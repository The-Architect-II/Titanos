"""Broker credential vault (WP2): records, backends, references and owner-only files.

Per-platform lanes (base tier, on every contract-job leg):
- Windows: a real DPAPI round trip; the blob does not contain the plaintext; the blob, the vault directory and
  entropy.bin are owner-only with a protected DACL; another install's entropy cannot open the blob.
- macOS: a real Keychain round trip in a TEMPORARY keychain (create-keychain/unlock-keychain over stdin), never the
  login keychain. Everywhere: a fake `security` records argv and stdin, and the secret is absent from argv.
- Linux: a real secret-tool round trip when secret-tool and a session bus are present; otherwise that cell is skipped as
  'untried' (never reported as a pass). Everywhere: a fake secret-tool records argv and stdin.
- Everywhere: sealed-file with the specified scrypt parameters (the maxmem regression), known-answer vectors, flipped-byte
  tamper failures, a wrong passphrase, and refusal of a passphrase file other accounts can read; env: values never
  written (the whole TITANOS_HOME tree is scanned); file: ACL refusal re-checked on every use; atomic rotation that
  deletes the old value; seal() -> vault_sealed; ambient OPENAI_API_KEY and ANTHROPIC_BASE_URL ignored without a record;
  opening the vault creates nothing until the first write; index.json metadata only; mtime re-read across instances; the
  bounded replace retry; no API that returns a value.
Synthetic values only; the netguard is installed (no test here opens a network connection).
"""
import dataclasses
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from broker_mocks import SENTINELS, netguard  # noqa: E402

netguard.install()

import httpx  # noqa: E402

from titanos.broker import config as broker_config  # noqa: E402
from titanos.broker import redact  # noqa: E402
from titanos.broker.config import Upstream  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.profiles import PROFILES  # noqa: E402
from titanos.broker.role import check_serve_preconditions  # noqa: E402
from titanos.broker.types import KeyStoreSummary  # noqa: E402
from titanos.broker.vault import CredentialHandle, fingerprint  # noqa: E402
from titanos.broker.vault import acl, refs, sealed_file  # noqa: E402
from titanos.broker.vault.keychain import KeychainBackend, delete_keychain, quote, temporary_keychain  # noqa: E402
from titanos.broker.vault.secret_service import SecretServiceBackend  # noqa: E402
from titanos.broker.vault.store import Vault, open_vault, platform_default  # noqa: E402
from titanos.settings import Settings  # noqa: E402

WINDOWS = sys.platform == 'win32'
ORIGIN = 'https://api.openai.com:443'
OTHER_ORIGIN = 'http://127.0.0.1:18080'
VALUE = 'sk-test-' + 'a1b2c3d4' * 4
NEW_VALUE = 'sk-test-' + 'e5f6a7b8' * 4


class MemoryBackend:
    """An inspectable in-memory VaultBackend (every stored value is visible to the test)."""

    def __init__(self, name):
        self.name, self.values, self.loads = name, {}, 0

    def available(self):
        return True

    def store(self, id, value):
        self.values[id] = bytes(value)

    def load(self, id):
        self.loads += 1
        if id not in self.values:
            raise BrokerError('missing', 'credential_unavailable')
        return self.values[id]

    def delete(self, id):
        self.values.pop(id, None)


class Unavailable(MemoryBackend):
    def available(self):
        return False


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def tree_bytes(home: Path) -> bytes:
    return b''.join(p.read_bytes() for p in sorted(home.rglob('*')) if p.is_file())


def leaked(home: Path, value: str) -> bool:
    data = tree_bytes(home)
    return value.encode('utf-8') in data or value.encode('utf-16-le') in data


def applied(handle: CredentialHandle, url='https://api.openai.com/v1/models') -> httpx.Request:
    request = httpx.Request('GET', url)
    handle.apply(request)
    return request


def broad_acl(path: Path) -> None:
    """Make path readable by other accounts (Authenticated Users on Windows, group/other on POSIX)."""
    if WINDOWS:
        acl._apply_sddl(path, f'D:P(A;;FA;;;{acl.current_user_sid()})(A;;FR;;;AU)')
    else:
        os.chmod(path, 0o644)


class VaultCase(unittest.TestCase):
    def setUp(self):
        netguard.reset_attempts()
        self.home = Path(tempfile.mkdtemp(prefix='titanos-vault-'))
        self.addCleanup(shutil.rmtree, self.home, True)
        self.name = platform_default()
        self.memory = MemoryBackend(self.name)
        self.clock = Clock()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [])

    def vault(self, settings=None, backend=None, **kw):
        return Vault(settings or Settings(), self.home, backends={self.name: backend or self.memory},
                     clock=self.clock, **kw)

    def assertCode(self, code, call, *args, **kw):
        with self.assertRaises(BrokerError) as ctx:
            call(*args, **kw)
        self.assertEqual(ctx.exception.code, code, ctx.exception)
        return ctx.exception

    def files(self):
        return sorted(p.relative_to(self.home).as_posix() for p in self.home.rglob('*'))


class Lazy(VaultCase):
    def test_opening_the_vault_creates_nothing_until_the_first_write(self):
        vault = open_vault(Settings(), self.home)
        self.assertIn(vault.backend_name, ('dpapi', 'keychain', 'secret-service', 'refs-only', 'unavailable'))  # refs-only: Linux without secret-tool
        self.assertEqual(vault.posture(), vault.backend_name)
        self.assertEqual(vault.records(), [])
        self.assertEqual(vault.summary()['credentials'], 0)
        self.assertEqual(vault.referenced_env_vars(), frozenset())
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin=ORIGIN, template='bearer')
        self.assertCode('not_found', vault.record, 'openai')
        self.assertEqual(self.files(), [])
        vault.put_env_ref('openai', 'WP2_TEST_PROVIDER_KEY', upstream_id='openai', origin=ORIGIN)
        self.assertIn('broker/vault/index.json', self.files())

    def test_importing_the_wp2_modules_does_no_io(self):
        code = '; '.join([
            'import os',
            'import titanos.broker.egress, titanos.broker.vault.store as store, titanos.broker.vault.acl as acl',
            'import titanos.broker.vault.dpapi as dpapi, titanos.broker.vault.keychain, titanos.broker.vault.refs',
            'import titanos.broker.vault.secret_service, titanos.broker.vault.sealed_file',
            'from titanos.settings import Settings',
            'vault = store.open_vault(Settings(), os.environ["TITANOS_HOME"])',
            # no Windows library is bound by importing or opening (the backend probe in backend_name comes later)
            'print(acl._Win._instance is None, dpapi._Crypt._instance is None, vault.sealed)',
        ])
        env = dict(os.environ, TITANOS_HOME=str(self.home), PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1')
        result = __import__('subprocess').run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                                               env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.split(), ['True', 'True', 'False'])
        self.assertEqual(self.files(), [])

    def test_seal_refuses_every_resolution(self):
        vault = self.vault()
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('openai', True, None, None)
        vault.handle('openai', origin=ORIGIN, template='bearer')
        with self.assertLogs('titanos.broker', 'WARNING') as logs:
            vault.seal('managed role')
            vault.seal('managed role')
        self.assertEqual(logs.output, ['WARNING:titanos.broker:managed role: 1 credentials configured and ignored'])
        for call in (lambda: vault.handle('openai', origin=ORIGIN, template='bearer'),
                     lambda: vault.validation_handle('openai', origin=ORIGIN, template='bearer'),
                     lambda: vault.put_secret('other', VALUE, upstream_id='openai', origin=ORIGIN),
                     lambda: vault.stage_rotation('openai', NEW_VALUE)):
            self.assertCode('vault_sealed', call)


class Records(VaultCase):
    def test_index_is_metadata_only_and_values_are_registered_for_scrubbing(self):
        vault = self.vault()
        record = vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        self.assertEqual((record.source, record.backend, record.validation), ('vault', self.name, 'unvalidated'))
        self.assertEqual(record.fingerprint, fingerprint(VALUE))
        self.assertNotIn(VALUE[8:16], record.fingerprint)
        index = json.loads((self.home / 'broker' / 'vault' / 'index.json').read_text(encoding='utf-8'))
        self.assertEqual(set(index['credentials']['openai']), set(record.to_dict()))
        self.assertFalse(leaked(self.home, VALUE))
        self.assertEqual(redact.scrub('upstream said ' + VALUE), 'upstream said <redacted>')
        self.assertTrue(acl.is_owner_only(self.home / 'broker' / 'vault' / 'index.json'))
        self.assertTrue(acl.is_owner_only(self.home / 'broker' / 'vault'))

    def test_handle_needs_validation_and_the_exact_origin(self):
        vault = self.vault()
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin=ORIGIN, template='bearer')
        vault.mark_validated('openai', True, None, fingerprint(VALUE))
        handle = vault.handle('openai', origin=ORIGIN, template='bearer')
        self.assertNotIn(VALUE, repr(handle) + str(handle.fingerprint))
        self.assertTrue(hmac.compare_digest(applied(handle).headers['authorization'], 'Bearer ' + VALUE))
        with self.assertRaises(BrokerError):
            applied(handle, 'https://api.openai.com.example.invalid/v1/models')
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin='https://api.openai.com:8443',
                        template='bearer')
        vault.mark_validated('openai', True, 'skipped', None)
        self.assertEqual(vault.record('openai').validation, 'skipped')
        vault.handle('openai', origin=ORIGIN, template='bearer')
        vault.mark_validated('openai', False, 'upstream_auth', None)
        self.assertEqual(vault.record('openai').validation, 'failed:upstream_auth')
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin=ORIGIN, template='bearer')

    def test_handle_enforces_purpose_and_upstream_binding(self):
        vault = self.vault()
        vault.put_secret('adv', VALUE, upstream_id='advisor', origin=ORIGIN, purpose='advisor')
        vault.put_secret('inf', NEW_VALUE, upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('adv', True, None, None)
        vault.mark_validated('inf', True, None, None)
        # an advisor credential never yields an inference handle (the default purpose), nor the reverse
        self.assertCode('credential_unavailable', vault.handle, 'adv', origin=ORIGIN, template='bearer')
        vault.handle('adv', origin=ORIGIN, template='bearer', purpose='advisor')
        self.assertCode('credential_unavailable', vault.handle, 'inf', origin=ORIGIN, template='bearer',
                        purpose='advisor')
        # a credential bound to upstream openai serves no other upstream sharing the origin
        vault.handle('inf', origin=ORIGIN, template='bearer', upstream_id='openai')
        self.assertCode('credential_unavailable', vault.handle, 'inf', origin=ORIGIN, template='bearer',
                        upstream_id='openai-2')

    def test_input_refusals(self):
        vault = self.vault()
        for kwargs in ({'id': 'Bad Id'}, {'secret': 'short'}, {'secret': ' ' + VALUE}, {'secret': VALUE + '\n'},
                       {'origin': 'https://api.openai.com'}, {'origin': 'https://api.openai.com:443/v1'},
                       {'upstream_id': 'store'}, {'purpose': 'acquisition'}):
            args = dict(id='openai', secret=VALUE, upstream_id='openai', origin=ORIGIN, purpose='inference', **{})
            args.update(kwargs)
            with self.subTest(kwargs=list(kwargs)):
                self.assertCode('invalid_request', vault.put_secret, args.pop('id'), args.pop('secret'), **args)
        self.assertCode('invalid_request', vault.put_env_ref, 'openai', 'OPENAI_BASE_URL', upstream_id='openai',
                        origin=ORIGIN)
        self.assertCode('invalid_request', vault.put_file_ref, 'openai', 'relative/key.txt', upstream_id='openai',
                        origin=ORIGIN)
        self.assertEqual(self.files(), [])

    def test_base_url_change_invalidates_and_validation_rebinds(self):
        vault = self.vault()
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('openai', True, None, None)
        self.assertEqual(vault.invalidate_for_upstream('openai', origin=OTHER_ORIGIN), ['openai'])
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin=ORIGIN, template='bearer')
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin=OTHER_ORIGIN, template='bearer')
        probe = vault.validation_handle('openai', origin=OTHER_ORIGIN, template='bearer')
        self.assertEqual(probe.origin, OTHER_ORIGIN)
        vault.mark_validated('openai', True, None, fingerprint(VALUE), origin=OTHER_ORIGIN)
        self.assertEqual(vault.record('openai').origin, OTHER_ORIGIN)
        vault.handle('openai', origin=OTHER_ORIGIN, template='bearer')
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin=ORIGIN, template='bearer')
        self.assertEqual(vault.invalidate_for_upstream('nothing'), [])

    def test_rotation_is_atomic_and_deletes_the_old_value(self):
        vault = self.vault()
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('openai', True, None, None)
        staged_fp = vault.stage_rotation('openai', NEW_VALUE)
        self.assertEqual(staged_fp, fingerprint(NEW_VALUE))
        self.assertEqual(vault.staged('openai')['fingerprint'], staged_fp)
        # the live value keeps serving while the staged one is validated
        live = applied(vault.handle('openai', origin=ORIGIN, template='bearer'))
        self.assertTrue(hmac.compare_digest(live.headers['authorization'], 'Bearer ' + VALUE))
        probe = applied(vault.validation_handle('openai', origin=ORIGIN, template='bearer', staged=True))
        self.assertTrue(hmac.compare_digest(probe.headers['authorization'], 'Bearer ' + NEW_VALUE))
        self.assertEqual(set(self.memory.values), {'openai', 'openai.next'})
        record = vault.commit_rotation('openai')
        self.assertEqual((record.fingerprint, record.validation), (fingerprint(NEW_VALUE), 'ok'))
        self.assertIsNotNone(record.rotated_at)
        self.assertEqual(set(self.memory.values), {'openai'})
        self.assertNotIn(VALUE.encode(), b''.join(self.memory.values.values()))
        self.assertIsNone(vault.staged('openai'))
        after = applied(vault.handle('openai', origin=ORIGIN, template='bearer'))
        self.assertTrue(hmac.compare_digest(after.headers['authorization'], 'Bearer ' + NEW_VALUE))
        vault.stage_rotation('openai', VALUE)
        vault.discard_rotation('openai')
        self.assertEqual(set(self.memory.values), {'openai'})
        self.assertCode('not_found', vault.commit_rotation, 'openai')
        vault.delete('openai')
        self.assertEqual(self.memory.values, {})
        self.assertEqual(vault.records(), [])

    def test_secret_cache_ttl(self):
        vault = self.vault(Settings(broker_secret_cache_seconds=60))
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('openai', True, None, None)
        for _ in range(3):
            vault.handle('openai', origin=ORIGIN, template='bearer')
        self.assertEqual(self.memory.loads, 1)
        self.clock.now += 61
        vault.handle('openai', origin=ORIGIN, template='bearer')
        self.assertEqual(self.memory.loads, 2)
        vault.mark_validated('openai', True, None, None)  # an index change drops the cache
        vault.handle('openai', origin=ORIGIN, template='bearer')
        self.assertEqual(self.memory.loads, 3)

    def test_index_is_re_read_when_another_writer_changes_it(self):
        server, cli = self.vault(), self.vault()
        self.assertEqual(server.records(), [])
        cli.put_env_ref('openai', 'WP2_TEST_PROVIDER_KEY', upstream_id='openai', origin=ORIGIN)
        self.assertEqual([r.id for r in server.records()], ['openai'])
        cli.mark_validated('openai', True, 'skipped', None)
        self.assertEqual(server.record('openai').validation, 'skipped')

    def test_last_used_is_written_at_most_once_a_minute(self):
        vault = self.vault()
        vault.put_env_ref('openai', 'WP2_TEST_PROVIDER_KEY', upstream_id='openai', origin=ORIGIN)
        vault.touch('openai')
        first = vault.record('openai').last_used_at
        self.assertIsNotNone(first)
        self.assertFalse(vault.touch_due('openai'))
        self.clock.now += 61
        self.assertTrue(vault.touch_due('openai'))

    def test_no_store_means_vault_unavailable_naming_both_options(self):
        preconditions = dict(bootstrap_key='boot-' + 'b' * 24, keystore=KeyStoreSummary(), explicit_port=False,
                             audit_ok=True)
        for settings, backend in ((Settings(), Unavailable(self.name)), (Settings(broker_vault_backend='refs-only'), None)):
            vault = self.vault(settings, backend=backend)
            with self.subTest(backend=settings.broker_vault_backend):
                error = self.assertCode('vault_unavailable', vault.put_secret, 'openai', VALUE, upstream_id='openai',
                                        origin=ORIGIN)
                self.assertEqual(error.status, 503)
                # spec: "If none is available, auto becomes refs-only"
                self.assertEqual(vault.posture(), 'refs-only')
                self.assertEqual(vault.summary()['backend'], 'refs-only')
                if settings.broker_vault_backend == 'auto':
                    for option in ('env:', 'file:', 'sealed-file', 'TITANOS_BROKER_VAULT_PASSPHRASE_FILE'):
                        self.assertIn(option, str(error))
                # a headless broker host without an OS store still starts on references
                broker = dataclasses.replace(settings, role='broker')
                check_serve_preconditions(broker, vault_backend=vault.posture(), **preconditions)
                self.assertNotIn(VALUE, str(error))
        # an explicitly configured store that cannot be used stays 'unavailable', which the broker role refuses
        explicit = Settings(broker_vault_backend=self.name)
        vault = self.vault(explicit, backend=Unavailable(self.name))
        self.assertEqual(vault.posture(), 'unavailable')
        self.assertCode('vault_unavailable', vault.put_secret, 'openai', VALUE, upstream_id='openai', origin=ORIGIN)
        with self.assertRaises(ValueError):
            check_serve_preconditions(dataclasses.replace(explicit, role='broker'), vault_backend=vault.posture(),
                                      **preconditions)
        self.assertEqual(self.files(), [])

    def test_no_api_returns_a_value(self):
        public = {name for name in dir(Vault) if not name.startswith('_')}
        for forbidden in ('get', 'get_secret', 'read', 'reveal', 'value', 'secret', 'resolve', 'load', 'export'):
            self.assertNotIn(forbidden, public)
        vault = self.vault()
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('openai', True, None, None)
        for value in (vault.records(), vault.summary(), vault.record('openai').to_dict(), vault.staged('openai'),
                      repr(vault.handle('openai', origin=ORIGIN, template='bearer'))):
            self.assertNotIn(VALUE, repr(value))


class References(VaultCase):
    VAR = 'WP2_TEST_PROVIDER_KEY'

    def test_env_values_are_never_written(self):
        environ = {self.VAR: VALUE, 'PATH': os.environ.get('PATH', '')}
        vault = self.vault(environ=environ)
        record = vault.put_env_ref('openai', self.VAR, upstream_id='openai', origin=ORIGIN)
        self.assertEqual((record.source, record.env_var, record.fingerprint), ('env', self.VAR, fingerprint(VALUE)))
        vault.mark_validated('openai', True, None, fingerprint(VALUE))
        handle = vault.handle('openai', origin=ORIGIN, template='bearer')
        self.assertTrue(hmac.compare_digest(applied(handle).headers['authorization'], 'Bearer ' + VALUE))
        vault.touch('openai')
        vault.invalidate_for_upstream('openai')
        self.assertEqual(vault.referenced_env_vars(), frozenset({self.VAR}))
        self.assertTrue(self.files())
        self.assertFalse(leaked(self.home, VALUE))
        environ.pop(self.VAR)
        vault.mark_validated('openai', True, None, None)
        self.clock.now += 10_000
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin=ORIGIN, template='bearer')

    def test_env_refusals(self):
        self.assertCode('credential_unavailable', refs.resolve_env, 'OPENAI_BASE_URL', {'OPENAI_BASE_URL': VALUE})
        self.assertCode('credential_unavailable', refs.resolve_env, self.VAR, {})
        self.assertCode('credential_unavailable', refs.resolve_env, self.VAR, {self.VAR: 'short'})
        self.assertCode('credential_unavailable', refs.resolve_env, 'BAD-NAME', {'BAD-NAME': VALUE})
        self.assertEqual(refs.resolve_env(self.VAR, {self.VAR: VALUE + '\n'}), VALUE)
        # Titanos' own variables (the bootstrap key above all) are never an upstream credential, at bind and at use
        for name in ('TITANOS_API_KEY', 'titanos_api_key', 'TITANOS_BROKER_VAULT_PASSPHRASE_FILE'):
            with self.subTest(name=name):
                self.assertCode('credential_unavailable', refs.resolve_env, name, {name: VALUE})
                self.assertCode('invalid_request', self.vault(environ={name: VALUE}).put_env_ref, 'openai', name,
                                upstream_id='openai', origin=ORIGIN)

    def test_ambient_provider_variables_are_ignored_without_a_record(self):
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': VALUE, 'ANTHROPIC_BASE_URL': 'http://127.0.0.1:9/x',
                                          'OPENAI_BASE_URL': 'http://127.0.0.1:9/y', 'ANTHROPIC_API_KEY': NEW_VALUE}):
            vault = open_vault(Settings(), self.home)
            self.assertEqual(vault.records(), [])
            for cid in ('openai', 'anthropic', 'openai_api_key'):
                self.assertCode('credential_unavailable', vault.handle, cid, origin=ORIGIN, template='bearer')
            anthropic = Upstream(profile='anthropic')
            self.assertEqual(anthropic.effective_base_url(), PROFILES['anthropic'].default_base_url)
            self.assertEqual(anthropic.origin(allow_http_lan=False), 'https://api.anthropic.com:443')
            self.assertEqual(Upstream(profile='openai').origin(allow_http_lan=False), ORIGIN)
            self.assertEqual(vault.referenced_env_vars(), frozenset())
        self.assertEqual(self.files(), [])

    def test_file_references_recheck_permissions_on_every_use(self):
        keyfile = self.home / 'keys' / 'openai.key'
        acl.write_owner_only(keyfile, (VALUE + '\n').encode())
        self.assertEqual(refs.resolve_file(str(keyfile)), VALUE)
        vault = self.vault(Settings(broker_secret_cache_seconds=300))
        vault.put_file_ref('openai', str(keyfile), upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('openai', True, None, None)
        vault.handle('openai', origin=ORIGIN, template='bearer')  # now cached
        broad_acl(keyfile)
        self.assertFalse(acl.is_owner_only(keyfile))
        self.assertCode('credential_unavailable', refs.resolve_file, str(keyfile))
        self.assertCode('credential_unavailable', vault.handle, 'openai', origin=ORIGIN, template='bearer')
        self.assertCode('credential_unavailable', vault.put_file_ref, 'other', str(keyfile), upstream_id='openai',
                        origin=ORIGIN)
        self.assertCode('credential_unavailable', refs.resolve_file, str(self.home / 'missing.key'))
        self.assertCode('credential_unavailable', refs.resolve_file, 'relative.key')
        self.assertFalse(leaked(self.home / 'broker', VALUE))


class OwnerOnlyFiles(VaultCase):
    def test_write_owner_only_and_directories(self):
        target = self.home / 'private' / 'blob.bin'
        acl.write_owner_only(target, b'payload')
        self.assertEqual(target.read_bytes(), b'payload')
        for path in (target, target.parent):
            summary = acl.acl_summary(path)
            self.assertTrue(summary.owner_only, summary)
            if WINDOWS:
                self.assertTrue(summary.protected)
            else:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600)
        child = target.parent / 'inherited.txt'
        child.write_text('x')
        if WINDOWS:
            self.assertTrue(acl.is_owner_only(child))  # the directory's inheritable owner-only ACEs
        broad_acl(child)
        self.assertFalse(acl.is_owner_only(child))
        self.assertFalse(acl.is_owner_only(self.home / 'missing'))
        self.assertEqual([p.name for p in target.parent.iterdir() if p.name.endswith('.tmp')], [])

    def test_temporary_file_is_owner_only_from_creation(self):
        target = self.home / 'created.bin'
        target.parent.mkdir(parents=True, exist_ok=True)
        seen = []
        real = broker_config.os.replace

        def inspect(src, dst):
            seen.append(acl.acl_summary(src))  # the temporary file, before it becomes the target
            return real(src, dst)
        with mock.patch.object(acl, 'make_owner_only', side_effect=AssertionError('post-hoc ACL change')), \
                mock.patch.object(broker_config.os, 'replace', inspect):
            acl.write_owner_only(target, b'payload')
        self.assertEqual(len(seen), 1)
        self.assertTrue(seen[0].owner_only, seen[0])
        if WINDOWS:
            self.assertTrue(seen[0].protected)  # created with the protected file DACL, not the parent's inherited one
        self.assertEqual(target.read_bytes(), b'payload')

    def test_replace_is_retried_on_windows_sharing_errors(self):
        target = self.home / 'retry.bin'
        acl.write_owner_only(target, b'one')
        real, calls = os.replace, []

        def flaky(src, dst):
            calls.append(dst)
            if len(calls) <= 2:
                raise PermissionError(13, 'sharing violation (simulated)')
            return real(src, dst)
        with mock.patch.object(broker_config.os, 'replace', flaky):
            acl.write_owner_only(target, b'two')
        self.assertEqual(len(calls), 3)
        self.assertEqual(target.read_bytes(), b'two')
        self.assertEqual(sorted(p.name for p in self.home.iterdir()), ['retry.bin'])


@unittest.skipUnless(WINDOWS, 'DPAPI exists only on Windows')
class Dpapi(VaultCase):
    def test_real_round_trip_with_owner_only_files(self):
        vault = open_vault(Settings(), self.home)
        self.assertEqual(vault.backend_name, 'dpapi')
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('openai', True, None, None)
        directory = self.home / 'broker' / 'vault'
        blob, entropy = directory / 'openai.dpapi', directory / 'entropy.bin'
        self.assertFalse(leaked(self.home, VALUE))
        self.assertGreater(len(blob.read_bytes()), len(VALUE))
        self.assertEqual(len(entropy.read_bytes()), 32)
        for path in (blob, entropy, directory, directory / 'index.json'):
            summary = acl.acl_summary(path)
            self.assertTrue(summary.owner_only and summary.protected, (path.name, summary))
        fresh = open_vault(Settings(), self.home)  # a second process' view: decrypts with the same user and entropy
        handle = fresh.handle('openai', origin=ORIGIN, template='bearer')
        self.assertTrue(hmac.compare_digest(applied(handle).headers['authorization'], 'Bearer ' + VALUE))
        fresh.stage_rotation('openai', NEW_VALUE)
        self.assertTrue((directory / 'openai.next.dpapi').is_file())
        fresh.commit_rotation('openai')
        self.assertFalse((directory / 'openai.next.dpapi').exists())
        rotated = open_vault(Settings(), self.home).handle('openai', origin=ORIGIN, template='bearer')
        self.assertTrue(hmac.compare_digest(applied(rotated).headers['authorization'], 'Bearer ' + NEW_VALUE))

    def test_another_install_entropy_cannot_open_the_blob(self):
        from titanos.broker.vault.dpapi import DpapiBackend
        first = DpapiBackend(self.home / 'a')
        first.store('openai', VALUE.encode())
        second = DpapiBackend(self.home / 'b')
        second.store('other', NEW_VALUE.encode())  # creates b's own entropy
        shutil.copy(first.path('openai'), second.path('openai'))
        self.assertTrue(hmac.compare_digest(first.load('openai'), VALUE.encode()))
        self.assertCode('credential_unavailable', second.load, 'openai')


FAKE_TOOL = r'''
import json, os, sys
state = sys.argv[1]
args = sys.argv[2:]
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
stdin = sys.stdin.read() if (args[:1] == ['-i'] or args[:1] == ['store']) else ''
with open(os.path.join(state, 'log.jsonl'), 'a', encoding='utf-8') as log:
    log.write(json.dumps({'argv': sys.argv[1:], 'stdin': stdin, 'env': sorted(os.environ)}) + '\n')
db_path = os.path.join(state, 'db.json')
db = json.load(open(db_path, encoding='utf-8')) if os.path.exists(db_path) else {}

def tokens(line):
    out, cur, quoted, i = [], None, False, 0
    while i < len(line):
        c = line[i]
        if quoted:
            if c == '\\':
                i += 1; cur += line[i]
            elif c == '"':
                quoted = False
            else:
                cur += c
        elif c == '"':
            quoted, cur = True, cur or ''
        elif c == ' ':
            if cur is not None:
                out.append(cur); cur = None
        else:
            cur = (cur or '') + c
        i += 1
    if cur is not None:
        out.append(cur)
    return out

def options(words, valued=('-s', '-a', '-w')):
    found, i = {}, 0
    while i < len(words):
        if words[i] == '-w' and '-w' not in valued:
            i += 1
        elif words[i] in valued:
            found[words[i][1]] = words[i + 1]; i += 2
        elif words[i] == '-U':
            i += 1
        else:
            found['keychain'] = words[i]; i += 1
    return found

kind = os.path.basename(state)
if kind == 'security':
    if args[:1] == ['-i']:
        for line in stdin.splitlines():
            words = tokens(line)
            if words and words[0] == 'add-generic-password':
                o = options(words[1:])
                db[o.get('keychain', '') + '|' + o['s'] + '|' + o['a']] = o['w']
    else:
        o = options(args[1:], valued=('-s', '-a'))
        key = o.get('keychain', '') + '|' + o['s'] + '|' + o['a']
        if args[0] == 'find-generic-password':
            if key not in db:
                sys.exit(44)
            sys.stdout.write(db[key] + '\n')
        elif args[0] == 'delete-generic-password':
            if db.pop(key, None) is None:
                sys.exit(44)
else:
    attrs = dict(zip(args[1::2], args[2::2])) if args[0] != 'store' else dict(zip(args[2::2], args[3::2]))
    key = attrs['service'] + '|' + attrs['account']
    if args[0] == 'store':
        db[key] = stdin
    elif args[0] == 'lookup':
        if key not in db:
            sys.exit(1)
        sys.stdout.write(db[key])
    elif args[0] == 'clear':
        db.pop(key, None)
json.dump(db, open(db_path, 'w', encoding='utf-8'))
'''


class HelperTools(VaultCase):
    TRICKY = 'sk-test-"quoted"\\back\\slash-' + 'c0ffee00' * 3

    def fake(self, kind: str) -> list[str]:
        state = self.home / kind
        state.mkdir()
        script = self.home / f'fake_{kind.replace("-", "_")}.py'
        script.write_text(FAKE_TOOL, encoding='utf-8')
        return [sys.executable, str(script), str(state)]

    def log(self, kind: str) -> list[dict]:
        return [json.loads(line) for line in (self.home / kind / 'log.jsonl').read_text(encoding='utf-8').splitlines()]

    def environ(self):
        env = dict(os.environ)
        env.update({'OPENAI_API_KEY': SENTINELS['leak'], 'ANTHROPIC_BASE_URL': 'http://127.0.0.1:9'})
        return env

    def test_fake_security_records_argv_and_stdin(self):
        exe = self.fake('security')
        backend = KeychainBackend(keychain=self.home / 'test.keychain-db', instance_id='0123abcd' * 4, executable=exe,
                                  environ=self.environ())
        self.assertTrue(backend.available())
        backend.store('openai', self.TRICKY.encode())
        self.assertEqual(backend.load('openai'), self.TRICKY.encode())
        backend.delete('openai')
        self.assertCode('credential_unavailable', backend.load, 'openai')
        entries = self.log('security')
        writes = [e for e in entries if e['argv'][1:] == ['-i']]
        self.assertEqual(len(writes), 1)
        self.assertIn('add-generic-password -U -s "titanos.broker" -a "' + '0123abcd' * 4 + '/openai" -w "',
                      writes[0]['stdin'])
        for entry in entries:
            self.assertFalse(any(self.TRICKY in arg or 'quoted' in arg for arg in entry['argv']))
            self.assertNotIn('OPENAI_API_KEY', entry['env'])
            self.assertNotIn('ANTHROPIC_BASE_URL', entry['env'])
            keychain = str(self.home / 'test.keychain-db')
            self.assertTrue(keychain in entry['argv'] or quote(keychain) in entry['stdin'])
        with self.assertRaises(ValueError):
            backend.store('bad', b'line-one\nline-two')

    def test_fake_secret_tool_records_argv_and_stdin(self):
        exe = self.fake('secret-tool')
        env = self.environ()
        env['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=/run/user/fixture/bus'
        backend = SecretServiceBackend(instance_id='0123abcd' * 4, executable=exe, environ=env)
        self.assertTrue(backend.available())
        self.assertFalse(SecretServiceBackend(executable=exe, environ={'PATH': ''}).available())
        backend.store('openai', self.TRICKY.encode())
        self.assertEqual(backend.load('openai'), self.TRICKY.encode())
        backend.delete('openai')
        self.assertCode('credential_unavailable', backend.load, 'openai')
        entries = self.log('secret-tool')
        stores = [e for e in entries if e['argv'][1] == 'store']
        self.assertEqual(stores[0]['argv'][1:], ['store', '--label=Titanos', 'service', 'titanos.broker', 'account',
                                                 '0123abcd' * 4 + '/openai'])
        self.assertEqual(stores[0]['stdin'], self.TRICKY)
        for entry in entries:
            self.assertFalse(any(self.TRICKY in arg for arg in entry['argv']))
            self.assertNotIn('OPENAI_API_KEY', entry['env'])
            self.assertIn('DBUS_SESSION_BUS_ADDRESS', entry['env'])

    def test_vault_over_a_helper_backend_uses_the_instance_id(self):
        exe = self.fake('secret-tool')
        env = self.environ()
        env['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=/run/user/fixture/bus'
        vault = Vault(Settings(broker_vault_backend='secret-service'), self.home, environ=env)
        vault._backends['secret-service'] = SecretServiceBackend(instance_id=vault.instance_id, executable=exe,
                                                                 environ=env)
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        instance = (self.home / 'broker' / 'instance-id').read_text().strip()
        self.assertRegex(instance, r'^[0-9a-f]{32}$')
        self.assertTrue(acl.is_owner_only(self.home / 'broker' / 'instance-id'))
        self.assertTrue(any(arg == instance + '/openai' for e in self.log('secret-tool') for arg in e['argv']))
        vault.mark_validated('openai', True, None, None)
        handle = vault.handle('openai', origin=ORIGIN, template='bearer')
        self.assertTrue(hmac.compare_digest(applied(handle).headers['authorization'], 'Bearer ' + VALUE))

    def test_helper_children_never_inherit_env_reference_variables(self):
        exe = self.fake('secret-tool')
        env = self.environ()
        env['DBUS_SESSION_BUS_ADDRESS'] = 'unix:path=/run/user/fixture/bus'
        env['MY_PROVIDER_TOKEN'] = 'sk-test-' + 'f00dfeed' * 4  # synthetic; the name does not end in _API_KEY
        vault = Vault(Settings(broker_vault_backend='secret-service'), self.home, environ=env)
        backend = vault._backend('secret-service')  # the vault's own construction (deny wired in)
        backend._executable = exe
        vault.put_env_ref('custom', 'MY_PROVIDER_TOKEN', upstream_id='custom', origin=OTHER_ORIGIN)
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        entries = self.log('secret-tool')
        self.assertTrue(entries)
        for entry in entries:
            self.assertNotIn('MY_PROVIDER_TOKEN', entry['env'])
            self.assertNotIn(env['MY_PROVIDER_TOKEN'], json.dumps(entry))
            self.assertIn('DBUS_SESSION_BUS_ADDRESS', entry['env'])
        keychain = KeychainBackend(instance_id='0123abcd' * 4, executable=self.fake('security'), environ=env,
                                   deny=vault.referenced_env_vars)
        keychain.store('openai', VALUE.encode())
        for entry in self.log('security'):
            self.assertNotIn('MY_PROVIDER_TOKEN', entry['env'])

    @unittest.skipUnless(sys.platform == 'darwin' and Path('/usr/bin/security').exists(), 'real Keychain: macOS only')
    def test_real_keychain_round_trip_in_a_temporary_keychain(self):
        path = self.home / 'titanos-test.keychain-db'
        temporary_keychain(path)
        self.addCleanup(delete_keychain, path)
        backend = KeychainBackend(keychain=path, instance_id='0123abcd' * 4)
        self.assertTrue(backend.available())
        backend.store('openai', self.TRICKY.encode())
        self.assertEqual(backend.load('openai'), self.TRICKY.encode())
        backend.delete('openai')
        self.assertCode('credential_unavailable', backend.load, 'openai')

    def test_real_secret_tool_round_trip(self):
        backend = SecretServiceBackend(instance_id='titanos-test-' + os.urandom(4).hex())
        if not backend.available():
            self.skipTest('untried: secret-tool or a session bus is not available here, so the real Secret Service cell '
                          'is untried (not a pass)')
        try:
            backend.store('openai', VALUE.encode())
            self.assertEqual(backend.load('openai'), VALUE.encode())
        finally:
            backend.delete('openai')


class SealedFile(VaultCase):
    PASSPHRASE = b'correct horse battery staple'
    SALT = bytes(range(32))
    NONCE = bytes(range(100, 116))
    PLAINTEXT = b'test-sealed-file-value-0001'
    # TSF1 known answer for (PASSPHRASE, id 'openai', PLAINTEXT, SALT, NONCE): magic, version, salt, nonce, ciphertext,
    # tag. Computed once with the specified construction; the test also recomputes it independently of the module.
    KAT_BLOB = ('54534631' '01' '000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f'
                '6465666768696a6b6c6d6e6f70717273'
                'e7d99211869a1ad52e1a72428dd4d702216db1bd85d426ee106cce'
                '67270fb58a50e025f01c2023f554de6c0e3934c3c09759d68bee6f2e96491fc2')

    def test_scrypt_parameters_need_maxmem(self):
        params = dict(salt=self.SALT, n=sealed_file.SCRYPT_N, r=sealed_file.SCRYPT_R, p=sealed_file.SCRYPT_P, dklen=64)
        with self.assertRaises(ValueError):
            hashlib.scrypt(self.PASSPHRASE, **params)  # the default 32 MiB limit refuses n=2**15, r=8
        self.assertEqual(len(hashlib.scrypt(self.PASSPHRASE, maxmem=sealed_file.SCRYPT_MAXMEM, **params)), 64)
        enc, mac = sealed_file.derive_keys(self.PASSPHRASE, self.SALT)
        self.assertEqual((len(enc), len(mac)), (32, 32))

    def test_known_answer_vectors(self):
        # RFC 7914 section 12 (scrypt) and RFC 4231 test case 2 (HMAC-SHA256): the primitives
        self.assertEqual(hashlib.scrypt(b'password', salt=b'NaCl', n=1024, r=8, p=16, dklen=64).hex(),
                         'fdbabe1c9d3472007856e7190d01e9fe7c6ad7cbc8237830e77376634b373162'
                         '2eaf30d92e22a3886ff109279d9830dac727afb94a83ee6d8360cbdfa2cc0640')
        self.assertEqual(hmac.new(b'Jefe', b'what do ya want for nothing?', hashlib.sha256).hexdigest(),
                         '5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843')
        # the construction, recomputed independently of the module's helpers
        key = hashlib.scrypt(self.PASSPHRASE, salt=self.SALT, n=2 ** 15, r=8, p=1, maxmem=64 * 1024 * 1024, dklen=64)
        enc, mac = key[:32], key[32:]
        stream = b''.join(hmac.new(enc, self.NONCE + i.to_bytes(8, 'big'), hashlib.sha256).digest() for i in range(1))
        ciphertext = bytes(a ^ b for a, b in zip(self.PLAINTEXT, stream))
        tag = hmac.new(mac, b'TSF1' + b'\x01' + (6).to_bytes(2, 'big') + b'openai' + self.NONCE + ciphertext,
                       hashlib.sha256).digest()
        expected = b'TSF1\x01' + self.SALT + self.NONCE + ciphertext + tag
        blob = sealed_file.seal_bytes(self.PASSPHRASE, 'openai', self.PLAINTEXT, salt=self.SALT, nonce=self.NONCE)
        self.assertEqual(blob, expected)
        self.assertEqual(blob.hex(), self.KAT_BLOB)
        self.assertEqual(sealed_file.open_bytes(self.PASSPHRASE, 'openai', blob), self.PLAINTEXT)
        long_value = b'x' * 100  # several keystream blocks
        self.assertEqual(sealed_file.open_bytes(self.PASSPHRASE, 'id', sealed_file.seal_bytes(self.PASSPHRASE, 'id',
                                                                                              long_value)), long_value)

    def test_tamper_wrong_passphrase_and_id_binding(self):
        blob = sealed_file.seal_bytes(self.PASSPHRASE, 'openai', self.PLAINTEXT, salt=self.SALT, nonce=self.NONCE)
        positions = {'magic': 0, 'version': 4, 'salt': 5, 'nonce': 40, 'ciphertext': 60, 'tag': len(blob) - 1}
        for name, position in positions.items():
            with self.subTest(byte=name):
                tampered = bytearray(blob)
                tampered[position] ^= 0x01
                with self.assertRaises(sealed_file.SealedFileError):
                    sealed_file.open_bytes(self.PASSPHRASE, 'openai', bytes(tampered))
        with self.assertRaises(sealed_file.SealedFileError):
            sealed_file.open_bytes(b'wrong passphrase value', 'openai', blob)
        with self.assertRaises(sealed_file.SealedFileError):
            sealed_file.open_bytes(self.PASSPHRASE, 'other', blob)
        with self.assertRaises(sealed_file.SealedFileError):
            sealed_file.open_bytes(self.PASSPHRASE, 'openai', blob[:40])

    def test_vault_on_sealed_file_and_the_passphrase_file_acl(self):
        passphrase = self.home / 'secrets' / 'vault.pass'
        acl.write_owner_only(passphrase, self.PASSPHRASE + b'\n')
        settings = Settings(broker_vault_backend='sealed-file', broker_vault_passphrase_file=str(passphrase))
        vault = open_vault(settings, self.home)
        self.assertEqual(vault.posture(), 'sealed-file')
        vault.put_secret('openai', VALUE, upstream_id='openai', origin=ORIGIN)
        vault.mark_validated('openai', True, None, None)
        sealed = self.home / 'broker' / 'vault' / 'openai.sealed'
        self.assertTrue(acl.is_owner_only(sealed))
        self.assertFalse(leaked(self.home / 'broker', VALUE))
        handle = open_vault(settings, self.home).handle('openai', origin=ORIGIN, template='bearer')
        self.assertTrue(hmac.compare_digest(applied(handle).headers['authorization'], 'Bearer ' + VALUE))
        data = bytearray(sealed.read_bytes())
        data[-1] ^= 1
        sealed.write_bytes(bytes(data))
        self.assertCode('credential_unavailable', open_vault(settings, self.home).handle, 'openai', origin=ORIGIN,
                        template='bearer')
        broker = dataclasses.replace(settings, role='broker')
        preconditions = dict(bootstrap_key='boot-' + 'b' * 24, keystore=KeyStoreSummary(), explicit_port=False,
                             audit_ok=True)
        check_serve_preconditions(broker, vault_backend=open_vault(broker, self.home).posture(), **preconditions)
        broad_acl(passphrase)
        refused = open_vault(settings, self.home)
        self.assertEqual(refused.posture(), 'unavailable')
        with self.assertRaises(ValueError):  # broker role refuses to start on this posture
            check_serve_preconditions(broker, vault_backend=refused.posture(), **preconditions)
        error = self.assertCode('vault_unavailable', refused.put_secret, 'other', VALUE, upstream_id='openai',
                                origin=ORIGIN)
        self.assertIn('owner-only', str(error))
        self.assertCode('vault_unavailable', refused.handle, 'openai', origin=ORIGIN, template='bearer')

    def test_sealed_file_is_never_auto_selected(self):
        vault = open_vault(Settings(), self.home)
        self.assertNotEqual(vault.backend_name, 'sealed-file')


if __name__ == '__main__':
    unittest.main()
