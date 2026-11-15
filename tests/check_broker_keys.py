"""Broker inbound key store (WP3): mint, verify, expire, revoke and rotate without a restart; the token is returned once
and keys.json holds only its SHA-256 (the whole broker tree is scanned); the unknown-id dummy compare; broker-role
budget and expiry rule (budget_required); the last-key guard (last_key_required); to_principal scopes and constraints;
an offline mint by a second KeyStore instance seen after the mtime check; has_records stays true when every key has
expired; the offline keyless reset and the sticky enforcement marker; a failed audit append persists nothing; a
transient read error clears on the next recheck; last_used_at pacing and its non-blocking lock; the owner-only file
mode. Synthetic keys only; no server, model or network (the netguard refuses any non-loopback connect)."""
import dataclasses
from datetime import datetime, timedelta, timezone
import hmac
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos.broker import keys as keys_module  # noqa: E402
from titanos.broker.audit import AuditChainError, AuditLog  # noqa: E402
from titanos.broker.errors import BrokerError  # noqa: E402
from titanos.broker.keys import TOKEN_RE, KeyPatch, KeySpec, KeyStore, is_private  # noqa: E402
from titanos.broker.redact import scrub  # noqa: E402
from titanos.settings import Settings  # noqa: E402


def install_netguard():
    """tests/broker_mocks/netguard.py (WP4), loaded by file path so the mock package itself is not imported; without that
    file, a minimal guard that refuses every non-loopback socket connect."""
    path = ROOT / 'tests' / 'broker_mocks' / 'netguard.py'
    if path.is_file():
        spec = importlib.util.spec_from_file_location('titanos_test_netguard', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.install()
        return

    def loopback(address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            return ipaddress.ip_address(str(host).split('%', 1)[0]).is_loopback
        except ValueError:
            return False
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded(original):
        def call(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6) and not loopback(address):
                raise ConnectionRefusedError('netguard: non-loopback connect refused in a broker check')
            return original(self, address)
        return call
    socket.socket.connect = guarded(connect)
    socket.socket.connect_ex = guarded(connect_ex)


install_netguard()
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now=NOW):
        self.now = now
        self.mono = 1000.0

    def __call__(self):
        return self.now

    def monotonic(self):
        return self.mono

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)
        self.mono += seconds


class KeyStoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.path = self.home / 'broker' / 'keys.json'
        self.clock = Clock()
        self.audit = AuditLog(self.home / 'broker' / 'audit', clock=self.clock)
        self.settings = Settings()

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, settings=None, **kw):
        kw.setdefault('clock', self.clock)
        kw.setdefault('monotonic', self.clock.monotonic)
        return KeyStore(self.path, settings or self.settings, self.audit, **kw)

    def tree_text(self):
        out = []
        for path in self.home.rglob('*'):
            if path.is_file():
                out.append(path.read_bytes().decode('utf-8', 'replace'))
        return '\n'.join(out)


class Lifecycle(KeyStoreCase):
    def test_mint_returns_the_secret_once_and_stores_only_the_digest(self):
        store = self.store()
        record, token = store.mint(KeySpec(name='ci', scopes=['inference']), created_by='bootstrap')
        self.assertRegex(token, TOKEN_RE)
        self.assertTrue(token.startswith('tk_' + record.id + '_'))
        secret_part = token.split('_', 2)[2]
        self.assertEqual(len(secret_part), 43)
        self.assertFalse(hasattr(record, 'token') or hasattr(record, 'secret'))
        document = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(document['version'], 1)
        stored = document['keys'][record.id]
        self.assertEqual(stored['sha256'], keys_module.token_digest(token))
        self.assertNotIn('secret', stored)
        everything = self.tree_text()  # keys.json, the audit log and the lock files
        self.assertNotIn(token, everything)
        self.assertNotIn(secret_part, everything)
        self.assertNotIn('sha256', json.dumps(record.public()))
        self.assertNotIn(token, json.dumps([r.public() for r in store.list()]))
        self.assertEqual(store.verify(token), record)
        self.assertNotIn(token, scrub('client sent ' + token))  # registered with the redaction registry at mint

    def test_verify_wrong_secret_unknown_id_and_non_token(self):
        store = self.store()
        record, token = store.mint(KeySpec(name='a', scopes=['inference']), created_by='bootstrap')
        wrong = token[:-4] + ('AAAA' if not token.endswith('AAAA') else 'BBBB')
        self.assertIsNone(store.verify(wrong))
        unknown = 'tk_' + 'z' * 10 + token[13:]
        self.assertNotEqual(unknown.split('_')[1], record.id)
        with patch.object(keys_module.hmac, 'compare_digest', wraps=hmac.compare_digest) as compare:
            self.assertIsNone(store.verify(unknown))
        self.assertEqual(compare.call_count, 1, 'an unknown id still runs one constant-time compare')
        self.assertEqual(compare.call_args.args[0], keys_module._DUMMY_DIGEST)
        with patch.object(keys_module.hmac, 'compare_digest', wraps=hmac.compare_digest) as compare:
            self.assertIsNone(store.verify('local'))
        self.assertEqual(compare.call_count, 1)
        self.assertIsNone(store.verify(None))

    def test_expiry_and_revocation_take_effect_without_restart(self):
        store = self.store()
        record, token = store.mint(KeySpec(name='short', scopes=['inference'],
                                           expires_at=NOW + timedelta(minutes=5)), created_by='bootstrap')
        self.assertIsNotNone(store.verify(token))
        self.clock.advance(301)
        with self.assertRaises(BrokerError) as caught:
            store.verify(token)
        self.assertEqual((caught.exception.code, caught.exception.status), ('key_expired', 401))
        other, token2 = store.mint(KeySpec(name='r', scopes=['inference']), created_by='bootstrap')
        self.assertIsNotNone(store.verify(token2))
        store.revoke(other.id)
        with self.assertRaises(BrokerError) as caught:
            store.verify(token2)
        self.assertEqual((caught.exception.code, caught.exception.status), ('key_revoked', 401))
        self.assertEqual(store.revoke(other.id).revoked_at, store.get(other.id).revoked_at)  # idempotent
        self.assertFalse(store.has_active())

    def test_rotate_returns_a_new_secret_once_and_retires_the_old(self):
        store = self.store()
        record, old = store.mint(KeySpec(name='rot', scopes=['inference']), created_by='bootstrap')
        rotated, new = store.rotate(record.id)
        self.assertEqual(rotated.id, record.id)
        self.assertNotEqual(old, new)
        self.assertRegex(new, TOKEN_RE)
        self.assertIsNone(store.verify(old))
        self.assertEqual(store.verify(new).id, record.id)
        self.assertNotIn(new.split('_', 2)[2], self.tree_text())
        store.revoke(record.id)
        with self.assertRaises(BrokerError):
            store.rotate(record.id)

    def test_update_patch_semantics(self):
        store = self.store()
        record, _ = store.mint(KeySpec(name='p', scopes=['inference'], note='n', limits={'rpm': 5}),
                               created_by='bootstrap')
        updated = store.update(record.id, KeyPatch(limits={'rpm': 9}, note=None))
        self.assertEqual(updated.limits['rpm'], 9)
        self.assertIsNone(updated.note)
        self.assertEqual(updated.name, 'p')
        with self.assertRaises(Exception):
            KeyPatch(scopes=None)
        with self.assertRaises(Exception):
            KeySpec(name='x', scopes=['not-a-scope'])
        with self.assertRaises(Exception):
            KeySpec(name='x', scopes=['inference'], unknown=1)
        with self.assertRaises(Exception):
            KeySpec(name='sk-' + 'a' * 30, scopes=['inference'])  # secret-shaped names are refused
        with self.assertRaises(BrokerError) as caught:
            store.get('aaaaaaaaaa')
        self.assertEqual(caught.exception.status, 404)

    def test_audit_records_lifecycle_without_secrets(self):
        store = self.store()
        record, token = store.mint(KeySpec(name='aud', scopes=['inference']), created_by='bootstrap')
        _, token2 = store.rotate(record.id)
        store.update(record.id, KeyPatch(note='x'))
        store.revoke(record.id)
        kinds = [r['kind'] for r in self.audit.query()]
        self.assertEqual(kinds, ['key.mint', 'key.rotate', 'key.update', 'key.revoke'])
        self.assertTrue(self.audit.verify().ok)
        text = self.tree_text()
        for value in (token, token2):
            self.assertNotIn(value.split('_', 2)[2], text)

    @unittest.skipUnless(os.name == 'posix' or keys_module._acl() is not None, 'no ACL inspection on this platform yet')
    def test_keys_file_is_owner_only(self):
        store = self.store()
        store.mint(KeySpec(name='acl', scopes=['inference']), created_by='bootstrap')
        self.assertIs(is_private(self.path), True)


class Rules(KeyStoreCase):
    def test_broker_role_requires_budget_and_expiry_for_egress_scopes(self):
        broker = dataclasses.replace(self.settings, role='broker')
        store = self.store(broker, bootstrap_key_present=True)
        soon = NOW + timedelta(days=30)
        cases = [
            dict(name='a', scopes=['inference', 'cloud:openai'], expires_at=soon),  # no budget
            dict(name='b', scopes=['cloud'], budget={'usd_per_day': '1'}),  # no expiry
            dict(name='c', scopes=['cloud:store'], budget={'usd_per_day': '1'}, expires_at=NOW + timedelta(days=400)),
            dict(name='d', scopes=['admin'], expires_at=soon),
            dict(name='e', scopes=['cloud'], budget={'hard': True}, expires_at=soon),  # a budget without any bound
        ]
        for spec in cases:
            with self.subTest(spec=spec['name']):
                with self.assertRaises(BrokerError) as caught:
                    store.mint(KeySpec(**spec), created_by='bootstrap')
                self.assertEqual((caught.exception.code, caught.exception.status), ('budget_required', 400))
        self.assertFalse(self.path.exists() and json.loads(self.path.read_text())['keys'])
        ok, _ = store.mint(KeySpec(name='ok', scopes=['inference', 'cloud:openai'], budget={'usd_per_day': '5.00'},
                                   expires_at=soon), created_by='bootstrap')
        local, _ = store.mint(KeySpec(name='local', scopes=['inference']), created_by='bootstrap')
        with self.assertRaises(BrokerError) as caught:
            store.update(ok.id, KeyPatch(expires_at=None))
        self.assertEqual(caught.exception.code, 'budget_required')
        with self.assertRaises(BrokerError):
            store.update(local.id, KeyPatch(scopes=['cloud']))
        # Standalone role: the same specs mint, and summary() reports them for serve's broker-role precondition.
        plain = self.store()
        record, _ = plain.mint(KeySpec(name='free', scopes=['cloud']), created_by='bootstrap')
        self.assertIn(record.id, plain.summary().cloud_keys_missing_budget)
        self.assertNotIn(ok.id, plain.summary().cloud_keys_missing_budget)

    def test_summary_is_the_types_keystoresummary_serve_preconditions_take(self):
        # R13: KeyStore.summary() returns types.KeyStoreSummary (frozen, secret-free) for role.check_serve_preconditions.
        from titanos.broker.role import check_serve_preconditions
        from titanos.broker.types import KeyStoreSummary
        store = self.store()
        empty = store.summary()
        self.assertIs(type(empty), KeyStoreSummary)
        self.assertEqual(empty, KeyStoreSummary())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            empty.active = 1
        admin, token = store.mint(KeySpec(name='adm', scopes=['admin']), created_by='cli')
        free, _ = store.mint(KeySpec(name='free', scopes=['cloud']), created_by='cli')
        summary = store.summary()
        self.assertIs(type(summary), KeyStoreSummary)
        self.assertEqual((summary.has_records, summary.active, summary.active_admin), (True, 2, 1))
        self.assertEqual(set(summary.cloud_keys_missing_budget), {admin.id, free.id})
        self.assertNotIn(token, repr(summary))
        self.assertEqual(check_serve_preconditions(self.settings, bootstrap_key=None, keystore=summary,
                                                   explicit_port=False, vault_backend='refs-only', audit_ok=True), [])
        broker = dataclasses.replace(self.settings, role='broker')
        with self.assertRaises(ValueError) as caught:
            check_serve_preconditions(broker, bootstrap_key='b' * 32, keystore=summary, explicit_port=False,
                                      vault_backend='refs-only', audit_ok=True)
        self.assertIn(free.id, str(caught.exception))

    def test_last_admin_key_guard_on_network_bind(self):
        store = self.store(bind_loopback=False)
        first, _ = store.mint(KeySpec(name='admin1', scopes=['admin']), created_by='cli')
        with self.assertRaises(BrokerError) as caught:
            store.revoke(first.id)
        self.assertEqual((caught.exception.code, caught.exception.status), ('last_key_required', 409))
        self.assertTrue(caught.exception.no_retry)
        with self.assertRaises(BrokerError):
            store.update(first.id, KeyPatch(scopes=['inference']))
        with self.assertRaises(BrokerError):
            store.update(first.id, KeyPatch(expires_at=NOW - timedelta(seconds=1)))
        second, _ = store.mint(KeySpec(name='admin2', scopes=['keys:admin']), created_by='cli')
        store.revoke(first.id)
        self.assertIsNotNone(store.get(first.id).revoked_at)
        with self.assertRaises(BrokerError):
            store.revoke(second.id)
        # With a bootstrap key, or on a loopback bind, the guard does not apply.
        self.store(bind_loopback=False, bootstrap_key_present=True).revoke(second.id)
        loop = self.store(bind_loopback=True)
        third, _ = loop.mint(KeySpec(name='admin3', scopes=['admin']), created_by='cli')
        loop.revoke(third.id)
        summary = loop.summary()
        self.assertEqual((summary.has_records, summary.active, summary.active_admin), (True, 0, 0))

    def test_to_principal_scopes_constraints_limits_and_budget(self):
        store = self.store()
        record, _ = store.mint(KeySpec(name='p', scopes=['inference', 'cloud:openai', 'models:read'],
                                       constraints={'models': ['gpt-*@openai', 'qwen3-*'], 'upstreams': ['openai'],
                                                    'target_classes': ['local', 'cloud'], 'locations': ['auto']},
                                       limits={'rpm': 60, 'tpm': 200000, 'concurrent': 4},
                                       budget={'usd_per_day': '5.00', 'tokens_per_day': 1000},
                                       default_location='prefer_local', expires_at=NOW + timedelta(days=10)),
                               created_by='bootstrap')
        p = store.to_principal(record)
        self.assertEqual((p.id, p.kind, p.key_id, p.name), (record.id, 'key', record.id, 'p'))
        self.assertEqual(p.scopes, frozenset({'inference', 'cloud:openai', 'models:read'}))
        self.assertEqual(p.constraints.models, ('gpt-*@openai', 'qwen3-*'))
        self.assertEqual(p.constraints.upstreams, ('openai',))
        self.assertEqual(p.constraints.target_classes, frozenset({'local', 'cloud'}))
        self.assertEqual(p.constraints.locations, frozenset({'auto'}))
        self.assertEqual((p.limits.rpm, p.limits.tpm, p.limits.concurrent), (60, 200000, 4))
        from decimal import Decimal
        self.assertEqual(p.budget.usd_per_day, Decimal('5.00'))
        self.assertEqual(p.budget.tokens_per_day, 1000)
        self.assertEqual(p.default_location, 'prefer_local')
        self.assertEqual(p.expires_at, NOW + timedelta(days=10))
        parent, _ = store.mint(KeySpec(name='parent', scopes=['inference', 'delegate']), created_by='bootstrap')
        self.assertEqual(store.to_principal(parent).kind, 'parent')
        default, _ = store.mint(KeySpec(name='d', scopes=['inference']), created_by='bootstrap')
        dp = store.to_principal(default)
        self.assertEqual(dp.constraints.models, ('*',))
        self.assertEqual(dp.constraints.target_classes, frozenset({'local', 'cloud', 'router'}))
        self.assertIsNone(dp.limits)
        self.assertIsNone(dp.budget)


class Reload(KeyStoreCase):
    def test_second_instance_mint_is_seen_after_the_mtime_check(self):
        server = self.store()
        self.assertFalse(server.has_records())
        cli = KeyStore(self.path, self.settings, self.audit, clock=self.clock, offline=True)  # the offline CLI
        record, token = cli.mint(KeySpec(name='offline', scopes=['inference']), created_by='cli')
        # within the one-second pacing window the server still answers from its table
        self.assertFalse(server.has_records())
        self.assertIsNone(server.verify(token))
        self.clock.mono += 1.5
        self.assertTrue(server.has_records())
        self.assertEqual(server.verify(token).id, record.id)
        cli.revoke(record.id)
        self.clock.mono += 1.5
        with self.assertRaises(BrokerError) as caught:
            server.verify(token)
        self.assertEqual(caught.exception.code, 'key_revoked')

    def test_has_records_stays_true_when_every_key_expired(self):
        store = self.store()
        store.mint(KeySpec(name='e', scopes=['admin'], expires_at=NOW + timedelta(seconds=10)), created_by='cli')
        self.clock.advance(3600)
        self.assertTrue(store.has_records())
        self.assertFalse(store.has_active())
        summary = store.summary()
        self.assertTrue(summary.has_records)
        self.assertEqual((summary.active, summary.active_admin), (0, 0))

    def test_unreadable_table_fails_closed(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{not json', encoding='utf-8')
        store = self.store()
        with self.assertLogs('titanos.broker', 'ERROR'):
            self.assertTrue(store.has_records())
        self.assertIsNone(store.verify('tk_' + 'a' * 10 + '_' + 'A' * 43))
        with self.assertRaises(BrokerError), self.assertLogs('titanos.broker', 'ERROR'):
            store.mint(KeySpec(name='x', scopes=['inference']), created_by='cli')

    def test_reset_keyless_is_offline_only_and_audited(self):
        server = self.store()
        server.mint(KeySpec(name='a', scopes=['admin']), created_by='cli')
        with self.assertRaises(BrokerError):
            server.reset_keyless()
        cli = self.store(offline=True)
        self.assertEqual(cli.reset_keyless(), 1)
        self.assertFalse(self.path.exists())
        self.clock.mono += 1.5
        self.assertFalse(server.has_records())
        self.assertEqual(self.audit.query(kind='key.reset_keyless')[-1]['detail'], {'removed': 1})

    def test_enforcement_survives_a_deleted_table(self):
        store = self.store()
        store.mint(KeySpec(name='a', scopes=['admin']), created_by='cli')
        self.assertTrue(store.marker_path.is_file())
        self.path.unlink()  # a manual delete or sync tooling, not the offline reset
        self.clock.mono += 1.5
        with self.assertLogs('titanos.broker', 'ERROR'):
            self.assertTrue(store.has_records(), 'only reset_keyless returns to keyless')
        with self.assertLogs('titanos.broker', 'ERROR'):
            self.assertTrue(self.store().summary().has_records)
        cli = self.store(offline=True)
        self.assertEqual(cli.reset_keyless(), 0)
        self.assertFalse(store.marker_path.exists())
        self.clock.mono += 1.5
        self.assertFalse(store.has_records())

    def test_a_failed_audit_append_persists_nothing(self):
        store = self.store()
        record, _ = store.mint(KeySpec(name='a', scopes=['inference']), created_by='cli')
        before = self.path.read_bytes()
        day_file = next((self.home / 'broker' / 'audit').glob('*.jsonl'))
        with day_file.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write('{"seq": 4, "ts"')  # a torn final line: the chain tail is unreadable
        for change in (lambda: store.mint(KeySpec(name='b', scopes=['inference']), created_by='cli'),
                       lambda: store.revoke(record.id), lambda: store.rotate(record.id),
                       lambda: store.update(record.id, KeyPatch(note='x'))):
            with self.assertRaises(AuditChainError):
                change()
        self.assertEqual(self.path.read_bytes(), before, 'no unaudited change and no orphan key')
        self.assertEqual([r.id for r in store.list()], [record.id])
        self.assertIsNone(store.get(record.id).revoked_at)

    def test_a_transient_read_error_clears_on_the_next_recheck(self):
        store = self.store()
        record, token = store.mint(KeySpec(name='t', scopes=['inference']), created_by='cli')
        self.clock.mono += 1.5
        os.utime(self.path, ns=(time.time_ns(), time.time_ns() + 10_000_000))  # force a reload
        real = Path.read_text

        def flaky(path, *a, **kw):
            if path == self.path:
                raise PermissionError('sharing violation')
            return real(path, *a, **kw)
        with patch.object(Path, 'read_text', flaky), self.assertLogs('titanos.broker', 'ERROR'):
            self.assertIsNone(store.verify(token), 'fails closed while unreadable')
        self.assertTrue(store.has_records())
        self.clock.mono += 1.5
        self.assertEqual(store.verify(token).id, record.id, 'the next recheck reads the file again')

    def test_last_used_at_never_waits_for_a_busy_lock(self):
        store = self.store()
        record, token = store.mint(KeySpec(name='busy', scopes=['inference']), created_by='cli')
        other = threading.Event()
        done = threading.Event()

        def hold():
            with store._lock:
                other.set()
                done.wait(5)
        holder = threading.Thread(target=hold)
        holder.start()
        try:
            other.wait(5)
            with patch.object(KeyStore, '_refresh'), patch.object(KeyStore, '_write') as write:
                started = time.monotonic()
                self.assertEqual(store.verify(token).id, record.id)
                self.assertLess(time.monotonic() - started, 1.0)
                write.assert_not_called()
        finally:
            done.set()
            holder.join()
        with keys_module.broker_lock(self.path, timeout=1):  # the CLI holds the file lock
            self.clock.advance(120)
            started = time.monotonic()
            self.assertEqual(store.verify(token).id, record.id)
            self.assertLess(time.monotonic() - started, 1.0)
        self.assertIsNone(store.get(record.id).last_used_at)

    @unittest.skipUnless(os.name == 'posix', 'POSIX modes only')
    def test_an_existing_open_broker_directory_is_tightened(self):
        self.path.parent.mkdir(parents=True)
        os.chmod(self.path.parent, 0o755)  # what config.broker_lock's plain mkdir can leave behind
        self.store().mint(KeySpec(name='m', scopes=['inference']), created_by='cli')
        self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)

    def test_last_used_at_is_written_at_most_once_per_minute(self):
        store = self.store()
        record, token = store.mint(KeySpec(name='u', scopes=['inference']), created_by='cli')
        with patch.object(KeyStore, '_write', wraps=store._write) as write:
            store.verify(token)
            store.verify(token)
            self.clock.advance(30)
            store.verify(token)
            self.assertEqual(write.call_count, 1)
            self.clock.advance(31)
            store.verify(token)
            self.assertEqual(write.call_count, 2)
        stored = json.loads(self.path.read_text(encoding='utf-8'))['keys'][record.id]
        self.assertTrue(re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', stored['last_used_at']))


if __name__ == '__main__':
    unittest.main()
