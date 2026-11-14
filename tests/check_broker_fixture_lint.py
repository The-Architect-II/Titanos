"""Fixture linter for the broker wire fixtures and mocks (WP4).

Every file under tests/fixtures/providers/ and tests/broker_mocks/ is scanned for:
- provider secret shapes (sk-, sk-proj-, sk-ant-, xai-, gsk_, nvapi-, hf_, AIza, tk_<id>_<43> tokens; the shapes of
  titanos.broker.redact.SECRET_PATTERNS) and 'Bearer <token>' whose token is not a SENTINELS value, a recorder placeholder
  (<sentinel:...>/<foreign:...>) or an ALL-CAPS placeholder such as YOUR_KEY;
- credential header values (authorization, x-api-key, xi-api-key, api-key, x-goog-api-key, cookie) that are not sentinels;
- real provider hostnames (titanos.broker.profiles.hostnames()) in a fixture object that also carries a credential header with
  a non-sentinel value;
- RFC1918 IPv4 addresses, drive-letter paths and every hygiene class of tools/public_export.json detection_patterns;
- in fixture rows (not the header's source_url): URL hosts other than 127.0.0.1, example.invalid (and subdomains) and
  192.0.2.0/24.
The linter's own self-tests build each offending sample at runtime, so no offending literal exists in a repository file.
"""
import ipaddress
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from titanos.broker.profiles import hostnames  # noqa: E402
from titanos.broker.redact import SECRET_PATTERNS  # noqa: E402
from broker_mocks.fixtures import FIXTURES_DIR, SENTINELS  # noqa: E402

MOCKS_DIR = ROOT / 'tests' / 'broker_mocks'
POLICY = json.loads((ROOT / 'tools' / 'public_export.json').read_text(encoding='utf-8'))
HYGIENE = {name: re.compile(pattern) for name, pattern in POLICY['detection_patterns'].items()}


def _shape(prefix: str) -> re.Pattern:
    """The one redact pattern whose literal prefix is `prefix` (selected by text, not by position)."""
    found = [pattern for pattern in SECRET_PATTERNS if (prefix + '[') in pattern.pattern]
    if len(found) != 1:
        raise LookupError(f'expected exactly one redact pattern for the {prefix!r} key shape, found {len(found)}')
    return found[0]


SHAPES = {name: _shape(prefix) for name, prefix in (('sk-ant', 'sk-ant-'), ('sk-proj', 'sk-proj-'), ('sk', 'sk-'),
                                                    ('xai', 'xai-'), ('gsk', 'gsk_'), ('nvapi', 'nvapi-'), ('hf', 'hf_'),
                                                    ('AIza', 'AIza'), ('tk', 'tk_'))}
BEARER = re.compile(r'(?i)\bbearer\s+([A-Za-z0-9._~+/=<>:{}\[\]\'-]{8,})')
DRIVE_PATH = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]')
RFC1918 = HYGIENE['rfc1918_ipv4']
URL = re.compile(r'(?i)\b(?:https?|wss?)://([^/\s"\'?#<>\\]+)')
CREDENTIAL_KEYS = {'authorization', 'proxy-authorization', 'x-api-key', 'xi-api-key', 'api-key', 'x-goog-api-key', 'cookie'}
# accepted in place of a credential: recorder spellings, ALL-CAPS placeholders (YOUR_KEY) and any '<...>' or '{...}'
# documentation placeholder (no provider secret starts with '<' or '{')
PLACEHOLDER = re.compile(r'^(<.*|\{.*|[A-Z][A-Z0-9_]{2,})$')
ALLOWED_HOSTS = ('127.0.0.1', 'example.invalid')
PROVIDER_HOSTS = hostnames()


def acceptable_token(token: str) -> bool:
    return token in SENTINELS.values() or bool(PLACEHOLDER.match(token))


def acceptable_credential(value) -> bool:
    if not isinstance(value, str):
        return False
    token = re.sub(r'(?i)^(bearer|basic|token)\s+', '', value.strip())
    return acceptable_token(token)


def allowed_host(host: str) -> bool:
    host = host.rsplit('@', 1)[-1]
    host = host[1:host.index(']')] if host.startswith('[') else host.split(':', 1)[0]
    host = host.lower().rstrip('.')
    if host in ALLOWED_HOSTS or host.endswith('.example.invalid'):
        return True
    try:
        return ipaddress.ip_address(host) in ipaddress.ip_network('192.0.2.0/24')
    except ValueError:
        return False


def _credentials(value) -> list:
    """Every (credential key, value) pair anywhere inside a JSON value."""
    out = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in CREDENTIAL_KEYS:
                out.append((key, item))
            out += _credentials(item)
    elif isinstance(value, list):
        for item in value:
            out += _credentials(item)
    return out


def _walk(value, found):
    """(credential pairs in the object's subtree, provider hostnames named in it) per JSON object, recursively."""
    if isinstance(value, dict):
        text = json.dumps(value)
        found.append((_credentials(value), [h for h in PROVIDER_HOSTS if h in text]))
        for item in value.values():
            _walk(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk(item, found)


def lint_text(name: str, text: str, *, fixture: bool) -> list:
    """[(name, line, class)] for every finding (no matched text)."""
    findings = []

    def add(pos, kind):
        findings.append((name, text.count('\n', 0, pos) + 1, kind))

    for kind, pattern in SHAPES.items():
        for match in pattern.finditer(text):
            if not acceptable_token(match.group(0)):
                add(match.start(), 'secret_shape:' + kind)
    for match in BEARER.finditer(text):
        if not acceptable_token(match.group(1).rstrip('.,;)"\'')):
            add(match.start(), 'bearer_non_sentinel')
    for kind, pattern in HYGIENE.items():
        for match in pattern.finditer(text):
            add(match.start(), 'hygiene:' + kind)
    for match in DRIVE_PATH.finditer(text):
        add(match.start(), 'drive_letter_path')
    if not fixture:
        return findings
    lines = text.split('\n')
    offset = 0
    for index, line in enumerate(lines):
        start, offset = offset, offset + len(line) + 1
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            add(start, 'invalid_json_row')
            continue
        scanned = row
        if index == 0 and isinstance(row, dict):
            scanned = {k: v for k, v in row.items() if k != 'source_url'}
        for match in URL.finditer(json.dumps(scanned)):
            if not allowed_host(match.group(1)):
                add(start, 'foreign_host')
        found = []
        _walk(scanned, found)
        for creds, hosts in found:
            bad = [k for k, v in creds if not acceptable_credential(v)]
            if bad:
                add(start, 'credential_header_non_sentinel')
                if hosts:
                    add(start, 'provider_host_with_credential')
    return findings


def lint_tree() -> list:
    findings = []
    for base, fixture in ((FIXTURES_DIR, True), (MOCKS_DIR, False)):
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix in ('.jsonl', '.py', '.json', '.md', '.txt'):
                name = path.relative_to(ROOT).as_posix()
                findings += lint_text(name, path.read_text(encoding='utf-8'), fixture=path.suffix == '.jsonl')
    return findings


def _header_row(**extra):
    row = {'source_url': 'https://docs.example.invalid/ref', 'retrieved': '2026-10-01', 'transcriber_note': 'x', 'kind': 'json'}
    row.update(extra)
    return json.dumps(row)


class TreeIsClean(unittest.TestCase):
    def test_fixtures_and_mocks_pass_the_linter(self):
        self.assertTrue(list(FIXTURES_DIR.glob('*/*.jsonl')), 'no fixtures found')
        findings = lint_tree()
        self.assertFalse(findings, '; '.join(f'{n}:{line} {kind}' for n, line, kind in findings[:40]))

    def test_sentinels_are_not_secret_shaped(self):
        for value in SENTINELS.values():
            self.assertGreaterEqual(len(value), 8)
            self.assertFalse(any(p.search(value) for p in SHAPES.values()), value)


class LinterCatchesEachClass(unittest.TestCase):
    """Each offending sample is assembled at runtime and must be flagged with its class."""

    def kinds(self, text, fixture=True):
        return {kind for _, _, kind in lint_text('sample.jsonl', text, fixture=fixture)}

    def test_secret_shapes(self):
        filler = 'Q7' * 24
        samples = {
            'sk': 's' + 'k-' + filler,
            'sk-proj': 's' + 'k-proj-' + filler,
            'sk-ant': 's' + 'k-ant-api03-' + filler,
            'xai': 'x' + 'ai-' + filler,
            'gsk': 'g' + 'sk_' + filler,
            'nvapi': 'n' + 'vapi-' + filler,
            'hf': 'h' + 'f_' + filler,
            'AIza': 'A' + 'Iza' + filler[:35],
            'tk': 't' + 'k_abcdefgh23_' + (filler * 2)[:43],
        }
        for kind, secret in samples.items():
            text = _header_row() + '\n' + json.dumps({'body': {'note': secret}}) + '\n'
            self.assertIn('secret_shape:' + kind, self.kinds(text), kind)

    def test_bearer_non_sentinel_and_credential_headers(self):
        token = 'opaque' + 'Token' + '0123456789'
        text = _header_row() + '\n' + json.dumps({'body': {'echo': 'B' + 'earer ' + token}}) + '\n'
        self.assertIn('bearer_non_sentinel', self.kinds(text))
        ok = _header_row() + '\n' + json.dumps({'body': {'echo': 'Bearer ' + SENTINELS['openai'],
                                                         'other': 'Bearer YOUR_KEY', 'rec': 'Bearer <sentinel:openai>'}})
        self.assertEqual(self.kinds(ok + '\n'), set())
        header = _header_row(request={'path': '/v1/x', 'headers': {'x-api-key': token}})
        self.assertIn('credential_header_non_sentinel', self.kinds(header + '\n' + json.dumps({'body': {}}) + '\n'))

    def test_provider_host_with_non_sentinel_credential(self):
        host = sorted(PROVIDER_HOSTS)[0]
        row = {'request': {'url': 'https://' + host + '/v1/chat/completions', 'headers': {'xi-api-key': 'live' + 'value99'}}}
        kinds = self.kinds(_header_row() + '\n' + json.dumps(row) + '\n')
        self.assertIn('provider_host_with_credential', kinds)
        self.assertIn('foreign_host', kinds)
        sentinel_row = {'request': {'note': host, 'headers': {'xi-api-key': SENTINELS['elevenlabs']}}}
        self.assertNotIn('provider_host_with_credential', self.kinds(_header_row() + '\n' + json.dumps(sentinel_row) + '\n'))

    def test_hosts_paths_and_hygiene_classes(self):
        private = '.'.join(['10', '20', '30', '40'])
        self.assertIn('hygiene:rfc1918_ipv4', self.kinds(_header_row() + '\n' + json.dumps({'body': private}) + '\n'))
        drive = chr(68) + ':' + '/' + 'records/run.json'
        self.assertIn('drive_letter_path', self.kinds(_header_row() + '\n' + json.dumps({'body': drive}) + '\n'))
        profile = chr(67) + ':' + '\\' + 'Users' + '\\' + 'someone' + '\\x'
        self.assertIn('hygiene:windows_user_profile', self.kinds(json.dumps({'body': profile}) + '\n', fixture=False))
        home = '/' + 'home' + '/someone/x'
        self.assertIn('hygiene:linux_home', self.kinds(json.dumps({'body': home}) + '\n', fixture=False))
        foreign = 'https://' + 'api.' + 'invalid-host.test/v1'
        self.assertIn('foreign_host', self.kinds(_header_row() + '\n' + json.dumps({'body': {'url': foreign}}) + '\n'))
        for host in ('http://127.0.0.1:9/x', 'https://example.invalid/a.png', 'https://cdn.example.invalid/b',
                     'http://192.0.2.7/x'):
            self.assertEqual(self.kinds(_header_row() + '\n' + json.dumps({'body': {'url': host}}) + '\n'), set(), host)
        # the header's source_url may name the public reference host
        self.assertEqual(self.kinds(json.dumps({'source_url': 'https://' + sorted(PROVIDER_HOSTS)[0] + '/docs',
                                                'retrieved': '2026-10-01', 'transcriber_note': 'x', 'kind': 'json'})
                                    + '\n' + json.dumps({'body': {}}) + '\n'), set())


if __name__ == '__main__':
    unittest.main(verbosity=1)
