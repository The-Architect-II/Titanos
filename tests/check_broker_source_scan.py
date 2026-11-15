"""Broker source scans (WP2): where credentials, grants and network clients may appear in src/titanos.

- In src/titanos/broker the outbound auth header names (Authorization, x-api-key, xi-api-key, api-key, x-goog-api-key) are
  set only in egress.py: no subscript assignment, header dict literal, (name, value) header pair, setdefault/update or
  keyword with one of those names elsewhere. The one dynamic setter is vault.CredentialHandle.apply (it fills the
  profiles.AUTH_TEMPLATES header), and .apply( is called only in egress.py. AUTH_TEMPLATES is consulted only by
  profiles.py, vault/__init__.py, vault/store.py and egress.py.
- Secret.reveal( appears only in broker/egress.py and broker/vault/* (anywhere in src/titanos).
- EgressGrant( is constructed only in broker/router.py (GrantIssuer) anywhere in src/titanos.
- Outside the broker, the only places in src/titanos that set Authorization are native.py (2), client.py (1),
  mcp_server.py (1) and cli.py (1: the gateway registration bearer, tests/check_ops_gateway_bearer.py).
- No trust_env=True or follow_redirects=True in the broker; httpx.Client/AsyncClient are constructed only in egress.py
  and always with trust_env=False and follow_redirects=False; no module-level httpx request helper (httpx.get, ...).
- socket, ssl, urllib.request, http.client, requests and aiohttp are imported only by egress.py; subprocess only by
  vault/keychain.py (the vault helper runner) and never with shell=True.
- Provider default hostnames (profiles.hostnames()) appear only in profiles.py within the broker.
Reads source files only.
"""
import ast
from collections import Counter
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from titanos.broker.profiles import hostnames  # noqa: E402

SRC = ROOT / 'src' / 'titanos'
BROKER = SRC / 'broker'
AUTH_HEADERS = frozenset({'authorization', 'x-api-key', 'xi-api-key', 'api-key', 'x-goog-api-key'})
CONTAINERS = (ast.Tuple, ast.List, ast.Set, ast.Dict)
NETWORK_MODULES = ('socket', 'ssl', 'urllib.request', 'http.client', 'requests', 'aiohttp')
HTTPX_HELPERS = frozenset({'get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'request', 'stream'})
# cli.py: register_with_gateway's one bearer (hivemind register-service / serve --register-with-gateway), checked by
# tests/check_ops_gateway_bearer.py
OUTSIDE_AUTHORIZATION = {'native.py': 2, 'client.py': 1, 'mcp_server.py': 1, 'cli.py': 1}


def rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def sources(root: Path):
    for path in sorted(root.rglob('*.py')):
        if '__pycache__' not in path.parts:
            yield path, ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def header_name(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        value = node.value.decode('latin-1') if isinstance(node.value, bytes) else node.value
        if value.lower() in AUTH_HEADERS:
            return value.lower()
    return None


def header_sets(tree) -> list[tuple[int, str]]:
    """(line, header) for every place that sets an outbound auth header by name."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and header_name(target.slice):
                    found.append((node.lineno, header_name(target.slice)))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if header_name(key) and not isinstance(value, CONTAINERS) and not (
                        isinstance(value, ast.Constant) and value.value is None):
                    found.append((node.lineno, header_name(key)))
        elif isinstance(node, ast.Tuple) and len(node.elts) == 2 and header_name(node.elts[0]) \
                and not isinstance(node.elts[1], (ast.Constant, *CONTAINERS)):
            found.append((node.lineno, header_name(node.elts[0])))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ('setdefault', '__setitem__', 'add_header', 'putheader') \
                    and node.args and header_name(node.args[0]):
                found.append((node.lineno, header_name(node.args[0])))
            for keyword in node.keywords:
                if keyword.arg and keyword.arg.lower() in AUTH_HEADERS:
                    found.append((node.lineno, keyword.arg.lower()))
    return found


def calls(tree, attr: str) -> list[int]:
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr]


def name_calls(tree, name: str) -> list[int]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == name) or (isinstance(func, ast.Attribute) and func.attr == name):
                out.append(node.lineno)
    return out


def keyword_true(tree, keyword: str) -> list[int]:
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
            for kw in node.keywords if kw.arg == keyword and not (isinstance(kw.value, ast.Constant)
                                                                   and kw.value.value is False)]


def imports(tree) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f'{node.module}.{alias.name}' for alias in node.names)
    return names


class SourceScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.broker = {rel(path): tree for path, tree in sources(BROKER)}
        cls.everything = {rel(path): tree for path, tree in sources(SRC)}
        assert 'broker/egress.py' in cls.broker and 'broker/vault/store.py' in cls.broker

    def test_auth_headers_are_set_only_in_egress(self):
        offenders = {name: header_sets(tree) for name, tree in self.broker.items()
                     if name != 'broker/egress.py' and header_sets(tree)}
        self.assertEqual(offenders, {})

    def test_the_dynamic_setter_is_credential_handle_apply_called_from_egress(self):
        users = sorted(name for name, tree in self.broker.items()
                       if any(isinstance(node, ast.Name) and node.id == 'AUTH_TEMPLATES' for node in ast.walk(tree)))
        self.assertEqual(users, ['broker/egress.py', 'broker/profiles.py', 'broker/vault/__init__.py',
                                 'broker/vault/store.py'])
        apply_callers = sorted(name for name, tree in self.broker.items() if calls(tree, 'apply'))
        self.assertEqual(apply_callers, ['broker/egress.py'])
        vault_init = self.broker['broker/vault/__init__.py']
        setters = []
        for node in ast.walk(vault_init):
            if isinstance(node, ast.FunctionDef):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Assign) and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute)
                                                             and t.value.attr == 'headers' for t in inner.targets):
                        setters.append(node.name)
        self.assertEqual(setters, ['apply'])

    def test_reveal_only_in_egress_and_the_vault(self):
        callers = sorted(name for name, tree in self.everything.items() if calls(tree, 'reveal'))
        self.assertTrue(all(name == 'broker/egress.py' or name.startswith('broker/vault/') for name in callers), callers)
        self.assertIn('broker/vault/__init__.py', callers)

    def test_egress_grant_constructed_only_by_the_router(self):
        constructors = sorted(name for name, tree in self.everything.items() if name_calls(tree, 'EgressGrant'))
        self.assertTrue(set(constructors) <= {'broker/router.py'}, constructors)

    def test_authorization_outside_the_broker_is_the_allowlist(self):
        counts = Counter()
        for name, tree in self.everything.items():
            if name.startswith('broker/'):
                continue
            for _, header in header_sets(tree):
                counts[name] += 1
        self.assertEqual(dict(counts), OUTSIDE_AUTHORIZATION)

    def test_httpx_posture(self):
        for keyword in ('trust_env', 'follow_redirects'):
            offenders = {name: lines for name, tree in self.broker.items() if (lines := [
                line for line in keyword_true(tree, keyword)])}
            self.assertEqual(offenders, {}, f'{keyword} must only ever be passed as False in the broker')
        constructors = {}
        for name, tree in self.broker.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'httpx':
                    if node.func.attr in ('Client', 'AsyncClient'):
                        keywords = {kw.arg: kw.value for kw in node.keywords}
                        for required in ('trust_env', 'follow_redirects'):
                            value = keywords.get(required)
                            self.assertTrue(isinstance(value, ast.Constant) and value.value is False,
                                            f'{name}:{node.lineno} {node.func.attr} without {required}=False')
                        constructors.setdefault(name, []).append(node.lineno)
                    self.assertNotIn(node.func.attr, HTTPX_HELPERS, f'{name}:{node.lineno} uses httpx.{node.func.attr}')
        self.assertEqual(sorted(constructors), ['broker/egress.py'])

    def test_network_and_subprocess_imports(self):
        for name, tree in self.broker.items():
            found = imports(tree)
            network = sorted(m for m in found if m in NETWORK_MODULES or m.split('.')[0] in ('requests', 'aiohttp'))
            if name != 'broker/egress.py':
                self.assertEqual(network, [], name)
            if name != 'broker/vault/keychain.py':
                self.assertNotIn('subprocess', found, name)
            self.assertEqual(keyword_true(tree, 'shell'), [], f'{name} passes shell=True')

    def test_provider_hostnames_only_in_profiles(self):
        hosts = hostnames()
        self.assertIn('api.openai.com', hosts)
        offenders = {}
        for path in sorted(BROKER.rglob('*.py')):
            if path.name == 'profiles.py' and path.parent == BROKER:
                continue
            text = path.read_text(encoding='utf-8').lower()
            hits = sorted(host for host in hosts if host in text)
            if hits:
                offenders[rel(path)] = hits
        self.assertEqual(offenders, {})

    def test_scanner_catches_each_setting_form(self):
        samples = {
            "headers['Authorization'] = 'Bearer ' + key": 1,
            "h = {'x-api-key': key}": 1,
            "h = {b'xi-api-key': ('/v1/voices',)}": 0,
            "pairs = [(b'authorization', b'Bearer ' + key)]": 1,
            "AUTH = {'bearer': ('Authorization', 'Bearer {secret}')}": 0,
            "headers.setdefault('api-key', key)": 1,
            "headers.update(Authorization=key)": 1,
            "value = headers.get('x-goog-api-key')": 0,
        }
        for source, expected in samples.items():
            with self.subTest(source=source):
                self.assertEqual(len(header_sets(ast.parse(source))), expected)


if __name__ == '__main__':
    unittest.main()
