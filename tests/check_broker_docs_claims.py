"""Every support claim in the broker documentation maps to a cell of the broker matrix (docs/broker_matrix.json).

The documents carry their evidence as HTML comments next to the statement (invisible when rendered):
- <!-- matrix: ID[=MODE], ... -->          each cell exists, its state is pass or live-pass, and its mode is MODE when given;
- <!-- matrix-untried: ID[=MODE], ... -->  the cell's state is untried;
- <!-- matrix-na: ID[=MODE], ... -->       the cell's state is na;
- <!-- matrix-live: ID, ... -->            the cell's state is live-pass;
- <!-- matrix-count: STATE[@PROFILE]=N --> exactly N cells (of that provider profile) have that state;
- <!-- matrix-deferred: NAME, ... -->      each name is a deferred profile of the matrix vocabulary;
- <!-- claims-legend -->                   the next table defines the vocabulary and is not itself a claim.
Inside Markdown tables a '|' in a cell id is written '\\|'.

Checks:
- every token in docs/*.md (except the generated BROKER_MATRIX.md) resolves as above, and no unknown matrix-* token kind
  is used;
- no support claim without a cell: in docs/BROKER*.md and docs/TOPOLOGY_ROLES.md every block (paragraph, list item or
  table row, outside fenced code) that uses a claim word (supported, supports, mock-validated, live-validated, proven,
  untried, live-pass, pass (mock)) carries at least one matrix token;
- drift against the code: the BROKER.md provider table lists every profile of broker.profiles exactly once; its error
  table lists every code of broker.errors with the table's HTTP status and no-retry flag; docs/CONFIGURATION.md names every
  broker, role, TLS and managed setting by its TITANOS_ variable; docs/TOPOLOGY_ROLES.md names every reserved managed port;
  docs/MCP.md names every broker MCP tool;
- relative links of the broker documents resolve to existing files and headings;
- the documents carry no lab locator (tools/public_export.json detection patterns) and not the unannounced
  component name of decision 2;
- the parser itself on synthetic text: an unbacked claim, a missing cell, a wrong state and a wrong mode are each found.
Static: reads files only; the netguard is installed and no connection is attempted."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from broker_mocks import netguard  # noqa: E402

netguard.install()

from titanos.broker.errors import ERROR_CODES  # noqa: E402
from titanos.broker.mcp_tools import TOOL_NAMES  # noqa: E402  (imports mcp, a base dependency)
from titanos.broker.profiles import PROFILES  # noqa: E402
from titanos.broker.role import RESERVED_PORTS  # noqa: E402
from titanos.settings import Settings  # noqa: E402

MATRIX = DOCS / 'broker_matrix.json'
GENERATED = {'BROKER_MATRIX.md'}
CLAIM_DOCS = ('BROKER.md', 'BROKER_SECURITY.md', 'TOPOLOGY_ROLES.md')
TOKEN = re.compile(r'<!--\s*(matrix(?:-[a-z]+)?|claims-legend)\s*(?::\s*(.*?))?\s*-->', re.S)
KINDS = {'matrix': ('pass', 'live-pass'), 'matrix-untried': ('untried',), 'matrix-na': ('na',),
         'matrix-live': ('live-pass',)}
CLAIM_WORDS = re.compile(r'(?i)\b(supported|supports|mock-validated|live-validated|proven|untried|live-pass)\b|'
                         r'pass \(mock')
BROKER_SETTING = re.compile(r'^(role|broker_.+|tls_certfile|tls_keyfile|behind_tls_proxy|managed_.+)$')
LINK = re.compile(r'\]\(([^)\s]+)\)')


def load_cells(path=MATRIX):
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['cells'], data['vocabulary']


def tokens(text):
    """(kind, argument string, line number) for every claim token of a document."""
    out = []
    for match in TOKEN.finditer(text):
        line = text.count('\n', 0, match.start()) + 1
        out.append((match.group(1), (match.group(2) or '').replace('\\|', '|').strip(), line))
    return out


def resolve(kind, argument, cells, vocabulary):
    """Problems of one token (empty when it holds)."""
    problems = []
    if kind == 'claims-legend':
        return problems
    if kind == 'matrix-count':
        found = re.fullmatch(r'([a-z-]+)(?:@([a-z0-9_]+))?=(\d+)', argument)
        if not found:
            return [f'malformed matrix-count {argument!r}']
        state, profile, wanted = found.group(1), found.group(2), int(found.group(3))
        count = sum(1 for cell in cells.values() if cell['state'] == state and (
            profile is None or cell.get('profile') == profile))
        if count != wanted:
            problems.append(f'matrix-count {argument}: the matrix has {count}')
        return problems
    if kind == 'matrix-deferred':
        deferred = set(vocabulary.get('deferred_profiles', ()))
        for name in (part.strip() for part in argument.split(',')):
            if name not in deferred:
                problems.append(f'{name!r} is not a deferred profile of the matrix')
        return problems
    if kind not in KINDS:
        return [f'unknown token kind {kind!r}']
    if not argument:
        return [f'{kind} names no cell']
    for item in (part.strip() for part in argument.split(',')):
        cell_id, _, mode = item.partition('=')
        cell = cells.get(cell_id)
        if cell is None:
            problems.append(f'{kind}: no cell {cell_id!r}')
            continue
        if cell['state'] not in KINDS[kind]:
            problems.append(f'{kind}: cell {cell_id!r} is {cell["state"]}')
        if mode and cell.get('mode') != mode:
            problems.append(f'{kind}: cell {cell_id!r} has mode {cell.get("mode")}, the document says {mode}')
    return problems


def blocks(text):
    """Claim units outside fenced code: each table row, each list item with its continuation lines, each paragraph.
    Yields (first line number, block text, exempt); exempt marks the block (a whole table, or one paragraph or list item)
    that directly follows a claims-legend marker."""
    lines = text.split('\n')
    out, current, start, fenced, legend, in_table = [], [], 0, False, False, False

    def flush():
        nonlocal current, legend
        if current:
            out.append((start, '\n'.join(current), legend))
            legend = False
        current = []

    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('```'):
            flush()
            fenced = not fenced
            continue
        if fenced:
            continue
        if stripped.startswith('|'):
            flush()
            out.append((number, line, legend))
            in_table = True
            continue
        if in_table:
            legend = False
            in_table = False
        if TOKEN.fullmatch(stripped) and 'claims-legend' in stripped:
            flush()
            legend = True
            continue
        if not stripped or stripped.startswith('#'):
            flush()
            continue
        if re.match(r'(- |\d+\. )', stripped):
            flush()
        if not current:
            start = number
        current.append(line)
    flush()
    return out


def unbacked_claims(text):
    """Blocks that use a claim word and carry no matrix token."""
    return [(number, block.strip()[:120]) for number, block, exempt in blocks(text)
            if not exempt and CLAIM_WORDS.search(TOKEN.sub('', block)) and not any(
                kind != 'claims-legend' for kind, _, _ in tokens(block))]


def slug(heading):
    text = re.sub(r'<[^>]+>', '', heading.strip().lower())
    text = re.sub(r'[^\w\- ]', '', text)
    return text.replace(' ', '-')


def anchors(path):
    return {slug(line.lstrip('#')) for line in path.read_text(encoding='utf-8').splitlines() if line.startswith('#')}


def table_rows(text, header):
    """Rows of the first Markdown table whose header line starts with header, split into stripped cells."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(header):
            rows = []
            for row in lines[index + 2:]:
                if not row.startswith('|'):
                    break
                cells = [cell.strip() for cell in re.split(r'(?<!\\)\|', row.strip())[1:-1]]
                rows.append(cells)
            return rows
    return []


class DocsClaims(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells, cls.vocabulary = load_cells()
        cls.docs = {path.name: path.read_text(encoding='utf-8') for path in sorted(DOCS.glob('*.md'))
                    if path.name not in GENERATED}

    def setUp(self):
        netguard.reset_attempts()

    def tearDown(self):
        self.assertEqual(netguard.attempts(), [], 'a non-loopback connection was attempted')

    def test_every_token_resolves(self):
        seen = 0
        for name, text in self.docs.items():
            for kind, argument, line in tokens(text):
                seen += 1
                for problem in resolve(kind, argument, self.cells, self.vocabulary):
                    self.fail(f'docs/{name}:{line}: {problem}')
        self.assertGreater(seen, 50, 'the broker documents carry their matrix tokens')

    def test_claim_documents_cite_cells(self):
        for name in CLAIM_DOCS:
            text = self.docs[name]
            self.assertTrue(any(kind.startswith('matrix') for kind, _, _ in tokens(text)), f'{name} cites no cell')
            self.assertTrue('BROKER_MATRIX.md' in text, f'{name} does not point to the matrix')

    def test_no_support_claim_without_a_cell(self):
        names = sorted(n for n in self.docs if n.startswith('BROKER') or n == 'TOPOLOGY_ROLES.md')
        self.assertEqual(set(CLAIM_DOCS) - set(names), set())
        found = [f'docs/{name}:{number}: {block!r}' for name in names for number, block in unbacked_claims(self.docs[name])]
        self.assertEqual(found, [], 'support claims without a matrix cell')

    def test_provider_table_lists_every_profile(self):
        rows = table_rows(self.docs['BROKER.md'], '| Profile | Kind |')
        listed = [re.match(r'`([a-z0-9_]+)`', row[0]).group(1) for row in rows]
        self.assertEqual(sorted(listed), sorted(PROFILES), 'BROKER.md provider table and broker.profiles disagree')
        self.assertEqual(len(listed), len(set(listed)))
        for row in rows:
            profile = re.match(r'`([a-z0-9_]+)`', row[0]).group(1)
            self.assertEqual(row[1], PROFILES[profile].kind, f'{profile}: kind')

    def test_error_table_matches_the_code_table(self):
        rows = table_rows(self.docs['BROKER.md'], '| Code | HTTP | No retry |')
        documented = {}
        for row in rows:
            code = row[0].strip('`')
            documented[code] = (int(row[1]), row[2] == 'yes')
        expected = {code: (spec.status, spec.no_retry) for code, spec in ERROR_CODES.items()}
        self.assertEqual(sorted(documented), sorted(expected), 'BROKER.md error table and broker.errors disagree')
        for code, value in expected.items():
            self.assertEqual(documented[code], value, f'{code}: (status, no_retry)')

    def test_configuration_names_every_broker_setting(self):
        text = self.docs['CONFIGURATION.md']
        names = [f.name for f in Settings.__dataclass_fields__.values() if BROKER_SETTING.match(f.name)]
        self.assertGreater(len(names), 30)
        for name in names:
            self.assertTrue('TITANOS_' + name.upper() in text, f'docs/CONFIGURATION.md does not document {name}')
        for extra in ('TITANOS_BEHIND_TLS_PROXY', 'forwarded_allow_ips'):
            self.assertTrue(extra in text, f'docs/CONFIGURATION.md does not mention {extra}')

    def test_topology_names_the_reserved_ports(self):
        text = self.docs['TOPOLOGY_ROLES.md']
        for port in sorted(RESERVED_PORTS):
            self.assertTrue(str(port) in text, f'docs/TOPOLOGY_ROLES.md does not name reserved port {port}')
        for needle in ('--role managed', '{bind}'):
            self.assertTrue(needle in text, f'docs/TOPOLOGY_ROLES.md does not mention {needle}')

    def test_mcp_doc_names_every_broker_tool(self):
        text = self.docs['MCP.md']
        for tool in TOOL_NAMES:
            self.assertTrue(f'`{tool}`' in text, f'docs/MCP.md does not name the {tool} tool')

    def test_links_resolve(self):
        for name in (*CLAIM_DOCS, 'README.md'):
            path = (ROOT / name) if name == 'README.md' else DOCS / name
            text = path.read_text(encoding='utf-8')
            for target in LINK.findall(text):
                if re.match(r'[a-z]+:', target):
                    continue
                file_part, _, anchor = target.partition('#')
                linked = (path.parent / file_part).resolve() if file_part else path
                self.assertTrue(linked.exists(), f'{name}: link to {target} has no file')
                if anchor and linked.suffix == '.md' and name != 'README.md':
                    self.assertTrue(anchor in anchors(linked), f'{name}: link to {target} has no heading')
        self.assertTrue('docs/BROKER.md' in (ROOT / 'README.md').read_text(encoding='utf-8'), 'README.md links docs/BROKER.md')

    def test_no_private_reference(self):
        policy = json.loads((ROOT / 'tools' / 'public_export.json').read_text(encoding='utf-8'))
        patterns = {name: re.compile(pattern) for name, pattern in policy['detection_patterns'].items()}
        for name in (*CLAIM_DOCS, 'BROKER_MATRIX.md'):
            text = (DOCS / name).read_text(encoding='utf-8')
            for label, pattern in patterns.items():
                self.assertIsNone(pattern.search(text), f'docs/{name}: {label} pattern matches')

    def test_parser_finds_each_kind_of_fault(self):
        cell_id, cell = next((k, v) for k, v in sorted(self.cells.items()) if v['state'] == 'pass' and v.get('mode'))
        escaped = cell_id.replace('|', '\\|')
        self.assertEqual(unbacked_claims('The thing is supported everywhere.\n'),
                         [(1, 'The thing is supported everywhere.')])
        self.assertEqual(unbacked_claims(f'It is supported. <!-- matrix: {cell_id} -->\n'), [])
        self.assertEqual(unbacked_claims('<!-- claims-legend -->\n| untried | meaning |\n|---|---|\n| untried | x |\n'), [])
        self.assertEqual(unbacked_claims('<!-- claims-legend -->\nA legend: untried means x.\n\nThis is untried.\n'),
                         [(4, 'This is untried.')])
        self.assertEqual(len(unbacked_claims('```\nsupported\n```\n- a list item that is untried\n')), 1)
        self.assertEqual(resolve('matrix', 'no.such|cell|off', self.cells, self.vocabulary),
                         ["matrix: no cell 'no.such|cell|off'"])
        self.assertTrue(resolve('matrix-na', cell_id, self.cells, self.vocabulary))
        self.assertTrue(resolve('matrix', f'{cell_id}=Z', self.cells, self.vocabulary))
        self.assertEqual(resolve('matrix', f'{cell_id}={cell["mode"]}', self.cells, self.vocabulary), [])
        self.assertTrue(resolve('matrix-count', 'pass=0', self.cells, self.vocabulary))
        self.assertTrue(resolve('matrix-deferred', 'not_a_profile', self.cells, self.vocabulary))
        self.assertTrue(resolve('matrix-bogus', cell_id, self.cells, self.vocabulary))
        self.assertEqual(tokens(f'| x <!-- matrix: {escaped} --> |')[0][1], cell_id)


if __name__ == '__main__':
    unittest.main(verbosity=2)
