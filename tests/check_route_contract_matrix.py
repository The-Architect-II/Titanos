"""The route-to-typed-contract matrix (docs/route_contract_matrix.json, docs/ROUTE_CONTRACT_MATRIX.md) is a fixed point of its
generator and is bound to the live app, the MCP registry and the source.

Checks:
- rebuilding with tools/build_route_contract_matrix.py into a temporary directory reproduces the committed JSON byte for byte
  (the committed copy is read with CRLF normalized to LF, because core.autocrlf checkouts rewrite text files; the rebuilt copy must
  carry no CR at all) and the committed Markdown apart from the source HEAD short hash in its Basis line;
- every operation of app.openapi() appears exactly once in the matrix and every matrix row is such an operation;
- every MCP tool of create_mcp() is in the matrix and maps to at least one existing operation; every resource and template maps to
  an operation or to a route outside the schema (/openapi.json);
- every semantics citation resolves: the file exists, the cited line exists and contains the named identifier; handler, tool,
  resource, prompt, client and doc_gaps citations resolve the same way; every `file.py:line` reference inside a note, including
  the `, :line` continuations that follow it, names an existing line of an existing file; every row carries an origin (table,
  derived, shared);
- the reclassified rows of the review rounds carry the class the source shows (EXPECTED_CLASSES) and a documentation gap is
  recorded on the operation's doc_gaps, never as a declared-only semantics row;
- every field of every AssetJob member (ImageJob, SpeechJob, VideoJob, MusicJob) has a semantics row on POST /api/jobs;
- the semantics the INT-01 remaining condition names (music and anonymous-diarization control combinations, stateful Responses) are
  marked as not implemented (ignored, rejected, rejected-in-job or declared-only), never as implemented, both on
  POST /v1/music/generations and on POST /api/jobs task=music, the operation CLI music, Client.submit_music and MCP submit_job call;
- the shared rows of POST /api/jobs are exact copies of the POST /v1/music/generations rows and cover every MusicJob field; the
  derived `location` rows exist on exactly the operations whose live request model declares the field, and request.fields of every
  JSON operation equals the live model's declared fields;
- the POST /v1/responses rows that cite the local adapter (responses.py) or the vision check that follows it carry the
  local-backends qualifier and name the native branch that precedes them; the llama-cpp native passthrough row exists, is
  implemented and cites that branch (server.py:1768), whose order against the adapter call and the vision check is read from the
  source; no semantics note in the matrix says clamp (the music duration is validated against the bundle range, never adjusted);
- the public-routes set equals EXPECTED_PUBLIC of tests/check_security_public_routes.py;
- the explicit tables of the generator are current: family paths are operations, CLI subcommands equal the argparse help, a CLI
  route restriction stays inside what its Client methods reach, the `mcp` bridge is joined to exactly the union of the MCP tool
  operations, and the generated files carry no lab-private reference (tools/public_export.json detection patterns);
- the broker extensions follow the live tables: every non-public operation is `scoped` with the broker.scopes.ROUTE_SCOPES entry
  that broker.scopes.required_scope enforces on a concrete path (per Gemini action where a surface decides it), its credential
  headers are Authorization: Bearer plus the family vendor header of broker.auth.VENDOR_HEADERS, every broker surface sits on
  exactly one operation of its method, the gemini and broker families hold exactly their paths, and GET /api/version is
  recorded as public in managed role only;
- the build is independent of local broker state: with TITANOS_HOME pointing at a home populated through the offline CLI
  (broker.json with an upstream, keys.json with a key, a vault index with an env reference) and TITANOS_ROLE=broker, the
  output is byte-identical (the netguard is installed; nothing leaves loopback).

In-process only: the app is created with the runtime catalog's hardware probes stubbed by the generator (and restored afterwards);
no server listens, no model loads, no device is touched, nothing is written outside a temporary directory. Run with -B. Without a
git checkout the Basis line reads `source HEAD unknown`, which the Markdown comparison accepts."""
import ast
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from broker_mocks import netguard  # noqa: E402

netguard.install()  # the populated-home build runs the offline CLI: nothing may leave loopback

import build_route_contract_matrix as builder  # noqa: E402

HASH = re.compile(r'source HEAD `(?:[0-9a-f]{7,40}|unknown)`')
NOT_IMPLEMENTED = {'ignored', 'rejected', 'rejected-in-job', 'declared-only'}
ORIGINS = ('table', 'derived', 'shared')
MUSIC_SOURCE, MUSIC_TARGET, MUSIC_QUALIFIER = 'POST /v1/music/generations', 'POST /api/jobs', 'task=music'
# The INT-01 remaining condition (docs/progress.json): unexecuted modality/control combinations, especially music and anonymous
# diarization, plus the stateful Responses surface the adapter declares absent. Each must be recorded as not implemented.
MUSIC_GAPS = {
    'reference_audio_base64': 'rejected-in-job',
    'reference_audio_base64 (child runner)': 'rejected-in-job',
    'timeout': 'rejected',
    'progress.completed/total': 'declared-only',
    'progress.native_cancelled': 'declared-only',
    'variant other than xl-sft': 'rejected-in-job',
    'model_sharding / tensor_parallel': 'declared-only',
    'duration outside the bundle range': 'rejected-in-job',
    'planner flags without a planner route': 'rejected-in-job',
}
KNOWN_GAPS = {(MUSIC_SOURCE, control): status for control, status in MUSIC_GAPS.items()}
# CLI music, Client.submit_music and MCP submit_job call POST /api/jobs task=music, so every music gap must be recorded there too.
KNOWN_GAPS.update({(MUSIC_TARGET, f'{MUSIC_QUALIFIER}: {control}'): status for control, status in MUSIC_GAPS.items()})
KNOWN_GAPS.update({
    ('POST /v1/audio/diarizations', 'stream'): 'rejected',
    ('POST /v1/audio/diarizations', 'location'): 'ignored',
    ('POST /v1/audio/diarizations', 'speaker identity / calibrated confidence'): 'declared-only',
    ('POST /v1/audio/diarizations', 'cancellation inside native segmentation/clustering'): 'declared-only',
    ('POST /v1/audio/diarizations', 'num_speakers / cluster_threshold on clips <= 10 s'): 'rejected',
    ('POST /v1/responses', 'store=true'): 'rejected',
    ('POST /v1/responses', 'previous_response_id'): 'rejected',
    ('POST /v1/responses', 'conversation'): 'rejected',
    ('POST /v1/responses', 'background=true'): 'rejected',
    ('POST /v1/responses', 'cancellation (nonstream, local backends)'): 'ignored',
})
# Rows whose class the review rounds fixed against the source; each must keep the class the cited code shows.
EXPECTED_CLASSES = {
    ('POST /v1/music/generations', 'capability progress claim'): 'implemented',
    ('POST /v1/music/generations', 'progress.stage'): 'implemented',
    ('POST /api/jobs', 'deadline / timeout'): 'rejected',
    ('POST /api/jobs', 'model (all tasks)'): 'implemented',
    ('POST /api/jobs', 'task=image: response_format'): 'ignored',
    ('POST /api/jobs', 'task=speech: language (transformers speech)'): 'rejected-in-job',
    ('POST /api/jobs', 'task=video: pipeline_options without input_reference'): 'rejected-in-job',
    ('POST /v1/videos', 'pipeline_options without input_reference'): 'rejected-in-job',
    ('POST /v1/chat/completions', 'cancellation (nonstream, local backends, require_loaded=false)'): 'ignored',
    ('POST /v1/chat/completions', 'cancellation (nonstream, local backends, require_loaded=true)'): 'implemented',
    ('GET /readyz', 'unauthenticated access (keyed mode)'): 'rejected',
    ('POST /v1/audio/analysis', 'model / worker (ASR stage)'): 'implemented',
    ('POST /v1/audio/analysis', 'response_format other than verbose_json'): 'rejected',
    ('POST /v1/responses', 'input items / roles / images / instructions / sampling / usage (llama-cpp native)'): 'implemented',
}
# POST /v1/responses forwards a llama-cpp native model to native.proxy before the local adapter (responses.py chat_request) and
# the vision check run (server.py:1768-1770): every row that cites the adapter or that check describes local backends only and
# carries a parenthesised qualifier naming them; the schema-level 422 rows (Payload extra=forbid, store Literal) hold for every
# backend and carry none.
RESPONSES_OP = 'POST /v1/responses'
RESPONSES_NATIVE_BRANCH = 'src/titanos/server.py:1774'
RESPONSES_VISION_CHECK = 'src/titanos/server.py:1783'
RESPONSES_SCHEMA_CITATIONS = {'src/titanos/server.py:381', 'src/titanos/server.py:619'}
RESPONSES_NATIVE_ROW = 'input items / roles / images / instructions / sampling / usage (llama-cpp native)'
LOCAL_BACKENDS = re.compile(r'\((?:[^()]*, )?local backends(?:, [^()]*)?\)')
# `file.py:line` (optionally `-line`) references inside notes; bare file names resolve through NOTE_REFERENCE_DIRS in order. A
# reference may continue with `, :line` (optionally `-line`) items that name further lines of the same file.
NOTE_REFERENCE = re.compile(r'(?<![\w./-])((?:[\w-]+/)*[\w-]+\.(?:py|md|json)):(\d+)(?:-(\d+))?')
NOTE_CONTINUATION = re.compile(r',\s*:(\d+)(?:-(\d+))?')
NOTE_REFERENCE_DIRS = ('src/titanos', 'tools/music_runtime', 'tools', 'tests', 'docs')


def read_lines(path):
    return (ROOT / path).read_text(encoding='utf-8', errors='replace').splitlines()


class RouteContractMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='titanos-route-matrix-check-')
        out = Path(cls.temporary.name) / 'out'
        with contextlib.redirect_stdout(io.StringIO()):
            builder.main(['--out', str(out)])
        cls.rebuilt_json = (out / builder.JSON_OUT).read_bytes()
        cls.rebuilt_md = (out / builder.MD_OUT).read_bytes()
        cls.committed_json = (ROOT / 'docs' / builder.JSON_OUT).read_bytes()
        cls.committed_md = (ROOT / 'docs' / builder.MD_OUT).read_bytes()
        cls.data = json.loads(cls.rebuilt_json)
        cls.rows = cls.data['operations']
        cls.ops = [f"{r['method']} {r['path']}" for r in cls.rows]
        # a second in-process app for the live comparisons (the generator's own app lived inside main())
        cls.app = builder.build_app(Path(cls.temporary.name) / 'live')
        cls.lines = {}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def source(self, path):
        if path not in self.lines:
            self.assertTrue((ROOT / path).is_file(), f'cited file {path} does not exist')
            self.lines[path] = read_lines(path)
        return self.lines[path]

    def assert_citation(self, citation, identifier, context):
        path, _, line = citation.rpartition(':')
        self.assertTrue(path and line.isdigit(), f'{context}: citation {citation!r} is not path:line')
        lines = self.source(path)
        number = int(line)
        self.assertTrue(1 <= number <= len(lines), f'{context}: {citation} is past the end of the file ({len(lines)} lines)')
        self.assertIn(identifier, lines[number - 1], f'{context}: line {citation} does not contain {identifier!r}: {lines[number - 1].strip()[:120]!r}')

    # (a) fixed point
    def test_rebuild_reproduces_the_committed_json_byte_for_byte(self):
        self.assertNotIn(b'\r', self.rebuilt_json, 'the generator writes LF newlines only')
        self.assertEqual(self.committed_json.replace(b'\r\n', b'\n'), self.rebuilt_json,
                         'docs/route_contract_matrix.json is not the generator output; run python tools/build_route_contract_matrix.py')

    def test_rebuild_reproduces_the_committed_markdown_apart_from_the_head_hash(self):
        self.assertNotIn(b'\r', self.rebuilt_md)
        rebuilt = self.rebuilt_md.decode('utf-8')
        committed = self.committed_md.replace(b'\r\n', b'\n').decode('utf-8')
        self.assertRegex(rebuilt.splitlines()[2], HASH, 'the Basis line names the source HEAD short hash')
        self.assertRegex(committed.splitlines()[2], HASH)
        self.assertEqual(HASH.sub('source HEAD `<hash>`', committed), HASH.sub('source HEAD `<hash>`', rebuilt),
                         'docs/ROUTE_CONTRACT_MATRIX.md is not the generator output; run python tools/build_route_contract_matrix.py')
        self.assertIn('tools/build_route_contract_matrix.py', rebuilt.splitlines()[2])
        self.assertIn('## Declared but not implemented', rebuilt)
        self.assertIn('## Shared semantics', rebuilt)

    # (b) coverage of app.openapi()
    def test_every_openapi_operation_appears_exactly_once_and_vice_versa(self):
        schema = self.app.openapi()
        declared = Counter(f'{method.upper()} {path}' for path, item in schema['paths'].items() for method in item)
        self.assertEqual(set(declared.values()), {1}, 'app.openapi() declares an operation twice')
        self.assertEqual(Counter(self.ops), declared)
        self.assertEqual(self.data['counts']['operations'], len(declared))
        self.assertEqual(self.data['counts']['paths'], len(schema['paths']))
        self.assertEqual(self.data['openapi_version'], schema['openapi'])
        self.assertEqual(self.ops, sorted(self.ops, key=lambda op: (op.split(' ', 1)[1], op.split(' ', 1)[0])), 'rows are sorted by path then method')
        for row in self.rows:
            self.assertIn(row['auth'], self.data['vocabulary']['auth'])
            self.assertIn(row['family'], self.data['vocabulary']['families'])
            self.assertIn(row['registry_gate'], self.data['vocabulary']['registry_gate'])
            self.assertEqual(row['auth'] == 'public', row['path'] in self.data['public_routes'], row['path'])

    # (c) MCP tools
    def test_every_mcp_tool_maps_to_an_existing_operation(self):
        live = set(self.app.state.mcp._tool_manager._tools)
        recorded = {tool['name']: tool for tool in self.data['mcp_tools']}
        self.assertEqual(set(recorded), live, 'the matrix tool list differs from create_mcp()')
        self.assertEqual(self.data['counts']['mcp_tools'], len(live))
        operations = set(self.ops)
        for name, tool in recorded.items():
            self.assertTrue(tool['operations'], f'MCP tool {name} maps to no HTTP operation')
            for op in tool['operations']:
                self.assertIn(op, operations, f'MCP tool {name} names an operation the app does not register: {op}')
            self.assertEqual(tool['unlisted_routes'], [], f'MCP tool {name} calls a route outside the schema')
        joined = {t for row in self.rows for t in row['mcp_tools']}
        self.assertEqual(joined, live, 'every tool is joined to at least one row and every joined name is a tool')
        live_resources = {str(r.uri) for r in self.app.state.mcp._resource_manager._resources.values()}
        live_templates = {t.uri_template for t in self.app.state.mcp._resource_manager._templates.values()}
        self.assertEqual({r['uri'] for r in self.data['mcp_resources']}, live_resources)
        self.assertEqual({t['uri_template'] for t in self.data['mcp_resource_templates']}, live_templates)
        unlisted = {f"{m} {u['path']}" for u in self.data['unlisted_routes'] for m in u['method'].split(',')}
        for entry in self.data['mcp_resources'] + self.data['mcp_resource_templates']:
            label = entry.get('uri') or entry.get('uri_template')
            self.assertTrue(entry['operations'] or entry['unlisted_routes'], f'MCP resource {label} maps to nothing')
            for op in entry['operations']:
                self.assertIn(op, operations, label)
            for route in entry['unlisted_routes']:
                self.assertIn(route, unlisted, label)
        self.assertEqual({p['name'] for p in self.data['mcp_prompts']}, set(self.app.state.mcp._prompt_manager._prompts))

    def assert_note_references(self, note, context):
        """Every file:line reference a note makes, and every `, :line` continuation that follows it, names an existing line of that
        file (the content is not checked: notes carry no identifier). Returns the number of resolved line references."""
        resolved = 0
        for match in NOTE_REFERENCE.finditer(note):
            name, first, last = match.groups()
            if '/' in name:
                candidates = [name]
            else:
                candidates = [f'{d}/{name}' for d in NOTE_REFERENCE_DIRS if (ROOT / d / name).is_file()]
            self.assertTrue(candidates and (ROOT / candidates[0]).is_file(), f'{context}: note references {name}:{first}, which is no file under {NOTE_REFERENCE_DIRS}')
            lines = self.source(candidates[0])
            spans = [(first, last)]
            end = match.end()
            while (continuation := NOTE_CONTINUATION.match(note, end)) is not None:
                spans.append(continuation.groups())
                end = continuation.end()
            for start, stop in spans:
                for number in (int(start), int(stop or start)):
                    self.assertTrue(1 <= number <= len(lines), f'{context}: note references {name}:{number}, past the end of {candidates[0]} ({len(lines)} lines)')
                resolved += 1
        return resolved

    def test_note_reference_continuations_resolve(self):
        # the continuation form `file.py:N, :M` is used by the tables; the checker must resolve every item, and must refuse an
        # out-of-range continuation (a synthetic note against this test file itself)
        own = 'tests/check_route_contract_matrix.py'
        self.assertEqual(self.assert_note_references(f'see {own}:1, :2, :3-4 and jobs.py:1', 'synthetic'), 4)
        with self.assertRaises(AssertionError):
            self.assert_note_references(f'see {own}:1, :999999', 'synthetic')
        notes = [s['note'] for r in self.rows for s in r['semantics']]
        resolved = sum(self.assert_note_references(note, 'tables') for note in notes)
        direct = sum(len(NOTE_REFERENCE.findall(note)) for note in notes)
        self.assertGreater(resolved, direct, 'the tables carry continuation references; the check must resolve them, not only the leading file:line')

    # (d) citations
    def test_every_semantics_citation_resolves(self):
        seen = set()
        for row in self.rows:
            op = f"{row['method']} {row['path']}"
            for s in row['semantics']:
                self.assertIn(s['status'], self.data['vocabulary']['semantics'], (op, s['control']))
                self.assertNotIn((op, s['control']), seen, 'duplicate semantics row')
                seen.add((op, s['control']))
                self.assertTrue(s['identifier'].strip(), (op, s['control']))
                self.assertFalse(re.match(r'[A-Za-z]:[\\/]', s['citation']), 'citations are repository-relative')
                self.assert_citation(s['citation'], s['identifier'], f'{op} / {s["control"]}')
                self.assert_note_references(s['note'], f'{op} / {s["control"]}')
                self.assertIn(s['origin'], ORIGINS, (op, s['control']))
                self.assertIn(s['origin'], self.data['vocabulary']['semantics_origin'])
                self.assertEqual(s['origin'] == 'shared', s['shared_from'] is not None, (op, s['control']))
            for gap in row['doc_gaps']:
                self.assert_citation(gap['citation'], gap['identifier'], f'{op} doc_gaps')
                self.assert_note_references(gap['note'], f'{op} doc_gaps')
        self.assertEqual(len(seen), self.data['counts']['semantics'])
        self.assertEqual(self.data['counts']['doc_gaps'], sum(len(r['doc_gaps']) for r in self.rows))
        for shared in self.data['operations']:
            for s in shared['shared_semantics_from']:
                self.assert_note_references(s['note'], f"{shared['method']} {shared['path']} shared_semantics_from")
        for row in self.data['cli_subcommands']:
            self.assert_note_references((row['local'] or '') + ' ' + (row['bridge'] or '') + ' ' + (row['note'] or ''), f"CLI {row['name']}")
        self.assertEqual(self.data['counts']['semantics_by_origin']['table'], len(builder.SEMANTICS))
        self.assertEqual(self.data['counts']['semantics_by_origin'], {o: Counter(x['origin'] for r in self.rows for x in r['semantics']).get(o, 0) for o in ORIGINS})
        self.assertEqual(self.data['counts']['semantics_by_status'], {s: Counter(x['status'] for r in self.rows for x in r['semantics']).get(s, 0) for s in builder.STATUSES})
        self.assertEqual(set(self.data['vocabulary']['semantics']), set(builder.STATUSES))

    def test_shared_semantics_reach_the_operation_the_interfaces_call(self):
        by_op = {f"{r['method']} {r['path']}": r for r in self.rows}
        jobs, music = by_op[MUSIC_TARGET], by_op[MUSIC_SOURCE]
        self.assertEqual([(s['operation'], s['qualifier']) for s in jobs['shared_semantics_from']], [(MUSIC_SOURCE, MUSIC_QUALIFIER)])
        self.assertEqual([(s['operation'], s['qualifier']) for s in music['shared_semantics_to']], [(MUSIC_TARGET, MUSIC_QUALIFIER)])
        for row in self.rows:
            if row not in (jobs, music):
                self.assertEqual((row['shared_semantics_from'], row['shared_semantics_to']), ([], []), f"{row['method']} {row['path']}")
        shared = {s['control']: s for s in jobs['semantics'] if s['origin'] == 'shared'}
        self.assertTrue(music['semantics'])
        self.assertEqual(set(shared), {f"{MUSIC_QUALIFIER}: {s['control']}" for s in music['semantics']})
        for s in music['semantics']:
            twin = shared[f"{MUSIC_QUALIFIER}: {s['control']}"]
            self.assertEqual((twin['status'], twin['citation'], twin['identifier'], twin['note'], twin['shared_from']),
                             (s['status'], s['citation'], s['identifier'], s['note'], MUSIC_SOURCE), s['control'])
        # the interfaces that reach music all call POST /api/jobs, so the copied gap rows sit on the operation they use
        self.assertIn('submit_job', jobs['mcp_tools'])
        self.assertIn('music', jobs['cli'])
        self.assertIn('submit_music', jobs['client'])
        from titanos.server import MusicJob, MusicRequest
        self.assertTrue(issubclass(MusicJob, MusicRequest))
        controls = [s['control'] for s in music['semantics']]
        for field in MusicJob.model_fields:
            if field not in ('task', 'model'):
                self.assertTrue(any(c == field or c.startswith(field + ' ') for c in controls), f'MusicJob field {field} has no music semantics row')
        rendered = self.rebuilt_md.decode('utf-8')
        self.assertIn(f'runs the {len(shared)} semantics rows of `{MUSIC_SOURCE}`', rendered)
        if not (music['mcp_tools'] or music['cli'] or music['client']):
            self.assertIn(f'via `{MUSIC_TARGET}` ({MUSIC_QUALIFIER}): submit_job', rendered)

    def test_derived_location_rows_cover_exactly_the_models_that_declare_the_field(self):
        from fastapi.routing import APIRoute
        live_fields = {}
        for route in self.app.routes:
            if isinstance(route, APIRoute) and route.include_in_schema:
                params = [p for p in route.dependant.body_params if type(p.field_info).__name__ not in ('Form', 'File')]
                if params:
                    names = {name for p in params for c in builder.model_classes(p.field_info.annotation) for name in c.model_fields}
                    for method in route.methods:
                        live_fields[f'{method} {route.path_format}'] = names
        recorded_fields = {f"{r['method']} {r['path']}": r['request']['fields'] for r in self.rows if r['request']['kind'] == 'json'}
        self.assertEqual(set(recorded_fields), set(live_fields))
        for op, names in live_fields.items():
            self.assertEqual(recorded_fields[op], sorted(names), op)
        for field, status, citation, identifier, note in builder.DERIVED_FIELD_SEMANTICS:
            expected = {op for op, names in live_fields.items() if field in names}
            self.assertTrue(expected, field)
            recorded = {f"{r['method']} {r['path']}": s for r in self.rows for s in r['semantics'] if s['control'] == field}
            self.assertEqual(set(recorded), expected, f'derived {field} rows differ from the models that declare the field')
            for op, s in recorded.items():
                self.assertEqual((s['status'], s['origin'], s['citation'], s['identifier']), (status, 'derived', citation, identifier), op)
        self.assertGreaterEqual(len({op for op, names in live_fields.items() if 'location' in names}), 13)

    def test_handler_tool_resource_and_client_citations_resolve(self):
        for row in self.rows:
            self.assert_citation(row['handler'], '@app.', f"handler of {row['method']} {row['path']}")
            self.assert_citation(row['handler'], row['method'].lower(), f"handler of {row['method']} {row['path']}")
        for u in self.data['unlisted_routes']:
            if re.search(r':\d+$', u['handler']):
                self.assert_citation(u['handler'], '@app.', f"handler of unlisted {u['path']}")
        for tool in self.data['mcp_tools']:
            self.assert_citation(tool['source'], '@mcp.tool', f"MCP tool {tool['name']}")
        for entry in self.data['mcp_resources'] + self.data['mcp_resource_templates']:
            self.assert_citation(entry['source'], '@mcp.resource', entry.get('uri') or entry.get('uri_template'))
        for prompt in self.data['mcp_prompts']:
            self.assert_citation(prompt['source'], '@mcp.prompt', prompt['name'])
        for method in self.data['client_methods']:
            self.assert_citation(method['source'], f"def {method['name']}(", f"Client.{method['name']}")

    # (e) INT-01 remaining condition
    def test_known_unimplemented_semantics_are_not_marked_implemented(self):
        progress = json.loads((ROOT / 'docs' / 'progress.json').read_text(encoding='utf-8'))
        item = next(i for i in progress['items'] if i['id'] == 'INT-01')
        remaining = item['remaining_condition'].lower()
        self.assertIn('music', remaining)
        self.assertIn('diarization', remaining)
        by_key = {(f"{r['method']} {r['path']}", s['control']): s['status'] for r in self.rows for s in r['semantics']}
        for key, expected in KNOWN_GAPS.items():
            self.assertIn(key, by_key, f'known gap {key} is not in the matrix')
            self.assertEqual(by_key[key], expected, f'known gap {key} is recorded as {by_key[key]}, expected {expected}')
            self.assertIn(by_key[key], NOT_IMPLEMENTED)
        for op in ('POST /v1/music/generations', 'POST /v1/audio/diarizations', 'POST /v1/responses'):
            statuses = [status for (o, _), status in by_key.items() if o == op]
            self.assertTrue(statuses, f'{op} has no semantics rows')
            self.assertTrue(any(s in NOT_IMPLEMENTED for s in statuses), f'{op} records no unimplemented semantics')
        rendered = self.rebuilt_md.decode('utf-8').split('## Declared but not implemented', 1)[1].split('## Semantics per operation', 1)[0]
        for (op, control), _ in KNOWN_GAPS.items():
            self.assertIn(f'- {control}:', rendered, f'{op} / {control} missing from the Declared-but-not-implemented section')

    def test_reclassified_rows_keep_the_class_the_source_shows(self):
        by_key = {(f"{r['method']} {r['path']}", s['control']): s for r in self.rows for s in r['semantics']}
        for key, expected in EXPECTED_CLASSES.items():
            self.assertIn(key, by_key, f'expected row {key} is not in the matrix')
            self.assertEqual(by_key[key]['status'], expected, key)
        # the superseded rows are gone: the unqualified chat cancellation row, the ignored deadline row, the docstring row
        for stale in (('POST /v1/chat/completions', 'cancellation (nonstream, local backends)'), ('POST /api/jobs', 'deadline'),
                      ('POST /api/jobs', 'MCP submit_job docstring (image, speech or video)'), ('GET /readyz', 'unauthenticated access')):
            self.assertNotIn(stale, by_key)
        for (op, control), s in by_key.items():
            self.assertFalse(s['status'] == 'declared-only' and 'docstring' in control.lower(),
                             f'{op} / {control}: a docstring that under-declares a code path belongs in doc_gaps, not declared-only')
        by_op = {f"{r['method']} {r['path']}": r for r in self.rows}
        self.assertEqual([g['citation'].rpartition(':')[0] for g in by_op['POST /api/jobs']['doc_gaps']], ['src/titanos/mcp_server.py'])
        self.assertIn('## Documentation gaps', self.rebuilt_md.decode('utf-8'))
        self.assertEqual(by_key[('POST /v1/responses', 'cancellation (nonstream, local backends)')]['citation'], 'src/titanos/server.py:1708')
        self.assertIn('server.py:1785', by_key[('POST /v1/responses', 'cancellation (nonstream, local backends)')]['note'])

    def test_responses_local_adapter_rows_are_qualified_and_the_native_passthrough_is_recorded(self):
        rows = {s['control']: s for r in self.rows if f"{r['method']} {r['path']}" == RESPONSES_OP for s in r['semantics']}
        adapter = [c for c, s in rows.items() if s['origin'] == 'table'
                   and (s['citation'].rpartition(':')[0] == 'src/titanos/responses.py' or s['citation'] == RESPONSES_VISION_CHECK)]
        self.assertGreaterEqual(len(adapter), 15, adapter)
        for control in adapter:
            self.assertRegex(control, LOCAL_BACKENDS, f'{RESPONSES_OP} / {control} cites the local adapter without the local-backends qualifier')
            self.assertIn('server.py:1774', rows[control]['note'], f'{control}: the note must name the native branch that precedes the adapter')
        schema = [c for c, s in rows.items() if s['citation'] in RESPONSES_SCHEMA_CITATIONS]
        self.assertGreaterEqual(len(schema), 1, schema)
        # the hosted-platform hints are accepted and named in X-Titanos-Ignored, unservable values are typed 400s, for every
        # backend (titanos.request_compat runs before the schema)
        compat = {c: s for c, s in rows.items() if s['citation'].startswith('src/titanos/request_compat.py:')}
        ignored = next(c for c, s in compat.items() if s['status'] == 'ignored')
        for name in ('include', 'reasoning', 'truncation', 'service_tier', 'prompt_cache_key', 'text.verbosity'):
            self.assertIn(name, ignored)
        self.assertEqual({c for c, s in compat.items() if s['status'] == 'rejected'},
                         {'store=true', 'previous_response_id', 'background=true'})
        for control in compat:
            self.assertNotRegex(control, LOCAL_BACKENDS, f'{control}: request_compat holds for every backend')
        for control in schema:
            self.assertNotRegex(control, LOCAL_BACKENDS, f'{control}: a schema-level 422 holds for every backend')
        self.assertIn('cancellation (nonstream, local backends)', rows)
        self.assertNotIn('cancellation (stream)', rows)
        native = rows[RESPONSES_NATIVE_ROW]
        self.assertEqual((native['status'], native['citation'], native['identifier'], native['origin']), ('implemented', RESPONSES_NATIVE_BRANCH, 'native', 'table'))
        for expected in ('chat_request', 'vision check', 'tests/check_responses.py:293', 'not from a test'):
            self.assertIn(expected, native['note'])
        self.assertIn('tools / tool_choice / text (llama-cpp native)', rows)
        # the source order the rows assert: the native branch, then the adapter import and call, then the vision check
        lines = self.source('src/titanos/server.py')
        branch, vision = int(RESPONSES_NATIVE_BRANCH.rpartition(':')[2]), int(RESPONSES_VISION_CHECK.rpartition(':')[2])
        self.assertIn('payload.model in native.models', lines[branch - 1])
        self.assertIn('native.proxy', lines[branch + 1])
        call = next(n for n in range(branch + 2, vision) if 'chat_request(payload' in lines[n - 1])
        self.assertLess(branch + 1, call)
        self.assertLess(call, vision)
        self.assertIn('unsupported_capability', lines[vision - 1])
        # the passthrough test the native row cites starts at the cited line and installs a native.proxy stand-in
        passthrough = self.source('tests/check_responses.py')
        self.assertIn('native_calls', passthrough[292])
        self.assertTrue(any('native.proxy' in line for line in passthrough[292:322]))
        self.assertTrue(any('503' in line for line in passthrough[292:322]))
        # the music duration is validated against the bundle range, never adjusted into it: no note in the matrix says clamp
        for r in self.rows:
            for s in r['semantics']:
                self.assertNotIn('clamp', s['note'].lower(), (f"{r['method']} {r['path']}", s['control']))
        music = {s['control']: s for r in self.rows if f"{r['method']} {r['path']}" == MUSIC_SOURCE for s in r['semantics']}
        self.assertIn('ValueError', music['duration']['note'])
        self.assertEqual((music['duration outside the bundle range']['status'], music['duration outside the bundle range']['identifier']),
                         ('rejected-in-job', 'Music duration must be'))
        self.assertIn(f'{MUSIC_QUALIFIER}: duration outside the bundle range', {s['control'] for r in self.rows if f"{r['method']} {r['path']}" == MUSIC_TARGET for s in r['semantics']})

    def test_every_asset_job_field_has_a_semantics_row_on_post_api_jobs(self):
        from titanos.server import ImageJob, SpeechJob, VideoJob, MusicJob
        jobs = next(r for r in self.rows if f"{r['method']} {r['path']}" == MUSIC_TARGET)
        controls = [s['control'] for s in jobs['semantics']]
        for member in (ImageJob, SpeechJob, VideoJob, MusicJob):
            for field in member.model_fields:
                if field == 'task':
                    continue
                pattern = re.compile(r'(?<![\w])' + re.escape(field) + r'(?![\w])')
                self.assertTrue(any(pattern.search(c) for c in controls), f'AssetJob member {member.__name__} field {field} has no semantics row on {MUSIC_TARGET}')
        self.assertEqual(set(jobs['request']['fields']), {f for m in (ImageJob, SpeechJob, VideoJob, MusicJob) for f in m.model_fields})

    # (f) public routes
    def test_public_routes_equal_the_security_test_expectation(self):
        tree = ast.parse((ROOT / 'tests' / 'check_security_public_routes.py').read_text(encoding='utf-8'))
        expected = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'EXPECTED_PUBLIC' for t in node.targets))
        self.assertEqual(set(self.data['public_routes']), set(expected))
        from titanos import server
        self.assertEqual(set(self.data['public_routes']), set(server.PUBLIC_ROUTES))
        public_rows = [r for r in self.rows if r['auth'] == 'public']
        self.assertEqual({(r['method'], r['path']) for r in public_rows}, {('GET', '/healthz'), ('GET', '/v1/health/ready')},
                         'the public routes inside the OpenAPI schema are GET /healthz and the broker readiness probe GET /v1/health/ready '
                         '(GET /api/version is public only in managed role, so it is absent here); the Studio and documentation views are '
                         'hidden from it')
        hidden = {u['path'] for u in self.data['unlisted_routes']}
        self.assertEqual(hidden, {'/', '/ui/{name}', '/docs', '/docs/assets/{name}', '/openapi.json', '/mcp'})

    # tables
    def test_generator_tables_are_current(self):
        paths = {r['path'] for r in self.rows}
        for family, members in builder.FAMILY_PATHS.items():
            self.assertIn(family, builder.FAMILY_MEANING)
            self.assertTrue(set(members) <= paths, f'FAMILY_PATHS[{family}] names unknown paths: {sorted(set(members) - paths)}')
        for op in builder.RESPONSE_SHAPES:
            self.assertIn(op, self.ops)
        from titanos import cli
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(['--help'])
        listed = re.search(r'\{([^}]+)\}', buffer.getvalue(), re.S)
        self.assertIsNotNone(listed, 'argparse help lists the subcommands')
        subcommands = {name.strip() for name in re.sub(r'\s+', '', listed.group(1)).split(',') if name.strip()}
        self.assertEqual(set(builder.CLI_SUBCOMMANDS), subcommands, 'CLI_SUBCOMMANDS differs from the argparse subcommands')
        self.assertEqual({row['name'] for row in self.data['cli_subcommands']}, subcommands)
        from titanos.client import Client
        client_methods = {name for name in vars(Client) if not name.startswith('_') and callable(vars(Client)[name])}
        self.assertEqual({row['name'] for row in self.data['client_methods']}, client_methods)
        reach = {row['name']: set(row['operations']) for row in self.data['client_methods']}
        for name, entry in builder.CLI_SUBCOMMANDS.items():
            kinds = ('local' in entry) + ('bridge' in entry) + bool(entry.get('client') or entry.get('routes'))
            self.assertEqual(kinds, 1, f'CLI {name} is exactly one of local, bridge or route-bearing')
            self.assertTrue(set(entry.get('client', [])) <= client_methods, name)
            self.assertTrue(set(entry.get('routes', [])) <= set(self.ops), name)
            self.assertTrue(set(entry) <= {'local', 'bridge', 'client', 'routes', 'note'}, name)
            if entry.get('routes') and entry.get('client'):
                reachable = set().union(*(reach[m] for m in entry['client']))
                self.assertTrue(set(entry['routes']) <= reachable, f'CLI {name} restricts to routes its Client methods do not reach')
        for name in builder.CLIENT_OVERRIDES:
            self.assertIn(name, client_methods)
        # the mcp subcommand is the stdio bridge (cli.py run_stdio -> create_mcp(base_url).run): joined to exactly the MCP tool operations, never local
        bridges = {row['name'] for row in self.data['cli_subcommands'] if row['bridge']}
        self.assertEqual(bridges, {'mcp'})
        mcp_row = next(row for row in self.data['cli_subcommands'] if row['name'] == 'mcp')
        self.assertIsNone(mcp_row['local'])
        self.assertEqual(mcp_row['operations'], sorted({op for tool in self.data['mcp_tools'] for op in tool['operations']}))
        cli_source = (ROOT / 'src' / 'titanos' / 'cli.py').read_text(encoding='utf-8')
        self.assertIn('run_stdio(args.url)', cli_source)
        # save_asset handed bytes sends nothing: only subcommands that hand it a result dict may be joined to GET /v1/files/{name};
        # the mcp bridge reaches it through the artifact_bytes helper of the MCP tools
        files = {row['name'] for row in self.data['cli_subcommands'] if 'GET /v1/files/{name}' in row['operations']}
        self.assertEqual(files, {'edit', 'image', 'jobs', 'mcp', 'music', 'video'})
        models = next(row for row in self.data['cli_subcommands'] if row['name'] == 'models')
        self.assertEqual(models['operations'], ['GET /v1/models'])

    def test_generated_files_carry_no_lab_reference(self):
        policy = json.loads((ROOT / 'tools' / 'public_export.json').read_text(encoding='utf-8'))
        patterns = {name: re.compile(pattern) for name, pattern in policy['detection_patterns'].items()}
        for label, payload in (('json', self.rebuilt_json), ('md', self.rebuilt_md)):
            text = payload.decode('utf-8')
            for name, pattern in patterns.items():
                self.assertIsNone(pattern.search(text), f'{label}: {name} pattern matches the generated matrix')
        for row in self.rows:
            self.assertFalse(re.match(r'[A-Za-z]:[\\/]', row['handler']), row['handler'])

    # (g) broker extensions: scopes, credential headers, surfaces, families, role-only public routes
    def test_scopes_credential_headers_and_surfaces_follow_the_live_tables(self):
        from titanos.broker import native as native_lane
        from titanos.broker import scopes
        from titanos.broker.auth import VENDOR_HEADERS
        from titanos.broker.routes import GEMINI_ACTIONS
        from titanos.broker.surfaces import SURFACES
        self.assertEqual(set(self.data['vocabulary']['auth']), {'public', 'scoped'})
        placed = Counter()
        for row in self.rows:
            op = f"{row['method']} {row['path']}"
            for name in row['broker_surfaces']:
                placed[name] += 1
                self.assertEqual(SURFACES[name].method, row['method'], (op, name))
            if row['auth'] == 'public':
                self.assertEqual((row['scope'], row['scope_by_action'], row['credential_headers']), (None, {}, []), op)
                self.assertIsNone(scopes.required_scope(row['method'], row['path']), op)
                continue
            listed = scopes.scope_for_template(row['method'], row['route_path'])
            self.assertIsNotNone(listed, f'{op} is not listed in broker.scopes.ROUTE_SCOPES (it would need admin)')
            self.assertEqual(row['scope'], listed, op)
            concrete = re.sub(r'\{[^}]+\}', 'x1', row['route_path'])
            enforced = listed
            if listed != scopes.AUTHENTICATED and native_lane.bare_may_win(row['method'], concrete):
                # a native provider row shares this shape (OpenAI /v1/videos): the Guard also accepts native, and
                # BrokerDispatch requires the listed scope when the Titanos route answers
                enforced = listed + '|native'
            self.assertEqual(scopes.required_scope(row['method'], concrete), enforced,
                             f'{op}: the enforced scope differs from the table')
            for action, scope in row['scope_by_action'].items():
                self.assertEqual(scopes.required_scope(row['method'], concrete + ':' + action), scope, (op, action))
            vendor = sorted(h.decode('ascii') for h, prefixes in VENDOR_HEADERS.items()
                            if any(row['path'] == p or row['path'].startswith(p + '/') for p in prefixes))
            self.assertEqual(row['credential_headers'], ['Authorization: Bearer'] + vendor, op)
        self.assertEqual(set(placed), set(SURFACES), 'every broker surface is served by an operation of this build')
        self.assertEqual(set(placed.values()), {1}, 'a broker surface sits on exactly one operation')
        gemini = next(r for r in self.rows if (r['method'], r['path']) == ('POST', '/v1beta/models/{model_action}'))
        self.assertEqual(set(gemini['scope_by_action']), set(GEMINI_ACTIONS))
        self.assertEqual((gemini['scope_by_action']['embedContent'], gemini['scope_by_action']['generateContent']),
                         ('inference:embeddings', 'inference:chat'))
        self.assertEqual(Counter(r['scope'] for r in self.rows if r['scope']), Counter(self.data['counts']['scopes']))
        for vendor in ('x-api-key', 'xi-api-key', 'x-goog-api-key'):
            self.assertIn(vendor, self.rebuilt_md.decode('utf-8'))
        mcp = next(u for u in self.data['unlisted_routes'] if u['path'] == '/mcp')
        self.assertTrue(mcp['auth'].startswith(f"scoped ({scopes.required_scope('POST', '/mcp')} "), mcp['auth'])

    def test_families_and_role_only_public_routes(self):
        self.assertEqual(builder.FAMILY_ORDER, list(builder.FAMILY_MEANING))
        self.assertEqual(set(builder.FAMILY_PATHS), set(builder.FAMILY_MEANING) - {'native'})
        for row in self.rows:
            expected = next((f for f, paths in builder.FAMILY_PATHS.items() if row['path'] in paths), 'native')
            self.assertEqual(row['family'], expected, row['path'])
            self.assertEqual(row['family'] == 'broker', row['path'].startswith(builder.BROKER_PREFIXES), row['path'])
            self.assertEqual(row['family'] == 'gemini', row['path'].startswith('/v1beta/'), row['path'])
        for family in ('gemini', 'broker', 'openai', 'anthropic', 'elevenlabs'):
            self.assertIn(f'## Family: {family}', self.rebuilt_md.decode('utf-8'))
        from titanos.broker import scopes
        self.assertEqual(set(self.data['role_only_public_routes']), set(scopes.MANAGED_PUBLIC_ROUTES))
        version = [(r['method'], r['auth'], r['family']) for r in self.rows if r['path'] == '/api/version']
        self.assertEqual(version, [('GET', 'scoped', 'ollama')], 'GET /api/version is a scoped Ollama-family operation of this build; public only in managed role')
        self.assertEqual(self.data['counts']['public_routes'], len(self.data['public_routes']))
        self.assertIn('managed role', self.rebuilt_md.decode('utf-8').split('## Family:', 1)[0])

    def test_build_is_independent_of_a_populated_titanos_home(self):
        import os
        from unittest import mock
        from titanos.cli import main as cli_main
        home = Path(self.temporary.name) / 'populated-home'
        home.mkdir()
        variable = 'WP9_ROUTE_MATRIX_SYNTHETIC_CREDENTIAL'
        env = {'TITANOS_HOME': str(home), variable: 'test-openai-not-a-key-' + os.urandom(6).hex()}
        commands = (['broker', 'upstream', 'add', 'oai', '--profile', 'openai', '--models', 'gpt-4o-mini'],
                    ['broker', 'credential', 'set', 'oai-key', '--upstream', 'oai', '--from-env', variable, '--no-validate'],
                    ['broker', 'policy', 'set', '--cloud-policy', 'explicit'],
                    ['keys', 'create', '--name', 'route-matrix', '--scope', 'inference', '--expires', '30d'])
        netguard.reset_attempts()
        with mock.patch.dict(os.environ, env):
            for argv in commands:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        cli_main(argv)
                    except SystemExit as exc:
                        self.assertIn(exc.code, (0, None), argv)
        for name in ('broker.json', 'broker/keys.json', 'broker/vault/index.json'):
            self.assertTrue((home / name).is_file(), f'the populated home lacks {name}')
        out = Path(self.temporary.name) / 'out-populated'
        with mock.patch.dict(os.environ, {'TITANOS_HOME': str(home), 'TITANOS_ROLE': 'broker'}), contextlib.redirect_stdout(io.StringIO()):
            builder.main(['--out', str(out)])
        self.assertEqual(netguard.attempts(), [], 'the populated-home build attempted a non-loopback connection')
        self.assertEqual((out / builder.JSON_OUT).read_bytes(), self.rebuilt_json, 'a populated broker home changed the route matrix JSON')
        self.assertEqual((out / builder.MD_OUT).read_bytes(), self.rebuilt_md, 'a populated broker home changed the route matrix Markdown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
