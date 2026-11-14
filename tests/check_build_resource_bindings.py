"""Bind packaged declarative catalog bytes and split MCP registrations; no native work."""
import ast
import fnmatch
import hashlib
from pathlib import Path
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]

def check():
    tree = ast.parse((ROOT / 'tools/build_binary.py').read_text(encoding='utf-8'))
    expressions = []
    for function, variable in [('build_plan', 'modules'), ('main', 'current_files')]:
        node = next(value for value in tree.body if isinstance(value, ast.FunctionDef) and value.name == function)
        value = next(statement.value for statement in node.body if isinstance(statement, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == variable for target in statement.targets))
        expressions.append(compile(ast.Expression(value), 'actual source-map expression', 'eval'))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        catalog = root / 'src/titanos/data/native-profiles.json'
        catalog.parent.mkdir(parents=True)
        catalog.write_text('{"profiles": []}\n')
        scope = {'ROOT': root, 'digest': digest}
        initial = eval(expressions[0], scope)
        assert initial['src/titanos/data/native-profiles.json'] == digest(catalog)
        assert eval(expressions[1], scope) == initial
        catalog.write_text('{"profiles": ["changed"]}\n')
        assert eval(expressions[1], scope) != initial
    settings = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    patterns = settings['tool']['setuptools']['package-data']['titanos']
    assert any(fnmatch.fnmatch('data/native-profiles.json', pattern) for pattern in patterns)
    actual = eval(expressions[0], {'ROOT': ROOT, 'digest': digest})
    assert actual['src/titanos/data/native-profiles.json'] == digest(ROOT / 'src/titanos/data/native-profiles.json')

    # The frozen-build MCP gate: build_binary.MCP_TOOL_SOURCES is the one list of registration modules and expected_mcp_tools the
    # set the gate expects (tests/check_offline_binary.py asserts that set against the real create_mcp registration). No module
    # list is repeated here, so a module added to the tuple is covered without a change to this check.
    functions = {'declared_mcp_tools', 'mcp_tool_source_paths', 'expected_mcp_tools'}
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in functions)
             or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'MCP_TOOL_SOURCES' for target in node.targets))]
    assert len(nodes) == len(functions) + 1, 'MCP_TOOL_SOURCES must be one module-level assignment beside the three gate functions'
    scope = {'ast': ast, 'Path': Path, 'ROOT': ROOT}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual MCP declarations', 'exec'), scope)
    declared, sources = scope['declared_mcp_tools'], scope['MCP_TOOL_SOURCES']
    assert isinstance(sources, tuple) and len(sources) == len(set(sources)) >= 4, sources
    assert {'src/titanos/mcp_server.py', 'src/titanos/broker/mcp_tools.py'} <= set(sources)
    paths = scope['mcp_tool_source_paths'](ROOT)
    assert paths == [ROOT / name for name in sources] and all(path.is_file() for path in paths)
    full = scope['expected_mcp_tools'](ROOT)
    assert {'chat', 'discover_models', 'plan_runtime_setup', 'prepare_runtime', 'retry_runtime_setup', 'broker_status', 'whoami'} <= full
    assert full == declared(paths) == set().union(*(declared(path) for path in paths))
    with tempfile.TemporaryDirectory() as directory:
        first, second = Path(directory) / 'first.py', Path(directory) / 'second.py'
        for path in (first, second): path.write_text('@server.tool()\ndef duplicate(): pass\n')
        try: declared([first, second])
        except ValueError: pass
        else: raise AssertionError('Cross-module duplicate tool registration accepted')
        try: scope['mcp_tool_source_paths'](Path(directory))
        except ValueError: pass
        else: raise AssertionError('A tree missing the declaration sources was accepted')
    print(f'PASS: native catalog included, mutation detected, package-data declared; {len(full)} MCP tools declared across the '
          f'{len(sources)} build_binary.MCP_TOOL_SOURCES modules (the set the frozen-build gate expects); no model/native calls')

if __name__ == '__main__': check()
