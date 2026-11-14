"""Source/notices packaging checks using inert bytes, not native runtimes/models."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sherpa_package_check', ROOT / 'tools/sherpa_package.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

def rejected(call):
    try: call()
    except (ValueError, FileNotFoundError): return
    raise AssertionError('Invalid packaging input was accepted')

def check():
    platform = 'windows' if sys.platform == 'win32' else 'linux'
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / 'source'
        source.mkdir()
        (source / 'LICENSE').write_text('inert license fixture')
        binary = b'inert native-binary fixture, never loaded'
        manifest = {'runtime_version': '1.13.8', 'source_git_revision': 'fixture',
                    'files': {'LICENSE': {'bytes': (source / 'LICENSE').stat().st_size, 'sha256': helper.digest(source / 'LICENSE')}},
                    'components': [{'name': 'sherpa-onnx', 'version': '1.13.8', 'license': 'Apache-2.0',
                                    'license_files': ['LICENSE'], 'source_url': 'https://example.invalid/source'},
                                   {'name': 'eSpeak NG', 'version': 'fixture', 'license': 'GPL-3.0-or-later',
                                    'license_files': ['LICENSE'], 'source_url': 'https://example.invalid/espeak'}],
                    'native_files': {platform: [{'name': 'fixture.so', 'bytes': len(binary), 'sha256': hashlib.sha256(binary).hexdigest()}]}}
        path = source / 'SOURCE_BUNDLE.json'
        path.write_text(json.dumps(manifest))
        plan = helper.source_plan(source)
        output = root / 'output'
        (output / '_internal').mkdir(parents=True)
        (output / '_internal/fixture.so').write_bytes(binary)
        for name, value in [('NATIVE_COMPONENTS.json', []), ('sbom.cdx.json', {'components': []}),
                            ('DEPENDENCIES.json', {'components': [{'name': 'sherpa-onnx-core', 'license_files': []}]})]:
            (output / name).write_text(json.dumps(value))
        (output / 'THIRD_PARTY_NOTICES.md').write_text('Base notices\n')
        (output / 'LICENSE').write_text('Titanos Apache source license, unchanged')
        before = (output / 'LICENSE').read_bytes()
        record = helper.include_native_sources(output, plan)
        assert record['native_files'][0]['packaged_paths'] == ['_internal/fixture.so']
        assert (output / 'THIRD_PARTY_SOURCES/sherpa-onnx-1.13.8/LICENSE').is_file()
        notices_text = (output / 'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')
        assert 'sherpa-onnx-core' in notices_text and 'Base notices' in notices_text, 'the generated Sherpa notice section must name the sherpa-onnx-core distribution explicitly'
        licenses = {item['licenses'][0]['expression'] for item in json.loads((output / 'sbom.cdx.json').read_text())['components']}
        assert licenses == {'Apache-2.0', 'GPL-3.0-or-later'}
        assert json.loads((output / 'DEPENDENCIES.json').read_text())['components'][0]['license_files']
        assert (output / 'LICENSE').read_bytes() == before
        (source / 'LICENSE').write_text('changed')
        rejected(lambda: helper.source_plan(source))
        rejected(lambda: helper.include_native_sources(output, plan))
        for name in ['../escape', '/absolute', 'C:/foreign', 'x\\y']:
            rejected(lambda name=name: helper.relative(name))

    # The frozen-build MCP gate reads its module list from build_binary.MCP_TOOL_SOURCES and its expected set from
    # expected_mcp_tools; this check binds to those names rather than repeating a module list or a tool count.
    tree = ast.parse((ROOT / 'tools/build_binary.py').read_text(encoding='utf-8'))
    functions = {'declared_mcp_tools', 'mcp_tool_source_paths', 'expected_mcp_tools'}
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in functions)
             or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'MCP_TOOL_SOURCES' for target in node.targets))]
    assert len(nodes) == len(functions) + 1, 'MCP_TOOL_SOURCES must be one module-level assignment beside the three gate functions'
    namespace = {'ast': ast, 'Path': Path, 'ROOT': ROOT}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / 'tools/build_binary.py'), 'exec'), namespace)
    declarations, sources = namespace['declared_mcp_tools'], namespace['MCP_TOOL_SOURCES']
    assert 'src/titanos/mcp_server.py' in sources and 'src/titanos/broker/mcp_tools.py' in sources, sources
    assert {'chat', 'list_model_providers', 'submit_model_acquisition'} <= declarations(ROOT / 'src/titanos/mcp_server.py')
    expected = namespace['expected_mcp_tools'](ROOT)
    assert expected == declarations(namespace['mcp_tool_source_paths'](ROOT))
    assert declarations(ROOT / 'src/titanos/mcp_server.py') < expected, 'the gate expects every MCP_TOOL_SOURCES module, not mcp_server.py alone'
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / 'mcp.py'
        source.write_text('@server.tool()\ndef ordinary(): pass\n@server.tool(name="renamed")\nasync def second(): pass\n')
        assert declarations(source) == {'ordinary', 'renamed'}
        source.write_text('@server.tool(name="same")\ndef first(): pass\n@server.tool(name="same")\ndef second(): pass\n')
        rejected(lambda: declarations(source))
        source.write_text('VALUE=1\n')
        rejected(lambda: declarations(source))
    print(f'PASS: source/native hash binding, shipped notices and mixed SBOM, unchanged Titanos license, tamper/traversal refusal, '
          f'{len(expected)} MCP declarations over the {len(sources)} build_binary.MCP_TOOL_SOURCES modules and duplicate/empty rejection; '
          f'no native/model execution')

if __name__ == '__main__': check()
