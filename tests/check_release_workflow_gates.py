"""Release workflow gates: the source-package and public-export hygiene checks run in the draft-release job before any
artifact is built or attached, and the manual check matrix keeps running them.

Reads .github/workflows/release.yml and check.yml only; no network, model, service, build or git operation. PyYAML is used
when importable; otherwise (and always, as a cross-check) the `run:` step commands are read positionally from the text.
"""
import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / '.github/workflows/release.yml'
CHECK = ROOT / '.github/workflows/check.yml'
GATES = ('tests/check_source_package.py', 'tests/check_public_export_hygiene.py')

RUN_LINE = re.compile(r'^(?P<indent>\s*)(?:-\s+)?run:\s*(?P<value>.*?)\s*$')


def run_commands_from_text(path):
    """Every `run:` step command in file order (all jobs), without a YAML parser; block scalars are joined by newlines."""
    lines = path.read_text(encoding='utf-8').splitlines()
    commands = []
    index = 0
    while index < len(lines):
        match = RUN_LINE.match(lines[index])
        index += 1
        if not match: continue
        value = match.group('value')
        if value and value[0] in '|>':
            indent, block = len(match.group('indent')), []
            while index < len(lines) and (not lines[index].strip() or len(lines[index]) - len(lines[index].lstrip()) > indent):
                block.append(lines[index].strip()); index += 1
            value = '\n'.join(block)
        commands.append(value.strip())
    return commands


def run_commands_from_yaml(path):
    """{job name: [run command, ...]} in file order via PyYAML, or None when PyYAML is not importable."""
    if importlib.util.find_spec('yaml') is None: return None
    import yaml
    document = yaml.safe_load(path.read_text(encoding='utf-8'))
    return {name: [str(step['run']).strip() for step in job['steps'] if 'run' in step] for name, job in document['jobs'].items()}


def runs_script(command, script):
    tokens = command.split()
    return bool(tokens) and tokens[0] == 'python' and script in tokens


def first_index(commands, predicate, label):
    hits = [index for index, command in enumerate(commands) if predicate(command)]
    assert len(hits) == 1, f'{label}: expected exactly one step, found {len(hits)}'
    return hits[0]


def check():
    for path in (RELEASE, CHECK): assert path.is_file(), path
    release_text = run_commands_from_text(RELEASE)
    check_text = run_commands_from_text(CHECK)
    release_yaml = run_commands_from_yaml(RELEASE)
    check_yaml = run_commands_from_yaml(CHECK)
    if release_yaml is not None:
        # the positional reader is only trusted when it agrees with the real parser wherever that parser is available
        assert list(release_yaml) == ['package'], list(release_yaml)
        assert release_yaml['package'] == release_text, (release_yaml['package'], release_text)
        assert [command for job in check_yaml.values() for command in job] == check_text
        assert all(any(runs_script(command, gate) for command in check_yaml['contract']) for gate in GATES), \
            'check.yml contract job must keep both gates'
    release = release_text

    # (a) both gate commands are present in the release job, with -B so the checkout stays free of bytecode
    gate_index = {gate: first_index(release, lambda command, gate=gate: runs_script(command, gate), gate) for gate in GATES}
    for gate in GATES:
        assert release[gate_index[gate]] == f'python -B {gate}', release[gate_index[gate]]

    # (b) each gate runs before the release build, the sdist build and the step that attaches artifacts
    build = first_index(release, lambda command: runs_script(command, 'tools/build_release.py'), 'tools/build_release.py')
    sdist = first_index(release, lambda command: command.split()[:3] == ['python', '-m', 'build'] and '--sdist' in command.split(),
                        'python -m build --sdist')
    attach = first_index(release, lambda command: command.startswith('gh release create'), 'gh release create')
    for gate, index in gate_index.items():
        assert index < build < sdist < attach, f'{gate} at step {index}; build {build}, sdist {sdist}, attach {attach}'
    # the gates import setuptools (and tools/build_binary imports packaging, a dependency of build): installed earlier in the job
    install = [index for index, command in enumerate(release) if command.startswith('python -m pip install') and 'setuptools' in command]
    assert install and min(install) < min(gate_index.values()), 'setuptools must be installed before the gates run'

    # (c) the attach step still uploads the release directory and the sdist to the draft
    tokens = release[attach].split()
    assert {'release/*', 'dist/*'} <= set(tokens), tokens
    assert attach == len(release) - 1, 'nothing runs after the artifacts are attached'

    # the manual matrix still runs both gates, so their coverage there did not move into release.yml
    for gate in GATES:
        assert any(runs_script(command, gate) for command in check_text), f'check.yml must still run {gate}'

    parser = 'PyYAML and positional readers agree' if release_yaml is not None else 'positional reader only (PyYAML not importable)'
    print(f'PASS: release.yml runs {len(GATES)} public-export gates at steps {sorted(gate_index.values())} before build_release '
          f'(step {build}), sdist (step {sdist}) and gh release create (step {attach}) which attaches release/* and dist/*; '
          f'check.yml keeps both gates; {parser}; no build, network or git operation')


if __name__ == '__main__': check()
