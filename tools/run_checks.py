"""Run the Titanos check suite by tier, mirroring .github/workflows/check.yml.

tests/checks.json is the one list of checks. Its CI tiers mirror the workflow jobs step for step, and --sync-check
fails when the two drift, so a local run and CI execute the same commands in the same order.

    python -B tools/run_checks.py --list
    python -B tools/run_checks.py --tier base --fast
    python -B tools/run_checks.py --tier base --tier extras --keep-going --report checks-report.json
    python -B tools/run_checks.py --sync-check

Every listed check is a contract or regression script over mocked or tiny fixtures. Passing them qualifies no model,
accelerator or package; model and hardware acceptance stays with the `manual` tier's scripts and tools/qualify.py.
Checks run with the interpreter that runs this script, from the repository root, with PYTHONDONTWRITEBYTECODE=1.
The runner writes nothing under the repository except the optional --report file.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'tests' / 'checks.json'
# workflow steps that install or invoke this runner are not checks; --sync-check ignores them
IGNORED_STEP_PREFIXES = ('python -m pip install', 'npm install', 'npx playwright install', 'python -B tools/run_checks.py')
RUN_LINE = re.compile(r'^(?P<indent>\s*)(?:-\s+)?run:\s*(?P<value>.*?)\s*$')
JOB_LINE = re.compile(r'^  (?P<name>[A-Za-z0-9_-]+):\s*$')
SCOPE = ('contract and regression scripts over mocked or tiny fixtures; a pass is not model, accelerator, package or '
         'presentation qualification')


def load_manifest():
    if not MANIFEST.is_file():
        raise SystemExit(f'missing {MANIFEST.relative_to(ROOT).as_posix()}')
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not isinstance(data.get('tiers'), dict):
        raise SystemExit('tests/checks.json: expected schema_version 1 with a tiers object')
    for name, tier in data['tiers'].items():
        for entry in tier.get('checks', []):
            if not isinstance(entry.get('run'), str) or not entry['run'].split():
                raise SystemExit(f'tests/checks.json: tier {name} has an entry without a run command')
    return data


def script_of(run):
    """The tests/... or tools/... path a command runs, or None (module or -c invocations)."""
    for token in shlex.split(run, posix=True)[1:]:
        if token.startswith(('tests/', 'tools/')):
            return token
    return None


def skip_reason(tier, entry, fast):
    """None when the entry can run on this host, else a short reason (mirrors the workflow's job placement)."""
    if tier.get('manual'):
        return 'manual: ' + (entry.get('note') or 'needs real inputs')
    if entry.get('ci_only'):
        return 'CI only: ' + entry['ci_only']
    systems = entry.get('os') or tier.get('os')
    if systems and platform.system() not in systems:
        return 'needs ' + '/'.join(systems)
    for module in list(tier.get('needs', [])) + list(entry.get('needs', [])):
        if importlib.util.find_spec(module) is None:
            return 'missing Python module ' + module
    tokens = shlex.split(entry['run'], posix=True)
    if tokens[0] == 'node' and shutil.which('node') is None:
        return 'node is not installed'
    node_module = entry.get('node_module') or tier.get('node_module')
    if node_module and not (ROOT / 'node_modules' / node_module).exists():
        return f'node module {node_module} is not installed under node_modules/'
    if entry.get('requires_path') and not (ROOT / entry['requires_path']).exists():
        return entry['requires_path'] + ' is not in this tree'
    if fast and entry.get('slow'):
        return 'slow; skipped by --fast'
    return None


def command(run):
    tokens = shlex.split(run, posix=True)
    if tokens[0] == 'python':
        tokens[0] = sys.executable
    elif tokens[0] == 'node':
        tokens[0] = shutil.which('node') or 'node'
    return tokens


def untried_report(stdout):
    """Whether a check's stdout carries a lane report whose overall result is untried (the sdk tier's broker lanes exit
    0 when no SDK or fixture is available here and say so in their JSON report); such a run is shown as UNTRIED."""
    return bool(re.search(r'"lane":', stdout or '')) and bool(re.search(r'"result":\s*"untried"', stdout or ''))


def run_entry(entry, verbose):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    untried = False
    if verbose:
        proc = subprocess.run(command(entry['run']), cwd=ROOT, env=env)
        tail = ''
    else:
        proc = subprocess.run(command(entry['run']), cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding='utf-8', errors='replace')
        tail = '\n'.join((proc.stdout + proc.stderr).splitlines()[-40:])
        untried = proc.returncode == 0 and untried_report(proc.stdout)
    return proc.returncode, round(time.monotonic() - started, 2), tail, untried


def workflow_jobs(path):
    """{job name: [run command, ...]} in file order, read positionally like tests/check_release_workflow_gates.py."""
    lines = path.read_text(encoding='utf-8').splitlines()
    jobs, current, in_jobs = {}, None, False
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.rstrip() == 'jobs:':
            in_jobs = True
            continue
        if not in_jobs:
            continue
        job = JOB_LINE.match(line)
        if job:
            current = job.group('name')
            jobs[current] = []
            continue
        match = RUN_LINE.match(line)
        if not match or current is None:
            continue
        value = match.group('value')
        if value and value[0] in '|>':
            indent, block = len(match.group('indent')), []
            while index < len(lines) and (not lines[index].strip() or len(lines[index]) - len(lines[index].lstrip()) > indent):
                block.append(lines[index].strip())
                index += 1
            value = '\n'.join(block)
        jobs[current].append(value.strip())
    return jobs


def sync_check(data):
    """tests/checks.json mirrors the workflow jobs it names, and every check file is in a tier."""
    problems = []
    expected = {}
    for name, tier in data['tiers'].items():
        mapping = tier.get('workflow')
        if not mapping:
            continue
        key = (mapping['file'], mapping['job'])
        expected.setdefault(key, []).extend(entry['run'] for entry in tier['checks'])
    parsed = {}
    for (file, job), runs in expected.items():
        if file not in parsed:
            path = ROOT / file
            if not path.is_file():
                problems.append(f'{file}: workflow file named by tests/checks.json is missing')
                continue
            parsed[file] = workflow_jobs(path)
        jobs = parsed.get(file, {})
        if job not in jobs:
            problems.append(f'{file}: job {job} named by tests/checks.json is missing (jobs: {", ".join(jobs)})')
            continue
        actual = [step for step in jobs[job] if not step.startswith(IGNORED_STEP_PREFIXES)]
        if actual != runs:
            problems.append(f'{file} job {job}: workflow steps and tests/checks.json differ')
            for position, (step, run) in enumerate(zip(actual + [None] * (len(runs) - len(actual)),
                                                       runs + [None] * (len(actual) - len(runs)))):
                if step != run:
                    problems.append(f'    step {position}: workflow {step!r} / manifest {run!r}')
    helpers = set(data.get('helpers', []))
    files = sorted(path.relative_to(ROOT).as_posix() for pattern in ('check*.py', 'check*.js')
                   for path in (ROOT / 'tests').glob(pattern))
    placement = {}
    for name, tier in data['tiers'].items():
        for entry in tier['checks']:
            script = script_of(entry['run'])
            if script and script.startswith('tests/'):
                placement.setdefault(script, []).append(name)
                if not (ROOT / script).is_file():
                    problems.append(f'tests/checks.json tier {name}: {script} does not exist')
    ci_tiers = {name for name, tier in data['tiers'].items()
                if (tier.get('workflow') or {}).get('file') == '.github/workflows/check.yml'}
    for file in files:
        if file in helpers:
            continue
        tiers = placement.get(file, [])
        if not tiers:
            problems.append(f'{file} is in no tier of tests/checks.json')
        elif len({t for t in tiers if t in ci_tiers}) > 1:
            problems.append(f'{file} is in more than one check.yml tier: {", ".join(tiers)}')
    for helper in helpers:
        if not (ROOT / helper).is_file():
            problems.append(f'helper {helper} named by tests/checks.json does not exist')
    if problems:
        print('FAIL: tests/checks.json and the workflows disagree')
        for problem in problems:
            print('  ' + problem)
        return 1
    mirrored = ', '.join(f'{file.rsplit("/", 1)[-1]} job {job}' for file, job in expected)
    print(f'PASS: tests/checks.json mirrors {mirrored}; {len(files) - len(helpers & set(files))} check files, each in a tier; '
          f'{sum(len(t["checks"]) for t in data["tiers"].values())} entries; no check was run')
    return 0


def list_tiers(data, names, fast):
    for name in names:
        tier = data['tiers'][name]
        mapping = tier.get('workflow')
        where = f"{mapping['file']} job {mapping['job']}" if mapping else 'not in a workflow'
        print(f"{name}: {len(tier['checks'])} checks; {tier.get('description', '')}")
        print(f'  where: {where}')
        for line in tier.get('install', []):
            print(f'  install: {line}')
        for entry in tier['checks']:
            reason = skip_reason(tier, entry, fast)
            marker = 'skip: ' + reason if reason else 'run'
            note = ('  # ' + entry['note']) if entry.get('note') else ''
            print(f"  [{marker}] {entry['run']}{note}")
        print()


def main(argv=None):
    data = load_manifest()
    tier_names = list(data['tiers'])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     epilog='tiers: ' + ', '.join(tier_names) + ', all (every tier except manual)')
    parser.add_argument('--tier', action='append', choices=tier_names + ['all'], default=None,
                        help='tier to run or list; repeatable (default: base)')
    parser.add_argument('--list', action='store_true', help='print the tiers and what would run or skip on this host')
    parser.add_argument('--fast', action='store_true', help='skip checks marked slow in tests/checks.json')
    parser.add_argument('--keep-going', action='store_true', help='run every check and report all failures')
    parser.add_argument('--verbose', action='store_true', help='stream check output instead of showing a tail on failure')
    parser.add_argument('--report', type=Path, help='write a JSON report of this run')
    parser.add_argument('--sync-check', action='store_true',
                        help='verify that tests/checks.json mirrors the workflow jobs and covers every check file; runs nothing')
    args = parser.parse_args(argv)
    if args.sync_check:
        return sync_check(data)
    selected = args.tier or (tier_names if args.list else ['base'])
    if 'all' in selected:
        selected = [name for name in tier_names if not data['tiers'][name].get('manual')]
    if args.list:
        list_tiers(data, selected, args.fast)
        return 0
    results = []
    failed = False
    for name in selected:
        tier = data['tiers'][name]
        print(f"== {name}: {tier.get('description', '')}")
        for entry in tier['checks']:
            reason = skip_reason(tier, entry, args.fast)
            if reason:
                print(f"SKIP          {entry['run']}  ({reason})")
                results.append({'tier': name, 'run': entry['run'], 'status': 'skipped', 'reason': reason})
                continue
            code, seconds, tail, untried = run_entry(entry, args.verbose)
            status = 'FAIL' if code else ('UNTRIED' if untried else 'PASS')
            print(f"{status} {seconds:7.2f}s {entry['run']}", flush=True)
            results.append({'tier': name, 'run': entry['run'], 'status': status.lower(), 'seconds': seconds, 'exit_code': code})
            if code:
                failed = True
                if tail:
                    print(tail)
                if not args.keep_going:
                    break
        if failed and not args.keep_going:
            break
    counts = {status: sum(1 for result in results if result['status'] == status)
              for status in ('pass', 'fail', 'skipped', 'untried')}
    untried_note = f", {counts['untried']} untried" if counts['untried'] else ''
    print(f"{'FAIL' if failed else 'PASS'}: {counts['pass']} passed, {counts['fail']} failed, {counts['skipped']} skipped"
          f"{untried_note}; {SCOPE}")
    if args.report:
        report = {'tiers': selected, 'fast': args.fast, 'platform': platform.platform(), 'python': sys.version.split()[0],
                  'node': shutil.which('node') is not None, 'passed': not failed, 'results': results, 'scope': SCOPE}
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
