"""Build the hash locks of a pinned-interpreter install: tools/locks/python-lock-<platform>.txt, one per platform.

A pinned install unpacks the CPython build tools/python-lock.json names (pinned by sha256 and size), makes a virtual
environment from it, installs this project's wheel and the source_wheels offline with --no-deps, then installs one lock
with pip --require-hashes --no-deps --only-binary=:all: from the indexes the JSON lists. So the lock names every
package the engine needs, each on one line as name==version followed by the sha256 of every wheel of that version for
the platform on those indexes (pip may take any of them), with no environment marker, URL, path, option or extra, and
no package twice: the grammar parse_lock checks.

What it holds: pyproject.toml's dependencies and the extras the JSON lists, evaluated for the platform, with the JSON's
pins, plus what each source wheel requires, less leave_out; titanos-engine and the source wheels themselves install
from their own wheels and are not in it. Versions come from uv (uv pip compile for the platform, CPython 3.12, wheels
only, across the indexes); hashes from the indexes' simple pages.

    python tools/build_python_lock.py --uv path/to/uv            # rebuild every platform's lock (network)
    python tools/build_python_lock.py --check                    # offline: grammar and coverage of the committed locks

build_source_wheels (also used by tools/build_release.py) builds each source wheel from a git archive of its pinned
commit, with LF line endings and SOURCE_DATE_EPOCH at the commit time, so the wheel is the same on every builder.
"""
import argparse
import html.parser
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / 'tools' / 'python-lock.json'
LOCKS = ROOT / 'tools' / 'locks'
ENGINE = 'titanos-engine'
MAX_PACKAGES = 4000
REQUIREMENT = re.compile(r'^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==([0-9A-Za-z][0-9A-Za-z.!+_-]*)$')
HASH = re.compile(r'^--hash=sha256:([0-9a-fA-F]{64})$')
ENVIRONMENTS = {
    'win_amd64': {'sys_platform': 'win32', 'platform_system': 'Windows', 'platform_machine': 'AMD64', 'os_name': 'nt'},
    'linux_x86_64': {'sys_platform': 'linux', 'platform_system': 'Linux', 'platform_machine': 'x86_64',
                     'os_name': 'posix'},
}


def lock_path(platform):
    return LOCKS / f'python-lock-{platform}.txt'


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def load_pins():
    return json.loads(PINS.read_text(encoding='utf-8'))


def lock_lines(text):
    """(line number, logical line): a trailing backslash joins the next line; # at the start or after a space or tab
    starts a comment, as pip reads a requirements file."""
    out, current, start = [], '', 0
    for number, raw in enumerate(text.split('\n'), 1):
        raw = raw.rstrip('\r')
        if not current:
            start = number
        comment = re.search(r'(^|[ \t])#', raw)
        if comment:
            raw = raw[:comment.start()]
        if raw.endswith('\\'):
            current += raw[:-1] + ' '
            continue
        out.append((start, current + raw))
        current = ''
    if current:
        out.append((start, current))
    return out


def parse_lock(text):
    """[(name, version, [sha256, ...])] of a lock, or ValueError naming the first line that breaks the grammar."""
    text = text.removeprefix('\ufeff')
    packages, seen = [], {}
    for number, line in lock_lines(text):
        fields = line.split()
        if not fields:
            continue
        req = fields[0]
        for bad, why in (('-', 'is an option'), (';', 'has an environment marker'), ('://', 'is a URL'),
                         ('@', 'is a direct reference'), ('/', 'is a path'), ('\\', 'is a path'),
                         ('[', 'names extras'), ('===', 'pins with ===')):
            if (req.startswith(bad) if bad == '-' else bad in req):
                raise ValueError(f"the lock's line {number}: {req!r} {why}")
        match = REQUIREMENT.match(req)
        if not match:
            raise ValueError(f"the lock's line {number}: {req!r} is not name==version")
        hashes = []
        for field in fields[1:]:
            found = HASH.match(field)
            if not found and field.startswith(';'):
                raise ValueError(f"the lock's line {number}: {match[1]} has an environment marker")
            if not found:
                raise ValueError(f"the lock's line {number}: {field!r} after {match[1]} is not --hash=sha256:<64 hex>")
            hashes.append(found[1].lower())
        if not hashes:
            raise ValueError(f"the lock's line {number}: {req} has no --hash=sha256:<hex>")
        key = normalized(match[1])
        if key in seen:
            raise ValueError(f"the lock's line {number}: {match[1]} is named again (first on line {seen[key]})")
        seen[key] = number
        packages.append((match[1], match[2], hashes))
        if len(packages) > MAX_PACKAGES:
            raise ValueError(f'the lock names more than {MAX_PACKAGES} packages')
    return packages


def build_source_wheels(out, pins=None):
    """Build each source wheel the pins name into out from a git archive of its commit; the wheel paths."""
    pins = pins or load_pins()
    out.mkdir(parents=True, exist_ok=True)
    wheels = []
    for name, source in pins['source_wheels'].items():
        with tempfile.TemporaryDirectory(prefix='titanos-source-wheel-') as tmp:
            repo, tree = Path(tmp) / 'repo', Path(tmp) / 'tree'
            git = ['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf']
            subprocess.run(git + ['init', '-q', str(repo)], check=True)
            subprocess.run(git + ['-C', str(repo), 'fetch', '-q', '--depth', '1', source['url'], source['commit']],
                           check=True)
            stamp = subprocess.run(git + ['-C', str(repo), 'log', '-1', '--format=%ct', source['commit']], check=True,
                                   capture_output=True, text=True).stdout.strip()
            archive = Path(tmp) / 'source.tar'
            subprocess.run(git + ['-C', str(repo), 'archive', '--format=tar', '-o', str(archive), source['commit']],
                           check=True)
            with tarfile.open(archive) as handle:
                handle.extractall(tree, filter='data')
            before = set(out.glob('*.whl'))
            subprocess.run([sys.executable, '-m', 'pip', 'wheel', '-q', '--no-deps', '--no-build-isolation',
                            '--wheel-dir', str(out), str(tree)], check=True,
                           env={**os.environ, 'SOURCE_DATE_EPOCH': stamp})
            built = [path for path in set(out.glob('*.whl')) - before if normalized(path.name.split('-')[0]) ==
                     normalized(name)]
            if len(built) != 1:
                raise SystemExit(f'building {name} at {source["commit"]} gave {len(built)} wheels')
            wheels.append(built[0])
    return wheels


def wheel_requires(wheel):
    with zipfile.ZipFile(wheel) as archive:
        meta = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
        text = archive.read(meta).decode('utf-8')
    return [line.split(':', 1)[1].strip() for line in text.splitlines() if line.startswith('Requires-Dist:')]


def environment(platform, pins):
    return {**ENVIRONMENTS[platform], 'python_version': pins['python'], 'python_full_version': pins['python'] + '.0',
            'implementation_name': 'cpython', 'platform_python_implementation': 'CPython', 'extra': ''}


def requirements(platform, pins, source_requires):
    """The requirements the lock must satisfy on the platform: (Requirement, origin) without markers or direct refs."""
    from packaging.requirements import Requirement
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
    wanted = [(text, 'dependencies') for text in project['dependencies']]
    for extra in pins['extras']:
        wanted += [(text, f'extra {extra}') for text in project['optional-dependencies'][extra]]
    for name, texts in source_requires.items():
        wanted += [(text, name) for text in texts]
    env, leave_out, sources = environment(platform, pins), set(pins['leave_out']), set(map(normalized, pins['source_wheels']))
    out = []
    for text, origin in wanted:
        req = Requirement(text)
        if req.marker is not None and not req.marker.evaluate(env):
            continue
        if normalized(req.name) in leave_out or normalized(req.name) in sources or normalized(req.name) == ENGINE:
            continue
        if req.url:
            raise SystemExit(f'{origin}: {text} is a direct reference; list it under source_wheels')
        req.marker = None
        out.append((req, origin))
    return out


def resolve(uv, platform, pins, reqs):
    """[(name, version)] uv chooses for the requirements and pins on the platform (wheels only, every index)."""
    with tempfile.TemporaryDirectory(prefix='titanos-lock-') as tmp:
        source = Path(tmp) / 'requirements.in'
        lines = [str(req) for req, _ in reqs] + pins['pins']
        source.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        target = Path(tmp) / 'requirements.txt'
        command = [uv, 'pip', 'compile', str(source), '--output-file', str(target), '--python-version', pins['python'],
                   '--python-platform', pins['platforms'][platform]['resolver_platform'], '--only-binary', ':all:',
                   '--index-url', pins['indexes'][0], '--index-strategy', 'unsafe-best-match', '--no-header',
                   '--no-annotate', '--quiet']
        for extra in pins['indexes'][1:]:
            command += ['--extra-index-url', extra]
        subprocess.run(command, check=True, env={k: v for k, v in os.environ.items() if not k.startswith('UV_')})
        chosen = []
        for _, line in lock_lines(target.read_text(encoding='utf-8')):
            line = line.strip()
            if not line:
                continue
            match = REQUIREMENT.match(line.split()[0])
            if not match:
                raise SystemExit(f'uv wrote {line!r}, not name==version')
            chosen.append((match[1], match[2]))
    return chosen


class Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.files, self._href, self._attrs = [], None, {}

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self._attrs = dict(attrs)
            self._href = self._attrs.get('href')

    def handle_data(self, data):
        if self._href is not None:
            digest = urllib.parse.urlsplit(self._href).fragment
            if digest.startswith('sha256=') and 'data-yanked' not in self._attrs:
                self.files.append((data.strip(), digest[len('sha256='):]))
            self._href = None


PAGES = {}


def index_files(index, name):
    """[(file name, sha256)] of a package's page on a simple index (PEP 691 JSON or the HTML page), yanked files skipped."""
    url = f'{index.rstrip("/")}/{normalized(name)}/'
    if url in PAGES:
        return PAGES[url]
    request = urllib.request.Request(url, headers={'Accept': 'application/vnd.pypi.simple.v1+json, text/html;q=0.1'})
    try:
        with urllib.request.urlopen(request, timeout=120) as reply:
            body, kind = reply.read(), reply.headers.get('Content-Type', '')
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            PAGES[url] = []
            return []
        raise
    if 'json' in kind:
        files = [(f['filename'], f['hashes']['sha256']) for f in json.loads(body)['files']
                 if f.get('hashes', {}).get('sha256') and not f.get('yanked')]
    else:
        parser = Links()
        parser.feed(body.decode('utf-8'))
        files = parser.files
    PAGES[url] = files
    return files


def supported_tags(platform, pins):
    from packaging import tags
    version = tuple(int(part) for part in pins['python'].split('.'))
    if platform == 'win_amd64':
        platforms = ['win_amd64']
    else:  # every manylinux wheel: pip on a newer glibc prefers the newest one it can run
        platforms = [f'manylinux_2_{minor}_x86_64' for minor in range(60, 4, -1)]
        platforms += ['manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64', 'linux_x86_64']
    return (set(tags.cpython_tags(version, abis=[f'cp{version[0]}{version[1]}', 'abi3', 'none'], platforms=platforms))
            | set(tags.compatible_tags(version, f'cp{version[0]}{version[1]}', platforms)))


def wheel_hashes(name, version, platform, pins):
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from packaging.version import Version
    accepted, wanted, found = supported_tags(platform, pins), Version(version), {}
    for index in pins['indexes']:
        for filename, digest in index_files(index, name):
            try:
                wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(filename)
            except InvalidWheelFilename:
                continue
            if normalized(wheel_name) == normalized(name) and wheel_version == wanted and wheel_tags & accepted:
                found[digest.lower()] = filename
    if not found:
        raise SystemExit(f'{name}=={version}: no {platform} wheel on {", ".join(pins["indexes"])}')
    return sorted(found)


def render(platform, pins, packages):
    head = [f'# Hash lock for a pinned CPython {pins["python"]} install on {platform}, built by tools/build_python_lock.py',
            "# from pyproject.toml's dependencies and the " + ' and '.join(pins['extras']) + ' extras with tools/python-lock.json.',
            '# Each line: one package at one version and the sha256 of every wheel of that version for this platform on',
            '# ' + ' '.join(pins['indexes']),
            f'# {ENGINE} and ' + ', '.join(pins['source_wheels']) + ' install from their own wheels and are not listed.']
    body = [f'{name}==' + version + ''.join(f' --hash=sha256:{digest}' for digest in hashes)
            for name, version, hashes in sorted(packages, key=lambda row: normalized(row[0]))]
    return '\n'.join(head + body) + '\n'


def build(uv, platforms, pins):
    with tempfile.TemporaryDirectory(prefix='titanos-source-wheels-') as tmp:
        source_requires = {name: wheel_requires(path) for name, path in
                           zip(pins['source_wheels'], build_source_wheels(Path(tmp), pins))}
    LOCKS.mkdir(exist_ok=True)
    for platform in platforms:
        reqs = requirements(platform, pins, source_requires)
        chosen = resolve(uv, platform, pins, reqs)
        packages = [(name, version, wheel_hashes(name, version, platform, pins)) for name, version in chosen]
        text = render(platform, pins, packages)
        parse_lock(text)
        lock_path(platform).write_text(text, encoding='utf-8', newline='\n')
        print(f'{lock_path(platform).relative_to(ROOT).as_posix()}: {len(packages)} packages')
    (LOCKS / 'source-requires.json').write_text(json.dumps(source_requires, indent=2, sort_keys=True) + '\n',
                                                encoding='utf-8', newline='\n')


def check(pins=None):
    """Problems with the committed locks: grammar, and every requirement (and pin) satisfied by its locked version."""
    from packaging.version import Version
    pins = pins or load_pins()
    problems = []
    recorded = LOCKS / 'source-requires.json'
    if not recorded.is_file():
        return [f'{recorded.relative_to(ROOT).as_posix()} is missing (run tools/build_python_lock.py)']
    source_requires = json.loads(recorded.read_text(encoding='utf-8'))
    if sorted(source_requires) != sorted(pins['source_wheels']):
        problems.append('tools/locks/source-requires.json does not cover the source wheels tools/python-lock.json names')
    for platform in pins['platforms']:
        path = lock_path(platform)
        if not path.is_file():
            problems.append(f'{path.relative_to(ROOT).as_posix()} is missing')
            continue
        try:
            packages = parse_lock(path.read_text(encoding='utf-8'))
        except ValueError as exc:
            problems.append(f'{path.name}: {exc}')
            continue
        locked = {normalized(name): version for name, version, _ in packages}
        for name in [ENGINE, *pins['source_wheels'], *pins['leave_out']]:
            if normalized(name) in locked:
                problems.append(f'{path.name}: {name} is locked, but it installs from its own wheel or is not to be installed')
        for req, origin in requirements(platform, pins, source_requires):
            version = locked.get(normalized(req.name))
            if version is None:
                problems.append(f'{path.name}: {req.name} ({origin}) is not locked')
            elif req.specifier and not req.specifier.contains(Version(version), prereleases=True):
                problems.append(f'{path.name}: {req.name}=={version} does not satisfy {req} ({origin})')
        for pin in pins['pins']:
            name, version = pin.split('==')
            if locked.get(normalized(name)) != version:
                problems.append(f'{path.name}: {pin} is pinned, locked {locked.get(normalized(name))}')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--uv', default='uv', help='the uv program that resolves versions')
    parser.add_argument('--platform', action='append', help='only this platform (repeatable)')
    parser.add_argument('--check', action='store_true', help='check the committed locks offline and write nothing')
    args = parser.parse_args()
    pins = load_pins()
    if args.check:
        problems = check(pins)
        for problem in problems:
            print('FAIL:', problem)
        if not problems:
            print('PASS: ' + ', '.join(lock_path(p).name for p in pins['platforms']) + ' parse and cover the requirements')
        raise SystemExit(1 if problems else 0)
    build(args.uv, args.platform or list(pins['platforms']), pins)


if __name__ == '__main__':
    main()
