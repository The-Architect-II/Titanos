"""Build the public branch of Titanos from a private commit, without the private history.

The private repository's history carries lab records that never leave it (tools/public_export.json excluded_paths and
private_names). The public repository therefore starts from a fresh root commit whose tree is the private commit minus
those records. This script produces that tree and commits it once on an orphan branch inside a separate git worktree.

What it reads: only git objects of the source commit (`git ls-tree`, `git cat-file --batch`), never the working tree, so
untracked files, local edits and checkout line-ending conversion cannot leak in. Every file is written with its exact
index bytes; after the commit the script asserts that every unchanged file has the same blob id as in the source commit.

What it drops (all recorded in the report and in the marker file .titanos-public-tree):
- paths under tools/public_export.json `excluded_paths` (case-insensitive, a directory drops its whole tree);
- files whose name is a key of tools/public_export.json `private_names`, and paths listed in its `private_files`;
- tracked files that the source commit's .gitignore would ignore (force-added by mistake);
- tools/public_branch/ (this builder's own private-side inputs) and any path listed in tools/public_branch/remove.txt;
- paths given with --drop.

What it changes:
- tools/public_branch/overrides/<path> in the source commit replaces <path> in the public tree (whole-file overrides);
- the marker file .titanos-public-tree is written at the repository root. Its presence tells tree-aware checks that this
  is a public tree (tools/public_export.json `tree_kinds`: the environment variable TITANOS_TREE_KIND wins when set, then
  the marker, then private); in a public tree tests/check_public_export_hygiene.py asserts that the private records are
  absent instead of present. The marker's content is a small JSON record of the source commit and the absent records.

Every other kept file, the workflows included, is written unchanged. The workflow step that runs the archived controller's
tests from tools/validation/gen01 stays in the public tree: it is guarded by `hashFiles(...)` and skips where the tree is
absent, and tests/checks.json mirrors the workflow step for step (`tools/run_checks.py --sync-check` fails when the two
differ), so removing the step here would break the public tree's own contract job.

What it checks before committing: every written text file is scanned with the policy's detection classes, the fixture
allowlist, the exporting machine's own name (COMPUTERNAME, HOSTNAME, the interpreter's node name) and, when the private
host map is present (tools/lab_hosts.private.json next to tools/lab_paths.py, or the file TITANOS_LAB_HOSTS_FILE names;
never tracked), the names it denies (tools/lab_paths.private_names(), class private_denylist): the same rules
tests/check_public_export_hygiene.py applies to the export copies; one hit aborts. Then the tree is written and staged in the
new worktree (so `git ls-files` there lists exactly the public tree) and the release gates run inside it (`--no-checks` skips
them). Only when everything passes is the single root commit made. References from the remaining documents into the
dropped records are counted and reported; they dangle by design (docs/DISTRIBUTION.md).

What it never does: touch the source repository's working tree or branches, rewrite or force-push anything, push at all.
Publishing is a manual step printed at the end. `--dry-run` computes and prints the report without creating a worktree.

Requirements: git 2.42 or newer creates the orphan worktree in one step (`git worktree add --orphan`); with an older git
(2.23 or newer) the builder checks the source commit out detached and switches to the unborn orphan branch instead, which
leaves the same empty worktree.

Usage:
  python tools/build_public_branch.py --dry-run [--report public-branch-report.json]
  python tools/build_public_branch.py --worktree ../titanos-public [--branch public] [--source-ref HEAD]
                                      [--drop PATH ...] [--identity "Name <email>"] [--no-checks] [--report FILE]
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import lab_paths  # noqa: E402  (the private host map's names, read the way tests/check_public_export_hygiene.py reads them)

POLICY_PATH = 'tools/public_export.json'
MARKER_PATH = '.titanos-public-tree'
BUILDER_DIR = 'tools/public_branch/'
OVERRIDES_DIR = BUILDER_DIR + 'overrides/'
REMOVE_LIST = BUILDER_DIR + 'remove.txt'
WORKFLOWS_DIR = '.github/workflows/'
TREE_KIND_VARIABLE = 'TITANOS_TREE_KIND'
MACHINE_NAME_CLASS = 'exporting_machine_name'
DENYLIST_CLASS = 'private_denylist'
# files whose job is to name the dropped records (rules, not references); they are left out of the dangling-reference count,
# as are the workflows (their guarded GEN-01 step names its own tree)
RULE_FILES = {'MANIFEST.in', '.dockerignore', '.gitattributes', '.gitignore', POLICY_PATH, MARKER_PATH, 'tools/build_binary.py',
              'tools/build_public_branch.py', 'tests/check_source_package.py', 'tests/check_public_export_hygiene.py'}
DEFAULT_CHECKS = (['-B', 'tests/check_source_package.py'], ['-B', 'tests/check_public_export_hygiene.py'],
                  ['-B', 'tests/check_release_workflow_gates.py'])


class BuildError(Exception):
    pass


def git(repo, *args, binary=False, check=True, stdin=None):
    proc = subprocess.run(['git', *args], cwd=str(repo), capture_output=True, input=stdin, check=False)
    if check and proc.returncode:
        raise BuildError('git ' + ' '.join(args[:3]) + ' failed: ' + proc.stderr.decode('utf-8', 'replace').strip())
    return proc.stdout if binary else proc.stdout.decode('utf-8')


def tracked_files(repo, ref):
    """{path: blob id} of every file in the source commit (index blobs, not the working tree)."""
    out = git(repo, 'ls-tree', '-r', '-z', '--full-tree', ref, binary=True)
    entries = {}
    for record in out.split(b'\0'):
        if not record:
            continue
        meta, path = record.split(b'\t', 1)
        mode, kind, sha = meta.decode().split(' ')
        if kind != 'blob':
            raise BuildError('unsupported tree entry (submodule?): ' + path.decode('utf-8', 'replace'))
        if mode not in ('100644', '100755'):
            raise BuildError('unsupported file mode ' + mode + ' for ' + path.decode('utf-8', 'replace') + ' (symlinks are not published)')
        entries[path.decode('utf-8')] = sha
    return entries


def read_blobs(repo, ref, paths):
    """{path: bytes} through one `git cat-file --batch` process."""
    request = ''.join(ref + ':' + path + '\n' for path in paths).encode('utf-8')
    out = git(repo, 'cat-file', '--batch', binary=True, stdin=request)
    blobs, position = {}, 0
    for path in paths:
        newline = out.index(b'\n', position)
        header = out[position:newline].decode('utf-8')
        if header.endswith(' missing'):
            raise BuildError('missing blob ' + ref + ':' + path)
        size = int(header.rsplit(' ', 1)[1])
        start = newline + 1
        blobs[path] = out[start:start + size]
        position = start + size + 1
    return blobs


def blob_or_none(repo, ref, path):
    proc = subprocess.run(['git', 'cat-file', 'blob', ref + ':' + path], cwd=str(repo), capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def gitignored(repo, ref, paths):
    """Tracked paths that the source commit's .gitignore would ignore (they were force-added and are not published)."""
    rules = blob_or_none(repo, ref, '.gitignore')
    if not rules or not paths:
        return []
    with tempfile.NamedTemporaryFile('wb', suffix='.gitignore', delete=False) as handle:
        handle.write(rules)
        excludes = handle.name
    try:
        proc = subprocess.run(['git', '-c', 'core.excludesFile=' + excludes, 'check-ignore', '--no-index', '--stdin', '-z'],
                              cwd=str(repo), capture_output=True, input='\0'.join(paths).encode('utf-8'))
    finally:
        os.unlink(excludes)
    if proc.returncode not in (0, 1):
        raise BuildError('git check-ignore failed: ' + proc.stderr.decode('utf-8', 'replace'))
    return sorted(p.decode('utf-8') for p in proc.stdout.split(b'\0') if p)


def load_policy(repo, ref):
    raw = blob_or_none(repo, ref, POLICY_PATH)
    if raw is None:
        raise BuildError(POLICY_PATH + ' is not in ' + ref)
    policy = json.loads(raw.decode('utf-8'))
    excluded = [str(p).replace('\\', '/').strip('/') for p in policy['excluded_paths']]
    if not excluded or '' in excluded:
        raise BuildError(POLICY_PATH + ' must name the excluded repository paths')
    return policy, excluded


def pyproject_version(blobs):
    text = blobs.get('pyproject.toml', b'').decode('utf-8', 'replace')
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else 'unknown'


def select(repo, ref, tracked, policy, excluded, drops):
    """Split the tracked paths into kept and dropped, with the reason for every drop."""
    excluded_lower = [p.casefold() for p in excluded]
    private_names = {str(n).casefold() for n in policy.get('private_names', {})}
    private_files = {str(p).replace('\\', '/').strip('/').casefold() for p in policy.get('private_files', [])}
    remove_list = blob_or_none(repo, ref, REMOVE_LIST)
    listed = {line.strip().replace('\\', '/') for line in (remove_list or b'').decode('utf-8').splitlines()
              if line.strip() and not line.lstrip().startswith('#')}
    ignored = set(gitignored(repo, ref, sorted(tracked)))
    dropped = {'excluded_paths': [], 'private_names': [], 'gitignore': [], 'builder': [], 'remove_list': [], 'option': []}
    kept = []
    for path in sorted(tracked):
        lower = path.casefold()
        if any(lower == p or lower.startswith(p + '/') for p in excluded_lower):
            dropped['excluded_paths'].append(path)
        elif Path(path).name.casefold() in private_names or lower in private_files:
            dropped['private_names'].append(path)
        elif path in ignored:
            dropped['gitignore'].append(path)
        elif lower.startswith(BUILDER_DIR):
            dropped['builder'].append(path)
        elif path in listed or any(path.startswith(entry.rstrip('/') + '/') for entry in listed):
            dropped['remove_list'].append(path)
        elif path in drops or any(path.startswith(entry.rstrip('/') + '/') for entry in drops):
            dropped['option'].append(path)
        else:
            kept.append(path)
    unknown = sorted(d for d in drops if d not in tracked and not any(p.startswith(d.rstrip('/') + '/') for p in tracked))
    if unknown:
        raise BuildError('--drop names paths that are not in ' + ref + ': ' + ', '.join(unknown))
    return kept, dropped


def transform(blobs, kept, policy, excluded, dropped, source_commit, version, overrides):
    """Apply the overrides and write the marker; every other kept file keeps its source bytes (the workflows included: see the
    module docstring). Returns (files {path: bytes}, notes {path: description})."""
    files = {path: blobs[path] for path in kept}
    notes = {}
    for path, content in overrides.items():
        files[path] = content
        notes[path] = 'replaced by ' + OVERRIDES_DIR + path
    absent = sorted(set(excluded) | set(dropped['private_names']) | {str(p).replace('\\', '/').strip('/') for p in policy.get('private_files', [])})
    marker = {
        'schema_version': 1,
        'tree_kind': 'public',
        'builder': 'tools/build_public_branch.py',
        'source_commit': source_commit,
        'source_version': version,
        'absent_records': absent,
        'note': ('This tree was built from the private repository at source_commit without its history. The paths under '
                 'absent_records are lab-private records that stay in the private repository (tools/public_export.json '
                 'excluded_paths, private_names, private_files); links into them do not resolve here by design. The presence '
                 'of this file selects the public tree kind for tree-aware checks (tools/public_export.json tree_kinds): '
                 'tests/check_public_export_hygiene.py then asserts that those records are absent instead of present. The rules '
                 'that exclude them (MANIFEST.in, .dockerignore, tools/public_export.json) stay in place so an accidental return '
                 'stays excluded.'),
    }
    files[MARKER_PATH] = (json.dumps(marker, indent=2) + '\n').encode('utf-8')
    notes[MARKER_PATH] = 'written by the builder'
    return files, notes


def load_overrides(repo, ref, tracked):
    prefix = OVERRIDES_DIR
    return {path[len(prefix):]: blob for path, blob in read_blobs(repo, ref, [p for p in tracked if p.startswith(prefix)]).items()}


def machine_names():
    return {value.strip() for value in (os.environ.get('COMPUTERNAME'), os.environ.get('HOSTNAME'), platform.node())
            if value and len(value.strip()) >= 3 and value.strip().casefold().split('.')[0] != 'localhost'}


def private_denylist():
    """(pattern or None, count) for the names the private host map denies (tools/lab_paths.private_names(): its host keys and deny
    list), whole-word and case-insensitive, built exactly as tests/check_public_export_hygiene.py builds its private_denylist class.
    The map is never tracked (tools/public_export.json never_tracked); without it the class is empty and the generic classes and the
    exporting machine's own name still apply. The names are never printed."""
    names = lab_paths.private_names()
    if not names:
        return None, 0
    pattern = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r')(?![A-Za-z0-9])',
                         re.IGNORECASE | re.ASCII)
    return pattern, len(names)


def scan(files, policy, denylist):
    """[(path, line, class)] over every UTF-8 file, the same rules as tests/check_public_export_hygiene.py: the policy's detection
    classes outside the fixture allowlist, the exporting machine's own name (never tolerated) and the private host map's names
    (private_denylist(), never tolerated). Diagnostics carry file, line and class only. Files that are not UTF-8 are returned in
    `skipped`."""
    patterns = {name: re.compile(pattern) for name, pattern in policy['detection_patterns'].items()}
    allowlist = {entry['path']: set(entry['literals']) for entry in policy.get('fixture_allowlist', [])}
    literals = set().union(*allowlist.values()) if allowlist else set()
    allowlist[POLICY_PATH] = {form for literal in literals for form in (literal, json.dumps(literal)[1:-1])}
    names = machine_names()
    machine = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in sorted(names)) + r')(?![A-Za-z0-9])',
                         re.IGNORECASE | re.ASCII) if names else None
    hits, skipped = [], []
    for path in sorted(files):
        try:
            text = files[path].decode('utf-8')
        except UnicodeDecodeError:
            skipped.append(path)
            continue
        allowed = allowlist.get(path, set())
        for label, pattern in patterns.items():
            for match in pattern.finditer(text):
                if match.group(0) not in allowed:
                    hits.append((path, text.count('\n', 0, match.start()) + 1, label))
        for label, pattern in ((MACHINE_NAME_CLASS, machine), (DENYLIST_CLASS, denylist)):
            if pattern is not None:
                for match in pattern.finditer(text):
                    hits.append((path, text.count('\n', 0, match.start()) + 1, label))
    return hits, skipped


def dangling_references(files, excluded, dropped_paths):
    """{path: line count} of remaining text files that name a dropped record (rule files and the workflows left out): a dropped
    directory is matched as `<dir>/`, a dropped file by its path; under docs/, also by the docs-relative spelling that Markdown links
    use."""
    directories = {e.casefold() for e in excluded if any(p.casefold().startswith(e.casefold() + '/') for p in dropped_paths)}
    needles = {d + '/' for d in directories} | {e.casefold() for e in excluded if e.casefold() not in directories} | {p.casefold() for p in dropped_paths}
    docs_relative = {n[len('docs/'):] for n in needles if n.startswith('docs/')}
    counts = {}
    for path in sorted(files):
        if path in RULE_FILES or path.startswith(WORKFLOWS_DIR):
            continue
        try:
            text = files[path].decode('utf-8')
        except UnicodeDecodeError:
            continue
        local = needles | (docs_relative if path.startswith('docs/') else set())
        count = sum(1 for line in text.casefold().splitlines() if any(needle in line for needle in local))
        if count:
            counts[path] = count
    return counts


def run_checks(worktree, python, checks):
    env = {**os.environ, 'PYTHONPATH': 'src', 'PYTHONDONTWRITEBYTECODE': '1', TREE_KIND_VARIABLE: 'public'}
    results = []
    for check in checks:
        started = time.monotonic()
        proc = subprocess.run([python, *check], cwd=str(worktree), env=env, capture_output=True, text=True)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
        results.append({'command': 'python ' + ' '.join(check), 'returncode': proc.returncode,
                        'seconds': round(time.monotonic() - started, 1), 'tail': tail})
        print(f"check: {results[-1]['command']}: {'pass' if proc.returncode == 0 else 'FAIL (rc ' + str(proc.returncode) + ')'} in {results[-1]['seconds']} s")
        if proc.returncode:
            print('\n'.join('    ' + line for line in tail))
    return results


def check_target(repo, worktree, branch):
    """Refuse early: the worktree path must be new and outside the repository, and the branch must not exist yet."""
    if worktree.exists():
        raise BuildError('refusing to reuse an existing path: ' + str(worktree))
    if worktree.resolve().parent == worktree.resolve() or repo.resolve() in worktree.resolve().parents or worktree.resolve() == repo.resolve():
        raise BuildError('the worktree must be outside the source repository: ' + str(worktree))
    if subprocess.run(['git', 'rev-parse', '--verify', '--quiet', 'refs/heads/' + branch], cwd=str(repo), capture_output=True).returncode == 0:
        raise BuildError(f'branch {branch} already exists; remove it first (git branch -D {branch}) or choose --branch')


def prepare_worktree(repo, worktree, branch, source_commit):
    """An empty worktree on the unborn orphan branch `branch`; returns a short note on how it was made."""
    check_target(repo, worktree, branch)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(['git', 'worktree', 'add', '--orphan', '-b', branch, str(worktree)], cwd=str(repo), capture_output=True)
    if proc.returncode == 0:
        return 'git worktree add --orphan'
    stderr = proc.stderr.decode('utf-8', 'replace').strip()
    if 'orphan' not in stderr or 'unknown option' not in stderr:
        raise BuildError('git worktree add failed: ' + stderr)
    # git before 2.42 has no `worktree add --orphan`: check the source commit out detached, then switch to the unborn orphan branch,
    # which removes every tracked file from the worktree (git 2.23 or newer) and leaves it empty for write_tree()
    git(repo, 'worktree', 'add', '--detach', str(worktree), source_commit)
    git(worktree, 'switch', '--orphan', branch)
    leftover = sorted(p.relative_to(worktree).as_posix() for p in worktree.rglob('*') if p.is_file() and p.relative_to(worktree).parts[0] != '.git')
    if leftover:
        raise BuildError('the orphan worktree is not empty after the fallback: ' + ', '.join(leftover[:10]))
    return 'git worktree add --detach, then git switch --orphan (git before 2.42)'


def write_tree(worktree, files):
    for path, content in files.items():
        target = worktree / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def stage_tree(worktree):
    """Stage every written file before the checks run: `git ls-files` in the worktree then lists exactly the public tree, so a
    tree-aware check (tests/check_public_export_hygiene.py reads the tracked set) scans the tree and not the worktree's own `.git`
    pointer file, whose gitdir line names the private repository's location. Exact bytes: no line-ending conversion on the way in."""
    git(worktree, '-c', 'core.autocrlf=false', '-c', 'core.safecrlf=false', 'add', '-A', '.')


def commit_tree(worktree, message, identity):
    config = []
    if identity:
        match = re.fullmatch(r'\s*(.+?)\s*<([^<>]+)>\s*', identity)
        if not match:
            raise BuildError('--identity must look like "Name <email>"')
        config = ['-c', 'user.name=' + match.group(1), '-c', 'user.email=' + match.group(2)]
    git(worktree, *config, '-c', 'core.autocrlf=false', '-c', 'core.safecrlf=false', 'commit', '-q', '-m', message)
    return git(worktree, 'rev-parse', 'HEAD').strip()


def verify_commit(worktree, files, source_blobs, notes):
    if git(worktree, 'rev-list', '--count', 'HEAD').strip() != '1':
        raise BuildError('the public branch must have exactly one commit')
    if git(worktree, 'status', '--porcelain').strip():
        raise BuildError('the worktree is not clean after the commit')
    committed = tracked_files(worktree, 'HEAD')
    if set(committed) != set(files):
        raise BuildError('committed paths differ from the built tree: ' + repr(sorted(set(committed) ^ set(files))[:20]))
    drifted = sorted(path for path, sha in committed.items() if path not in notes and source_blobs.get(path) != sha)
    if drifted:
        raise BuildError('unchanged files must keep their source blob ids; drifted: ' + repr(drifted[:20]))
    return len(committed) - len(notes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repo', type=Path, default=ROOT, help='the private repository (default: the one containing this script)')
    parser.add_argument('--source-ref', default='HEAD', help='commit to publish (default HEAD)')
    parser.add_argument('--worktree', type=Path, help='new directory outside the repository for the orphan branch (for example ../titanos-public)')
    parser.add_argument('--branch', default='public', help='orphan branch name (default public); must not exist yet')
    parser.add_argument('--drop', action='append', default=[], metavar='PATH', help='additional tracked path to leave out (repeatable; recorded in the report)')
    parser.add_argument('--identity', help='"Name <email>" for the root commit (default: the repository git identity)')
    parser.add_argument('--python', default=sys.executable, help='interpreter for the checks run inside the worktree')
    parser.add_argument('--no-checks', action='store_true', help='skip the release gates inside the worktree')
    parser.add_argument('--dry-run', action='store_true', help='compute and print the report; create nothing')
    parser.add_argument('--report', type=Path, help='write the JSON report to this file (outside the repository)')
    args = parser.parse_args(argv)
    if not args.dry_run and args.worktree is None:
        parser.error('--worktree is required unless --dry-run')

    repo = args.repo.resolve()
    ref = args.source_ref
    source_commit = git(repo, 'rev-parse', '--verify', ref + '^{commit}').strip()
    if not args.dry_run:
        check_target(repo, args.worktree.resolve(), args.branch)
    tracked = tracked_files(repo, ref)
    policy, excluded = load_policy(repo, ref)
    drops = [d.replace('\\', '/').strip('/') for d in args.drop]
    kept, dropped = select(repo, ref, tracked, policy, excluded, drops)
    blobs = read_blobs(repo, ref, kept)
    version = pyproject_version(blobs)
    overrides = load_overrides(repo, ref, tracked)
    files, notes = transform(blobs, kept, policy, excluded, dropped, source_commit, version, overrides)
    denylist, denied = private_denylist()
    hits, skipped = scan(files, policy, denylist)
    dropped_paths = [p for group in dropped.values() for p in group]
    dangling = dangling_references(files, excluded, dropped_paths)
    for path in list(files):
        if any(path.casefold() == e.casefold() or path.casefold().startswith(e.casefold() + '/') for e in excluded):
            raise BuildError('an excluded path survived selection: ' + path)

    report = {
        'source_ref': ref, 'source_commit': source_commit, 'source_version': version, 'tracked_files': len(tracked),
        'kept_files': len(kept), 'written_files': len(files), 'dropped': dropped, 'dropped_total': len(dropped_paths),
        'changed_or_new': notes, 'overrides_applied': sorted(overrides),
        'scan': {'hits': [{'file': f, 'line': l, 'class': c} for f, l, c in hits], 'skipped_not_utf8': skipped,
                 'classes': sorted(policy['detection_patterns']) + [MACHINE_NAME_CLASS] + ([DENYLIST_CLASS] if denylist else []),
                 # how many names the private host map denies (tools/lab_paths.private_names()); the names themselves are never written
                 'private_denylist_names': denied},
        'dangling_references': dangling, 'dry_run': args.dry_run, 'worktree': None, 'branch': args.branch, 'commit': None, 'checks': [],
    }
    print(f'source {ref} = {source_commit[:12]} (Titanos {version}): {len(tracked)} tracked files, {len(dropped_paths)} dropped, {len(files)} in the public tree')
    for reason, paths in dropped.items():
        if paths:
            print(f'  dropped ({reason}): {len(paths)}: ' + ', '.join(paths[:6]) + (' ...' if len(paths) > 6 else ''))
    for path, note in notes.items():
        print(f'  changed or new: {path}: {note}')
    print(f"  scan ({', '.join(report['scan']['classes'])}; {denied} private-map name(s)): {len(hits)} hit(s)"
          + (f", {len(skipped)} file(s) not UTF-8 skipped: {', '.join(skipped)}" if skipped else ''))
    for path, line, kind in hits[:40]:
        print(f'    {path}:{line} {kind}')
    print(f'  references into dropped records remaining in {len(dangling)} file(s) (dangle by design): '
          + ', '.join(f'{p} ({n})' for p, n in sorted(dangling.items(), key=lambda item: -item[1])[:12]) + (' ...' if len(dangling) > 12 else ''))

    def finish(code):
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print('report written to', args.report)
        return code

    if hits:
        print('ABORT: the public tree carries lab-private references (file, line, class above); nothing was created')
        return finish(1)
    if args.dry_run:
        print('dry run: nothing created')
        return finish(0)

    worktree = args.worktree.resolve()
    report['worktree_made_by'] = prepare_worktree(repo, worktree, args.branch, source_commit)
    report['worktree'] = str(worktree)
    write_tree(worktree, files)
    stage_tree(worktree)
    print(f'{len(files)} files written and staged in {worktree} (branch {args.branch}, unborn; {report["worktree_made_by"]})')
    if not args.no_checks:
        report['checks'] = run_checks(worktree, args.python, [list(c) for c in DEFAULT_CHECKS])
        if any(result['returncode'] for result in report['checks']):
            print(f'checks failed: no commit was made; the worktree is kept for inspection (files staged, nothing committed). Remove it with:\n'
                  f'  git -C "{repo}" worktree remove --force "{worktree}" && git -C "{repo}" branch -D {args.branch}')
            return finish(1)
    message = (f'Titanos {version}: public source tree\n\nBuilt by tools/build_public_branch.py from private commit {source_commit} '
               f'without its history. {len(dropped_paths)} lab-private paths named by tools/public_export.json are not part of this tree '
               f'(see {MARKER_PATH}).')
    report['commit'] = commit_tree(worktree, message, args.identity)
    unchanged = verify_commit(worktree, files, tracked, notes)
    print(f'committed {report["commit"][:12]} on branch {args.branch}: {len(files)} files, {unchanged} byte-identical to {ref}, {len(notes)} changed or new')
    print('next steps are manual and never automatic:\n'
          f'  inspect:  git -C "{worktree}" log --stat\n'
          f'  publish:  git -C "{worktree}" push <empty public repository url> {args.branch}:main\n'
          f'  discard:  git -C "{repo}" worktree remove --force "{worktree}" && git -C "{repo}" branch -D {args.branch}')
    return finish(0)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BuildError as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(2)
