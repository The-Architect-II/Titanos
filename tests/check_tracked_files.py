"""Tracked-file guard: what the repository tracks stays source, never weights, audio, archives, credentials or secrets.

Over `git ls-files` (a checkout) or the files under the root (a source copy without .git), it fails on a file larger than
1 MiB outside LARGE_FILES, on a weight, audio, array, archive or native-binary extension, on a credential file name, and on
a secret-shaped string in a text file: Hugging Face, NVIDIA, OpenAI and Anthropic, AWS, Google, GitHub and Slack token
shapes and PEM private keys, in UTF-8 or UTF-16 text. Findings name the file, line and rule, never the matched text.
SECRET_SENTINELS lists the deliberate test sentinels by file. A self-test first proves every rule fires on synthetic
inputs built at run time (so this file holds no secret-shaped literal). No model, process, GPU or network; git is run
read-only (`git ls-files`)."""
import codecs
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1024 * 1024
# generated or vendored files over MAX_BYTES, each with the check or catalog that binds its bytes
LARGE_FILES = frozenset({'docs/execution_matrix.html', 'docs/execution_matrix.json',  # tools/render_execution_matrix.py --check
                         'src/titanos/web/swagger-ui-bundle.js'})  # tools/licenses/vendored_assets.json
BLOCKED_SUFFIXES = frozenset({
    '.safetensors', '.gguf', '.onnx', '.onnx_data', '.pt', '.pth', '.ckpt', '.bin', '.npy', '.npz', '.pkl', '.pickle',
    '.h5', '.msgpack', '.tflite', '.dlc', '.engine', '.plan', '.pdiparams', '.pdmodel', '.traineddata',
    '.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus', '.mp4',
    '.exe', '.dll', '.pyd', '.so', '.dylib', '.msi', '.whl', '.zip', '.7z', '.tar', '.tgz', '.gz', '.xz', '.bz2',
    '.key', '.pem', '.p12', '.pfx'})
BLOCKED_NAMES = frozenset({'keys.json', 'token', 'token.json', 'tokens.json', 'credentials.json', 'secrets.json',
                           'id_rsa', 'id_ed25519', '.env', '.netrc', '.pypirc'})
ALLOWED_NAMES = frozenset({'.env.example'})
SECRET_RULES = {
    'huggingface': r'\bhf_[A-Za-z0-9]{30,}',
    'nvapi': r'\bnvapi-[A-Za-z0-9_-]{40,}',
    'openai_anthropic': r'\bsk-(?:proj-|ant-(?:api|admin)\d\d-)?[A-Za-z0-9_-]{32,}',
    'aws_access_key': r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b',
    'google_api_key': r'\bAIza[0-9A-Za-z_-]{35}\b',
    'github': r'\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})',
    'slack': r'\bxox[abposr]-[A-Za-z0-9-]{10,}',
    'private_key': r'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----',
}
PATTERNS = {name: re.compile(rule) for name, rule in SECRET_RULES.items()}
# deliberate, non-secret sentinels the test suite needs, by file
SECRET_SENTINELS = {'tests/check_broker_ui.js': frozenset({'sk-' + 'test-wp8-studio-sentinel-0123456789'})}  # split: no literal
SKIPPED_DIRS = frozenset({'.git', '.venv', 'venv', 'env', 'build', 'dist', 'release', 'node_modules', '__pycache__'})


def tracked(root):
    """Repository-relative POSIX paths: git's index in a checkout, else every file under root outside SKIPPED_DIRS."""
    try:
        listed = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True, check=True, timeout=60)
        names = [name for name in listed.stdout.decode('utf-8').split('\0') if name]
        if names:
            return sorted(name for name in names if (root / name).is_file())
    except (OSError, subprocess.SubprocessError):
        pass
    return sorted(path.relative_to(root).as_posix() for path in root.rglob('*')
                  if path.is_file() and not SKIPPED_DIRS.intersection(path.relative_to(root).parts[:-1])
                  and not any(part.endswith('.egg-info') for part in path.relative_to(root).parts))


def decoded(data):
    """Text of a file, or None for binary data. UTF-16 is recognised by its BOM or by NUL bytes in alternate positions."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode('utf-16', errors='replace')
    sample = data[:4096]
    if b'\0' in sample:
        if sample[1::2].count(0) > len(sample) // 3:
            return data.decode('utf-16-le', errors='replace')
        return None
    return data.decode('utf-8', errors='replace')


def findings(root, names):
    problems = []
    for name in names:
        path, pure = root / name, PurePosixPath(name)
        size = path.stat().st_size
        if size > MAX_BYTES and name not in LARGE_FILES:
            problems.append((name, 0, f'larger than {MAX_BYTES} bytes'))
        lowered = pure.name.lower()
        if lowered not in ALLOWED_NAMES and (lowered in BLOCKED_NAMES or lowered.startswith('.env.')
                                            or any(lowered.endswith(suffix) for suffix in BLOCKED_SUFFIXES)):
            problems.append((name, 0, 'weight, audio, archive, binary or credential file'))
        text = decoded(path.read_bytes())
        if text is None:
            continue
        allowed = SECRET_SENTINELS.get(name, frozenset())
        for number, line in enumerate(text.splitlines(), 1):
            for rule, pattern in PATTERNS.items():
                if any(match.group(0) not in allowed for match in pattern.finditer(line)):
                    problems.append((name, number, 'secret-shaped string: ' + rule))
    return problems


def self_test():
    """Every rule fires on a synthetic tree, and allowed files and sentinels stay quiet."""
    samples = {'huggingface': 'hf_' + 'Ab1' * 11, 'nvapi': 'nvapi-' + 'x9_' * 15, 'openai_anthropic': 'sk-proj-' + 'Q7z' * 12,
               'aws_access_key': 'AKIA' + 'Z7' * 8, 'google_api_key': 'AIza' + 'k' * 35, 'github': 'ghp_' + 'a1' * 18,
               'slack': 'xoxb-' + '12345-abcde', 'private_key': '-----BEGIN ' + 'OPENSSH PRIVATE KEY-----'}
    assert set(samples) == set(SECRET_RULES)
    with tempfile.TemporaryDirectory(prefix='titanos-tracked-') as folder:
        root = Path(folder)
        for rule, value in samples.items():
            (root / f'{rule}.txt').write_text(f'line one\nvalue = "{value}"\n', encoding='utf-8')
        (root / 'wide.txt').write_bytes(('token: ' + samples['huggingface']).encode('utf-16'))
        (root / 'big.md').write_bytes(b'a' * (MAX_BYTES + 1))
        for name in ('weights.safetensors', 'voice.flac', 'keys.json', '.env.local', 'cache.npz'):
            (root / name).write_bytes(b'\x00\x01')
        (root / '.env.example').write_text('OPENAI_API_KEY=\n', encoding='utf-8')
        (root / 'clean.py').write_text('# sk-short and hf_ alone are not tokens\n', encoding='utf-8')
        found = findings(root, sorted(path.name for path in root.iterdir()))
        by_file = {}
        for name, number, rule in found:
            by_file.setdefault(name, set()).add((number, rule))
        for rule in samples:
            assert by_file.get(f'{rule}.txt') == {(2, 'secret-shaped string: ' + rule)}, (rule, by_file.get(f'{rule}.txt'))
        assert by_file['wide.txt'] == {(1, 'secret-shaped string: huggingface')}, by_file['wide.txt']
        assert by_file['big.md'] == {(0, f'larger than {MAX_BYTES} bytes')}
        for name in ('weights.safetensors', 'voice.flac', 'keys.json', '.env.local', 'cache.npz'):
            assert by_file[name] == {(0, 'weight, audio, archive, binary or credential file')}, name
        assert '.env.example' not in by_file and 'clean.py' not in by_file, by_file
    return len(samples)


def main():
    rules = self_test()
    names = tracked(ROOT)
    problems = findings(ROOT, names)
    for name, number, rule in problems:
        print(f'{name}:{number}: {rule}', file=sys.stderr)
    stale = sorted(name for name in LARGE_FILES | set(SECRET_SENTINELS) if name not in names)
    if stale:
        print('allowlisted files that are no longer tracked: ' + ', '.join(stale), file=sys.stderr)
    if problems or stale:
        raise SystemExit(1)
    print(json.dumps({'check': 'tracked_files', 'result': 'pass', 'files': len(names), 'secret_rules': rules,
                      'scope': 'tracked file sizes, extensions, names and secret-shaped text; not a full secret scanner'}))


if __name__ == '__main__':
    main()
