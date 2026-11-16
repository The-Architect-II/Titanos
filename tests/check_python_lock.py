"""The pinned-interpreter install's inputs: tools/python-lock.json and the committed locks (tools/build_python_lock.py).

Offline. Each lock parses under the grammar an installer that runs pip with --require-hashes --no-deps on it applies
(name==version and one or more --hash=sha256:<64 hex> per line; no marker, URL, path, @ reference, extras, ===, option or
repeated name; continuations and comments as pip reads them), and its versions satisfy pyproject.toml's dependencies,
the listed extras and the source wheels' requirements for that platform, with the pins exact; titanos-engine, the
source wheels and the leave_out names are not in it. The pins file names an https CPython build per platform by sha256
and size with a relative python_bin, one to eight https indexes without credentials, query or fragment, and each
source wheel at a full commit.
"""
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import build_python_lock as B  # noqa: E402

DIGEST = '0' * 64


def refused(text, words):
    try:
        B.parse_lock(text)
    except ValueError as exc:
        assert words in str(exc), (text, str(exc))
        return
    raise AssertionError(f'accepted {text!r}')


def check_grammar():
    good = (f'\ufeff# comment\nalpha==1.0 --hash=sha256:{DIGEST} \\\n    --hash=sha256:{"A" * 64}\n'
            f'beta==2.0+cu128 --hash=sha256:{DIGEST}  # trailing comment\n\n')
    assert B.parse_lock(good) == [('alpha', '1.0', [DIGEST, 'a' * 64]), ('beta', '2.0+cu128', [DIGEST])]
    line = f' --hash=sha256:{DIGEST}'
    refused('-e .' + line, 'is an option')
    refused('--index-url https://example.invalid/simple', 'is an option')
    refused("alpha==1.0;sys_platform=='win32'" + line, 'environment marker')
    refused('alpha==1.0 ; python_version>"3"', 'environment marker')
    refused('https://example.invalid/a.whl' + line, 'is a URL')
    refused('alpha@1.0' + line, 'direct reference')
    refused('./alpha.whl' + line, 'is a path')
    refused('alpha[extra]==1.0' + line, 'names extras')
    refused('alpha===1.0' + line, 'pins with ===')
    refused('alpha>=1.0' + line, 'is not name==version')
    refused('alpha==1.0', 'has no --hash')
    refused('alpha==1.0 --hash=sha256:abc', 'is not --hash=sha256')
    refused('alpha==1.0 --hash=md5:' + DIGEST[:32], 'is not --hash=sha256')
    refused(f'Alpha_One==1.0{line}\nalpha-one==2.0{line}', 'is named again (first on line 1)')


def check_pins():
    pins = B.load_pins()
    assert pins['schema_version'] == 1 and re.fullmatch(r'3\.\d+', pins['python']), pins['python']
    indexes = pins['indexes']
    assert 1 <= len(indexes) <= 8 and len(set(indexes)) == len(indexes), indexes
    for index in indexes:
        parts = urllib.parse.urlsplit(index)
        assert parts.scheme == 'https' and parts.hostname and not (parts.username or parts.password or parts.query
                                                                    or parts.fragment), index
    for name, source in pins['source_wheels'].items():
        assert source['url'].startswith('https://') and re.fullmatch(r'[0-9a-f]{40}', source['commit']), (name, source)
    assert set(pins['platforms']) == set(B.ENVIRONMENTS), pins['platforms']
    for platform, row in pins['platforms'].items():
        python = row['interpreter']
        assert python['url'].startswith('https://') and re.fullmatch(r'[0-9a-f]{64}', python['sha256']), platform
        assert isinstance(python['size'], int) and python['size'] > 0, platform
        parts = python['python_bin'].split('/')
        assert '\\' not in python['python_bin'] and not python['python_bin'].startswith('/') \
            and '.' not in parts and '..' not in parts, python['python_bin']


def check_locks():
    problems = B.check()
    assert not problems, '\n'.join(problems)
    for platform in B.load_pins()['platforms']:
        data = B.lock_path(platform).read_bytes()
        assert b'\r' not in data, f'{platform}: the lock has CR bytes (it is a release asset; .gitattributes keeps LF)'


def main():
    check_grammar()
    check_pins()
    check_locks()
    print('check_python_lock: ok')


if __name__ == '__main__':
    main()
