"""File links through the cluster facade: X-Forwarded-Host and X-Forwarded-Proto, or a relative path.

Image, video, speech and music answers link to /v1/files/<name>. A client of the cluster facade can't reach the
engine's loopback address, so a loopback peer (the facade beside the engine) that sends X-Forwarded-Host gets links on
that host and X-Forwarded-Proto's scheme, and a relative path when the host is malformed. Without the header, and from
any other peer, links name the address the client asked, as before. Covered at the helper (server.link_base) and
end to end through a speech job whose stand-in runtime returns a WAV. No model, GPU or network.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
HOME = tempfile.TemporaryDirectory(prefix='titanos-forwarded-links-home-', ignore_cleanup_errors=True)
os.environ['TITANOS_HOME'] = HOME.name  # nothing from this check reaches ~/.titanos

from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402
from titanos import server  # noqa: E402
from titanos.config import Config, ModelSpec  # noqa: E402
from titanos.settings import Settings  # noqa: E402

WAV = (b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00'
       b'data\x00\x00\x00\x00')


def base(headers=(), peer='127.0.0.1', root_path=''):
    scope = {'type': 'http', 'method': 'GET', 'scheme': 'http', 'server': ('127.0.0.1', 11435), 'path': '/x',
             'root_path': root_path, 'query_string': b'', 'client': (peer, 50000) if peer else None,
             'headers': [(name.lower().encode('latin-1'), value.encode('latin-1')) for name, value in headers]}
    return server.link_base(Request(scope))


def check_helper():
    own = 'http://127.0.0.1:11435/'
    assert base() == own
    assert base([('X-Forwarded-Host', 'pair.local:9443'), ('X-Forwarded-Proto', 'https')]) == 'https://pair.local:9443/'
    assert base([('X-Forwarded-Host', '198.51.100.20:8080')]) == 'http://198.51.100.20:8080/'  # no proto: this scheme
    assert base([('X-Forwarded-Host', '[fd00::5]:9443'), ('X-Forwarded-Proto', 'HTTPS')]) == 'https://[fd00::5]:9443/'
    assert base([('X-Forwarded-Host', 'a.example, b.example'), ('X-Forwarded-Proto', 'https, http')]) == \
        'https://a.example/'  # a proxy chain: the first (client-facing) value
    assert base([('X-Forwarded-Host', 'pair.local'), ('X-Forwarded-Proto', 'gopher')]) == 'http://pair.local/'
    assert base([('X-Forwarded-Host', 'pair.local')], peer='::1') == 'http://pair.local/'
    assert base([('X-Forwarded-Host', 'pair.local')], root_path='/engine') == 'http://pair.local/engine/'
    for bad in ('evil.example/path', 'user@evil.example', 'a b', '', '-x.example', 'host:99999x', '[zz]'):
        assert base([('X-Forwarded-Host', bad)]) == '/', bad  # malformed: relative, never the loopback address
    assert base([('X-Forwarded-Host', 'evil.example/p')], root_path='/engine') == '/engine/'
    for peer in ('192.0.2.10', 'testclient', None):
        assert base([('X-Forwarded-Host', 'pair.local')], peer=peer) == own, peer  # only the facade beside it


class Runtime:
    device = 'cpu'

    def infer(self, task, data, emit=None, cancel=None):
        assert task == 'speech', task
        return WAV


def check_speech_job():
    with tempfile.TemporaryDirectory(prefix='titanos-forwarded-links-') as scratch:
        config = Config(models={'voice': ModelSpec(path=scratch, task='speech', device='cpu')})
        app = server.create_app(config, output_dir=Path(scratch) / 'out', settings=Settings(), runtime_catalog=False)
        engine = app.state.engine.target
        engine.runtime, engine.current, engine.states['voice'] = Runtime(), 'voice', 'loaded'
        with TestClient(app, base_url='http://127.0.0.1:11435', client=('127.0.0.1', 50000)) as client:
            def job(headers):
                accepted = client.post('/api/jobs', json={'task': 'speech', 'model': 'voice', 'input': 'Hi.'},
                                       headers=headers)
                assert accepted.status_code == 202, accepted.text[:300]
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    record = client.get('/api/jobs/' + accepted.json()['id']).json()
                    if record['status'] not in ('queued', 'running'):
                        assert record['status'] == 'completed', record
                        return record['result']['url']
                    time.sleep(0.05)
                raise AssertionError('the speech job did not finish')

            url = job({'X-Forwarded-Host': 'pair.local:9443', 'X-Forwarded-Proto': 'https'})
            assert url.startswith('https://pair.local:9443/v1/files/') and url.endswith('.wav'), url
            relative = job({'X-Forwarded-Host': 'not a host'})
            assert relative.startswith('/v1/files/'), relative
            direct = job({})
            assert direct.startswith('http://127.0.0.1:11435/v1/files/'), direct
            for link in (url, relative, direct):
                served = client.get('/v1/files/' + link.rsplit('/', 1)[-1])
                assert served.status_code == 200 and served.content == WAV, served.status_code


def main():
    check_helper()
    check_speech_job()
    print('check_forwarded_links: ok')


if __name__ == '__main__':
    main()
