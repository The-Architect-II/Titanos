"""Two real GPU worker processes launched by a self-contained Titanos executable."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import httpx

from titanos.config import Config, ModelSpec, WorkerSpec, write_config
from titanos.devices import inventory, pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpus', nargs=2, default=['0', '1'])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    executable = args.executable.resolve()
    devices = [item for item in inventory(True)['devices'] if item['present']]
    selected = pin(args.gpus, devices)
    result = {'passed': False, 'executable': str(executable), 'gpus': selected,
              'scope': 'Two real local GPU replicas from the frozen executable; no sharding or tensor parallelism claim'}
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='titanos-frozen-workers-') as temp:
            root = Path(temp)
            cfg = Config(models={'chat': ModelSpec(path=str(args.model.resolve()), task='chat')},
                         workers={name: WorkerSpec(models=['chat'], gpus=[gpu], threads=4, profile=True)
                                  for name, gpu in zip(['a', 'b'], selected)})
            write_config(root / 'models.json', cfg)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            env = {k: v for k, v in os.environ.items() if k not in {'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'TITANOS_API_KEY'}}
            env.update(TITANOS_HOME=str(root), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            system = Path(env.get('SystemRoot', 'C:/Windows'))
            env['PATH'] = os.pathsep.join(map(str, [system / 'System32', system, system / 'System32' / 'Wbem']))
            with (root / 'server.log').open('w+', encoding='utf-8') as log:
                process = subprocess.Popen([str(executable), 'serve', '--port', str(port)], cwd=root, env=env, stdout=log, stderr=log)
                try:
                    with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=180) as client:
                        deadline = time.monotonic() + 30
                        while True:
                            try:
                                if client.get('/healthz').status_code == 200:
                                    break
                            except httpx.HTTPError:
                                pass
                            assert process.poll() is None and time.monotonic() < deadline, 'Frozen server did not start'
                            time.sleep(0.05)
                        loaded = client.post('/api/models/load', json={'model': 'chat'})
                        assert loaded.status_code == 200, loaded.text
                        body = {'model': 'chat', 'messages': [{'role': 'user', 'content': 'Write ten detailed sentences about exploring a forest.'}],
                                'temperature': 0, 'max_tokens': 128}
                        overlap = False
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            calls = [pool.submit(client.post, '/v1/chat/completions', json={**body, 'worker': name}) for name in ['a', 'b']]
                            while not all(call.done() for call in calls):
                                rows = client.get('/api/workers').json()['data']
                                overlap |= sum(bool(row['busy']) for row in rows) == 2
                                time.sleep(0.01)
                            replies = [call.result() for call in calls]
                        assert all(reply.status_code == 200 for reply in replies), [reply.text for reply in replies]
                        rows = client.get('/api/workers').json()['data']
                        assert len({row['pid'] for row in rows}) == 2, rows
                        assert overlap, 'Requests did not overlap in separate workers'
                        for row in rows:
                            assert row['current_model'] == 'chat' and row['completed_requests'] == 1, row
                            assert row['observed_gpu_indices'] == [0], row
                            assert row['verified_devices'][0]['uuid'] == row['assignment']['gpus'][0], row
                        result.update(passed=True, overlap=overlap, workers=rows,
                                      replies=[reply.json()['choices'][0]['message']['content'] for reply in replies])
                except Exception:
                    log.flush()
                    log.seek(0)
                    result['server_log_tail'] = log.read()[-10000:]
                    raise
                finally:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    process.wait(timeout=30)
            for attempt in range(60):
                try:
                    (root / 'server.log').unlink()
                    break
                except PermissionError:
                    if attempt == 59:
                        raise
                    time.sleep(0.05)
    except Exception as exc:
        result.update(passed=False, error=str(exc))
        raise
    finally:
        result['seconds'] = round(time.monotonic() - started, 2)
        args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
