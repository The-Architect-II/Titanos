"""Daybreak blocker 2 (active monitoring) on Windows (prints SKIP and exits 0 elsewhere). Independent proof, never Titanos itself:
a fake `titanos` package in the cell work dir answers the kit's HTTP calls under `python -m titanos`.
  1. guarded_call refreshes the readings at most ~2 s apart on the real clock, and a STOP that appears while the call is blocked
     surfaces within one poll interval.
  2. wait_admission: STOP written by the third (admitting) reading is seen immediately before success -> not admitted, 'stop'.
  3. serve_and_exercise, contained, task music: the fake server writes STOP while the music job is 'running'; the trip is
     'stopped_during_request' within one guard interval of the STOP write, the kit cancels exactly the owned job (DELETE
     /api/jobs/<id> reaches the fake server), the owned tree alone is reaped, and the attempt is untried with every sample retained.
  4. serve_and_exercise, artifact floor: the fake server writes STOP and never answers the model load; the trip is
     'stopped_during_model_load' within one guard interval, the server is terminated and the record is untried with samples."""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'matrix_kit'))
import containment  # noqa: E402
import run_cells  # noqa: E402

if os.name != 'nt' or not containment.available():
    print(json.dumps({'skipped': True, 'reason': 'Windows Job Object containment is not available on this host', 'os': os.name}))
    sys.exit(0)

MIB = 1 << 20
failures, observations = [], {}


def expect(cond, msg):
    if not cond:
        failures.append(msg)


# 1. guarded_call cadence on the real clock, and a mid-call STOP surfaces within one poll interval
with tempfile.TemporaryDirectory() as tmp:
    reads = []
    guard = containment.CellGuard(None, None, None, None, None, None, None, commit_reader=lambda: reads.append(time.monotonic()) or (100 << 30), gpu_free_reader=lambda u: None)
    containment.guarded_call(guard, 'probe', lambda: time.sleep(6.5))
    gaps = [b - a for a, b in zip(reads, reads[1:])]
    expect(len(reads) >= 4 and max(gaps) <= 2.1, f'readings refreshed at most ~2 s apart while the call blocked: {len(reads)} reads, gaps {[round(g, 3) for g in gaps]}')
    observations['guarded_call_gaps_s'] = [round(g, 3) for g in gaps]
    stop = Path(tmp) / 'STOP'
    guard2 = containment.CellGuard(str(stop), None, None, None, None, None, None, commit_reader=lambda: 100 << 30, gpu_free_reader=lambda u: None)
    written = {}

    def write_stop():
        stop.write_text('stop', encoding='utf-8')
        written['t'] = time.monotonic()
    threading.Timer(1.0, write_stop).start()
    try:
        containment.guarded_call(guard2, 'model_load', lambda: time.sleep(30))
        expect(False, 'a STOP during the blocked call must trip')
    except containment.GuardTrip as exc:
        latency = time.monotonic() - written['t']
        expect(exc.kind == 'stopped' and exc.phase == 'model_load' and latency <= 2.1, f'STOP surfaced within one poll interval: {exc} after {latency:.2f}s')
        observations['guarded_call_stop_latency_s'] = round(latency, 3)

# 2. wait_admission: STOP appearing with the third good reading is caught immediately before success
with tempfile.TemporaryDirectory() as tmp:
    stop = Path(tmp) / 'STOP'
    n = []

    def late_stop():
        n.append(1)
        if len(n) == 3:
            stop.write_text('stop', encoding='utf-8')
        return 50 << 30
    r = run_cells.wait_admission({'commit': 1 << 30, 'gpu': None}, None, str(stop), lambda m: None, commit_reader=late_stop, gpu_reader=lambda u: None, sleep=lambda s: None)
    expect(r['admitted'] is False and r['reason'] == 'stop' and len(n) == 3, f'success is never returned after STOP appeared: {r}')
    r = run_cells.wait_admission({'commit': 1 << 30, 'gpu': None}, None, str(stop), lambda m: None, commit_reader=lambda: 50 << 30, gpu_reader=lambda u: None, sleep=lambda s: None)
    expect(r['admitted'] is False and r['reason'] == 'stop' and r['samples'] == [], f'STOP at the top of the first iteration: {r}')

# 3./4. a fake Titanos server: healthz, load (optionally hanging after writing STOP), a music job that writes STOP while running, cancel log
FAKE_SERVER = '''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
port = int(sys.argv[sys.argv.index('--port') + 1])
STOP, HANG_LOAD, CANCEL_LOG = os.environ.get('KIT_TEST_STOP'), os.environ.get('KIT_TEST_HANG_LOAD'), os.environ.get('KIT_TEST_CANCEL_LOG')


def write_stop():
    Path(STOP).write_text(json.dumps({'written_at': time.time()}), encoding='utf-8')


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def drain(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n:
            self.rfile.read(n)

    def do_GET(self):
        if self.path == '/healthz':
            return self.reply(200, {'ok': True})
        if self.path.startswith('/api/jobs/'):
            if not Path(STOP).exists():
                write_stop()
            return self.reply(200, {'id': 'job-1', 'status': 'running'})
        if self.path == '/api/models/loaded':
            return self.reply(200, {'data': [{'id': 'music'}]})
        self.reply(404, {})

    def do_POST(self):
        self.drain()
        if self.path == '/api/models/load':
            if HANG_LOAD:
                write_stop()
                time.sleep(600)
            return self.reply(200, {'ok': True})
        if self.path == '/v1/music/generations':
            return self.reply(202, {'id': 'job-1'})
        self.reply(200, {})

    def do_DELETE(self):
        with Path(CANCEL_LOG).open('a', encoding='utf-8') as f:
            f.write(f'DELETE {self.path} {time.time()}\\n')
        self.reply(200, {'id': 'job-1', 'status': 'cancelled'})


ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
'''
cpu_slot = {'slot': 'win-x64-cpu', 'device': 'cpu', 'device_named': 'cpu', 'runtime': 'cpu'}


def fake_cell(tmp, name, hang_load):
    root = Path(tmp) / name
    (root / 'titanos').mkdir(parents=True)
    (root / 'titanos' / '__init__.py').write_text('', encoding='utf-8')
    (root / 'titanos' / '__main__.py').write_text(FAKE_SERVER, encoding='utf-8')
    stop, cancel_log = Path(tmp) / f'{name}.STOP', Path(tmp) / f'{name}.cancel.log'
    entry = {'model_id': 'fake', 'task': 'chat', 'backend': 'transformers', 'path': str(ROOT / 'tests'), 'spec': {},
             'env': {'KIT_TEST_STOP': str(stop), 'KIT_TEST_CANCEL_LOG': str(cancel_log), **({'KIT_TEST_HANG_LOAD': '1'} if hang_load else {})}}
    cfg, _ = run_cells.build_config(entry, cpu_slot, 'single', None, 'auto')
    entry['task'] = 'music'  # the fake server ignores the config; the kit's request path is the music polling loop
    return entry, cfg, root, stop, cancel_log


def stop_to_trip_seconds(stop, rec):
    written = json.loads(stop.read_text(encoding='utf-8'))['written_at']
    tripped = datetime.fromisoformat(rec['resource_qualification']['trip']['utc']).timestamp()
    return tripped - written


with tempfile.TemporaryDirectory() as tmp:
    # 3. contained, STOP while the music job is running
    entry, cfg, root, stop, cancel_log = fake_cell(tmp, 'music', hang_load=False)
    resource = {'kind': 'contained_resident', 'reason': 'first run', 'floors': {'commit': 512 * MIB + 1, 'gpu': None}, 'margin_bytes': 2 << 30, 'admission_samples': [],
                'job_limit': 512 * MIB, 'job_limit_source': 'first_run_containment_cap', 'reserve_bytes': 1, 'profile_key': hashlib.sha256(b'blocker2-profile').hexdigest()}
    rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', None, 30, stop_file=str(stop), resource=resource)
    rq = rec['resource_qualification']
    latency = stop_to_trip_seconds(stop, rec) if stop.exists() and rq.get('trip') else None
    expect(rec['state'] == 'untried' and rec['error'].startswith('stopped_during_request: STOP file present') and rec['exercise']['state'] == 'untried',
           f"STOP during the active music request leaves the attempt untried with the trip: {rec['state']} {rec['error'][:160]}")
    expect(latency is not None and latency <= 2.5, f'the trip followed the STOP write within one guard interval: {latency}s')
    cancel = rq['cleanup'].get('music_job_cancel') or {}
    log_text = cancel_log.read_text(encoding='utf-8') if cancel_log.exists() else ''
    expect(cancel.get('job') == 'job-1' and cancel.get('http') == 200 and 'DELETE /api/jobs/job-1' in log_text, f'exactly the owned music job was cancelled on our server: {cancel} log {log_text!r}')
    trip = rq.get('trip') or {}
    expect(rq['cleanup']['active_processes_after'] == 0 and rq['cleanup']['job_handle_closed'] and trip.get('kind') == 'stopped' and trip.get('phase') == 'request',
           f'the owned tree alone was reaped and the handle closed after a request-phase STOP trip: trip {trip} cleanup {rq["cleanup"]}')
    phases = [s['phase'] for s in rq['samples']]
    # the healthz loop may reuse the launch sample (taken under 2 s earlier), so 'server_start' is not required to appear by name
    expect(len(rq['samples']) >= 4 and {'launch', 'model_load', 'request'} <= set(phases) and all('peak_job_commit_bytes' in s for s in rq['samples'] if s['phase'] != 'launch'),
           f'samples retained across every phase with the owned-job peak: {phases}')
    expect(any(step.get('loaded') for step in rec['steps']) and not any(step.get('request_verified') for step in rec['steps']), f'the load completed and the request never verified: {rec["steps"]}')
    observations['music_trip'] = {'error': rec['error'][:120], 'stop_to_trip_s': round(latency, 3) if latency is not None else None, 'cancel': cancel, 'samples': len(rq['samples']), 'phases': sorted(set(phases))}
    # 4. artifact floor, the model load hangs after writing STOP
    entry, cfg, root, stop, cancel_log = fake_cell(tmp, 'load', hang_load=True)
    rec = run_cells.serve_and_exercise(entry, cfg, root, sys.executable, '1', None, 30, stop_file=str(stop), resource={'kind': 'artifact_floor', 'floors': {'commit': 1, 'gpu': None}, 'margin_bytes': 2 << 30, 'admission_samples': []})
    rq = rec['resource_qualification']
    latency = stop_to_trip_seconds(stop, rec) if stop.exists() and rq.get('trip') else None
    expect(rec['state'] == 'untried' and rec['error'].startswith('stopped_during_model_load: STOP file present') and rq['verdict'] == 'not_contained',
           f"STOP during a hanging model load leaves the attempt untried with the trip: {rec['state']} {rec['error'][:160]}")
    expect(latency is not None and latency <= 2.5, f'the load trip followed the STOP write within one guard interval: {latency}s')
    expect(rec['server_returncode'] is not None and not any(step.get('loaded') for step in rec['steps']) and len(rq['samples']) >= 2 and 'music_job_cancel' not in rq['cleanup'],
           f'the server was ended, nothing was recorded as loaded, samples retained, no foreign cancel: rc {rec["server_returncode"]} steps {rec["steps"]} cleanup {rq["cleanup"]}')
    observations['load_trip'] = {'error': rec['error'][:120], 'stop_to_trip_s': round(latency, 3) if latency is not None else None, 'samples': len(rq['samples'])}

assert not failures, '\n'.join(failures)
print(json.dumps({'passed': True, 'observations': observations}, default=str))
