"""Plan and prepare checksum-pinned native runtimes in owned, versioned storage."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zipfile

from .config import home
from .hardware import discover, read_location, ReleaseRedirect


class InstallCancelled(RuntimeError):
    pass


def _check(cancelled):
    if cancelled is not None and cancelled.is_set():
        raise InstallCancelled('Native runtime preparation cancelled')


def _hash(path, cancelled=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            _check(cancelled)
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='native-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _identity(value):
    fields = {key: value[key] for key in ('runtime', 'release', 'profile', 'os', 'arch', 'assets', 'license')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def plan(catalog, profile, target=None):
    """Read a trusted catalog; planning never downloads or executes its runtime."""
    catalog = str(catalog)
    raw = read_location(catalog)
    if len(raw) > 1024 * 1024:
        raise ValueError('Native runtime catalog exceeds 1 MiB')
    data = json.loads(raw)
    if data.get('version') != 1 or data.get('runtime') != 'llama.cpp':
        raise ValueError('Unsupported native runtime catalog')
    if any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value)
           for value in (data.get('release'), profile)):
        raise ValueError('Invalid native release/profile name')
    item = data['profiles'][profile]
    hardware = discover()
    if (item['os'], item['arch']) != (hardware['os'], hardware['arch']):
        raise ValueError('Native runtime profile does not match this OS/architecture')
    assets = []
    for artifact in item['assets']:
        digest = artifact.get('sha256', '').lower()
        if not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Every runtime asset requires a SHA-256 pin')
        location = artifact.get('url') or artifact['file']
        if urllib.parse.urlparse(catalog).scheme in {'http', 'https'}:
            location = urllib.parse.urljoin(catalog, location)
        elif not urllib.parse.urlparse(location).scheme:
            location = str(Path(catalog).expanduser().resolve().parent / location)
        parsed = urllib.parse.urlparse(location)
        local = not parsed.scheme or (os.name == 'nt' and len(parsed.scheme) == 1)
        if not local and parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}):
            raise ValueError('Runtime downloads require HTTPS or loopback HTTP')
        assets.append({'location': location, 'sha256': digest})
    if not assets or len(assets) > 8:
        raise ValueError('Native runtime requires 1..8 pinned assets')
    result = {'schema_version': 1, 'runtime': 'llama.cpp', 'release': data['release'], 'profile': profile,
              'os': item['os'], 'arch': item['arch'], 'assets': assets,
              'license': item.get('license', data.get('license', 'unknown')),
              'license_url': item.get('license_url', data.get('license_url')),
              'source': data.get('source', 'https://github.com/ggml-org/llama.cpp'),
              'catalog': catalog, 'catalog_sha256': hashlib.sha256(raw).hexdigest(),
              'requires_user_python': False, 'hardware': hardware}
    result['fingerprint'] = _identity(result)
    result['destination'] = str(Path(target).expanduser().resolve() if target else
                                home() / 'native' / (data['release'] + '-' + profile + '-' + result['fingerprint'][:12]))
    result['status'] = 'present_unverified' if Path(result['destination']).exists() else 'absent'
    return result


def _receipt(destination, expected=None, cancelled=None):
    destination = Path(destination).resolve()
    receipt = json.loads((destination / 'receipt.json').read_text(encoding='utf-8'))
    if expected is not None and receipt.get('fingerprint') != expected:
        raise ValueError('Runtime destination has a different installation identity')
    for name, digest in receipt.get('files', {}).items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination) or not path.is_file() or _hash(path, cancelled) != digest:
            raise ValueError('Installed native runtime changed since preparation: ' + name)
    executable = Path(receipt['executable']).resolve()
    if not executable.is_relative_to(destination) or not executable.is_file() or _hash(executable, cancelled) != receipt['binary_sha256']:
        raise ValueError('Installed native executable changed since installation')
    return receipt


def _extract(archive, staged, cancelled, progress):
    expanded = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for member in stream.infolist():
                _check(cancelled)
                name = member.filename.replace('\\', '/')
                output = (staged / name).resolve()
                mode = member.external_attr >> 16
                if (not output.is_relative_to(staged) or ':' in name or stat.S_ISLNK(mode)):
                    raise ValueError('Unsafe native archive path')
                expanded += member.file_size
                if expanded > 8 * 1024**3:
                    raise ValueError('Expanded runtime exceeds 8 GiB')
                if member.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(member) as source, output.open('wb') as target:
                    while chunk := source.read(1024 * 1024):
                        _check(cancelled)
                        target.write(chunk)
                if os.name != 'nt' and mode:
                    output.chmod(mode & 0o777)
                progress('extracting', expanded, None)
    else:
        with tarfile.open(archive) as stream:
            for member in stream:
                _check(cancelled)
                expanded += member.size
                if expanded > 8 * 1024**3:
                    raise ValueError('Expanded runtime exceeds 8 GiB')
                stream.extract(member, staged, filter='data')
                progress('extracting', expanded, None)


def prepare(prepared_plan, *, progress=None, cancelled=None, activate=True):
    """Prepare a server-created plan, verify on-disk reuse, then atomically activate.

    Network callers select catalog profile names through the owning server. Never
    accept arbitrary client-authored plans/URLs as authorization to run binaries.
    """
    from filelock import FileLock
    value = dict(prepared_plan)
    if value.get('fingerprint') != _identity(value):
        raise ValueError('Native runtime plan changed after planning')
    arch = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(platform.machine().lower(), platform.machine().lower())
    if (value['os'], value['arch']) != (platform.system().lower(), arch):
        raise ValueError('Native runtime plan does not match this OS/architecture')
    progress = progress or (lambda stage, completed, total: None)
    dest = Path(value['destination']).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    _check(cancelled)
    # ponytail: one lock per immutable target prevents duplicate publication.
    with FileLock(str(dest) + '.lock', timeout=0):
        if dest.exists():
            progress('verifying_reuse', 0, None)
            receipt = _receipt(dest, value['fingerprint'], cancelled)
            _check(cancelled)
            if activate:
                _write(home() / 'native-runtime.json', receipt)
            return {**receipt, 'reused': True}
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix='native-install-') as tmp:
            root = Path(tmp)
            staged = root / 'runtime'
            staged.mkdir()
            for i, artifact in enumerate(value['assets']):
                _check(cancelled)
                location = artifact['location']
                parsed = urllib.parse.urlparse(location)
                local = not parsed.scheme or (os.name == 'nt' and len(parsed.scheme) == 1)
                archive = root / str(i)
                handlers = [ReleaseRedirect()]
                if parsed.hostname in {'127.0.0.1', 'localhost', '::1'}:
                    handlers.append(urllib.request.ProxyHandler({}))
                source = open(location, 'rb') if local else urllib.request.build_opener(*handlers).open(location, timeout=10)
                with source, archive.open('wb') as output:
                    total = os.fstat(source.fileno()).st_size if local else int(source.headers.get('Content-Length', 0)) or None
                    completed = 0
                    progress('downloading', completed, total)
                    while chunk := source.read(1024 * 1024):
                        _check(cancelled)
                        completed += len(chunk)
                        if completed > 2 * 1024**3:
                            raise ValueError('Native archive exceeds 2 GiB')
                        output.write(chunk)
                        progress('downloading', completed, total)
                _check(cancelled)
                progress('verifying', completed, total)
                if _hash(archive, cancelled) != artifact['sha256']:
                    raise ValueError('Native runtime checksum mismatch')
                _extract(archive, staged, cancelled, progress)
            executables = list(staged.rglob('llama-server.exe' if os.name == 'nt' else 'llama-server'))
            if len(executables) != 1:
                raise ValueError('Expected exactly one llama-server executable')
            _check(cancelled)
            progress('checking_executable', 0, None)
            from .process_command import module_command
            version = subprocess.check_output(module_command('titanos.native_process', '--query-version', 30, executables[0]),
                                              text=True, stderr=subprocess.STDOUT, timeout=35)
            files = {str(path.relative_to(staged)): _hash(path, cancelled)
                     for path in staged.rglob('*') if path.is_file()}
            executable = dest / executables[0].relative_to(staged)
            receipt = {**value, 'status': 'prepared', 'executable': str(executable), 'files': files,
                       'binary_sha256': files[str(executables[0].relative_to(staged))],
                       'version_output': version, 'native_inference_qualified': False,
                       'prepared_at': time.time()}
            _write(staged / 'receipt.json', receipt)
            _check(cancelled)
            os.rename(staged, dest)
        if activate:
            _write(home() / 'native-runtime.json', receipt)
        progress('prepared', 1, 1)
        return {**receipt, 'reused': False}


def install(catalog, profile, target=None, *, progress=None, cancelled=None):
    return prepare(plan(catalog, profile, target), progress=progress, cancelled=cancelled)


def executable_path(explicit=None):
    if explicit:
        value = Path(explicit).expanduser().resolve()
    else:
        receipt = json.loads((home() / 'native-runtime.json').read_text(encoding='utf-8'))
        if receipt.get('files'):
            receipt = _receipt(receipt['destination'], receipt.get('fingerprint'))
        value = Path(receipt['executable'])
        if not value.is_file() or _hash(value) != receipt['binary_sha256']:
            raise ValueError('Installed native executable changed since installation')
    if not value.is_file():
        raise ValueError('llama-server executable not found; run native-install or set --runtime')
    return value


class RuntimeSetup:
    """One existing app's native setup queue; immutable receipts survive restart.

    Catalog is server configuration, never client-provided executable policy.
    The app closes this owner at shutdown, just like its acquisition store.
    """
    def __init__(self, catalog, root=None):
        from filelock import FileLock
        self.catalog = str(catalog)
        self.root = Path(root or home() / 'runtime-setups')
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = FileLock(str(self.root / 'owner.lock'), thread_local=False)
        self.lock = threading.RLock()
        self.events = {}
        self.closed = False
        self.started = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='titanos-runtime-setup')

    def _start(self):
        with self.lock:
            if self.started:return
            if self.closed:raise ValueError('Runtime setup owner is closed')
            self.owner.acquire(timeout=0)
            try:
                for path in self.root.glob('*.json'):
                    record = json.loads(path.read_text(encoding='utf-8'))
                    if record['status'] in {'queued', 'running', 'cancelling'}:
                        record.update(status='interrupted', error='Previous app stopped; retry explicitly')
                        _write(path, record)
                self.started=True
            except BaseException:
                self.owner.release()
                raise

    def plan(self, profile):
        self._start()
        return plan(self.catalog, profile)

    def get(self, identifier):
        if not self.closed:self._start()
        if not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('Invalid runtime setup identity')
        with self.lock:
            return json.loads((self.root / (identifier + '.json')).read_text(encoding='utf-8'))

    def submit(self, profile):
        return self._submit(self.plan(profile))

    def list(self):
        self._start()
        with self.lock:
            return {'object': 'list', 'data': [json.loads(p.read_text(encoding='utf-8'))
                    for p in sorted(self.root.glob('*.json'))]}

    def retry(self, identifier):
        record = self.get(identifier)
        if record['status'] not in {'failed', 'cancelled', 'interrupted'}:
            raise ValueError('Only failed, cancelled or interrupted runtime setups can retry')
        return self._submit(record['plan'], retry_of=identifier)

    def _submit(self, value, retry_of=None):
        if self.closed:
            raise ValueError('Runtime setup owner is closed')
        identifier = uuid.uuid4().hex
        record = {'id': identifier, 'object': 'runtime.setup', 'status': 'queued', 'plan': value,
                  'created_at': time.time(), 'progress': {'stage': 'queued', 'completed': 0, 'total': None, 'unit': 'bytes'}}
        if retry_of:
            record['retry_of'] = retry_of
        event = threading.Event()
        with self.lock:
            self.events[identifier] = event
            _write(self.root / (identifier + '.json'), record)

        def save():
            record['updated_at'] = time.time()
            _write(self.root / (identifier + '.json'), record)

        def progress(stage, completed, total):
            with self.lock:
                if event.is_set():
                    record['status'] = 'cancelling'
                record['progress'] = {'stage': stage, 'completed': completed, 'total': total,
                                      'unit': 'files' if stage == 'extracting' else 'steps' if stage in ('prepared', 'checking_executable', 'verifying_reuse') else 'bytes'}
                save()

        def run():
            try:
                with self.lock:
                    _check(event)
                    record['status'] = 'running'
                    save()
                receipt = prepare(value, progress=progress, cancelled=event)
                with self.lock:
                    record.update(status='prepared', receipt=receipt, error=None)
            except InstallCancelled as exc:
                with self.lock:
                    record.update(status='cancelled', error=str(exc))
            except Exception as exc:
                with self.lock:
                    record.update(status='failed', error=str(exc))
            finally:
                with self.lock:
                    save()
                    self.events.pop(identifier, None)
        self.executor.submit(run)
        return self.get(identifier)

    def cancel(self, identifier):
        with self.lock:
            record = self.get(identifier)
            event = self.events.get(identifier)
            if event is not None:
                event.set()
                record['status'] = 'cancelling'
                _write(self.root / (identifier + '.json'), record)
            return record

    def close(self):
        self.closed = True
        for event in list(self.events.values()):
            event.set()
        self.executor.shutdown(wait=True)
        if self.started:
            self.owner.release()
            self.started=False
