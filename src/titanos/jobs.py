"""Bounded local video jobs with atomic status records and explicit cancellation.

Owner scoping (broker): a job record may carry owner_key_id (the issued key that submitted it) and decision_id (the
broker Decision of a cloud job); both keys are written only when set, so keyless and bootstrap records keep their exact
shape. public() never exposes owner_key_id. Idempotency-Key matching also requires an equal owner_key_id, so the same
key from two keys yields two jobs. list(), get(), delete() and cancel() take an optional principal: an admin principal
(bootstrap, keyless loopback, or a key holding admin) sees every job; any other principal sees only the jobs whose owner_key_id equals its
key id, and a job it cannot see answers 404 not_found. Keyless loopback principals see every job, as they see every
artifact (broker.ownership).

Client lifecycle contract (asynchronous clients such as Cauldron):
- submit with an Idempotency-Key replays the existing record (same body) or answers 409 idempotency_conflict (other
  body); a deleted record no longer claims its key. A replayed interrupted record stays interrupted: retry it with a new
  key (the record says retryable).
- list(request_key=...) finds a job by the hash of the client's Idempotency-Key (the server hashes the header value), so
  a client that lost the submit answer, or restarted, can recover the job id; status= and task= filter the list.
- cancel() answers the record plus cancel_accepted: true when the job was still queued, running or cancelling (the
  record then reads cancelling until execution stops, and always ends in a terminal status: cancelled, or completed or
  failed when the work finished first), false when it was already terminal (the record is returned unchanged).
  cancel_requested_at is kept on the record. cancel_accepted is part of the answer only, never stored.
- a restart marks queued, running and cancelling records interrupted, with interrupted_at and retryable: true; the
  work is not replayed.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
import copy
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid
from .engine import EngineError

_native_generation_observer = ContextVar('native_generation_observer', default=None)


def _is_admin(principal):
    if principal is None:
        return True
    if getattr(principal, 'kind', None) in ('bootstrap', 'anonymous_loopback'):
        return True
    return 'admin' in (getattr(principal, 'scopes', None) or ())


def visible(record, principal):
    """Whether principal may see the job record (None means an unscoped, in-process caller)."""
    if _is_admin(principal):
        return True
    return record.get('owner_key_id') == getattr(principal, 'key_id', None)

class VideoJobs:
    def __init__(self, root, settings=None):
        from .settings import load_settings
        self.settings=settings or load_settings()
        self.root=Path(root)
        self.lock=threading.RLock()
        self.executor=None
        self.events={}
        self._owner_lock=None

    def start(self):
        if self._owner_lock:
            return
        from filelock import FileLock, Timeout
        self.root.mkdir(parents=True,exist_ok=True)
        owner=FileLock(self.root/'owner.lock',thread_local=False)
        try:
            owner.acquire(timeout=0)
        except Timeout as exc:
            raise EngineError('Another server owns this job directory; use a separate TITANOS_HOME',409,'job_store_busy') from exc
        try:
            for path in self.root.glob('*.json'):
                try:
                    record=json.loads(path.read_text())
                    if record['id']!=path.stem or not re.fullmatch(r'[a-f0-9]{32}',record['id']):raise ValueError('Invalid identity')
                    if record['status'] in {'queued','running','cancelling'}:
                        record.update(status='interrupted',error='Previous server stopped before completion; job was not replayed',
                                      interrupted_at=time.time(),retryable=True)
                        self.write(record)
                    elif record['updated_at'] < time.time()-self.settings.artifact_ttl_seconds:
                        path.unlink()
                except (ValueError,KeyError,TypeError):
                    path.rename(path.with_suffix('.invalid-'+uuid.uuid4().hex))
                    logging.warning('Preserved an unreadable job record outside the active queue')
            self.executor=ThreadPoolExecutor(max_workers=self.settings.job_workers,thread_name_prefix='titanos-assets')
            self._owner_lock=owner
        except Exception:
            owner.release()
            raise

    def write(self, record):
        record['updated_at']=time.time()
        fd,name=tempfile.mkstemp(dir=self.root,prefix='job-')
        try:
            with os.fdopen(fd,'w') as stream:
                json.dump(record,stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name,self.root/(record['id']+'.json'))
        finally:
            Path(name).unlink(missing_ok=True)

    def list(self, task=None, limit=100, principal=None, status=None, request_key=None):
        # request_key is the sha256 of a client's Idempotency-Key, as stored at submit
        with self.lock:
            self.start()
            records = [json.loads(p.read_text()) for p in self.root.glob('*.json')]
            records = [r for r in records if r['status']!='deleted' and (task is None or r.get('task', 'video') == task)
                       and (status is None or r['status'] == status)
                       and (request_key is None or r.get('request_key') == request_key)
                       and visible(r, principal)]
            records.sort(key=lambda r: r['created_at'], reverse=True)
            return {'object': 'list', 'data': [self.public(r) for r in records[:limit]]}

    @staticmethod
    def public(record):
        return {k: v for k, v in record.items() if k not in {'request_key', 'fingerprint', 'owner_key_id'}}

    def owner(self, identifier):
        """The owner_key_id of a job record (None when unowned or unknown)."""
        with self.lock:
            self.start()
            if not re.fullmatch(r'[a-f0-9]{32}',identifier):
                return None
            path=self.root/(identifier+'.json')
            if not path.is_file():
                return None
            return json.loads(path.read_text()).get('owner_key_id')

    def get(self, identifier, principal=None):
        with self.lock:
            self.start()
            if not re.fullmatch(r'[a-f0-9]{32}',identifier):
                raise EngineError('Job not found',404,'not_found')
            path=self.root/(identifier+'.json')
            if not path.is_file():
                raise EngineError('Job not found',404,'not_found')
            record=json.loads(path.read_text())
            if record['status']=='deleted':raise EngineError('Job deleted',404,'not_found')
            if not visible(record, principal):raise EngineError('Job not found',404,'not_found')
            evidence=getattr(self.events.get(identifier),'native_generation',None)
            if evidence is not None:record['native_generation']=copy.deepcopy(evidence)
            return self.public(record)

    def delete(self,identifier,principal=None):
        with self.lock:
            self.get(identifier,principal)
            if identifier in self.events:
                raise EngineError('Cancel the running job and wait for cancellation before deleting it',409,'job_active')
            record=json.loads((self.root/(identifier+'.json')).read_text())
            record.update(status='deleted')
            self.write(record)
            return {'id':identifier,'object':'video','deleted':True}

    def submit(self, model, generate, task='video', request_key=None, fingerprint=None, cleanup=None,
               owner_key_id=None, decision_id=None, on_settled=None):
        # on_settled() runs once the job can no longer run (replayed from its Idempotency-Key, finished, failed,
        # cancelled, or not dispatched), so a caller's admission is always released; it may be called again (idempotent)
        with self.lock:
            self.start()
            if request_key:
                for path in self.root.glob('*.json'):
                    previous = json.loads(path.read_text())
                    if previous.get('status') == 'deleted':
                        continue
                    if previous.get('request_key') == request_key and previous.get('owner_key_id') == owner_key_id:
                        if previous.get('fingerprint') != fingerprint:
                            raise EngineError('Idempotency key already used for a different request',409,'idempotency_conflict')
                        if on_settled is not None:on_settled()
                        return self.public(previous)
            if len(self.events)>=self.settings.queue_capacity:
                raise EngineError('Asset queue is full',429,'queue_full')
            identifier=uuid.uuid4().hex
            record={'id':identifier,'object':'video.job' if task == 'video' else 'asset.job','task':task,'model':model,'status':'queued','created_at':time.time(),'execution_started':False}
            if request_key:
                record.update(request_key=request_key,fingerprint=fingerprint)
            if owner_key_id is not None:
                record['owner_key_id']=owner_key_id
            if decision_id is not None:
                record['decision_id']=decision_id
            self.write(record)
            event=self.events[identifier]=threading.Event()
            def observe_native(value):
                with self.lock:
                    if self.events.get(identifier) is event:
                        if value.get('job_id') != identifier:raise ValueError('Native evidence job identity mismatch')
                        event.native_generation=copy.deepcopy(value)
            def finish(failure=None,result=None):
                # cancel() uses this same lock: it either wins before the final
                # decision, or observes a published terminal record with no event.
                with self.lock:
                    discarded=False
                    def discard():
                        nonlocal discarded
                        if discarded or cleanup is None:
                            return
                        discarded=True
                        try:
                            cleanup()
                        except Exception as exc:
                            record['cleanup_error']=f'{type(exc).__name__}: {exc}'
                            record.setdefault('error','Artifact cleanup failed: '+str(exc))
                            if failure is not None:
                                failure.add_note('Artifact cleanup failed: '+record['cleanup_error'])
                            logging.exception('Artifact cleanup failed for job %s',identifier)
                    record.update(status='cancelled' if event.is_set() else 'failed' if failure is not None else 'completed')
                    requested=getattr(event,'cancel_requested_at',None)
                    if requested is not None:
                        record['cancel_requested_at']=requested
                    evidence=getattr(event,'native_generation',None)
                    if evidence is not None:
                        try:record['native_generation']=copy.deepcopy(evidence)
                        except Exception as exc:
                            record.pop('native_generation',None)
                            record['native_generation_error']={'stage':'terminal_snapshot','type':type(exc).__name__,'job_id':identifier}
                        else:record['native_generation']['job_result_disposition']='published' if record['status']=='completed' else 'discarded' if result is not None else 'no_result'
                    if failure is not None:
                        record['error']=str(failure)
                    if record['status']=='completed':
                        record['result']=result
                    else:
                        record.pop('result',None)
                        discard()
                    try:
                        self.write(record)
                    except Exception as exc:
                        record.pop('result',None)
                        if 'native_generation' in record:
                            record['native_generation']['job_result_disposition']='discarded' if result is not None else 'no_result'
                        record.update(status='failed',publication_error=f'{type(exc).__name__}: {exc}')
                        record.setdefault('error','Job publication failed: '+str(exc))
                        if failure is not None:
                            failure.add_note('Job publication failed: '+record['publication_error'])
                        discard()
                        try:
                            self.write(record)
                        except Exception as retry_error:
                            if failure is not None:
                                failure.add_note(f'Could not persist failed publication: {type(retry_error).__name__}: {retry_error}')
                            logging.exception('Could not persist failed publication for job %s',identifier)
                    finally:
                        self.events.pop(identifier,None)
            def run():
                failure=None
                result=None
                binding=_native_generation_observer.set((identifier,observe_native))
                try:
                    with self.lock:
                        # submit() can enqueue this callback before thread startup
                        # raises. A failed dispatch has already released its event.
                        if self.events.get(identifier) is not event:
                            return
                        if event.is_set():
                            raise EngineError('Cancelled',499,'cancelled')
                        record['status']='running'
                        record['execution_started']=True
                        self.write(record)
                    result=generate(event)
                except Exception as exc:
                    failure=exc
                finally:
                    _native_generation_observer.reset(binding)
                finish(failure,result)
            def settled_run():
                try:
                    run()
                finally:
                    if on_settled is not None:on_settled()
            try:
                self.executor.submit(copy_context().run,settled_run)
            except Exception as exc:
                finish(failure=exc)
                if on_settled is not None:on_settled()
                raise
            return self.public(record)

    def cancel(self, identifier, principal=None):
        with self.lock:
            self.get(identifier, principal)
            record=json.loads((self.root/(identifier+'.json')).read_text())
            accepted=identifier in self.events
            if accepted:
                event=self.events[identifier]
                event.set()
                if getattr(event,'cancel_requested_at',None) is None:
                    event.cancel_requested_at=time.time()
                record.update(status='cancelling',cancel_requested_at=event.cancel_requested_at)
                self.write(record)
            return {**self.public(record),'cancel_accepted':accepted}

    def close(self):
        with self.lock:
            for event in self.events.values():
                event.set()
        if self.executor:
            self.executor.shutdown(wait=True)
        if self._owner_lock:
            self._owner_lock.release()
            self._owner_lock=None
