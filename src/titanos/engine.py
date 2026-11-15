"""Model lifecycle and inference, with no external agent or orchestration service."""

import gc
import threading
import time

from .config import Config


class EngineError(Exception):
    def __init__(self, message, status=400, code="invalid_request_error"):
        super().__init__(message)
        self.status, self.code = status, code


def _finish_inference(callback, result, error):
    if callback is not None:
        try:
            callback(result, error)
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note('Request ownership cleanup also failed: ' + type(cleanup).__name__)


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.current = None
        self.runtime = None
        self.states = {name: "unloaded" for name in config.models}
        self.started = int(time.time())

    def validate(self, name, task=None):
        if name not in self.config.models:
            raise EngineError(f"Unknown model {name!r}", 404, "model_not_found")
        supported = {"completion": {"chat"}, "chat": {"chat", "vision"}, "image-edit": {"image"}}
        if task and self.config.models[name].task not in supported.get(task, {task}):
            raise EngineError(f"Model {name!r} does not support {task}")

    def _runtime_fault(self):
        return getattr(self.runtime, 'runtime_fault', None) if self.runtime is not None else None

    def _unavailable(self):
        fault = self._runtime_fault()
        if fault is not None:
            with self.state_lock:
                if self.current:self.states[self.current] = 'unavailable'
            raise EngineError('Native runtime is unavailable and remains resident; restart its owning process after device re-admission', 503, 'device_unavailable')

    def _require_loaded(self, name):
        """Caller owns self.lock; state observation never authorizes a load."""
        self._unavailable()
        with self.state_lock:
            if self.current != name or self.runtime is None or self.states[name] != 'loaded':
                raise EngineError('The requested model is not already loaded', 409, 'model_not_loaded')

    def _loaded_details(self, name):
        """Passive child metadata refresh while its parent retains the busy lease."""
        with self.lock:
            if name == self.current and self._runtime_fault() is not None:
                return {'backend': self.config.models[name].backend, 'device': self.runtime.device,
                        'status': 'unavailable', 'available': False, 'resident': True,
                        'runtime_fault': self._runtime_fault()}
            self._require_loaded(name)
            if hasattr(self.runtime, 'details'):
                return self.runtime.details()
            return {'backend': self.config.models[name].backend, 'device': self.runtime.device}

    def catalog(self, loaded_only=False):
        with self.state_lock:
            return {"object": "list", "data": [
                {"id": name, "object": "model", "created": self.started, "owned_by": "local",
                 "task": spec.task, "status": self.states[name],
                 **({"resident": True, "runtime_fault": self._runtime_fault()} if name == self.current and self._runtime_fault() is not None else {})}
                for name, spec in self.config.models.items()
                if not loaded_only or self.states[name] in {"loaded", "unavailable"}
            ]}

    def _clear(self):
        self._unavailable()
        if self.runtime is not None and hasattr(self.runtime, 'close'):
            self.runtime.close()
        with self.state_lock:
            if self.current:
                self.states[self.current] = "unloaded"
            self.current, self.runtime = None, None
        gc.collect()
        import sys
        torch = sys.modules.get("torch")
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()

    def _load(self, name):
        self._unavailable()
        if name == self.current and self.runtime is not None:
            return
        self._clear()
        with self.state_lock:
            self.states[name] = "loading"
        try:
            spec = self.config.models[name]
            if spec.task == 'vad':
                from .vad import SileroVAD
                runtime = SileroVAD(spec)
            elif spec.task == 'diarization':
                from .diarization import SherpaDiarizationModel
                runtime = SherpaDiarizationModel(spec)
            elif spec.parallelism == "tensor_parallel":
                from .parallel import TensorParallelModel
                runtime = TensorParallelModel(spec)
            elif spec.backend == "qwen3-tts":
                from .qwen_tts_backend import QwenTTSModel
                runtime=QwenTTSModel(spec)
            elif spec.backend == "sherpa-onnx":
                from .sherpa_tts import SherpaTTSModel
                runtime=SherpaTTSModel(spec)
            elif spec.backend == "ace-step":
                from .music_backend import AceStepModel
                runtime=AceStepModel(spec)
            elif spec.backend == 'tesseract':
                from .tesseract import TesseractModel
                runtime = TesseractModel(spec)
            elif spec.backend == "onnx-diffusers":
                from .onnx_images import ONNXImageModel
                runtime = ONNXImageModel(spec)
            elif spec.backend in {"onnx", "openvino"}:
                from .compiled import CompiledModel
                runtime = CompiledModel(spec)
            elif spec.backend == "openvino-genai":
                if spec.task == 'image':
                    from .genai_images import GenAIImageModel
                    runtime = GenAIImageModel(spec)
                else:
                    from .genai import GenAIModel
                    runtime = GenAIModel(spec)
            else:
                from .backends import LoadedModel
                runtime = LoadedModel(spec)
        except Exception as exc:
            failed = getattr(exc, 'failed_runtime', None)
            if failed is not None and getattr(failed, 'runtime_fault', None) is not None:
                with self.state_lock:
                    self.current, self.runtime = name, failed
                    self.states[name] = 'unavailable'
                raise EngineError(str(exc), 503, 'device_unavailable') from exc
            with self.state_lock:
                self.states[name] = "error"
            raise EngineError(f"Could not load {name}: {exc}", 503, "model_load_failed") from exc
        with self.state_lock:
            self.current, self.runtime = name, runtime
            self.states[name] = "loaded"

    def load(self, name, worker=None):
        self.local_worker(worker)
        self.validate(name)
        with self.lock:
            self._load(name)
        return {"model": name, "status": "loaded"}

    def details(self, name, worker=None):
        self.local_worker(worker)
        self.validate(name)
        with self.lock:
            if name == self.current and self._runtime_fault() is not None:
                return {'backend':self.config.models[name].backend, 'device':self.runtime.device,
                        'status':'unavailable', 'available':False, 'resident':True, 'runtime_fault':self._runtime_fault()}
            self._load(name)
            if hasattr(self.runtime, "details"):
                return self.runtime.details()
            return {"backend": self.config.models[name].backend, "device": self.runtime.device}

    def loaded_details(self, name):
        """Runtime details only when `name` is the resident, loaded model; never loads and never waits behind a running
        inference (the engine lock is held for a whole request): when the lock is busy the caller keeps the passive view."""
        if not self.lock.acquire(blocking=False):
            return None
        try:
            if name != self.current or self.runtime is None or self.states.get(name) != 'loaded' or self._runtime_fault() is not None:
                return None
            if not hasattr(self.runtime, 'details'):
                return None
            try:
                return self.runtime.details()
            except Exception:
                return None
        finally:
            self.lock.release()

    def unload(self, name=None, worker=None):
        self.local_worker(worker)
        if name:
            self.validate(name)
        with self.lock:
            if name is None or name == self.current:
                self._clear()
        return {"model": name, "status": self.states.get(name, "unloaded")}

    @staticmethod
    def local_worker(worker):
        if worker is not None:
            raise EngineError("This server has no replica worker assignment", 400)

    def close(self):
        self.unload()

    def infer(self, name, task, data, emit=None, cancel=None, *, require_loaded=False,
              on_admitted=None, on_finished=None):
        self.local_worker(data.get('worker'))
        self.validate(name, task)
        if type(require_loaded) is not bool:
            raise EngineError('require_loaded must be a boolean')
        data = {key: value for key, value in data.items() if key != 'require_loaded'}
        if task == 'speech':
            from .openai_params import speech_voice
            data = speech_voice(self.config.models[name], data)  # client voice names through the model's voice_map
        if task == 'diarization':
            from .diarization import prepare
            try:
                prepare(self.config.models[name], data)
            except (ValueError, TypeError) as exc:
                raise EngineError(str(exc)) from exc
        # ponytail: one resident model and serialized inference; run more processes
        # for independent devices. Add batching only after measured demand.
        with self.lock:
            if cancel and cancel.is_set():
                raise EngineError("Request cancelled", 499, "cancelled")
            if require_loaded:
                self._require_loaded(name)
            else:
                self._load(name)
            result = error = None
            try:
                if require_loaded and on_admitted is not None:
                    on_admitted()
                if require_loaded and cancel and cancel.is_set():
                    raise EngineError('Request cancelled', 499, 'cancelled')
                try:
                    result = self.runtime.infer(task, data, emit, cancel)
                    return result
                except EngineError:
                    if self._runtime_fault() is not None:
                        with self.state_lock:self.states[name] = 'unavailable'
                    raise
                except (ValueError, TypeError) as exc:
                    raise EngineError(str(exc)) from exc
                except Exception as exc:
                    if self.config.models[name].parallelism == 'tensor_parallel':
                        self._clear()
                        with self.state_lock:
                            self.states[name] = 'error'
                    raise EngineError(f"Inference failed: {exc}", 500, "inference_error") from exc
            except BaseException as exc:
                error = exc
                raise
            finally:
                if require_loaded:
                    _finish_inference(on_finished, result, error)
