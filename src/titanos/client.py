"""Small synchronous client for the standalone Titanos HTTP API.

Only httpx (an engine dependency) is needed; importing this module never loads
model frameworks. Streaming yields the original JSON events, including tools
and usage. Job wait timeouts leave the server job running.

Routing hints: Client(location=..., target=...) sends X-Titanos-Location and
X-Titanos-Target on every request; an inference call overrides them with the
option keys _location and _target, which are removed before the body is sent.
Client.provenance(response) and client.last_provenance expose the X-Titanos-*
provenance headers (served by, served model, fallbacks, decision id).

Broker, protocol and admin methods mirror the HTTP routes one to one. Secrets
(credential values, minted key tokens) travel only in request or response
bodies: the Client never stores them, and callers must never log them.
"""

import base64
from contextlib import ExitStack
from dataclasses import dataclass
import ipaddress
import json
import math
import mimetypes
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote, urljoin, urlsplit

import httpx
from .retrieval_contract import RerankResult
from .retrieval_workflow_contract import RetrievalAnswerResult
from .audio_contract import VADResult, DiarizationResult, AudioAnalysisResult
from .pair_contract import PairClassificationResult


class TitanosError(ValueError):
    """HTTP or stream error, retaining the server's structured error fields."""

    def __init__(self, message, status_code=None, error=None):
        self.status_code = status_code
        self.error = error or {}
        self.code = self.error.get("code") or self.error.get("type")
        super().__init__((f"HTTP {status_code}: " if status_code else "") + message)


def _origin(url):
    parts = urlsplit(str(url))
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Use an HTTP(S) server URL without embedded credentials")
    return parts.scheme, parts.hostname.lower(), parts.port or (443 if parts.scheme == "https" else 80)


def _check(response):
    if response.is_success:
        return response
    response.read()
    try:
        detail = response.json().get("error", {})
        if not isinstance(detail, dict):
            detail = {"message": str(detail)}
    except (ValueError, AttributeError):
        detail = {}
    raise TitanosError(detail.get("message") or response.text or response.reason_phrase,
                       response.status_code, detail)


LOCATIONS = ("local_only", "prefer_local", "auto", "cloud_only")
HINT_OPTIONS = {"_location": "X-Titanos-Location", "_target": "X-Titanos-Target"}


def _hint_value(name, value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be a non-empty single-line string")
    if name in {"location", "_location"} and value not in LOCATIONS:
        raise ValueError(f"{name} must be one of {', '.join(LOCATIONS)}")
    return value


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, repr=False)
class CreatedKey:
    """A minted or rotated inbound key. secret is shown once by the server; the repr never prints it."""
    id: str
    secret: str
    record: dict

    def __repr__(self):
        return f"CreatedKey(id={self.id!r}, secret=<redacted>, record={self.record!r})"


class Client:
    def __init__(self, base_url=None, api_key=None, timeout=None, *, transport=None, location=None, target=None):
        if timeout is None:
            from .settings import load_settings
            timeout = load_settings().client_timeout
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("HTTP timeout must be finite and positive")
        self.base_url = (base_url or os.environ.get("TITANOS_URL") or "http://127.0.0.1:11435").rstrip("/")
        self.origin = _origin(self.base_url)
        if urlsplit(self.base_url).query or urlsplit(self.base_url).fragment:
            raise ValueError("Server URL cannot contain a query or fragment")
        key = os.environ.get("TITANOS_API_KEY", "") if api_key is None else api_key
        headers = {"Authorization": "Bearer " + key} if key else {}
        if location is not None:
            headers["X-Titanos-Location"] = _hint_value("location", location)
        if target is not None:
            headers["X-Titanos-Target"] = _hint_value("target", target)
        self.last_provenance = {}
        self.http = httpx.Client(headers=headers,
                                 timeout=timeout, trust_env=False, follow_redirects=False, transport=transport)

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _url(self, route):
        if not route.startswith("/") or route.startswith("//"):
            raise ValueError("API route must start with a single slash")
        return self.base_url + route

    @staticmethod
    def _hints(kwargs):
        """Move the per-call _location/_target option keys of a JSON or form body into request headers."""
        found = {}
        for field in ("json", "data"):
            body = kwargs.get(field)
            if isinstance(body, dict) and any(name in body for name in HINT_OPTIONS):
                body = dict(body)
                for name, header in HINT_OPTIONS.items():
                    value = body.pop(name, None)
                    if value is not None:
                        found[header] = _hint_value(name, value)
                kwargs[field] = body
        if found:
            kwargs["headers"] = {**(kwargs.get("headers") or {}), **found}
        return kwargs

    @staticmethod
    def provenance(response):
        """The X-Titanos-* provenance headers (and X-Request-ID) of a response, lower-cased names to values."""
        return {name.lower(): value for name, value in response.headers.items()
                if name.lower().startswith("x-titanos-") or name.lower() == "x-request-id"}

    def request(self, method, route, **kwargs):
        try:
            response = self.http.request(method, self._url(route), **self._hints(kwargs))
            self.last_provenance = self.provenance(response)
            return _check(response)
        except httpx.HTTPError as exc:
            raise TitanosError(f"Request failed: {exc}") from exc

    def _json(self, method, route, **kwargs):
        return self.request(method, route, **kwargs).json()

    def info(self):
        return self._json("GET", "/api/info")

    def models(self, loaded=False):
        return self._json("GET", "/api/models/loaded" if loaded else "/v1/models")

    def load(self, model, worker=None):
        return self._json("POST", "/api/models/load", json={"model": model, "worker": worker})

    def unload(self, model, worker=None):
        return self._json("POST", "/api/models/unload", json={"model": model, "worker": worker})

    def details(self, model, worker=None):
        return self._json("POST", "/api/models/details", json={"model": model, "worker": worker})

    def capabilities(self, model):
        return self._json("POST", "/api/models/capabilities", json={"model": model})

    def providers(self):
        return self._json("GET", "/api/model-providers")

    def discover_models(self, provider, task, query="", limit=20):
        return self._json("GET", "/api/model-providers/" + quote(provider, safe="") + "/search",
                          params={"task": task, "query": query, "limit": limit})

    def inspect_model_source(self, provider, identifier, revision=None):
        return self._json("POST", "/api/model-providers/inspect",
                          json={"provider": provider, "identifier": identifier, "revision": revision})

    def import_model_link(self, url):
        return self._json("POST", "/api/model-providers/import", json={"url": url})

    def plan_acquisition(self, **request):
        return self._json("POST", "/api/acquisitions/plan", json=request)

    def submit_acquisition(self, plan, *, idempotency_key=None):
        if hasattr(plan, "model_dump"):
            plan = plan.model_dump(mode="json")
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        return self._json("POST", "/api/acquisitions", json=plan, headers=headers)

    def acquisitions(self, limit=100):
        return self._json("GET", "/api/acquisitions", params={"limit": limit})

    def acquisition(self, identifier, **kwargs):
        return self._json("GET", "/api/acquisitions/" + quote(identifier, safe=""), **kwargs)

    def cancel_acquisition(self, identifier):
        return self._json("DELETE", "/api/acquisitions/" + quote(identifier, safe=""))

    def retry_acquisition(self, identifier):
        return self._json("POST", "/api/acquisitions/" + quote(identifier, safe="") + "/retry")

    def activate_acquisition(self, identifier, *, accepted_license_sha256=None, load=True, runtime_profile=None):
        return self._json("POST", "/api/acquisitions/" + quote(identifier, safe="") + "/activate",
                          json={"accepted_license_sha256": accepted_license_sha256, "load": load,
                                **({'runtime_profile':runtime_profile} if runtime_profile is not None else {})})

    def delete_acquired_weights(self,identifier):
        return self._json("DELETE","/api/acquisitions/"+quote(identifier,safe="")+"/weights")

    def wait_acquisition(self, identifier, *, timeout=None, interval=0.25):
        if timeout is not None and (not math.isfinite(timeout) or timeout<=0):
            raise ValueError("Acquisition wait timeout must be positive")
        if not math.isfinite(interval) or interval<=0:
            raise ValueError("Acquisition poll interval must be positive")
        started = time.monotonic()
        while True:
            value = self.acquisition(identifier)
            if value["status"] in {"prepared", "active", "failed", "cancelled", "interrupted"}:
                return value
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError("Acquisition wait timed out; the server operation remains active")
            time.sleep(interval)

    def plan_runtime_setup(self,profile):
        return self._json("POST","/api/runtime-setups/plan",json={"profile":profile})

    def runtime_profiles(self):
        return self._json("GET","/api/runtime-profiles")

    def submit_runtime_setup(self,profile):
        return self._json("POST","/api/runtime-setups",json={"profile":profile})

    def runtime_setups(self):
        return self._json("GET","/api/runtime-setups")

    def runtime_setup(self,identifier):
        return self._json("GET","/api/runtime-setups/"+quote(identifier,safe=""))

    def cancel_runtime_setup(self,identifier):
        return self._json("DELETE","/api/runtime-setups/"+quote(identifier,safe=""))

    def retry_runtime_setup(self,identifier):
        return self._json("POST","/api/runtime-setups/"+quote(identifier,safe="")+"/retry")

    def wait_runtime_setup(self,identifier,*,timeout=None,interval=.25):
        if timeout is not None and (not math.isfinite(timeout) or timeout<=0):
            raise ValueError("Runtime setup timeout must be positive")
        if not math.isfinite(interval) or interval<=0:raise ValueError("Runtime setup interval must be positive")
        started=time.monotonic()
        while True:
            value=self.runtime_setup(identifier)
            if value["status"] in {"prepared","failed","cancelled","interrupted"}:return value
            if timeout is not None and time.monotonic()-started>=timeout:
                raise TimeoutError("Runtime setup wait timed out; the server operation remains active")
            time.sleep(interval)

    def assignment_devices(self, *, model=None, refresh=False):
        params={**({'model':model} if model is not None else {}),**({'refresh':'true'} if refresh else {})}
        return self._json("GET","/api/assignments/devices",**({'params':params} if params else {}))

    def assignment(self,model):
        return self._json("GET","/api/assignments/"+quote(model,safe=""))

    def save_assignment(self,**payload):
        return self._json("POST","/api/assignments",json=payload)

    def _stream(self, route, payload):
        try:
            with self.http.stream("POST", self._url(route), **self._hints({"json": payload})) as response:
                self.last_provenance = self.provenance(response)
                _check(response)
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise TitanosError("Expected an event stream from the server")
                data = []
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif not line and data:
                        value, data = "\n".join(data), []
                        if value == "[DONE]":
                            return
                        event = json.loads(value)
                        if "error" in event:
                            detail = event["error"]
                            raise TitanosError(detail.get("message", "Inference failed"), error=detail)
                        yield event
                raise TitanosError("Event stream ended before [DONE]; the result may be incomplete")
        except httpx.HTTPError as exc:
            raise TitanosError(f"Stream failed: {exc}") from exc

    def _sse_events(self, route, payload, *, terminal, params=None):
        """Server-sent events without a [DONE] sentinel (Anthropic Messages, Gemini alt=sse): yields each event's JSON
        data. With terminal (an event type such as message_stop) the stream must reach it; without, it ends at the end of
        the stream. An error event (Anthropic {type: error}, Gemini {error: ...}) raises TitanosError."""
        try:
            with self.http.stream("POST", self._url(route), params=params, **self._hints({"json": payload})) as response:
                self.last_provenance = self.provenance(response)
                _check(response)
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise TitanosError("Expected an event stream from the server")
                data, name = [], None
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        name = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif not line and data:
                        value, data = "\n".join(data), []
                        event = json.loads(value)
                        kind = event.get("type", name) if isinstance(event, dict) else name
                        name = None
                        if isinstance(event, dict) and (kind == "error" or ("error" in event and "type" not in event)):
                            detail = event.get("error") if isinstance(event.get("error"), dict) else {"message": str(event.get("error"))}
                            raise TitanosError(detail.get("message", "Inference failed"), error=detail)
                        yield event
                        if terminal is not None and kind == terminal:
                            return
                if terminal is not None:
                    raise TitanosError(f"Event stream ended before {terminal}; the result may be incomplete")
        except httpx.HTTPError as exc:
            raise TitanosError(f"Stream failed: {exc}") from exc

    def _byte_stream(self, route, payload, *, params=None):
        """A binary response body (ElevenLabs streaming text-to-speech) as it arrives."""
        try:
            with self.http.stream("POST", self._url(route), params=params, **self._hints({"json": payload})) as response:
                self.last_provenance = self.provenance(response)
                _check(response)
                yield from response.iter_bytes()
        except httpx.HTTPError as exc:
            raise TitanosError(f"Stream failed: {exc}") from exc

    def chat(self, model, messages, *, stream=False, **options):
        payload = {**options, "model": model, "messages": messages, "stream": stream}
        return self._stream("/v1/chat/completions", payload) if stream else self._json("POST", "/v1/chat/completions", json=payload)

    def embeddings(self, model, input, **options):
        """Embed text with explicit truncate policy and per-input fidelity reporting."""
        return self._json("POST", "/v1/embeddings", json={**options, "model": model, "input": input})

    def rerank(self, model: str, query: str, documents: list[str], *, top_n: int | None = None,
               return_documents: bool = False, max_input_tokens: int | None = None,
               truncate: bool | None = None, **options) -> RerankResult:
        """Return typed cross-encoder relevance logits and stable original indices."""
        payload = {**options, 'model': model, 'query': query, 'documents': documents,
                   'top_n': top_n, 'return_documents': return_documents,
                   'max_input_tokens': max_input_tokens, 'truncate': truncate}
        return RerankResult.model_validate(self._json('POST', '/v1/rerank', json=payload))

    def retrieval_answer(self, llm_model, query, documents, *, embedding_model, rerank_model, **options) -> RetrievalAnswerResult:
        """Return source-linked claims or explicit abstention/failure; quotes do not prove entailment."""
        payload={**options,'llm_model':llm_model,'embedding_model':embedding_model,'rerank_model':rerank_model,'query':query,'documents':documents}
        return RetrievalAnswerResult.model_validate(self._json('POST','/v1/retrieval/answer',json=payload))

    def classify_pairs(self, model, pairs, *, max_input_tokens=None, truncate=None, **options) -> PairClassificationResult:
        """Preserve native categorical labels/logits; softmax scores are uncalibrated."""
        payload = {**options, 'model':model, 'pairs':pairs}
        if max_input_tokens is not None:payload['max_input_tokens'] = max_input_tokens
        if truncate is not None:payload['truncate'] = truncate
        return PairClassificationResult.model_validate(self._json('POST','/v1/classifications/text-pairs',json=payload))

    def transcribe(self, model, file, **options):
        """Transcribe audio; optional prompt supplies Whisper context, not output replacement."""
        return self._audio_upload("/v1/audio/transcriptions", model, file, options)

    def vad(self, model, file, **options):
        """Detect speech intervals in a local audio file; each request resets recurrent state."""
        payload = {**options, 'model': model, 'audio_base64': base64.b64encode(Path(file).read_bytes()).decode('ascii')}
        return VADResult.model_validate(self._json('POST', '/v1/audio/vad', json=payload))

    def diarize(self, model, file, **options):
        """Return anonymous request-local speaker intervals, preserving overlap."""
        payload = {**options, 'model': model, 'audio_base64': base64.b64encode(Path(file).read_bytes()).decode('ascii')}
        return DiarizationResult.model_validate(self._json('POST', '/v1/audio/diarizations', json=payload))

    def analyze_audio(self, model, file, *, diarization_model, **options):
        """ASR, anonymous speaker alignment and optional transcript-text sentiment."""
        payload = {**options, 'model': model, 'diarization_model': diarization_model,
                   'audio_base64': base64.b64encode(Path(file).read_bytes()).decode('ascii')}
        for key in ('diarization_options', 'text_analysis_options', 'vad_options'):
            if payload.get(key) is not None and hasattr(payload[key], 'model_dump'):
                payload[key] = payload[key].model_dump(mode='json', exclude_unset=True)
        return AudioAnalysisResult.model_validate(self._json('POST', '/v1/audio/analysis', json=payload))

    def translate_audio(self, model, file, **options):
        result = self._audio_upload("/v1/audio/translations", model, file, options)
        if options.get('include_provenance'):
            from .translation_contract import AudioTranslationResult
            return AudioTranslationResult.model_validate(result)
        return result

    def _audio_upload(self, route, model, file, options):
        path = Path(file)
        options = dict(options)
        if options.get('vad_options') is not None:
            value = options['vad_options']
            if hasattr(value, 'model_dump'):
                value = value.model_dump(mode='json', exclude_none=True)
            options['vad_options'] = json.dumps(value, allow_nan=False)
        if 'timestamp_granularities' in options:
            options['timestamp_granularities[]'] = options.pop('timestamp_granularities')
        with path.open("rb") as source:
            response = self.request("POST", route,
                                    data={k: v for k, v in {**options, "model": model}.items() if v is not None},
                                    files={"file": (path.name, source, mimetypes.guess_type(path.name)[0] or "application/octet-stream")})
        return response.text if options.get("response_format") == "text" else response.json()

    def translate(self, model, text, target_language, **options):
        result = self._json("POST", "/v1/translations", json={**options, "model": model, "text": text, "target_language": target_language})
        if options.get('include_provenance'):
            from .translation_contract import TextTranslationResult
            return TextTranslationResult.model_validate(result)
        return result

    def sentiment(self, model, text, **options):
        """Classify expressed text sentiment; returned scores are not calibrated probabilities."""
        return self._json('POST', '/v1/sentiment', json={**options, 'model': model, 'text': text})

    def compose_sentiment(self, classifier_model, llm_model, text, **options):
        """Require agreement while retaining both stage results, errors and abstentions."""
        return self._json('POST', '/v1/sentiment/compose', json={**options,
            'classifier_model': classifier_model, 'llm_model': llm_model, 'text': text})

    def ocr(self, model, image, **options):
        return self._json("POST", "/v1/images/ocr", json={**options, "model": model, "image_data_url": image_part(image)["image_url"]["url"]})

    def depth(self, model, image, **options):
        from .depth import DepthResult
        return DepthResult.model_validate(self._json('POST', '/v1/images/depth',
            json={**options, 'model': model, 'image_data_url': image_part(image)['image_url']['url']}))

    def segment(self, model, image, *, box=None, points=None, point_labels=None,
                multimask=None, mask_threshold=None, **options):
        """Return aligned RGBA edit masks from a box or labeled points in original image coordinates."""
        from .segmentation import SegmentationResult
        payload = {**options, 'model': model, 'image_data_url': image_part(image)['image_url']['url']}
        payload.update({key: value for key, value in {
            'box': box, 'points': points, 'point_labels': point_labels,
            'multimask': multimask, 'mask_threshold': mask_threshold}.items() if value is not None})
        return SegmentationResult.model_validate(self._json('POST', '/v1/images/segment', json=payload))

    def detect(self, model, image, **options):
        return self._json("POST", "/v1/images/detect", json={**options, "model": model, "image_data_url": image_part(image)["image_url"]["url"]})

    def extension(self, operation, model, **options):
        field = 'classifier_model' if operation == 'sentiment_compose' else 'llm_model' if operation=='retrieval_answer' else 'model'
        return self._json("POST", "/v1/extensions/infer", json={**options, "operation": operation, field: model})

    def speech(self, model, input, **options):
        return self.request("POST", "/v1/audio/speech", json={**options, "model": model, "input": input}).content

    def image(self, model, prompt, **options):
        return self._json("POST", "/v1/images/generations", json={**options, "model": model, "prompt": prompt})

    def edit(self, model, image, prompt, *, mask=None, mask_base64=None, preserve_unmasked=None, image_guidance_scale=None, **options):
        """Edit an image; image_guidance_scale is specific to instruction-edit models."""
        if mask is not None and mask_base64 is not None:
            raise ValueError("Use mask or mask_base64, not both")
        path = Path(image)
        with ExitStack() as stack:
            source = stack.enter_context(path.open("rb"))
            files = {"image": (path.name, source, mimetypes.guess_type(path.name)[0] or "application/octet-stream")}
            if mask is not None:
                mask_path = Path(mask)
                files["mask"] = (mask_path.name, stack.enter_context(mask_path.open("rb")), "image/png")
            return self._json("POST", "/v1/images/edits",
                              data={k: v for k, v in {**options, "model": model, "prompt": prompt,
                                    "mask_base64": mask_base64, "preserve_unmasked": preserve_unmasked,
                                    "image_guidance_scale":image_guidance_scale}.items() if v is not None},
                              files=files)

    def video(self, model, prompt, **options):
        return self._json("POST", "/v1/videos/generations", json={**options, "model": model, "prompt": prompt})

    def submit_job(self, task, model, *, idempotency_key=None, **options):
        if task not in {"image", "speech", "video", "music"}:
            raise ValueError("Asset jobs support image, speech, video or music")
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        return self._json("POST", "/api/jobs", json={**options, "task": task, "model": model}, headers=headers)

    def submit_video(self, model, prompt, *, idempotency_key=None, **options):
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        return self._json("POST", "/api/video/jobs", json={**options, "model": model, "prompt": prompt}, headers=headers)

    def submit_music(self,model,prompt,*,idempotency_key=None,**options):
        return self.submit_job("music",model,idempotency_key=idempotency_key,prompt=prompt,**options)

    def jobs(self):
        return self._json("GET", "/api/jobs")

    def job(self, identifier, **kwargs):
        return self._json("GET", "/api/jobs/" + quote(identifier, safe=""), **kwargs)

    def cancel_job(self, identifier):
        return self._json("DELETE", "/api/jobs/" + quote(identifier, safe=""))

    def wait_job(self, identifier, *, timeout=None, poll_interval=1, poll_retries=3):
        """Retry transient status-read disconnects without resubmitting inference."""
        if timeout is None:
            from .settings import load_settings
            timeout = load_settings().client_timeout
        if not all(math.isfinite(v) and v > 0 for v in (timeout, poll_interval)):
            raise ValueError("Job timeout and poll interval must be finite and positive")
        if not isinstance(poll_retries, int) or isinstance(poll_retries, bool) or poll_retries < 0:
            raise ValueError("poll_retries must be a nonnegative integer")
        deadline = time.monotonic() + timeout
        failures = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Job {identifier} did not finish within {timeout:g}s; it has not been cancelled")
            try:
                record = self.job(identifier, timeout=remaining)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Job {identifier} did not finish within {timeout:g}s; it has not been cancelled")
                failures = 0
            except TitanosError as exc:
                if not isinstance(exc.__cause__, (httpx.NetworkError, httpx.RemoteProtocolError)) or failures >= poll_retries:
                    raise
                failures += 1
                record = None
            if record and record["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return record
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))

    def save_asset(self, asset, destination):
        """Atomically save bytes or one result's b64_json/url, preserving on error.

        Authenticated downloads are restricted to this server's origin and never
        follow redirects. Call this once for each item in an image result's data.
        """
        dest = Path(destination)
        url, content = None, None
        if isinstance(asset, bytes):
            content = asset
        elif "b64_json" in asset:
            content = base64.b64decode(asset["b64_json"], validate=True)
        elif "url" in asset:
            url = urljoin(self.base_url + "/", asset["url"])
            if _origin(url) != self.origin:
                raise ValueError("Artifact URL must belong to the configured Titanos server origin")
        else:
            raise ValueError("Asset must be bytes or contain b64_json or url")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".titanos-", dir=dest.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                if url:
                    try:
                        with self.http.stream("GET", url) as response:
                            _check(response)
                            for chunk in response.iter_bytes():
                                output.write(chunk)
                    except httpx.HTTPError as exc:
                        raise TitanosError(f"Asset download failed: {exc}") from exc
                else:
                    output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, dest)
        finally:
            Path(temp).unlink(missing_ok=True)
        return dest

    # -- protocol surfaces (Anthropic, OpenAI moderations and Responses handles, Gemini, ElevenLabs) -------------------
    def messages(self, model, messages, *, max_tokens, stream=False, **options):
        """Anthropic Messages. stream=True yields the SSE event objects and ends after message_stop."""
        payload = {**options, "model": model, "messages": messages, "max_tokens": max_tokens, "stream": stream}
        if stream:
            return self._sse_events("/v1/messages", payload, terminal="message_stop")
        return self._json("POST", "/v1/messages", json=payload)

    def count_tokens(self, model, messages, **options):
        """Anthropic count_tokens: an Anthropic upstream, or a local llama-cpp model counted natively."""
        return self._json("POST", "/v1/messages/count_tokens", json={**options, "model": model, "messages": messages})

    def moderations(self, model, input):
        return self._json("POST", "/v1/moderations", json={"model": model, "input": input})

    def response(self, identifier):
        """A stored Responses object (handle-bound: only the key that created it, or an admin)."""
        return self._json("GET", "/v1/responses/" + quote(identifier, safe=""))

    def delete_response(self, identifier):
        return self._json("DELETE", "/v1/responses/" + quote(identifier, safe=""))

    def cancel_response(self, identifier):
        return self._json("POST", "/v1/responses/" + quote(identifier, safe="") + "/cancel")

    def response_input_items(self, identifier):
        return self._json("GET", "/v1/responses/" + quote(identifier, safe="") + "/input_items")

    def gemini_generate(self, model, contents, *, stream=False, **options):
        """Gemini generateContent; stream=True uses streamGenerateContent with alt=sse and ends at the end of the stream."""
        payload = {**options, "contents": contents}
        if stream:
            action = quote(model, safe="/") + ":streamGenerateContent"
            return self._sse_events(f"/v1beta/models/{action}", payload, terminal=None, params={"alt": "sse"})
        action = quote(model, safe="/") + ":generateContent"
        return self._json("POST", f"/v1beta/models/{action}", json=payload)

    def gemini_embed(self, model, content, *, batch=False):
        """Gemini embedContent, or batchEmbedContents for a list when batch=True. A string becomes one text part."""
        def part(value):
            return {"parts": [{"text": value}]} if isinstance(value, str) else value
        if batch:
            items = content if isinstance(content, (list, tuple)) else [content]
            name = model if model.startswith("models/") else "models/" + model
            action = quote(model, safe="/") + ":batchEmbedContents"
            return self._json("POST", f"/v1beta/models/{action}",
                              json={"requests": [{"model": name, "content": part(item)} for item in items]})
        action = quote(model, safe="/") + ":embedContent"
        return self._json("POST", f"/v1beta/models/{action}", json={"content": part(content)})

    def gemini_count_tokens(self, model, contents):
        action = quote(model, safe="/") + ":countTokens"
        return self._json("POST", f"/v1beta/models/{action}", json={"contents": contents})

    def elevenlabs_speech(self, voice_id, text, *, model_id=None, output_format="mp3_44100_128", stream=False, **options):
        """ElevenLabs text-to-speech: the audio bytes, or an iterator of byte chunks with stream=True."""
        payload = {**options, "text": text, **({"model_id": model_id} if model_id is not None else {})}
        params = {"output_format": output_format} if output_format else None
        if stream:
            return self._byte_stream("/v1/text-to-speech/" + quote(voice_id, safe="") + "/stream", payload, params=params)
        return self.request("POST", "/v1/text-to-speech/" + quote(voice_id, safe=""), json=payload, params=params).content

    def speech_to_text(self, model, file, **options):
        """ElevenLabs speech-to-text (multipart model_id and file)."""
        path = Path(file)
        with path.open("rb") as source:
            return self.request("POST", "/v1/speech-to-text",
                                data={k: v for k, v in {**options, "model_id": model}.items() if v is not None},
                                files={"file": (path.name, source, mimetypes.guess_type(path.name)[0] or "application/octet-stream")}).json()

    def voices(self):
        return self._json("GET", "/v1/voices")

    # -- broker status and observability -------------------------------------------------------------------------------
    def broker_status(self):
        return self._json("GET", "/api/broker")

    def broker_profiles(self):
        return self._json("GET", "/api/broker/profiles")

    def broker_catalog(self, refresh=False):
        return self._json("GET", "/api/broker/catalog", params={"refresh": "true" if refresh else "false"})

    def broker_explain(self, surface, body, headers=None, key_id=None):
        """Dry-run routing decision (zero egress); key_id (admin only) explains for another key."""
        return self._json("POST", "/api/broker/explain", json={"surface": surface, "body": body, "headers": headers or {},
                                                               **({"key_id": key_id} if key_id is not None else {})})

    def broker_env_candidates(self):
        """Which well-known provider credential variable NAMES are set in the server process (names and booleans only,
        never values)."""
        return self._json("GET", "/api/broker/env-candidates")

    def broker_connect(self, provider, *, secret=None, env=None, file=None, upstream=None, credential_id=None,
                       models=None, base_url=None, enable=True, validate=True, aliases=False, anonymous_cloud=None,
                       allow_store=None):
        """One-step provider setup: create or update the upstream from its profile, store and validate the credential
        (exactly one of secret, env (a variable name on the server host) or file; none only for profiles without auth),
        enable it, raise the cloud policy from off to explicit and return the discovered models (never a secret)."""
        body = {"provider": provider, "enable": enable, "validate": validate, "aliases": aliases}
        if any(value is not None for value in (secret, env, file)):
            self._secure_transport()
            source = self._credential_source(secret, env, file)
            body["credential"] = {("value" if name == "secret" else name): value for name, value in source.items()}
        for name, value in (("upstream", upstream), ("credential_id", credential_id), ("models", models),
                            ("base_url", base_url), ("anonymous_cloud", anonymous_cloud), ("allow_store", allow_store)):
            if value is not None:
                body[name] = list(value) if name == "models" else value
        return self._json("POST", "/api/broker/connect", json=body)

    def broker_decisions(self, limit=100, **filters):
        return self._json("GET", "/api/broker/decisions",
                          params={"limit": limit, **{k: v for k, v in filters.items() if v is not None}})

    def broker_usage(self, group_by=("key", "upstream", "model", "day"), since=None, until=None, key_id=None):
        groups = group_by if isinstance(group_by, str) else ",".join(group_by)
        params = {k: v for k, v in {"group_by": groups, "since": since, "until": until, "key_id": key_id}.items() if v is not None}
        return self._json("GET", "/api/broker/usage", params=params)

    def broker_metrics(self):
        return self._json("GET", "/api/broker/metrics")

    def broker_reload(self):
        return self._json("POST", "/api/broker/reload")

    # -- upstreams ------------------------------------------------------------------------------------------------------
    def upstreams(self):
        return self._json("GET", "/api/broker/upstreams")

    def upstream(self, identifier):
        return self._json("GET", "/api/broker/upstreams/" + quote(identifier, safe=""))

    def put_upstream(self, identifier, **spec):
        """Create or replace an upstream (broker.json Upstream fields; credential names an existing credential id)."""
        return self._json("PUT", "/api/broker/upstreams/" + quote(identifier, safe=""), json=spec)

    def delete_upstream(self, identifier):
        return self._json("DELETE", "/api/broker/upstreams/" + quote(identifier, safe=""))

    def test_upstream(self, identifier):
        return self._json("POST", "/api/broker/upstreams/" + quote(identifier, safe="") + "/test")

    def upstream_health(self, identifier):
        return self._json("GET", "/api/broker/upstreams/" + quote(identifier, safe="") + "/health")

    def reset_breaker(self, identifier):
        return self._json("POST", "/api/broker/upstreams/" + quote(identifier, safe="") + "/breaker/reset")

    # -- credentials (write-only secrets) -------------------------------------------------------------------------------
    def _secure_transport(self):
        """Credential writes go over TLS or to a loopback server only (the server refuses them otherwise too)."""
        scheme, host, _ = self.origin
        if scheme != "https" and not _loopback(host):
            raise ValueError("Credential writes need an https server URL or a loopback server")

    @staticmethod
    def _credential_source(secret, env, file):
        sources = {k: v for k, v in {"secret": secret, "env": env, "file": file}.items() if v is not None}
        if len(sources) != 1:
            raise ValueError("Give exactly one of secret, env or file")
        return sources

    def credentials(self):
        """Credential metadata and fingerprints only; values are never returned."""
        return self._json("GET", "/api/broker/credentials")

    def set_credential(self, identifier, *, upstream, secret=None, env=None, file=None, purpose="inference", validate=True):
        """Store a credential bound to one upstream: exactly one of secret (sent once in the JSON body, never kept on this
        Client), env (a variable name on the server host) or file (an owner-only path on the server host). Never log
        the secret. Refused before sending for an http server URL that is not loopback."""
        self._secure_transport()
        body = {"upstream": upstream, **self._credential_source(secret, env, file), "purpose": purpose, "validate": validate}
        return self._json("PUT", "/api/broker/credentials/" + quote(identifier, safe=""), json=body)

    def rotate_credential(self, identifier, *, secret=None, env=None, file=None):
        """Replace a credential's value or reference (validated before it serves). Never log the secret."""
        self._secure_transport()
        return self._json("POST", "/api/broker/credentials/" + quote(identifier, safe="") + "/rotate",
                          json=self._credential_source(secret, env, file))

    def validate_credential(self, identifier):
        self._secure_transport()
        return self._json("POST", "/api/broker/credentials/" + quote(identifier, safe="") + "/validate")

    def delete_credential(self, identifier):
        self._secure_transport()
        return self._json("DELETE", "/api/broker/credentials/" + quote(identifier, safe=""))

    # -- published aliases, virtual models, policy, advisors and prices ------------------------------------------------
    def aliases(self):
        return self._json("GET", "/api/broker/models")

    def put_alias(self, name, target, capabilities=(), inject=None):
        body = {"target": target, **({"capabilities": list(capabilities)} if capabilities else {}),
                **({"inject": inject} if inject else {})}
        return self._json("PUT", "/api/broker/models/" + quote(name, safe="/"), json=body)

    def delete_alias(self, name):
        return self._json("DELETE", "/api/broker/models/" + quote(name, safe="/"))

    def virtual_models(self):
        return self._json("GET", "/api/broker/virtual-models")

    def put_virtual_model(self, name, **spec):
        return self._json("PUT", "/api/broker/virtual-models/" + quote(name, safe="/"), json=spec)

    def delete_virtual_model(self, name):
        return self._json("DELETE", "/api/broker/virtual-models/" + quote(name, safe="/"))

    def policy(self):
        return self._json("GET", "/api/broker/policy")

    def set_policy(self, **fields):
        """Change the named policy fields; the others keep their current values (read first, then PUT the whole policy)."""
        current = self._json("GET", "/api/broker/policy")
        names = ("cloud_policy", "default_location", "anonymous_cloud", "allow_http_lan", "max_fallbacks", "auto_model")
        unknown = set(fields) - set(names)
        if unknown:
            raise ValueError("Unknown policy fields: " + ", ".join(sorted(unknown)))
        body = {name: current.get(name) for name in names if name in current}
        body.update(fields)
        return self._json("PUT", "/api/broker/policy", json=body)

    def advisors(self):
        return self._json("GET", "/api/broker/advisors")

    def put_advisor(self, identifier, **spec):
        return self._json("PUT", "/api/broker/advisors/" + quote(identifier, safe=""), json=spec)

    def test_advisor(self, identifier):
        return self._json("POST", "/api/broker/advisors/" + quote(identifier, safe="") + "/test")

    def prices(self):
        return self._json("GET", "/api/broker/prices")

    def put_prices(self, rows):
        """Replace the operator price rows (Decimal strings); Titanos ships no prices."""
        return self._json("PUT", "/api/broker/prices", json=list(rows))

    # -- inbound keys and audit -----------------------------------------------------------------------------------------
    def keys(self):
        return self._json("GET", "/api/keys")

    def create_key(self, **spec):
        """Mint a key. The returned CreatedKey.secret is shown once; store it immediately and never log it."""
        value = self._json("POST", "/api/keys", json=spec)
        secret = value.pop("secret")
        return CreatedKey(value["id"], secret, value)

    def key(self, identifier):
        return self._json("GET", "/api/keys/" + quote(identifier, safe=""))

    def update_key(self, identifier, **patch):
        return self._json("PATCH", "/api/keys/" + quote(identifier, safe=""), json=patch)

    def revoke_key(self, identifier):
        return self._json("POST", "/api/keys/" + quote(identifier, safe="") + "/revoke")

    def rotate_key(self, identifier):
        """A new secret for the same key id (the old one stops working at once); shown once."""
        value = self._json("POST", "/api/keys/" + quote(identifier, safe="") + "/rotate")
        secret = value.pop("secret")
        return CreatedKey(value["id"], secret, value)

    def whoami(self):
        return self._json("GET", "/api/keys/self")

    def audit(self, **filters):
        return self._json("GET", "/api/audit", params={k: v for k, v in filters.items() if v is not None})

    def audit_verify(self):
        return self._json("GET", "/api/audit/verify")


def image_part(file):
    """Encode a local image as the inline image part accepted by vision chat."""
    path = Path(file)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if not mime.startswith("image/"):
        raise ValueError("Vision input must be an image file")
    return {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}
