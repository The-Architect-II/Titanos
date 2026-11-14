"""Standalone implementations of HiveMind's reusable extended inference shapes.

Only wire conventions are shared. Every operation runs a registered local model.
"""

import asyncio
import base64
import binascii
import hashlib
import json
import math
import threading
import time
from typing import Annotated, Literal

import anyio
from fastapi import File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .engine import EngineError
from .settings import current_settings as load_settings
from .sentiment_contract import (SentimentResult, SentimentInput, SentimentComposition, SentimentStage, SentimentStageError,
                                 llm_instruction, parse_llm_output, build_result, compose_results, sentiment_json_schema)
from .retrieval_contract import RerankResult
from .retrieval_workflow_contract import RetrievalAnswerRequest, RetrievalAnswerResult
from .audio_contract import (VADResult, ASRMetadata, TranscriptionText, ASRVADGate, DiarizationOptions,
                             DiarizationResult, AudioAnalysisResult, AttributedASRSegment,
                             timestamp_attribution, attribution_status)
from .audio_gating import ASRGateFields, VADGateOptions, resolve_gate
from .translation_contract import (AudioTranslationResult, TextTranslationResult, TranslationStage,
                                   NMTProvenance, NMTMeasurements, LanguageSelection, text_digest)
from .pair_contract import TextPair, PairClassificationResult, validate_pairs, text_hash as pair_text_hash, pair_hash
from .depth import DepthResult
from .segmentation import SegmentationPrompt, SegmentationResult
from .tesseract import OCRControls, CONTROL_NAMES as OCR_CONTROL_NAMES, controls as ocr_controls


class ExtensionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=128)
    worker: str | None = Field(default=None, max_length=128)
    timeout: int | None = Field(default=None, gt=0, description="Inference deadline in milliseconds, including time waiting for the model")
    location: Literal["local_only", "auto", "prefer_local", "cloud_only"] = "local_only"


class TranslationRequest(ExtensionPayload):
    text: str = Field(min_length=1)
    source_language: str = Field(default="eng_Latn", min_length=1, max_length=64)
    target_language: str = Field(min_length=1, max_length=64)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    include_provenance: bool = Field(default=False, strict=True, description='Return the versioned attributed translation result; ordinary output is unchanged when omitted')

    @model_validator(mode="after")
    def bounded(self):
        settings = load_settings()
        if len(self.text) > settings.max_text_chars:
            raise ValueError("text exceeds TITANOS_MAX_TEXT_CHARS")
        try:
            for value in (self.text, self.source_language, self.target_language):value.encode('utf-8')
        except UnicodeEncodeError as exc:
            raise ValueError('Translation text and language IDs must be valid Unicode scalar text') from exc
        if self.max_tokens is not None and self.max_tokens > settings.max_tokens:
            raise ValueError("max_tokens exceeds TITANOS_MAX_TOKENS")
        return self


class PairClassificationRequest(ExtensionPayload):
    pairs: list[TextPair] = Field(min_length=1)
    max_input_tokens: int | None = Field(default=None, strict=True, ge=1)
    truncate: bool | None = Field(default=None, strict=True)

    @model_validator(mode='after')
    def bounded(self):
        validate_pairs(self.pairs)
        if self.max_input_tokens is not None and self.max_input_tokens > load_settings().pair_max_input_tokens:
            raise ValueError('max_input_tokens exceeds TITANOS_PAIR_MAX_INPUT_TOKENS')
        return self


class SentimentRequest(ExtensionPayload):
    text: str = Field(min_length=1)
    input_modality: Literal['text', 'asr_transcript'] = 'text'
    language: str | None = Field(default=None, min_length=1, max_length=64)
    asr_uncertain: bool = Field(default=False, strict=True, description='Caller-declared uncertainty; false does not certify ASR quality')
    min_score: float | None = Field(default=None, strict=True, ge=0, le=1)
    min_margin: float | None = Field(default=None, strict=True, ge=0, le=1)
    max_input_tokens: int | None = Field(default=None, strict=True, ge=1)
    max_tokens: int | None = Field(default=None, strict=True, ge=1)
    temperature: float | None = Field(default=None, strict=True, ge=0, le=2)
    top_p: float | None = Field(default=None, strict=True, gt=0, le=1)
    prompt_version: Literal['weights-v1', 'label-v2', 'label-v3', 'facets-v1'] | None = None
    decoding: Literal['auto', 'validate', 'label_choices', 'facet_choices', 'native_json_schema'] | None = None

    @model_validator(mode='after')
    def bounded(self):
        settings = load_settings()
        if not self.text.strip() or len(self.text) > settings.max_text_chars:
            raise ValueError('text must be nonblank and within TITANOS_MAX_TEXT_CHARS')
        if self.asr_uncertain and self.input_modality != 'asr_transcript':
            raise ValueError('asr_uncertain applies to an explicitly marked ASR transcript')
        if self.max_input_tokens is not None and self.max_input_tokens > settings.sentiment_max_input_tokens:
            raise ValueError('max_input_tokens exceeds TITANOS_SENTIMENT_MAX_INPUT_TOKENS')
        if self.max_tokens is not None and self.max_tokens > settings.max_tokens:
            raise ValueError('max_tokens exceeds TITANOS_MAX_TOKENS')
        return self


class SentimentCompositionRequest(SentimentRequest):
    model: str = Field(alias='classifier_model', min_length=1, max_length=128)
    llm_model: str = Field(min_length=1, max_length=128)
    llm_worker: str | None = Field(default=None, max_length=128)
    policy: Literal['require_agreement'] = 'require_agreement'


class RerankRequest(ExtensionPayload):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)
    top_n: int | None = Field(default=None, strict=True, ge=1)
    return_documents: bool = Field(default=False, strict=True)
    max_input_tokens: int | None = Field(default=None, strict=True, ge=1)
    truncate: bool | None = Field(default=None, strict=True)

    @model_validator(mode='after')
    def bounded(self):
        settings = load_settings()
        if not self.query.strip() or len(self.query) > settings.max_prompt_chars:
            raise ValueError('query must be nonblank and within TITANOS_MAX_PROMPT_CHARS')
        if len(self.documents) > settings.rerank_max_documents:
            raise ValueError('documents exceeds TITANOS_RERANK_MAX_DOCUMENTS')
        if any(not doc.strip() or len(doc) > settings.max_text_chars for doc in self.documents):
            raise ValueError('documents must be nonblank and within TITANOS_MAX_TEXT_CHARS')
        if self.max_input_tokens is not None and self.max_input_tokens > settings.rerank_max_input_tokens:
            raise ValueError('max_input_tokens exceeds TITANOS_RERANK_MAX_INPUT_TOKENS')
        return self


Byte = Annotated[int, Field(strict=True, ge=0, le=255)]


class ImageInput(ExtensionPayload):
    image_base64: str | None = None
    image_data_url: str | None = None
    image: str | None = None
    image_buffer: str | list[list[Byte]] | None = None

    @model_validator(mode="after")
    def one_image(self):
        fields = [self.image_base64, self.image_data_url, self.image, self.image_buffer]
        if sum(value is not None for value in fields) != 1:
            raise ValueError("Provide exactly one inline image: image_base64, image_data_url, image, or image_buffer")
        if isinstance(self.image_buffer, list) and (len(self.image_buffer) != 1 or not self.image_buffer[0]):
            raise ValueError("image_buffer must contain exactly one nonempty encoded image byte array")
        return self


class OCRRequest(ImageInput, OCRControls):
    prompt: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def bounded(self):
        settings = load_settings()
        if self.prompt is not None and len(self.prompt) > settings.max_prompt_chars:
            raise ValueError("prompt exceeds TITANOS_MAX_PROMPT_CHARS")
        if self.max_tokens is not None and self.max_tokens > settings.max_tokens:
            raise ValueError("max_tokens exceeds TITANOS_MAX_TOKENS")
        return self


class DepthRequest(ImageInput):
    pass


class SegmentationRequest(ImageInput,SegmentationPrompt):
    pass


class DetectionRequest(ImageInput):
    threshold: float | None = Field(default=None, ge=0, le=1, description='Native detection score threshold; omitted inherits model then process policy')
    nms: bool | None = Field(default=None, strict=True, description='Enable class-aware filtering of overlapping boxes; false overrides an enabled default')
    nms_iou: float | None = Field(default=None, strict=True, ge=0, le=1, description='IoU above which lower-score same-class boxes are suppressed when NMS is enabled; zero is valid')


class ASRRequest(ExtensionPayload, ASRGateFields):
    audio_base64: str = Field(min_length=1)
    prompt: str | None = Field(default=None,min_length=1,strict=True)
    language: str | None = Field(default=None, max_length=64)
    max_tokens: int | None = Field(default=None, gt=0)
    response_format: Literal['json', 'verbose_json'] = 'json'
    timestamp_granularities: list[Literal['segment', 'word']] = Field(default_factory=list, max_length=1)

    @model_validator(mode='after')
    def bounded_audio(self):
        if self.prompt is not None and (not self.prompt.strip() or len(self.prompt)>load_settings().max_prompt_chars):
            raise ValueError('ASR prompt must be nonblank and within TITANOS_MAX_PROMPT_CHARS')
        if self.max_tokens is not None and self.max_tokens > load_settings().max_tokens:
            raise ValueError('max_tokens exceeds TITANOS_MAX_TOKENS')
        if self.timestamp_granularities and self.response_format != 'verbose_json':
            raise ValueError('Timestamps require verbose_json')
        return self


class DiarizationRequest(ExtensionPayload, DiarizationOptions):
    audio_base64: str = Field(min_length=1)


class TranscriptAnalysisOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    asr_uncertain: bool = Field(default=True, strict=True, description='Transcript uncertainty, not a claim about tone or a measured ASR confidence')
    min_score: float | None = Field(default=None, strict=True, ge=0, le=1)
    min_margin: float | None = Field(default=None, strict=True, ge=0, le=1)
    max_input_tokens: int | None = Field(default=None, strict=True, ge=1)
    max_tokens: int | None = Field(default=None, strict=True, ge=1)
    temperature: float | None = Field(default=None, strict=True, ge=0, le=2)
    top_p: float | None = Field(default=None, strict=True, gt=0, le=1)
    prompt_version: Literal['weights-v1', 'label-v2', 'label-v3', 'facets-v1'] | None = None
    decoding: Literal['auto', 'validate', 'label_choices', 'facet_choices', 'native_json_schema'] | None = None


class AudioAnalysisRequest(ASRRequest):
    max_tokens: int | None = Field(default=None, strict=True, gt=0)
    response_format: Literal['verbose_json'] = 'verbose_json'
    diarization_model: str = Field(min_length=1, max_length=128)
    diarization_worker: str | None = Field(default=None, min_length=1, max_length=128)
    diarization_options: DiarizationOptions = Field(default_factory=DiarizationOptions)
    text_analysis_model: str | None = Field(default=None, min_length=1, max_length=128)
    text_analysis_worker: str | None = Field(default=None, min_length=1, max_length=128)
    text_analysis_options: TranscriptAnalysisOptions | None = None

    @model_validator(mode='after')
    def stage_options(self):
        if self.text_analysis_model is None and (self.text_analysis_worker is not None or self.text_analysis_options is not None):
            raise ValueError('text_analysis_worker/options require text_analysis_model')
        return self


class VADRequest(ExtensionPayload):
    audio_base64: str = Field(min_length=1)
    sample_rate: Literal[8000, 16000] | None = None
    threshold: float | None = Field(default=None, strict=True, gt=0, lt=1)
    min_speech_ms: float | None = Field(default=None, strict=True, ge=0)
    min_silence_ms: float | None = Field(default=None, strict=True, gt=0)
    speech_pad_ms: float | None = Field(default=None, strict=True, ge=0)
    return_frames: bool = Field(default=False, strict=True)


class AudioTranslationRequest(ExtensionPayload, ASRGateFields):
    audio_base64: str = Field(min_length=1)
    language: str | None = Field(default=None, max_length=64)
    translation_model: str | None = Field(default=None, max_length=128)
    translation_worker: str | None = Field(default=None, max_length=128)
    source_language: str | None = Field(default=None, max_length=64, description="NMT source language ID for a composed ASR + NMT translation")
    target_language: str | None = Field(default=None, max_length=64, description="English language ID used by the selected NMT tokenizer; omitted inherits the translation model default, then eng_Latn")
    max_tokens: int | None = Field(default=None, gt=0)
    response_format: Literal['json', 'verbose_json'] = 'json'
    include_provenance: bool = Field(default=False, strict=True, description='Return versioned ASR/NMT stages; observed ASR language is distinct from caller NMT tokenizer IDs')

    @model_validator(mode="after")
    def english_only(self):
        if self.target_language is not None and self.target_language not in {"eng_Latn", "en", "en_XX", "eng", "English"}:
            raise ValueError("Audio translations produce English; use text translations for other targets")
        if self.translation_model and not self.source_language:
            raise ValueError("source_language is required with translation_model; ASR language hints and NMT language IDs can differ")
        if self.translation_worker and not self.translation_model:
            raise ValueError("translation_worker requires translation_model")
        if self.max_tokens is not None and self.max_tokens > load_settings().max_tokens:
            raise ValueError("max_tokens exceeds TITANOS_MAX_TOKENS")
        return self


class TranslationOperation(TranslationRequest):
    operation: Literal["translation"]


class OCROperation(OCRRequest):
    operation: Literal["ocr"]


class DepthOperation(DepthRequest):
    operation: Literal['depth']


class SegmentationOperation(SegmentationRequest):
    operation: Literal['segmentation']


class DetectionOperation(DetectionRequest):
    operation: Literal["detection"]


class AudioTranslationOperation(AudioTranslationRequest):
    operation: Literal["audio_translation"]


class ASROperation(ASRRequest):
    operation: Literal['audio_transcription']


class VADOperation(VADRequest):
    operation: Literal['vad']


class DiarizationOperation(DiarizationRequest):
    operation: Literal['diarization']


class AudioAnalysisOperation(AudioAnalysisRequest):
    operation: Literal['audio_analysis']


class SentimentOperation(SentimentRequest):
    operation: Literal['sentiment']


class SentimentCompositionOperation(SentimentCompositionRequest):
    operation: Literal['sentiment_compose']


class RerankOperation(RerankRequest):
    operation: Literal['rerank']


class PairClassificationOperation(PairClassificationRequest):
    operation: Literal['pair_classification']


class RetrievalAnswerOperation(RetrievalAnswerRequest):
    operation: Literal['retrieval_answer']


ExtensionRequest = Annotated[TranslationOperation | OCROperation | DetectionOperation | DepthOperation | SegmentationOperation | AudioTranslationOperation | ASROperation | VADOperation | DiarizationOperation | AudioAnalysisOperation | SentimentOperation | SentimentCompositionOperation | RerankOperation | RetrievalAnswerOperation | PairClassificationOperation,
                             Field(discriminator="operation")]


def _inline_bytes(value, media):
    if value.startswith("data:"):
        if not value.startswith("data:" + media + "/") or ";base64," not in value:
            raise EngineError(f"Expected an inline {media} base64 data URL")
        value = value.split(";base64,", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise EngineError("Invalid inline base64 media") from exc
    maximum = load_settings().max_input_image_bytes if media == "image" else load_settings().max_body_bytes
    if not raw or len(raw) > maximum:
        raise EngineError("Inline media is empty or exceeds the configured byte limit", 413, "request_too_large")
    return raw


def _image(payload, size_check=None):
    from .backends import image_input
    value = next(value for value in (payload.image_base64, payload.image_data_url, payload.image, payload.image_buffer) if value is not None)
    raw = bytes(value[0]) if isinstance(value, list) else _inline_bytes(value, "image")
    try:
        url = "data:image/png;base64," + base64.b64encode(raw).decode()
        return image_input(url) if size_check is None else image_input(url, size_check=size_check)
    except ValueError as exc:
        raise EngineError(str(exc)) from exc


CLOUD_RUN_TASKS = frozenset({'translation', 'transcription', 'chat'})


def register_api_plus(app, engine, config, chat_defaults=None, native_gateway=None, broker=None):
    """Attach extensions to Titanos' existing authentication/error/lifecycle stack.

    With a broker, cloud-eligible stages resolve their model through broker.StageRouter (placement from body.location,
    default local_only; key scopes and constraints, the cloud policy and the role apply): text translation (cloud chat
    with the versioned broker-translation-v1 prompt), ASR after the local VAD gate, and the LLM of sentiment and
    sentiment_compose (through GatewayRouter, so decoding resolves to validate). A cloud target runs through
    CloudInvoker on the stage's worker thread, never inside Engine.lock. Local-only stages (OCR, depth, detection,
    segmentation, VAD, diarization, rerank, pair classification, embeddings) refuse cloud_only with 400
    broker_capability_unavailable; include_provenance with a cloud target is 400 provenance_unavailable_for_cloud, and a
    retrieval answer needs a local llama-cpp model (exact native context accounting). While the broker is inactive
    (no broker.json and a keyless or bootstrap caller) every stage runs exactly as before."""
    from .broker.errors import BrokerError

    def current_principal():
        from .broker.context import current_principal as principal
        return principal()

    def stage_for(model, task, location):
        """The cloud StageTarget for (model, task) at location, or None when the stage runs the existing local path."""
        location = location or 'local_only'
        from .broker.stages import CLOUD_TASKS
        if task not in CLOUD_TASKS:
            if location == 'cloud_only':
                raise BrokerError(f'{task} stages run on local models only', 'broker_capability_unavailable')
            return None
        principal = current_principal()
        if broker is None or not broker.active(principal):
            if location == 'cloud_only':
                raise BrokerError('No cloud target is configured', 'location_unsatisfiable', origin='policy')
            return None
        stage = broker.stage_router().resolve_stage(model, task, location=location, principal=principal)
        return stage if stage.kind == 'cloud' else None

    def spec_of(model, task, location):
        stage = stage_for(model, task, location)
        return (stage.spec if stage is not None else config.models.get(model)), stage

    def gateway(location, principal=None):
        """NativePool for the stage, or broker.GatewayRouter (same signature) while the broker is active."""
        native = native_gateway() if native_gateway is not None else None
        if broker is not None:
            principal = principal if principal is not None else current_principal()
            if broker.active(principal):
                return broker.gateway(native, principal, location or 'local_only')
        if native is None:
            raise EngineError('Native gateway is unavailable', 503, 'worker_unavailable')
        return native

    def cloud_run(stage, task, values, cancelled, deadline):
        """A cloud stage on this worker thread (CloudInvoker.infer_sync), shaped like the local engine result."""
        data = {k: v for k, v in values.items() if not str(k).startswith('_')
                and k not in {'worker', 'location', 'timeout', 'require_loaded', 'include_provenance'}}
        if task == 'transcription':
            data = {'audio': values.get('audio'), 'language': values.get('language'),
                    'task': 'translate' if values.get('translation') else 'transcribe'}
        result = broker.invoker().infer_sync(stage.target, task, data, cancel=cancelled, deadline=deadline,
                                             principal=current_principal())
        if task == 'transcription' and values.get('response_format') != 'verbose_json':
            return result['text']
        return result

    def run(payload, task, values, caller_cancel=None, deadline=None):
        milliseconds = payload.timeout or int(load_settings().client_timeout * 1000)
        if deadline is not None:
            milliseconds = min(milliseconds, math.ceil((deadline - time.monotonic()) * 1000))
        if milliseconds <= 0:
            raise EngineError('Request deadline expired before model execution', 504, 'deadline_exceeded')
        expired = threading.Event()
        def timed_out():
            return expired.is_set() or (deadline is not None and time.monotonic() >= deadline)
        class Cancellation:
            def is_set(self):
                return timed_out() or (caller_cancel is not None and caller_cancel.is_set())
            def wait(self, timeout=None):
                until = None if timeout is None else time.monotonic() + timeout
                while not self.is_set():
                    remaining = None if until is None else until - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        break
                    expired.wait(.01 if remaining is None else min(.01, remaining))
                return self.is_set()
        cancelled = Cancellation()
        def check_cancel():
            if caller_cancel is not None and caller_cancel.is_set():
                raise EngineError('Caller cancelled inference', 499, 'cancelled')
            if timed_out():
                raise EngineError('Inference exceeded the configured deadline', 504, 'deadline_exceeded')
        location = getattr(payload, 'location', 'local_only')
        stage = None
        if task in CLOUD_RUN_TASKS:
            stage = stage_for(payload.model, task, location)
        elif location == 'cloud_only':
            raise BrokerError(f'{task} stages run on local models only', 'broker_capability_unavailable')
        if stage is not None and payload.worker is not None:
            raise BrokerError('worker selects a local worker; it cannot be used with a cloud target', 'invalid_request')
        timer = threading.Timer(milliseconds / 1000, expired.set)
        timer.daemon = True
        timer.start()
        try:
            check_cancel()
            if stage is not None:
                # cloud stage: CloudInvoker on this worker thread, outside Engine.lock and the NativePool
                result = cloud_run(stage, task, values, cancelled, time.monotonic() + milliseconds / 1000)
                check_cancel()
                return result
            result = engine.infer(payload.model, task, {**values, "worker": payload.worker}, cancel=cancelled)
            try:
                check_cancel()
            except EngineError:
                if task in {'image', 'image-edit'} and isinstance(result, list):
                    for image in result:
                        getattr(image, 'close', lambda: None)()
                raise
            return result
        except EngineError:
            check_cancel()
            raise
        finally:
            timer.cancel()

    def translate(payload, caller_cancel=None, deadline=None):
        def boundary():
            if caller_cancel is not None and caller_cancel.is_set():
                raise EngineError('Translation cancelled', 499, 'cancelled')
            if deadline is not None and time.monotonic() >= deadline:
                raise EngineError('Translation exceeded its shared deadline', 504, 'deadline_exceeded')
        boundary()
        requested_source = payload.source_language if 'source_language' in payload.model_fields_set else None
        source_origin = 'request' if requested_source is not None else 'request_default'
        if "source_language" not in payload.model_fields_set and payload.model in config.models:
            source = config.models[payload.model].defaults.get("source_language")
            if source:
                if not isinstance(source, str) or not source.strip():
                    raise EngineError('Model source_language default must be a nonblank language ID')
                payload = payload.model_copy(update={"source_language": source})
                source_origin = 'model_default'
        if payload.include_provenance:
            if stage_for(payload.model, 'translation', payload.location) is not None:
                raise BrokerError('Attributed translation provenance needs a local NMT model; a cloud target has no '
                                  'loaded checkpoint to attest', 'provenance_unavailable_for_cloud')
            engine.validate(payload.model, 'translation')
            if payload.source_language == payload.target_language:
                text = payload.text
                stage = TranslationStage(stage='nmt', model=payload.model, status='passthrough', reason='equal_language_ids',
                    input_text=text, output_text=text, input_sha256=text_digest(text), output_sha256=text_digest(text))
                boundary()
                return TextTranslationResult(model=payload.model, text=text, translations=[{'text':text}],
                    source_language=payload.source_language, target_language=payload.target_language, translated=False,
                    requested_source_language=requested_source, source_language_origin=source_origin, stages=[stage]).model_dump(mode='json')
            spec = config.models[payload.model]
            if spec.backend != 'transformers' or spec.parallelism != 'single' or spec.allow_pickle or spec.trust_remote_code:
                raise EngineError('Attributed NMT requires a safe local single-device Transformers model', 400, 'unsupported_capability')
        result = run(payload, "translation", payload.model_dump(exclude_none=True), caller_cancel, deadline)
        boundary()
        if not isinstance(result, dict) or not isinstance(result.get('text'), str):
            raise EngineError('NMT returned invalid text', 502, 'invalid_backend_response')
        text = result["text"]
        response = {"model": payload.model, "text": text, "translations": [{"text": text}],
                "source_language": payload.source_language, "target_language": payload.target_language,
                "translated": result.get("translated", True)}
        if payload.include_provenance:
            try:
                provenance = NMTProvenance.model_validate(result['provenance'])
                selection = LanguageSelection.model_validate(result['language_selection'])
                measurements = NMTMeasurements.model_validate(result['measurements'])
                if (result.get('translated') is not True or result['input_sha256'] != text_digest(payload.text)
                        or result['output_sha256'] != text_digest(text)
                        or measurements.processed_input_sha256 != text_digest(selection.translation_prefix + payload.text)
                        or selection.registered_source_language != spec.defaults.get('source_language')
                        or selection.registered_target_language != spec.defaults.get('target_language')
                        or provenance.repository != spec.source_repo or provenance.revision != spec.source_revision):
                    raise ValueError('NMT returned mismatched input, output or loaded model identity')
                stage = TranslationStage(stage='nmt', model=payload.model, status='completed', input_text=payload.text,
                    output_text=text, input_sha256=result['input_sha256'], output_sha256=result['output_sha256'],
                    provenance=provenance, language_selection=selection, measurements=measurements)
                response = TextTranslationResult(**response, requested_source_language=requested_source,
                    source_language_origin=source_origin, stages=[stage]).model_dump(mode='json')
            except (ValueError, TypeError, KeyError) as exc:
                raise EngineError('NMT returned missing or inconsistent loaded provenance/measurements', 502, 'invalid_backend_response') from exc
        boundary()
        return response

    def classify_pairs(payload, caller_cancel=None, deadline=None):
        from .text_inputs import token_budget, truncation_policy
        raw = run(payload, 'pair-classification', {'pairs':[pair.model_dump() for pair in payload.pairs],
            'max_input_tokens':payload.max_input_tokens,'truncate':payload.truncate}, caller_cancel, deadline)
        try:
            result = PairClassificationResult.model_validate({**raw, 'model':payload.model})
            pairs, ids = validate_pairs(payload.pairs)
            spec = config.models[payload.model]
            settings = load_settings()
            limit = token_budget(payload.max_input_tokens, spec.defaults, 'max_input_tokens', settings.pair_max_input_tokens)
            if type(limit) is not int or not 1 <= limit <= settings.pair_max_input_tokens:
                raise ValueError('Invalid pair input limit')
            if len(result.results) != len(pairs) or len(result.labels) > settings.pair_max_labels:
                raise ValueError('Backend changed pair or native-label count')
            if result.truncation.enabled != truncation_policy(payload.truncate, spec.defaults, settings.pair_truncate) or result.truncation.max_tokens > limit:
                raise ValueError('Backend changed the pair truncation policy')
            if result.provenance.repository != spec.source_repo or result.provenance.revision != spec.source_revision:
                raise ValueError('Loaded classifier identity differs from the registration')
            for pair, identity, row in zip(pairs, ids, result.results):
                if (row.id != identity or row.id_source != ('caller' if pair.id is not None else 'index_default')
                        or row.input.premise_sha256 != pair_text_hash(pair.premise)
                        or row.input.hypothesis_sha256 != pair_text_hash(pair.hypothesis)
                        or row.input.pair_sha256 != pair_hash(pair.premise,pair.hypothesis)):
                    raise ValueError('Backend changed pair input identity or order')
            return result
        except (ValueError,TypeError,KeyError) as exc:
            raise EngineError('Invalid pair-classifier output, input fidelity or provenance',502,'invalid_backend_response') from exc

    def rerank(payload, caller_cancel=None, deadline=None, native_complete=None):
        from pydantic import ValidationError
        from .text_inputs import truncation_policy, token_budget
        raw = run(payload, 'rerank', {'query': payload.query, 'documents': payload.documents,
                                     'max_input_tokens': payload.max_input_tokens, 'truncate': payload.truncate}, caller_cancel, deadline)
        try:
            result = RerankResult.model_validate({**raw, 'model': payload.model})
        except (ValidationError, TypeError) as exc:
            raise EngineError('Invalid reranking backend response', 502, 'invalid_backend_response') from exc
        indices = [row.index for row in result.results]
        if len(indices) != len(payload.documents) or set(indices) != set(range(len(payload.documents))):
            raise EngineError('Reranker did not return every original document index', 502, 'invalid_backend_response')
        if len(result.input_details) != len(payload.documents) or {row.index for row in result.input_details} != set(range(len(payload.documents))):
            raise EngineError('Reranker omitted input-fidelity details', 502, 'invalid_backend_response')
        for row in result.input_details:
            if (row.input_sha256 != hashlib.sha256(payload.documents[row.index].encode('utf-8')).hexdigest()
                    or row.query_sha256 != hashlib.sha256(payload.query.encode('utf-8')).hexdigest()
                    or row.processed_tokens > row.input_tokens
                    or row.truncated != (row.processed_tokens < row.input_tokens)):
                raise EngineError('Reranker input-fidelity details do not match the request', 502, 'invalid_backend_response')
        tokens = sum(row.processed_tokens for row in result.input_details)
        if result.usage.prompt_tokens != tokens or result.usage.total_tokens != tokens:
            raise EngineError('Reranker token usage does not match its processed inputs', 502, 'invalid_backend_response')
        if (not result.truncation.enabled and any(row.truncated for row in result.input_details)
                or any(row.processed_tokens > result.truncation.max_tokens for row in result.input_details)):
            raise EngineError('Reranker violated its reported truncation policy', 502, 'invalid_backend_response')
        spec = config.models[payload.model]
        expected_policy = truncation_policy(payload.truncate, spec.defaults, load_settings().rerank_truncate)
        expected_limit = token_budget(payload.max_input_tokens, spec.defaults, 'max_input_tokens', load_settings().rerank_max_input_tokens)
        if result.truncation.enabled != expected_policy or result.truncation.strategy != 'only_second' or result.truncation.max_tokens > expected_limit:
            raise EngineError('Reranker did not honor the requested input policy', 502, 'invalid_backend_response')
        result.results.sort(key=lambda row: (-row.relevance_score, row.index))
        result.results = result.results[:payload.top_n]
        for row in result.results:
            row.document = {'text': payload.documents[row.index]} if payload.return_documents else None
        return result

    def retrieval_answer(payload, caller_cancel=None, deadline=None, native_complete=None):
        from .embedding_output import format_embeddings
        from .retrieval_workflow import execute
        from .server import Chat, Embeddings
        settings=load_settings()
        if stage_for(payload.llm_model, 'chat', getattr(payload, 'location', 'local_only')) is not None:
            # the workflow binds its answer to exact native context accounting and checkpoint provenance
            raise BrokerError('Retrieval answers need a local llama-cpp model (exact native context accounting); '
                              'a cloud LLM cannot provide it', 'broker_capability_unavailable')
        def validate_worker(model, worker):
            spec=config.models[model]
            if spec.backend=='llama-cpp':
                if native_gateway is None:raise EngineError('Native gateway is unavailable',503,'worker_unavailable')
                gateway(getattr(payload, 'location', 'local_only')).candidates(model,worker)
            elif worker is not None and (worker not in config.workers or model not in config.workers[worker].models):
                raise EngineError('No matching workflow worker assignment',404,'worker_not_found')
            elif any(assignment.mode!='single' for name,assignment in config.workers.items() if model in assignment.models and (worker is None or name==worker)):
                raise EngineError('Workflow text encoders require single-device workers',400,'unsupported_capability')
        def input_limit(model, explicit, key, maximum):
            if model not in config.models:raise EngineError('Unknown workflow model',404,'model_not_found')
            limit=explicit if explicit is not None else config.models[model].defaults.get(key,maximum)
            if type(limit) is not int or not 1<=limit<=maximum:
                raise EngineError('Workflow model input defaults exceed current limits')
            return limit
        encoder_limit=input_limit(payload.embedding_model,payload.embedding_max_tokens,'max_tokens',settings.embedding_max_tokens)
        rank_limit=input_limit(payload.rerank_model,payload.rerank_max_input_tokens,'max_input_tokens',settings.rerank_max_input_tokens)
        def prepare_chat(messages):
            values={'model':payload.llm_model,'worker':payload.llm_worker,'messages':messages,'stream':False}
            values.update({key:getattr(payload,key) for key in ['max_tokens','temperature','top_p'] if getattr(payload,key) is not None})
            try:
                chat=Chat.model_validate(values)
                if chat_defaults is not None:chat=chat_defaults(chat)
            except ValueError as exc:raise EngineError('Workflow generation controls or context exceed chat limits') from exc
            data=chat.model_dump(exclude_none=True)
            if any(data.get(key) for key in ['tools','tool_choice','parallel_tool_calls']):
                raise EngineError('Retrieval answers use a source-reference schema, not model-default tool calls',400,'unsupported_capability')
            if data.get('max_completion_tokens') is not None:data['max_tokens']=data.pop('max_completion_tokens')
            return data
        def embed(texts):
            request=Embeddings(model=payload.embedding_model,worker=payload.embedding_worker,input=texts,max_tokens=encoder_limit,truncate=False,encoding_format='float')
            owner=ExtensionPayload(model=request.model,worker=request.worker,timeout=payload.timeout)
            return format_embeddings(request,run(owner,'embeddings',request.model_dump(),caller_cancel,deadline))
        def rank(documents):
            request=RerankRequest(model=payload.rerank_model,worker=payload.rerank_worker,query=payload.query,documents=documents,
                top_n=None,return_documents=False,max_input_tokens=rank_limit,truncate=False,timeout=payload.timeout)
            return rerank(request,caller_cancel,deadline)
        if native_complete is None:raise EngineError('Retrieval answers require the live native dispatcher',503,'worker_unavailable')
        return execute(payload,config=config,embed=embed,rerank=rank,native_complete=native_complete,prepare_chat=prepare_chat,
            validate_worker=validate_worker,cancel=caller_cancel,deadline=deadline)

    def image_extension(payload, task, values, caller_cancel, deadline):
        if caller_cancel is not None and caller_cancel.is_set():
            raise EngineError('Image inference cancelled before decode', 499, 'cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise EngineError('Image inference deadline expired before decode', 504, 'deadline_exceeded')
        image = _image(payload)
        try:
            # run checks the same absolute deadline after decode and before inference.
            return run(payload, task, {**values, "image": image}, caller_cancel, deadline)
        finally:
            image.close()

    def ocr(payload, caller_cancel=None, deadline=None):
        engine.validate(payload.model, 'image-to-text')
        spec = config.models[payload.model]
        if spec.backend == 'tesseract':
            try:
                ocr_controls(payload.model_dump(exclude_none=True), spec.defaults)
            except ValueError as exc:
                raise EngineError(str(exc)) from exc
        elif any(getattr(payload, key) is not None for key in OCR_CONTROL_NAMES):
            raise EngineError('Native OCR controls require a Tesseract backend')
        result = image_extension(payload, "image-to-text", payload.model_dump(exclude_none=True), caller_cancel, deadline)
        response = {"model": payload.model, "text": result["text"], "lines": result["text"].splitlines(), "task": "image-to-text"}
        if spec.backend == 'tesseract':
            response.update({key: result[key] for key in ('raw_text', 'ocr', 'geometry') if key in result})
        return response

    def depth(payload, caller_cancel=None, deadline=None):
        from .depth import DepthResult, check_dimensions
        # Header dimensions bound dense output before RGB conversion or model work.
        def check_size(width, height):
            try:
                check_dimensions(width, height)
            except ValueError as exc:
                raise EngineError(str(exc), 413, 'request_too_large') from exc
        if caller_cancel is not None and caller_cancel.is_set():
            raise EngineError('Depth inference cancelled before decode', 499, 'cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise EngineError('Depth inference deadline expired before decode', 504, 'deadline_exceeded')
        image = _image(payload, size_check=check_size)
        try:
            result = run(payload, 'depth-estimation', {'model': payload.model, 'image': image}, caller_cancel, deadline)
            try:
                parsed = DepthResult.model_validate({**result, 'model': payload.model})
                if (parsed.width, parsed.height) != image.size:
                    raise ValueError('Depth result geometry differs from the decoded input')
                if parsed.input_rgb_sha256 != hashlib.sha256(image.tobytes()).hexdigest():
                    raise ValueError('Depth result input identity differs from the decoded RGB image')
                if caller_cancel is not None and caller_cancel.is_set():
                    raise EngineError('Depth inference cancelled during response validation', 499, 'cancelled')
                if deadline is not None and time.monotonic() >= deadline:
                    raise EngineError('Depth inference deadline expired during response validation', 504, 'deadline_exceeded')
                return parsed
            except (ValueError, TypeError) as exc:
                raise EngineError('Depth backend returned invalid image-aligned values', 502, 'invalid_backend_response') from exc
        finally:
            image.close()

    def detect(payload, caller_cancel=None, deadline=None):
        values = {name: getattr(payload, name) for name in ('threshold', 'nms', 'nms_iou')}
        result = image_extension(payload, "object-detection", values, caller_cancel, deadline)
        return {"model": payload.model, **result}

    def segment(payload,caller_cancel=None,deadline=None):
        from .segmentation import resolve_prompt,check_dimensions,check_cancel
        spec=config.models.get(payload.model)
        prompt=resolve_prompt(payload.model_dump(exclude_none=True),spec.defaults if spec else {})
        def boundary():
            check_cancel(caller_cancel)
            if deadline is not None and time.monotonic()>=deadline:
                raise EngineError('Segmentation deadline exceeded',504,'deadline_exceeded')
        def check_size(width,height):
            try:
                check_dimensions(width,height,load_settings().segmentation_max_masks if prompt.multimask else 1)
                prompt.check_geometry(width,height)
            except ValueError as exc:raise EngineError(str(exc)) from exc
        boundary();image=_image(payload,check_size)
        try:
            boundary()
            result=run(payload,'image-segmentation',{'model':payload.model,'image':image,**prompt.model_dump(exclude_none=True)},caller_cancel,deadline)
            try:
                parsed=SegmentationResult.model_validate(result)
                if (parsed.model!=payload.model or (parsed.width,parsed.height)!=image.size
                        or parsed.input_rgb_sha256!=hashlib.sha256(image.tobytes()).hexdigest() or parsed.prompt!=prompt):
                    raise ValueError('Segmentation result differs from its source image or prompt')
                parsed.verify_pixels()
            except (ValueError,TypeError,OSError) as exc:
                raise EngineError('Segmentation backend returned invalid aligned masks: '+str(exc),502,'invalid_backend_response') from exc
            boundary();return parsed
        finally:image.close()

    def llm_parameters(payload, spec):
        """Validate static LLM controls before a composition loads its first stage."""
        settings = load_settings()
        prompt_version = payload.prompt_version or spec.defaults.get('sentiment_prompt_version', settings.sentiment_prompt_version)
        decoding = payload.decoding or spec.defaults.get('sentiment_decoding', settings.sentiment_decoding)
        if not isinstance(prompt_version, str) or prompt_version not in {'weights-v1', 'label-v2', 'label-v3', 'facets-v1'}:
            raise EngineError('Invalid sentiment model default: unknown prompt version')
        if not isinstance(decoding, str) or decoding not in {'auto', 'validate', 'label_choices', 'facet_choices', 'native_json_schema'}:
            raise EngineError('Invalid sentiment model default: unknown decoding mode')
        if decoding == 'auto':
            decoding = 'validate'
            if spec.backend == 'transformers':
                if prompt_version in {'label-v2', 'label-v3'}:
                    decoding = 'label_choices'
                elif prompt_version == 'facets-v1':
                    decoding = 'facet_choices'
            elif spec.backend == 'llama-cpp':
                decoding = 'native_json_schema'
        if decoding not in {'validate', 'label_choices', 'facet_choices', 'native_json_schema'}:
            raise EngineError('Unknown sentiment decoding mode')
        if decoding == 'label_choices' and (prompt_version not in {'label-v2', 'label-v3'} or spec.backend != 'transformers'):
            raise EngineError('Label-choice decoding requires a categorical label prompt and a Transformers chat model', 400, 'unsupported_capability')
        if decoding == 'facet_choices' and (prompt_version != 'facets-v1' or spec.backend != 'transformers'):
            raise EngineError('Facet-choice decoding requires facets-v1 and a Transformers chat model', 400, 'unsupported_capability')
        if decoding == 'native_json_schema' and spec.backend != 'llama-cpp':
            raise EngineError('Native JSON-schema decoding requires an owned llama-cpp model',400,'unsupported_capability')
        if spec.backend in ('llama-cpp', 'cloud'):
            if native_gateway is None and spec.backend == 'llama-cpp':
                raise EngineError('Native gateway is unavailable',503,'worker_unavailable')
            gateway(payload.location).candidates(payload.model,payload.worker)
        try:
            instruction = llm_instruction(prompt_version)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        from .server import Chat
        messages = [{'role': 'system', 'content': instruction},
                    {'role': 'user', 'content': json.dumps({'text': payload.text}, ensure_ascii=False)}]
        chat_values = {'model': payload.model, 'worker': payload.worker, 'messages': messages}
        for name in ['max_tokens', 'temperature', 'top_p']:
            if getattr(payload, name) is not None:
                chat_values[name] = getattr(payload, name)
        try:
            chat = Chat.model_validate(chat_values)
            if chat_defaults is not None:
                chat = chat_defaults(chat)
        except ValueError as exc:
            raise EngineError('Sentiment LLM prompt and controls exceed the configured chat limits: ' + str(exc)) from exc
        data = chat.model_dump(exclude_none=True)
        if data.get('max_completion_tokens') is not None:
            data['max_tokens'] = data.pop('max_completion_tokens')
        if decoding == 'label_choices':
            data['_sentiment_label_choices'] = True
        elif decoding == 'facet_choices':
            data['_sentiment_facet_choices'] = True
        elif decoding == 'native_json_schema':
            data['response_format'] = {'type':'json_schema','json_schema':{'name':'titanos_sentiment','strict':True,'schema':sentiment_json_schema(prompt_version)}}
        return data, messages, prompt_version, decoding

    def prepare_sentiment(payload):
        spec, _ = spec_of(payload.model, 'chat', payload.location)
        if spec is None:
            raise EngineError('Unknown model', 404, 'model_not_found')
        if spec.task not in {'sentiment', 'chat'} or spec.backend not in {'transformers', 'openvino-genai', 'llama-cpp', 'cloud'}:
            raise EngineError('Sentiment requires a dedicated classifier or a local Transformers/GenAI/llama-cpp chat model', 400, 'unsupported_capability')
        values = payload.model_dump()
        settings = load_settings()
        for name, default in [('min_score', settings.sentiment_min_score), ('min_margin', settings.sentiment_min_margin)]:
            if values[name] is None:
                values[name] = spec.defaults.get(name, default)
        try:
            payload = type(payload).model_validate(values)
        except ValueError as exc:
            raise EngineError('Invalid sentiment model defaults: ' + str(exc)) from exc
        if spec.task == 'sentiment':
            if any(getattr(payload, name) is not None for name in ['max_tokens', 'temperature', 'top_p', 'prompt_version', 'decoding']):
                raise EngineError('Generation controls apply to LLM sentiment; use max_input_tokens for a classifier')
            limit = payload.max_input_tokens if payload.max_input_tokens is not None else spec.defaults.get('max_input_tokens', settings.sentiment_max_input_tokens)
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= settings.sentiment_max_input_tokens:
                raise EngineError('Classifier input limit exceeds current sentiment settings')
            return payload, spec, limit
        if payload.max_input_tokens is not None:
            raise EngineError('max_input_tokens is a classifier control; LLM prompt/context limits use the existing chat path')
        return payload, spec, llm_parameters(payload, spec)

    def sentiment(payload, caller_cancel=None, deadline=None, native_complete=None):
        if deadline is None:
            deadline = time.monotonic() + (payload.timeout or int(load_settings().client_timeout * 1000)) / 1000
        payload, spec, prepared = prepare_sentiment(payload)
        if spec.task == 'sentiment':
            limit = prepared
            raw = run(payload, 'sentiment', {'text': payload.text, 'max_input_tokens': limit}, caller_cancel, deadline)
            if raw.get('score_kind') != 'softmax_uncalibrated':
                raise EngineError('Classifier returned unsupported score semantics', 502, 'invalid_backend_response')
            try:
                return build_result(payload, raw, 'dedicated_classifier', raw['provenance'])
            except (ValueError, TypeError, KeyError) as exc:
                raise EngineError('Invalid classifier result: ' + str(exc), 502, 'invalid_backend_response') from exc
        if payload.max_input_tokens is not None:
            raise EngineError('max_input_tokens is a classifier control; LLM prompt/context limits use the existing chat path')
        data, messages, prompt_version, decoding = prepared
        if spec.backend in ('llama-cpp', 'cloud'):  # a cloud LLM goes through GatewayRouter (CloudInvoker)
            if native_complete is None:
                raise EngineError('Native completion requires the live request dispatcher',503,'worker_unavailable')
            answer = native_complete(data,deadline,caller_cancel)
        else:
            answer = run(payload, 'chat', data, caller_cancel, deadline)
        raw = parse_llm_output(answer.get('text'), answer.get('finish_reason'), prompt_version)
        actual = answer.get('provenance') or {}
        provenance = {'repository': spec.source_repo, 'revision': spec.source_revision, 'backend': spec.backend,
                      'device': actual.get('device'), 'checkpoint_digest': actual.get('checkpoint_digest'),
                      'prompt_version': prompt_version,
                      'prompt_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}
        return build_result(payload, raw, 'llm', provenance,
                            decoding if decoding in {'label_choices', 'facet_choices', 'native_json_schema'} else 'prompt_then_validation')

    def sentiment_compose(payload, caller_cancel=None, deadline=None, native_complete=None):
        if deadline is None:
            deadline = time.monotonic() + (payload.timeout or int(load_settings().client_timeout * 1000)) / 1000
        # Validate both roles and requests before loading either model.
        roles = [(payload.model, 'sentiment'), (payload.llm_model, 'chat')]
        specs = {}
        for name, task in roles:
            spec = specs[name] = (spec_of(name, 'chat', payload.location)[0] if task == 'chat'
                                  else config.models.get(name))
            if spec is None:
                raise EngineError('Unknown composition model', 404, 'model_not_found')
            if spec.task != task or spec.backend not in ({'transformers'} if task == 'sentiment' else {'transformers', 'openvino-genai', 'llama-cpp', 'cloud'}):
                raise EngineError('Composition requires an explicit classifier and local chat model', 400, 'unsupported_capability')
        common = {key: getattr(payload, key) for key in ['text', 'language', 'input_modality', 'asr_uncertain', 'min_score', 'min_margin', 'location']}
        classifier = SentimentRequest(model=payload.model, worker=payload.worker, max_input_tokens=payload.max_input_tokens, **common)
        llm = SentimentRequest(model=payload.llm_model, worker=payload.llm_worker,
            **{key: getattr(payload, key) for key in ['max_tokens', 'temperature', 'top_p', 'prompt_version', 'decoding']}, **common)
        llm_parameters(llm, specs[llm.model])
        stages = []
        for request in [classifier, llm]:
            remaining = math.ceil((deadline - time.monotonic()) * 1000)
            if caller_cancel is not None and caller_cancel.is_set():
                error = SentimentStageError(code='cancelled', message='Caller cancelled composition', http_status=499)
                stages.append(SentimentStage(model=request.model, status='not_run', error=error))
                continue
            if stages and stages[0].status != 'completed':
                error = SentimentStageError(code='preceding_stage_failed', message='The classifier stage did not complete', http_status=424)
                stages.append(SentimentStage(model=request.model, status='not_run', error=error))
                continue
            if remaining <= 0:
                error = SentimentStageError(code='deadline_exceeded', message='Composition exhausted its shared deadline', http_status=504)
                stages.append(SentimentStage(model=request.model, status='not_run', error=error))
                continue
            try:
                result = sentiment(request.model_copy(update={'timeout': remaining}), caller_cancel, deadline, native_complete)
                stages.append(SentimentStage(model=request.model, status='completed', result=result))
            except EngineError as exc:
                stages.append(SentimentStage(model=request.model, status='failed', error=SentimentStageError(
                    code=exc.code, message=str(exc), http_status=exc.status)))
        identity = SentimentInput(sha256=hashlib.sha256(payload.text.encode('utf-8')).hexdigest(), characters=len(payload.text),
            modality=payload.input_modality, language=payload.language, asr_uncertain=payload.asr_uncertain)
        return compose_results(*stages, input_data=identity)

    async def dispatch_cancellable(request, payload, operation):
        # Retain the worker until cooperative cleanup. HTTP disconnects and MCP
        # task cancellation share one caller event across the composition stages.
        cancelled = threading.Event()
        deadline = time.monotonic() + (payload.timeout or int(load_settings().client_timeout * 1000)) / 1000
        loop = asyncio.get_running_loop()
        # the request's principal, captured here: native_complete runs its coroutine on the loop thread, whose context
        # does not carry this request's ContextVars
        principal = current_principal()
        location = getattr(payload, 'location', 'local_only')
        def native_complete(data, absolute_deadline, caller_cancel, context_max_tokens=None):
            async def complete():
                if native_gateway is None and (broker is None or not broker.active(principal)):
                    raise EngineError('Native gateway is unavailable',503,'worker_unavailable')
                options={'deadline':absolute_deadline,'cancel':caller_cancel}
                if context_max_tokens is not None:options['context_max_tokens']=context_max_tokens
                return await gateway(location, principal).complete(data,**options)
            # Called only inside invoke's worker thread; the request loop stays free.
            return asyncio.run_coroutine_threadsafe(complete(),loop).result()
        def invoke():
            remaining = math.ceil((deadline - time.monotonic()) * 1000)
            if remaining <= 0:
                raise EngineError('Request expired before stage execution', 504, 'deadline_exceeded')
            arguments = (payload.model_copy(update={'timeout': remaining}), cancelled, deadline)
            return operation(*arguments, native_complete) if operation in {sentiment, sentiment_compose, retrieval_answer, audio_analysis} else operation(*arguments)
        worker = asyncio.create_task(asyncio.to_thread(invoke))
        try:
            while not worker.done():
                if await request.is_disconnected():
                    cancelled.set()
                await asyncio.wait({worker}, timeout=.05)
            return await worker
        except asyncio.CancelledError:
            cancelled.set()
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.shield(worker)
                except Exception:
                    pass  # The caller remains cancelled even if backend cleanup fails.
            raise

    def asr_stage(payload, raw, values, caller_cancel=None, deadline=None):
        deadline = deadline if deadline is not None else time.monotonic() + (payload.timeout or int(load_settings().client_timeout * 1000)) / 1000
        def boundary():
            if caller_cancel is not None and caller_cancel.is_set():
                raise EngineError('Caller cancelled audio pipeline', 499, 'cancelled')
            if time.monotonic() >= deadline:
                raise EngineError('Audio pipeline deadline expired', 504, 'deadline_exceeded')
        boundary()
        stage = stage_for(payload.model, 'transcription', payload.location)
        if stage is None:
            engine.validate(payload.model, 'transcription')
        # a cloud ASR target gate-resolves with defaults={} (the VAD gate itself always runs locally)
        defaults = stage.spec.defaults if stage is not None else config.models[payload.model].defaults
        try:
            selection = resolve_gate(payload.model_dump(), defaults, load_settings())
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        gate = None
        if selection['enabled']:
            selected = selection['model']
            if selected == payload.model:
                raise EngineError('ASR cannot use itself as its VAD gate')
            engine.validate(selected, 'vad')
            options = dict(selection['options'])
            measurement = vad(VADRequest(model=selected, audio_base64='AA==', timeout=payload.timeout,
                location='local_only', return_frames=False, **options), caller_cancel, deadline, raw=raw)
            boundary()
            gate = ASRVADGate(model=selected, decision='speech_detected' if measurement.segments else 'no_speech_detected',
                asr_executed=False, model_source=selection['model_source'], enabled_source=selection['enabled_source'], measurement=measurement)
            if not measurement.segments:
                if payload.response_format != 'verbose_json':
                    return '', gate, deadline
                skipped = {'task':'translate' if values.get('translation') else 'transcribe', 'text':'',
                    'duration':measurement.duration, 'sample_rate':measurement.sample_rate, 'decoded_samples':measurement.decoded_samples,
                    'language':None, 'language_source':'unavailable', 'requested_language':payload.language,
                    'segments':None, 'words':None,
                    'timestamps':{'granularity':None, 'source':'unavailable', 'supported':[], 'reason':'ASR was skipped after VAD reported no speech'},
                    'quality':{'confidence':None, 'avg_logprob':None, 'no_speech_prob':None, 'status':'unavailable',
                               'reason':'ASR was not executed; VAD is a separate uncalibrated speech-activity measurement'},
                    'backend':None, 'duration_source':'vad_decoded_audio', 'vad_gate':gate.model_dump(mode='json')}
                return skipped, gate, deadline
        boundary()
        result = run(payload, 'transcription', {'audio':raw, **values}, caller_cancel, deadline)
        boundary()
        if gate is not None:
            gate = gate.model_copy(update={'asr_executed':True})
            if payload.response_format == 'verbose_json':
                if not isinstance(result, dict):
                    raise EngineError('ASR returned invalid metadata', 502, 'invalid_backend_response')
                result = {**result, 'vad_gate':gate.model_dump(mode='json'), 'duration_source':'asr_decoded_audio'}
        return result, gate, deadline

    def transcription(payload, caller_cancel=None, deadline=None, raw=None):
        raw = raw if raw is not None else _inline_bytes(payload.audio_base64, 'audio')
        result, _, _ = asr_stage(payload, raw, {'language': payload.language,
            **({'prompt':payload.prompt} if payload.prompt is not None else {}),
            'max_tokens': payload.max_tokens, 'response_format': payload.response_format,
            'timestamp_granularities': payload.timestamp_granularities}, caller_cancel, deadline)
        if payload.response_format == 'verbose_json':
            try:
                return ASRMetadata.model_validate(result).model_dump(mode='json', exclude_unset=True)
            except ValueError as exc:
                raise EngineError('ASR returned invalid metadata', 502, 'invalid_backend_response') from exc
        if not isinstance(result, str):
            raise EngineError('ASR returned invalid text', 502, 'invalid_backend_response')
        return {'text': result}

    def vad(payload, caller_cancel=None, deadline=None, raw=None):
        raw = raw if raw is not None else _inline_bytes(payload.audio_base64, 'audio')
        data = {key: getattr(payload, key) for key in ('sample_rate', 'threshold', 'min_speech_ms', 'min_silence_ms', 'speech_pad_ms', 'return_frames')}
        result = run(payload, 'vad', {'audio': raw, **data}, caller_cancel, deadline)
        try:
            parsed = VADResult.model_validate({**result, 'model': payload.model})
            if parsed.input_sha256 != hashlib.sha256(raw).hexdigest():
                raise ValueError('Wrong VAD input identity')
            return parsed
        except (ValueError, TypeError) as exc:
            raise EngineError('VAD returned invalid measurements', 502, 'invalid_backend_response') from exc

    def audio_boundary(cancel, deadline):
        if cancel is not None and cancel.is_set():
            raise EngineError('Caller cancelled audio analysis', 499, 'cancelled')
        if deadline is not None and time.monotonic() >= deadline:
            raise EngineError('Audio analysis exhausted its shared deadline', 504, 'deadline_exceeded')

    def audio_worker(model, task, worker, location='local_only'):
        spec, stage = spec_of(model, task, location)
        if spec is None:
            raise EngineError('Unknown audio workflow model', 404, 'model_not_found')
        if spec.task != task:
            raise EngineError('Audio workflow model has the wrong task', 400, 'unsupported_capability')
        if stage is not None:
            if worker is not None:
                raise BrokerError('worker selects a local worker; it cannot be used with a cloud target',
                                  'invalid_request')
            return spec
        if spec.backend == 'llama-cpp':
            if native_gateway is None:
                raise EngineError('Native gateway is unavailable', 503, 'worker_unavailable')
            gateway(location).candidates(model, worker)
        elif worker is not None and (worker not in config.workers or model not in config.workers[worker].models):
            raise EngineError('No matching audio workflow worker assignment', 404, 'worker_not_found')
        else:
            engine.validate(model, task)
        return spec

    def prepare_diarization(payload):
        from .diarization import prepare
        spec = audio_worker(payload.model, 'diarization', payload.worker)
        try:
            return prepare(spec, {key: getattr(payload, key) for key in DiarizationOptions.model_fields})
        except ValueError as exc:
            raise EngineError('Invalid diarization controls: ' + str(exc)) from exc

    def check_diarization_audio(values, raw, caller_cancel, deadline):
        # Sherpa bypasses clustering on <=10s clips. Check nondefault controls
        # before any load; keep the existing stage decoders and original bytes.
        if values['num_speakers'] == 0 and values['cluster_threshold'] == .5:
            return
        from .diarization import validate_duration_controls
        from .media import read_audio
        class Budget:
            def is_set(self):
                return (caller_cancel is not None and caller_cancel.is_set()) or time.monotonic() >= deadline
        try:
            wave = read_audio(raw, 16000, Budget())
            validate_duration_controls(values, len(wave) / 16000)
        except (EngineError, ValueError) as exc:
            audio_boundary(caller_cancel, deadline)
            raise EngineError('Diarization controls are incompatible with this audio: ' + str(exc)) from exc
        audio_boundary(caller_cancel, deadline)

    def diarization(payload, caller_cancel=None, deadline=None, raw=None, controls_checked=False):
        deadline = deadline if deadline is not None else time.monotonic() + (payload.timeout or int(load_settings().client_timeout * 1000)) / 1000
        audio_boundary(caller_cancel, deadline)
        values = prepare_diarization(payload)
        raw = raw if raw is not None else _inline_bytes(payload.audio_base64, 'audio')
        audio_boundary(caller_cancel, deadline)
        if not controls_checked:
            check_diarization_audio(values, raw, caller_cancel, deadline)
        result = run(payload, 'diarization', {'audio': raw, **values}, caller_cancel, deadline)
        try:
            parsed = DiarizationResult.model_validate({**result, 'model': payload.model})
            if parsed.input_sha256 != hashlib.sha256(raw).hexdigest():
                raise ValueError('Wrong diarization input identity')
            settings = load_settings()
            if parsed.duration > settings.max_audio_seconds or parsed.decoded_samples > settings.max_audio_samples:
                raise ValueError('Diarization exceeded the shared audio size budget')
            if any(parsed.parameters.get(key) != value for key, value in values.items()):
                raise ValueError('Diarization did not honor effective request controls')
        except (ValueError, TypeError, KeyError) as exc:
            raise EngineError('Diarization returned invalid measurements: ' + str(exc), 502, 'invalid_backend_response') from exc
        audio_boundary(caller_cancel, deadline)
        return parsed

    def audio_analysis(payload, caller_cancel=None, deadline=None, native_complete=None):
        deadline = deadline if deadline is not None else time.monotonic() + (payload.timeout or int(load_settings().client_timeout * 1000)) / 1000
        audio_boundary(caller_cancel, deadline)
        asr = ASRRequest.model_validate({key: getattr(payload, key) for key in ASRRequest.model_fields})
        spec = audio_worker(asr.model, 'transcription', asr.worker, payload.location)
        # Validate static effective ASR controls without turning model defaults into request provenance.
        effective = asr.model_dump()
        for key in ('prompt', 'max_tokens'):
            if effective[key] is None and key in spec.defaults:
                effective[key] = spec.defaults[key]
        try:
            if effective['max_tokens'] is not None and type(effective['max_tokens']) is not int:
                raise ValueError('ASR model max_tokens must be an integer')
            ASRRequest.model_validate(effective)
            selection = resolve_gate(asr.model_dump(), spec.defaults, load_settings())
        except ValueError as exc:
            raise EngineError('Invalid ASR workflow controls: ' + str(exc)) from exc
        if selection['enabled']:
            gate_model = selection['model']
            gate_spec = audio_worker(gate_model, 'vad', selection['options'].get('worker'))
            gate_values = {**gate_spec.defaults, **selection['options']}
            try:
                gate_request = VADRequest(model=gate_model, audio_base64='AA==', **gate_values)
                for key in ('min_speech_ms', 'min_silence_ms', 'speech_pad_ms'):
                    value = getattr(gate_request, key)
                    if value is not None and value > load_settings().max_audio_seconds * 1000:
                        raise ValueError('VAD timing control exceeds the shared audio duration budget')
            except ValueError as exc:
                raise EngineError('Invalid VAD workflow controls: ' + str(exc)) from exc
        diarizer = DiarizationRequest(model=payload.diarization_model, worker=payload.diarization_worker,
            audio_base64=payload.audio_base64, timeout=payload.timeout, location=payload.location,
            **payload.diarization_options.model_dump(exclude_unset=True))
        diarization_values = prepare_diarization(diarizer)
        analysis_options = (payload.text_analysis_options or TranscriptAnalysisOptions()).model_dump(exclude_none=True)
        if payload.text_analysis_model is not None:
            text_spec, _ = spec_of(payload.text_analysis_model, 'chat', payload.location)
            if text_spec is None:
                raise EngineError('Unknown text-analysis model', 404, 'model_not_found')
            audio_worker(payload.text_analysis_model, text_spec.task, payload.text_analysis_worker, payload.location)
            prepare_sentiment(SentimentRequest(model=payload.text_analysis_model, worker=payload.text_analysis_worker,
                text='Static controls preflight', input_modality='asr_transcript', timeout=payload.timeout,
                location=payload.location, **analysis_options))
        audio_boundary(caller_cancel, deadline)
        raw = _inline_bytes(payload.audio_base64, 'audio')
        check_diarization_audio(diarization_values, raw, caller_cancel, deadline)
        transcription_result = ASRMetadata.model_validate(transcription(asr, caller_cancel, deadline, raw=raw))
        audio_boundary(caller_cancel, deadline)
        if (transcription_result.decoded_samples > load_settings().max_audio_samples
                or transcription_result.duration > load_settings().max_audio_seconds
                or abs(transcription_result.duration - transcription_result.decoded_samples / transcription_result.sample_rate) > 1e-8):
            raise EngineError('ASR exceeded the shared audio size budget', 502, 'invalid_backend_response')
        measured = diarization(diarizer, caller_cancel, deadline, raw=raw, controls_checked=True)
        # Same original bytes, independently decoded/resampled by existing stages.
        tolerance = max(1 / transcription_result.sample_rate, 1 / measured.sample_rate)
        if (transcription_result.decoded_samples > load_settings().max_audio_samples
                or transcription_result.duration > load_settings().max_audio_seconds
                or abs(transcription_result.duration - transcription_result.decoded_samples / transcription_result.sample_rate) > 1e-8
                or abs(transcription_result.duration - measured.duration) > tolerance):
            raise EngineError('Audio stages disagree about the original decoded duration or size budget', 502, 'invalid_backend_response')
        units = transcription_result.segments
        granularity = 'segment'
        if units is None:
            units = transcription_result.words
            granularity = 'word'
        alignment = granularity if units is not None else 'unavailable'
        attributed = []
        for unit in units or []:
            audio_boundary(caller_cancel, deadline)
            row = unit.model_dump()
            if granularity == 'word':
                row['text'] = row.pop('word')
            try:
                reason = timestamp_attribution(row['start'], row['end'], transcription_result.duration)
            except ValueError as exc:
                raise EngineError('Invalid ASR alignment: ' + str(exc), 502, 'invalid_backend_response') from exc
            known = reason == 'interval_intersection'
            speakers = sorted({segment.speaker for segment in measured.segments
                if known and max(row['start'], segment.start) < min(row['end'], segment.end)}, key=lambda name: int(name.split('_')[1]))
            overlap = any(known and max(row['start'], span.start) < min(row['end'], span.end) for span in measured.overlaps)
            attributed.append(AttributedASRSegment(**row, speakers=speakers, overlap=overlap,
                attribution=reason, granularity=granularity))
        text_result, text_status = None, 'not_requested'
        if payload.text_analysis_model is not None:
            text_status = 'not_run_empty_transcript'
            if transcription_result.text.strip():
                audio_boundary(caller_cancel, deadline)
                text_result = sentiment(SentimentRequest(model=payload.text_analysis_model, worker=payload.text_analysis_worker,
                    text=transcription_result.text, language=transcription_result.language, input_modality='asr_transcript',
                    timeout=payload.timeout, location=payload.location, **analysis_options), caller_cancel, deadline, native_complete)
                text_status = 'completed'
        try:
            result = AudioAnalysisResult(model=payload.model, text=transcription_result.text, transcription=transcription_result,
                diarization=measured, speaker_segments=attributed, speaker_alignment=alignment,
                speaker_attribution_status=attribution_status(attributed),
                text_analysis=text_result, text_analysis_status=text_status)
        except ValueError as exc:
            raise EngineError('Invalid composed audio result: ' + str(exc), 502, 'invalid_backend_response') from exc
        audio_boundary(caller_cancel, deadline)
        return result

    def translate_audio(payload, caller_cancel=None, deadline=None, raw=None):
        raw = raw if raw is not None else _inline_bytes(payload.audio_base64, "audio")
        if payload.translation_model:
            # Validate both ends before the first model can execute.
            translation_stage = stage_for(payload.translation_model, 'translation', payload.location)
            if translation_stage is None:
                engine.validate(payload.translation_model, "translation")
            elif payload.include_provenance:
                raise BrokerError('Attributed translation provenance needs a local NMT model',
                                  'provenance_unavailable_for_cloud')
            if payload.target_language is None:
                defaults = (translation_stage.spec.defaults if translation_stage is not None
                            else config.models[payload.translation_model].defaults)
                target = defaults.get('target_language', 'eng_Latn')
                if not isinstance(target, str) or not target.strip():
                    raise EngineError('Translation model target_language default must be a nonblank English language ID')
                try:
                    values = payload.model_dump()
                    values.pop('operation', None)
                    payload = AudioTranslationRequest.model_validate({**values, 'target_language': target})
                except ValueError as exc:
                    raise EngineError(str(exc)) from exc
        stage_payload = payload.model_copy(update={'response_format':'verbose_json'}) if payload.include_provenance else payload
        result, gate, deadline = asr_stage(stage_payload, raw, {"language": payload.language,
                    "max_tokens": payload.max_tokens, "translation": not bool(payload.translation_model),
                    'response_format': stage_payload.response_format,
                    'timestamp_granularities': []}, caller_cancel, deadline)
        try:
            metadata = ASRMetadata.model_validate(result).model_dump(mode='json', exclude_unset=True) if stage_payload.response_format == 'verbose_json' else None
        except (ValueError, TypeError) as exc:
            raise EngineError('ASR returned invalid metadata', 502, 'invalid_backend_response') from exc
        text = metadata['text'] if metadata else result
        asr_text = text
        skipped = bool(gate and not gate.asr_executed)
        stages = []
        if payload.include_provenance:
            try:
                if metadata['requested_language'] != payload.language:
                    raise ValueError('ASR metadata changed the caller language hint')
                stages.append(TranslationStage(stage='asr', model=payload.model,
                    status='skipped' if skipped else 'completed', reason='no_speech_detected' if skipped else None,
                    input_sha256=hashlib.sha256(raw).hexdigest(), output_sha256=text_digest(text), output_text=text,
                    asr_metadata=ASRMetadata.model_validate(metadata)))
            except (ValueError, TypeError) as exc:
                raise EngineError('ASR returned inconsistent stage attribution', 502, 'invalid_backend_response') from exc
        translation_executed = not bool(payload.translation_model) and (gate is None or gate.asr_executed)
        if payload.translation_model and not skipped:
            remaining = math.ceil((deadline - time.monotonic()) * 1000)
            if remaining <= 0:
                raise EngineError("Audio translation exceeded the configured deadline", 504, "deadline_exceeded")
            translated = translate(TranslationRequest(model=payload.translation_model, text=text,
                                  source_language=payload.source_language, target_language=payload.target_language,
                                  worker=payload.translation_worker, timeout=remaining, max_tokens=payload.max_tokens,
                                  include_provenance=payload.include_provenance, location=payload.location),
                                  caller_cancel, deadline)
            text = translated["text"]
            translation_executed = translated.get('translated', True)
            if payload.include_provenance:
                stages.extend(TextTranslationResult.model_validate(translated).stages)
        elif payload.translation_model and payload.include_provenance:
            stages.append(TranslationStage(stage='nmt', model=payload.translation_model, status='skipped', reason='no_speech_detected',
                input_text=asr_text, output_text='', input_sha256=text_digest(asr_text), output_sha256=text_digest('')))
        if payload.include_provenance:
            return AudioTranslationResult(model=payload.model, translation_model=payload.translation_model, text=text,
                input_audio_sha256=hashlib.sha256(raw).hexdigest(), requested_asr_language=metadata['requested_language'],
                observed_asr_language=metadata['language'], nmt_source_language=payload.source_language if payload.translation_model else None,
                nmt_target_language=payload.target_language if payload.translation_model else None,
                translation_executed=translation_executed, stages=stages).model_dump(mode='json')
        response = {"text": text, "model": payload.model, "translation_model": payload.translation_model,
                    "language": "en", "task": "audio_translation"}
        if metadata:
            response.update(duration=metadata['duration'], sample_rate=metadata['sample_rate'],
                decoded_samples=metadata['decoded_samples'], segments=None, words=None,
                timestamps={'source': 'unavailable', 'reason': 'Translation alignment is not supported'},
                quality=metadata['quality'], source_language=metadata['language'])
            if gate:
                response.update(vad_gate=gate.model_dump(mode='json'), duration_source=metadata['duration_source'],
                                translation_executed=translation_executed)
        return response

    CLOUD_STAGES = {'translation': 'text translation through cloud chat (broker-translation-v1 prompt)',
                    'audio_transcription': 'cloud ASR after the local VAD gate',
                    'audio_translation': 'cloud ASR after the local VAD gate; cloud text translation with translation_model',
                    'audio_analysis': 'cloud ASR and text-analysis LLM; diarization and VAD stay local',
                    'sentiment': 'cloud chat LLM (decoding validate)',
                    'sentiment_compose': 'cloud chat LLM role (the classifier stays local)'}

    def runtime_view():
        """runtime 'standalone-local' (unchanged) while no broker is configured; 'local+broker' with the cloud-eligible
        stages and the placement values when broker.json enables a target."""
        if broker is None or not broker.configured:
            return {'runtime': 'standalone-local'}
        return {'runtime': 'local+broker', 'cloud_stages': CLOUD_STAGES,
                'locations': ['local_only', 'auto', 'prefer_local', 'cloud_only'], 'default_location': 'local_only'}

    @app.get("/v1/extensions")
    def extensions():
        operations = {"translation": "translation", "ocr": "image-to-text", "detection": "object-detection", "depth": "depth-estimation", "segmentation":"image-segmentation", "audio_translation": "transcription", 'audio_transcription': 'transcription', 'vad': 'vad', 'diarization':'diarization', 'audio_analysis':'transcription', 'sentiment': 'sentiment', 'sentiment_compose': 'sentiment', 'rerank': 'rerank', 'retrieval_answer':'chat', 'pair_classification':'pair-classification'}
        return {"object": "list", "data": [{"operation": operation, "task": task,
                "models": [name for name, spec in config.models.items() if
                           ((spec.task == 'chat' and spec.backend == 'llama-cpp') if operation=='retrieval_answer' else
                           (spec.task == task or (operation == 'sentiment' and spec.task == 'chat' and spec.backend in {'transformers', 'openvino-genai', 'llama-cpp'})))],
                "availability": "registered models require load and checkpoint compatibility validation"}
                for operation, task in operations.items()], "schema": TypeAdapter(ExtensionRequest).json_schema(),
                **runtime_view(), "media_inputs": "inline bytes only",
                'audio_analysis':{'route':'/v1/audio/analysis','diarization_route':'/v1/audio/diarizations',
                    'speaker_labels':'anonymous_per_request','overlaps':'preserved_interval_intersections',
                    'text_analysis_scope':'transcript_text_only_not_tone_or_identity',
                    'decode_policy':'full_original_bytes_per_stage_shared_limits_and_deadline',
                    'diarization_output_schema':DiarizationResult.model_json_schema(),
                    'output_schema':AudioAnalysisResult.model_json_schema()},
                'pair_classification':{'route':'/v1/classifications/text-pairs','inputs':['premise','hypothesis'],
                    'labels':'native loaded config order; no semantic remapping','score_kind':'softmax_uncalibrated',
                    'truncation':'reject by default; opt-in premise-only truncation preserves hypothesis',
                    'output_schema':PairClassificationResult.model_json_schema()},
                'translation_provenance':{'request_option':'include_provenance', 'default':False,
                    'nmt_scope':'safe single-device Transformers checkpoint bytes verified before/after load; registered repository/revision',
                    'asr_scope':'actual existing ASR backend metadata and registered model ID; no loaded ASR checkpoint fingerprint claimed',
                    'text_result_schema':TextTranslationResult.model_json_schema(),
                    'audio_result_schema':AudioTranslationResult.model_json_schema()},
                "retrieval_answer":{'inputs':'inline bounded passages; no fetching or truncation','stream':False,
                    'prompt_versions':['retrieval-answer-v1','retrieval-answer-v2'],
                    'default_prompt_version':load_settings().retrieval_prompt_version,
                    'prompt_precedence':['request.prompt_version','answer_model.defaults.retrieval_prompt_version','TITANOS_RETRIEVAL_PROMPT_VERSION'],
                    'roles':{'embedding':'single safe Transformers embeddings','rerank':'single safe scalar Transformers reranker','llm':'owned llama-cpp chat'},
                    'models':{role:[name for name,spec in config.models.items() if spec.task==task and spec.backend==backend and spec.parallelism=='single' and not spec.allow_pickle and not spec.trust_remote_code]
                        for role,task,backend in [('embedding','embeddings','transformers'),('rerank','rerank','transformers'),('llm','chat','llama-cpp')]},
                    'citation_integrity':'literal quotation and supplied ID checks','semantic_entailment':'not_automatically_verified'}}

    @app.post("/v1/translations")
    async def translations(payload: TranslationRequest, request: Request):
        return await dispatch_cancellable(request, payload, translate)

    @app.post("/v1/images/ocr")
    async def images_ocr(payload: OCRRequest, request: Request):
        return await dispatch_cancellable(request, payload, ocr)

    @app.post('/v1/images/depth', response_model=DepthResult)
    async def images_depth(payload: DepthRequest, request: Request):
        return await dispatch_cancellable(request, payload, depth)

    @app.post('/v1/images/segment',response_model=SegmentationResult)
    async def images_segment(payload:SegmentationRequest,request:Request):
        return await dispatch_cancellable(request,payload,segment)

    @app.post("/v1/images/detect")
    async def images_detect(payload: DetectionRequest, request: Request):
        return await dispatch_cancellable(request, payload, detect)

    @app.post('/v1/sentiment', response_model=SentimentResult)
    async def text_sentiment(payload: SentimentRequest, request: Request):
        return await dispatch_cancellable(request, payload, sentiment)

    @app.post('/v1/sentiment/compose', response_model=SentimentComposition)
    async def compose_sentiment(payload: SentimentCompositionRequest, request: Request):
        return await dispatch_cancellable(request, payload, sentiment_compose)

    @app.post('/v1/rerank', response_model=RerankResult, response_model_exclude_none=True)
    async def rerank_documents(payload: RerankRequest, request: Request):
        return await dispatch_cancellable(request, payload, rerank)

    @app.post('/v1/retrieval/answer', response_model=RetrievalAnswerResult, response_model_exclude_none=True)
    async def answer_from_documents(payload: RetrievalAnswerRequest, request: Request):
        return await dispatch_cancellable(request,payload,retrieval_answer)

    @app.post('/v1/classifications/text-pairs', response_model=PairClassificationResult)
    async def pair_classification(payload: PairClassificationRequest, request: Request):
        return await dispatch_cancellable(request,payload,classify_pairs)

    @app.post('/v1/audio/diarizations', response_model=DiarizationResult)
    async def audio_diarizations(payload: DiarizationRequest, request: Request):
        return await dispatch_cancellable(request, payload, diarization)

    @app.post('/v1/audio/analysis', response_model=AudioAnalysisResult)
    async def analyze_audio(payload: AudioAnalysisRequest, request: Request):
        return await dispatch_cancellable(request, payload, audio_analysis)

    @app.post('/v1/audio/vad', response_model=VADResult)
    async def audio_vad(payload: VADRequest, request: Request):
        return await dispatch_cancellable(request, payload, vad)

    TRANSCRIPTION_FORM_FIELDS = frozenset({'file', 'model', 'language', 'worker', 'prompt', 'response_format', 'timestamp_granularities[]',
                                           'max_tokens', 'vad_enabled', 'vad_model', 'vad_options', 'timeout'})
    TRANSLATION_FORM_FIELDS = frozenset({'file', 'model', 'language', 'translation_model', 'source_language', 'target_language', 'worker',
                                         'translation_worker', 'response_format', 'include_provenance', 'timestamp_granularities[]',
                                         'vad_enabled', 'vad_model', 'vad_options', 'timeout', 'max_tokens'})

    def refuse_unknown_form_fields(form, allowed):
        # Multipart routes only see declared Form()/File() parameters; upstream knobs such as temperature, include[], stream or
        # chunking_strategy would otherwise be dropped silently. Unknown fields are refused like unknown JSON fields.
        unknown = sorted(set(form.keys()) - set(allowed))
        if unknown:
            raise EngineError('Unsupported multipart fields: ' + ', '.join(unknown) + '; supported: ' + ', '.join(sorted(allowed)), 400, 'unsupported_capability')

    @app.post('/v1/audio/transcriptions', response_model=ASRMetadata | TranscriptionText, response_model_exclude_unset=True)
    async def audio_transcriptions(request: Request, file: Annotated[UploadFile, File()], model: Annotated[str, Form()],
            language: Annotated[str | None, Form()] = None, worker: Annotated[str | None, Form(max_length=128)] = None,
            prompt: Annotated[str | None,Form()] = None,
            response_format: Annotated[Literal['json', 'text', 'verbose_json'], Form()] = 'json',
            timestamp_granularities: Annotated[list[Literal['segment', 'word']] | None, Form(alias='timestamp_granularities[]')] = None,
            max_tokens: Annotated[int | None, Form(gt=0)] = None,
            vad_enabled: Annotated[bool | None, Form()] = None,
            vad_model: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
            vad_options: Annotated[str | None, Form()] = None,
            timeout: Annotated[int | None, Form(gt=0)] = None):
        form=await request.form()
        refuse_unknown_form_fields(form, TRANSCRIPTION_FORM_FIELDS)
        if len(form.getlist('prompt'))>1 or 'prompt' in form and prompt is None:
            raise EngineError('ASR prompt must be supplied once and must not be empty')
        try:
            payload = ASRRequest(model=model, audio_base64='AA==', language=language, worker=worker,
                prompt=prompt,
                response_format='json' if response_format == 'text' else response_format,
                timestamp_granularities=timestamp_granularities or [], max_tokens=max_tokens, timeout=timeout,
                vad_enabled=vad_enabled, vad_model=vad_model,
                vad_options=VADGateOptions.model_validate_json(vad_options) if vad_options is not None else None)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        raw = await file.read(load_settings().max_body_bytes + 1)
        if not raw or len(raw) > load_settings().max_body_bytes:
            raise EngineError('Audio is empty or exceeds TITANOS_MAX_BODY_BYTES', 413, 'request_too_large')
        result = await dispatch_cancellable(request, payload,
            lambda current, cancel, deadline: transcription(current, cancel, deadline, raw))
        return PlainTextResponse(result['text']) if response_format == 'text' else result

    @app.post("/v1/extensions/infer")
    async def extension_infer(payload: ExtensionRequest, request: Request):
        operations = {"translation": translate, "ocr": ocr, "detection": detect, "depth": depth, "segmentation":segment, "audio_translation": translate_audio, 'audio_transcription': transcription, 'vad': vad, 'diarization':diarization, 'audio_analysis':audio_analysis, 'sentiment': sentiment, 'sentiment_compose': sentiment_compose, 'rerank': rerank, 'retrieval_answer':retrieval_answer, 'pair_classification':classify_pairs}
        operation = operations[payload.operation]
        if payload.operation in {'ocr', 'detection', 'depth', 'segmentation', 'translation', 'sentiment', 'sentiment_compose', 'rerank', 'audio_transcription', 'audio_translation', 'vad', 'diarization', 'audio_analysis', 'retrieval_answer', 'pair_classification'}:
            return await dispatch_cancellable(request, payload, operation)
        return await asyncio.to_thread(operation, payload)

    @app.post("/v1/audio/translations")
    async def audio_translations(request: Request, file: Annotated[UploadFile, File()], model: Annotated[str, Form()],
                           language: Annotated[str | None, Form()] = None,
                           translation_model: Annotated[str | None, Form()] = None,
                           source_language: Annotated[str | None, Form()] = None,
                           target_language: Annotated[str | None, Form()] = None,
                           worker: Annotated[str | None, Form()] = None,
                           translation_worker: Annotated[str | None, Form()] = None,
                           response_format: Annotated[Literal["json", "text", 'verbose_json'], Form()] = "json",
                           include_provenance: Annotated[bool, Form()] = False,
                           timestamp_granularities: Annotated[list[str] | None, Form(alias='timestamp_granularities[]')] = None,
                           vad_enabled: Annotated[bool | None, Form()] = None,
                           vad_model: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
                           vad_options: Annotated[str | None, Form()] = None,
                           timeout: Annotated[int | None, Form()] = None,
                           max_tokens: Annotated[int | None, Form()] = None):
        from pydantic import ValidationError
        refuse_unknown_form_fields(await request.form(), TRANSLATION_FORM_FIELDS)
        if timestamp_granularities is not None:
            raise EngineError('Audio translation timestamps are not supported')
        if include_provenance and response_format == 'text':
            raise EngineError('Translation provenance requires a JSON response format')
        try:
            payload = AudioTranslationRequest(model=model, audio_base64="AA==", language=language,
                       translation_model=translation_model, source_language=source_language,
                       target_language=target_language, worker=worker, translation_worker=translation_worker,
                       timeout=timeout, max_tokens=max_tokens, response_format='json' if response_format == 'text' else response_format,
                       include_provenance=include_provenance,
                       vad_enabled=vad_enabled, vad_model=vad_model,
                       vad_options=VADGateOptions.model_validate_json(vad_options) if vad_options is not None else None)
        except ValidationError as exc:
            raise EngineError(str(exc)) from exc
        raw = await file.read(load_settings().max_body_bytes + 1)
        if not raw or len(raw) > load_settings().max_body_bytes:
            raise EngineError("Audio is empty or exceeds TITANOS_MAX_BODY_BYTES", 413, "request_too_large")
        result = await dispatch_cancellable(request, payload,
            lambda current, cancel, deadline: translate_audio(current, cancel, deadline, raw=raw))
        return PlainTextResponse(result["text"]) if response_format == "text" else result

    app.state.inference_run = run
    app.state.dispatch_cancellable = dispatch_cancellable
    app.state.extension_handlers = {"translation": translate, "ocr": ocr, "detection": detect, "depth": depth, "segmentation":segment, "audio_translation": translate_audio, 'audio_transcription': transcription, 'vad': vad, 'diarization':diarization, 'audio_analysis':audio_analysis, 'sentiment': sentiment, 'sentiment_compose': sentiment_compose, 'rerank': rerank, 'retrieval_answer':retrieval_answer, 'pair_classification':classify_pairs}
