"""Validated process settings, with one predictable TITANOS_* environment name.

No framework import or credential introspection. Call load_settings() at process
startup; explicit non-None overrides take precedence over environment values.
"""

from dataclasses import asdict, dataclass, field, fields
from contextvars import ContextVar

_active_settings = ContextVar("titanos_settings", default=None)
import math
import os
import re
import sys


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 11435
    threads: int = field(default_factory=lambda: min(8, os.cpu_count() or 1))
    client_timeout: float = 600.0
    max_body_bytes: int = 64 * 1024 * 1024
    body_read_seconds: float = 30.0
    body_read_min_rate: int = 256 * 1024
    body_read_concurrency: int = 8
    body_read_wait_seconds: float = 2.0
    max_tensor_elements: int = 4_000_000
    max_tokens: int = 8192
    default_max_tokens: int = 256
    default_temperature: float = 0.7
    default_top_p: float = 1.0
    max_images: int = 8
    max_image_dimension: int = 2048
    max_image_pixels: int = 2048 ** 2
    max_input_image_bytes: int = 20 * 1024 * 1024
    max_input_image_pixels: int = 16_000_000
    tesseract_executable: str = ''
    ocr_language: str = 'eng'
    ocr_psm: int = 3
    ocr_max_output_bytes: int = 8 * 1024 * 1024
    ocr_max_tsv_rows: int = 100_000
    ocr_max_model_bytes: int = 512 * 1024 * 1024
    detection_threshold: float = 0.5
    detection_nms: bool = False
    detection_nms_iou: float = 0.7
    max_generated_images: int = 4
    segmentation_max_points: int = 64
    segmentation_max_masks: int = 3
    segmentation_multimask: bool = False
    segmentation_mask_threshold: float = 0.0
    max_steps: int = 100
    default_image_size: str = "512x512"
    default_steps: int = 25
    max_video_frames: int = 129
    default_video_frames: int = 17
    max_video_fps: int = 60
    default_video_fps: int = 8
    video_codec: str = "libx264"
    video_bitrate: str = "2M"
    video_diagnostics_dir: str = ''
    video_diagnostics_max_bytes: int = 256 * 1024 * 1024
    max_audio_seconds: float = 600.0
    max_audio_samples: int = 20_000_000
    max_music_samples: int = 60_000_000
    max_music_output_bytes: int = 128 * 1024 * 1024
    asr_vad_model: str = ''
    asr_vad_mode: str = 'auto'
    vad_sample_rate: int = 16000
    vad_threshold: float = 0.5
    vad_min_speech_ms: float = 250.0
    vad_min_silence_ms: float = 100.0
    vad_speech_pad_ms: float = 30.0
    vad_max_frames: int = 20_000
    diarization_num_speakers: int = 0
    diarization_cluster_threshold: float = 0.5
    diarization_min_duration_on: float = 0.3
    diarization_min_duration_off: float = 0.5
    diarization_max_speakers: int = 32
    diarization_max_chunks: int = 600
    embedding_max_tokens: int = 512
    embedding_max_batch: int = 64
    embedding_truncate: bool = False
    rerank_max_input_tokens: int = 512
    rerank_max_documents: int = 64
    rerank_truncate: bool = False
    pair_max_input_tokens: int = 512
    pair_max_pairs: int = 32
    pair_max_labels: int = 256
    pair_truncate: bool = False
    retrieval_max_documents: int = 32
    retrieval_top_k: int = 8
    retrieval_top_n: int = 3
    retrieval_context_tokens: int = 2048
    retrieval_prompt_version: str = 'retrieval-answer-v2'
    retrieval_max_claims: int = 8
    retrieval_max_citations_per_claim: int = 4
    retrieval_max_quote_chars: int = 2048
    retrieval_max_quote_matches: int = 32
    sentiment_max_input_tokens: int = 512
    sentiment_min_score: float = 0.6
    sentiment_min_margin: float = 0.15
    sentiment_prompt_version: str = 'label-v2'
    sentiment_decoding: str = 'auto'
    max_messages: int = 128
    max_tools: int = 32
    max_stop_sequences: int = 4
    max_stop_chars: int = 128
    max_choices: int = 8
    default_edit_strength: float = 0.75
    default_image_guidance_scale: float = 1.5
    preserve_unmasked: bool = True
    max_text_chars: int = 100_000
    max_prompt_chars: int = 8000
    max_speech_chars: int = 5000
    queue_capacity: int = 8
    job_workers: int = 1
    artifact_ttl_seconds: float = 86400.0
    acquisition_max_bytes: int = 8 * 1024 * 1024 * 1024
    acquisition_workers: int = 1
    acquisition_retention_seconds: float = 7 * 86400.0
    model_page_timeout: float = 15.0
    model_page_max_bytes: int = 2_000_000
    model_page_max_redirects: int = 3
    model_page_max_links: int = 50
    model_page_max_text_chars: int = 2_000
    model_inspection_max_files: int = 4096
    model_inspection_max_metadata_bytes: int = 2_000_000
    model_inspection_max_depth: int = 16
    limit_concurrency: int = 64; health_port: int = 0  # 0: none; else health routes outside the cap (health_listener)
    keep_alive_timeout: int = 5
    log_level: str = "info"; log_file: str = ''; log_max_bytes: int = 64 * 1024 * 1024; log_backups: int = 4; log_max_age_days: float = 0.0  # log_rotation
    native_context_size: int = 8192; native_max_slot_context: int = 0  # 0: the GGUF trained context (native_context)
    native_slots: int = 4
    native_threads: int = 4
    native_batch_size: int = 0
    native_ubatch_size: int = 0
    native_gpu_layers: int | str = 'auto'; exclude_gpu_uuids: str = ''  # comma list; never placed on (native placement)
    device_probe_timeout: float = 30.0
    device_inventory_ttl: float = 30.0; native_admission: str = 'auto'  # off, auto or on (titanos.admission)
    native_startup_timeout: int = 300; admission_reserve_mib: int = 0; admission_free_mib: int = 0  # titanos.admission
    native_queue_capacity: int = 32
    native_queue_timeout: float = 30.0
    replica_queue_capacity: int = 32
    replica_queue_timeout: float = 30.0
    worker_threads: int = 4
    worker_timeout: float = 600.0
    # Cloud broker (src/titanos/broker): non-secret paths, switches and bounds only; credentials never live in Settings.
    role: str = 'standalone'
    broker_config: str = ''
    broker_cloud_ceiling: str = 'fallback'
    broker_egress: str = 'internet'
    broker_vault_backend: str = 'auto'
    broker_vault_passphrase_file: str = ''
    broker_connect_timeout: float = 10.0
    broker_first_byte_timeout: float = 120.0
    broker_stream_idle_timeout: float = 120.0
    broker_max_fallbacks: int = 2
    broker_breaker_failures: int = 3
    broker_breaker_base_seconds: float = 2.0
    broker_breaker_max_seconds: float = 300.0
    broker_health_ttl: float = 300.0
    broker_probe_idle_cutoff: float = 900.0
    broker_catalog_ttl: float = 3600.0
    broker_advisor_timeout_ms: int = 300
    broker_decision_log_size: int = 1000
    broker_decision_log_file: bool = False
    broker_usage_retention_days: int = 90
    broker_audit_retention_days: int = 365
    broker_secret_cache_seconds: float = 300.0
    broker_debug_upstream_urls: bool = False
    broker_default_max_output_tokens: int = 4096
    broker_max_upstream_json_bytes: int = 16 * 1024 * 1024
    broker_max_upstream_media_bytes: int = 64 * 1024 * 1024
    broker_auth_failures_per_minute: int = 20
    tls_certfile: str = ''
    tls_keyfile: str = ''
    behind_tls_proxy: bool = False
    managed_delegation: str = 'off'
    managed_deadline_headers: str = 'x-titanos-deadline-ms'
    managed_trace_headers: str = 'x-request-id,traceparent,x-hivemind-trace'
    unknown_fields: str = 'strict'  # ignore: managed role only (titanos.request_compat.unknown_fields)

    def __post_init__(self):
        nonnegative = {"default_temperature", "native_max_slot_context", "log_backups", "log_max_age_days", "health_port", "admission_reserve_mib", "admission_free_mib", "native_gpu_layers", "native_batch_size", "native_ubatch_size", "keep_alive_timeout", "sentiment_min_score", "sentiment_min_margin", 'vad_min_speech_ms', 'vad_speech_pad_ms', 'detection_threshold', 'detection_nms_iou','model_page_max_redirects', 'diarization_num_speakers', 'diarization_cluster_threshold', 'diarization_min_duration_on', 'diarization_min_duration_off', 'broker_max_fallbacks'}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == 'native_gpu_layers':
                if not (type(value) is int and value >= 0 or type(value) is str and value in {'auto', 'all'}):
                    raise ValueError('TITANOS_NATIVE_GPU_LAYERS must be auto, all or a nonnegative integer')
                continue
            if f.type is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"TITANOS_{f.name.upper()} must be a boolean")
            elif f.type in {int, float}:
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or (isinstance(value, float) and not math.isfinite(value))
                        or (f.type is float and isinstance(value, int) and abs(value) > sys.float_info.max)):
                    raise ValueError(f"TITANOS_{f.name.upper()} must be a finite number")
                if f.type is int and not isinstance(value, int):
                    raise ValueError(f"TITANOS_{f.name.upper()} must be an integer")
                if f.name != 'segmentation_mask_threshold' and (value < 0 or (value == 0 and f.name not in nonnegative)):
                    raise ValueError(f"TITANOS_{f.name.upper()} must be {'nonnegative' if f.name in nonnegative else 'positive'}")
            elif not isinstance(value, str) or (not value.strip() and f.name not in {'asr_vad_model', 'tesseract_executable', 'video_diagnostics_dir', 'broker_config', 'broker_vault_passphrase_file', 'tls_certfile', 'tls_keyfile', 'exclude_gpu_uuids', 'log_file'}):
                raise ValueError(f"TITANOS_{f.name.upper()} must be a nonempty string")
        if self.video_diagnostics_dir:
            from pathlib import Path
            if not Path(self.video_diagnostics_dir).expanduser().is_absolute():
                raise ValueError('TITANOS_VIDEO_DIAGNOSTICS_DIR must be an absolute operator-owned directory')
        if (self.model_page_timeout>60 or self.model_page_max_bytes>8_000_000
                or self.model_page_max_redirects>5 or self.model_page_max_links>200
                or self.model_page_max_text_chars>10_000):
            raise ValueError('Model-page fetch settings exceed their safe operator ceilings')
        if self.ocr_psm not in {3,4,5,6,7,8,9,10,11,13}:
            raise ValueError('TITANOS_OCR_PSM requires a non-OSD recognition mode: 3..11 or 13')
        if len(self.ocr_language) > 128 or not re.fullmatch(r'[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*', self.ocr_language):
            raise ValueError('TITANOS_OCR_LANGUAGE requires local traineddata language IDs joined with +')
        if self.asr_vad_mode not in {'auto', 'on', 'off'}:
            raise ValueError('TITANOS_ASR_VAD_MODE must be auto, on or off')
        if self.pair_max_labels < 2:
            raise ValueError('TITANOS_PAIR_MAX_LABELS must permit at least two native classes')
        from .retrieval_prompts import validate_version
        validate_version(self.retrieval_prompt_version)
        if len(self.asr_vad_model) > 128 or (self.asr_vad_model and not self.asr_vad_model.strip()):
            raise ValueError('TITANOS_ASR_VAD_MODEL must be empty or a registered model name of at most128 characters')
        if self.port > 65535:
            raise ValueError("TITANOS_PORT must be between 1 and 65535")
        if self.vad_sample_rate not in {8000, 16000} or not 0 < self.vad_threshold < 1:
            raise ValueError('VAD sample rate must be 8000/16000 and threshold between zero and one')
        if self.diarization_num_speakers > self.diarization_max_speakers:
            raise ValueError('TITANOS_DIARIZATION_NUM_SPEAKERS exceeds TITANOS_DIARIZATION_MAX_SPEAKERS')
        if max(self.diarization_min_duration_on, self.diarization_min_duration_off) > self.max_audio_seconds:
            raise ValueError('Diarization minimum durations exceed TITANOS_MAX_AUDIO_SECONDS')
        if self.default_edit_strength > 1:
            raise ValueError("TITANOS_DEFAULT_EDIT_STRENGTH must be at most 1")
        if self.default_image_guidance_scale < 1:
            raise ValueError('TITANOS_DEFAULT_IMAGE_GUIDANCE_SCALE must be at least one')
        if self.detection_threshold > 1 or self.detection_nms_iou > 1:
            raise ValueError('Detection score and NMS IoU thresholds must be at most one')
        if self.default_top_p > 1 or self.default_temperature > 2:
            raise ValueError("Default top_p must be at most 1 and temperature at most 2")
        if self.sentiment_min_score > 1 or self.sentiment_min_margin > 1:
            raise ValueError('Sentiment score and margin thresholds must be at most one')
        if self.sentiment_prompt_version not in {'weights-v1', 'label-v2', 'label-v3', 'facets-v1'}:
            raise ValueError('TITANOS_SENTIMENT_PROMPT_VERSION must be weights-v1, label-v2, label-v3 or facets-v1')
        if self.sentiment_decoding not in {'auto', 'validate', 'label_choices', 'facet_choices', 'native_json_schema'}:
            raise ValueError('TITANOS_SENTIMENT_DECODING must be auto, validate, label_choices, facet_choices or native_json_schema')
        if self.native_batch_size and self.native_ubatch_size > self.native_batch_size:
            raise ValueError('TITANOS_NATIVE_UBATCH_SIZE cannot exceed TITANOS_NATIVE_BATCH_SIZE')
        if self.log_level not in {"critical", "error", "warning", "info", "debug", "trace"}:
            raise ValueError("TITANOS_LOG_LEVEL must be critical, error, warning, info, debug or trace")
        for default, maximum in (("default_max_tokens", "max_tokens"), ("default_steps", "max_steps"),
                                 ("default_video_frames", "max_video_frames"), ("default_video_fps", "max_video_fps")):
            if getattr(self, default) > getattr(self, maximum):
                raise ValueError(f"TITANOS_{default.upper()} exceeds TITANOS_{maximum.upper()}")
        if not re.fullmatch(r"\d+x\d+", self.default_image_size):
            raise ValueError("TITANOS_DEFAULT_IMAGE_SIZE must be WIDTHxHEIGHT")
        width, height = map(int, self.default_image_size.split("x"))
        if any(v < 32 or v % 8 or v > self.max_image_dimension for v in (width, height)) or width * height > self.max_image_pixels:
            raise ValueError("Default image dimensions must be multiples of 8, at least 32, and within configured dimension/pixel limits")
        if self.native_context_size < 512:
            raise ValueError("TITANOS_NATIVE_CONTEXT_SIZE must be at least 512")
        self._check_broker(); _check_engine(self)

    def _check_broker(self):
        """Broker settings: choices, paths, the TLS pair and the bounded fallback/advisor knobs (no I/O, no secret)."""
        for name, choices in (('role', ('standalone', 'broker', 'managed')),
                              ('broker_cloud_ceiling', ('off', 'explicit', 'fallback')),
                              ('broker_egress', ('internet', 'lan', 'loopback-only')),
                              ('broker_vault_backend', ('auto', 'dpapi', 'keychain', 'secret-service', 'sealed-file', 'refs-only')),
                              ('managed_delegation', ('off', 'parent')), ('unknown_fields', ('strict', 'ignore'))):
            if getattr(self, name) not in choices:
                raise ValueError(f"TITANOS_{name.upper()} must be one of {', '.join(choices)}")
        if bool(self.tls_certfile) != bool(self.tls_keyfile):
            raise ValueError('TITANOS_TLS_CERTFILE and TITANOS_TLS_KEYFILE must be set together')
        from pathlib import Path
        for name in ('broker_config', 'broker_vault_passphrase_file', 'tls_certfile', 'tls_keyfile'):
            value = getattr(self, name)
            if value and (value != value.strip() or not Path(value).expanduser().is_absolute()):
                raise ValueError(f'TITANOS_{name.upper()} must be empty or an absolute path')
        if self.broker_vault_backend == 'sealed-file' and not self.broker_vault_passphrase_file:
            raise ValueError('TITANOS_BROKER_VAULT_BACKEND=sealed-file requires TITANOS_BROKER_VAULT_PASSPHRASE_FILE')
        if self.broker_max_fallbacks > 5:
            raise ValueError('TITANOS_BROKER_MAX_FALLBACKS must be at most 5')
        if self.broker_advisor_timeout_ms > 5000:
            raise ValueError('TITANOS_BROKER_ADVISOR_TIMEOUT_MS must be at most 5000')
        if self.broker_breaker_base_seconds > self.broker_breaker_max_seconds:
            raise ValueError('TITANOS_BROKER_BREAKER_BASE_SECONDS exceeds TITANOS_BROKER_BREAKER_MAX_SECONDS')
        for name in ('managed_deadline_headers', 'managed_trace_headers'):
            names = [part.strip() for part in getattr(self, name).split(',')]
            if not all(re.fullmatch(r'[A-Za-z0-9-]{1,64}', part) for part in names) or len(names) > 16:
                raise ValueError(f'TITANOS_{name.upper()} must be a comma list of at most 16 HTTP header names')
        if self.behind_tls_proxy and not _loopback_host(self.host):
            raise ValueError('TITANOS_BEHIND_TLS_PROXY is valid only with a loopback bind')

    @classmethod
    def from_env(cls, environ=None, **overrides):
        env = os.environ if environ is None else environ
        values = {}
        known = {f.name for f in fields(cls)}
        if set(overrides) - known:
            raise ValueError("Unknown settings: " + ", ".join(sorted(set(overrides) - known)))
        for f in fields(cls):
            key = "TITANOS_" + f.name.upper()
            if overrides.get(f.name) is not None:
                values[f.name] = overrides[f.name]
            elif key in env:
                try:
                    if f.name == 'native_gpu_layers':
                        literal = env[key].strip().lower()
                        values[f.name] = literal if literal in {'auto', 'all'} else int(literal)
                    elif f.type is bool:
                        literal = env[key].strip().lower()
                        if literal not in {'true', 'false', '1', '0'}:
                            raise ValueError('Use true, false, 1 or 0')
                        values[f.name] = literal in {'true', '1'}
                    else:
                        values[f.name] = f.type(env[key].strip())
                except (TypeError, ValueError, AttributeError) as exc:
                    kind = 'integer, auto or all' if f.name == 'native_gpu_layers' else f.type.__name__
                    raise ValueError(f"{key} must be a valid {kind}") from exc
        return cls(**values)

    def describe(self):
        """JSON-safe current values, defaults and environment names, never keys."""
        defaults = asdict(type(self)())
        return {"settings": {f.name: {"value": getattr(self, f.name), "default": defaults[f.name],
                                      "env": "TITANOS_" + f.name.upper(),
                                      "type": 'int_or_auto_all' if f.name == 'native_gpu_layers' else f.type.__name__}
                             for f in fields(self)}}


def _loopback_host(host):
    """True for a loopback IP literal or the name localhost (the bind rule of `titanos serve`)."""
    import ipaddress
    try:
        return ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        return host == 'localhost'


def load_settings(**overrides):
    return Settings.from_env(**overrides)


def current_settings():
    return _active_settings.get() or load_settings()


def _check_engine(settings):
    """Engine settings checked after the broker's (Settings.__post_init__): the per-slot context ceiling."""
    if 0 < settings.native_max_slot_context < 512:
        raise ValueError("TITANOS_NATIVE_MAX_SLOT_CONTEXT must be 0 (each GGUF's trained context) or at least 512")
    if settings.health_port > 65535 or settings.health_port and settings.health_port == settings.port:
        raise ValueError('TITANOS_HEALTH_PORT must be 0 (none) or a port other than TITANOS_PORT, at most 65535')
    if settings.log_file and not os.path.isabs(os.path.expanduser(settings.log_file)):
        raise ValueError('TITANOS_LOG_FILE must be an absolute path')
    if settings.log_max_bytes < 4096:
        raise ValueError('TITANOS_LOG_MAX_BYTES must be at least 4096')
    if settings.native_admission not in ('off', 'auto', 'on'):
        raise ValueError('TITANOS_NATIVE_ADMISSION must be off, auto or on')
    from .devices import normalized_uuid
    for item in filter(None, (part.strip() for part in settings.exclude_gpu_uuids.split(','))):
        if not item.startswith('GPU-') or not normalized_uuid(item.removeprefix('GPU-')):
            raise ValueError('TITANOS_EXCLUDE_GPU_UUIDS must be a comma list of complete NVIDIA GPU UUIDs (GPU-...)')
