"""Configured ASR gate selection. No framework imports or implicit model discovery."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class VADGateOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    worker: str | None = Field(default=None, min_length=1, max_length=128)
    sample_rate: Literal[8000, 16000] | None = None
    threshold: float | None = Field(default=None, strict=True, gt=0, lt=1)
    min_speech_ms: float | None = Field(default=None, strict=True, ge=0)
    min_silence_ms: float | None = Field(default=None, strict=True, gt=0)
    speech_pad_ms: float | None = Field(default=None, strict=True, ge=0)


class ASRGateFields(BaseModel):
    vad_enabled: bool | None = Field(default=None, strict=True, description='Omitted inherits; false explicitly disables the configured VAD gate')
    vad_model: str | None = Field(default=None, strict=True, min_length=1, max_length=128, description='Explicit registered VAD model; no model is downloaded or discovered implicitly')
    vad_options: VADGateOptions | None = None


GATE_FIELDS = {'vad_enabled', 'vad_model', 'vad_options'}


def resolve_gate(request, defaults, settings):
    """Resolve omission separately from false, and merge only explicit VAD knobs."""
    parsed = ASRGateFields.model_validate({k:v for k,v in defaults.items() if k in GATE_FIELDS})
    request = ASRGateFields.model_validate({k:v for k,v in request.items() if k in GATE_FIELDS})
    model = request.vad_model if request.vad_model is not None else parsed.vad_model if parsed.vad_model is not None else settings.asr_vad_model or None
    model_source = 'request' if request.vad_model is not None else 'model_default' if parsed.vad_model is not None else 'process_default'
    enabled = request.vad_enabled if request.vad_enabled is not None else parsed.vad_enabled
    enabled_source = 'request' if request.vad_enabled is not None else 'model_default' if parsed.vad_enabled is not None else 'process_default'
    if enabled is None:
        enabled = bool(model) if settings.asr_vad_mode == 'auto' else settings.asr_vad_mode == 'on'
    if enabled and not model:
        raise ValueError('Enabled ASR VAD gating requires an explicit registered vad_model')
    options = parsed.vad_options.model_dump(exclude_none=True) if parsed.vad_options else {}
    if request.vad_options:
        options.update(request.vad_options.model_dump(exclude_none=True))
    if not enabled and request.vad_options is not None and request.vad_enabled is not False:
        raise ValueError('vad_options require an enabled, configured VAD gate')
    return {'enabled':enabled,'model':model,'options':options,'model_source':model_source,'enabled_source':enabled_source}
