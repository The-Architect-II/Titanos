"""Opt-in translation attribution; tokenizer IDs are not detected audio languages."""
import hashlib
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .audio_contract import ASRMetadata


def text_digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


class Value(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class NMTProvenance(Value):
    repository: str | None
    revision: str | None
    checkpoint_digest: str = Field(pattern=r'^[a-f0-9]{64}$')
    backend: Literal['transformers']
    device: str = Field(min_length=1)
    model_class: str = Field(min_length=1)
    tokenizer_class: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    identity_scope: Literal['checkpoint_bytes_verified_before_and_after_load'] = 'checkpoint_bytes_verified_before_and_after_load'
    identifiers_are_registered: Literal[True] = True


class LanguageSelection(Value):
    mode: Literal['fixed_pair', 'multilingual']
    source_language: str = Field(min_length=1)
    target_language: str = Field(min_length=1)
    registered_source_language: str | None = None
    registered_target_language: str | None = None
    tokenizer_source_language: str | None
    tokenizer_target_language: str | None
    forced_bos_token_id: int | None = Field(default=None, strict=True, ge=0)
    translation_prefix: str = ''
    language_detection_performed: Literal[False] = False

    @model_validator(mode='after')
    def supported_selection(self):
        if self.mode == 'fixed_pair':
            if (self.registered_source_language, self.registered_target_language) != (self.source_language, self.target_language):
                raise ValueError('Fixed-direction translation must match the declared language pair')
            if ((self.tokenizer_source_language is not None and self.tokenizer_source_language != self.source_language)
                    or (self.tokenizer_target_language is not None and self.tokenizer_target_language != self.target_language)):
                raise ValueError('Fixed-direction tokenizer choices differ from the declared pair')
        elif (self.forced_bos_token_id is None or self.tokenizer_source_language != self.source_language
                or self.tokenizer_target_language != self.target_language):
            raise ValueError('Multilingual selection must bind the source and forced target token')
        return self


class NMTMeasurements(Value):
    processed_input_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    input_tokens: int = Field(strict=True, ge=1)
    input_token_limit: int = Field(strict=True, ge=1)
    output_sequence_tokens: int = Field(strict=True, ge=1)
    output_sequence_includes_special_tokens: Literal[True] = True
    inference_seconds: float = Field(strict=True, ge=0)
    timing_scope: Literal['tokenization_generation_decode_excludes_model_load'] = 'tokenization_generation_decode_excludes_model_load'
    input_truncated: Literal[False] = False

    @model_validator(mode='after')
    def within_limit(self):
        if self.input_tokens > self.input_token_limit:
            raise ValueError('NMT consumed inputs exceed the reported untruncated limit')
        return self


class TranslationStage(Value):
    stage: Literal['asr', 'nmt']
    model: str = Field(min_length=1)
    status: Literal['completed', 'passthrough', 'skipped']
    reason: Literal['equal_language_ids', 'no_speech_detected'] | None = None
    input_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    output_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    input_text: str | None = None
    output_text: str
    provenance: NMTProvenance | None = None
    language_selection: LanguageSelection | None = None
    measurements: NMTMeasurements | None = None
    asr_metadata: ASRMetadata | None = None

    @model_validator(mode='after')
    def coherent(self):
        if text_digest(self.output_text) != self.output_sha256:
            raise ValueError('Stage output text does not match its hash')
        if self.input_text is not None and text_digest(self.input_text) != self.input_sha256:
            raise ValueError('Stage input text does not match its hash')
        if self.status != 'completed':
            if self.reason is None or any(value is not None for value in (self.provenance, self.language_selection, self.measurements)):
                raise ValueError('Skipped/passthrough stages cannot invent loaded NMT provenance or measurements')
            if self.status == 'passthrough' and (self.stage != 'nmt' or self.reason != 'equal_language_ids' or self.input_text != self.output_text):
                raise ValueError('Equal-language passthrough must preserve the exact input')
            if self.status == 'skipped' and (self.reason != 'no_speech_detected' or self.output_text):
                raise ValueError('No-speech stages return empty output')
        elif self.reason is not None:
            raise ValueError('Completed stages do not have a skip reason')
        if self.stage == 'nmt':
            if self.input_text is None or self.asr_metadata is not None:
                raise ValueError('NMT stages require exact text input, not ASR metadata')
            if self.status == 'completed' and any(value is None for value in (self.provenance, self.language_selection, self.measurements)):
                raise ValueError('Completed NMT stages require loaded provenance, tokenizer choices and measurements')
            if self.status == 'completed' and self.measurements.processed_input_sha256 != text_digest(self.language_selection.translation_prefix + self.input_text):
                raise ValueError('NMT processed-input identity differs from its explicit tokenizer prefix')
        elif self.asr_metadata is None or self.asr_metadata.text != self.output_text:
            raise ValueError('ASR stage must preserve its real backend metadata/text')
        elif any(value is not None for value in (self.provenance, self.language_selection, self.measurements)):
            raise ValueError('ASR metadata cannot be labeled as loaded NMT provenance')
        elif self.status == 'completed' and not self.asr_metadata.backend:
            raise ValueError('Completed ASR stage requires an observed backend')
        elif self.status == 'skipped' and (self.asr_metadata.backend is not None or self.asr_metadata.vad_gate is None or self.asr_metadata.vad_gate.asr_executed):
            raise ValueError('Skipped ASR must be attributed to its actual no-speech gate')
        return self


class TextTranslationResult(Value):
    object: Literal['text.translation'] = 'text.translation'
    schema_version: Literal['1'] = '1'
    model: str
    text: str
    translations: list[dict[str, str]]
    source_language: str
    target_language: str
    requested_source_language: str | None
    source_language_origin: Literal['request', 'model_default', 'request_default']
    translated: bool
    stages: list[TranslationStage]

    @model_validator(mode='after')
    def coherent(self):
        if len(self.stages) != 1 or self.stages[0].stage != 'nmt':
            raise ValueError('Text translation has exactly one NMT stage')
        stage = self.stages[0]
        if stage.model != self.model or stage.output_text != self.text or self.translations != [{'text': self.text}]:
            raise ValueError('Text translation differs from its stage output')
        if self.translated != (stage.status == 'completed'):
            raise ValueError('Translation status differs from actual NMT execution')
        if (self.source_language_origin == 'request') != (self.requested_source_language is not None):
            raise ValueError('Requested source-language identity must retain omission')
        if self.requested_source_language is not None and self.requested_source_language != self.source_language:
            raise ValueError('Explicit caller source-language ID must not be rewritten')
        if stage.language_selection and (stage.language_selection.source_language != self.source_language or stage.language_selection.target_language != self.target_language):
            raise ValueError('Tokenizer choices differ from requested NMT language IDs')
        return self


class AudioTranslationResult(Value):
    object: Literal['audio.translation'] = 'audio.translation'
    schema_version: Literal['1'] = '1'
    task: Literal['audio_translation'] = 'audio_translation'
    model: str
    translation_model: str | None
    text: str
    language: Literal['en'] = 'en'
    input_audio_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    requested_asr_language: str | None
    observed_asr_language: str | None
    nmt_source_language: str | None
    nmt_target_language: str | None
    translation_executed: bool
    stages: list[TranslationStage]

    @model_validator(mode='after')
    def coherent(self):
        expected = ['asr', 'nmt'] if self.translation_model is not None else ['asr']
        if [stage.stage for stage in self.stages] != expected:
            raise ValueError('Audio translation stages do not match its requested composition')
        asr = self.stages[0]
        if asr.asr_metadata.task != ('transcribe' if self.translation_model else 'translate'):
            raise ValueError('ASR task must match direct translation versus ASR-to-NMT composition')
        if asr.model != self.model or asr.input_sha256 != self.input_audio_sha256 or self.stages[-1].output_text != self.text:
            raise ValueError('Audio translation differs from its attributed stages')
        if self.observed_asr_language != asr.asr_metadata.language or self.requested_asr_language != asr.asr_metadata.requested_language:
            raise ValueError('Observed and requested ASR language must preserve backend metadata')
        if self.translation_model:
            nmt = self.stages[1]
            if nmt.model != self.translation_model or nmt.input_text != asr.output_text or nmt.input_sha256 != asr.output_sha256:
                raise ValueError('NMT must receive the exact ASR output without rewriting')
            if nmt.language_selection and (nmt.language_selection.source_language != self.nmt_source_language or nmt.language_selection.target_language != self.nmt_target_language):
                raise ValueError('Audio NMT language IDs differ from tokenizer choices')
            executed = nmt.status == 'completed'
        else:
            if self.nmt_source_language is not None or self.nmt_target_language is not None:
                raise ValueError('Direct ASR translation does not invent NMT language IDs')
            executed = asr.status == 'completed'
        if self.translation_executed != executed:
            raise ValueError('Translation execution differs from stage status')
        return self
