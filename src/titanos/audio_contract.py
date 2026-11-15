"""Typed standalone audio activity measurements; no identity or affect inference."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .sentiment_contract import SentimentResult


class AudioResult(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class SpeechInterval(AudioResult):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)


class VADFrame(SpeechInterval):
    speech_probability: float = Field(ge=0, le=1)
    padded_samples: int = Field(ge=0)


class VADResult(AudioResult):
    model: str
    task: Literal['vad'] = 'vad'
    sample_rate: Literal[8000, 16000]
    duration: float = Field(gt=0)
    decoded_samples: int = Field(gt=0)
    frame_samples: int = Field(gt=0)
    context_samples: int = Field(gt=0)
    state_reset: Literal['per_request'] = 'per_request'
    score_kind: Literal['speech_probability_uncalibrated'] = 'speech_probability_uncalibrated'
    segments: list[SpeechInterval]
    frames: list[VADFrame] | None = None
    parameters: dict
    input_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    provenance: dict

    @model_validator(mode='after')
    def coherent(self):
        if abs(self.duration - self.decoded_samples / self.sample_rate) > 1e-8:
            raise ValueError('VAD duration and sample count disagree')
        for collection in [self.segments, self.frames or []]:
            prior = 0
            for item in collection:
                if not prior <= item.start_sample < item.end_sample <= self.decoded_samples:
                    raise ValueError('VAD intervals must be ordered, disjoint and inside the decoded input')
                if abs(item.start - item.start_sample / self.sample_rate) > 1e-8 or abs(item.end - item.end_sample / self.sample_rate) > 1e-8:
                    raise ValueError('VAD seconds and sample coordinates disagree')
                prior = item.end_sample
        return self


class TranscriptionText(AudioResult):
    text: str


class ASRSegment(AudioResult):
    id: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    text: str


class ASRWord(AudioResult):
    id: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    word: str


class ASRVADGate(AudioResult):
    model: str
    decision: Literal['no_speech_detected', 'speech_detected']
    asr_executed: bool
    audio_policy: Literal['full_original_audio'] = 'full_original_audio'
    model_source: Literal['request', 'model_default', 'process_default']
    enabled_source: Literal['request', 'model_default', 'process_default']
    measurement: VADResult


class ASRPromptConditioning(AudioResult):
    source: Literal['request','model_default']
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    characters: int = Field(gt=0)
    text_tokens: int = Field(gt=0)
    prompt_id_tokens: int = Field(gt=1)
    context_token_limit: int = Field(gt=0)
    decoder_prefix_budget: int = Field(gt=0)
    decoder_capacity: int = Field(gt=0)
    max_new_tokens: int | None = Field(default=None,gt=0)
    previous_token_reserve: int = Field(gt=0)
    available_new_tokens: int = Field(gt=0)
    conditioning: Literal['first-segment']='first-segment'
    truncated: Literal[False]=False

    @model_validator(mode='after')
    def coherent_budget(self):
        if (self.prompt_id_tokens!=self.text_tokens+1 or self.text_tokens>self.context_token_limit
                or self.context_token_limit!=self.decoder_capacity//2-1 or self.previous_token_reserve<self.prompt_id_tokens
                or self.decoder_prefix_budget+self.previous_token_reserve+self.available_new_tokens!=self.decoder_capacity
                or self.max_new_tokens is not None and self.max_new_tokens>self.available_new_tokens):
            raise ValueError('Whisper prompt conditioning reports an inconsistent decoder budget')
        return self


class ASRMetadata(AudioResult):
    task: Literal['transcribe', 'translate']
    text: str
    duration: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    decoded_samples: int = Field(gt=0)
    language: str | None
    language_source: Literal['backend', 'unavailable']
    requested_language: str | None
    segments: list[ASRSegment] | None
    words: list[ASRWord] | None
    timestamps: dict
    quality: dict
    backend: str | None
    prompt_conditioning: ASRPromptConditioning | None = None
    vad_gate: ASRVADGate | None = None
    duration_source: Literal['asr_decoded_audio', 'vad_decoded_audio'] | None = None


class DiarizationOptions(AudioResult):
    num_speakers: int | None = Field(default=None, strict=True, ge=0)
    cluster_threshold: float | None = Field(default=None, strict=True, ge=0)
    min_duration_on: float | None = Field(default=None, strict=True, ge=0)
    min_duration_off: float | None = Field(default=None, strict=True, ge=0)


class AnonymousSpeaker(AudioResult):
    id: str = Field(pattern=r'^speaker_[0-9]+$')


class DiarizationSegment(SpeechInterval):
    speaker: str = Field(pattern=r'^speaker_[0-9]+$')


class SpeakerOverlap(SpeechInterval):
    speakers: list[str] = Field(min_length=2)


def speaker_overlaps(segments, sample_rate=16000):
    """Exact half-open intersections; a change of active speakers starts a new span."""
    from collections import Counter
    events = {}
    for segment in segments:
        row = segment.model_dump() if isinstance(segment, BaseModel) else segment
        events.setdefault(row['start_sample'], []).append((row['speaker'], 1))
        events.setdefault(row['end_sample'], []).append((row['speaker'], -1))
    active, previous, output = Counter(), None, []
    for point in sorted(events):
        speakers = sorted((name for name, count in active.items() if count > 0), key=lambda name: int(name.split('_')[1]))
        if previous is not None and point > previous and len(speakers) > 1:
            if output and output[-1]['end_sample'] == previous and output[-1]['speakers'] == speakers:
                output[-1].update(end_sample=point, end=point / sample_rate)
            else:
                output.append({'speakers': speakers, 'start': previous / sample_rate, 'end': point / sample_rate,
                               'start_sample': previous, 'end_sample': point})
        for name, change in events[point]:
            active[name] += change
        previous = point
    return output


class DiarizationResult(AudioResult):
    model: str
    task: Literal['diarization'] = 'diarization'
    sample_rate: Literal[16000] = 16000
    duration: float = Field(gt=0)
    decoded_samples: int = Field(gt=0)
    input_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    speakers: list[AnonymousSpeaker]
    segments: list[DiarizationSegment]
    overlaps: list[SpeakerOverlap]
    parameters: dict
    provenance: dict

    @model_validator(mode='after')
    def coherent(self):
        if abs(self.duration - self.decoded_samples / self.sample_rate) > 1e-8:
            raise ValueError('Diarization duration and sample count disagree')
        names = [speaker.id for speaker in self.speakers]
        if names != ['speaker_' + str(i) for i in range(len(names))]:
            raise ValueError('Speaker labels must be contiguous anonymous request-local IDs')
        prior = -1
        for segment in self.segments:
            if (segment.speaker not in names or segment.start_sample < prior
                    or not 0 <= segment.start_sample < segment.end_sample <= self.decoded_samples):
                raise ValueError('Speaker intervals must be ordered and inside the decoded input')
            if (abs(segment.start - segment.start_sample / self.sample_rate) > 1e-8
                    or abs(segment.end - segment.end_sample / self.sample_rate) > 1e-8):
                raise ValueError('Speaker seconds and sample coordinates disagree')
            prior = segment.start_sample
        if set(names) != {segment.speaker for segment in self.segments}:
            raise ValueError('Speaker inventory must match the actual interval labels')
        if [row.model_dump() for row in self.overlaps] != speaker_overlaps(self.segments, self.sample_rate):
            raise ValueError('Overlaps must preserve the exact simultaneous speaker intervals')
        return self


def timestamp_attribution(start, end, duration):
    """Classify raw ASR coordinates without clipping or inventing alignment."""
    import math
    for value in (start, end):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError('ASR timestamps must be finite nonnegative numbers or null')
    if start is not None and end is not None and end < start:
        raise ValueError('ASR timestamps must not be reversed')
    if any(value is not None and value > duration for value in (start, end)):
        return 'timestamps_outside_audio'
    if start is None or end is None:
        return 'timestamps_unavailable'
    return 'interval_intersection'


def attribution_status(rows):
    usable = sum(row.attribution == 'interval_intersection' for row in rows)
    return 'complete' if rows and usable == len(rows) else 'partial' if usable else 'unavailable'


class AttributedASRSegment(ASRSegment):
    speakers: list[str]
    overlap: bool
    attribution: Literal['interval_intersection', 'timestamps_unavailable', 'timestamps_outside_audio']
    granularity: Literal['segment', 'word']


class AudioAnalysisResult(AudioResult):
    model: str
    task: Literal['audio_analysis'] = 'audio_analysis'
    text: str
    transcription: ASRMetadata
    diarization: DiarizationResult
    speaker_segments: list[AttributedASRSegment]
    speaker_alignment: Literal['segment', 'word', 'unavailable']
    speaker_attribution_status: Literal['complete', 'partial', 'unavailable']
    text_analysis: SentimentResult | None = None
    text_analysis_scope: Literal['transcript_text_only'] = 'transcript_text_only'
    text_analysis_status: Literal['not_requested', 'completed', 'not_run_empty_transcript']
    attribution_policy: Literal['intersect_backend_timestamps_no_per_speaker_transcript'] = 'intersect_backend_timestamps_no_per_speaker_transcript'

    @model_validator(mode='after')
    def consistent(self):
        if self.text != self.transcription.text:
            raise ValueError('Composed text must preserve the original ASR transcript')
        if (self.text_analysis_status == 'completed') != (self.text_analysis is not None):
            raise ValueError('Text analysis state and result disagree')
        if self.text_analysis_status == 'not_run_empty_transcript' and self.text.strip():
            raise ValueError('Only an empty transcript can skip requested text analysis')
        asr = self.transcription
        tolerance = max(1 / asr.sample_rate, 1 / self.diarization.sample_rate)
        if (abs(asr.duration - asr.decoded_samples / asr.sample_rate) > 1e-8
                or abs(asr.duration - self.diarization.duration) > tolerance):
            raise ValueError('Audio stages must describe the same decoded duration')
        if asr.segments is not None and asr.words is not None:
            raise ValueError('Select one authoritative ASR alignment granularity')
        units, granularity = (asr.segments, 'segment') if asr.segments is not None else (asr.words, 'word')
        if self.speaker_alignment != (granularity if units is not None else 'unavailable') or len(self.speaker_segments) != len(units or []):
            raise ValueError('Attributed units must exactly preserve the available ASR alignment')
        for original, attributed in zip(units or [], self.speaker_segments):
            text = original.text if granularity == 'segment' else original.word
            if (attributed.id, attributed.start, attributed.end, attributed.text, attributed.granularity) != (original.id, original.start, original.end, text, granularity):
                raise ValueError('Speaker attribution must not rewrite ASR text or timestamps')
            reason = timestamp_attribution(original.start, original.end, asr.duration)
            known = reason == 'interval_intersection'
            names = sorted({row.speaker for row in self.diarization.segments
                if known and max(original.start, row.start) < min(original.end, row.end)}, key=lambda name: int(name.split('_')[1]))
            overlap = any(known and max(original.start, row.start) < min(original.end, row.end) for row in self.diarization.overlaps)
            if (attributed.speakers != names or attributed.overlap != overlap
                    or attributed.attribution != reason):
                raise ValueError('Attributed speaker IDs/overlap must match real interval intersections')
        if self.speaker_attribution_status != attribution_status(self.speaker_segments):
            raise ValueError('Speaker attribution status must reflect usable ASR timestamps')
        if self.text_analysis is not None:
            import hashlib
            identity = self.text_analysis.input
            if (identity.modality != 'asr_transcript' or identity.characters != len(self.text)
                    or identity.sha256 != hashlib.sha256(self.text.encode('utf-8')).hexdigest()):
                raise ValueError('Text analysis must bind the unmodified ASR transcript')
        return self
