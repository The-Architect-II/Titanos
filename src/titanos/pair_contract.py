"""Native categorical pair labels and uncalibrated scores, without inferred meanings."""
import hashlib
import json
import math
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .retrieval_contract import RetrievalUsage, RerankProvenance
from .settings import current_settings

Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
Number = Annotated[float, Field(strict=True)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1)]

def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def pair_hash(premise, hypothesis):
    return text_hash(json.dumps([premise, hypothesis], ensure_ascii=False, separators=(',', ':')))

def softmax(logits):
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = math.fsum(weights)
    return [value / total for value in weights]

class Value(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class TextPair(Value):
    id: str | None = Field(default=None, min_length=1, max_length=128)
    premise: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)

    @model_validator(mode='after')
    def valid_text(self):
        for text in [self.premise, self.hypothesis, *([self.id] if self.id is not None else [])]:
            if not text.strip():raise ValueError('Pair text and IDs must be nonblank')
            try:text.encode('utf-8')
            except UnicodeEncodeError as exc:raise ValueError('Pair input must be Unicode scalar text') from exc
        return self

def validate_pairs(values):
    if not isinstance(values, list) or not values:
        raise ValueError('pairs must be a nonempty array')
    settings = current_settings()
    if len(values) > settings.pair_max_pairs:
        raise ValueError('Pair count exceeds TITANOS_PAIR_MAX_PAIRS')
    pairs = [value if isinstance(value, TextPair) else TextPair.model_validate(value) for value in values]
    if any(len(text) > settings.max_text_chars for pair in pairs for text in (pair.premise, pair.hypothesis)):
        raise ValueError('Pair text exceeds TITANOS_MAX_TEXT_CHARS')
    ids = [pair.id if pair.id is not None else f'pair-{index}' for index, pair in enumerate(pairs)]
    if len(set(ids)) != len(ids):raise ValueError('Caller and generated pair IDs must be unique')
    return pairs, ids

class NativeLabel(Value):
    index: int = Field(strict=True, ge=0)
    label: str = Field(min_length=1)

class PairInputDetail(Value):
    premise_sha256: Digest
    hypothesis_sha256: Digest
    pair_sha256: Digest
    input_tokens: int = Field(strict=True, ge=1)
    processed_tokens: int = Field(strict=True, ge=1)
    truncated: bool = Field(strict=True)
    original_token_ids_sha256: Digest
    processed_token_ids_sha256: Digest

    @model_validator(mode='after')
    def coherent(self):
        if self.processed_tokens > self.input_tokens or self.truncated != (self.processed_tokens < self.input_tokens):
            raise ValueError('Pair truncation must match actual token counts')
        if not self.truncated and self.original_token_ids_sha256 != self.processed_token_ids_sha256:
            raise ValueError('Untruncated token identities changed')
        if self.truncated and self.original_token_ids_sha256 == self.processed_token_ids_sha256:
            raise ValueError('Truncated token sequence cannot keep the same token identity')
        return self

class PairTruncation(Value):
    enabled: bool = Field(strict=True)
    side: Literal['left', 'right']
    strategy: Literal['only_first'] = 'only_first'
    preserved_sequence: Literal['hypothesis'] = 'hypothesis'
    max_tokens: int = Field(strict=True, ge=1)

class PairPrediction(Value):
    index: int = Field(strict=True, ge=0)
    id: str = Field(min_length=1, max_length=128)
    id_source: Literal['caller', 'index_default']
    label_index: int = Field(strict=True, ge=0)
    label: str
    logits: list[Number] = Field(min_length=2)
    scores: list[Probability] = Field(min_length=2)
    input: PairInputDetail

class PairProvenance(RerankProvenance):
    device: str = Field(min_length=1)
    label_source: Literal['loaded_config_id2label'] = 'loaded_config_id2label'
    model_class: str = Field(min_length=1)
    model_type: str = Field(min_length=1)

class PairClassificationResult(Value):
    object: Literal['text_pair.classification'] = 'text_pair.classification'
    schema_version: Literal['1'] = '1'
    model: str
    labels: list[NativeLabel] = Field(min_length=2)
    results: list[PairPrediction] = Field(min_length=1)
    score_kind: Literal['softmax_uncalibrated'] = 'softmax_uncalibrated'
    logit_kind: Literal['native_classifier_logits'] = 'native_classifier_logits'
    calibrated: Literal[False] = False
    semantic_label_mapping_applied: Literal[False] = False
    tie_break: Literal['lowest_class_index'] = 'lowest_class_index'
    truncation: PairTruncation
    usage: RetrievalUsage
    provenance: PairProvenance

    @model_validator(mode='after')
    def coherent(self):
        if [item.index for item in self.labels] != list(range(len(self.labels))) or len({item.label for item in self.labels}) != len(self.labels):
            raise ValueError('Native label indices must be contiguous with unique names')
        if [row.index for row in self.results] != list(range(len(self.results))) or len({row.id for row in self.results}) != len(self.results):
            raise ValueError('Pair results must preserve input order and unique IDs')
        for row in self.results:
            if len(row.logits) != len(self.labels) or len(row.scores) != len(self.labels):
                raise ValueError('Every pair must retain every native class logit and score')
            if any(not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7) for a,b in zip(row.scores,softmax(row.logits))):
                raise ValueError('Scores must be the disclosed uncalibrated softmax of logits')
            winner = max(range(len(row.logits)), key=row.logits.__getitem__)
            if row.label_index != winner or row.label != self.labels[winner].label:
                raise ValueError('Winning native label must follow logits, with lowest-index ties')
            if row.input.processed_tokens > self.truncation.max_tokens or (row.input.truncated and not self.truncation.enabled):
                raise ValueError('Pair input did not honor the declared token policy')
        tokens = sum(row.input.processed_tokens for row in self.results)
        if self.usage.prompt_tokens != tokens or self.usage.total_tokens != tokens:
            raise ValueError('Classifier usage must count the actual processed input tokens')
        return self
