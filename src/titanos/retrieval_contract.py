"""Public retrieval results; relevance logits are not probability estimates."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class RetrievalValue(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class InputDetail(RetrievalValue):
    index: int = Field(ge=0, strict=True)
    input_tokens: int = Field(ge=1, strict=True)
    processed_tokens: int = Field(ge=1, strict=True)
    truncated: bool = Field(strict=True)
    input_sha256: str = Field(pattern='^[a-f0-9]{64}$')
    query_sha256: str | None = Field(default=None, pattern='^[a-f0-9]{64}$')


class TruncationInfo(RetrievalValue):
    enabled: bool = Field(strict=True)
    side: Literal['left', 'right']
    strategy: Literal['single_sequence', 'only_second']
    max_tokens: int = Field(ge=1, strict=True)


class RankedDocument(RetrievalValue):
    index: int = Field(ge=0, strict=True)
    relevance_score: float = Field(strict=True)
    document: dict[str, str] | None = None


class RetrievalUsage(RetrievalValue):
    prompt_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)


class RerankProvenance(RetrievalValue):
    repository: str | None = None
    revision: str | None = None
    checkpoint_digest: str = Field(pattern='^[a-f0-9]{64}$')
    backend: Literal['transformers']
    device: str


class RerankResult(RetrievalValue):
    object: Literal['list'] = 'list'
    model: str
    results: list[RankedDocument]
    score_kind: Literal['cross_encoder_logit']
    usage: RetrievalUsage
    input_details: list[InputDetail]
    truncation: TruncationInfo
    provenance: RerankProvenance
