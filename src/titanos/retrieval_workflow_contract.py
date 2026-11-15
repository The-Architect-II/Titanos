"""Stateless retrieval answer contracts; checked quotations are not entailment proof."""
import hashlib
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .retrieval_contract import InputDetail, RerankResult, TruncationInfo
from .settings import current_settings
from .retrieval_prompts import PromptVersion


class Value(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Document(Value):
    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=512)
    source_uri: str | None = Field(default=None, max_length=2048)


class RetrievalAnswerRequest(Value):
    query: str = Field(min_length=1)
    documents: list[Document] = Field(min_length=1)
    embedding_model: str = Field(min_length=1, max_length=128)
    rerank_model: str = Field(min_length=1, max_length=128)
    llm_model: str = Field(min_length=1, max_length=128)
    prompt_version: PromptVersion | None = Field(default=None, description='Request overrides answer-model retrieval_prompt_version, then the process default')
    embedding_worker: str | None = Field(default=None, max_length=128)
    rerank_worker: str | None = Field(default=None, max_length=128)
    llm_worker: str | None = Field(default=None, max_length=128)
    retrieve_top_k: int | None = Field(default=None, strict=True, ge=1)
    context_top_n: int | None = Field(default=None, strict=True, ge=1)
    embedding_max_tokens: int | None = Field(default=None, strict=True, ge=1)
    rerank_max_input_tokens: int | None = Field(default=None, strict=True, ge=1)
    context_max_tokens: int | None = Field(default=None, strict=True, ge=1)
    max_tokens: int | None = Field(default=None, strict=True, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    timeout: int | None = Field(default=None, strict=True, gt=0)
    return_documents: bool = Field(default=True, strict=True)
    truncate: Literal[False] = False
    stream: Literal[False] = False

    @model_validator(mode='after')
    def bounded(self):
        settings = current_settings()
        if not self.query.strip() or len(self.query) > settings.max_prompt_chars:
            raise ValueError('query exceeds the configured nonblank prompt limit')
        if len(self.documents) > settings.retrieval_max_documents or len(self.documents)+1 > settings.embedding_max_batch:
            raise ValueError('corpus exceeds the retrieval/embedding batch limit (query occupies one entry)')
        if len({doc.id for doc in self.documents}) != len(self.documents) or any(not doc.id.strip() or not doc.text.strip() or len(doc.text)>settings.max_text_chars for doc in self.documents):
            raise ValueError('documents require unique nonblank IDs and bounded nonblank text')
        for name, limit in [('embedding_max_tokens',settings.embedding_max_tokens),('rerank_max_input_tokens',settings.rerank_max_input_tokens),('max_tokens',settings.max_tokens)]:
            if getattr(self,name) is not None and getattr(self,name)>limit:
                raise ValueError(name+' exceeds the process limit')
        if min(self.retrieve_top_k or settings.retrieval_top_k,len(self.documents)) > settings.rerank_max_documents:
            raise ValueError('retrieved candidate count exceeds rerank document limit')
        for text in [self.query,*[value for doc in self.documents for value in (doc.id,doc.text,doc.title,doc.source_uri) if value is not None]]:
            try:text.encode('utf-8')
            except UnicodeEncodeError as exc:raise ValueError('query and document fields must contain valid Unicode scalar text') from exc
        return self


class ModelProvenance(Value):
    repository: str | None = None
    revision: str | None = None
    checkpoint_digest: str | None = Field(default=None, pattern='^[a-f0-9]{64}$')
    backend: str
    device: str | None = None
    pooling: str | None = None
    normalization: str | None = None
    identifiers_are_registered: Literal[True] = True


class StageError(Value):
    code: str
    message: str
    http_status: int


class StageUsage(Value):
    model_config = ConfigDict(extra='allow',allow_inf_nan=False)
    prompt_tokens: int = Field(strict=True,ge=0)
    total_tokens: int = Field(strict=True,ge=0)
    completion_tokens: int | None = Field(default=None,strict=True,ge=0)


class Stage(Value):
    name: Literal['embedding','selection','rerank','generation','validation']
    status: Literal['completed','failed','not_run'] = 'not_run'
    model: str | None = None
    requested_worker: str | None = None
    worker: str | None = None
    binary_sha256: str | None = Field(default=None,pattern='^[a-f0-9]{64}$')
    runtime_digest: str | None = Field(default=None,pattern='^[a-f0-9]{64}$')
    input_sha256: str | None = None
    output_sha256: str | None = None
    elapsed_seconds: float = Field(default=0, ge=0)
    usage: StageUsage | None = None
    provenance: ModelProvenance | None = None
    error: StageError | None = None

    @model_validator(mode='after')
    def coherent(self):
        if self.status=='completed' and self.error is not None or self.status=='failed' and self.error is None:
            raise ValueError('Stage completion/failure must match its error record')
        return self


class Source(Value):
    document_id: str
    original_index: int = Field(ge=0)
    text_sha256: str
    title: str | None = None
    source_uri: str | None = None
    text: str | None = None


class Candidate(Value):
    document_id: str
    original_index: int = Field(ge=0)
    text_sha256: str
    cosine_similarity: float


class ContextAccounting(Value):
    method: Literal['native_template_and_tokenizer']
    prompt_sha256: str = Field(pattern='^[a-f0-9]{64}$')
    input_tokens: int = Field(strict=True,ge=1)
    requested_input_limit: int = Field(strict=True,ge=1)
    model_context_tokens: int = Field(strict=True,ge=1)
    output_reservation: int = Field(strict=True,ge=1)
    verified_against_usage: Literal[True]


class ContextTrace(Value):
    prompt_version: PromptVersion = 'retrieval-answer-v1'
    messages_sha256: str
    source_ids: list[str]
    messages: list[dict[str,str]] | None = None
    accounting: ContextAccounting | None = None


class Citation(Value):
    document_id: str
    quote: str
    document_sha256: str
    quote_sha256: str
    spans: list[tuple[int,int]]
    offsets: Literal['unicode_codepoints'] = 'unicode_codepoints'


class Claim(Value):
    text: str
    citations: list[Citation]


class SupportValidation(Value):
    citation_integrity: Literal['verified','invalid','not_applicable'] = 'not_applicable'
    semantic_entailment: Literal['not_automatically_verified'] = 'not_automatically_verified'


class RetrievalAnswerResult(Value):
    object: Literal['retrieval.answer'] = 'retrieval.answer'
    schema_version: Literal['1', '2'] = '1'
    status: Literal['answered','abstained','failed']
    reasons: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    query_sha256: str
    corpus_sha256: str
    documents: list[Source]
    candidates: list[Candidate] = Field(default_factory=list)
    rerank_candidate_ids: list[str] = Field(default_factory=list)
    rerank_result: RerankResult | None = None
    embedding_input_details: list[InputDetail] = Field(default_factory=list)
    embedding_truncation: TruncationInfo | None = None
    context: ContextTrace | None = None
    stages: list[Stage]
    support_validation: SupportValidation = Field(default_factory=SupportValidation)

    @model_validator(mode='after')
    def coherent(self):
        if self.context is not None and self.schema_version != ('1' if self.context.prompt_version=='retrieval-answer-v1' else '2'):
            raise ValueError('Result schema version must match the selected retrieval prompt contract')
        if [stage.name for stage in self.stages] != ['embedding','selection','rerank','generation','validation']:
            raise ValueError('Workflow has exactly the declared ordered stages')
        if self.status!='answered' and self.claims:
            raise ValueError('Failed/abstained workflows cannot expose answer claims')
        if self.status=='answered' and (not self.claims or self.support_validation.citation_integrity!='verified' or any(stage.status!='completed' for stage in self.stages)):
            raise ValueError('Answered workflow requires completed stages and checked citations')
        if self.status=='failed' and not any(stage.status=='failed' for stage in self.stages):
            raise ValueError('Failure requires a typed stage failure')
        if self.status=='abstained' and (any(stage.status!='completed' for stage in self.stages) or self.support_validation.citation_integrity!='not_applicable'):
            raise ValueError('Model abstention requires completed stages, not a validation failure')
        if [source.original_index for source in self.documents] != list(range(len(self.documents))) or len({source.document_id for source in self.documents})!=len(self.documents):
            raise ValueError('Original document identities must stay ordered and unique')
        sources={source.document_id:source for source in self.documents}
        for source in self.documents:
            if source.text is not None and hashlib.sha256(source.text.encode('utf-8')).hexdigest()!=source.text_sha256:
                raise ValueError('Returned document text differs from its identity')
        for candidate in self.candidates:
            source=sources.get(candidate.document_id)
            if source is None or (candidate.original_index,candidate.text_sha256)!=(source.original_index,source.text_sha256):
                raise ValueError('Retrieval candidate mapping differs from the original corpus')
        if self.rerank_candidate_ids != [candidate.document_id for candidate in self.candidates]:
            raise ValueError('Rerank-local index mapping must preserve the candidate order')
        if self.context is not None and (len(set(self.context.source_ids))!=len(self.context.source_ids) or not set(self.context.source_ids)<=set(self.rerank_candidate_ids)):
            raise ValueError('Context contains an unselected or duplicate source')
        if self.context is not None and self.context.messages is not None:
            encoded=json.dumps(self.context.messages,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False)
            if hashlib.sha256(encoded.encode('utf-8')).hexdigest()!=self.context.messages_sha256:
                raise ValueError('Returned context differs from the supplied-context hash')
        for claim in self.claims:
            if not claim.text.strip() or not claim.citations:raise ValueError('Every claim needs a source citation')
            for citation in claim.citations:
                source=sources.get(citation.document_id)
                if self.context is None or citation.document_id not in self.context.source_ids or source is None or citation.document_sha256!=source.text_sha256:
                    raise ValueError('Citation does not name its supplied source')
                if not citation.quote.strip() or not citation.spans:
                    raise ValueError('Citation has no literal source quote')
                if hashlib.sha256(citation.quote.encode('utf-8')).hexdigest()!=citation.quote_sha256:
                    raise ValueError('Citation quote differs from its identity')
                if any(start<0 or end-start!=len(citation.quote) for start,end in citation.spans):
                    raise ValueError('Invalid citation span')
                if source.text is not None and any(source.text[start:end]!=citation.quote for start,end in citation.spans):
                    raise ValueError('Citation span does not match supplied text')
        return self
