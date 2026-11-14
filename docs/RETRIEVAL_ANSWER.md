# Answer from supplied passages

`POST /v1/retrieval/answer` runs a stateless embedding → rerank → owned native LLM workflow. It accepts inline passages with stable caller IDs. It does not fetch URLs, read caller paths, create a vector store or require PAIR/HiveMind.

The result distinguishes `answered`, genuine model-declared `abstained`, and `failed`. Malformed output, missing citations, an unknown document ID or a nonliteral quote produces **failed**, a typed validation error and no exposed answer claims. A quotation that occurs in a source does not prove that it supports the claim: `semantic_entailment` always says `not_automatically_verified`. Source URLs/titles are copied only from caller metadata and are not verified authenticity claims.

## Request

```json
{
  "embedding_model": "encoder",
  "rerank_model": "reranker",
  "llm_model": "answer-model",
  "prompt_version": "retrieval-answer-v2",
  "query": "Question about the supplied passages",
  "documents": [{"id": "caller-passage-id", "text": "Caller-supplied passage text"}],
  "retrieve_top_k": 4,
  "context_top_n": 2,
  "embedding_max_tokens": 256,
  "rerank_max_input_tokens": 512,
  "context_max_tokens": 2048,
  "max_tokens": 256,
  "temperature": 0,
  "top_p": 1,
  "timeout": 60000,
  "return_documents": true
}
```

The example values are field-shape placeholders, not a source citation or quality result. Each document has a unique nonblank `id` and nonblank `text`; optional `title` and `source_uri` are attribution data only. Text remains unchanged, including case and line endings. The first workflow requires a safe, single-mode Transformers encoder, a safe scalar Transformers reranker and an owned llama.cpp chat model. Unsupported roles fail before inference. It accepts only `truncate:false` and `stream:false`; the raw primitive endpoints retain their existing behavior.

The query occupies index0 of the embedding batch; document i occupies index i+1. Batch size must fit `TITANOS_EMBEDDING_MAX_BATCH`. Embeddings use the existing attention-mask mean pooling and L2 normalization, with loaded checkpoint/device/pooling provenance. Provenance is additive on `/v1/embeddings`; legacy or symlink layouts without a complete safe fingerprint remain usable through that primitive but cannot silently qualify for this workflow. No model weights or dependency formats were changed.

Cosine selection returns stable original document indices and IDs; exact ties use original index. `retrieve_top_k` is an upper bound. Rerank-local indices refer to that selected list, with the complete mapping retained in `rerank_candidate_ids`. Rerank scores remain raw cross-encoder logits. `context_top_n` selects from the reranker's existing sorted order; cosine and logit scores are never averaged or called probabilities.

`embedding_worker`, `rerank_worker` and `llm_worker` optionally pin existing assignments. Requested assignments are retained separately from an observed native worker ID. An unreported automatic text-worker identity remains absent; the actual loaded encoder/reranker device and fingerprint are still recorded. The first workflow rejects non-single text-encoder worker modes.

Request input budgets override model defaults; process settings provide the remaining defaults. Generation controls reuse the existing shared Chat defaults and limits. Workflow settings are:

| Environment setting | Default |
|---|---:|
| `TITANOS_RETRIEVAL_MAX_DOCUMENTS` |32|
| `TITANOS_RETRIEVAL_TOP_K` |8|
| `TITANOS_RETRIEVAL_TOP_N` |3|
| `TITANOS_RETRIEVAL_CONTEXT_TOKENS` |2048|
| `TITANOS_RETRIEVAL_PROMPT_VERSION` |`retrieval-answer-v2`|
| `TITANOS_RETRIEVAL_MAX_CLAIMS` |8|
| `TITANOS_RETRIEVAL_MAX_CITATIONS_PER_CLAIM` |4|
| `TITANOS_RETRIEVAL_MAX_QUOTE_CHARS` |2048|
| `TITANOS_RETRIEVAL_MAX_QUOTE_MATCHES` |32|

Existing HTTP body, text, embedding batch/token and rerank count/token bounds also apply. A lower limit remains binding. Oversized passages fail instead of being silently cropped. Repeated quotes retain all matching Unicode-codepoint spans up to the configured match limit; exceeding it fails with `ambiguous_citation` instead of allocating an unbounded span list.

## Context and source checks

The source-reading policy resolves from non-null request `prompt_version`, then the answer model's non-null `defaults.retrieval_prompt_version`, then `TITANOS_RETRIEVAL_PROMPT_VERSION`. The candidate process default is `retrieval-answer-v2`. Unknown versions and malformed defaults fail before any workflow model executes. The model default is valid only on an owned native chat registration configured through the trusted local registry/CLI; the existing HTTP restriction on native loader registration remains. It does not change ordinary chat prompts.

v2 tells the model to separate factual assertions from commands addressed to the assistant. It must ignore those commands while continuing to use relevant supported facts in the same passage. Instruction-like text alone is not a reason to discard a passage or abstain. Insufficient, ambiguous or unresolved contradictory evidence still requires abstention. This is a versioned prompt policy, not a deterministic model guarantee: real consumed-case development and new independent gates must qualify the candidate. No source text is removed, no output is repaired and no request is automatically retried.

The exact original `retrieval-answer-v1` instruction remains selectable. v1 returns result `schema_version:"1"`; v2 explicitly returns `schema_version:"2"`. The returned `context.prompt_version` identifies the selected policy, with the exact supplied messages/hash. Older strict typed clients must upgrade to read schema2 or explicitly select v1 (per request, model default, or process setting). v1 retains the prior schema1 fields and serialization. New clients accept both. A model abstention reports behavior; it does not establish that an answer is absent from the source.

Both policies place the original query and selected `{document_id,text}` records in one data message. The result retains its exact messages/hash and source order. `return_documents:false` omits full document text and messages, while retaining their hashes, selected IDs and the validated quotations. Reconstruct the same context from the original request, source order and versioned instruction when full text is omitted. The producer citation JSON schema and strict literal validator are shared across prompt versions; neither prompt adds semantic-entailment verification.

With this workflow's explicit context budget, the same leased native worker applies its chat template and tokenizes the complete prompt before generation, then reports its context capacity. No character estimate is called a token count. Input plus output reservation must fit the actual capacity and the requested input limit. The returned prompt-token usage must match preparation; an unsupported accounting endpoint or mismatch fails explicitly and does not expose claims. This optional path leaves ordinary native completion behavior unchanged. It follows the [pinned llama.cpp b10809 server endpoints](https://raw.githubusercontent.com/ggml-org/llama.cpp/b10809/tools/server/README.md); actual model/runtime accounting still requires a separate qualification run.

Native JSON-schema decoding restricts answer structure and cited IDs to the selected context. A second strict validator checks duplicate JSON keys, completion status, claim/citation bounds, exact source membership and literal quote matches. It never repairs a quote or invents a source location. `answered` requires nonempty cited claims; `abstained` requires an empty claims array. Invalid output is not counted as successful abstention.

The typed `retrieval.answer` result contains original document identities, cosine candidates, rerank result/mapping, supplied context and ordered embedding/selection/rerank/generation/validation stages. A failed stage retains earlier completed results and marks later stages `not_run`. Usage stays per stage. Native generation retains its actual worker and runtime/executable fingerprints. The shared absolute deadline and caller cancellation cover queues, input preparation, model work, context preparation and validation. Cancellation is cooperative at model boundaries; it does not pretend to preempt a kernel instantly.

## Interfaces

```python
from titanos.client import Client

with Client() as client:
    result = client.retrieval_answer(
        "answer-model", query, documents,
        embedding_model="encoder", rerank_model="reranker",
        retrieve_top_k=4, context_top_n=2, timeout=60000,
    )
    if result.status == "answered":
        for claim in result.claims:
            print(claim.text, claim.citations)
```

```text
titanos retrieval-answer answer-model "Question" --embedding-model encoder --rerank-model reranker --documents-file passages.json --retrieve-top-k 4 --context-top-n 2 --deadline-ms 60000
```

Add `--prompt-version retrieval-answer-v1` for the original policy and schema1 client compatibility, or `--prompt-version retrieval-answer-v2` for the candidate facts/commands policy. Python accepts the same `prompt_version` keyword; HTTP, the typed MCP tool and the extension accept the same field. The browser places this selection in advanced settings with policy descriptions. Its abstention message says the model did not answer, rather than claiming the supplied context lacks an answer.

The CLI reads a UTF-8 JSON document array locally. MCP exposes the typed `retrieval_answer` tool; `extension_infer` uses operation `retrieval_answer`. `/v1/extensions` lists compatible model roles and models. This is a Titanos extension, not an invented OpenAI/HLI standard route. The browser's **Answer from your documents** panel selects models, uploads a passage JSON file, controls limits and displays claims with literal quotations/source IDs. It renders text rather than executing or following source HTML/URIs, and the Stop button aborts the request.

Model-free checks exercise mapping, invalid/unsupported output, real abstention separation, partial stages, cancellation, native context-accounting failures, client/CLI/MCP and a controlled DOM flow. They do not establish retrieval/answer quality, entailment, a live browser acceptance result or native package qualification. The project owner controls the admitted all-MiniLM-L6-v2 encoder, compatible scalar-reranker selection and native Qwen7B roster, the untouched semantic gate and later native-runtime release. The earlier MiniLM reranker missed its separate relevance gate; its historical results are preserved and do not qualify the workflow. BGE reranker admission is a separate owner-controlled candidate, with no model-specific code path or automatic fallback. No model execution was performed for this source change.
