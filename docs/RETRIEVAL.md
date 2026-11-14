# Text retrieval

Titanos provides normalized embeddings and a separate trained cross-encoder reranker. Models are external local checkpoints; HiveMind, PAIR and a hosted API are not required.

## Explicit embedding input policy

Overlength input now fails before model forward by default. The earlier Transformers and compiled adapters silently truncated text; the compiled adapter also used its graph length without honoring the request's smaller token limit. This change intentionally makes that loss of input visible.

```json
{"model":"encoder","input":["first text","second text"],"max_tokens":256,"truncate":false}
```

Send `truncate:true` to allow cropping, set `defaults.truncate` on that model, or set `TITANOS_EMBEDDING_TRUNCATE=true` for the process. Precedence is explicit request, model default, process default (`false`). A request `max_tokens` overrides `defaults.max_tokens`, then `TITANOS_EMBEDDING_MAX_TOKENS` supplies the default budget. The tokenizer/model context and fixed graph length remain hard upper bounds. Fixed graphs are padded to their required shape after the smaller requested token budget is applied; padding is not counted as consumed input.

The OpenAI envelope remains `object`, `model`, `data` and `usage`, with float or base64 embeddings. Two additive fields describe input fidelity:

```json
{
  "input_details":[{"index":0,"input_tokens":700,"processed_tokens":256,"truncated":true,"input_sha256":"<SHA256 of the original UTF-8 text>"}],
  "truncation":{"enabled":true,"side":"right","strategy":"single_sequence","max_tokens":256}
}
```

Counts include special tokens and exclude padding. `usage.prompt_tokens` and `total_tokens` describe processed tokens. Original text hashes do not change when truncation is enabled. Every entry is checked before any forward in an embedding batch, so an invalid later entry cannot silently produce partial results. The same policy/reporting flows through `/api/embed`, legacy `/api/embeddings`, the Python client and MCP `embed`.

```shell
titanos embed encoder "some text" --max-tokens 256 --no-truncate
titanos embed encoder --input-file texts.json --truncate
```

OpenAI Python clients can use `extra_body={"truncate": True, "max_tokens": 256}`. Controlled compatibility checks cover ordinary SDK base64 decoding and strict float-response parsing. Clients needing the extra fields can inspect the SDK response's extra metadata or use Titanos's client.

## Trained reranking

Register an author-verified, scalar sequence-classification cross-encoder with `--task rerank --backend transformers`. It must use local safetensors and built-in Transformers code. The adapter rejects missing/incompatible weights, multi-output heads and multi-device placement. It jointly tokenizes each query/document pair and calls `AutoModelForSequenceClassification`; it does not derive ranks from embedding cosine distance.

```shell
titanos add reranker /path/to/checkpoint --task rerank --backend transformers --device cpu
titanos rerank reranker "query text" "document one" "document two" --top-n 1 --return-documents
titanos rerank reranker "query text" --documents-file documents.json --max-input-tokens 512 --no-truncate --deadline-ms 10000
```

`POST /v1/rerank` accepts:

```json
{"model":"reranker","query":"query text","documents":["document one","document two"],"top_n":1,"return_documents":true,"max_input_tokens":512,"truncate":false}
```

The response has `object:"list"`, `model`, `results`, `score_kind:"cross_encoder_logit"`, `usage`, `input_details`, `truncation` and checkpoint `provenance`. Each result contains its original zero-based `index` and finite `relevance_score`. Higher scores rank first; exact ties retain ascending original index. Scores can be negative or greater than one: they are raw relevance logits, not calibrated probabilities, and are not promised comparable across queries or models.

`top_n` is an upper bound; absent means all documents, and a value above the document count returns all. All documents are scored before selecting the result subset. `return_documents:true` includes `document:{"text":...}` with the original supplied text; false omits document contents. Input details cover every original document, including those outside `top_n`, and include both the original document's `input_sha256` and `query_sha256`.

Reranking rejects an overlong pair before any forward by default. Explicit `truncate:true` permits truncating only the document, according to the tokenizer's reported side; the full query is preserved. A query that cannot fit with the pair's special tokens and document space is rejected. Pair counts include query, document and special tokens. `defaults.max_input_tokens` supplies a per-model budget, bounded by `TITANOS_RERANK_MAX_INPUT_TOKENS` (default 512) and the actual model context. `defaults.truncate` overrides `TITANOS_RERANK_TRUNCATE` (default false), and explicit request values take precedence. `TITANOS_RERANK_MAX_DOCUMENTS` defaults to 64; query/document character and HTTP body limits also apply.

Cancellation is checked before tokenization, before and after each document forward, and at the shared deadline. It cannot interrupt a single native kernel in progress. The `timeout` request field is milliseconds including model-queue wait; the CLI exposes it as `--deadline-ms`. The ordinary client `timeout` controls HTTP waiting separately.

```python
from titanos.client import Client

with Client() as client:
    result = client.rerank("reranker", "query text", ["document one", "document two"], top_n=1)
    print(result.results[0].index, result.results[0].relevance_score)
```

The Python method returns the typed `RerankResult`. MCP exposes the same typed request/result through `rerank`; `extension_infer` also supports `operation:"rerank"`. Discover registered candidates and schemas through `/v1/extensions`.

## Selected model and proof boundary

The admitted candidate is [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a), pinned at `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Its Apache-2.0 model card describes English MS MARCO passage ranking and direct Transformers logit scoring. The author safetensors file is 90,870,598 bytes, SHA-256 `821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae`. The inherited `_name_or_path` metadata mentions the L12 parent; the actual architecture configuration has six layers and a scalar scoring head.

Source tests use controlled tensors/scorers to prove limits, cancellation, transport schemas and ordering. Those tests are not relevance evidence. Real source and packaged-runtime relevance results require separate model/host/commit admission and the predeclared independent query/document gates. Checkpoint provenance reports registered repository/revision and the actual local file fingerprint; registration metadata alone is not an upstream authenticity claim.
