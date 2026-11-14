# Native text-pair classification

`pair-classification` runs a local Transformers `AutoModelForSequenceClassification` over **premise first, hypothesis second**. It retains every native output label, logit and softmax score. It does not map NLI labels to sentiment or relevance, integrate into RAG, apply a reliability policy, or claim that a predicted entailment label proves a fact.

Register an existing safe checkpoint with `task:"pair-classification"`, `backend:"transformers"`, and single-device placement. Model configuration must describe a categorical head with at least two outputs and a complete, unique `id2label` mapping. Generic names such as `LABEL_0` remain generic; no semantic names are invented. Explicit regression and multilabel configurations, scalar heads, incompatible/missing weights and label overrides fail. An absent `problem_type` follows the standard single-label interpretation of this explicit task registration; the adapter cannot reconstruct an undeclared training objective.

```json
{
  "model": "pair-model",
  "pairs": [
    {"id": "check-a", "premise": "The lamp is switched off.", "hypothesis": "The lamp is switched on."}
  ],
  "max_input_tokens": 512,
  "truncate": false
}
```

Send this to `POST /v1/classifications/text-pairs`, or use `operation:"pair_classification"` at `/v1/extensions/infer`. MCP provides typed `classify_pairs` and the existing extension tool. `/v1/extensions`, `/api/info` and model capabilities disclose this family and its schemas without claiming model quality.

```python
result = client.classify_pairs("pair-model", [
    {"id": "check-a", "premise": "The lamp is switched off.", "hypothesis": "The lamp is switched on."}
], max_input_tokens=512, truncate=False)
print(result.results[0].label, result.results[0].logits)
```

```text
titanos add pair-model ./models/pair-model --task pair-classification --device cpu
titanos classify-pairs pair-model --pairs-file pairs.json --max-input-tokens 512 --no-truncate --deadline-ms 30000
```

The CLI reads the JSON array from stdin when `--pairs-file` is omitted. Python and CLI omit absent token/truncation controls; explicit false remains false. Source/model storage and downloads remain separate; no inference request acquires weights.

## Browser form

In Studio, add or select a model with task `pair-classification`. Enter the **Premise / source text** and **Hypothesis / statement**; Run sends one pair directly to `/v1/classifications/text-pairs`. Text is sent unchanged and the server supplies the pair ID. The generic prompt and image input are hidden for this task. Batch IDs and multiple pairs remain available through the API, CLI, Python client and MCP.

Leave the input token limit and truncation selection blank to inherit model/server settings. **Reject oversized pairs** sends `truncate:false`; **Allow premise-only truncation** sends `truncate:true`. The result displays every native class index, label, raw logit and uncalibrated score, plus actual original/processed token counts. The full typed result includes input hashes, truncation policy and model provenance. Predicted labels do not establish truth or support a RAG reliability claim.

The main Studio form accepts one active request. Its model, input text, file, settings and conversation are captured before asynchronous work. Refresh may update the catalog, while model changes, Clear, Unload and another Run stay blocked until cleanup. Stop aborts the request that owns the form; late responses do not create results or chat history. If an asset-job ID arrives after Stop, Studio cancels that owned job before releasing the form. Cancellation is a client request, not a guarantee of immediate backend/kernel preemption.

Run `node tests/check_studio_pairs_ui.js` for the actual HTML/app source with DOM, fetch and deferred-response doubles. It covers request omission, native score rendering, errors, refresh/selection/duplicate-submit/Stop ownership, chat history, file-read snapshots and existing task controls. This source check does not establish real browser interaction or trained-model quality; those need separate qualification.

## Results and identities

The typed result is `object:"text_pair.classification"`, `schema_version:"1"`. `labels` preserves native class-index order. Every input pair has its original index and ID, full `logits`/`scores` arrays in that same order, and the winning native label/index. Exact logit ties select the lowest class index. Omitted IDs become `pair-<index>` and are marked `index_default`; explicit IDs are marked `caller`. Any collision between explicit and generated IDs is rejected before inference.

`score_kind:"softmax_uncalibrated"`, `calibrated:false` and `semantic_label_mapping_applied:false` state the interpretation. Scores are a numerically stable softmax of the reported native logits, not calibrated probabilities or independent correctness measurements. Output order follows input order; this is not a ranked-document result. Classifier usage counts processed input tokens, not generated text tokens.

Before and after loading, the existing checkpoint-fingerprint helper hashes contained local safetensors/config/tokenizer bytes. Changed files, replaced head parameters and inconsistent loaded label maps are rejected. Actual model parameters/buffers must reside on one requested device. Repository/revision are registered identifiers; the checkpoint digest and observed device have separate evidence. A cache snapshot linking weights outside its directory is not supported by this new task's strict identity scope; prepare contained local files explicitly. No existing task's fingerprint or loader behavior is relaxed.

## Input budgets and truncation

| Setting | Default |
|---|---:|
| `TITANOS_PAIR_MAX_PAIRS` | 32 |
| `TITANOS_PAIR_MAX_INPUT_TOKENS` | 512 |
| `TITANOS_PAIR_MAX_LABELS` | 256 |
| `TITANOS_PAIR_TRUNCATE` | false |

The token limit resolves from non-null request `max_input_tokens`, then model `defaults.max_input_tokens`, then the process limit. The effective limit also respects the tokenizer/model's actual supported input positions. Request/model values cannot exceed the process limit. Truncation resolves from non-null request `truncate`, then model `defaults.truncate`, then the process default. Pair-model defaults accept those two controls only; generation/label-remapping settings do not apply.

Every complete pair is tokenized without truncation before any forward call. By default, one oversized pair rejects the whole batch before model execution. With explicit truncation, only the premise is cropped on the tokenizer's declared left/right side; the complete hypothesis is preserved. If the hypothesis plus required special tokens leaves no room for a premise token, reject the request rather than truncate the hypothesis.

Each row reports both original text hashes, the ordered pair hash, original/processed token counts, the truncation flag, and original/processed token-ID hashes. Those token-ID hashes cover the unpadded model sequences including their special tokens, not decoded character spans. The shared tokenizer helper's existing embedding and rerank defaults remain unchanged.

All pairs are checked before inference; one forward per pair keeps cancellation boundaries explicit. The existing request deadline and cancellation signal cover preparation and model work. A cancelled or failed batch does not return a partial result as successful classification. Cancellation remains cooperative; source tests do not prove arbitrary native kernel preemption or all hardware.

`tests/check_pair_classification.py` uses loader/tensor/HTTP/MCP doubles to verify native labels, scalar/multilabel rejection, head/fingerprint binding, input limits, premise-only truncation, finite scores, stable IDs/ties, cancellation and interface omission. Actual model behavior and any later NLI/RAG reliability policy require separately frozen trained-model qualification. No trained model or private held-out case was used to develop this source slice.
