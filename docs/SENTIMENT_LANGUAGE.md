# Sentiment language and composition

Text sentiment describes expressed evaluation, not actual feelings or audio affect. This source increment adds categorical LLM output and an explicit two-stage composition. Runtime/model quality is separately qualified; source support is not a semantic acceptance claim.

`POST /v1/sentiment` keeps `model`, `text`, input metadata, thresholds and normal generation controls. Select the task prompt with `prompt_version` (`weights-v1`, `label-v2` or `label-v3`) and decoding with `decoding` (`auto`, `validate`, `label_choices`). Request values override model `defaults.sentiment_prompt_version` / `defaults.sentiment_decoding`, which override `TITANOS_SENTIMENT_PROMPT_VERSION` / `TITANOS_SENTIMENT_DECODING`. Defaults are `label-v2` / `auto`.

Existing `text.sentiment` modes in this development revision use **schema_version 2**, including dedicated classifiers and the legacy weighted prompt. The added decision/provenance fields are an explicit wire-schema change: strict 0.5.4 schema-1 Python/MCP consumers must upgrade their response model. The `/v1` route prefix is the API namespace, not the result schema version. Existing archived schema-1 receipts are unchanged. The optional facet task uses schema3; existing modes keep their schema2 fields without a new null facet field. Composition is a new object with its own schema version1 and typed stage results carrying their own schema version.

`weights-v1` preserves the original strict model-output weights-plus-abstain JSON contract and score/margin policy, not the old HTTP response schema. Generated weights are explicitly uncalibrated. The earlier malformed SmolLM2 output and Qwen universal abstention remain useful regressions; selecting a new mode does not relabel their historical results.

`label-v3` is an optional second categorical task prompt. It checks opposing aspects first and separates context-poor evaluation from clear factual neutrality. It uses the same categorical output contract and finite decoder as label-v2; prompt versions remain distinct in provenance. Its semantics require model evaluation, not just schema validation. The default remains label-v2 until separately qualified.

`label-v2` requests only a categorical JSON label: negative, neutral, positive, mixed or unknown. It removes the independent abstain boolean and the demand to invent numerical confidence. A clear label returns schema_version `2`, `decision_basis: label`, `scores: null`, `score_kind: none`, `score_thresholds_applied: false` and the exact `model_label`. The response still shows configured score thresholds, but explicitly does not apply them when no scores exist. Mixed maps to public label unknown with reason `mixed_polarity`; unknown maps to `model_abstained`. Caller-marked ASR uncertainty always remains abstention. No one-hot confidence is fabricated.

With a Transformers chat backend, `auto` selects `label_choices` for label-v2. A finite prefix trie restricts every generated token to one of the five JSON label outputs and EOS. Tokenizer round-trip validation is required. This is a finite output grammar, not arbitrary JSON-schema support or evidence that the chosen label is correct. Low token limits/cancellation can leave an incomplete output; strict validation then returns abstention. `validate` uses the prompt followed by strict validation only. OpenVINO GenAI supports that validation mode here; requesting label_choices on an unsupported backend fails explicitly. `constraint_mode` and provenance `prompt_version`/`prompt_sha256` report the actual path.

## Explicit composition

```json
{
  "classifier_model": "cardiff",
  "llm_model": "qwen",
  "text": "The item works well.",
  "policy": "require_agreement",
  "timeout": 60000,
  "prompt_version": "label-v2",
  "decoding": "auto"
}
```

Send this to `POST /v1/sentiment/compose`. The request shares text/language/ASR metadata and optional threshold overrides. `max_input_tokens` applies to the classifier; `max_tokens`, `temperature`, `top_p`, prompt version and decoding apply to the LLM. `worker` selects the classifier worker and `llm_worker` the LLM worker.

The only policy is `require_agreement`: both completed stages must decide the same polarity. A stage abstention is retained and keeps the overall label unknown. Differing decided labels return `agreement: disagree` with `stage_disagreement`. No component scores are averaged or converted into a joint probability; top-level scores are always null.

The typed `text.sentiment.composition` response contains input identity and two stages, `classifier` and `llm`. Each stage has its model, status (`completed`, `failed`, `not_run`), exact result or an explicit error (`code`, `message`, `http_status`). A failed classifier skips the LLM with `preceding_stage_failed`. Deadline failure is visible in the affected stage. Transport HTTP200 can carry a complete failure report: always inspect the top-level status (`ok`, `abstained`, `failed`) before using its label. Invalid requests/model roles are rejected before inference.

One deadline is shared by both stages; the second stage receives only the remaining milliseconds. Model loads/native calls remain cooperatively cancellable, so this logical deadline does not promise immediate preemption of a blocked native call. Owned-job wall/resource guards remain the separate hard execution boundary. One resident model per engine means composition can evict/reload sequentially; use existing explicit worker assignments when independently resident stages are appropriate.

The same request is exposed as extension operation `sentiment_compose`, Python `Client.compose_sentiment(classifier_model, llm_model, text, **options)`, and MCP tool `compose_sentiment` with a typed output schema. CLI:

```text
titanos sentiment-compose cardiff qwen "The item works well." --prompt-version label-v2 --decoding auto --deadline-ms 60000
```

CLI `--deadline-ms` controls the shared inference deadline; `--timeout` remains the HTTP transport timeout in seconds. Direct `sentiment` also accepts those prompt/decoding/deadline controls. The legacy weighted mode remains explicitly selectable.

HTTP disconnect and MCP task cancellation signal the active sentiment backend and prevent the next composition stage. Both stages share the caller event; deadline expiry remains a separate `deadline_exceeded` error. A cancelled stage has error code `cancelled`, and an earlier completed stage stays in the composition record. The async handler retains the worker through cooperative cleanup. This does not forcibly interrupt a native kernel or model-file load; it checks cancellation at backend boundaries and after generation, and never accepts a post-cancellation normal result as success. A disconnected caller cannot receive the resulting error body. Controlled tests exercise disconnect and actual MCP tool task cancellation in each stage; transport-level native cancellation still needs qualification.

Tests use controlled backends and old development regressions only. Independent held-out cases must remain unseen by the builder; all-abstained output cannot satisfy the useful-decision gates. Qualify the chosen model/prompt/decoder on real native runtime before claiming SENT-02/SENT-03/SENT-04 complete.

Time waiting for the request worker is debited from the same deadline. If the deadline has already expired before any stage can execute, the request returns HTTP504 `deadline_exceeded` without loading a model. A deadline reached during a composition stage is recorded in that stage's error.

The optional `facets-v1` task asks for independent boolean observations, retained exactly as `model_facets`: `positive_evaluation`, `negative_evaluation`, and `neutral_statement`. The last flag means a self-contained unevaluative statement; it does not verify factual truth. `decoding: auto` selects `facet_choices` on Transformers, a finite language of exactly eight boolean records. `validate` provides prompting plus strict validation only. This is not a generic JSON grammar. Explicit `facet_choices` requires this task and a Transformers chat backend. Existing prompt/decoding/default behavior is unchanged.

Facet responses use schema3, `decision_basis: facets`, null scores, no invented numeric confidence, and null `model_label` because the model reported flags rather than one label. Model-reported flags are not externally verified evidence. The declared derivation is:

| Positive flag | Negative flag | Neutral-statement flag | Derived disposition |
|---|---|---|---|
| true | true | either | unknown, abstained with `mixed_polarity` |
| true | false | either | positive |
| false | true | either | negative |
| false | false | true | neutral |
| false | false | false | unknown, abstained with `insufficient_evidence` |

All reported flags remain visible, including a neutral flag alongside polarity. Caller-declared ASR uncertainty still abstains; malformed/truncated flags abstain with `invalid_model_output` and no fabricated flags. Composition uses the same require-agreement policy, shared absolute deadline and caller-cancellation event. Use the same request shape or `titanos sentiment qwen "The item works well." --prompt-version facets-v1 --decoding auto`. This task still requires a separate model-quality gate; eight valid record shapes do not establish eight correct semantic interpretations.

An owned `llama-cpp` chat model can run the same weighted, label or facet tasks. `auto` selects `native_json_schema`; that name is also an explicit `decoding` value. The task schema is sent directly through `NativePool.complete` to the worker's native chat endpoint. It is distinct from the finite Transformers choices and from `validate` (prompting plus post-validation). Output still passes the same strict task parser; schema support does not replace cross-field checks such as weighted sums or prove a correct semantic decision. Unsupported native/schema requests fail explicitly without a fallback or replay.

The internal completion boundary uses the existing owned worker lease and process lifecycle. One absolute deadline and caller event cover queueing, startup and response reads; cancellation closes the response, joins startup cleanup and releases the lease. A per-request callback resolves the current gateway after registry reload, and schedules its coroutine on the active request loop from the stage's worker thread. It never calls Titanos's own public HTTP route. Native result provenance reports the actual selected device and a digest of all declared GGUF files; `/api/models/details` also exposes their hashes, the launcher hash and the adjacent runtime-library fingerprint. Those file fingerprints alone do not prove which DLLs loaded; actual module-path qualification remains separate.

For split GGUFs, set `native.model_file` to the first shard and `native.model_files` to the complete original shard list including that first file. Titanos fingerprints each file and llama.cpp handles the split set without merging. Optional `native.batch_size`/`native.ubatch_size` become the native batch flags. Their environment defaults are `TITANOS_NATIVE_BATCH_SIZE` and `TITANOS_NATIVE_UBATCH_SIZE`; zero leaves the upstream defaults unchanged. Explicit nonzero model options override environment defaults. H4's proposed CPU controls are context4096,slot1,threads4,batch256,ubatch128; these are qualification settings, not a general hardware guarantee.

Permanent pool shutdown marks its workers closed, signals and joins any retained hashing/startup thread, and serializes process/client cleanup. A lease selected before shutdown cannot start afterward. Ordinary model unload uses reversible stop instead, so the same registered worker can load again. The executable hash reuses its already streamed runtime fingerprint.
