# Text sentiment

Titanos exposes explicit text sentiment through `POST /v1/sentiment`, the `sentiment` extension operation and MCP tool, `titanos sentiment`, and `Client.sentiment`. Calls use the existing authentication, model registry, worker selection, model lock and lifecycle. This feature does not monitor messages or connect to a conversation service.

The [CPU text bundle](CPU_TEXT_BUNDLE.md) packages the runtime and MCP dependencies for a native standalone application with external model weights.

Register a local safetensors classifier with `task: sentiment` and `backend: transformers`. A compatible two- or three-class sequence-classification head is required. Native labels must mean negative, neutral or positive; unknown label names require a complete explicit `defaults.label_mapping`. Binary models retain their two classes. Regression/multilabel heads, incompatible classification weights, remote checkpoint code and runtime pickle loading are rejected.

```json
{
  "models": {
    "polarity": {
      "path": "D:/models/prepared-sentiment",
      "task": "sentiment",
      "backend": "transformers",
      "device": "cpu",
      "dtype": "float32",
      "defaults": {"max_input_tokens": 512, "min_score": 0.6, "min_margin": 0.15}
    }
  }
}
```

Use actual repository/revision fields when known. They are registered identifiers; the dedicated adapter separately fingerprints actual checkpoint/config/tokenizer bytes. Keep the model license, attribution and any conversion record with external model assets.

```python
from titanos.client import Client
with Client("http://127.0.0.1:11435") as client:
    result = client.sentiment("polarity", "The repair worked well.", min_score=0.7)
    print(result["status"], result["label"], result["scores"])
```

```text
titanos --config models.json serve
titanos sentiment polarity "The repair worked well." --min-score 0.7
```

Use `-` as CLI text to read stdin. `--options` accepts the same request fields. The HTTP request also supports `input_modality: asr_transcript` and caller-declared `asr_uncertain: true`. A true uncertainty flag forces abstention while retaining model scores; false means the caller did not declare uncertainty, not that ASR quality was verified. `language` is caller metadata, not language detection.

In the browser, register or select a `sentiment` classifier, enter text and generate its typed result. The settings expose input-token, minimum-score and minimum-margin overrides. Chat models retain the chat interface; use the explicit API, CLI, Python client or MCP sentiment call to classify with an LLM.

## Result and uncertainty

The typed `text.sentiment` result includes the model/method, status, selected label or `unknown`, every supported class score, score semantics, thresholds, abstention reasons, input SHA-256/length/token count when available, and model provenance. `audio_affect` is null. Dedicated scores are labeled `classifier_softmax_uncalibrated`; they sum to one but are not an empirical probability of correctness. `calibrated` is always false in this slice.

The decision abstains below `min_score`, when the top-two score margin is below `min_margin`, or when ASR uncertainty is declared. These are configurable heuristics, not calibrated operating points or reliable domain-shift detection. Scores can be confidently wrong, and uncertainty does not create a neutral or mixed class absent from the model.

Request thresholds override model defaults, then process defaults. Environment settings are `TITANOS_SENTIMENT_MIN_SCORE`, `TITANOS_SENTIMENT_MIN_MARGIN` and `TITANOS_SENTIMENT_MAX_INPUT_TOKENS`. The classifier applies the lower of the requested/configured input limit and usable checkpoint/tokenizer context. It tokenizes without truncation and rejects excess input before the forward pass. `TITANOS_MAX_TEXT_CHARS` still applies. No text is silently shortened.

## Existing LLM route

A registered local Transformers or OpenVINO GenAI `chat` model may use the same endpoint. The adapter requests exactly three polarity weights and an abstention flag, then performs one strict JSON validation pass. The result explicitly reports `constraint_mode: prompt_then_validation` and `llm_generated_weights_uncalibrated`. This is not grammar-constrained decoding or probability calibration.

Malformed, fenced, duplicated-key, non-finite, extra-field, non-normalized or length-truncated model output becomes typed `unknown`/`abstained` with `invalid_model_output`; it is never repaired, retried or converted to neutral. A model-declared abstention is retained. Registered provenance and the exact prompt hash are returned; unavailable weight fingerprints/device observations remain null rather than being invented. The native llama.cpp chat API retains its separate JSON-schema path; that backend is outside this first sentiment wrapper.

LLM `max_tokens`, temperature and top-p follow the ordinary request/model/process chat-default policy. Classifiers reject those generation controls and use `max_input_tokens`. LLMs reject the classifier-specific input-token control; existing chat/context limits bound their complete wrapper prompt. Very small text limits can reject the fixed instruction overhead with a clear error before inference.

The extension timeout uses the existing cooperative cancellation event, including queue waiting and checks around model execution. Model loading and a single classifier forward pass cannot be forcibly preempted by that event. It is not a hard process wall-clock kill guarantee; callers requiring that boundary must use their owned-process supervision.

## Scope

This describes sentiment expressed by text or a transcript. It does not measure voice tone, prosody, speaker emotion or a person's actual feelings. There is no automatic activation on user messages. Raw-audio affect, aspect sentiment, a mixed classification head, calibrated probabilities and additional language/domain qualifications require separate model/runtime work.

The initial private evaluation uses synthetic proposed labels, not a human-annotated deployment benchmark. Keep format validity, coverage/abstention, classification agreement, calibration, latency and native package qualification as separate results. A poor result from one cached small LLM and prompt is not a conclusion about all LLMs.
