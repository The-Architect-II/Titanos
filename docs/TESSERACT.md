# Native printed OCR

Register a directory containing external `eng.traineddata` (or other matching LSTM language files):

```console
titanos add print-ocr /models/tessdata --task image-to-text --backend tesseract --device cpu
titanos ocr print-ocr page.png --psm 3 --language eng --include-geometry
```

The HTTP endpoint remains `POST /v1/images/ocr`. Python uses `client.ocr(model, image, psm=3, include_geometry=True)`. MCP uses the typed `extension_infer` operation `ocr`, available over HTTP and stdio. No HiveMind service or binary is involved. Existing Transformer OCR remains available as its own backend.

In a native bundle the executable resolves only to `_internal/native/tesseract/tesseract.exe` (or the native platform basename). Source installations can set `TITANOS_TESSERACT_EXECUTABLE` to an absolute, trusted executable outside the registered model directory. There is no model-folder executable discovery, PATH search, automatic installation or download. The current source-built package has native Windows AMD64 startup and four-line CLI qualification; other builds require their own qualification.

Tesseract uses CPU LSTM recognition (`OEM1`) and external `.traineddata`. It accepts `language`, `psm`, `dpi`, `preserve_interword_spaces`, `char_whitelist`, `char_blacklist`, `include_geometry`, `max_output_bytes` and the shared request `timeout` in milliseconds. PSM3 is the generic default; PSM7 is a single line. OSD modes are excluded from this profile. `prompt`, `max_tokens`, temperature and other language-model decoding controls fail explicitly. Model defaults accept only the native OCR control fields; request values override defaults, then process settings supply omitted values.

Process controls are `TITANOS_OCR_LANGUAGE` (eng), `TITANOS_OCR_PSM` (3), `TITANOS_OCR_MAX_OUTPUT_BYTES` (8 MiB), `TITANOS_OCR_MAX_TSV_ROWS` (100000) and `TITANOS_OCR_MAX_MODEL_BYTES` (512 MiB combined selected data). Existing image byte/pixel and client timeout settings apply. Explicit per-request byte limits can lower the process limit; raising the process settings permits larger jobs. Page formats are decoded through Titanos/Pillow into a temporary RGB PNG, which the minimal native build supports.

Each request owns its process tree, files and deadline. Windows launches suspended into a private kill-on-close Job before execution. Unix launches in a new session and kills the owned process group during cleanup. Cancellation/output-limit failures close the same owned tree. Native stdout/stderr and generated text/TSV share the byte budget. Temporary files are removed when the request exits. Runtime/model identities are returned as hashes; model data is rechecked after inference. No Torch import is needed by this adapter. The cleanup rules for that Job are in [Windows native command cleanup](WINDOWS_COMMAND_CLEANUP.md).

The current release lane is Windows AMD64. The Unix process-group path is source-only and unqualified; it does not claim cleanup after an uncatchable parent crash. Windows Job closure covers parent death as well as ordinary cancellation. Output-size checks run during the native process and once it stops; the process can briefly write past its budget between checks, but oversized output is rejected before being read into a response.

`text` removes exactly one native terminal CRLF or LF, and `raw_text` preserves all decoded native text. There is no case, spelling, whitespace or punctuation correction. `lines` follows `text.splitlines()`. Optional `geometry.rows` preserves TSV order and native page/block/paragraph/line/word IDs, boxes in original input pixels and word confidence 0–100; structural -1 values remain distinct and confidence is uncalibrated. Invalid or oversized geometry fails, rather than being clamped. TSV geometry remains source-contract tested until the new native API build is qualified.

Both binary builders accept `--tesseract-dir` pointing to the verified, model-free native runtime directory. They require a Pillow profile, verify every source/runtime/license manifest entry, copy the whole bundle under `_internal/native/tesseract`, incorporate its components into notices/SBOM and bind the copy in the final manifest. The English model archive stays external. No native model execution occurs during source checks or bundle-input verification.

Validation: `tests/check_tesseract.py` uses synthetic OCR files and harmless child processes to check API shapes, rejection before decode, temporary-file cleanup, output bounds, deadlines and descendant cancellation. This is separate from real OCR quality and the subsequent standalone HTTP/CLI/Python/MCP qualification.
