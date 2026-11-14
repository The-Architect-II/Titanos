# Studio language and workflow contract

Studio supports exactly English (`en`), Spanish (`es`), Japanese (`ja`) and
Simplified Chinese (`zh-CN`). Simple remains the initial mode. Language and mode
preferences retain the existing safe browser-storage and OS-language fallback.

Simple keeps the model, task, prompt, results and everyday controls visible.
The **Runtime** panel, removal of registrations, and backend/device implementation
tags are available in Advanced. An already-open runtime-details panel is hidden
in Simple without discarding its state. The media section is called **Jobs**,
**Trabajos**, **ジョブ**, or **任务**, with matching refresh, empty-state and
queued-job guidance.

Owned UI copy uses explicit semantic keys in `web/i18n.js`. Static HTML labels
carry `data-i18n` attributes; dynamic controls, messages and accessible labels use
the private `ui(...)` message marker and targeted `setContent`/`setUiAttribute`
bindings. Language changes update those marked elements. They never look up
translations by matching arbitrary DOM text.

This includes discovery, compatibility/license and pinned-plan presentation,
acquisition and runtime progress/retry/cancellation, registration, lifecycle,
device assignment, generation settings, music controls, result explanations,
asset-job controls and document-retrieval labels/status. The retrieval workflow
keeps its input controls separate from translated caption spans so changing
language cannot replace a question field or reset its contents.

Prompts, lyrics, returned text, document passages, claims, quotations, source
URIs, model/device IDs, protocol values, file names and raw diagnostic JSON stay
original. A string equal to an English UI label is still user content unless the
application explicitly constructs a marked UI message. Interpolation is one
pass; literal `{{...}}` inside user content remains literal. Provider notes are
kept verbatim in Advanced technical details; the primary compatibility guidance
is localized. Known structured API errors receive localized next-step hints,
while their backend diagnostic message remains unchanged.

Changing locale or Simple/Advanced mode preserves the selected model, per-model
assignment drafts, pending-operation ownership, field values, license checkbox
state and partial/stopped result labels. Existing model/epoch guards continue to
discard stale assignment and lifecycle responses.

Device policy and compatibility guidance are translated in all four languages:
automatic recommendation versus observed placement, explicit CPU, saved GPU
assignments, restricted visibility, missing runtimes, incompatible homogeneous
groups, independent replicas and backend-specific tensor parallel limits.
Stable device IDs and raw inventory diagnostics remain unchanged. Switching
language or mode never submits an assignment or replaces an explicit runtime
profile. Dirty assignment drafts must be saved or discarded before a policy
change; masked saved devices stay pinned and show an unavailable warning.

## Music readiness and controls

For ACE-Step, any acquisition plan runtime status other than `available` blocks
Download and run before acquisition starts. A missing runtime points the user to
a complete matching music distribution; `unsupported_adapter` points to a
supported variant/runtime. The UI does not download runtimes or write receipts
to work around that state. llama.cpp's existing managed runtime preparation
workflow remains available.

Music seed, guidance scale and steps appear in Advanced only when their exact
capability flags are true. Values survive language/mode changes. Undeclared
controls are omitted from the music request even if another model left a value
in the field. Blank duration remains omitted so the normal request/model/server
default precedence applies.

## Verification

`node tests/check_studio_modes_i18n.js` checks exact locale catalogs, placeholders,
preference storage and fallback. Existing Node UI tests retain form/payload,
assignment-race, cancellation, lifecycle, retrieval and responsive-design checks.

`node tests/check_studio_workflow_i18n.js` exercises the actual HTML/scripts in
headless Chromium against intercepted fixtures. It checks all four languages,
live locale changes, discovery/plans and known error hints, raw text/citation
preservation, assignment drafts, license acknowledgement, music readiness gates,
declared music controls, preserved llama setup and 390-pixel layout. It makes no
model or real service calls and launches Chromium with GPU rendering disabled.

The optional browser test uses an installed Playwright module and browser; it
does not install or download them. `TITANOS_PLAYWRIGHT_MODULE` and
`TITANOS_CHROME_PATH` can point to existing installations.
`TITANOS_I18N_EVIDENCE_DIR` selects a separate output directory for screenshots and
results. Fixture/browser checks establish UI behavior; they do not establish
native inference, a packaged release or a model's output quality.

`node tests/check_studio_accelerators.js` exercises the actual page in headless
Chromium with intercepted v2 inventory, assignment and runtime-profile APIs.
It covers explicit CPU/automatic policy, homogeneous sharding, distinct replicas,
native versus framework tensor limits, masked saved devices, and GPU runtime
preparation despite an available older CPU runtime. No real device probe,
model call, server listener or primary desktop input occurs.
`TITANOS_ACCELERATOR_UI_EVIDENCE` selects its fixture evidence directory; the
same installed Playwright/browser variables above apply.

`node tests/check_studio_workflow_i18n.js --simple-only` runs the narrow
presentation check: technical controls hidden in Simple and retained in
Advanced, the four creation labels, and unchanged model/prompt/result/settings
state. This mode issues no fixture inference or mutation requests.
