# Titanos Studio design

Studio uses the Titanos v2 identity (honeycomb T, wordmark, lime cube) in the
"Cabinet" theme: the HM3 release-edition look at operator density. Numbered
bays carry the structure, violet is the structural ink, lime means ready or go,
and cyan means a request leaves this computer. The header acknowledges the
lineage ("Born of HiveMind") in plain small text; Titanos remains a standalone
local inference application and needs no HiveMind component.

This theme replaced the flat neutral theme on 2 Oct 2026. It is built from the
style brief and the judged prototypes of that day; human design acceptance by
the project owner is not claimed here.

## Identity

- `/ui/titanos-wordmark-v2.svg` is the integrated wordmark. Its honeycomb T is
  the actual first letter in Titanos; there is no second plain T. The lime cube
  is the lowercase-i dot. The header renders it at 168px wide with no filter.
  The "tanos" letters are still live text in the SVG (Segoe UI, then Arial);
  outlining them, and a small-size favicon variant, are open identity decisions.
- `/ui/titanos-honeycomb-t-v2.svg` is the icon and SVG favicon.
- Beside the wordmark sits a violet **STUDIO** plate, then **Born of HiveMind ·
  Standalone local inference** as one 12px muted line (hidden below 1180px).
  The footer is one sentence: "Titanos runs on its own. It does not need
  HiveMind."

Both SVGs are local package assets with accessible title/description content.
Nothing in Studio loads an external font, script, stylesheet or image: the
display face and every glyph are inlined in `style.css` as `data:` URIs.

## Theme contract

One `:root` block in `src/titanos/web/style.css` defines the tokens. Contrast
ratios (WCAG 2.x) are against the canvas `#050509` and the bay `#0D0D18`;
`tests/check_studio_design_ui.js` recomputes every pair below on all five
surfaces (canvas, bay, row, raised bay, selected fill).

| Token | Value | Role | vs canvas | vs bay |
|---|---|---|---|---|
| `--bg` | `#050509` | canvas, input wells, `theme-color` | - | - |
| `--bay` | `#0D0D18` | bay fill | - | - |
| `--cart` | `#11111F` | result cards, tiles, candidate cards | - | - |
| `--bay-hi` | `#16162A` | hover, disabled RUN | - | - |
| `--slot` | `#22104A` | selected model fill, step numerals | - | - |
| `--seam` / `--seam-hi` | `#2A2640` / `#3A3458` | hairlines, sub-bay dashes, lattice; never a control's only boundary | 1.4 / 1.8 | - |
| `--ctl` | `#6E5CA8` | control boundary: inputs, quiet buttons | 3.64:1 | 3.46:1 |
| `--frame` | `#8000FF` | 2px bay frames, hard offset shadows, plates, structural buttons (white text 6.25:1) | 3.26:1 | 3.09:1 |
| `--frame-hi` | `#A855FF` | input hover border | 5.24:1 | 4.97:1 |
| `--violet-ink` | `#B488FF` | violet text: links, sub-indices, step numerals, chevrons | 7.66:1 | 7.27:1 |
| `--lime` | `#A6FF00` | ready, the one go button per region, focus ring | 16.40:1 | 15.56:1 |
| `--lime-ink` | `#0F1A00` | text on lime (14.49:1) | - | - |
| `--lime-shadow` | `#4E7D00` | RUN's hard press shadow | - | - |
| `--cyan` | `#32D4FF` | the request leaves this computer: cloud glyphs, route note, External tags, cloud spines | 11.62:1 | 11.03:1 |
| `--magenta` | `#D66BFF` | counts and cost: stream counter, result and job counts, priced cost | 7.18:1 | 6.81:1 |
| `--warn` | `#FFC857` | warnings, one-time secret box | 13.23:1 | 12.55:1 |
| `--error` | `#FF4D6D` | errors, danger buttons, failed spines | 6.33:1 | 6.01:1 |
| `--text` / `--text-2` / `--text-3` | `#FFFFFF` / `#C9C1E8` / `#8F88AD` | primary, secondary, muted text | 20.35 / 11.89 / 6.10 | 19.30 / 11.29 / 5.78 |

Rules:

- **Bays.** Each top-level section is a numbered bay: a 2px `--frame` border
  drawn as a chamfered layer (top-left and bottom-right, 12px at 45°), a `--bay`
  fill and a violet plate on the top frame ("01 // MODELS"). Only bays and the
  selected model row carry the violet frame; rows, sub-bays, inputs and quiet
  buttons use seams or `--ctl`.
- **Hierarchy.** The hard `drop-shadow(4px 4px 0 var(--frame))` (zero blur)
  marks the working bays only: 03 Run a model, 04 Results once it holds a
  result, and a collapsible bay while it is open. A collapsed bay (02, 06, 07)
  is a slim plate row: the plate with its disclosure caret inside it, and its
  one-sentence summary, which wraps at narrow widths and is never ellipsized
  mid-word.
- **Lit corner.** A bay that holds something live (a loaded model in the rail,
  a request in the composer, a streaming result, a running job) draws a 2px lime
  line exactly on its top-left chamfer.
- **One gradient.** The only gradient is the 2px violet-to-lime route rule under
  the header. No blur, glow, frosted surface, text shadow, pill, hover lift,
  scale or rotation. RUN moves 3px when pressed (`translate`, its only motion).
- **Type.** Body text is the system stack (`-apple-system`, `BlinkMacSystemFont`,
  `Segoe UI`, `Roboto`, `Noto Sans`, `Helvetica Neue`, `Arial`, `sans-serif`;
  a CJK stack for Japanese and Simplified Chinese); ids, readouts, receipts and
  job rows use the monospace stack. The vendored display face, Chakra Petch
  (SemiBold and Bold Italic, Latin subsets), is used only for plates, numerals
  (step numbers, sub-indices, counters, provider monograms) and RUN.
  Uppercase and letter-spacing appear only on the plates, the boot line and RUN;
  everything else is sentence case at 12px or larger.
- **Buttons are quiet by default** (transparent, `--ctl` border). Lime fills
  only the `.go` action of a bay, at most one per bay: Search in 02, RUN in 03,
  Answer with sources in 06, Connect in 07. Violet `.structural` buttons are
  state changes that are not "go": Load, Add model, the server-key Connect in
  the rail, Save assignment, Download and run, Activate after review and Save
  credential. Remove, Revoke, Stop and Cancel use the error colour.
  `check_studio_design_ui.js` fails on a lime button outside `.go`, and the
  browser test counts the lime fills of every rendered bay.
- **State glyphs.** Pointy-top hex glyphs, always beside a state word: hollow
  (not loaded), half (loading), the three-tone lime cube (ready), violet core
  (running, queued), red slash (error), amber notch (warning), cyan orbit
  (cloud route). Colour is never the only signal.
- **Radius** is 2px on controls and 0 elsewhere; nothing exceeds 6px.
- **Focus** is a 2px outline at 2px offset on every focusable element: lime on
  fields, links, rows and summaries, white on buttons, red on an error message.
  It never relies on a glow. RUN clips itself to its chamfer, which would clip
  its own outline, so its white ring is drawn on the unclipped wrapper
  (`#submit-wrap:has(#submit:focus-visible)`); the mobile section links draw
  their hex on `::before` and stay unclipped. The browser test compares RUN
  with and without keyboard focus.
- **Forced colors** (Windows High Contrast): buttons and RUN get a 2px
  `ButtonText` border without clipping, bays a `CanvasText` outline and plates
  a border, and the hard shadows are dropped, so the structure survives when
  the palette is replaced.
- **Reflow.** Nothing scrolls sideways at 320 CSS px in any locale or mode:
  boot values shrink and wrap, a long plate title ends in an ellipsis, and
  buttons may wrap their label at 640px and below.
- **Motion.** 120ms colour transitions, the cube's three-step appearance, the
  stepping violet cell of a running job and the progress lattice's current
  cell. `prefers-reduced-motion: reduce` stops all of them.
- CSS carries no copy: `content` is empty, a middle dot or the step counter.

## Layout and behavior

- **Header** (60px, sticky on desktop): wordmark, STUDIO plate, lineage line,
  Mode and Language selects, and the connection line: a cube glyph, "Connected,
  N models" and the server's host:port in mono, then Refresh.
- **01 Models** (300px sticky rail): local rows first, then a "Cloud routes"
  divider and the cloud rows. Every row is two lines (about 50px; three when a meta chip wraps): a hex glyph,
  the model id in mono and the state word, then the task (for a cloud row also
  the provider, in cyan; in Advanced also backend and device) and a **Details**
  disclosure named "Details for <model>". A meta chip that does not fit wraps
  to its own line rather than shrinking into an ellipsis. An id never breaks mid-token: it ends
  in an ellipsis with the full id as its title, and a cloud id's `@upstream`
  suffix is dimmed. A loaded model's id carries a 2px lime underline. The rail
  scrolls inside a hairline boundary that ends on a mono cue ("4 more below")
  when rows are out of view; its empty space shows the honeycomb lattice. The
  rail shows whole rows only: the scroller is cut back to the last row that
  fits (never through a row's text), a row sliced by the top edge is brought
  into view once a scroll settles, and the cue is recounted on every scroll,
  resize and disclosure toggled in the rail. Register a local path and
  Connection scroll inside their own share below the rows (the rows keep about
  four), so every field stays reachable. At 900px and below the rail becomes a
  horizontal strip of whole cells above the work area (three across, two at
  640px and below), with a "3 of 12" position cue in its readout.
- **02 Add a model**: a collapsible bay, open on the first load only when no
  local model is registered; afterwards it follows the user. Search results,
  plans and progress live inside it; progress is a row of ten hex cells beside
  the native progress element.
- **03 Run a model** (composer): steps numbered in hex cells in the left gutter
  (Model, input, Settings, Advanced configuration, Run); the numbers come from a
  CSS counter, so Simple shows 1-3 and Advanced 1-5. Under the model select:
  for a cloud model a cyan route note ("This route leaves your machine. Titanos
  sends only what you run here."), then a dotted-leader readout (route, state,
  device, task; in Advanced also backend, mode and source). The device block is
  a one-line readout of the device policy that opens on demand. Settings and
  Advanced configuration are dashed sub-bays; a cloud model shows neither the
  device block, the device assignment nor the require-loaded switch. The last
  step states the run state ("Ready. Press Run." or "Not loaded. Load it first,
  or Run loads it if this server allows."), not the task again. The status line
  is the bay's last line: it collapses to zero height while empty but stays
  rendered as the live region; while a job runs it follows the job (Queued,
  Running, Cancelling).
- **04 Results**: while empty, a boot readout (`> TITANOS_READY` and dotted
  lines for engine, models, cloud, jobs and the selected model), a hive map
  with exactly one cell per model (no placeholder cells; none at all with no
  model), and an empty-state sentence with the empty lattice below it, never
  behind it. After the first result the readout collapses to one line. Each
  result card has a spine (violet while streaming, lime when recorded locally,
  cyan when recorded from a cloud route, red on failure, a seam for your own
  prompt), prose capped at 75 characters per line, a live token or chunk
  counter for streams, images in a well of at least 96px with their pixel size
  under them, and a receipt under a dashed tear line: "Recorded result" (or
  "Failed"), "External" for cloud routes, then the provenance from the response
  headers ("Served by cloud:openai · gpt-4o-mini@openai · fallbacks 0") or the
  model and device, the time, tokens or size, the cost when the call was
  priced, and the time the result was recorded (a job reopened from Jobs keeps
  the job's own finish time). Separators never start a line. Counts are
  singular or plural ("1 result", "2 results").
- **05 Jobs**: a scoreboard with column headings (state, started, job, task,
  model, time, actions), a count readout ("1 running · 2 done · 1 failed · 1
  cancelled") that adds up to the rows, real elapsed seconds that keep counting
  on running rows, one violet cell stepping along a six-cell hex lattice on
  running rows (static under reduced motion), and the note inline under a
  failed row (red) or a cancelled row (amber).
- **06 Answer from your documents** and **07 Cloud and routing** are
  collapsible bays after Jobs. Bay 06 groups its form into numbered sub-cells
  on a compact grid: 06.1 Models (three selects and Refresh), 06.2 Passages and
  question, and 06.3 Retrieval and answer settings (Advanced, folded); Answer
  with sources and Stop share one action row.
- **Offline.** When Titanos stops answering, the header shows an amber notch
  glyph and "Connection needed", the status line says "Titanos is not
  reachable at <host:port>. Start it or press Refresh." instead of the
  browser's error, RUN is disabled, the rail dims and no bay keeps its lit
  corner.
- Below 900px a sticky strip of hex numbers (01-07) links to the bays. At 640px
  and below the model row wraps (select on its own line, buttons sharing the
  next), controls become single-column, RUN fills the width and the job rows
  stack. No horizontal overflow at any width.

## Cloud and routing (bay 07)

Shown in both modes when the server reports a broker block and the principal
holds `broker:read`. A cyan note states that these routes leave this computer;
a two-column readout gives role, cloud policy, vault, upstreams, egress and
keyless use. Numbered sub-sections follow: **07.1 Connect a provider** and
**07.2 Cloud routes** are open; credentials, aliases and virtual models,
policy, usage and decisions, and keys fold away and appear only in Advanced.

- Connect a provider is a form, not a grid: a provider select built from
  `GET /api/broker/profiles` (marked "connected" only when its upstream is
  enabled and has a credential, "NAME found" when a variable is set, or
  "configured, not enabled"), a key choice (a variable from this server's
  environment by name, from `GET /api/broker/env-candidates`, or a pasted key
  in a write-only field; each radio has its own label and the select or key
  field is named by the option text; focus never changes the chosen option,
  so Tab passes the variable picker and the key field on its way to Connect,
  and only picking a variable or typing a key chooses that option; the panel
  re-renders for each provider and after a connect, and keyboard focus stays on
  the control that had it, so Arrow keys step through the providers and Enter
  on Connect leaves focus on Connect; the write-only key field is emptied and
  swapped for a fresh one on submit, and when Enter was pressed in it the fresh
  field takes focus, here and in the expert credential form), an
  optional base URL (Advanced, or
  required when the profile has none), and for the keyless Studio a switch that
  allows keyless use of the new provider. Connect is the panel's one lime
  action. A broker refusal is shown with the server's own reason, without the
  hints written for local model routes. After a connect, the next steps are
  Studio's own translated sentences built from the answer's fields.
- Forms in the expert sections are top-aligned grids with paired actions in one
  row (Save credential beside Remove credential); the usage and decision tables
  say "No cloud requests recorded yet." or "No routing decisions recorded yet."
  instead of showing bare headings.
- Provider tiles show a hex monogram, the name, an Online/Disabled/Breaker open
  chip, and a readout (credential and variable name, fingerprint, validation,
  state, catalog). The enable switch is a native checkbox labelled with the
  current state word.
- The one-time key secret sits in a box with a 2px warning border.

## Verification boundary

`node tests/check_studio_design_ui.js` checks the identity references, every
token and the contrast of each pair in use, the banned tells (blur, glows, text
shadows, extra gradients, transforms, pills, radii above 6px, the concept-page
and leftover colours, Inter and Orbitron, external URLs, marketing hero), that
uppercase, tracking and the display face stay on plates, numerals and RUN, the
violet roles, the quiet default button and the `.go` count per file, the shadow
hierarchy, the unclipped focus rings, forced colors, the inlined font against
`fonts/*.woff2` and its licence, the live regions, breakpoints, reduced motion
and focus. `node
tests/check_broker_ui.js` checks the Cloud and routing panel with a DOM double:
write-only secrets, the R29 Studio header, the R30/R31 connect flow and its
typed messages, scope gating, one lime action and focus kept across re-renders. `node
tests/check_studio_broker_browser.js` renders the panel in a real browser
against a real server and the loopback OpenAI mock and checks computed styles
(no gradient or shadow, caps and tracking only on plates, radii at most 6px,
one lime action in the panel and at most one in every bay, error-coloured
danger buttons, a cyan provider badge, a monospace provenance line, no
horizontal overflow), RUN's visible keyboard focus, the accessible names of the
connect fields and the rail's Details controls, and the keyless connect, whose
environment option is chosen and submitted by keyboard alone (the chosen source
is asserted at every Tab stop), the Provider select stepped by Arrow keys
with focus kept on it after each re-render, and the typed credential and the
pasted connect key each submitted by Enter in their own field, with focus kept
in the fresh field. `node tests/check_studio_workflow_i18n.js`
renders Studio at 1600x1100, 390x844 and 320x800 (all four locales, Simple and
Advanced) and fails on horizontal overflow; with `--simple-only` it also loads a
longer catalog and checks the rail at 1440x900, 1024x768 and 390x844 (no row
sliced by the scroller, no squeezed meta chip, the cue equal to the rows below
after scrolling and after a Details toggle, Connection and its key field
reachable while Register a local path is open) and that every collapsed bay
sentence wraps in all four locales. `docs/assets/studio-simple.png` and
`docs/assets/studio-providers.png` record two desktop renderings; their
provenance is in `docs/assets/README.md`. These checks establish source
behavior; native packaging, real-model quality and deployment remain separate
gates, and so does the owner's design acceptance.
