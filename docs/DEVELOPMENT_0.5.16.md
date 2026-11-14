# 0.5.16 development changes

> **Historical record (Titanos 0.5.16).** This page is kept for provenance and is not updated. It describes that version's source and recorded results only; current behaviour is documented in [prompted SAM masks](SEGMENTATION.md). Results recorded here do not qualify the current source or any later package.

Prompted SAM segmentation accepts one image and either a bounding box or labeled points in the decoded image coordinate
frame. It returns native mask alternatives as binary RGBA edit masks: transparent pixels are selected, opaque pixels remain outside
the selection. Scores are the model's finite, uncalibrated IoU estimates; no semantic class name, calibrated confidence or perfect contour is invented.

HTTP `/v1/images/segment`, the `segmentation` extension, CLI, Python client and
typed MCP expose the same controls. Native alternative identities, processor
alignment, image/mask hashes and geometry remain explicit. A single-mask request
still works when the configured output cap is one. Input/output/cancellation
boundaries and the builtin eager/SDPA loading policy are checked before avoidable
work. The CPU text/vision profile includes SAM class imports without new numerical
dependencies; models remain external.

Source tests and import checks do not prove learned mask quality or native
execution. Existing relative-depth and image-edit results retain their exact
version/model/device scope. This release also normalizes copied FFmpeg diagnostic
prefixes while preserving original native inputs and source archives.
