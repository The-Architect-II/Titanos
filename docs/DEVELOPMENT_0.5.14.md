# 0.5.14 development changes

> **Historical record (Titanos 0.5.14).** This page is kept for provenance and is not updated. It describes that version's source and recorded results only; current behaviour is documented in [masked image editing](image-masks.md) and [relative depth](DEPTH.md). Results recorded here do not qualify the current source or any later package.

This increment adds binary-alpha masked editing through actual SD/SDXL
inpainting, optional disclosed RGB preservation, and typed relative Depth
Anything inference. The common HTTP/CLI/Python/MCP surfaces retain explicit
input, configuration and cancellation boundaries. Studio adds source/mask
uploads for editing; depth currently uses the API, MCP, CLI and Python client.

Independent source review caught and repaired three depth boundaries: output
budget rejection before pixel conversion, external-backbone loading options,
and cancellation observed during tensor transfer. The image output cleanup now
attempts every returned image even when one close fails, retaining an earlier
generation/write error. The native builder also finds CPython notices in the
standard-library layout used by the Mac standalone interpreter and rejects a
package when that notice is missing.

Read [mask semantics](image-masks.md), [relative-depth semantics](DEPTH.md),
the [CPU text/vision profile](BUILD_TEXT_VISION.md), and
[GenAI image profile](GENAI_IMAGES.md) for exact supported groups and controls.
No new numerical dependency or model weights are baked into this source wheel.
The MCP catalog adds `estimate_depth`; old task and tool names remain available.

Source checks use doubles and real image parsing. Those checks do not establish
trained-model output quality, accelerator execution or native cancellation
responsiveness. Earlier releases, failures and hardware qualification remain
bound to their exact versions. Full-engine and current-film acceptance remain
separate from this development increment.
