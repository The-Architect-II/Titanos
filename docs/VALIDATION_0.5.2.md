# 0.5.2 hardware validation

> **Historical record (Titanos 0.5.2).** This page is kept for provenance and is not updated. It describes that version's source and recorded results only; current behaviour is documented in [hardware, runtime builds, and distribution](HARDWARE.md), [DirectML on Windows](DIRECTML.md) and [capability and qualification matrix](CAPABILITY_MATRIX.md). Results recorded here do not qualify the current source or any later package.

This source successor adds the runtime selection changes exercised during the native hardware window. The
existing 0.5.1 source and application archives are preserved. A source change, runtime test, and newly
packaged binary are separate claims; each binary's own build manifest and validation receipt determine its qualification.

| Hardware | Runtime and observed workload | Evidence boundary |
|---|---|---|
| AMD Radeon 840M | DirectML; ImageNet MobileNetV2 classification and MiniLM embeddings | GPU-only ORT node profiles, CPU fallback disabled, ordinary CLI/TCP passed |
| Ryzen 7800X3D Radeon iGPU | DirectML; MobileNetV2 classification | Explicit DXGI AMD adapter, GPU-only node profile, ordinary CLI/TCP passed |
| NVIDIA RTX 3500 Ada | DirectML MiniLM embeddings; released 0.5.1 CUDA bundle trained chat | Explicit DXGI adapter for DML; frozen CUDA worker verifies physical GPU UUID and records GPU kernels; greedy CPU/GPU responses agree |
| Intel Iris Xe | DirectML and OpenVINO; MiniLM embeddings | Explicit runtime addresses, GPU profiles and CPU parity; multiple runtime views are not extra physical cards |
| Intel Arc B580 and A770 | OpenVINO; trained classifier and MiniLM embeddings | Fresh device identities, native GPU profiles, CPU agreement, ordinary HTTP and MCP inference on both cards |
| Intel AI Boost NPU | OpenVINO; trained classifier/embedding checks | Explicit NPU execution; prepared graph/operator constraints apply |
| AMD Ryzen AI 5 340 NPU | VitisAI/VAIML; MobileNetV2 classification | All 104 operators in hardware context; VitisAI node execution and no CPU EP nodes; FP32 checkpoint compiled to BF16 |
| Qualcomm Snapdragon NPU | QNN HTP; trained digit classifier | Native ARM64 execution and CPU agreement; input/model preparation constraints apply |
| Qualcomm Adreno GPU | DirectML from WindowsML wheel; trained digit classifier | Native ARM64 GPU node profile, fallback disabled, ordinary CLI/TCP passed |

The MobileNet input was an image containing a cat. GPU results kept the CPU's top
class, with maximum logit difference about 9.1e-6 on the tested Radeon paths.
The Ryzen NPU kept the same top class, with logit cosine similarity 0.99664 against
FP32 CPU; BF16 computation changes numerical values. Its initial compile took
about 572 seconds; a cached request then completed in seconds.

MiniLM used three sentences: two about cats/cups and one about a spacecraft.
The related pair scored about 0.635 cosine similarity versus 0.118 for the unrelated
sentence. DirectML/OpenVINO outputs matched CPU within roughly 2.2e-7 on the tested
devices. These samples check real trained inference and semantic behavior; they
are not full ImageNet or sentence-embedding benchmarks. The digit classifier
checks likewise separate trained-model accuracy from agreement with CPU.

Pinned public checkpoint identities:

- `onnxmodelzoo/mobilenetv2-12` at `23a2a25eaaa6fcd61865414c8a1acf13dda66b7a`; a fixed batch=1 derivative preserves every trained initializer.
- `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; Titanos exports fixed-shape ONNX/OpenVINO graphs and checks CPU parity.

The Intel A770 initially reported Windows driver Code 31. The
operator selected an existing signed Intel 6790 driver for the
paired B580 without rebooting. Both cards then reported Code 0,
and a fresh inference process loaded the 6790 OpenCL and
compiler DLL family. Subsequent 0.5.2 classifier, embedding, HTTP and MCP
tests passed on both cards. This was a lab driver
repair, not a Titanos auto-repair feature; Titanos does not install drivers. Unreachable hosts remain untested.
Apple-specific evidence remains in its own native receipts;
Core ML provider selection alone is not proof of Neural Engine execution.

Regression checks cover vendor ONNX version reporting, DirectML adapter
selection/software rejection, session flags, provider options, discovery identity
strength, indexed OpenVINO registration, and runtime profile selection. Those
synthetic tests do not substitute for the native trained-model results above.
