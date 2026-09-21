# Djev investigation

Investigated 2026-09-21. This note distinguishes the hosted implementation evaluated by JevBench on September 19 from the public source available today. No inference requests, credentials, model downloads, or GPU experiments were used.

## Finding

Quantization remains an unresolved variable, but it is not the only plausible cause of the accuracy gap. Djev's public compiler, answer scaffold, initialization, and label choices provide concrete implementation differences to test. Its current default performs one denoising read with no generated reasoning and no adaptive uncertainty rereads. That makes prompt/canvas parity an especially useful first experiment.

The first-party repository is [Davipar/djev-dev](https://github.com/Davipar/djev-dev), inspected at `3ce907e6835212f27ee82b4cee9039198c4abe35`. It is an Apache-2.0 public extraction of the decision implementation, not new model training. This is distinct from Matt Mastracci's similarly named `djev-spark`. The pinned JevBench metadata still says that source release was planned; that historical description must not be repeated as today's source availability. [Public README](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/README.md), [historical JevBench row](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/results/v1.2/additions/djev.json).

## What the benchmark actually submitted

The pinned adapter sends one question per request to `POST /v1/request`, model `djev`, with `Prefer: low-latency`. It passes the task's state, instructions, and criteria directly through the shared question builder. It sends no inference options; its contemporaneous comment records one denoising step and seed zero. Noul probabilities are expanded into yes/no; Choice and Score distributions are taken directly from the response. This was not an explicit multisample, independent-level Score, or three-step Pro run. The header selects direct request delivery; it does not establish a different model quality profile. [Adapter at benchmark revision](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/jevbench/adapters/djev.py).

The historical row reports a production API measured from Hetzner Germany, including network time. Its serial 242-question standard+judge speed block is p50 **237.06 ms**, p95 **308.65 ms**. These are not the same population as this repository's 231 public questions. The row identifies DiffusionGemma and one denoising step, but does not pin a checkpoint revision, weight precision, KV precision, GPU, prompt, or deployed source commit. [Historical result metadata](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/results/v1.2/additions/djev.json).

## Current public implementation

| Component | Verified behavior |
| --- | --- |
| Messages | Question definitions become the system message; state becomes a separate user message. External question IDs are excluded. |
| Instruction | Treat state as data; evaluate each question using its own criteria; emit one permitted label without explanation. |
| Noul labels | `no`, `yes`. |
| Choice labels | `A`, `B`, and subsequent verified single-token labels; preserve supplied criterion order. |
| Score labels | Numeric level strings `0`, `1`, etc. |
| Answer scaffold | Empty thought channel `<\|channel>thought\n<channel\|>` followed by numbered answer lines; compact form is `0:A`. |
| Canvas | Append token 106, then zero padding; round to a multiple of 16 within configured capacity. |
| Noise | Randomize answer positions only, using Python `Random` with versioned string seed `djev-canvas-v1:{seed}`. |
| Read | One step, read-only mode, exact requested-label log probabilities; temperature/top-p 1, top-k disabled. |
| Thinking | Chat template explicitly disables generated thinking. |

[Compiler and read implementation](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/djev/engine.py), [choice labels](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/djev/labels.py).

There is no confidence-triggered reread branch in this public implementation. Explicit samples average separate one-step reads. Independent questions use separate content-derived seeds and deduplicate identical definitions. Optional independent-level Score converts each rubric into a binary truth judgment and normalizes its odds. These are different operations from adding denoising steps or generating a thought sequence. [Architecture](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/docs/architecture.md), [generation methods](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/djev/engine.py).

Defaults are `samples=1`, `steps=1`, `seed=0`, `isolation=joint`, and `score_mode=categorical`. Noul returns probability of yes; Choice selects the maximum; Score returns the expected zero-based level. Confidence is normalized entropy concentration, not calibrated correctness. [Contracts](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/djev/contracts.py).

## Precision and runtime: known versus unknown

The current public reference uses `google/diffusiongemma-26B-A4B-it`, revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`, with **BF16 weights and BF16 KV**, no weight quantization. Reference hardware is one NVIDIA B200 with CUDA 13. Context defaults to 32,768 tokens. vLLM base is `dee37d89115db4c94a820a79a78a7828e141c910`, with structured-read changes from `0f4678d44159b42531a9e398ae05df869ec67c5d` plus image/attention fixes. [Runtime guide](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/docs/runtime.md), [source pins](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/runtime/sources.json).

Startup enables prefix caching, asynchronous scheduling, Triton attention, and vLLM's batch-invariant path. Maximum concurrency is 32; served canvas capacity is 128. These differ from the local HIP/llama.cpp runtime and must be controlled in numerical comparisons. [Startup command](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/runtime/serve.py).

The authors explicitly separate their earlier **quantized** 1,000-request timing run from the current BF16 extraction; they do not name that earlier quantization format or connect it to the JevBench deployment. They also state that the public seed namespace differs from prior hosted builds and disclose Score variation under mixed load. Therefore neither historical hosted precision nor bitwise reproduction of its benchmark answers can be inferred from today's repository. [Performance caveats](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/docs/performance.md).

Today's hosted OpenAPI mentions a three-step `djev-pro`; the public capability endpoint returned `enabled: false` and `features.pro: false` during this investigation. The base `djev` remains one-step. The public extraction only accepts one step. This is another reason to separate live documentation, available source, and historical benchmark settings. [Hosted schema](https://api.djev.dev/openapi.json), [deployment capabilities](https://api.djev.dev/config).

## Experiments this evidence supports

These are proposed tests, not demonstrated improvements:

1. Keep local weights fixed and test separate system/user messages plus the empty-thought scaffold. Change each independently where practical.
2. Keep the winning prompt fixed and compare native yes/no and numeric Score labels with the local universal choice codes.
3. Compare compact/padded canvases and controlled answer-slot noise across several seeds. Matching the number `seed=0` alone does not match the initial token canvas across implementations.
4. Test explicit two/four-sample averaging separately, reporting actual read counts, p50, p95, and changed answers. Djev's published default is not evidence for adaptive rereads.
5. Only after prompt, labels, canvas, and input tokens are controlled, compare GGUF precisions within one runtime; then compare runtimes with identical weights/inputs where feasible. A hosted djev-versus-local score cannot isolate quantization.

Record full probability vectors and per-case outcomes. Keep the known public benchmark as a regression set and use an untouched evaluation split for choosing improvements; do not select prompt changes solely on the questions already inspected.
