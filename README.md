# llama-cpp-system-one

A Rust implementation of the System One API for structured question answering with DiffusionGemma and llama.cpp.
> OpenJev - Open, Jev Compatible

This independent learning project builds on the work at [codiv.ai](https://codiv.ai/). Credit goes to Codiv for the System One approach and API design, Google DeepMind for [DiffusionGemma](https://ai.google.dev/gemma/docs/diffusiongemma/model_card), and [llama.cpp](https://github.com/ggml-org/llama.cpp) contributors for native inference support. Rust handles request validation, prompt construction, and response mapping.

Run structured questions with DiffusionGemma through llama.cpp on AMD ROCm/HIP hardware. Send a state and questions; get yes/no probabilities, choices, or rubric scores through the System One API. Image questions also require a compatible vision projector.

## The masked canvas

A canvas is a block of token positions. For this API, we fix the question labels and leave one answer position per question:

```text
Prompt: state + questions + answer codes (A = yes, B = no, ...)

Canvas: Question 1       Question 2
        Answer: [ ? ]   Answer: [ ? ]
                 ^               ^
             answer slot     answer slot
```

The brackets mark unknown answers. We fill those positions with seeded random vocabulary tokens, excluding the special mask token. After caching the prompt, we evaluate the whole canvas once. Bidirectional attention lets each slot use the surrounding canvas and prompt context. We read the logits at each slot and normalize them over its allowed answer codes.

With `think=0`, the cached prompt ends with an empty `<|channel>thought\n<channel|>` channel before the answer canvas. This framing generates no thought tokens and keeps the default at one decoder read. Requests with `think > 0` generate their own bounded thought instead.

DiffusionGemma's full text generator refines a noisy canvas over multiple denoising steps. By default, this service takes one read with fixed surrounding text and returns the answer distributions. Optional extensions add denoising steps, noise samples, a bounded thought, sequential question chunks, and images. See [diffusion and canvas inference](docs/inference.md) for the model explanation, a worked example, and the limits of these probabilities.

## Run

You need Rust 1.88+, a C/C++ compiler, CMake 3.24+, Make or Ninja, libclang, and a DiffusionGemma GGUF model. Obtain the model yourself; Cargo builds the pinned llama.cpp sources. For AMD GPUs, install ROCm/HIP and follow the [build guide](docs/build.md#rocmhip).

```bash
git submodule update --init --recursive
export DIFFUSION_MODEL="$HOME/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
export DIFFUSION_MMPROJ="$HOME/models/diffusiongemma/mmproj-diffusiongemma-26b-a4b-f16.gguf"

# CPU
cargo run --release --locked -p llama-cpp-system-one -- --bind 127.0.0.1:8080

# AMD GPU: after configuring ROCm/HIP
export AMDGPU_TARGETS=gfx1151
cargo run --release --locked -p llama-cpp-system-one --features hip,native -- --bind 127.0.0.1:8080

```

For images, start the service with a compatible projector (see [image setup](docs/build.md#image-input)):

```bash
cargo run --release --locked -p llama-cpp-system-one --features hip,native -- -m "$DIFFUSION_MODEL" --mmproj "$DIFFUSION_MMPROJ"
```

Choose one server command. In another terminal:

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  --data-binary @examples/system-one.json
```

The example asks three questions about a construction material using the default options, including `think=0`. The default context is 8,192 tokens; change it with `--context-size`. Set `TYPESAFE_API_KEY` on the server to enable bearer authentication, then add `-H "Authorization: Bearer $TYPESAFE_API_KEY"` to client calls. The default listener is `127.0.0.1:8080`.

Try the [hot dog photo example](docs/api.md#hot-dog-photo), including the [bundled JPEG](examples/hotdog.jpg), [ready-to-send request](examples/hotdog.json), and startup instructions for the vision projector.

Use [JavaScript SDK examples](examples/javascript/README.md) for application code. The [API reference](docs/api.md) covers request types, model aliases, and errors. Use the [extensions](docs/api.md#extensions) for `steps`, `samples`, `think`, `sequential`, and `images`. Text defaults remain `steps=1`, `samples=1`, and `think=0`; image requests require `--mmproj` or `DIFFUSION_MMPROJ`.

## Benchmarks

### JevBench public cases

The updated implementation scores **189/231 (81.8%)**, up from **165/231 (71.4%)**, using the same Q4_K_M weights and no generated thinking. Empty-channel prefill fixes 31 answers and regresses seven. It matches published OpenJev's total on these public cases and is five answers behind djev. This is not the full official benchmark or an unseen holdout.

Matched local HTTP measurements use context 8,192, seed 42, one request at a time, one excluded warmup, a 120-second timeout, and no retries:

| Implementation | Correct | Valid | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Original, `think=0` | 165/231 (71.4%) | 231/231 | 0.847 s | 8.589 s |
| Updated, `think=0` | **189/231 (81.8%)** | 231/231 | **1.009 s** | **9.381 s** |

The accuracy gain costs 19.1% higher p50 and 9.2% higher p95 in this serial comparison; it is not a speedup. All-attempt and valid-response percentiles are identical because no requests failed. Every HTTP probability vector exactly matches its corresponding research run. Updated accuracy is 48/48 easy, 69/72 standard, and 72/111 hard.

Full-set research confirmation also improved seed 123 from 163 to 185 correct and seed 2026 from 181 to 189. The controlled Q8-to-Q4 pilot found no consistent additional accuracy penalty on its 32 cases; total loss relative to BF16 remains unresolved. Prompt framing explains part of the gap without changing weights. The default context is now 8,192; all eight historical thinking/context failures return valid responses at that size. See the [experiment log](docs/research/experiments.md), [complete HTTP results](docs/research/http-results.json), and [test evidence](docs/research/validation-results.json).

The historical measurements below use the original implementation. On 2026-09-21, we ran all **231 public JevBench cases** against DiffusionGemma Q4_K_M on an AMD Ryzen AI MAX+ 395 / Radeon 8060S, using a HIP/native release build. Both runs used one request at a time, with no benchmark warmups or retries.

| Local configuration | Easy (48) | Standard (72) | Hard (111) | Overall (231) | Valid responses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `think=0` | 91.7% | 87.5% | 52.3% | **165/231 (71.4%)** | 231/231 |
| `think=1024` | 100.0% | 95.8% | 60.4% | **184/231 (79.7%)** | 223/231 |

Thinking fixed 28 baseline errors and lost 9 previously correct answers: **19 more correct answers, or +8.2 percentage points**. Eight long requests exceeded the unchanged 4,096-token context after reserving the thought budget; those count as incorrect.

The comparison below uses the same 231 public case IDs for accuracy and latency. Local rows identify the updated run and historical original runs; their timing protocols differ as described above. Published reference models were not rerun here; their figures come from [Benchmark Heaven's pinned per-case results](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/results/v1.2/jevbench-v1.2-per-task.json).

| Model / configuration | Correct | Accuracy | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Jev 1.13.0 (TypeSafe AI) | 200/231 | 86.6% | 0.665 s | 0.803 s |
| djev (Maisa, DiffusionGemma) | 194/231 | 84.0% | 0.239 s | 0.354 s |
| OpenJev (DiffusionGemma NVFP4, razorback16) | 189/231 | 81.8% | 0.246 s | 0.459 s |
| **Updated local Q4_K_M, `think=0`** | **189/231** | **81.8%** | **1.009 s** | **9.381 s** |
| SemIf (Qwen3.5-4B, TheoLeeCJ) | 187/231 | 81.0% | 0.194 s | 0.538 s |
| **Original local Q4_K_M, `think=1024`** | **184/231** | **79.7%** | **17.204 s** | **35.569 s** |
| **Original local Q4_K_M, `think=0`** | **165/231** | **71.4%** | **0.914 s** | **7.417 s** |

Latencies are raw caller wall times over all attempts, including failures, with no production-load adjustment. p50 is the median; p95 is the 95th percentile. Reference percentiles use linear interpolation over published per-case timings rounded to milliseconds. Local requests used loopback; Benchmark Heaven measured its deployments from Germany. Hardware, network paths, quantization, and inference settings differ, so these timings do not isolate model speed. Excluding the eight rejected requests, the thinking run's p50/p95 were **17.688/36.182 seconds**.

This subset excludes 303 decisions from the full benchmark and does not establish an official leaderboard score. See the [benchmark report and reproduction commands](benchmarks/jevbench/README.md), [thinking comparison](benchmarks/jevbench/think1024-2026-09-21.md), and data snapshots for [think=0](benchmarks/jevbench/results-2026-09-21.json) and [think=1024](benchmarks/jevbench/results-2026-09-21-think1024.json).

### System One comparison corpus

On 2026-09-19, we compared 72 requests and 84 questions per endpoint in one round, with concurrency one and no benchmark warmups or retries.

| Metric | Local `gemmadiffusion-0.1` | Hosted TypeSafe `jev-1.13.0` |
| --- | --- | --- |
| Question accuracy | 65/84 (77.4%) | 80/84 (95.2%) |
| Valid responses | 72/72 | 71/72 |
| p50 latency, valid responses | 588.4 ms | 339.9 ms |
| p95 latency, valid responses | 1,314.6 ms | 881.0 ms |
| Mean latency, all attempts | 683.6 ms | 838.4 ms |

The hosted endpoint timed out once at 30 seconds. We count that answer as incorrect and include the timeout in the all-attempt mean. Timings include network travel and SDK parsing. The saved run lacks hardware and quantization details; treat it as an exploratory result on this synthetic corpus. See [results and methodology](benchmarks/system-one/README.md#recorded-results-2026-09-19) and the [data snapshot](benchmarks/system-one/results-2026-09-19.json).

## Documentation

| Guide | Contents |
| --- | --- |
| [Build and hardware](docs/build.md) | ROCm/WSL, CUDA, native library overrides |
| [Diffusion and canvas inference](docs/inference.md) | Denoising, answer slots, probability math |
| [HTTP API](docs/api.md) | Question types, authentication, limits, errors |
| [Development](docs/development.md) | Crate layout, tests, recipes, SCM CLI |
| [JevBench](benchmarks/jevbench/README.md) | Public benchmark results, published comparisons, reproduction commands |
| [System One comparison](benchmarks/system-one/README.md) | Synthetic corpus, local/hosted results, scoring, runner settings |
