# JevBench public results

The default service configuration completed all **231 public cases** on 2026-09-21: **189/231 correct (81.8%)**, with **231/231 valid responses**. The [snapshot](results-2026-09-21-defaults.json) contains per-case predictions, probabilities, expected labels, latency, usage, and provenance hashes.

| Public tier | Correct | Valid responses | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Easy | 48/48 (100.0%) | 48/48 | 0.798 s | 0.903 s |
| Standard | 69/72 (95.8%) | 72/72 | 0.813 s | 0.906 s |
| Hard | 72/111 (64.9%) | 111/111 | 2.040 s | 10.619 s |
| All | **189/231 (81.8%)** | 231/231 | **0.912 s** | **8.124 s** |

Mean caller latency: 2.255 s. Calibration over valid distributions: ECE 0.0898; multiclass Brier 0.2687 (lower is better).

## Think extension rerun

With `think=1024` on the same binary and 8,192-token context, the service scored **189/231 (81.8%)**, with **231/231 valid responses**. Thinking fixed one baseline error and lost one previously correct answer, giving no net accuracy gain in this run.

| Public tier | think=0 | think=1024 | Thinking p50 | Thinking p95 |
| --- | ---: | ---: | ---: | ---: |
| Easy | 48/48 (100.0%) | 48/48 (100.0%) | 12.714 s | 45.085 s |
| Standard | 69/72 (95.8%) | 69/72 (95.8%) | 18.310 s | 33.876 s |
| Hard | 72/111 (64.9%) | 72/111 (64.9%) | 21.202 s | 37.765 s |
| All | **189/231 (81.8%)** | **189/231 (81.8%)** | **18.325 s** | **37.483 s** |

Thinking median/p95 latency was **18.325/37.483 seconds**, compared with **0.912/8.124 seconds** at `think=0`. Both runs used one excluded warmup; the thinking run used a 900-second timeout. See the [full report and reproduction command](think1024-defaults-2026-09-21.md) and [verified results snapshot](results-2026-09-21-defaults-think1024.json).

## Earlier runs

The [earlier baseline](baseline-2026-09-21.md) scored 165/231 (71.4%); the [historical `think=1024` run](think1024-2026-09-21.md) scored 184/231 (79.7%), with eight context-limit failures. Both used a 4,096-token context and no benchmark warmup. The current default run changes the zero-thought prompt framing, increases the context to 8,192, and includes one excluded warmup. The thinking configuration was subsequently rerun on the current build; see the comparison above.

Against the earlier baseline, the current `think=0` run fixed 31 answers and lost 7 previously correct answers; 193 kept the same correctness. That is 24 more correct answers, or +10.4 percentage points. This is not a controlled single-variable comparison. Historical snapshots remain unchanged.

## Published comparisons

We recomputed the reference figures for the same 231 public IDs from the [pinned upstream results](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/results/v1.2/jevbench-v1.2-per-task.json). We did not rerun these deployments.

| Model / configuration | Correct | Accuracy | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| Jev 1.13.0 (TypeSafe AI) | 200/231 | 86.6% | 0.665 s | 0.803 s |
| djev (Maisa, diffusion-gemma) | 194/231 | 84.0% | 0.239 s | 0.354 s |
| OpenJev (DiffusionGemma 26B-A4B NVFP4, razorback16) | 189/231 | 81.8% | 0.246 s | 0.459 s |
| SemIf, formerly OpenJev (Qwen3.5-4B, TheoLeeCJ) | 187/231 | 81.0% | 0.194 s | 0.538 s |
| **Local Q4_K_M, defaults** | **189/231** | **81.8%** | **0.912 s** | **8.124 s** |
| **Local Q4_K_M, `think=1024`** | **189/231** | **81.8%** | **18.325 s** | **37.483 s** |

## Method and scope

We used the upstream TypeSafe HTTP adapter and scorer at commit `fd51755eb0c0b546ca206d764faf3302feca913e`. Accuracy follows the scorer's argmax rule, including ordinal questions. HTTP and schema failures count as incorrect.

Requests ran serially in easy, standard, then hard order, after one excluded warmup with the tested thought budget. There were no retries; the adapter timeout was 120 seconds for `think=0` and 900 seconds for `think=1024`. p50/p95 use linear interpolation over all caller wall times, including failures. Model loading and the warmup are excluded. No other local inference or build workload was observed during either timed run. HTTP smoke checks ran after timing.

Reference timings use published values rounded to milliseconds, measured from Germany. Our measurements use local loopback. Hardware, network, quantization, and inference settings differ; these timings do not isolate model speed.

The official benchmark includes 534 decisions. This public subset includes 48 easy, 72 standard, and 111 hard cases; the other 303 decisions are unavailable in the public datasets. This fixed-seed run does not establish an official rank or unseen-task accuracy. Local compute cost is unpriced.

## Configuration

- AMD Ryzen AI MAX+ 395 / Radeon 8060S, WSL2, ROCm 7.2.1, `gfx1151`.
- HIP/native release build, eight CPU threads, all model layers offloaded, flash attention off.
- `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`; model SHA-256 `24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`.
- Default context 8,192, batch 512, seed 42; `steps=1`, `samples=1`, `think=0`, `sequential=false`.
- Empty, closed thought channel in prefill; no generated thought tokens.
- Pinned native sources: `12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4`.

The snapshot includes binary, model, Rust source, build-file, dataset, and scorer hashes to identify the measured implementation.

Run: `jevbench-defaults-2026-09-21`; timed requests started at `2026-09-21T20:14:44.105437+00:00` and finished at `2026-09-21T20:22:47.347031+00:00`.

## Reproduce

Configure the SDK and model paths using the [build guide](../../docs/build.md#rocmhip). Build the service, fetch the pinned harness, then run the local wrapper. It starts a service on port 8082 and stops that service when finished.

```bash
cargo build --release --locked -p llama-cpp-system-one --features hip,native
JEVBENCH_SOURCE=$(mktemp -d)
git clone https://github.com/fstandhartinger/jevbench.git "$JEVBENCH_SOURCE"
git -C "$JEVBENCH_SOURCE" checkout --detach fd51755eb0c0b546ca206d764faf3302feca913e

python3 -B scripts/jevbench-local.py \
  --binary "${CARGO_TARGET_DIR:-target}/release/llama-cpp-system-one" \
  --model "$DIFFUSION_MODEL" \
  --harness "$JEVBENCH_SOURCE" \
  --output "benchmarks/results/jevbench-$(date -u +%Y-%m-%dT%H-%M-%SZ)"
```

The wrapper uses Python's standard library and disables authentication only for its own loopback service. It writes the upstream summary, per-case records, raw evidence, service log, and source/model hashes to the selected output directory. Use a new directory for each run. Raw evidence stays under gitignored `benchmarks/results/`.

We verified all 231 task IDs and raw-response hashes and rescored each probability distribution with the pinned scorer. Formatting, workspace tests, Clippy, native inference tests, and HTTP smoke checks passed. Public case IDs and expected labels come from the [JevBench datasets](https://github.com/fstandhartinger/jevbench/tree/fd51755eb0c0b546ca206d764faf3302feca913e/datasets/public), under the [MIT license](LICENSE-jevbench).

Validation used the same HIP/native release configuration:

```bash
cargo fmt --all -- --check
cargo test --release --workspace --locked --features hip,native
cargo clippy --release --workspace --all-targets --locked --features hip,native -- -D warnings
cargo test --release -p llama-diffusion-structured --locked --features hip,native -- --ignored --nocapture --test-threads=1
```

The four ignored native tests require `DIFFUSION_MODEL`, and the image test also requires `DIFFUSION_MMPROJ`. The local benchmark wrapper ran `scripts/smoke-test.py` after the timed cases and then stopped its service.
