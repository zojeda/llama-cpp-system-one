# Development

[Back to README](../README.md)

## Workspace

| Crate | Responsibility |
| --- | --- |
| `llama-diffusion-sys` | Build the pinned native sources and generate C bindings. |
| `llama-diffusion-structured` | Own native resources, prepare tokens, evaluate the canvas, and provide the SCM CLI. |
| `system-one` | Validate requests, compile questions into slots, and map answers. |
| `llama-cpp-system-one` | Serve HTTP, check authentication, and manage the inference queue. |

We keep model/context ownership on a dedicated worker thread and blocking inference off Tokio executor threads. Native model and context types do not implement `Send` or `Sync`. We destroy contexts before models and release the backend after the last model.

Keep handwritten unsafe code in [`native.rs`](../crates/llama-diffusion-structured/src/native.rs) and document its safety invariants. The structured crate denies unsafe code outside that module. Each `lib.rs` declares modules and re-exports its public API; the server exposes `error` and `worker` modules as well.

| Crate | Modules |
| --- | --- |
| `llama-diffusion-structured` | `config`, `read`, `engine`, `denoise`, `images`, `probability`, `native`, `error` |
| `system-one` | `request`, `compiler`, `response`, `error` |
| `llama-cpp-system-one` | `http`, `handlers`, `middleware`, `worker`, `error` |

Keep protocol rules in `system-one`, HTTP policy in the server, and token/native operations in `llama-diffusion-structured`. Put unit tests beside the responsible code. Router tests exercise the HTTP contract without a model.

## Checks

Initialize the submodule and install the [build prerequisites](build.md) first. Regular checks use the CPU backend and do not load a model:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

For native inference changes, set `DIFFUSION_MODEL` and the [ROCm/WSL environment](build.md#rocmhip), then run:

```bash
cargo test -p llama-diffusion-structured --locked --features hip,native \
  native_reads_preserve_reproducibility_across_requests -- --ignored --nocapture
```

Use `--features cuda` for NVIDIA or omit GPU features for CPU. Do not use `--all-features`; HIP and CUDA are alternative backends. Against a running service, run `python3 scripts/smoke-test.py` with the server's `TYPESAFE_API_KEY` if configured.

Regular tests cover validation, probability math, error mapping, model aliases, request IDs, and queue behavior. The ignored native tests check reproducibility, extension behavior, and image prefill using real assets. Run `native_extensions_average_refine_think_and_chunk` with `DIFFUSION_MODEL`. Run `native_images_prefill_and_preserve_text_reproducibility` with both `DIFFUSION_MODEL` and `DIFFUSION_MMPROJ`. Use `-- --ignored --nocapture` and the appropriate backend features for each.

## Recipes

Install `just` and run `just` to list recipes:

```bash
just native-build        # Build the native sources.
just build --release     # Build the workspace.
just fmt                 # Format Rust source.
just check               # Type-check all targets.
just verify              # Run formatting, Clippy, and regular tests.
just test softmax        # Filter tests by name.
just doc                 # Build API documentation.
just serve               # Start the HTTP service.
just serve-release       # Build and start the release service.
just native-test         # Run the model-dependent test.
just smoke               # Check a running service.
```

Set `LLAMA_FEATURES=hip,native` or `LLAMA_FEATURES=cuda` to select a GPU backend for build, check, test, documentation, and run recipes. Preserve the runtime environment from the build guide. For example:

```bash
LLAMA_FEATURES=hip,native just serve-release --bind 127.0.0.1:8080
```

Set `DIFFUSION_MODEL` or pass `--model PATH` to an inference command. The `serve`, `serve-release`, and `scm` recipes forward extra arguments to their binaries.

## SCM CLI

The CLI classifies a material as a supplementary cementitious material (SCM), using `A = yes`, `B = no`, and the fixed canvas prefix `Is this material an SCM?\nAnswer: `.

```bash
cargo run --locked -p llama-diffusion-structured -- \
  -m "$DIFFUSION_MODEL" \
  -p "Ground granulated blast furnace slag is used in concrete." \
  --gpu-layers=-1 --seed 42 --json

# Equivalent recipe:
just scm "Ground granulated blast furnace slag is used in concrete." --json
```

Add the chosen GPU feature before `--` in the Cargo command. The output includes candidate token IDs, logits, probabilities, token counts, zero-based slot positions, initial noise, seed, and canvas forward time. See the [inference guide](inference.md) for their meaning.
