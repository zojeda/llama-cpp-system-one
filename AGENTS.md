# Repository Guidelines

## Project Structure & Module Organization

This Rust 2024 workspace requires Rust 1.88 or newer. Source lives under `crates/`:

- `llama-diffusion-sys`: generated C bindings and native library linking.
- `llama-diffusion-structured`: inference orchestration, safe native ownership, and the SCM CLI.
- `system-one`: request validation, question compilation, and response mapping.
- `llama-cpp-system-one`: Axum routes, authentication, and the bounded inference worker.

Unit tests live in each crate’s source modules. `examples/system-one.json` provides a request fixture; `scripts/smoke-test.py` exercises a running service. GGUF models are external assets and are not downloaded automatically.

## Build, Test, and Development Commands

Initialize the pinned DiffusionGemma submodule with `git submodule update --init --recursive`. Cargo builds its static native libraries through CMake; a C/C++ compiler, CMake 3.24+, and libclang are required. CPU is the default; use `--features hip` or `--features cuda` for GPU builds and `native` for host CPU optimizations. Follow [docs/build.md](docs/build.md) for GPU SDK setup and optional external source/shared-library overrides. Do not edit the submodule for integration changes; use the CMake wrapper in `llama-diffusion-sys/cmake`.

- `cargo build --workspace --locked`: build all crates.
- `cargo fmt --all -- --check`: check Rust formatting.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: run lint checks.
- `cargo test --workspace --locked`: run regular tests without loading a model; native libraries are built by Cargo.
- `cargo run --locked -p llama-cpp-system-one -- -m "$DIFFUSION_MODEL" --bind 127.0.0.1:8080`: start the service.
- `python3 scripts/smoke-test.py`: validate the running service.

## Coding Style & Naming Conventions

Use rustfmt defaults, four-space indentation, `snake_case` functions/modules, `UpperCamelCase` types, and `SCREAMING_SNAKE_CASE` constants. Keep handwritten unsafe code in `llama-diffusion-structured/src/native.rs`, document safety invariants, and honor workspace unsafe-code lints. Keep model/context ownership on the dedicated worker thread and blocking inference off Tokio executor threads.

## Testing Guidelines

Use inline `#[cfg(test)]` modules with `#[test]` or `#[tokio::test]`. Name tests after observable behavior, such as `unsupported_extensions_are_never_silently_ignored`. Cover changed validation, probability math, HTTP errors, and queue behavior; no numeric coverage threshold is configured.

For native inference changes, set `DIFFUSION_MODEL` and run:

```bash
cargo test -p llama-diffusion-structured --locked \
  native_reads_preserve_reproducibility_across_requests -- --ignored --nocapture
```

Add `--features hip,native` on the local ROCm/WSL machine, with SDK/runtime variables from [docs/build.md](docs/build.md#rocmhip), or `--features cuda` for NVIDIA. Do not use `--all-features`; HIP and CUDA are alternative backends.

## Commit & Pull Request Guidelines

This checkout was initialized as a Git repository when the native submodule was added; no earlier commit convention was available. Use concise, imperative subjects identifying the affected crate or behavior. PRs should describe the change, link relevant issues, report validation commands/results, and identify native backend or model prerequisites. Update the README and example request when changing API behavior.

## Security & Configuration

Keep keys and model files out of commits. Configure bearer authentication with `TYPESAFE_API_KEY`; authentication is disabled when no key is supplied. Avoid logging request state or credentials.
