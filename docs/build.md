# Build and hardware

[Back to README](../README.md)

Run these commands from the repository root. Install Rust 1.88+, a C/C++ compiler, CMake 3.24+, Make or Ninja, and libclang. Set `LIBCLANG_PATH` if bindgen cannot find libclang. Supply your own DiffusionGemma GGUF model.

## Native dependency

We pin the [llama.cpp](https://github.com/ggml-org/llama.cpp) submodule at `crates/llama-diffusion-sys/vendor/llama.cpp` to `12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4` from [DiffusionGemma PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423). Revisions without the DiffusionGemma C APIs cannot build this integration.

```bash
git submodule update --init --recursive
cargo build --workspace --locked
```

Cargo builds static llama.cpp and ggml libraries under `OUT_DIR` and generates bindings from the same headers. CMake uses Release mode for native code, including Rust debug builds. The gitlink pins the native source revision; `Cargo.lock` pins Rust dependencies.

## CPU

The default build uses CPU kernels without host-specific instruction sets, OpenMP, BLAS, or GPU SDK dependencies. Add `--features native` to optimize for the build machine; those binaries may require the same CPU features on other machines.

```bash
cargo run --release --locked -p llama-cpp-system-one -- \
  -m "$DIFFUSION_MODEL" --gpu-layers=0
```

The C++ runtime and platform libraries remain system dependencies. Cargo builds the bundled native code, so you do not need a separate llama.cpp installation.

## ROCm/HIP

Install the ROCm/HIP SDK and choose the deployment GPU's architecture. The example below uses `gfx1151`; replace it for other GPUs.

```bash
AMDGPU_TARGETS=gfx1151 cargo build -p llama-cpp-system-one --release --locked --features hip,native
```

For the development machine's ROCm/WSL setup:

```bash
export ROCM_PATH=/opt/rocm-7.2.1
export CMAKE_PREFIX_PATH="$ROCM_PATH"
export CMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++"
export AMDGPU_TARGETS=gfx1151
export GGML_HIP_NO_VMM=ON
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export DIFFUSION_MODEL="$HOME/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
cargo run --release --locked -p llama-cpp-system-one --features hip,native -- --bind 127.0.0.1:8080
```

Adjust the SDK path, GPU target, and model path to your machine. Keep the runtime environment set in the shell that starts the service.

If a build reports `HIP_ARCHITECTURES is empty for target "ggml-hip"`, export `AMDGPU_TARGETS=gfx1151` (or your GPU's architecture) and rerun the command. Earlier build scripts could delete CMake's saved architecture when the variable was unset. You do not need to clean the build directory. Subsequent builds keep the saved target; set `AMDGPU_TARGETS` again to change it. `CMAKE_HIP_ARCHITECTURES` takes precedence when both variables are set.

## CUDA

Install the CUDA toolkit and driver, then select the target GPU architecture:

```bash
CMAKE_CUDA_ARCHITECTURES=89 cargo build -p llama-cpp-system-one --release --locked --features cuda
```

The `hip`, `cuda`, and `native` features propagate through the server and structured inference crates to the native build. Select either HIP or CUDA; do not use `--all-features`.

Use `CMAKE_HIP_ARCHITECTURES`, `CMAKE_HIP_COMPILER`, `CMAKE_CUDA_COMPILER`, or `CMAKE_PREFIX_PATH` to select a toolchain. GPU builds link SDK libraries at runtime, so the loader must find them. A compiler or toolchain change may require `cargo clean -p llama-diffusion-sys` before rebuilding.

## Runtime settings

Both inference binaries accept these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--gpu-layers` | `-1` | Offload all layers; use `0` for CPU. |
| `--main-gpu` | `0` | Select the GPU. |
| `--context-size` | `4096` | Limit prompt plus canvas tokens. |
| `--batch-size` | `512` | Limit each prefill chunk and the full canvas. |
| `--seed` | `42` | Seed the initial answer-slot noise. |
| `--flash-attention` | Off | Enable with a supported native backend. |

We use one device for the prompt cache and disable tensor splitting. Flash attention defaults to off, matching the tested `-fa off` configuration. Validate CUDA and platforms beyond Linux with their SDKs.

## External native libraries

Set `LLAMA_CPP_DIR=/absolute/path/to/checkout` to build from another source checkout. Keep its DiffusionGemma APIs compatible with this integration.

To reuse a compiled shared library, set both `LLAMA_CPP_DIR` and `LLAMA_CPP_LIB_DIR` to matching headers and libraries. Cargo's backend features do not configure that external build. On Linux, add its library directory to `LD_LIBRARY_PATH`.

## Update the pin

For an intentional native dependency update:

```bash
git -C crates/llama-diffusion-sys/vendor/llama.cpp fetch origin pull/24423/head
git -C crates/llama-diffusion-sys/vendor/llama.cpp checkout --detach <reviewed-commit-sha>
# Run the regular checks and native reproducibility test before committing.
git add crates/llama-diffusion-sys/vendor/llama.cpp
```

Update the SHA in this guide and commit the gitlink with the parent project. Keep integration changes in `crates/llama-diffusion-sys/cmake`, outside the submodule. After provisioning submodules and Cargo dependencies, you can build with `--offline`; build scripts do not fetch source code.

## Image input

Supply a vision-projector GGUF compatible with DiffusionGemma's Gemma 4 26B-A4B vision encoder and the text model's embedding width. The text GGUF alone cannot process images. Projectors and model weights are external assets and are never downloaded automatically.

Set `DIFFUSION_MMPROJ=/path/to/mmproj.gguf` or pass `--mmproj /path/to/mmproj.gguf` to the server. The bundled build includes `mtmd` on CPU, HIP, and CUDA. The projector follows `--gpu-layers=0` for CPU execution; otherwise GPU use is enabled. Image encoding requests up to 280 patch tokens, with actual counts determined by the projector.

The CMake wrapper applies image-prefill embedding scaling and attention changes to a generated copy of the pinned native model source. No submodule files are changed. External source builds must match the overlay's source patterns. External shared-library mode also requires `libmtmd` beside `libllama`; image projector loading is disabled in that mode because Cargo cannot verify that its DiffusionGemma prefill has the integration changes.

Image decoder dependencies are Rust crates; WebP support does not require ffmpeg. See the [API examples](api.md#extensions) for payload formats and image limits.
