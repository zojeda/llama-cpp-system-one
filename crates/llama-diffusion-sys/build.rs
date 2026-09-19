use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_LIB_DIR");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let external_source = env::var_os("LLAMA_CPP_DIR");
    let external_libraries = env::var_os("LLAMA_CPP_LIB_DIR");
    assert!(
        external_libraries.is_none() || external_source.is_some(),
        "LLAMA_CPP_LIB_DIR requires LLAMA_CPP_DIR pointing to matching headers"
    );
    let source = external_source
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("vendor/llama.cpp"))
        .canonicalize()
        .expect("Run git submodule update --init --recursive, or set LLAMA_CPP_DIR");
    assert!(
        source.join("include/llama.h").is_file(),
        "Missing llama.cpp sources; run git submodule update --init --recursive"
    );
    let header = fs::read_to_string(source.join("include/llama.h"))
        .expect("Cannot read the llama.cpp header");
    for symbol in [
        "llama_diffusion_set_phase",
        "llama_diffusion_set_sc",
        "llama_vocab_mask",
    ] {
        assert!(
            header.contains(symbol),
            "Incompatible llama.cpp: missing {symbol}; use the pinned DiffusionGemma revision"
        );
    }

    if let Some(libraries) = external_libraries {
        let libraries = PathBuf::from(libraries)
            .canonicalize()
            .expect("LLAMA_CPP_LIB_DIR must point to the matching shared-library directory");
        println!("cargo::rerun-if-changed={}", libraries.display());
        println!("cargo::rustc-link-search=native={}", libraries.display());
        println!("cargo::rustc-link-lib=dylib=llama");
        println!(
            "cargo::warning=Using prebuilt llama.cpp; Cargo backend features do not configure this external library"
        );
    } else {
        build_native(&manifest, &source);
    }

    let bindings = bindgen::Builder::default()
        .header(source.join("include/llama.h").to_string_lossy())
        .clang_arg(format!("-I{}", source.join("ggml/include").display()))
        .allowlist_function("llama_(backend_init|backend_free|model_default_params|context_default_params|model_load_from_file|model_free|model_get_vocab|model_is_diffusion|init_from_model|free|n_ctx|n_batch|n_ubatch|set_causal_attn|diffusion_set_sc|diffusion_set_phase|diffusion_pkv_bytes_per_token|batch_init|batch_free|decode|get_logits|synchronize|tokenize|token_to_piece|vocab_n_tokens|vocab_mask|vocab_is_control)")
        .allowlist_var("LLAMA_.*")
        .derive_debug(false)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Cannot generate llama.cpp bindings; install libclang and use the matching headers");
    bindings
        .write_to_file(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("Cannot write llama.cpp bindings");
}

fn build_native(manifest: &Path, source: &Path) {
    let hip = cfg!(feature = "hip");
    let cuda = cfg!(feature = "cuda");
    assert!(!(hip && cuda), "Select either hip or cuda, not both");
    println!("cargo::rerun-if-changed=cmake/CMakeLists.txt");
    // Watch sources, including additions, without watching an external checkout's build outputs.
    for path in [
        "CMakeLists.txt",
        "cmake",
        "src",
        "include",
        "ggml",
        "vendor",
    ] {
        println!("cargo::rerun-if-changed={}", source.join(path).display());
    }
    let mut build = cmake::Config::new(manifest.join("cmake"));
    build
        .define("LLAMA_SOURCE_DIR", source)
        .profile("Release")
        .build_target("llama")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        // GGML_STATIC also requests static GPU SDKs, which HIP does not support.
        .define("GGML_STATIC", "OFF")
        .define("GGML_BACKEND_DL", "OFF")
        .define(
            "GGML_NATIVE",
            if cfg!(feature = "native") {
                "ON"
            } else {
                "OFF"
            },
        )
        .define("GGML_CPU", "ON")
        .define("GGML_HIP", if hip { "ON" } else { "OFF" })
        .define("GGML_CUDA", if cuda { "ON" } else { "OFF" });
    for option in [
        "LLAMA_BUILD_COMMON",
        "LLAMA_BUILD_TESTS",
        "LLAMA_BUILD_TOOLS",
        "LLAMA_BUILD_EXAMPLES",
        "LLAMA_BUILD_SERVER",
        "LLAMA_BUILD_APP",
        "LLAMA_BUILD_MTMD",
        "LLAMA_OPENSSL",
        "GGML_OPENMP",
        "GGML_OPENMP_FETCH",
        "GGML_BLAS",
        "GGML_METAL",
        "GGML_ACCELERATE",
        "GGML_CUDA_NCCL",
        "GGML_HIP_RCCL",
        "GGML_CPU_KLEIDIAI",
        // Upstream otherwise enables these even with GGML_NATIVE=OFF.
        "GGML_SSE42",
        "GGML_AVX",
        "GGML_AVX2",
        "GGML_BMI2",
        "GGML_FMA",
        "GGML_F16C",
    ] {
        build.define(option, "OFF");
    }
    for variable in [
        "AMDGPU_TARGETS",
        "CMAKE_HIP_ARCHITECTURES",
        "CMAKE_HIP_COMPILER",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_COMPILER",
    ] {
        println!("cargo::rerun-if-env-changed={variable}");
        if let Some(value) = env::var_os(variable) {
            build.define(variable, value);
        } else if variable == "AMDGPU_TARGETS" || variable.ends_with("_ARCHITECTURES") {
            // Do not retain an architecture selected by a previous environment.
            build.configure_arg(format!("-U{variable}"));
        }
    }
    println!("cargo::rerun-if-env-changed=GGML_HIP_NO_VMM");
    build.define(
        "GGML_HIP_NO_VMM",
        env::var_os("GGML_HIP_NO_VMM").unwrap_or_else(|| "ON".into()),
    );
    for variable in [
        "ROCM_PATH",
        "HIP_PATH",
        "HIPCXX",
        "CUDACXX",
        "CUDA_PATH",
        "CUDAToolkit_ROOT",
    ] {
        println!("cargo::rerun-if-env-changed={variable}");
    }
    let destination = build.build();
    let link = fs::read_to_string(destination.join("build/cargo-link.txt"))
        .expect("CMake did not generate native link metadata");
    print!("{link}");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap();
    if target_env != "msvc" {
        let cxx = if matches!(target_os.as_str(), "macos" | "ios" | "freebsd" | "openbsd") {
            "c++"
        } else {
            "stdc++"
        };
        println!("cargo::rustc-link-lib=dylib={cxx}");
        if target_os != "windows" {
            println!("cargo::rustc-link-lib=m");
            println!("cargo::rustc-link-lib=pthread");
        }
        if matches!(target_os.as_str(), "linux" | "android") {
            println!("cargo::rustc-link-lib=dl");
        }
    }
}
