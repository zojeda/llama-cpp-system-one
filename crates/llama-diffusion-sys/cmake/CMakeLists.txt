cmake_minimum_required(VERSION 3.24)
project(llama_diffusion_native LANGUAGES C CXX)

# Keep upstream untouched. CMake resolves SDK paths, including nonstandard installs.
# SDK targets discovered in backend subdirectories must be visible to file(GENERATE).
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
add_subdirectory("${LLAMA_SOURCE_DIR}" llama.cpp)

# Build an overlay of the pinned model implementation without changing the submodule.
# Image embeddings already have their scale applied by the projector. Each image
# is one prefill batch, allowing bidirectional attention within its patch block.
get_target_property(LLAMA_SOURCES llama SOURCES)
set(DIFFUSION_SOURCE "${LLAMA_SOURCE_DIR}/src/models/diffusion-gemma.cpp")
file(READ "${DIFFUSION_SOURCE}" DIFFUSION_CODE)
set(OLD_SCALE "ggml_scale(ctx0, inpL, sqrtf((float) n_embd))")
set(OLD_MASK "for (int64_t k = 0; k <= q; ++k) {     // causal: keys [0, q]")
foreach(EXPECTED "${OLD_SCALE}" "${OLD_MASK}")
    string(FIND "${DIFFUSION_CODE}" "${EXPECTED}" MATCH_POSITION)
    if(MATCH_POSITION EQUAL -1)
        message(FATAL_ERROR "DiffusionGemma image overlay needs the pinned source revision")
    endif()
endforeach()
string(REPLACE "${OLD_SCALE}" "ggml_scale(ctx0, inpL, ubatch.token ? sqrtf((float) n_embd) : 1.0f)" DIFFUSION_CODE "${DIFFUSION_CODE}")
string(REPLACE "void set_input(const llama_ubatch * /*ubatch*/) override" "void set_input(const llama_ubatch * ubatch) override" DIFFUSION_CODE "${DIFFUSION_CODE}")
string(REPLACE "${OLD_MASK}" "for (int64_t k = 0; k < (ubatch->embd ? n_kv : q + 1); ++k) {" DIFFUSION_CODE "${DIFFUSION_CODE}")
# SWA still limits previous text, but never splits the current image block.
string(REPLACE "if (swa && llama_hparams::is_masked_swa(hparams.n_swa, hparams.swa_type, k, q))" "if (swa && !(ubatch->embd && k >= off) && llama_hparams::is_masked_swa(hparams.n_swa, hparams.swa_type, k, q))" DIFFUSION_CODE "${DIFFUSION_CODE}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/diffusion-gemma.cpp" "${DIFFUSION_CODE}")
list(REMOVE_ITEM LLAMA_SOURCES "${DIFFUSION_SOURCE}" "models/diffusion-gemma.cpp")
set_property(TARGET llama PROPERTY SOURCES "${LLAMA_SOURCES}")
target_sources(llama PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/diffusion-gemma.cpp")
target_include_directories(llama PRIVATE "${LLAMA_SOURCE_DIR}/src/models")

set(CARGO_LINK "")
foreach(LIB mtmd llama ggml ggml-cpu)
    string(APPEND CARGO_LINK "cargo::rustc-link-search=native=$<TARGET_FILE_DIR:${LIB}>\ncargo::rustc-link-lib=static=${LIB}\n")
endforeach()

if(GGML_HIP)
    string(APPEND CARGO_LINK "cargo::rustc-link-search=native=$<TARGET_FILE_DIR:ggml-hip>\ncargo::rustc-link-lib=static=ggml-hip\n")
    string(APPEND CARGO_LINK "cargo::rustc-link-search=native=$<TARGET_LINKER_FILE_DIR:hip::amdhip64>\ncargo::rustc-link-lib=dylib=amdhip64\n")
    foreach(LIB hipblas rocblas)
        string(APPEND CARGO_LINK "cargo::rustc-link-search=native=$<TARGET_LINKER_FILE_DIR:roc::${LIB}>\ncargo::rustc-link-lib=dylib=${LIB}\n")
    endforeach()
endif()

if(GGML_CUDA)
    string(APPEND CARGO_LINK "cargo::rustc-link-search=native=$<TARGET_FILE_DIR:ggml-cuda>\ncargo::rustc-link-lib=static=ggml-cuda\n")
    foreach(LIB cudart cublas)
        string(APPEND CARGO_LINK "cargo::rustc-link-search=native=$<TARGET_LINKER_FILE_DIR:CUDA::${LIB}>\ncargo::rustc-link-lib=dylib=${LIB}\n")
    endforeach()
    if(WIN32)
        set(CUDA_DRIVER_NAME nvcuda)
    else()
        set(CUDA_DRIVER_NAME cuda)
    endif()
    string(APPEND CARGO_LINK "cargo::rustc-link-search=native=$<TARGET_LINKER_FILE_DIR:CUDA::cuda_driver>\ncargo::rustc-link-lib=dylib=${CUDA_DRIVER_NAME}\n")
endif()

string(APPEND CARGO_LINK "cargo::rustc-link-search=native=$<TARGET_FILE_DIR:ggml-base>\ncargo::rustc-link-lib=static=ggml-base\n")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/cargo-link.txt" CONTENT "${CARGO_LINK}")
