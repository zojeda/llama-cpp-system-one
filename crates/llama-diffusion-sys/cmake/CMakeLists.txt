cmake_minimum_required(VERSION 3.24)
project(llama_diffusion_native LANGUAGES C CXX)

# Keep upstream untouched. CMake resolves SDK paths, including nonstandard installs.
# SDK targets discovered in backend subdirectories must be visible to file(GENERATE).
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)
add_subdirectory("${LLAMA_SOURCE_DIR}" llama.cpp)

set(CARGO_LINK "")
foreach(LIB llama ggml ggml-cpu)
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
