// Research utility: measure additional Q8 -> Q4_K_M loss with a shared source.
// Link against the exact native libraries built by this workspace.
#include "llama.h"
#include <cstdio>
#include <filesystem>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: requantize-gguf INPUT_Q8.gguf NEW_OUTPUT_Q4.gguf\n");
        return 2;
    }
    if (!std::filesystem::is_regular_file(argv[1]) || std::filesystem::exists(argv[2])) {
        std::fprintf(stderr, "Require an existing input and a new output path\n");
        return 2;
    }
    auto params = llama_model_quantize_default_params();
    params.ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    params.allow_requantize = true;
    params.nthread = 8;
    params.max_buf_size = 512ULL * 1024 * 1024;
    return llama_model_quantize(argv[1], argv[2], &params) == 0 ? 0 : 1;
}
