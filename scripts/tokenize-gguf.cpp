// Inspect the pinned native tokenizer without loading model weights or a context.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: tokenize-gguf MODEL.gguf TEXT [TEXT ...]\n");
        return 2;
    }
    auto params = llama_model_default_params();
    params.vocab_only = true;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) return 1;
    const auto * vocab = llama_model_get_vocab(model);
    std::printf("[");
    for (int i = 2; i < argc; ++i) {
        const int length = static_cast<int>(std::strlen(argv[i]));
        std::vector<llama_token> tokens(length * 2 + 32);
        int n = llama_tokenize(vocab, argv[i], length, tokens.data(), static_cast<int>(tokens.size()), false, false);
        if (n < 0) {
            tokens.resize(-n);
            n = llama_tokenize(vocab, argv[i], length, tokens.data(), static_cast<int>(tokens.size()), false, false);
        }
        if (n < 0) { llama_model_free(model); return 1; }
        std::printf(i == 2 ? "[" : ",[");
        for (int j = 0; j < n; ++j) std::printf(j == 0 ? "%d" : ",%d", tokens[j]);
        std::printf("]");
    }
    std::printf("]\n");
    llama_model_free(model);
}
