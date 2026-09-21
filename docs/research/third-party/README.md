# Reference compiler attribution

The text-only reference compilers in `scripts/jev-experiments.py` adapt prompt
construction from these Apache-2.0 projects:

- [OpenJev](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/openjev/engine.py), release 0.2.0. Its engine credits the vLLM structured-read example and contributors.
- [djev dev](https://github.com/Davipar/djev-dev/blob/3ce907e6835212f27ee82b4cee9039198c4abe35/djev/engine.py), copyright 2026 Djev contributors. See the retained [NOTICE](djev-NOTICE.txt).

Both use the [Apache License 2.0](Apache-2.0.txt). The adaptations remove hosted
transport and schema caches, limit reference choice codes to those needed by
the public tasks, and preserve our yes/no probability ordering. They run through
the local Rust engine, with local noise and token preparation, and are not
claimed to reproduce either hosted implementation exactly. Weights are not
redistributed.
