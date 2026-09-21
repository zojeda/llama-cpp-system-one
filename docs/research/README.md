# Jev inference parity research

Completed on `research/jev-inference-parity`, starting from `d023974`. All planned experiment groups and final validation are complete. Empty-header prefill is now the implementation default, with context 8,192. Matched HTTP accuracy improved from 165/231 to 189/231; all responses are valid. The same framing also improved full-set scores at seeds 123 and 2026.

Execution is now tracked in the [experiment log](experiments.md), with [machine-readable results](experiment-results.json).

## Conclusion

**The published accuracy gap cannot be attributed to quantization alone.** Both reference projects differ from our prompt compiler and answer canvas. OpenJev also performs automatic rereads. With unchanged Q4 weights, empty-header prefill improved full-set scores from 165/163/181 to 189/185/189 across three fixed seeds. A controlled Q8-to-Q4 pilot found no consistent accuracy penalty on its 32 cases; this does not measure total loss from BF16 or establish precision equivalence.

The completed experiments support an empty thought scaffold with the existing weights and no generated thinking. Additional prompt, label, padding, reread, refinement, precision, and flash-attention comparisons did not establish a stronger production default. Proper generated thinking reached 30/32 on the development sample, but its p95 exceeded five minutes; it remains experimental.

Detailed, pinned primary-source investigations: [djev](djev.md) and [OpenJev](openjev.md). The reference is `razorback16/openjev`, not another similarly named project.

## Final implementation verification

| Matched local HTTP run | Correct / 231 | p50 | p95 |
| --- | ---: | ---: | ---: |
| Original | 165 (71.4%) | 0.847 s | 8.589 s |
| Updated | 189 (81.8%) | 1.009 s | 9.381 s |

Both runs use identical server arguments, the same model, one excluded warmup, and the pinned serial HTTP protocol. Each returned 231 valid responses and exactly reproduced the corresponding research probabilities. The updated run fixes 31 cases and regresses seven; p50 increased 19.1% and p95 9.2%. Formatting, Clippy, 22 regular tests, four model tests, and both HTTP smoke checks passed. [Full HTTP evidence](http-results.json), [validation](validation-results.json), and [experiment coverage](experiments.md#experiment-coverage).

## Historical evidence

All accuracy figures below use the same **231 public cases**, not the complete official benchmark. Reference answers are published results, not reruns performed here.

| Configuration | Correct | Valid | Local p50 | Local p95 |
| --- | ---: | ---: | ---: | ---: |
| Local Q4_K_M, think=0 | 165/231 (71.4%) | 231/231 | 0.914 s | 7.417 s |
| Local Q4_K_M, think=1024 | 184/231 (79.7%) | 223/231 | 17.204 s | 35.569 s |
| Published djev | 194/231 (84.0%) | — | — | — |
| Published OpenJev | 189/231 (81.8%) | — | — | — |

Local latency includes all attempts. Thinking's eight context rejections count as incorrect; its valid-response p50/p95 are 17.688/36.182 seconds. Published speed tests used other hardware, network paths, and task populations, so they are not inserted into this table as matched latency measurements. Sources: [baseline and reference comparison](../../benchmarks/jevbench/README.md), [thinking results](../../benchmarks/jevbench/think1024-2026-09-21.md).

Reanalysis of the saved results provides useful clues:

- Thinking fixed 28 baseline errors and lost nine correct answers, including three lost to context rejection. **26 of those 28 fixes reported only four output tokens.** Five requests used all 1,024 tokens; none changed correctness. Token counts do not reveal thought text or prove an empty thought, but they make framing a plausible explanation worth isolating.
- Against djev, there are 39 reference-only correct cases and ten local-only correct cases. Against OpenJev, those counts are 38 and 14. An improvement must account for regressions as well as recovered answers.
- Both references score 15/18 on `multi_hop`, versus our 7/18 baseline; on `temporal_numeric`, our baseline scores 5/15 versus djev's 4/15 and OpenJev's 2/15. The gap varies by task family.

See [derived counts and model metadata](baseline-analysis.json). Local token counts come from the tracked [baseline](../../benchmarks/jevbench/results-2026-09-21.json) and [thinking](../../benchmarks/jevbench/results-2026-09-21-think1024.json) snapshots. Reference disagreements were joined by task ID against the published per-task results at JevBench revision `fd51755eb0c0b546ca206d764faf3302feca913e`, retained in the original run's ignored `published-per-task.json` artifact.

## Differences to isolate

| Factor | Local saved baseline | OpenJev 0.2.0 source | Current public djev source |
| --- | --- | --- | --- |
| Message roles | Everything in one user turn | Questions in system; state in user | Questions in system; state in user |
| Empty thought scaffold at think=0 | Absent | Inside answer canvas | Inside answer canvas |
| Noul / Score label tokens | Alphabetic codes | yes/no / ordinal digits | no/yes / ordinal digits |
| Answer line | `Question 1\nAnswer: A` | `q1: A` | Compact numbered form such as `0:A` |
| Canvas suffix | No explicit turn-close/padding | Turn-close; PAD to multiple of 16 | Turn-close; PAD to multiple of 16 |
| Default reads | One | One, then three more when uncertain | One; no adaptive rereads |
| Slot noise | ChaCha8, fixed seed 42 | Python RNG, content-derived seed | Python RNG, versioned seed namespace |
| Weights | Mixed GGUF Q4_K_M | NVFP4 | BF16; historical hosted precision unknown |

Sources and limitations are detailed in the linked project investigations. The current djev extraction does not establish the exact historical hosted prompt or precision. OpenJev's benchmark identifies image tag 0.2.0, but not an immutable image digest or model revision.

Our GGUF is not uniformly four-bit: its 692 tensor descriptors contain 423 F32, 33 Q5_0, 194 Q4_K, 14 Q6_K, and 28 Q8_0 tensors; embeddings use Q6_K. These are tensor counts, not fractions of parameters. Its name metadata is `Dg_Rc0P1_Patched`; the header does not establish an immutable source checkpoint. OpenJev's current NVFP4 metadata describes a different precision allocation, including exclusions for attention, routing, and self-conditioning. Comparing these two format names does not measure quantization loss.

Local sources inspected: [compiler](../../crates/system-one/src/compiler.rs), [engine](../../crates/llama-diffusion-structured/src/engine.rs), [native wrapper](../../crates/llama-diffusion-structured/src/native.rs), and native submodule `12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4`.

## Experiment sequence

The table records the original experiment plan; see the [experiment log](experiments.md) for completed runs and results. Preserve a baseline arm and change one factor per comparison; do not combine several changes and assign the gain to one of them.

| ID | Controlled comparison | Question answered |
| --- | --- | --- |
| E0 | Reproduce think=0 at context 4,096; then 8,192 with all other settings fixed. Use 8,192 for subsequent paired arms if stable. | Establish repeatability and enough headroom for the eight previously rejected thinking requests. |
| E1 | Existing answer canvas versus an empty `<\|channel>thought\n<channel\|>` prefix, with think=0. Separately place the same scaffold in encoder prefill. | Does thought framing help without generating reasoning? Does bidirectional canvas placement matter? |
| E2 | Move question instructions to a system turn and state to a user turn; preserve wording. Then separately test reference wording and shorter numbered answer lines. | Does prompt organization explain part of the gap? |
| E3 | Replace Noul codes with yes/no and Score codes with numeric levels. Verify each label is one token in its full answer-template context. | Do semantic answer tokens help? Stratify by question type. |
| E4 | Add turn-close, then separately PAD to a multiple of 16. Keep noise restricted to answer slots. | Does fixed surrounding canvas text affect bidirectional inference? |
| E5 | One versus four fixed reads, averaging candidate probabilities. Then evaluate OpenJev's exact adaptive policy separately. | Is averaging beneficial, and how much latency does the gate save? |
| E6 | With a frozen compiler/canvas, compare higher-precision GGUF and Q4_K_M from the same checkpoint and conversion lineage. Cross baseline/reference-style formatting with both precisions. | Separate format effects, precision effects, and their interaction. |
| E7 | After one-step parity checks, test two/four denoising steps and generated thinking independently. | Do further decoder work or reasoning add value beyond framing and averaging? |
| E8 | Profile the accepted configurations before optimizing native execution. | Where can p50/p95 improve while preserving accuracy? |

For E5, OpenJev's threshold is **0.1 on an unnormalized partial entropy of full-vocabulary probabilities** for top-20 tokens plus candidate IDs. It is not entropy of the candidate-normalized distribution. Its automatic path has four reads total, not four additional reads. djev's default does not use this policy.

Before E6, capture exact prompt token IDs, initial canvas IDs, slot positions, candidate IDs, and selected logits in explicit research artifacts. A shared integer seed does not produce shared noise across Python and ChaCha8. Test a frozen input canvas when comparing backends. Record checkpoint, tokenizer, conversion, model-file hash, runtime commit, KV type, and precision. If matched source weights are unavailable or do not fit, label precision attribution unresolved rather than substituting a hosted score comparison.

Generated thinking also needs its own template check: OpenJev enables the tokenizer's thinking template; our engine opens the thought channel directly. The current model template includes a system `<|think|>` marker. Verify the intended tokenizer revision before treating those paths as equivalent.

## Measurement and acceptance

- Freeze a small development set before changing code, stratified by tier and question type without selecting on which system won. Use fixed seeds 42, 123, and 2026 with the same initial slot draws within each paired comparison. Reference-specific seed/RNG parity is a separate experiment.
- Use the public 231-case set for regression reporting. We have already inspected its aggregate outcomes, so it is not an untouched holdout. Obtain a separate unseen set before claiming general improvement.
- For each arm, retain full probability vectors, correct/total, valid/total, per-tier/type/family scores, fixes, regressions, and paired accuracy differences. Include all failures in the main denominator and report matched-valid accuracy separately.
- Report end-to-end **p50 and p95** for all attempts and valid responses separately. Use the same percentile definition, timeout, serial scheduling, warm-up, hardware, and cache policy. Record actual read/step counts, prompt/canvas/thought tokens, and context errors. Do not run competing GPU workloads during measurement.
- Recheck promising arms on all 231 cases and across the fixed seeds. Report uncertainty on paired gains; a one-case improvement is not sufficient evidence by itself. Keep experimental compiler modes reversible until correctness and latency tradeoffs are understood.

## Native performance leads, not established bugs

The pinned native implementation applies the canvas post-RMSNorm with self-conditioning both enabled and disabled. The vLLM source also applies it on the first step. This audit did **not** find the suspected missing first-step normalization.

The local wrapper copies full-vocabulary logits to a host vector, and the thought sampler performs CPU exponential/entropy calculations over those rows. A 64-row block with 262,144 vocabulary entries contains **64 MiB of f32 logits per step**. The bundled native API already exposes device sampling and device self-conditioning, which the wrapper does not use. These are profiling targets, not measured speedups. Also measure repeated prompt prefill during thinking and first-step self-conditioning graph work before changing either path. Sources: [native wrapper](../../crates/llama-diffusion-structured/src/native.rs), [sampler](../../crates/llama-diffusion-structured/src/denoise.rs), [bundled native API](../../crates/llama-diffusion-sys/vendor/llama.cpp/include/llama.h), [native model graph](../../crates/llama-diffusion-sys/vendor/llama.cpp/src/models/diffusion-gemma.cpp).
