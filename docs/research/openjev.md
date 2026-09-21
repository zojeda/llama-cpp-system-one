# OpenJev implementation research

Investigated 2026-09-21. Scope: `razorback16/openjev`, the DiffusionGemma NVFP4 entrant, **not** the unrelated `openjev-sglang`, SemIf, or OpenJev Verdict projects. No inference was run for this investigation.

**Finding:** OpenJev differs in prompt structure, thought framing, answer tokens, canvas contents, and automatic averaging. Its published advantage cannot currently be attributed to quantization alone. The first experiments should keep our existing Q4_K_M weights and change these factors individually.

## Source and benchmark provenance

| Component | Pinned evidence | Limitation |
| --- | --- | --- |
| JevBench | `fd51755eb0c0b546ca206d764faf3302feca913e` | Published reference runs, not reruns here. |
| Benchmarked OpenJev | Docker image `razorback16/openjev:0.2.0`, defaults except context 32,768; release source `91d5005effcf8cc0ecccaa9538ceabbb130fef59` | Benchmark records the image tag; no immutable image digest or embedded source commit was found. The release commit is the corresponding source, not independently proven image provenance. |
| Historical vLLM | `razorback16/vllm@9bbf7418e85020dc76da9f60cdfe6c4e912ec048` | Pinned by OpenJev 0.2.0 Dockerfile. |
| OpenJev current HEAD | `2050fdb8280d3094180870ac4df962f1bb44edca` | Adds MLX and changes the vLLM pin; should not substitute for the tested release. |
| Model | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | Benchmark does not pin the Hugging Face revision. Current metadata inspected at `ec4ff3df205028f4e81c954c2227f9312b3ec2ea` is identified separately below. |

Sources: [benchmark deployment plan](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/scripts/v1.1.3/r2_plan.json), [release Dockerfile](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/docker/Dockerfile), [current source](https://github.com/razorback16/openjev/tree/2050fdb8280d3094180870ac4df962f1bb44edca).

The benchmark used an RTX PRO 4500 Blackwell 32 GB in RunPod EU-RO-1. Requests travelled from Germany over public HTTP. Its v1.1 repeated answers matched 237/242, despite deterministic seed construction. Consequently, published results contain some run variability; neither accuracy gaps nor latency gaps constitute controlled backend comparisons. [Benchmark run conditions](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/RESULTS-v1.1.3.md#how-the-new-systems-ran)

Our saved exact-public-subset comparison is 189/231 for OpenJev, 165/231 locally without thinking, and 184/231 locally with `think=1024`. These are 231 shared public cases, not the complete official benchmark. [Local comparison](../../benchmarks/jevbench/README.md)

## Concrete implementation differences

| Factor | OpenJev 0.2.0 | Local baseline |
| --- | --- | --- |
| Message roles | Question instructions and allowed answers in a system turn; state alone in user turn | Instructions, state, and questions combined into one user turn |
| Prompt order | Questions before user state | State before questions |
| Answer template | `q1: A`, etc.; compact indexed format after ten questions | `Question 1\nAnswer: A`, etc. |
| Empty thought block | `<\|channel>thought\n<channel\|>` is fixed text at start of the **bidirectional canvas**, even with `think=0` | No thought scaffold for `think=0` |
| Noul labels | Actual `yes` / `no` tokens | Assigned answer codes such as `A` / `B` for true / false |
| Score labels | Actual ordinal digit tokens | Assigned alphabetic answer codes |
| Choice labels | Alphabetic candidates verified in complete answer-template context | Candidates checked independently for single-token encoding |
| Canvas end/width | Turn-close token 106, PAD token 0; rounds width up to multiples of 16, maximum 64 | Exact answer-template length; no explicit ending/padding |
| Default samples | One read, then three more if any slot's entropy exceeds 0.1; averages probabilities | One read, no automatic rereads |
| Seeds | First 32 bits of SHA-256 of sorted-key JSON containing state and question objects; Python `random.Random`; adds 7919 per sample | Fixed CLI seed 42 by default, ChaCha8; adds 7919 per sample |
| Context | 32,768 in benchmark | 4,096 in saved runs |

OpenJev preserves input question/option insertion order; local `serde_json` also enables `preserve_order`, so a default sorting difference is **not established**. OpenJev serializes nonstring state with Python `json.dumps(..., ensure_ascii=False)` including default spaces; local serialization uses compact JSON. Descriptions are trimmed in OpenJev and choice descriptions appear in parentheses. These smaller formatting differences are additional confounds. [Historical schema/compiler/template](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/openjev/engine.py#L60-L231), [seed construction](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/openjev/api.py#L150-L163), [local compiler](../../crates/system-one/src/compiler.rs), [local engine](../../crates/llama-diffusion-structured/src/engine.rs), [local defaults](../../crates/llama-cpp-system-one/src/main.rs).

**Entropy detail:** OpenJev averages restricted-label probabilities, but its reread trigger uses the unnormalized partial entropy of full-vocabulary probabilities returned for top-20 tokens plus explicitly requested label IDs. It does not use entropy of the renormalized candidate distribution. Explicit `samples` overrides this automatic policy. Match this distinction in experiments; a threshold of 0.1 on candidate-only entropy is a different algorithm. [Read/aggregation code](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/openjev/engine.py#L300-L393), [defaults](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/openjev/config.py#L15-L27)

**Probability semantics:** Both implementations normalize candidate logits at temperature 1 for the default one-step read. OpenJev requests exact log probabilities for every candidate; the historical vLLM path explicitly returns raw, untempered logits for read-only requests. Noul returns positive probability, choice returns argmax, and score returns expected ordinal value; confidence uses normalized entropy. These output-mapping choices are substantially aligned, even though answer-token choices differ. [OpenJev mapping](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/openjev/engine.py#L382-L412), [vLLM read-only logits](https://github.com/razorback16/vllm/blob/9bbf7418e85020dc76da9f60cdfe6c4e912ec048/vllm/model_executor/models/diffusion_gemma.py#L1539-L1574), [local mapping](../../crates/system-one/src/response.rs).

## Decoder and thinking details

The historical vLLM implementation initializes self-conditioning embeddings to zero, but still runs its self-conditioning module during the first denoising forward pass. With zero input and bias-free projections, the residual signal is zero; the module still performs an unweighted post-RMSNorm on input embeddings. The local audit found that pinned llama.cpp also applies post-RMSNorm whether self-conditioning is enabled or disabled, so this check did not reveal a missing normalization bug. Distinguish “no previous-step signal” from “skip the entire self-conditioning transform.” [Module](https://github.com/razorback16/vllm/blob/9bbf7418e85020dc76da9f60cdfe6c4e912ec048/vllm/model_executor/models/diffusion_gemma.py#L74-L104), [initialization](https://github.com/razorback16/vllm/blob/9bbf7418e85020dc76da9f60cdfe6c4e912ec048/vllm/model_executor/models/diffusion_gemma.py#L800-L841), [application](https://github.com/razorback16/vllm/blob/9bbf7418e85020dc76da9f60cdfe6c4e912ec048/vllm/model_executor/models/diffusion_gemma.py#L1020-L1080).

OpenJev's optional thought path calls the tokenizer's chat template with `enable_thinking=True`, opens the thought channel, generates until the closing channel or budget, then closes it before the structured read. Its ordinary read uses `enable_thinking=False` and puts the empty scaffold in the canvas. The model's currently published chat template adds `<|think|>` inside a system turn when thinking is enabled; the template otherwise ends after the model turn marker. Our local `think()` opens the thought channel but does not request that system marker. Verify historical tokenizer metadata before claiming exact historical token parity. [Historical thought path](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/openjev/engine.py#L244-L298), [pinned current model template](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4/blob/ec4ff3df205028f4e81c954c2227f9312b3ec2ea/chat_template.jinja), [local thought implementation](../../crates/llama-diffusion-structured/src/engine.rs).

## Quantization evidence

NVFP4 is also quantized; it is not a full-precision reference. Current model metadata describes four-bit floating-point weights and input activations, group size 16, and exclusion patterns covering attention, router, self-conditioning, embeddings/vision components, and the output head. It also lists an eight-bit floating KV cache scheme, but the actual serving KV cache dtype should be verified from runtime before asserting that configuration was used. This differs from GGUF Q4_K_M's allocation of precision. Neither format's name establishes which is more accurate for this task. [Pinned current model configuration](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4/blob/ec4ff3df205028f4e81c954c2227f9312b3ec2ea/config.json), [historical startup flags](https://github.com/razorback16/openjev/blob/91d5005effcf8cc0ecccaa9538ceabbb130fef59/docker/entrypoint.sh).

## Ranked hypotheses and experiments

1. **Thought framing and canvas structure.** On the existing Q4_K_M model, test only the empty thought scaffold in the canvas. Separately test scaffold in the prompt, turn-close suffix, and padding to a 16-token boundary. Bidirectional placement is not interchangeable with prefill placement. The existing thinking result is not a clean measure of reasoning: it also changes these surrounding tokens.
2. **Prompt roles and wording.** Hold the canvas fixed while changing to system questions/user state. Then change wording and the `q1: label` template separately. Save rendered prompt IDs and slot positions so a claimed match is inspectable.
3. **Automatic noise averaging.** Compare one read and four fixed draws using existing `samples`; then implement/evaluate the exact entropy-triggered policy. Log each initial slot token and distribution. Keep seed families paired; Python and ChaCha8 do not produce matching draws from matching integers.
4. **Semantic label tokens.** Test yes/no for noul and digits for scores separately from the preceding changes; use the complete-template single-token check. Stratify outcomes by question type.
5. **Native/backend parity.** The first-step post-RMSNorm check is aligned in source. Continue checking token IDs, canvas attention boundaries, and selected logits before adding further decoder steps. Any failure here precedes a quantization conclusion.
6. **Precision ladder.** After freezing the best understood prompt/decoder path, compare Q4_K_M to a higher-precision GGUF from the same source checkpoint and conversion lineage on the same backend. Ideally also compare quantizations under the same vLLM implementation. A Q4_K_M/llama.cpp versus NVFP4/vLLM comparison changes too many factors to isolate precision.

For each run record correct/total, valid/total, per-tier and per-type deltas, fixed/regressed IDs, p50/p95 latency, read counts, and prompt/canvas/thought token counts. Keep all 231 public cases in the denominator; report context failures separately. Use a fixed development subset for choosing variants, reserve an untouched evaluation subset, and test multiple seeds before accepting small gains. These are proposed experiments, not results.
