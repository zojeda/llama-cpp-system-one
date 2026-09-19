# Diffusion and canvas inference

[Back to README](../README.md)

## Text diffusion

An autoregressive language model generates one next token per decoding step. A text diffusion model predicts tokens across a block, called a **canvas**, using a sequence that starts with noise. During training, a denoising model learns to recover text from corrupted tokens. During generation, a sampler uses its predictions to refine the canvas over several steps.

A token can represent a word, part of a word, or punctuation. A canvas contains token positions, so an answer slot need not correspond to a whole word.

In masked diffusion, noise takes the form of a special mask token. DiffusionGemma uses **uniform state diffusion**: random vocabulary tokens supply the noise. Its sampler can replace uncertain tokens with fresh noise and revisit predictions as the context changes. Google describes this process in [Diffusion in Text Generation Explained](https://ai.google.dev/gemma/docs/diffusiongemma/explained).

Google built DiffusionGemma on the Gemma 4 26B A4B mixture-of-experts architecture. For full text generation, its encoder caches the prompt, and its decoder refines a 256-token canvas with bidirectional attention. After completing a block, it adds that block to the context and starts another. The [model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card) describes the full sampler and architecture.

## A restricted canvas for structured answers

Here we supply the surrounding text and reserve one token position for each answer. We use "masked canvas" to describe those unknown positions. The implementation fills them with random vocabulary tokens, excluding the special mask token.

Consider two questions about a material. For readability, this example abbreviates the prompt:

```text
Prompt
  State: Ground granulated blast furnace slag is used in concrete.
  Question 1: Is this an SCM?       A = yes, B = no
  Question 2: Which material?      A = scm, B = aggregate, C = reinforcement

Canvas before the read
  Question 1
  Answer: [random token]
  Question 2
  Answer: [random token]
```

Each bracket represents one token. The question prefixes remain fixed. The initial random tokens can come from outside the allowed answer codes.

We assign codes such as `A`, `B`, and `C` because each answer slot occupies one token. At model load, we verify that each code maps to a distinct token. Your external labels, such as `reinforcement`, can contain multiple tokens. After inference, we map codes back to those labels and restore your question IDs. The model receives numbered questions; it does not receive the IDs.

```mermaid
flowchart LR
    R[State and questions] --> P[Prompt and answer codes]
    P --> K[Prompt prefill and cache]
    C[Fixed canvas text plus noisy answer slots] --> D[One canvas evaluation]
    K --> D
    D --> L[Allowed logits at each slot]
    L --> S[Restricted softmax]
    S --> A[Probabilities and typed answers]
```

The [compiler](../crates/system-one/src/compiler.rs) constructs the prompt and slot prefixes. The [inference engine](../crates/llama-diffusion-structured/src/engine.rs) then:

1. Wraps the prompt in DiffusionGemma's text chat markers and tokenizes it.
2. Appends fixed prefixes and one seeded random token per answer slot to the canvas.
3. Prefills the prompt cache through `PKV_PREFILL`, in chunks up to the batch size.
4. Evaluates the full canvas with one `PKV_DECODE` call.
5. Reads the allowed candidate logits at each answer position and computes probabilities.

During the canvas read, each position can attend to positions on either side and use the prompt cache. The answer slots therefore share context; they are not independent model runs. The engine returns distributions after that read, without writing predicted answers into a second canvas or running a refinement loop. "One read" counts the canvas evaluation; prompt prefill adds work before it.

The prompt cache holds the attention keys and values computed during prefill. The decoder reuses those representations to condition its canvas predictions on the supplied state and questions.

## From logits to answers

A logit is an unnormalized score for a token. We keep the logits for the allowed answer codes and apply softmax over that set:

```text
p(i) = exp(logit(i) - max_logit) / sum_j exp(logit(j) - max_logit)
```

For example, logits `A = 2`, `B = 1` give probabilities of about `0.731` and `0.269`. These numbers illustrate the math; they are not a measured answer to the material example.

| Question type | Mapping |
| --- | --- |
| `noul` | Return the probability assigned to yes. |
| `choice` | Return the label with the largest probability and the distribution. |
| `score` | Return the expected zero-based rubric level: `sum_i i * p(i)`. |

For `choice` and `score`, we compute confidence as `1 - H(p) / ln(K)`, where `H(p) = -sum_i p(i) ln(p(i))` and `K` is the number of options. A uniform distribution gives 0; a distribution concentrated on one option gives 1. A single-option choice gives 1 by convention.

This normalization measures preference among the supplied options. It discards probability mass on other vocabulary tokens. Adding an option can change the distribution, and high entropy confidence can accompany a wrong answer. We have not calibrated these values as probabilities of correctness.

## Implementation details

The [native wrapper](../crates/llama-diffusion-structured/src/native.rs) uses `llama_diffusion_set_sc(model, nullptr, 0.0, 1.0, true)` for the canvas phase. The zero gate disables prior-step self-conditioning.

We seed `ChaCha8Rng` for the initial slot tokens. Matching requests and seeds reproduce that initialization. The C++ prototype uses `std::mt19937`, so equal numeric seeds across the two implementations need not produce equal canvases or probabilities.

We use the text-only `<|turn>` / `<turn|>` chat framing with `think=0`. The C++ prototype's common chat helper enables a thinking preface by default; the Rust prompt has seven fewer tokens with the tested GGUF. We refresh the prompt cache for each request.

The canvas length follows the question prefixes and slot count. It must fit in one `--batch-size` batch, and prompt plus canvas must fit in `--context-size`. We do not pad this restricted read to the full generator's 256-token canvas or split it into multiple reads.

`usage.input_tokens` counts prompt and canvas tokens; `output_tokens` is 0. The CLI's `forward_ms` covers the canvas evaluation and logit synchronization/download, excluding prompt prefill. The HTTP benchmark includes the request's full elapsed time.
