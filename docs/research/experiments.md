# Experiment log

The [experiment sequence](README.md#experiment-sequence) is complete on
`research/jev-inference-parity`. The implemented empty-header prefill default
improved all three tested seeds with unchanged Q4 weights. Final matched HTTP
runs confirm **165 → 189/231 correct**, with every response valid; p50 increased
from **0.847 to 1.009 seconds** and p95 from **8.589 to 9.381 seconds**.
Formatting, workspace tests, Clippy, four native/model tests, and HTTP smoke
checks passed. The default context is now 8,192.

## Experiment coverage

| Plan item | Evidence and current status |
| --- | --- |
| E0: baseline and context headroom | Both 4,096 and 8,192 reproduce every baseline probability on all 231 cases. All eight historical thinking/context failures return valid responses at 8,192; five are correct. |
| E1: empty thought framing | Canvas and prefill layouts completed all 231 cases at seeds 42, 123, and 2026. Prefill is the selected candidate; [confirmation results](confirmation-results.json). |
| E2: roles, wording, compact answers | Development comparisons and compiler control completed; none surpassed prefill. [Initial and contextual sweeps](experiment-results.json). |
| E3: semantic and contextual answer tokens | Semantic-label development tests completed. Contextual tokens also completed the full three-seed comparison and did not give a dependable gain; [paired evidence](confirmation-additional-context-paired.json). |
| E4: turn-close and canvas padding | Separate and combined development arms completed; no stronger candidate than prefill. [Initial sweep](experiment-results.json). |
| E5: fixed and adaptive reads | Tested legacy, reference-style, contextual, and selected-prefill layouts. Neither policy improved selected prefill; [refinement evidence](prefill-refinement-development.json). |
| E6: controlled precision comparison | Q8, Q4 derived from that exact Q8, and original Q4 each completed four matched 32-case layouts. [Results](precision-results.json) and [provenance](precision-sources.json). Total BF16-to-Q4 loss remains unmeasured. |
| E7: additional steps and thinking | Step-count comparisons completed, including selected prefill. All nine thinking/refinement arms completed (288 valid responses) with a 1,024-token thinking limit. |
| E8: native profiling and speed | First-step self-conditioning bypass preserved probabilities but showed no total-latency gain. [Phase profiling](no-thinking-phase-profile.json) identifies prefill as the main cost without thinking; Flash attention completed all 128 cases and is not selected: prefill lost two correct answers and measured slower. |
| Production acceptance | Candidate framing implemented. Formatting, workspace tests, Clippy, and all four model tests passed. Both HTTP smoke checks passed; full HTTP accuracy and probability parity are verified, with matched p50/p95 below. |

Only promising development candidates receive full-set confirmation. Repeated
seeds use the same public tasks; they are not independent holdout samples.

The candidate now defaults to context 8,192 in the library, service CLI, and SCM
CLI. Both full-set legacy context runs returned exactly the same probabilities,
and all eight historical context failures now return valid `think=1024` responses.
Five are correct; in-process p50/p95 are 31.168/48.449 seconds. This targeted
regression set is selected by historical failure and is not an independent
accuracy sample. [Full results](context-regression-results.json) retain each
case and trace. The larger default aligns startup with the tested research
configuration and provides room for reserved thought tokens, at the cost of more
cache memory. All native/model and HTTP checks passed.


## Completed: E0, 4,096-token baseline

The new persistent research driver reproduced **165/231 correct, 231/231 valid**.
Every probability vector and upstream-scored outcome exactly matches the tracked
September 21 baseline. In-process p50 was **0.756 s**, p95 **8.085 s**. These timings
exclude HTTP transport and use one excluded warm-up request; do not compare them
directly with the previous HTTP timings.

Raw run: `benchmarks/results/research-e0-4096/`. The [tracked results](experiment-results.json)
retain the per-case probabilities, diagnostics, timings, and run manifest.

The initial research runner compared integer ordinal gold labels with string
predictions, undercounting correct Score answers. Validation against the saved
baseline caught this. The runner now calls the pinned upstream JevBench scorer,
including its validation and tie-breaking rules. Original probability vectors
were rescored without new inference; the original intermediate output is retained
in the raw run for audit. All 231 corrected outcomes match the saved baseline.

## Completed: E0 headroom and first development sweep

Plan: [first-sweep-plan.json](first-sweep-plan.json). The baseline covers all 231
cases at context 8,192. Each other arm covers the [frozen 32-case development set](development-cases.json),
selected by a fixed hash within tier/type strata, independently of outcomes.
The full headroom baseline completed at 165/231, with all probabilities exactly
matching the 4,096-token run. Seed 42 is used first; promising configurations still require multiple seeds
and full-set confirmation.

The sweep tests empty thought scaffolds in canvas versus prefill, separate message
roles, compact answers, semantic labels, turn-close and padding, reference-style
combinations, OpenJev and current djev text compilers, fixed/adaptive rereads,
two/four denoising steps, and the zero-contribution self-conditioning bypass.
The reference compilers retain our runtime/noise, so these arms test formatting
effects rather than reproduce hosted systems.

Raw run: `benchmarks/results/research-first-sweep/`. A Q8 model download overlaps
this exploratory run. Treat latency here as preliminary; final timing runs must
exclude concurrent downloads and builds. No competing inference workload runs.

## Initial development findings (seed 42)

| Arm | Correct / 32 | Interpretation |
| --- | ---: | --- |
| Legacy baseline | 23 | Fixed development baseline. |
| Research compiler control | 23 | All probabilities exactly match the production compiler. |
| Skip zero-contribution self-conditioning | 23 | All probabilities unchanged; median paired decoder time ratio 0.932, end-to-end ratio 1.028. Preliminary timing does not establish a total-latency improvement. |
| Empty thought header in canvas | 26 | Five fixes, two regressions; no generated thought tokens. |
| Empty thought header in prefill | 27 | Gain with unchanged model precision and no generated thinking. |
| Separate system/user roles only | 16 | Regressed without the reference scaffold. |
| Compact answer lines only | 19 | Regressed on this development set. |
| Semantic Noul/Score labels only | 17 | Regressed; combined formats are reported in the completed follow-up. |
| Turn-close token only | 21 | Regressed without the complete reference canvas. |
| Turn-close plus padding | 25 | Padding changes outcomes with fixed Q4 weights. |
| Combined reference layout | 25 | Roles, compact answers, semantic labels, thought scaffold, turn-close and padding. |
| OpenJev wording/canvas | 25 | Local backend and RNG; matches the published development total but differs on two cases. |
| Current djev wording/canvas | 24 | Local backend and RNG; published djev gets 26 on these cases. |
| OpenJev adaptive rereads | 25 | No gain over its single-read text arm here. |
| Legacy adaptive rereads | 24 | One more correct than baseline. |
| Legacy four samples | 24 | Same total as adaptive sampling. |
| Legacy two steps | 26 | Three more correct than baseline. |
| Legacy four steps | 24 | More steps did not monotonically improve accuracy. |

These are development observations, not full-set gains. The same 32 cases scored
27/32 in the historical think=1024 run, but equal totals do not establish identical
answers. All 775 evaluations in the first sweep completed. A subsequent tokenizer
audit changed the next [follow-up plan](followup-plan.json): it now tests contextual
answer tokens alongside the winning prefill scaffold, reference compilers, and
additional reads/steps. Then verify selected changes on the full public set and
additional seeds before changing defaults.

## Tokenization finding

The first native trace test failed its newly added assumption that the legacy
segmented template equals tokenization of the complete answer. A vocabulary-only
native inspection confirmed the mismatch; [exact token arrays](tokenization-audit.json)
are retained. For example, the legacy `Answer: ` prefix ends in space token
236743 and uses bare `A` candidate 236776. The complete text `Answer: A` instead
ends in one space-prefixed `A` token, 562. Similarly, bare `yes` is 4443 but the
label in `q1: yes` is 11262. djev's compact `0:A` avoids this particular space
boundary, matching the legacy bare `A` token.

An opt-in `contextual_tokens` layout now resolves the label inside each prefix,
then verifies every candidate substitution against tokenization of the entire
group's answer text. Each label must occupy one distinct slot with all other
tokens fixed. This also recomputes canvas lengths and slot positions. The legacy
path remains available for controlled comparisons; contextual tokenization stays
opt-in. The native test now checks the legacy mismatch, contextual agreement,
exact trace positions, and zero-self-conditioning probability parity. The expanded
native test passed, including the context-aware read and diagnostic checks.

The follow-up measures its accuracy effect on the development set. The
first sweep's OpenJev arm used local segmented tokenization; its outcome therefore
does not establish parity with OpenJev's contextual candidate IDs.

## Completed: contextual-token follow-up

The 11-arm [follow-up plan](followup-plan.json) completed on the frozen 32 cases
at seed 42 and context 8,192. Raw run: `benchmarks/results/research-contextual-followup/`.
Downloads and conversion have finished; no other inference workload is active.
The first sweep remains a legacy-tokenization comparison, with its exact binary
hash and source snapshot retained.

All 352 requests were valid. With contextual candidate tokens: legacy scored
26/32, canvas header 26/32, prefill header 25/32, OpenJev text 25/32, and djev text
24/32. Adding four samples or adaptive rereads to contextual prefill reached
27/32; two steps scored 24/32 and four steps 25/32. Adaptive OpenJev remained
25/32. The legacy-token prefill control reproduced 27/32, so the more expensive
combinations did not improve this pilot's best total.

The [confirmation plan](confirmation-plan.json) selects the legacy baseline,
prefill header, canvas header, and contextual legacy answers for all 231 cases
at seeds 42, 123, and 2026. Selection favors simple individual changes; the
development set does not establish a winning production default. The
[paired report](contextual-vs-prefill.json) records fixes and regressions between
the prefill control and contextual legacy arm, separately by tier, type, and
family. Its exact McNemar p values are exploratory and unadjusted for selection.

Opt-in traces now include synchronized prefill/decode, host logits-copy, CPU
sampling, and entropy timings. These instrument the next runs; they were not
present in the completed follow-up binary. Ordinary service responses omit them.

## Completed: controlled precision comparison

The [precision plan](precision-plan.json) ran sequentially on Q8, its
derived Q4, and the original Q4, using the frozen 32 cases, seed 42, and context
8,192. Each model evaluates legacy, empty-header prefill, contextual legacy,
and contextual OpenJev layouts. The last arm crosses reference formatting with
precision; identical traces are required before attributing a paired difference
to the Q8-to-Q4 conversion. Raw directories are
`benchmarks/results/research-precision-{q8,derived-q4,original-q4}/`.

The controlled Q8 and derived-Q4 runs are complete, with all 256 requests valid.
Every paired prompt, initial canvas, slot position, and candidate ID matches
exactly across all 128 comparisons. [Raw measurements and manifests](precision-results.json)
retain the probability vectors and phase timings.

| Layout | Q8 correct / 32 | Derived Q4 correct / 32 | Q4 fixes / regressions |
| --- | ---: | ---: | ---: |
| Legacy | 23 | 24 | 2 / 1 |
| Empty-header prefill | 23 | 23 | 1 / 1 |
| Contextual legacy | 26 | 25 | 3 / 4 |
| Contextual OpenJev | 25 | 25 | 1 / 1 |

Paired details: [legacy](precision-legacy-paired.json),
[prefill](precision-prefill-paired.json),
[contextual legacy](precision-context_legacy-paired.json),
[contextual OpenJev](precision-context_openjev-paired.json).
All four exact McNemar tests have unadjusted p=1.0. This small pilot finds no
consistent accuracy penalty from the additional Q8-to-Q4 conversion; it does
not establish equivalence, exclude losses on other cases, or measure BF16 loss.
The original Q4 scored 23/32 (legacy), 27/32 (prefill), 26/32 (contextual legacy),
and 25/32 (contextual OpenJev), with all 128 requests valid. Every probability
vector exactly matches the corresponding earlier run before phase profiling was
added. Its 128 input traces also match Q8, despite the files' token-type metadata
differences. The original Q4's larger prefill score cautions against assuming
higher precision must win each small sample; its conversion lineage remains less
controlled than the Q8-derived Q4 pair.

## Completed full-set confirmation and inference queue

The [confirmation plan](confirmation-plan.json) completed all 231 cases at seed
42, writing `benchmarks/results/research-confirmation-42/`. The dependent queue
verified all 924 records and the completion marker, then ran the dependent queue in this order. All four items are complete:

1. All four confirmation arms on all 231 cases at seeds 123 and 2026:
   `benchmarks/results/research-confirmation-additional-seeds/`.
2. The nine-arm [thinking/refinement plan](thinking-plan.json) on 32 development cases:
   `benchmarks/results/research-thinking/`.
3. The eight thinking/context regressions described below:
   `benchmarks/results/research-thinking-context-regressions/`.
4. The four precision-plan layouts on the original Q4 with flash attention
   enabled, compared with the completed default-attention run:
   `benchmarks/results/research-flash-attention/`.

Each job must finish successfully before the next starts. The research binary
stayed fixed throughout this queue and every research arm selects its layout
explicitly. Source changes to the ordinary service default do not change these
comparisons.

All four full-set arms completed at seed 42:

| Configuration | Correct / 231 | Valid / 231 | In-process p50 | In-process p95 |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 165 (71.4%) | 231 | 0.837 s | 7.857 s |
| Empty-header prefill | 189 (81.8%) | 231 | 0.900 s | 7.660 s |
| Empty-header canvas | 187 (81.0%) | 231 | 0.777 s | 7.344 s |
| Contextual answer tokens | 167 (72.3%) | 231 | 0.797 s | 7.957 s |

Prefill fixes 31 cases and regresses seven, a gain of 24 correct answers
(10.39 percentage points). Its unadjusted exact McNemar p value is 0.000116;
the arm was selected on the development subset of these same public cases, so
this is not an untouched-holdout claim. The total equals the published OpenJev
189/231 and remains five cases below published djev's 194/231. Neither hosted
system was rerun for this comparison.

The canvas header fixes 31 and regresses nine (unadjusted exact McNemar
p=0.000680). Contextual tokens fix 31 and regress 29 (p=0.897), despite looking
promising on the development sample. Its tokenizer agreement is real, but does
not establish a dependable accuracy improvement. Paired cases and stratified
counts: [prefill](confirmation-prefill-paired.json),
[canvas](confirmation-canvas-paired.json),
[contextual tokens](confirmation-context_legacy-paired.json).

All 231 repeated baseline probability vectors exactly match the earlier
context-8,192 baseline. The candidate uses the same Q4 model, one read, one
decoder step, and zero generated thought tokens. The [confirmation snapshot](confirmation-results.json)
contains all 2,772 completed evaluations across the three seeds and four arms.
These in-process timings are not final HTTP latency.

Both additional-seed baselines completed with every response valid:
seed 123 scored 163/231 (in-process p50 0.906 s, p95 9.118 s), and seed 2026
scored 181/231 (p50 0.864 s, p95 8.094 s). This 18-case range across the three
baseline seeds exposes substantial sensitivity to the initial random canvas.
The completed candidate arms are compared against their matching
seed; the seed-42 baseline is not reused as their control. Selecting the
best observed seed alone would not establish a general improvement.

Prefill at seed 123 completed at **185/231**, versus its matching baseline's
163/231: 26 fixes, four regressions, and +9.52 percentage points. All 231
responses were valid. In-process p50/p95 were 0.935/8.324 s. The
[paired case list](prefill-seed123-paired.json) records the exact changes;
its unadjusted exact McNemar p value is 0.0000595.

Prefill at seed 2026 also completed with all responses valid: **181 to 189/231**,
14 fixes and six regressions (+3.46 percentage points; unadjusted exact McNemar
p=0.115). Its in-process p50/p95 were 0.934/8.474 s. The
[paired case list](prefill-seed2026-paired.json) retains the changes.
Across seeds 42/123/2026, mean correct counts rise from 169.7 to 187.7
(73.4% to 81.2%), and the observed range narrows from 18 cases to four.
These are repeated measurements of the same 231 public cases, not 693 independent
or unseen examples.

The canvas-header alternative scored 187/231 at every seed, with all responses
valid. The [completed framing comparison](framing-comparison.json) includes all
three arms and seeds, timings, and paired changes. Prefill's mean is slightly
higher (187.7 versus 187.0), and it scores two more correct at the retained
default seed 42. Prefill also keeps the existing answer canvas lengths and slot
positions. These are the reasons for retaining it as the candidate default;
the small difference does not establish that prefill wins every seed or workload.
Canvas p50/p95 were 0.851/8.154 s at seed 123 and 0.852/8.196 s at seed 2026.

The contextual-token arm finished at 144/231 for seed 123 and 178/231 for
seed 2026, versus matched legacy scores of 163 and 181. All responses were
valid. Together with 167 versus 165 at seed 42, this rejects contextual
tokenization as a dependable default improvement. Completed additional-seed
paired reports retain [prefill](confirmation-additional-prefill-paired.json),
[canvas](confirmation-additional-canvas-paired.json), and
[contextual-token](confirmation-additional-context-paired.json) changes.
The library-test build overlapped part of the seed-123 contextual arm; its
latency remains exploratory.

### Selected implementation

`Engine::read_with_options` now selects the tested empty thought-channel prefill
when `think=0`. Positive thinking budgets retain their existing generated-thought
path. Answer-token selection, sampling, denoising steps, and the CLI seed remain
unchanged. The example request now relies on the documented defaults.

This source change follows positive matched gains at all three seeds and passed
final native and HTTP checks. The research executable stayed fixed during the
queue. Its explicit `read_with_layout` calls preserve all control configurations,
including the legacy baseline, regardless of the ordinary API default.

The [seed audit](seed-sensitivity.json) verifies that all 231 single-question
requests start their answer slot with the same noise token within each seed:
42 uses token 58741 (`▁Sheffield`), 123 uses 236736 (`▁రావ`), and 2026 uses
14846 (`▁potentially`). These are vocabulary pieces, not generated reasoning.
The strings do not establish why a particular answer changes, but the observed
18-case range confirms that initialization needs controlled comparisons.
OpenJev's content-derived seeds and Python RNG remain distinct from this local
fixed-seed ChaCha8 policy.

The [per-family and reference comparison](prefill-full-set-analysis.json) shows
multi-hop improving from 7/18 to 13/18 and long-policy from 7/19 to 12/19;
temporal/numeric falls from 5/15 to 4/15. Equal totals do not mean identical
behavior: OpenJev and this candidate each have 13 uniquely correct cases in their
paired comparison. Relative to historical think=1024, prefill wins six cases
and loses one. Five wins replace historical context rejections; the remaining
win and the loss swap two formerly valid answers. On the 223 historically valid
requests, both score 184 correct. The separate headroom regression recovered valid responses on all eight historical failures, with five correct.

The selected legacy-token prefill layout was also tested with four samples,
adaptive reads, two steps, and four steps. None improved its development score;
the completed comparisons are reported below.

For thinking headroom, [context-regression-cases.json](context-regression-cases.json)
contains exactly the eight failed requests from the historical think=1024 run.
The [targeted plan](context-regression-plan.json) reran them at context 8,192
using `--case-ids docs/research/context-regression-cases.json`. All eight are valid
and five are correct. This regression set is reported separately from the
outcome-independent development sample.

### Model provenance

The existing Q4 file SHA-256 is
`24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5`, exactly matching
the pinned Unsloth artifact. [Source metadata](precision-sources.json) records the
repository revision and expected Q8 hash. Q8 download completed outside the repository. Its exact size and SHA-256 were
verified against the pinned publisher manifest. Metadata comparison found differing
tokenizer fields: `tokenizer.ggml.token_type`. These require inspection before
claiming precision-only parity. Nine token-type flags differ: tool/channel markers plus `<s>` and `</s>`.
The vocabulary, scores, merges, special-token IDs, and chat template hash-match.
The Q8-derived Q4 pair will preserve token-type metadata, avoiding this confound.

The publisher's card identifies the base model but not its immutable source
checkpoint or converter revision. To strengthen precision attribution, compare
Q8 with a Q4 file requantized from that exact Q8 source, holding tokenizer, prompt,
initial canvas, backend, and context fixed. This isolates additional Q8-to-Q4 loss,
not the total loss relative to BF16. Compare the original Q4 file separately.

The conversion driver is [requantize-gguf.cpp](../../scripts/requantize-gguf.cpp).
It refuses an existing output path, selects Q4_K_M with explicit requantization,
and uses eight CPU threads. [build-requantizer.py](../../scripts/build-requantizer.py)
links it to the libraries listed in Cargo's generated `cargo-link.txt`, keeping
the same pinned native implementation. Conversion completed successfully during
an idle inference window. Its verified SHA-256 is
`c26e504dcf7bfc3a89bc954a0e61ea1c08f0e6e93d7c99a00b730726084ec4da`.
The derived Q4 has exactly the source Q8 tokenizer metadata and the same tensor-type
counts as the original published Q4. Sixty-one tensors used the converter's
fallback types, matching the original Q4's 33 Q5_0 and 28 Q8_0 tensors. Example:

```bash
python3 scripts/build-requantizer.py \
  --link-manifest "$NATIVE_BUILD/cargo-link.txt" --output /tmp/jev-requantize
/tmp/jev-requantize "$Q8_MODEL" "$NEW_Q4_MODEL"
```

The [metadata inspector](../../scripts/gguf-research-metadata.py) records exact
tokenizer-value hashes and tensor descriptors without reading model payloads.
Opt-in `capture_trace` in the research layout records the actual prompt IDs,
initial canvas, candidate IDs, and positions for each structured read. It also
checks whether each prefix/candidate tokenization agrees with tokenizing their
concatenation. Traces are absent from ordinary service reads and support text
inputs only.

## Prefill refinement development results

The completed sampling arms in the ongoing thinking/refinement run do not
improve the selected framing: one read scored 27/32, while both four fixed reads
and adaptive sampling scored 26/32. Both lost `hard-opus-a-probability-08` and
fixed no cases. All responses were valid. The trace audit confirms that the
adaptive threshold triggered four reads for every case and produced exactly the
same 32 probability vectors as fixed four-read averaging. It saved no reads on
this sample. These results do not support increasing the default sample count.
Two denoising steps tied the one-step score at 27/32: they fixed
`hard-opus-a-long_policy-13` and lost `hard-opus-a-probability-08`. Four steps
scored 24/32, fixing no cases and losing three. All responses were valid.
There is no development accuracy gain to justify increasing the default steps.

| Prefill configuration | Correct / 32 | In-process p50 | In-process p95 |
| --- | ---: | ---: | ---: |
| One read, one step | 27 | 1.049 s | 8.186 s |
| Four fixed reads | 26 | 1.331 s | 9.678 s |
| Adaptive reads | 26 | 1.345 s | 9.195 s |
| Two steps | 27 | 1.071 s | 8.681 s |
| Four steps | 24 | 1.300 s | 9.072 s |

The [paired development evidence](prefill-refinement-development.json) records
completed arms, actual read/decoder counts, and exact regressions. These timings are exploratory; the final
HTTP comparison is measured separately.

## Legacy generated-thinking development result

At context 8,192, `think=1024` completed all 32 development cases with valid
responses and scored 27/32. It got exactly the same cases right and wrong as
empty-channel prefill, with no fixes or regressions. In-process p50/p95 were
21.703/36.577 seconds, versus 1.049/8.186 seconds for prefill without generation.
This does not justify enabling generated thinking by default.

The token traces resolve the earlier uncertainty about short outputs: 29 cases
reported four generated tokens, and every one repeated the open-channel,
`thought`, and newline tokens before closing. Their final prompt suffix was
`[100, 45518, 107, 100, 45518, 107, 101]`, versus the empty prefill header
`[100, 45518, 107, 101]`. These cases generated no substantive thought text.
Two cases reported one output token, and one exhausted the 1,024-token cap.
This audit applies to this completed development arm, not all historical
thinking outputs. [Results and exact case IDs](legacy-thinking-development.json).

Aggregate wall time in this arm was 41.9% decoder work, 29.3% CPU sampling,
23.3% prefill, and 4.7% full-logit copies. Those generation costs differ from the
no-thinking profile below. The intended-thinking-template comparisons add a
system turn and its thinking marker; they do not isolate the marker token alone.

## Thinking marker with the existing compiler

The completed `legacy_think1024_marker` arm scored **30/32**, with all responses
valid in process. Against empty-channel prefill's 27/32, it fixed
`hard-opus-b-tradeoff-07`, `hard-opus-c-long_policy-08`, and
`hard-sol-b-temporal_numeric-03`, with no regressions. These are development-set
results, not full-set confirmation; the exploratory exact paired p value is 0.25.

In-process **p50 was 54.820 s and p95 was 343.822 s**, versus 1.049/8.186 s for
empty-channel prefill. Outputs ranged from 96 to 1,024 tokens, with a median of
243.5; four cases reached the token cap. Nine requests took more than 120 seconds.
Only 23 answers were both correct and completed within 120 seconds. The pinned
HTTP adapter uses that timeout, so the 30/32 in-process score does not establish
an improved score under the original HTTP protocol. The observed durations are
not a simulated HTTP run: client timeouts and queued work were not exercised.

The [paired results and timing audit](thinking-template-development.json) retain
the exact slow-case IDs and token-cap cases. Phase totals were 40.0% decoder
work, 33.1% prefill, 22.5% CPU sampling, and 3.7% full-logit copies. The summary
exporter includes `over_120s_ids` so this limitation stays visible in the final
results. This arm remains experimental because of its runtime cost and limited
accuracy evidence.

The completed OpenJev-style wording/template arm scored **29/32**, fixing three
prefill errors and losing `hard-opus-a-probability-08`. In-process p50/p95 were
**35.156/344.051 seconds**. Seven requests exceeded 120 seconds; 25 were both
correct and finished within that duration. Its median output was 177 tokens,
and three cases reached the 1,024-token cap. This is a local format approximation,
not a rerun of the hosted OpenJev implementation. Both proper-thinking arms remain
experimental. [All 288 thinking/refinement records](thinking-results.json) include
per-tier, question-type, and family summaries.

## No-thinking phase profile

On the original-Q4 precision run, prompt prefill consumed 92.5–93.3% of
aggregate request wall time across the four 32-case layouts. Decoder work
consumed 6.4–7.2%. For the empty-header prefill arm, median prefill and decoder
times were 746.6 ms and 115.1 ms respectively. These are separate phase medians,
not additive estimates of request p50. Full-logit copy and generation-sampling
timers were zero because these requests generated no thought tokens.
The [phase totals](no-thinking-phase-profile.json) retain the measurements.
This supports investigating prefill performance; the earlier first-step
self-conditioning bypass did not demonstrate an end-to-end speedup. The flash
attention comparison is complete below. Generated-thinking phase costs are reported above.

## Flash-attention pilot

All four original-Q4 layouts completed 32 cases with flash attention enabled.
The default-attention control is the earlier original-Q4 precision run; every
paired prompt, initial canvas, slot position, and candidate-token trace matches.

| Layout | Default correct | Flash correct | Default p50 / p95 | Flash p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 23 | 23 | 0.778 / 7.449 s | 0.876 / 9.222 s |
| Empty-header prefill | 27 | 25 | 0.861 / 7.402 s | 0.937 / 8.938 s |
| Contextual legacy | 26 | 26 | 0.791 / 7.245 s | 0.865 / 8.759 s |
| Contextual OpenJev-style | 25 | 26 | 1.101 / 7.482 s | 1.136 / 9.332 s |

All responses are valid. Prefill loses `hard-opus-a-probability-08` and
`hard-opus-b-ambiguous-03`, with no fixes (exploratory exact paired p=0.5).
Legacy and contextual legacy each swap one correct case; contextual OpenJev-style
fixes one. These small differences do not establish a general accuracy effect,
but this pilot gives no reason to enable flash attention for the selected default.
The separate serial timing runs also show no speed benefit on this HIP machine;
these are exploratory in-process timings, not a randomized performance trial.

[Full results](flash-attention-results.json), [phase totals](flash-attention-profile.json),
and paired results for [legacy](flash-attention-legacy-paired.json),
[prefill](flash-attention-prefill-paired.json),
[contextual legacy](flash-attention-context_legacy-paired.json), and
[contextual OpenJev-style](flash-attention-context_openjev-paired.json) retain
all inputs, outcomes, and regressions.

## Validation so far

The four main completed compact exports contain 4,802 records: 1,358 in the initial
and contextual experiments, 384 in precision comparisons, 2,772 in full-set
confirmation, and 288 in thinking/refinement. The eight context regressions and 128 flash-attention cases are
exported separately, for 4,938 in-process evaluations in total. Each export includes tier/type/family summaries; their counts
reconcile with the overall arm summaries. An integrity check verified completion markers, expected counts,
unique `(arm, seed, task_id)` records within each run, and agreement between every
existing summary field and the per-case data. Python script syntax also passed.

Final source validation passed on the HIP/native release build:

- Formatting: `cargo fmt --all -- --check`.
- Workspace tests: `cargo test --release --workspace --locked --features hip,native`;
  22 regular tests passed, with four model tests reserved for the separate run.
- Clippy: `cargo clippy --release --workspace --all-targets --locked --features hip,native -- -D warnings`.
- Native/model tests: `cargo test --release -p llama-diffusion-structured --locked --features hip,native -- --ignored --nocapture --test-threads=1`;
  all four passed with the Q4 model and vision projector.
- Service build: `cargo build --release --locked -p llama-cpp-system-one --features hip,native`.

Native checks cover image prefill and subsequent text reproducibility, multiple
reads/steps, generated thinking, sequential and multi-chunk requests, all 128
answer codes, profile fields, and resetting experimental state between requests.
[Validation commands, exit statuses, log hashes, and source hashes](validation-results.json)
identify the tested implementation. No source changes have followed these checks.
Matched HTTP benchmarks and smoke checks run serially after all builds.

An earlier 27-second library build overlapped the exploratory contextual-token
arm at seed 123; `library-test-overlap.json` records that window in the raw run.
All framing arms had already finished. Final HTTP measurements exclude concurrent
builds. Research inference kept one fixed binary throughout the queue, SHA-256
`708d96bbea4e7ca360be2e1aba4c400b0180df9f344be1464ff8e1dc67408bc1`.

The research runner's final-exit check was verified with two simulated drivers:
both returned all three fixture answers, but only the driver exiting zero could
write a completion marker. A driver exiting seven was rejected after retaining
the answers. This prevents a late process failure from silently completing a run.

Source review also corrected an extreme-value case in experimental partial
entropy: normalization now stays in the shifted-logit domain so a large common
offset cannot erase `ln(sum(exp(...)))`. The existing entropy unit test now
checks a uniform vector at `f32::MAX` and passed in the final workspace suite.
This affects the experimental adaptive gate, which is disabled in the selected
default. This fix was made after freezing the experiment binary. Across 384 saved adaptive
trace values, the closest observed entropy was 0.0253 away from the 0.1 threshold.

## Reproduction

Build the driver with the documented HIP environment:

```bash
AMDGPU_TARGETS=gfx1151 cargo build --release --locked \
  -p llama-cpp-system-one --example research --features hip,native
python3 -B scripts/jev-experiments.py \
  --tasks benchmarks/results/jevbench-2026-09-21T11-32-44Z/tasks \
  --plan docs/research/first-sweep-plan.json \
  --output benchmarks/results/research-first-sweep-new \
  --model "$DIFFUSION_MODEL" --context 8192
```

The original benchmark setup creates the task directory and its sibling pinned
`harness/jevbench/scoring.py`; see [benchmark reproduction](../../benchmarks/jevbench/README.md).
The driver requires a fresh output directory, preserves requests/results, and
records binary/scorer hashes, task hashes, configuration, and a source diff.
Use `--seeds 42,123,2026` for repeated seed comparisons. Export compact results with
`scripts/jev-experiment-report.py RUN_DIRECTORY --output OUTPUT.json`.

For the final HTTP measurements, [jev-http-benchmark.py](../../scripts/jev-http-benchmark.py)
starts one specified server binary, waits for that process's readiness message,
performs one excluded warmup, runs the pinned HTTP harness, verifies that every
frozen task has exactly one result, and stops its server. Run it separately for
the preserved baseline and candidate, without other inference/build workloads:

```bash
python3 -B scripts/jev-http-benchmark.py \
  --binary "$BASELINE_BINARY" --model "$DIFFUSION_MODEL" \
  --fixture-run benchmarks/results/jevbench-2026-09-21T11-32-44Z \
  --output benchmarks/results/research-http-baseline
```

Use a fresh output directory for the candidate. Both runs use context 8,192,
seed 42, eight CPU threads, batch size 512, and serial HTTP requests. The runner
records binary/task/scorer hashes, arguments, raw responses, and type-7 p50/p95
for all attempts and valid responses separately. After measurement, it runs the
HTTP smoke test against the same server, covering the multi-question example,
probability mapping, and validation errors. It requires a clean server exit.
Both HTTP runs and smoke checks completed successfully. Argument parsing and
percentile edge cases were checked before starting them.

After both HTTP runs finish, export the comparison and verify exact probability
parity against the completed seed-42 research arms:

```bash
python3 -B scripts/jev-http-report.py \
  --baseline benchmarks/results/research-http-baseline \
  --candidate benchmarks/results/research-http-candidate \
  --confirmation benchmarks/results/research-confirmation-42 \
  --output docs/research/http-results.json
```

The exporter requires matched server arguments, task/scorer hashes, and complete
case coverage. It reapplies the pinned scorer, recomputes p50/p95, retains full
per-case results and paired regressions, and flags any probability differences
from research confirmation. Its simulation against saved historical data passed
the matching-data case, deliberate probability-change detection, and duplicate
record rejection. That simulation made no new inference requests and is not a
substitute for the real HTTP benchmarks.

## Final matched HTTP acceptance

Both complete 231-case runs used the same original-Q4 model, context 8,192,
seed 42, eight CPU threads, batch size 512, one excluded warmup, serial loopback
HTTP requests, a 120-second timeout, and no retries. No build or other inference
workload ran concurrently. Each server passed the HTTP smoke checks after timing
and exited cleanly.

| Implementation | Correct | Valid | All-attempt p50 | All-attempt p95 | Valid p50 | Valid p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 165/231 | 231/231 | 0.847 s | 8.589 s | 0.847 s | 8.589 s |
| Updated | 189/231 | 231/231 | 1.009 s | 9.381 s | 1.009 s | 9.381 s |

Every probability vector exactly matches the corresponding completed seed-42
research arm. The update fixes 31 cases and regresses seven, gaining 10.39
percentage points. Easy/standard/hard correct counts change from 44/63/58 to
48/69/72. The exploratory paired exact p value is 0.000116, unadjusted for
selection on the public development subset. The updated total equals published
OpenJev's 189 and is five below djev's 194; those endpoints were not rerun.

The latency tradeoff is **+19.1% p50 and +9.2% p95** in this serial comparison.
This supports an accuracy improvement, not a speed improvement. Phase profiling
points to prompt prefill as the dominant no-thinking cost. Timing variation and
fixed run order limit causal performance attribution.

[Full HTTP results, binary/configuration provenance, exact parity checks,
and stratified paired outcomes](http-results.json) retain the evidence.
Together with the 4,938 in-process experiment records, these runs provide 5,400
recorded evaluations; repeated public cases are not independent samples.
All E0–E8 experiment groups and production acceptance checks are complete.
Total BF16-to-Q4 degradation, unseen-task generalization, and improvements to
proper generated-thinking latency remain open research questions, rather than
claims established by these experiments.
