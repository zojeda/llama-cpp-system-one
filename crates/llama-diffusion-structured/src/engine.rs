//! Token preparation, repeated reads, and bounded diffusion generation.

use crate::native::{Batch, Native, PromptPart};
use crate::{Error, ModelConfig, ReadRequest, ReadResult, Result, SlotRead, restricted_softmax};
use crate::{ImageInput, ReadLayout, ReadOptions, ReadProfile, ReadTrace, denoise};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::{collections::HashSet, time::Instant};

const EMPTY_THOUGHT_CHANNEL: &str = "<|channel>thought\n<channel|>";

pub struct Engine {
    native: Native,
    codes: Vec<String>,
}

impl Engine {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let native = Native::load(config)?;
        for marker in ["<|turn>", "<turn|>"] {
            let tokens = native.tokenize(marker, false, true)?;
            if tokens.len() != 1 {
                return Err(Error::InvalidInput(format!(
                    "The model does not have the DiffusionGemma text chat marker {marker}"
                )));
            }
        }
        let mut engine = Self {
            native,
            codes: Vec::new(),
        };
        engine.codes = engine.find_codes(128)?;
        Ok(engine)
    }

    /// Verified single-token codes used to represent arbitrary external labels.
    pub fn codes(&self) -> &[String] {
        &self.codes
    }

    fn find_codes(&self, count: usize) -> Result<Vec<String>> {
        let mut codes = Vec::with_capacity(count);
        let mut seen = HashSet::new();
        for ch in ('A'..='Z').chain('a'..='z').chain('0'..='9') {
            let code = ch.to_string();
            let tokens = self.native.tokenize(&code, false, false)?;
            if tokens.len() == 1 && seen.insert(tokens[0]) {
                codes.push(code);
            }
        }
        for token in 0..self.native.n_vocab {
            if codes.len() >= count {
                break;
            }
            if seen.contains(&token) || token == self.native.mask {
                continue;
            }
            if let Some(code) = self.native.code_piece(token)
                && self.native.tokenize(&code, false, false)? == [token]
            {
                seen.insert(token);
                codes.push(code);
            }
        }
        if codes.len() < count {
            return Err(Error::InvalidInput(format!(
                "The vocabulary has fewer than {count} usable single-token answer codes"
            )));
        }
        Ok(codes)
    }

    pub fn read(&mut self, request: &ReadRequest, seed: u64) -> Result<ReadResult> {
        self.read_with_options(request, seed, ReadOptions::default(), &[])
    }

    pub fn read_with_options(
        &mut self,
        request: &ReadRequest,
        seed: u64,
        options: ReadOptions,
        images: &[ImageInput],
    ) -> Result<ReadResult> {
        // Cache the empty thought channel as answer framing without generating tokens.
        // Generated thinking opens and closes its own channel instead.
        let layout = ReadLayout {
            prefill_suffix: if options.think == 0 {
                EMPTY_THOUGHT_CHANNEL.into()
            } else {
                String::new()
            },
            ..Default::default()
        };
        self.read_with_layout(request, seed, options, images, &layout)
    }

    /// Run an explicit layout, bypassing the framing defaults of `read_with_options`.
    pub fn read_with_layout(
        &mut self,
        request: &ReadRequest,
        seed: u64,
        options: ReadOptions,
        images: &[ImageInput],
        layout: &ReadLayout,
    ) -> Result<ReadResult> {
        options.validate()?;
        // Reset on every request so an experimental setting cannot leak to later reads.
        self.native.skip_zero_self_conditioning = layout.skip_zero_self_conditioning;
        if layout.capture_trace && !images.is_empty() {
            return Err(Error::InvalidInput(
                "Token traces currently support text-only requests".into(),
            ));
        }
        if layout.adaptive_samples && options.samples != 1 {
            return Err(Error::InvalidInput(
                "Adaptive sampling requires samples=1".into(),
            ));
        }
        if options.think > 0 && !layout.prefill_suffix.is_empty() {
            return Err(Error::InvalidInput(
                "A prefill suffix cannot be combined with generated thinking".into(),
            ));
        }
        if layout.thinking_marker && layout.system_prompt.is_none() {
            return Err(Error::InvalidInput(
                "Thinking marker requires a system turn".into(),
            ));
        }
        if request.prompt.trim().is_empty() || request.slots.is_empty() {
            return Err(Error::InvalidInput(
                "A prompt and at least one slot are required".into(),
            ));
        }
        if !images.is_empty() && (options.think > 0 || options.sequential) {
            return Err(Error::InvalidInput(
                "Images cannot be combined with think or sequential".into(),
            ));
        }
        let mut parts = Vec::new();
        if let Some(system) = &layout.system_prompt {
            parts.push(PromptPart::Text(self.native.tokenize(
                "<|turn>system\n",
                true,
                true,
            )?));
            if layout.thinking_marker {
                parts.push(PromptPart::Text(self.native.tokenize(
                    "<|think|>\n",
                    false,
                    true,
                )?));
            }
            parts.push(PromptPart::Text(
                self.native.tokenize(system, false, false)?,
            ));
            parts.push(PromptPart::Text(self.native.tokenize(
                "<turn|>\n",
                false,
                true,
            )?));
        }
        parts.push(PromptPart::Text(self.native.tokenize(
            "<|turn>user\n",
            layout.system_prompt.is_none(),
            true,
        )?));
        parts.extend(self.native.image_parts(images)?);
        parts.push(PromptPart::Text(self.native.tokenize(
            request.prompt.trim(),
            false,
            false,
        )?));
        parts.push(PromptPart::Text(self.native.tokenize(
            "<turn|>\n<|turn>model\n",
            false,
            true,
        )?));
        let base_length: usize = parts.iter().map(PromptPart::len).sum();
        let prepared = request
            .slots
            .iter()
            .map(|slot| {
                if slot.candidates.is_empty() {
                    return Err(Error::InvalidInput("Each slot needs a candidate".into()));
                }
                let mut candidates = Vec::new();
                let mut prefix = self.native.tokenize(&slot.prefix, false, false)?;
                let mut context_matches = Vec::new();
                for candidate in &slot.candidates {
                    let tokens = self.native.tokenize(candidate, false, false)?;
                    if tokens.len() != 1 {
                        return Err(Error::InvalidInput(format!(
                            "Candidate {candidate:?} must encode to one distinct token"
                        )));
                    }
                    let full = if layout.capture_trace || layout.contextual_tokens {
                        self.native.tokenize(&format!("{}{candidate}", slot.prefix), false, false)?
                    } else { Vec::new() };
                    let token = if layout.contextual_tokens {
                        let (&token, fixed) = full.split_last().ok_or(Error::Tokenization)?;
                        if candidates.is_empty() { prefix = fixed.to_vec(); }
                        else if fixed != prefix {
                            return Err(Error::InvalidInput("Candidates must share a single final token in their answer context".into()));
                        }
                        token
                    } else { tokens[0] };
                    if candidates.contains(&token) {
                        return Err(Error::InvalidInput("Candidates must have distinct token IDs in their answer context".into()));
                    }
                    candidates.push(token);
                    if layout.capture_trace {
                        let mut segmented = prefix.clone();
                        segmented.push(token);
                        context_matches.push(full == segmented);
                    }
                }
                Ok(PreparedSlot {
                    prefix,
                    candidates,
                    context_matches,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let capacity = 64.min(self.native.batch_size);
        if layout.pad_to > capacity
            || (layout.pad_to > 0 && !capacity.is_multiple_of(layout.pad_to))
        {
            return Err(Error::InvalidInput(
                "Canvas padding must divide canvas capacity".into(),
            ));
        }
        let frame = CanvasFrame {
            prefix: self.native.tokenize(&layout.canvas_prefix, false, true)?,
            suffix: self.native.tokenize(&layout.canvas_suffix, false, true)?,
            pad_to: layout.pad_to,
            steps: options.steps,
            capture_entropy: layout.adaptive_samples,
            capture_trace: layout.capture_trace,
        };
        let overhead = frame.prefix.len() + frame.suffix.len();
        if overhead >= capacity {
            return Err(Error::InvalidInput(
                "Canvas frame leaves no room for answers".into(),
            ));
        }
        let lengths: Vec<_> = prepared.iter().map(|s| s.prefix.len() + 1).collect();
        let groups = chunk_ranges(&lengths, capacity - overhead)?;
        if layout.contextual_tokens {
            // Check whole groups as well as individual prefixes: BPE may cross slot boundaries.
            for range in &groups {
                let mut expected = Vec::new();
                let mut positions = Vec::new();
                for slot in &prepared[range.clone()] {
                    expected.extend(&slot.prefix);
                    positions.push(expected.len());
                    expected.push(slot.candidates[0]);
                }
                for (index, slot) in request.slots[range.clone()].iter().enumerate() {
                    for (candidate_index, candidate) in slot.candidates.iter().enumerate() {
                        let mut text = String::new();
                        for (other_index, other) in request.slots[range.clone()].iter().enumerate()
                        {
                            text.push_str(&other.prefix);
                            text.push_str(if other_index == index {
                                candidate
                            } else {
                                &other.candidates[0]
                            });
                        }
                        let mut variant = expected.clone();
                        variant[positions[index]] =
                            prepared[range.start + index].candidates[candidate_index];
                        if self.native.tokenize(&text, false, false)? != variant {
                            return Err(Error::InvalidInput("Answer labels must each occupy one distinct slot in the complete template".into()));
                        }
                    }
                }
            }
        }
        let canvas_reserve = if options.sequential {
            lengths.iter().sum()
        } else {
            groups
                .iter()
                .map(|r| lengths[r.clone()].iter().sum::<usize>())
                .max()
                .unwrap_or(0)
        };
        let canvas_reserve = frame.padded_len(canvas_reserve + overhead);
        let prefill_suffix = self.native.tokenize(&layout.prefill_suffix, false, true)?;
        let thought_reserve = prefill_suffix.len()
            + if options.think > 0 {
                options.think
                    + self
                        .native
                        .tokenize("<|channel>thought\n<channel|>", false, true)?
                        .len()
            } else {
                0
            };
        if base_length + canvas_reserve + thought_reserve > self.native.n_ctx {
            return Err(Error::InvalidInput(format!(
                "Prompt, thought budget, and canvas need {} tokens; context allows {}",
                base_length + canvas_reserve + thought_reserve,
                self.native.n_ctx
            )));
        }
        let mut result = ReadResult {
            slots: Vec::new(),
            prompt_tokens: 0,
            canvas_tokens: 0,
            output_tokens: 0,
            seed,
            forward_ms: 0.0,
            traces: Vec::new(),
            profile: None,
        };
        let mut profile = ReadProfile::default();
        let mut batch = Batch::new(self.native.batch_size)?;
        let mut suffix = prefill_suffix;
        if options.think > 0 {
            let thought = self.think(&parts, options.think, seed, &mut batch)?;
            suffix = thought.suffix;
            result.prompt_tokens += thought.input_tokens;
            result.output_tokens = thought.output_tokens;
            result.forward_ms += thought.forward_ms;
            profile.add(&thought.profile);
        }
        for (group_index, range) in groups.iter().enumerate() {
            let group = &prepared[range.clone()];
            let prefill_start = Instant::now();
            let prompt_length = self.native.prefill(&parts, &suffix, &mut batch)?;
            profile.prefill_ms += prefill_start.elapsed().as_secs_f64() * 1000.0;
            let group_seed = seed.wrapping_add(104729_u64.wrapping_mul(group_index as u64));
            let mut reads = Vec::new();
            let samples = if layout.adaptive_samples {
                4
            } else {
                options.samples
            };
            for sample in 0..samples {
                let sample_seed = group_seed.wrapping_add(7919_u64.wrapping_mul(sample as u64));
                let read =
                    self.read_canvas(group, prompt_length, sample_seed, &mut batch, &frame)?;
                result.prompt_tokens += prompt_length;
                result.canvas_tokens += read.canvas_tokens;
                result.forward_ms += read.forward_ms;
                profile.add(&read.profile);
                if let Some(initial_canvas) = read.initial_canvas {
                    let mut prompt_token_ids = Vec::new();
                    for part in &parts {
                        if let PromptPart::Text(tokens) = part {
                            prompt_token_ids.extend(tokens);
                        }
                    }
                    prompt_token_ids.extend(&suffix);
                    result.traces.push(ReadTrace {
                        prompt_token_ids,
                        initial_canvas,
                        slot_positions: read.slots.iter().map(|s| s.canvas_position).collect(),
                        candidate_tokens: read
                            .slots
                            .iter()
                            .map(|s| s.candidate_tokens.clone())
                            .collect(),
                        candidates_match_context: group
                            .iter()
                            .map(|s| s.context_matches.clone())
                            .collect(),
                        partial_entropy: layout.adaptive_samples.then_some(read.entropy),
                    });
                }
                let stop = layout.adaptive_samples && sample == 0 && read.entropy <= 0.1;
                reads.push(read.slots);
                if stop {
                    break;
                }
            }
            let averaged = average_reads(reads)?;
            if options.sequential {
                for (slot, read) in group.iter().zip(&averaged) {
                    suffix.extend(&slot.prefix);
                    suffix.push(slot.candidates[best_index(&read.probabilities)]);
                }
            }
            result.slots.extend(averaged);
        }
        result.profile = layout.capture_trace.then_some(profile);
        Ok(result)
    }

    fn read_canvas(
        &mut self,
        group: &[PreparedSlot],
        prompt_length: usize,
        seed: u64,
        batch: &mut Batch,
        frame: &CanvasFrame,
    ) -> Result<CanvasRead> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut canvas = frame.prefix.clone();
        let mut slots = Vec::new();
        for slot in group {
            canvas.extend(&slot.prefix);
            let position = canvas.len();
            let initial_token = denoise::noise(self.native.n_vocab, self.native.mask, &mut rng);
            canvas.push(initial_token);
            slots.push(SlotRead {
                canvas_position: position,
                absolute_position: prompt_length + position,
                initial_token,
                candidate_tokens: slot.candidates.clone(),
                logits: Vec::new(),
                probabilities: Vec::new(),
            });
        }
        canvas.extend(&frame.suffix);
        canvas.resize(frame.padded_len(canvas.len()), 0);
        let initial_canvas = frame.capture_trace.then(|| canvas.clone());
        let positions: Vec<_> = slots.iter().map(|s| s.canvas_position).collect();
        let mut previous = None;
        let mut inverse_temperature = 1.0;
        let mut entropy: f64 = 0.0;
        let mut profile = ReadProfile::default();
        let start = Instant::now();
        for step in 0..frame.steps {
            let decode_start = Instant::now();
            self.native.decode_canvas(
                batch,
                &canvas,
                prompt_length,
                previous.as_deref(),
                inverse_temperature,
            )?;
            profile.decode_ms += decode_start.elapsed().as_secs_f64() * 1000.0;
            profile.decode_calls += 1;
            if step + 1 == frame.steps {
                if frame.capture_entropy {
                    let copy_start = Instant::now();
                    let logits = self.native.all_logits()?;
                    profile.logits_copy_ms += copy_start.elapsed().as_secs_f64() * 1000.0;
                    let entropy_start = Instant::now();
                    let vocab = self.native.n_vocab as usize;
                    for slot in &slots {
                        let row = &logits
                            [slot.canvas_position * vocab..(slot.canvas_position + 1) * vocab];
                        entropy = entropy.max(crate::probability::partial_entropy(
                            row,
                            &slot.candidate_tokens,
                        )?);
                    }
                    profile.entropy_ms += entropy_start.elapsed().as_secs_f64() * 1000.0;
                }
                for slot in &mut slots {
                    slot.logits = self
                        .native
                        .logits(slot.canvas_position, &slot.candidate_tokens)?;
                    slot.probabilities = restricted_softmax(&slot.logits)?;
                }
            } else {
                let copy_start = Instant::now();
                let logits = self.native.all_logits()?;
                profile.logits_copy_ms += copy_start.elapsed().as_secs_f64() * 1000.0;
                let temperature = 0.4 + 0.4 * (frame.steps - step) as f64 / frame.steps as f64;
                let sampling_start = Instant::now();
                denoise::refine(
                    &mut canvas,
                    &logits,
                    &positions,
                    self.native.n_vocab as usize,
                    temperature,
                    self.native.mask,
                    &mut rng,
                )?;
                profile.sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.0;
                previous = Some(logits);
                inverse_temperature = (1.0 / temperature) as f32;
            }
        }
        Ok(CanvasRead {
            slots,
            canvas_tokens: canvas.len(),
            forward_ms: start.elapsed().as_secs_f64() * 1000.0,
            entropy,
            initial_canvas,
            profile,
        })
    }

    fn think(
        &mut self,
        parts: &[PromptPart],
        budget: usize,
        seed: u64,
        batch: &mut Batch,
    ) -> Result<Thought> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut suffix = self.native.tokenize("<|channel>thought\n", false, true)?;
        let close = self.native.tokenize("<channel|>", false, true)?;
        let turn_close = self.native.tokenize("<turn|>", false, true)?;
        if close.len() != 1 || turn_close.len() != 1 {
            return Err(Error::UnsupportedModel);
        }
        let mut output_tokens = 0;
        let mut input_tokens = 0;
        let mut forward_ms = 0.0;
        let mut profile = ReadProfile::default();
        while output_tokens < budget {
            let prefill_start = Instant::now();
            let prompt_length = self.native.prefill(parts, &suffix, batch)?;
            profile.prefill_ms += prefill_start.elapsed().as_secs_f64() * 1000.0;
            input_tokens += prompt_length;
            let count = 64.min(self.native.batch_size).min(budget - output_tokens);
            let mut canvas: Vec<_> = (0..count)
                .map(|_| denoise::noise(self.native.n_vocab, self.native.mask, &mut rng))
                .collect();
            let positions: Vec<_> = (0..count).collect();
            let mut previous = None;
            let mut previous_best = Vec::new();
            let mut inverse_temperature = 1.0;
            let start = Instant::now();
            for step in 0..48 {
                let decode_start = Instant::now();
                self.native.decode_canvas(
                    batch,
                    &canvas,
                    prompt_length,
                    previous.as_deref(),
                    inverse_temperature,
                )?;
                profile.decode_ms += decode_start.elapsed().as_secs_f64() * 1000.0;
                profile.decode_calls += 1;
                let copy_start = Instant::now();
                let logits = self.native.all_logits()?;
                profile.logits_copy_ms += copy_start.elapsed().as_secs_f64() * 1000.0;
                let temperature = 0.4 + 0.4 * (48 - step) as f64 / 48.0;
                let sampling_start = Instant::now();
                let predictions = denoise::refine(
                    &mut canvas,
                    &logits,
                    &positions,
                    self.native.n_vocab as usize,
                    temperature,
                    self.native.mask,
                    &mut rng,
                )?;
                profile.sampling_ms += sampling_start.elapsed().as_secs_f64() * 1000.0;
                let best: Vec<_> = predictions.iter().map(|p| p.best).collect();
                let stable = best == previous_best
                    && predictions.iter().map(|p| p.entropy).sum::<f64>() / (count as f64) < 0.005;
                previous_best = best;
                if stable {
                    break;
                }
                previous = Some(logits);
                inverse_temperature = (1.0 / temperature) as f32;
            }
            forward_ms += start.elapsed().as_secs_f64() * 1000.0;
            let stop = previous_best
                .iter()
                .position(|t| *t == close[0] || *t == turn_close[0]);
            let length = stop.unwrap_or(previous_best.len());
            suffix.extend(&previous_best[..length]);
            output_tokens += length + usize::from(stop.is_some());
            if stop.is_some() {
                break;
            }
        }
        suffix.extend(close);
        Ok(Thought {
            suffix,
            input_tokens,
            output_tokens,
            forward_ms,
            profile,
        })
    }
}

struct PreparedSlot {
    prefix: Vec<i32>,
    candidates: Vec<i32>,
    context_matches: Vec<bool>,
}
struct CanvasFrame {
    prefix: Vec<i32>,
    suffix: Vec<i32>,
    pad_to: usize,
    steps: usize,
    capture_entropy: bool,
    capture_trace: bool,
}
struct CanvasRead {
    slots: Vec<SlotRead>,
    canvas_tokens: usize,
    forward_ms: f64,
    entropy: f64,
    initial_canvas: Option<Vec<i32>>,
    profile: ReadProfile,
}
impl CanvasFrame {
    fn padded_len(&self, length: usize) -> usize {
        if self.pad_to == 0 {
            length
        } else {
            length.div_ceil(self.pad_to) * self.pad_to
        }
    }
}
struct Thought {
    suffix: Vec<i32>,
    input_tokens: usize,
    output_tokens: usize,
    forward_ms: f64,
    profile: ReadProfile,
}

fn chunk_ranges(lengths: &[usize], capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    let mut ranges = Vec::new();
    let (mut start, mut length) = (0, 0);
    for (i, &size) in lengths.iter().enumerate() {
        if size > capacity {
            return Err(Error::InvalidInput(format!(
                "A question's {size}-token template exceeds the {capacity}-token canvas"
            )));
        }
        if length + size > capacity {
            ranges.push(start..i);
            start = i;
            length = 0;
        }
        length += size;
    }
    if start < lengths.len() {
        ranges.push(start..lengths.len());
    }
    Ok(ranges)
}

fn best_index(probabilities: &[f64]) -> usize {
    probabilities.iter().enumerate().fold(
        0,
        |best, (i, p)| if *p > probabilities[best] { i } else { best },
    )
}

fn average_reads(mut reads: Vec<Vec<SlotRead>>) -> Result<Vec<SlotRead>> {
    let count = reads.len();
    if count == 0 {
        return Err(Error::InvalidLogits);
    }
    let mut means = reads.remove(0);
    for read in reads {
        if read.len() != means.len() {
            return Err(Error::InvalidLogits);
        }
        for (mean, slot) in means.iter_mut().zip(read) {
            if mean.candidate_tokens != slot.candidate_tokens
                || mean.probabilities.len() != slot.probabilities.len()
            {
                return Err(Error::InvalidLogits);
            }
            for (sum, p) in mean.probabilities.iter_mut().zip(slot.probabilities) {
                *sum += p;
            }
        }
    }
    for mean in &mut means {
        for p in &mut mean.probabilities {
            *p /= count as f64;
        }
        // An averaged distribution has no single native logit row.
        if count > 1 {
            mean.logits.clear();
        }
    }
    Ok(means)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_padding_and_chunking_keep_every_answer_within_capacity() {
        let frame = CanvasFrame {
            prefix: vec![100, 7, 101],
            suffix: vec![106],
            pad_to: 16,
            steps: 1,
            capture_entropy: false,
            capture_trace: false,
        };
        assert_eq!(frame.padded_len(4 + 12), 16);
        assert_eq!(frame.padded_len(4 + 13), 32);
        let lengths = [30, 30, 1];
        let groups = chunk_ranges(&lengths, 64 - frame.prefix.len() - frame.suffix.len()).unwrap();
        assert_eq!(groups, vec![0..2, 2..3]);
        for group in groups {
            assert!(frame.padded_len(4 + lengths[group].iter().sum::<usize>()) <= 64);
        }
        assert!(chunk_ranges(&[61], 60).is_err());
    }

    #[test]
    fn chunking_preserves_order_and_keeps_whole_questions() {
        assert_eq!(
            chunk_ranges(&[12, 20, 32, 1, 63], 64).unwrap(),
            vec![0..3, 3..5]
        );
        assert_eq!(chunk_ranges(&[64, 64], 64).unwrap(), vec![0..1, 1..2]);
        assert!(chunk_ranges(&[65], 64).is_err());
    }

    #[test]
    fn samples_average_probabilities_instead_of_logits_or_winning_labels() {
        let slot = |probabilities: Vec<f64>| SlotRead {
            canvas_position: 0,
            absolute_position: 1,
            initial_token: 7,
            candidate_tokens: vec![1, 2],
            logits: vec![9.0, 1.0],
            probabilities,
        };
        let means = average_reads(vec![
            vec![slot(vec![0.99, 0.01])],
            vec![slot(vec![0.25, 0.75])],
        ])
        .unwrap();
        assert_eq!(means[0].probabilities, vec![0.62, 0.38]);
        assert!(means[0].logits.is_empty());
    }

    #[test]
    #[ignore = "Requires DIFFUSION_MODEL and a working native backend"]
    fn native_extensions_average_refine_think_and_chunk() {
        let config = ModelConfig::new(std::env::var("DIFFUSION_MODEL").unwrap());
        let mut engine = Engine::load(&config).unwrap();
        let request = ReadRequest::scm("Ground granulated blast furnace slag is used in concrete.");
        let first = engine.read(&request, 42).unwrap();
        let second = engine.read(&request, 42 + 7919).unwrap();
        let averaged = engine
            .read_with_options(
                &request,
                42,
                ReadOptions {
                    samples: 2,
                    ..Default::default()
                },
                &[],
            )
            .unwrap();
        for (i, &p) in averaged.slots[0].probabilities.iter().enumerate() {
            assert!(
                (p - (first.slots[0].probabilities[i] + second.slots[0].probabilities[i]) / 2.0)
                    .abs()
                    < 1e-5
            );
        }
        assert_eq!(averaged.prompt_tokens, first.prompt_tokens * 2);
        assert_eq!(averaged.canvas_tokens, first.canvas_tokens * 2);
        let refined = engine
            .read_with_options(
                &request,
                42,
                ReadOptions {
                    steps: 3,
                    ..Default::default()
                },
                &[],
            )
            .unwrap();
        assert!((refined.slots[0].probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        let thought = engine
            .read_with_options(
                &request,
                42,
                ReadOptions {
                    think: 8,
                    ..Default::default()
                },
                &[],
            )
            .unwrap();
        assert!((1..=8).contains(&thought.output_tokens));
        assert!(thought.prompt_tokens > first.prompt_tokens);
        let many = ReadRequest {
            prompt: request.prompt.clone(),
            slots: vec![request.slots[0].clone(); 12],
        };
        for sequential in [false, true] {
            let read = engine
                .read_with_options(
                    &many,
                    42,
                    ReadOptions {
                        sequential,
                        ..Default::default()
                    },
                    &[],
                )
                .unwrap();
            assert_eq!(read.slots.len(), 12);
            assert_eq!(read.canvas_tokens, first.canvas_tokens * 12);
            assert!(
                read.slots
                    .iter()
                    .all(|slot| (slot.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-6)
            );
        }
        let largest_choice = ReadRequest {
            prompt: "Choose exactly one allowed answer code. Prefer A.".into(),
            slots: vec![crate::Slot {
                prefix: "Answer: ".into(),
                candidates: engine.codes()[..128].to_vec(),
            }],
        };
        let largest = engine.read(&largest_choice, 42).unwrap();
        assert_eq!(largest.slots[0].probabilities.len(), 128);
        assert!(
            largest.slots[0]
                .probabilities
                .iter()
                .all(|p| p.is_finite() && *p >= 0.0)
        );
        assert!((largest.slots[0].probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let repeated = engine.read(&request, 42).unwrap();
        for (a, b) in first.slots[0]
            .probabilities
            .iter()
            .zip(&repeated.slots[0].probabilities)
        {
            assert!(
                (a - b).abs() < 1e-5,
                "Extension state leaked into the next request"
            );
        }
        let error = engine
            .read_with_options(
                &request,
                42,
                ReadOptions::default(),
                &[ImageInput { bytes: vec![1] }],
            )
            .unwrap_err();
        assert!(error.to_string().contains("--mmproj"));
    }

    #[test]
    #[ignore = "Requires DIFFUSION_MODEL and DIFFUSION_MMPROJ"]
    fn native_images_prefill_and_preserve_text_reproducibility() {
        let mut config = ModelConfig::new(std::env::var("DIFFUSION_MODEL").unwrap());
        config.mmproj = Some(std::env::var("DIFFUSION_MMPROJ").unwrap().into());
        let mut engine = Engine::load(&config).unwrap();
        let request = ReadRequest::scm("Ground granulated blast furnace slag.");
        let before = engine.read(&request, 42).unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_pixel(224, 224, image::Rgb([255, 0, 0]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let question = ReadRequest {
            prompt: "What color is the image? A = red, B = blue".into(),
            slots: vec![crate::Slot {
                prefix: "Answer: ".into(),
                candidates: vec!["A".into(), "B".into()],
            }],
        };
        let image = ImageInput {
            bytes: png.into_inner(),
        };
        let read = engine
            .read_with_options(
                &question,
                42,
                ReadOptions {
                    steps: 2,
                    samples: 2,
                    ..Default::default()
                },
                &[image],
            )
            .unwrap();
        assert!(read.prompt_tokens > 100);
        assert!(read.slots[0].probabilities[0] > read.slots[0].probabilities[1]);
        let after = engine.read(&request, 42).unwrap();
        for (a, b) in before.slots[0]
            .probabilities
            .iter()
            .zip(&after.slots[0].probabilities)
        {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    #[ignore = "Requires DIFFUSION_MODEL and a working native backend"]
    fn native_reads_preserve_reproducibility_across_requests() {
        let config = ModelConfig::new(std::env::var("DIFFUSION_MODEL").unwrap());
        let mut engine = Engine::load(&config).unwrap();
        let request = ReadRequest::scm("Ground granulated blast furnace slag is used in concrete.");
        let first = engine.read(&request, 42).unwrap();
        let other = ReadRequest::scm(
            "This is a different and longer material description: steel reinforcement bars carry tensile forces in a reinforced concrete structure.",
        );
        engine.read(&other, 7).unwrap();
        let optimized = engine
            .read_with_layout(
                &request,
                42,
                ReadOptions::default(),
                &[],
                &ReadLayout {
                    prefill_suffix: EMPTY_THOUGHT_CHANNEL.into(),
                    skip_zero_self_conditioning: true,
                    capture_trace: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(optimized.traces.len(), 1);
        let profile = optimized.profile.as_ref().unwrap();
        assert_eq!(profile.decode_calls, 1);
        assert!(profile.prefill_ms > 0.0 && profile.decode_ms > 0.0);
        assert_eq!(profile.sampling_ms, 0.0);
        let trace = &optimized.traces[0];
        let header = engine
            .native
            .tokenize(EMPTY_THOUGHT_CHANNEL, false, true)
            .unwrap();
        assert!(trace.prompt_token_ids.ends_with(&header));
        assert_eq!(first.output_tokens, 0);
        assert_eq!(trace.prompt_token_ids.len(), optimized.prompt_tokens);
        assert_eq!(trace.initial_canvas.len(), optimized.canvas_tokens);
        // The legacy split loses the natural leading-space label token.
        assert!(trace.candidates_match_context.iter().flatten().any(|v| !v));
        for (index, slot) in optimized.slots.iter().enumerate() {
            assert_eq!(trace.slot_positions[index], slot.canvas_position);
            assert_eq!(
                trace.initial_canvas[slot.canvas_position],
                slot.initial_token
            );
            assert_eq!(trace.candidate_tokens[index], slot.candidate_tokens);
        }
        for (a, b) in first.slots[0]
            .probabilities
            .iter()
            .zip(&optimized.slots[0].probabilities)
        {
            assert!(
                (a - b).abs() < 1e-5,
                "Zero-SC bypass changed probabilities: {a} != {b}"
            );
        }
        let repeated = engine.read(&request, 42).unwrap();
        assert!(repeated.traces.is_empty());
        assert!(repeated.profile.is_none());
        let contextual = engine
            .read_with_layout(
                &request,
                42,
                ReadOptions::default(),
                &[],
                &ReadLayout {
                    contextual_tokens: true,
                    capture_trace: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            contextual.traces[0]
                .candidates_match_context
                .iter()
                .flatten()
                .all(|v| *v)
        );
        assert_ne!(
            contextual.slots[0].candidate_tokens,
            first.slots[0].candidate_tokens
        );
        assert_eq!(contextual.canvas_tokens + 1, first.canvas_tokens);
        assert_eq!(
            first.slots[0].initial_token,
            repeated.slots[0].initial_token
        );
        assert_eq!(first.slots[0].candidate_tokens.len(), 2);
        assert_eq!(first.canvas_tokens, 12);
        assert_eq!(first.slots[0].canvas_position, 11);
        for (a, b) in first.slots[0]
            .probabilities
            .iter()
            .zip(&repeated.slots[0].probabilities)
        {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }
}
