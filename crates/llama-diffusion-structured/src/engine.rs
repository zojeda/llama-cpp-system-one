//! Token preparation and the single-forward inference lifecycle.

use crate::native::{Batch, Native};
use crate::{Error, ModelConfig, ReadRequest, ReadResult, Result, SlotRead, restricted_softmax};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{collections::HashSet, time::Instant};

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
        if request.prompt.trim().is_empty() || request.slots.is_empty() {
            return Err(Error::InvalidInput(
                "A prompt and at least one slot are required".into(),
            ));
        }
        // Text-only DiffusionGemma user turn; user content cannot introduce control tokens.
        let mut prompt = self.native.tokenize("<|turn>user\n", true, true)?;
        prompt.extend(self.native.tokenize(request.prompt.trim(), false, false)?);
        prompt.extend(
            self.native
                .tokenize("<turn|>\n<|turn>model\n", false, true)?,
        );

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut canvas = Vec::new();
        let mut slots = Vec::with_capacity(request.slots.len());
        for slot in &request.slots {
            if slot.candidates.is_empty() {
                return Err(Error::InvalidInput(
                    "Each slot needs at least one candidate".into(),
                ));
            }
            let mut candidate_tokens = Vec::with_capacity(slot.candidates.len());
            for candidate in &slot.candidates {
                let tokens = self.native.tokenize(candidate, false, false)?;
                if tokens.len() != 1 || candidate_tokens.contains(&tokens[0]) {
                    return Err(Error::InvalidInput(format!(
                        "Candidate {candidate:?} must encode to exactly one distinct token (got {} tokens)",
                        tokens.len()
                    )));
                }
                candidate_tokens.push(tokens[0]);
            }
            canvas.extend(self.native.tokenize(&slot.prefix, false, false)?);
            let position = canvas.len();
            let noise = loop {
                let token = rng.random_range(0..self.native.n_vocab);
                if token != self.native.mask {
                    break token;
                }
            };
            canvas.push(noise);
            slots.push(SlotRead {
                canvas_position: position,
                absolute_position: prompt.len() + position,
                initial_token: noise,
                candidate_tokens,
                logits: Vec::new(),
                probabilities: Vec::new(),
            });
        }
        if prompt.len() + canvas.len() > self.native.n_ctx {
            return Err(Error::InvalidInput(format!(
                "Prompt and canvas need {} tokens, context allows {}",
                prompt.len() + canvas.len(),
                self.native.n_ctx
            )));
        }
        if canvas.len() > self.native.batch_size {
            return Err(Error::InvalidInput(format!(
                "The entire {}-token canvas must fit in one {}-token batch",
                canvas.len(),
                self.native.batch_size
            )));
        }

        let mut batch = Batch::new(self.native.batch_size)?;
        for (i, chunk) in prompt.chunks(self.native.batch_size).enumerate() {
            let offset = i * self.native.batch_size;
            self.native.prefill_phase(prompt.len(), offset);
            self.native.decode(&mut batch, chunk, offset, false)?;
        }
        self.native.synchronize();
        self.native.canvas_phase(prompt.len());
        let start = Instant::now();
        self.native
            .decode(&mut batch, &canvas, prompt.len(), true)?;
        for slot in &mut slots {
            slot.logits = self
                .native
                .logits(slot.canvas_position, &slot.candidate_tokens)?;
        }
        let forward_ms = start.elapsed().as_secs_f64() * 1000.0;
        for slot in &mut slots {
            slot.probabilities = restricted_softmax(&slot.logits)?;
        }
        Ok(ReadResult {
            slots,
            prompt_tokens: prompt.len(),
            canvas_tokens: canvas.len(),
            seed,
            forward_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let repeated = engine.read(&request, 42).unwrap();
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
