//! Restricted-canvas inputs and the diagnostics returned by a read.

use serde::{Deserialize, Serialize};

/// Explicit formatting controls for reproducible inference experiments.
/// User-provided text remains tokenized without interpreting control markers.
/// `Default` is the unframed control; [`crate::Engine::read_with_options`] selects service defaults.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReadLayout {
    pub system_prompt: Option<String>,
    pub canvas_prefix: String,
    pub canvas_suffix: String,
    pub prefill_suffix: String,
    pub pad_to: usize,
    pub thinking_marker: bool,
    /// OpenJev policy: up to four total reads when top-20-plus-label entropy exceeds 0.1.
    pub adaptive_samples: bool,
    /// Experiment: bypass a self-conditioning branch whose contribution is multiplied by zero.
    pub skip_zero_self_conditioning: bool,
    /// Opt-in research diagnostics; contains input tokens and must not be logged by the service.
    pub capture_trace: bool,
    /// Resolve the answer token inside its complete textual template.
    pub contextual_tokens: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ReadOptions {
    pub steps: usize,
    pub samples: usize,
    pub think: usize,
    pub sequential: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            steps: 1,
            samples: 1,
            think: 0,
            sequential: false,
        }
    }
}

impl ReadOptions {
    pub(crate) fn validate(self) -> crate::Result<()> {
        if !(1..=8).contains(&self.steps) || !(1..=32).contains(&self.samples) || self.think > 4096
        {
            return Err(crate::Error::InvalidInput(
                "Require steps=1..8, samples=1..32, think=0..4096".into(),
            ));
        }
        Ok(())
    }
}

/// Compressed image bytes, decoded on the inference worker with allocation limits.
#[derive(Clone, Debug)]
pub struct ImageInput {
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Slot {
    /// Fixed canvas text immediately before this slot.
    pub prefix: String,
    /// Each candidate must encode to exactly one distinct token.
    pub candidates: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReadRequest {
    pub prompt: String,
    pub slots: Vec<Slot>,
}

impl ReadRequest {
    pub fn scm(material: &str) -> Self {
        Self {
            prompt: format!(
                "{material}\n\nSCM means supplementary cementitious material. Use these answer codes:\nA = yes\nB = no"
            ),
            slots: vec![Slot {
                prefix: "Is this material an SCM?\nAnswer: ".into(),
                candidates: vec!["A".into(), "B".into()],
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SlotRead {
    pub canvas_position: usize,
    pub absolute_position: usize,
    pub initial_token: i32,
    pub candidate_tokens: Vec<i32>,
    pub logits: Vec<f64>,
    pub probabilities: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct ReadResult {
    pub slots: Vec<SlotRead>,
    pub prompt_tokens: usize,
    pub canvas_tokens: usize,
    pub output_tokens: usize,
    pub seed: u64,
    pub forward_ms: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub traces: Vec<ReadTrace>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<ReadProfile>,
}

/// Wall-clock phase timings for opt-in research traces; no request text.
#[derive(Debug, Default, Serialize)]
pub struct ReadProfile {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub logits_copy_ms: f64,
    pub sampling_ms: f64,
    pub entropy_ms: f64,
    pub decode_calls: usize,
}

impl ReadProfile {
    pub(crate) fn add(&mut self, other: &Self) {
        self.prefill_ms += other.prefill_ms;
        self.decode_ms += other.decode_ms;
        self.logits_copy_ms += other.logits_copy_ms;
        self.sampling_ms += other.sampling_ms;
        self.entropy_ms += other.entropy_ms;
        self.decode_calls += other.decode_calls;
    }
}

#[derive(Debug, Serialize)]
pub struct ReadTrace {
    pub prompt_token_ids: Vec<i32>,
    pub initial_canvas: Vec<i32>,
    pub slot_positions: Vec<usize>,
    pub candidate_tokens: Vec<Vec<i32>>,
    pub candidates_match_context: Vec<Vec<bool>>,
    pub partial_entropy: Option<f64>,
}
