//! Restricted-canvas inputs and the diagnostics returned by a read.

use serde::{Deserialize, Serialize};

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
    pub seed: u64,
    pub forward_ms: f64,
}
