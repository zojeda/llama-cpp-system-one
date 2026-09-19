//! Errors shared by the safe inference API and native backend.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Could not load the model")]
    ModelLoad,
    #[error("A DiffusionGemma model with prompt KV caching is required")]
    UnsupportedModel,
    #[error("Could not create the inference context")]
    ContextCreation,
    #[error("Tokenization failed")]
    Tokenization,
    #[error("Native decode failed with status {0}")]
    Decode(i32),
    #[error("The canvas forward returned no logits")]
    MissingLogits,
    #[error("Candidate logits must be finite and nonempty")]
    InvalidLogits,
    #[error("The backend lifecycle lock is poisoned")]
    BackendPoisoned,
}
