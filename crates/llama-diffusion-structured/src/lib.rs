//! A single restricted-canvas read, ported from llama.cpp's structured-cli.cpp.
//!
//! Configuration and read types form the public API. The engine handles token
//! preparation and inference phases; only the native module can use unsafe code.
#![deny(unsafe_code)]

mod config;
mod engine;
mod error;
#[allow(unsafe_code)]
mod native;
mod probability;
mod read;

pub use config::ModelConfig;
pub use engine::Engine;
pub use error::{Error, Result};
pub use probability::restricted_softmax;
pub use read::{ReadRequest, ReadResult, Slot, SlotRead};
