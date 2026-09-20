//! Raw bindings generated from the same llama.cpp sources as the linked library.
//! By default, Cargo builds the pinned vendored submodule; GPU backends are opt-in.
//! No inference policy or resource ownership belongs in this crate.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Bundled build applies the DiffusionGemma image-prefill integration.
pub const IMAGE_PREFILL_SUPPORTED: bool =
    matches!(env!("DIFFUSION_IMAGE_PREFILL").as_bytes(), b"1");
