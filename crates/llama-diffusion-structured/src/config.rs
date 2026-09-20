//! Model loading and context configuration.

use crate::{Error, Result};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub gpu_layers: i32,
    pub main_gpu: i32,
    pub context_size: u32,
    pub batch_size: u32,
    pub threads: i32,
    pub flash_attention: bool,
}

impl ModelConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            mmproj: None,
            gpu_layers: -1,
            main_gpu: 0,
            context_size: 4096,
            batch_size: 512,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get().min(8) as i32),
            flash_attention: false,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.context_size == 0
            || self.context_size > i32::MAX as u32
            || self.batch_size == 0
            || self.batch_size > self.context_size
            || self.threads < 1
            || self.main_gpu < 0
        {
            return Err(Error::InvalidInput("Require 0 < batch_size <= context_size <= i32::MAX, positive threads, and a nonnegative GPU index".into()));
        }
        Ok(())
    }
}
