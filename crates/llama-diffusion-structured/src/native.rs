use crate::{Error, ModelConfig, Result};
use llama_diffusion_sys as ffi;
use std::{ffi::CString, ptr::NonNull, sync::Mutex};

static BACKEND_USERS: Mutex<usize> = Mutex::new(0);

struct Backend;

impl Backend {
    fn acquire() -> Result<Self> {
        let mut users = BACKEND_USERS.lock().map_err(|_| Error::BackendPoisoned)?;
        if *users == 0 {
            // SAFETY: backend initialization and teardown are serialized by this mutex.
            unsafe { ffi::llama_backend_init() };
        }
        *users += 1;
        Ok(Self)
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let mut users = BACKEND_USERS.lock().unwrap_or_else(|e| e.into_inner());
        *users -= 1;
        if *users == 0 {
            // SAFETY: all models and contexts owned by these guards have been freed.
            unsafe { ffi::llama_backend_free() };
        }
    }
}

struct Model {
    ptr: NonNull<ffi::llama_model>,
    _backend: Backend,
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: this is the unique owner; Native drops its context first.
        unsafe { ffi::llama_model_free(self.ptr.as_ptr()) };
    }
}

struct Context(NonNull<ffi::llama_context>);

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the context is owned here and its model is still alive.
        unsafe {
            ffi::llama_synchronize(self.0.as_ptr());
            ffi::llama_free(self.0.as_ptr());
        }
    }
}

pub(crate) struct Batch {
    raw: ffi::llama_batch,
    capacity: usize,
}

impl Batch {
    pub fn new(capacity: usize) -> Result<Self> {
        let capacity_i32 = i32::try_from(capacity)
            .map_err(|_| Error::InvalidInput("Batch is too large".into()))?;
        // SAFETY: capacity is positive, checked by ModelConfig, and one sequence is used.
        let raw = unsafe { ffi::llama_batch_init(capacity_i32, 0, 1) };
        Ok(Self { raw, capacity })
    }

    fn fill(&mut self, tokens: &[i32], offset: usize, all_logits: bool) -> Result<()> {
        if tokens.is_empty() || tokens.len() > self.capacity {
            return Err(Error::InvalidInput("Invalid batch length".into()));
        }
        let end = offset
            .checked_add(tokens.len())
            .ok_or_else(|| Error::InvalidInput("Position overflow".into()))?;
        i32::try_from(end).map_err(|_| Error::InvalidInput("Position overflow".into()))?;
        self.raw.n_tokens = tokens.len() as i32;
        for (i, &token) in tokens.iter().enumerate() {
            // SAFETY: llama_batch_init allocated all arrays for capacity entries and one sequence.
            unsafe {
                *self.raw.token.add(i) = token;
                *self.raw.pos.add(i) = (offset + i) as i32;
                *self.raw.n_seq_id.add(i) = 1;
                *(*self.raw.seq_id.add(i)) = 0;
                *self.raw.logits.add(i) = i8::from(all_logits || i == tokens.len() - 1);
            }
        }
        Ok(())
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // SAFETY: the batch allocation is owned here and decoding no longer uses its arrays.
        unsafe { ffi::llama_batch_free(self.raw) };
    }
}

/// Raw pointers deliberately keep the model and its mutable phase state !Send and !Sync.
pub(crate) struct Native {
    context: Context,
    model: Model,
    vocab: NonNull<ffi::llama_vocab>,
    pub n_vocab: i32,
    pub mask: i32,
    pub n_ctx: usize,
    pub batch_size: usize,
    output_rows: usize,
}

impl Native {
    pub fn load(config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        let backend = Backend::acquire()?;
        let path = CString::new(config.model.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::InvalidInput("Model path contains a NUL byte".into()))?;
        // SAFETY: default structs come from the linked library; the path lives through loading.
        let model = unsafe {
            let mut params = ffi::llama_model_default_params();
            params.n_gpu_layers = config.gpu_layers;
            params.split_mode = ffi::llama_split_mode_LLAMA_SPLIT_MODE_NONE;
            params.main_gpu = config.main_gpu;
            let ptr = NonNull::new(ffi::llama_model_load_from_file(path.as_ptr(), params))
                .ok_or(Error::ModelLoad)?;
            Model {
                ptr,
                _backend: backend,
            }
        };
        // SAFETY: model is live; these functions only inspect its architecture and vocabulary.
        let (vocab, n_vocab, mask) = unsafe {
            if !ffi::llama_model_is_diffusion(model.ptr.as_ptr())
                || ffi::llama_diffusion_pkv_bytes_per_token(model.ptr.as_ptr(), false) == 0
            {
                return Err(Error::UnsupportedModel);
            }
            let vocab = NonNull::new(ffi::llama_model_get_vocab(model.ptr.as_ptr()).cast_mut())
                .ok_or(Error::UnsupportedModel)?;
            (
                vocab,
                ffi::llama_vocab_n_tokens(vocab.as_ptr()),
                ffi::llama_vocab_mask(vocab.as_ptr()),
            )
        };
        if n_vocab < 2 {
            return Err(Error::UnsupportedModel);
        }
        // SAFETY: model stays alive past context destruction; SC is reserved before context creation.
        let context = unsafe {
            ffi::llama_diffusion_set_sc(model.ptr.as_ptr(), std::ptr::null(), 0.0, 1.0, true);
            let mut params = ffi::llama_context_default_params();
            params.n_ctx = config.context_size;
            params.n_batch = config.batch_size;
            params.n_ubatch = config.batch_size;
            params.n_threads = config.threads;
            params.n_threads_batch = config.threads;
            params.flash_attn_type = if config.flash_attention {
                ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED
            } else {
                ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED
            };
            let ptr = NonNull::new(ffi::llama_init_from_model(model.ptr.as_ptr(), params))
                .ok_or(Error::ContextCreation)?;
            ffi::llama_set_causal_attn(ptr.as_ptr(), false);
            Context(ptr)
        };
        // SAFETY: context is initialized and owned by this object.
        let (n_ctx, batch_size) = unsafe {
            (
                ffi::llama_n_ctx(context.0.as_ptr()) as usize,
                ffi::llama_n_batch(context.0.as_ptr()).min(ffi::llama_n_ubatch(context.0.as_ptr()))
                    as usize,
            )
        };
        Ok(Self {
            context,
            model,
            vocab,
            n_vocab,
            mask,
            n_ctx,
            batch_size,
            output_rows: 0,
        })
    }

    pub fn tokenize(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let length = i32::try_from(text.len()).map_err(|_| Error::Tokenization)?;
        // SAFETY: the string has length bytes; a null output with zero capacity queries the size.
        let needed = unsafe {
            ffi::llama_tokenize(
                self.vocab.as_ptr(),
                text.as_ptr().cast(),
                length,
                std::ptr::null_mut(),
                0,
                add_special,
                parse_special,
            )
        };
        let count = needed.checked_abs().ok_or(Error::Tokenization)?;
        let mut tokens = vec![0; count as usize];
        if count == 0 {
            return Ok(tokens);
        }
        // SAFETY: the output vector has the exact capacity requested by the tokenizer.
        let written = unsafe {
            ffi::llama_tokenize(
                self.vocab.as_ptr(),
                text.as_ptr().cast(),
                length,
                tokens.as_mut_ptr(),
                count,
                add_special,
                parse_special,
            )
        };
        if written < 0 || written > count {
            return Err(Error::Tokenization);
        }
        tokens.truncate(written as usize);
        Ok(tokens)
    }

    pub fn code_piece(&self, token: i32) -> Option<String> {
        let mut bytes = [0_u8; 16];
        // SAFETY: callers enumerate valid vocabulary IDs; the buffer is writable for 16 bytes.
        let count = unsafe {
            if ffi::llama_vocab_is_control(self.vocab.as_ptr(), token) {
                return None;
            }
            ffi::llama_token_to_piece(
                self.vocab.as_ptr(),
                token,
                bytes.as_mut_ptr().cast(),
                16,
                0,
                false,
            )
        };
        if !(1..=16).contains(&count)
            || !bytes[..count as usize]
                .iter()
                .all(u8::is_ascii_alphanumeric)
        {
            return None;
        }
        String::from_utf8(bytes[..count as usize].to_vec()).ok()
    }

    pub fn prefill_phase(&mut self, prompt_length: usize, offset: usize) {
        self.output_rows = 0;
        // SAFETY: validated token counts fit i32; exclusive access prevents phase races.
        unsafe {
            ffi::llama_diffusion_set_sc(self.model.ptr.as_ptr(), std::ptr::null(), 0.0, 1.0, false);
            ffi::llama_diffusion_set_phase(
                self.model.ptr.as_ptr(),
                1,
                prompt_length as i32,
                offset as i32,
            );
        }
    }

    pub fn canvas_phase(&mut self, prompt_length: usize) {
        // SAFETY: prefill has populated this model's cache, and access is exclusive.
        unsafe {
            ffi::llama_diffusion_set_phase(self.model.ptr.as_ptr(), 2, prompt_length as i32, 0);
            ffi::llama_diffusion_set_sc(self.model.ptr.as_ptr(), std::ptr::null(), 0.0, 1.0, true);
        }
    }

    pub fn decode(
        &mut self,
        batch: &mut Batch,
        tokens: &[i32],
        offset: usize,
        all_logits: bool,
    ) -> Result<()> {
        if tokens
            .iter()
            .any(|&token| token < 0 || token >= self.n_vocab)
        {
            return Err(Error::InvalidInput(
                "Token is outside the vocabulary".into(),
            ));
        }
        batch.fill(tokens, offset, all_logits)?;
        self.output_rows = 0;
        // SAFETY: the populated batch owns valid arrays and tokens; context access is exclusive.
        let status = unsafe { ffi::llama_decode(self.context.0.as_ptr(), batch.raw) };
        if status != 0 {
            return Err(Error::Decode(status));
        }
        if all_logits {
            self.output_rows = tokens.len();
        }
        Ok(())
    }

    pub fn synchronize(&mut self) {
        // SAFETY: this object owns the live context.
        unsafe { ffi::llama_synchronize(self.context.0.as_ptr()) };
    }

    pub fn logits(&mut self, position: usize, candidates: &[i32]) -> Result<Vec<f64>> {
        if position >= self.output_rows || candidates.iter().any(|&t| t < 0 || t >= self.n_vocab) {
            return Err(Error::InvalidInput(
                "Invalid logit row or candidate token".into(),
            ));
        }
        // SAFETY: all canvas rows were requested, the indices are checked, and values are copied before reuse.
        unsafe {
            let logits = ffi::llama_get_logits(self.context.0.as_ptr());
            if logits.is_null() {
                return Err(Error::MissingLogits);
            }
            let row = logits.add(position * self.n_vocab as usize);
            Ok(candidates
                .iter()
                .map(|&token| f64::from(*row.add(token as usize)))
                .collect())
        }
    }
}
