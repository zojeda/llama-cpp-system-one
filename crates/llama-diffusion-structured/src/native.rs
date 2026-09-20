use crate::{Error, ImageInput, ModelConfig, Result};
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

struct Vision(NonNull<ffi::mtmd_context>);
impl Drop for Vision {
    fn drop(&mut self) {
        // SAFETY: unique ownership; Native drops the projector before its text model.
        unsafe { ffi::mtmd_free(self.0.as_ptr()) };
    }
}
struct Bitmap(NonNull<ffi::mtmd_bitmap>);
impl Drop for Bitmap {
    fn drop(&mut self) {
        // SAFETY: the bitmap is uniquely owned and tokenization has finished.
        unsafe { ffi::mtmd_bitmap_free(self.0.as_ptr()) };
    }
}
struct Chunks(NonNull<ffi::mtmd_input_chunks>);
impl Drop for Chunks {
    fn drop(&mut self) {
        // SAFETY: chunk data has been copied and no outstanding references remain.
        unsafe { ffi::mtmd_input_chunks_free(self.0.as_ptr()) };
    }
}

pub(crate) enum PromptPart {
    Text(Vec<i32>),
    Image { embeddings: Vec<f32>, tokens: usize },
}
impl PromptPart {
    pub fn len(&self) -> usize {
        match self {
            Self::Text(tokens) => tokens.len(),
            Self::Image { tokens, .. } => *tokens,
        }
    }
}

/// Raw pointers deliberately keep the model and its mutable phase state !Send and !Sync.
pub(crate) struct Native {
    vision: Option<Vision>,
    n_embd: usize,
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
        let vision = if let Some(path) = &config.mmproj {
            if !ffi::IMAGE_PREFILL_SUPPORTED {
                return Err(Error::InvalidInput(
                    "Image input requires the bundled native build with the image-prefill overlay"
                        .into(),
                ));
            }
            let path = CString::new(path.as_os_str().as_encoded_bytes())
                .map_err(|_| Error::InvalidInput("Projector path contains NUL".into()))?;
            // SAFETY: model is live and outlives the projector; path stays valid through loading.
            let vision = unsafe {
                let mut params = ffi::mtmd_context_params_default();
                params.use_gpu = config.gpu_layers != 0;
                params.n_threads = config.threads;
                params.warmup = false;
                params.image_max_tokens = 280;
                let ptr = NonNull::new(ffi::mtmd_init_from_file(
                    path.as_ptr(),
                    model.ptr.as_ptr(),
                    params,
                ))
                .ok_or_else(|| {
                    Error::InvalidInput("Could not load a compatible vision projector".into())
                })?;
                Vision(ptr)
            };
            // SAFETY: vision is initialized. Only Gemma-style, ordinary-position image batches are supported.
            if unsafe {
                !ffi::mtmd_support_vision(vision.0.as_ptr())
                    || ffi::mtmd_decode_use_mrope(vision.0.as_ptr())
            } {
                return Err(Error::InvalidInput(
                    "Projector must support vision with ordinary token positions".into(),
                ));
            }
            Some(vision)
        } else {
            None
        };
        // SAFETY: inspect the live model's required input embedding width.
        let n_embd = unsafe { ffi::llama_model_n_embd_inp(model.ptr.as_ptr()) } as usize;
        Ok(Self {
            vision,
            n_embd,
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

    pub fn image_parts(&mut self, images: &[ImageInput]) -> Result<Vec<PromptPart>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        if images.len() > 8 {
            return Err(Error::InvalidInput("At most 8 images are allowed".into()));
        }
        let vision = self.vision.as_ref().ok_or_else(|| Error::InvalidInput("Image input requires --mmproj with a compatible DiffusionGemma vision projector".into()))?.0;
        let mut parts = Vec::new();
        for image in images {
            let rgb = crate::images::decode(image)?;
            // SAFETY: rgb is packed RGB8 with exactly width * height * 3 bytes; mtmd copies it.
            let bitmap = Bitmap(
                NonNull::new(unsafe {
                    ffi::mtmd_bitmap_init(rgb.width(), rgb.height(), rgb.as_ptr())
                })
                .ok_or_else(|| Error::InvalidInput("Could not allocate image bitmap".into()))?,
            );
            // SAFETY: creates an owned empty chunk list.
            let chunks = Chunks(
                NonNull::new(unsafe { ffi::mtmd_input_chunks_init() })
                    .ok_or_else(|| Error::InvalidInput("Could not allocate image chunks".into()))?,
            );
            let part = ffi::mtmd_input_part {
                text: std::ptr::null(),
                bitmap: bitmap.0.as_ptr(),
            };
            let ptr = &part as *const _;
            // SAFETY: every pointer remains live through synchronous tokenization; output is owned here.
            let status = unsafe {
                ffi::mtmd_tokenize_from_parts(vision.as_ptr(), chunks.0.as_ptr(), &ptr, 1, false)
            };
            if status != 0 {
                return Err(Error::InvalidInput("Image preprocessing failed".into()));
            }
            // SAFETY: chunks is a live, populated list; each chunk remains live until all data is copied.
            unsafe {
                for i in 0..ffi::mtmd_input_chunks_size(chunks.0.as_ptr()) {
                    let chunk = ffi::mtmd_input_chunks_get(chunks.0.as_ptr(), i);
                    match ffi::mtmd_input_chunk_get_type(chunk) {
                        ffi::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT => {
                            let mut length = 0;
                            let data = ffi::mtmd_input_chunk_get_tokens_text(chunk, &mut length);
                            if length > 0 {
                                if data.is_null() {
                                    return Err(Error::InvalidInput(
                                        "Missing image delimiter tokens".into(),
                                    ));
                                }
                                parts.push(PromptPart::Text(
                                    std::slice::from_raw_parts(data, length).to_vec(),
                                ));
                            }
                        }
                        ffi::mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_IMAGE => {
                            let tokens = ffi::mtmd_input_chunk_get_n_tokens(chunk);
                            if tokens == 0 || tokens > self.batch_size {
                                return Err(Error::InvalidInput(format!(
                                    "Image needs {tokens} tokens in one batch; increase --batch-size"
                                )));
                            }
                            if ffi::mtmd_encode_chunk(vision.as_ptr(), chunk) != 0 {
                                return Err(Error::InvalidInput("Vision encoder failed".into()));
                            }
                            let data = ffi::mtmd_get_output_embd(vision.as_ptr());
                            if data.is_null() {
                                return Err(Error::InvalidInput(
                                    "Vision encoder returned no embeddings".into(),
                                ));
                            }
                            let length = tokens.checked_mul(self.n_embd).ok_or_else(|| {
                                Error::InvalidInput("Image embedding size overflow".into())
                            })?;
                            parts.push(PromptPart::Image {
                                embeddings: std::slice::from_raw_parts(data, length).to_vec(),
                                tokens,
                            });
                        }
                        _ => return Err(Error::InvalidInput("Only images are supported".into())),
                    }
                }
            }
        }
        Ok(parts)
    }

    pub fn prefill(
        &mut self,
        parts: &[PromptPart],
        suffix: &[i32],
        batch: &mut Batch,
    ) -> Result<usize> {
        let length = parts.iter().map(PromptPart::len).sum::<usize>() + suffix.len();
        if length > self.n_ctx {
            return Err(Error::InvalidInput("Prompt exceeds context size".into()));
        }
        let mut offset = 0;
        for part in parts {
            match part {
                PromptPart::Text(tokens) => {
                    for chunk in tokens.chunks(self.batch_size) {
                        self.prefill_phase(length, offset);
                        self.decode(batch, chunk, offset, false)?;
                        offset += chunk.len();
                    }
                }
                PromptPart::Image { embeddings, tokens } => {
                    self.prefill_phase(length, offset);
                    self.decode_image(batch, embeddings, *tokens, offset)?;
                    offset += tokens;
                }
            }
        }
        for chunk in suffix.chunks(self.batch_size) {
            self.prefill_phase(length, offset);
            self.decode(batch, chunk, offset, false)?;
            offset += chunk.len();
        }
        self.synchronize();
        Ok(length)
    }

    fn decode_image(
        &mut self,
        batch: &mut Batch,
        embeddings: &[f32],
        tokens: usize,
        offset: usize,
    ) -> Result<()> {
        if tokens == 0 || tokens > self.batch_size || embeddings.len() != tokens * self.n_embd {
            return Err(Error::InvalidInput("Invalid image embeddings".into()));
        }
        batch.fill(&vec![0; tokens], offset, false)?;
        let mut raw = batch.raw;
        raw.token = std::ptr::null_mut();
        // The C API uses a mutable pointer but only reads input embeddings.
        raw.embd = embeddings.as_ptr().cast_mut();
        self.output_rows = 0;
        // SAFETY: raw metadata arrays live in batch and embeddings contains n_tokens * n_embd floats.
        // Decode uploads inputs synchronously; synchronize before releasing the borrowed image buffer.
        let status = unsafe { ffi::llama_decode(self.context.0.as_ptr(), raw) };
        self.synchronize();
        if status != 0 {
            return Err(Error::Decode(status));
        }
        Ok(())
    }

    pub fn decode_canvas(
        &mut self,
        batch: &mut Batch,
        tokens: &[i32],
        prompt_length: usize,
        previous: Option<&[f32]>,
        inverse_temperature: f32,
    ) -> Result<()> {
        if let Some(previous) = previous
            && previous.len() != tokens.len() * self.n_vocab as usize
        {
            return Err(Error::InvalidInput(
                "Invalid self-conditioning shape".into(),
            ));
        }
        self.canvas_phase(prompt_length);
        if let Some(previous) = previous {
            // SAFETY: previous contains exactly one vocabulary row per canvas position and lives through decode.
            unsafe {
                ffi::llama_diffusion_set_sc(
                    self.model.ptr.as_ptr(),
                    previous.as_ptr(),
                    1.0,
                    inverse_temperature,
                    true,
                )
            };
        }
        let result = self.decode(batch, tokens, prompt_length, true);
        self.synchronize();
        // Remove the borrowed pointer on both success and error before its slice can be dropped.
        self.canvas_phase(prompt_length);
        result
    }

    pub fn all_logits(&mut self) -> Result<Vec<f32>> {
        if self.output_rows == 0 {
            return Err(Error::MissingLogits);
        }
        // SAFETY: decode requested all rows; llama_get_logits synchronizes, and the copy precedes any reuse.
        unsafe {
            let ptr = ffi::llama_get_logits(self.context.0.as_ptr());
            if ptr.is_null() {
                return Err(Error::MissingLogits);
            }
            Ok(std::slice::from_raw_parts(ptr, self.output_rows * self.n_vocab as usize).to_vec())
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires DIFFUSION_MODEL; exercises image embeddings without a projector"]
    fn native_image_embedding_batches_can_prefill_the_diffusion_encoder() {
        let config = ModelConfig::new(std::env::var("DIFFUSION_MODEL").unwrap());
        let mut native = Native::load(&config).unwrap();
        let parts = vec![
            PromptPart::Text(
                native
                    .tokenize("<|turn>user\n<|image>", true, true)
                    .unwrap(),
            ),
            PromptPart::Image {
                embeddings: vec![0.0; 4 * native.n_embd],
                tokens: 4,
            },
            PromptPart::Text(
                native
                    .tokenize("<image|>Describe it.<turn|>\n<|turn>model\n", false, true)
                    .unwrap(),
            ),
        ];
        let mut batch = Batch::new(native.batch_size).unwrap();
        let length = native.prefill(&parts, &[], &mut batch).unwrap();
        let canvas = native.tokenize("A", false, false).unwrap();
        native
            .decode_canvas(&mut batch, &canvas, length, None, 1.0)
            .unwrap();
        let logits = native.all_logits().unwrap();
        assert_eq!(logits.len(), canvas.len() * native.n_vocab as usize);
        assert!(logits.iter().all(|x| x.is_finite()));
    }
}
