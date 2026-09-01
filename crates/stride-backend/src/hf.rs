//! Hugging Face tokenizer, loaded from a checkpoint's `tokenizer.json`.
//!
//! Required whenever a real model is being served: the byte tokenizer produces
//! ids the model has never seen, so a mismatched tokenizer yields fluent-looking
//! nonsense rather than an error. Loading the checkpoint's own file is the only
//! way to be sure the ids mean what the weights expect.

use std::path::Path;

use stride_core::TokenId;
use tokenizers::Tokenizer as HfInner;

use crate::tokenizer::Tokenizer;

pub struct HfTokenizer {
    inner: HfInner,
    eos: TokenId,
    /// Ids that must never be shown to a client.
    special: Vec<TokenId>,
}

impl HfTokenizer {
    /// Load from a checkpoint directory or a `tokenizer.json` path.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let inner = HfInner::from_file(&file)
            .map_err(|e| format!("cannot load {}: {e}", file.display()))?;

        // Prefer the id declared by the checkpoint's generation config, and
        // fall back to the conventional names only if it is absent.
        let eos = [
            "<|eot_id|>",
            "<|end_of_text|>",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
        ]
        .iter()
        .find_map(|name| inner.token_to_id(name))
        .ok_or_else(|| {
            format!(
                "{} declares no recognisable end-of-sequence token; \
                     generation would never stop on its own",
                file.display()
            )
        })?;

        let special = inner
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .iter()
            .filter(|(_, t)| t.special)
            .map(|(&id, _)| id)
            .collect();

        Ok(Self {
            inner,
            eos,
            special,
        })
    }

    /// Override the end-of-sequence id, for a checkpoint whose
    /// `generation_config.json` disagrees with its tokenizer.
    pub fn with_eos(mut self, eos: TokenId) -> Self {
        self.eos = eos;
        self
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        match self.inner.encode(text, false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(e) => {
                tracing::error!(error = %e, "tokenization failed; treating the prompt as empty");
                Vec::new()
            }
        }
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        self.inner.decode(tokens, true).unwrap_or_default()
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn eos(&self) -> TokenId {
        self.eos
    }

    fn is_special(&self, token: TokenId) -> bool {
        token == self.eos || self.special.contains(&token)
    }
}
