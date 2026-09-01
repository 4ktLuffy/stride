//! Text to tokens and back.
//!
//! The runtime is deliberately generic over tokenization. A production
//! deployment slots in the model's own tokenizer; [`ByteTokenizer`] exists so
//! the full serving path can be exercised without a checkpoint on disk, and it
//! is exactly reversible, which makes it useful for round-trip tests that a
//! lossy BPE would fail for uninteresting reasons.

use stride_core::TokenId;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> String;
    fn vocab_size(&self) -> usize;
    /// Token that ends generation.
    fn eos(&self) -> TokenId;
    /// True for tokens that should never be shown to a client.
    fn is_special(&self, token: TokenId) -> bool {
        token == self.eos()
    }
}

/// One token per UTF-8 byte, plus a small block of special ids above them.
///
/// Byte-level means every string round-trips exactly and no text is
/// unrepresentable — including partial multi-byte characters, which matters
/// when tokens are streamed one at a time.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub const EOS: TokenId = 256;
    pub const BOS: TokenId = 257;
    pub const VOCAB: usize = 258;
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        text.as_bytes().iter().map(|&b| b as TokenId).collect()
    }

    /// Decode, replacing any byte sequence that is not valid UTF-8.
    ///
    /// A streamed response can cut a multi-byte character in half, so this has
    /// to be lossy rather than an error. Callers that stream should buffer
    /// with [`IncrementalDecoder`] instead of decoding token by token.
    fn decode(&self, tokens: &[TokenId]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter(|&&t| t < 256)
            .map(|&t| t as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        Self::VOCAB
    }

    fn eos(&self) -> TokenId {
        Self::EOS
    }

    fn is_special(&self, token: TokenId) -> bool {
        token >= 256
    }
}

/// Buffers bytes across tokens so a multi-byte character is only emitted once
/// it is complete.
///
/// Without this, streaming a character like `é` one byte at a time yields two
/// replacement characters instead of one letter.
#[derive(Debug, Default)]
pub struct IncrementalDecoder {
    pending: Vec<u8>,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one token, returning whatever text is now complete.
    pub fn push(&mut self, tokenizer: &dyn Tokenizer, token: TokenId) -> String {
        if tokenizer.is_special(token) {
            return String::new();
        }
        self.pending.push(token as u8);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                // Emit the valid prefix and keep the incomplete tail buffered.
                let valid = e.valid_up_to();
                if valid == 0 {
                    // Still incomplete, unless it can never become valid.
                    if self.pending.len() >= 4 {
                        let out = String::from_utf8_lossy(&self.pending).into_owned();
                        self.pending.clear();
                        return out;
                    }
                    return String::new();
                }
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                out
            }
        }
    }

    /// Flush anything still buffered at the end of a stream.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trips_exactly() {
        let t = ByteTokenizer;
        let text = "the quick brown fox";
        assert_eq!(t.decode(&t.encode(text)), text);
    }

    #[test]
    fn multibyte_text_round_trips_exactly() {
        let t = ByteTokenizer;
        for text in ["café", "日本語", "🙂 emoji", "ሰላም"] {
            assert_eq!(t.decode(&t.encode(text)), text, "failed on {text}");
        }
    }

    #[test]
    fn streaming_reassembles_split_characters() {
        let t = ByteTokenizer;
        let text = "café ሰላም 🙂";
        let mut d = IncrementalDecoder::new();
        let mut out = String::new();
        for token in t.encode(text) {
            out.push_str(&d.push(&t, token));
        }
        out.push_str(&d.finish());
        assert_eq!(out, text, "streamed output must match the whole string");
    }

    #[test]
    fn streaming_emits_nothing_for_an_incomplete_character() {
        let t = ByteTokenizer;
        let bytes = t.encode("é"); // two bytes
        let mut d = IncrementalDecoder::new();
        assert_eq!(d.push(&t, bytes[0]), "", "half a character is not text yet");
        assert_eq!(d.push(&t, bytes[1]), "é");
    }

    #[test]
    fn special_tokens_are_never_emitted_as_text() {
        let t = ByteTokenizer;
        let mut d = IncrementalDecoder::new();
        assert_eq!(d.push(&t, ByteTokenizer::EOS), "");
        assert!(t.is_special(ByteTokenizer::EOS));
        assert!(!t.is_special(b'a' as TokenId));
    }
}
