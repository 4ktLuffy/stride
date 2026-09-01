//! Per-request token streams.

use stride_core::{FinishReason, TokenId};
use tokio::sync::mpsc;

/// Token accounting for one completed sequence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Prompt tokens served from the prefix cache rather than recomputed.
    /// Reported because it is what a cached system prompt actually saves.
    pub cached_prompt_tokens: usize,
}

impl Usage {
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.completion_tokens
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token {
        /// Zero-based position in the completion.
        index: usize,
        token: TokenId,
        /// Decoded text. Empty when the token completes only part of a
        /// multi-byte character; the remainder arrives with a later token.
        text: String,
    },
    Done {
        reason: FinishReason,
        /// Any buffered bytes flushed at the end of the stream.
        trailing_text: String,
        usage: Usage,
    },
}

/// Receiving half of one request's token stream.
#[derive(Debug)]
pub struct TokenStream {
    rx: mpsc::Receiver<StreamEvent>,
}

impl TokenStream {
    pub(crate) fn new(rx: mpsc::Receiver<StreamEvent>) -> Self {
        Self { rx }
    }

    pub async fn next(&mut self) -> Option<StreamEvent> {
        self.rx.recv().await
    }

    /// Consume the whole stream and return the complete text with its usage.
    pub async fn collect(mut self) -> (String, Usage, Option<FinishReason>) {
        let mut text = String::new();
        let mut usage = Usage::default();
        let mut reason = None;
        while let Some(event) = self.rx.recv().await {
            match event {
                StreamEvent::Token { text: t, .. } => text.push_str(&t),
                StreamEvent::Done {
                    reason: r,
                    trailing_text,
                    usage: u,
                } => {
                    text.push_str(&trailing_text);
                    usage = u;
                    reason = Some(r);
                    break;
                }
            }
        }
        (text, usage, reason)
    }
}
