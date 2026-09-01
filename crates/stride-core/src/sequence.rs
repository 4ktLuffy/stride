use crate::{RequestId, SequenceId, ServiceClass, Tick, TokenId};

/// Why a sequence stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit `max_tokens`.
    Length,
    /// Emitted EOS or a configured stop token.
    Stop,
    /// Client went away or explicitly cancelled.
    Cancelled,
    /// Evicted under memory pressure and not recoverable.
    Preempted,
}

/// Lifecycle of a single generation stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// Admitted, holds no GPU blocks yet.
    Waiting,
    /// Prompt is being processed, possibly across several chunked steps.
    Prefilling,
    /// Generating one token per step.
    Decoding,
    /// Blocks released under pressure; tokens retained for recomputation.
    Swapped,
    Finished(FinishReason),
}

impl SequenceState {
    pub fn is_active(self) -> bool {
        matches!(self, SequenceState::Prefilling | SequenceState::Decoding)
    }

    pub fn is_finished(self) -> bool {
        matches!(self, SequenceState::Finished(_))
    }
}

/// One generation stream. Owns its token history; block ownership lives in the
/// KV-cache crate so that this type stays free of allocator concerns.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: SequenceId,
    pub request: RequestId,
    pub tenant: String,
    pub class: ServiceClass,
    pub state: SequenceState,

    /// Prompt followed by every generated token.
    tokens: Vec<TokenId>,
    prompt_len: usize,
    /// How much of `tokens` already has KV entries computed on the device.
    /// Trails `tokens.len()` while a prompt is being prefilled in chunks.
    computed: usize,
    /// Prompt tokens served from the prefix cache rather than recomputed.
    pub cached_prefix_len: usize,

    pub arrived_at: Tick,
    pub first_token_at: Option<Tick>,
    pub last_token_at: Option<Tick>,
    pub max_tokens: usize,
}

impl Sequence {
    pub fn new(
        request: RequestId,
        tenant: impl Into<String>,
        class: ServiceClass,
        prompt: Vec<TokenId>,
        max_tokens: usize,
        arrived_at: Tick,
    ) -> Self {
        let prompt_len = prompt.len();
        Self {
            id: SequenceId::next(),
            request,
            tenant: tenant.into(),
            class,
            state: SequenceState::Waiting,
            tokens: prompt,
            prompt_len,
            computed: 0,
            cached_prefix_len: 0,
            arrived_at,
            first_token_at: None,
            last_token_at: None,
            max_tokens,
        }
    }

    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn prompt(&self) -> &[TokenId] {
        &self.tokens[..self.prompt_len]
    }

    pub fn output(&self) -> &[TokenId] {
        &self.tokens[self.prompt_len..]
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }

    pub fn output_len(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    pub fn total_len(&self) -> usize {
        self.tokens.len()
    }

    pub fn computed_len(&self) -> usize {
        self.computed
    }

    /// Tokens whose KV entries still have to be computed this step.
    pub fn uncomputed_len(&self) -> usize {
        self.tokens.len() - self.computed
    }

    /// Mark `n` further tokens as having KV entries on the device.
    pub fn advance_computed(&mut self, n: usize) {
        self.computed = (self.computed + n).min(self.tokens.len());
    }

    /// Adopt `n` leading tokens whose KV blocks came from the prefix cache.
    ///
    /// May exceed the prompt: a preempted sequence recomputes its generated
    /// tokens too, and those may still be resident from before it was evicted.
    pub fn adopt_cached_prefix(&mut self, n: usize) {
        debug_assert!(n <= self.tokens.len());
        self.cached_prefix_len = n;
        self.computed = self.computed.max(n);
    }

    /// Append one generated token.
    pub fn push_token(&mut self, token: TokenId, now: Tick) {
        self.tokens.push(token);
        self.computed = self.tokens.len();
        if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
        }
        self.last_token_at = Some(now);
    }

    /// Drop KV state but keep tokens, so the sequence can be recomputed later.
    pub fn swap_out(&mut self) {
        self.computed = 0;
        self.cached_prefix_len = 0;
        self.state = SequenceState::Swapped;
    }

    pub fn finish(&mut self, reason: FinishReason) {
        self.state = SequenceState::Finished(reason);
    }

    /// Time-to-first-token, once the first token has been emitted.
    pub fn ttft_us(&self) -> Option<Tick> {
        self.first_token_at.map(|t| t.saturating_sub(self.arrived_at))
    }

    /// Mean inter-token latency across the generated tokens.
    pub fn mean_itl_us(&self) -> Option<f64> {
        let (first, last) = (self.first_token_at?, self.last_token_at?);
        let gaps = self.output_len().checked_sub(1)?;
        if gaps == 0 {
            return None;
        }
        Some(last.saturating_sub(first) as f64 / gaps as f64)
    }

    pub fn is_at_limit(&self) -> bool {
        self.output_len() >= self.max_tokens
    }
}
