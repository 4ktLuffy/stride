use crate::{Error, Result, TokenId};

/// Decoding parameters for one sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub stop_tokens: Vec<TokenId>,
    /// Fixed seed. `None` means the runtime picks one and reports it back, so
    /// every generation can be replayed.
    pub seed: Option<u64>,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            max_tokens: 128,
            min_tokens: 0,
            stop_tokens: Vec::new(),
            seed: None,
            ignore_eos: false,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding: deterministic given the same model and batch shape.
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            temperature: 0.0,
            max_tokens,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if !(0.0..=2.0).contains(&self.temperature) || self.temperature.is_nan() {
            return Err(Error::InvalidSampling(format!(
                "temperature must be in [0, 2], got {}",
                self.temperature
            )));
        }
        if !(0.0..=1.0).contains(&self.top_p) || self.top_p.is_nan() {
            return Err(Error::InvalidSampling(format!(
                "top_p must be in (0, 1], got {}",
                self.top_p
            )));
        }
        if self.top_p == 0.0 {
            return Err(Error::InvalidSampling("top_p must be > 0".into()));
        }
        if self.top_k == Some(0) {
            return Err(Error::InvalidSampling(
                "top_k must be >= 1, or None to disable".into(),
            ));
        }
        if self.max_tokens == 0 {
            return Err(Error::InvalidSampling("max_tokens must be >= 1".into()));
        }
        if self.min_tokens > self.max_tokens {
            return Err(Error::InvalidSampling(format!(
                "min_tokens ({}) exceeds max_tokens ({})",
                self.min_tokens, self.max_tokens
            )));
        }
        Ok(())
    }

    /// True when this configuration always picks the argmax token.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 || self.top_k == Some(1)
    }
}
