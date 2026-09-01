use crate::{RequestId, SamplingParams, Tick, TokenId};

/// Latency class a request is admitted under.
///
/// The scheduler does not treat these as raw priorities — a `Batch` request
/// that has been waiting long enough will outrank a freshly arrived
/// `Interactive` one. The class sets the deadline, not the rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ServiceClass {
    /// Chat and agent traffic. Tight time-to-first-token budget.
    Interactive,
    /// Throughput-oriented work that still has a deadline.
    Batch,
    /// Best effort. Runs only on capacity nothing else claimed.
    Background,
}

impl ServiceClass {
    /// Time-to-first-token budget in microseconds.
    pub const fn ttft_budget_us(self) -> Tick {
        match self {
            ServiceClass::Interactive => 500_000,     // 500 ms
            ServiceClass::Batch => 10_000_000,        // 10 s
            ServiceClass::Background => 120_000_000,  // 2 min
        }
    }

    /// Per-token budget once generation has started, in microseconds.
    pub const fn itl_budget_us(self) -> Tick {
        match self {
            ServiceClass::Interactive => 50_000,     // 50 ms
            ServiceClass::Batch => 200_000,          // 200 ms
            ServiceClass::Background => 1_000_000,   // 1 s
        }
    }
}

/// One admitted client request.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: RequestId,
    /// Isolation boundary. Cached prefix blocks are never shared across
    /// tenants, even when the token content is byte-identical.
    pub tenant: String,
    pub prompt: Vec<TokenId>,
    pub params: SamplingParams,
    pub class: ServiceClass,
    /// When the request entered the waiting queue.
    pub arrived_at: Tick,
    /// Number of independent sequences to generate.
    pub n: usize,
}

impl Request {
    pub fn new(tenant: impl Into<String>, prompt: Vec<TokenId>, arrived_at: Tick) -> Self {
        Self {
            id: RequestId::next(),
            tenant: tenant.into(),
            prompt,
            params: SamplingParams::default(),
            class: ServiceClass::Interactive,
            arrived_at,
            n: 1,
        }
    }

    pub fn with_params(mut self, params: SamplingParams) -> Self {
        self.params = params;
        self
    }

    pub fn with_class(mut self, class: ServiceClass) -> Self {
        self.class = class;
        self
    }

    pub fn prompt_len(&self) -> usize {
        self.prompt.len()
    }

    /// Absolute tick by which this request must emit its first token.
    pub fn ttft_deadline(&self) -> Tick {
        self.arrived_at.saturating_add(self.class.ttft_budget_us())
    }

    /// Microseconds of slack left against the TTFT deadline, negative once missed.
    pub fn ttft_slack_us(&self, now: Tick) -> i64 {
        self.ttft_deadline() as i64 - now as i64
    }
}
