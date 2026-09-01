//! Core types shared by every layer of the Stride runtime.
//!
//! Nothing in this crate allocates GPU memory or talks to a model. It defines
//! the vocabulary — requests, sequences, sampling parameters, service classes —
//! that the KV-cache allocator, the scheduler and the API layer all agree on.

mod error;
mod ids;
mod request;
mod sampling;
mod sequence;

pub use error::{Error, Result};
pub use ids::{RequestId, SequenceId};
pub use request::{Request, ServiceClass};
pub use sampling::SamplingParams;
pub use sequence::{FinishReason, Sequence, SequenceState};

/// A token id as produced by the tokenizer.
pub type TokenId = u32;

/// Monotonic logical clock, in microseconds since runtime start.
///
/// The runtime never reads the wall clock directly. Every component takes the
/// current tick as an argument, which is what makes the scheduler
/// deterministically replayable from a recorded trace.
pub type Tick = u64;
