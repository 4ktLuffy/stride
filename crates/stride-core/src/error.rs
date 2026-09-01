use crate::{RequestId, SequenceId};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no free KV blocks: {requested} requested, {available} available")]
    OutOfBlocks { requested: usize, available: usize },

    #[error("unknown sequence {0:?}")]
    UnknownSequence(SequenceId),

    #[error("unknown request {0:?}")]
    UnknownRequest(RequestId),

    #[error("request {0:?} was cancelled")]
    Cancelled(RequestId),

    #[error("prompt of {got} tokens exceeds the model context of {max}")]
    PromptTooLong { got: usize, max: usize },

    #[error("invalid sampling parameters: {0}")]
    InvalidSampling(String),
}
