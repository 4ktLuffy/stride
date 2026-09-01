//! Model geometry, quantization formats and capacity planning for large dense
//! and Mixture-of-Experts transformers.
//!
//! The runtime needs to answer three questions before it serves anything:
//! does the model fit, how much context is left, and what does one token cost.
//! This crate answers all three from arithmetic over the model's own shape, so
//! a deployment is validated at planning time rather than discovered to be
//! impossible under load.
//!
//! ```
//! use stride_model::{presets, MemoryPlan, ParallelConfig};
//!
//! // Llama-3.1-70B in BF16 across 4 H100s.
//! let model = presets::llama3_70b();
//! let plan = MemoryPlan::new(
//!     &model,
//!     ParallelConfig::tp(4),
//!     presets::H100_80GB,
//!     16,
//!     0.10,
//! ).unwrap();
//!
//! assert!(plan.num_blocks > 0);
//! assert!(plan.concurrent_sequences(4096) > 0);
//! ```

pub mod config;
pub mod dtype;
pub mod memory;
pub mod parallel;
pub mod presets;

pub use config::{AttentionConfig, FeedForward, ModelConfig};
pub use dtype::{DType, WeightFormat};
pub use memory::{decode_bandwidth_ceiling_tokens_per_s, MemoryPlan, PlanError};
pub use parallel::ParallelConfig;
