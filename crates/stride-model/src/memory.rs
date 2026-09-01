//! Capacity planning: what fits on a rank, and how much context is left over.
//!
//! Serving a large model is mostly an arithmetic problem before it is a
//! performance problem. Weights, activation working set and KV cache compete
//! for the same HBM, and the KV cache is whatever survives the other two. This
//! module does that subtraction explicitly so a deployment fails at planning
//! time rather than with an out-of-memory error mid-request.

use crate::config::ModelConfig;
use crate::parallel::ParallelConfig;
use crate::presets::GpuProfile;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PlanError {
    #[error(
        "{model}: weights need {weight_gib:.1} GiB per rank but {gpu} has {capacity_gib:.1} GiB. \
         Raise tensor or pipeline parallelism, or quantize the weights"
    )]
    WeightsDoNotFit {
        model: String,
        gpu: &'static str,
        weight_gib: f64,
        capacity_gib: f64,
    },

    #[error(
        "{model}: {leftover_gib:.1} GiB left for KV after weights and activations, \
         under the {needed_gib:.1} GiB one sequence of {context} tokens requires"
    )]
    NoRoomForContext {
        model: String,
        leftover_gib: f64,
        needed_gib: f64,
        context: usize,
    },

    #[error("invalid parallelism plan: {0}")]
    Parallel(String),
}

const GIB: f64 = (1024 * 1024 * 1024) as f64;

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryPlan {
    pub model: String,
    pub gpu: &'static str,
    pub parallel: ParallelConfig,
    pub block_size: usize,

    pub weight_bytes_per_rank: u64,
    pub activation_reserve_bytes: u64,
    pub kv_bytes_per_rank: u64,
    pub kv_bytes_per_token_per_rank: u64,

    /// KV blocks that fit on one rank.
    pub num_blocks: usize,
    /// Total tokens of KV those blocks hold.
    pub total_kv_tokens: usize,
}

impl MemoryPlan {
    /// Work out how much KV cache is left once weights and activations are paid
    /// for.
    ///
    /// `activation_fraction` reserves a share of what remains after weights for
    /// activations, workspace and fragmentation. It is a headroom knob, not a
    /// measurement — profile a real deployment and set it from what you see.
    pub fn new(
        model: &ModelConfig,
        parallel: ParallelConfig,
        gpu: GpuProfile,
        block_size: usize,
        activation_fraction: f64,
    ) -> Result<Self, PlanError> {
        parallel.validate(model).map_err(PlanError::Parallel)?;
        assert!(block_size > 0, "block_size must be positive");
        assert!(
            (0.0..1.0).contains(&activation_fraction),
            "activation_fraction must be in [0, 1)"
        );

        let weight_bytes = parallel.weight_bytes_per_rank(model);
        if weight_bytes >= gpu.memory_bytes {
            return Err(PlanError::WeightsDoNotFit {
                model: model.name.clone(),
                gpu: gpu.name,
                weight_gib: weight_bytes as f64 / GIB,
                capacity_gib: gpu.memory_bytes as f64 / GIB,
            });
        }

        let after_weights = gpu.memory_bytes - weight_bytes;
        let activation_reserve = (after_weights as f64 * activation_fraction) as u64;
        let kv_bytes = after_weights - activation_reserve;

        let per_token = parallel.kv_bytes_per_token_per_rank(model).max(1);
        let bytes_per_block = per_token * block_size as u64;
        let num_blocks = (kv_bytes / bytes_per_block) as usize;

        if num_blocks == 0 {
            return Err(PlanError::NoRoomForContext {
                model: model.name.clone(),
                leftover_gib: kv_bytes as f64 / GIB,
                needed_gib: bytes_per_block as f64 / GIB,
                context: block_size,
            });
        }

        Ok(MemoryPlan {
            model: model.name.clone(),
            gpu: gpu.name,
            parallel,
            block_size,
            weight_bytes_per_rank: weight_bytes,
            activation_reserve_bytes: activation_reserve,
            kv_bytes_per_rank: kv_bytes,
            kv_bytes_per_token_per_rank: per_token,
            num_blocks,
            total_kv_tokens: num_blocks * block_size,
        })
    }

    /// Sequences of `context` tokens that fit concurrently, ignoring sharing.
    ///
    /// This is the pessimistic figure. Prefix reuse raises it whenever requests
    /// share a system prompt, which is exactly the case the cache exists for.
    pub fn concurrent_sequences(&self, context: usize) -> usize {
        if context == 0 {
            return 0;
        }
        let blocks_each = context.div_ceil(self.block_size);
        self.num_blocks / blocks_each.max(1)
    }

    /// Longest single sequence the cache can hold.
    pub fn max_context(&self) -> usize {
        self.total_kv_tokens
    }

    pub fn weight_gib(&self) -> f64 {
        self.weight_bytes_per_rank as f64 / GIB
    }

    pub fn kv_gib(&self) -> f64 {
        self.kv_bytes_per_rank as f64 / GIB
    }

    /// A human-readable summary, for startup logs and capacity reviews.
    pub fn summary(&self) -> String {
        let p = self.parallel;
        format!(
            "{} on {}x {} (tp={} pp={} ep={})\n  \
             weights {:.1} GiB/rank | activations {:.1} GiB/rank | KV {:.1} GiB/rank\n  \
             {} blocks of {} tokens = {} tokens of KV per rank\n  \
             {} concurrent sequences at 4k context, {} at 32k",
            self.model,
            p.world_size(),
            self.gpu,
            p.tensor,
            p.pipeline,
            p.expert,
            self.weight_gib(),
            self.activation_reserve_bytes as f64 / GIB,
            self.kv_gib(),
            self.num_blocks,
            self.block_size,
            self.total_kv_tokens,
            self.concurrent_sequences(4096),
            self.concurrent_sequences(32_768),
        )
    }
}

/// Decode is bandwidth-bound: every generated token reads the active weights
/// from HBM once. This is the ceiling that ratio implies, before any batching,
/// kernel efficiency or cache traffic is accounted for.
///
/// It is an upper bound derived from published peak bandwidth, not a
/// prediction. A real deployment lands below it.
pub fn decode_bandwidth_ceiling_tokens_per_s(
    model: &ModelConfig,
    parallel: ParallelConfig,
    gpu: GpuProfile,
) -> f64 {
    let active_bytes_per_rank = model
        .weights
        .bytes_for(model.active_params() as usize) as f64
        / parallel.world_size() as f64;
    if active_bytes_per_rank <= 0.0 {
        return 0.0;
    }
    gpu.hbm_bandwidth_bytes_per_s as f64 / active_bytes_per_rank
}
