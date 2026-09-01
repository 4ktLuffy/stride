//! Sharding plans across GPUs, and what each rank ends up holding.

use serde::{Deserialize, Serialize};

use crate::config::{FeedForward, ModelConfig};

/// How a model is split across devices.
///
/// Tensor and pipeline parallelism multiply into the world size. Expert
/// parallelism overlays that grid rather than extending it: EP ranks are drawn
/// from the same devices, each holding a slice of the expert set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Splits every weight matrix across ranks. Cheap on NVLink, punishing
    /// across PCIe, because every layer ends in an all-reduce.
    pub tensor: usize,
    /// Splits layers into stages. Adds no per-layer collective, but leaves
    /// bubbles unless enough microbatches are in flight.
    pub pipeline: usize,
    /// Splits MoE experts across ranks. Turns the FFN into an all-to-all.
    pub expert: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tensor: 1,
            pipeline: 1,
            expert: 1,
        }
    }
}

impl ParallelConfig {
    pub fn tp(tensor: usize) -> Self {
        Self {
            tensor,
            ..Default::default()
        }
    }

    pub fn with_pipeline(mut self, pipeline: usize) -> Self {
        self.pipeline = pipeline;
        self
    }

    pub fn with_expert(mut self, expert: usize) -> Self {
        self.expert = expert;
        self
    }

    pub fn world_size(&self) -> usize {
        self.tensor * self.pipeline
    }

    /// Reject plans the model geometry cannot support.
    ///
    /// These are hard divisibility constraints, not preferences. A plan that
    /// fails here would either crash at load time or silently replicate what
    /// it was asked to shard.
    pub fn validate(&self, model: &ModelConfig) -> Result<(), String> {
        if self.tensor == 0 || self.pipeline == 0 || self.expert == 0 {
            return Err("parallel degrees must be at least 1".into());
        }
        let a = &model.attention;
        if !a.num_q_heads.is_multiple_of(self.tensor) {
            return Err(format!(
                "tp={}: {} query heads do not divide evenly",
                self.tensor, a.num_q_heads
            ));
        }
        if !a.num_kv_heads.is_multiple_of(self.tensor) {
            return Err(format!(
                "tp={}: {} KV heads do not divide evenly. Either lower tp, or \
                 replicate KV heads and accept the duplicated cache",
                self.tensor, a.num_kv_heads
            ));
        }
        if !model.num_layers.is_multiple_of(self.pipeline) {
            return Err(format!(
                "pp={}: {} layers do not divide evenly",
                self.pipeline, model.num_layers
            ));
        }
        match model.ffn {
            FeedForward::Moe { num_experts, .. } => {
                if num_experts % self.expert != 0 {
                    return Err(format!(
                        "ep={}: {num_experts} experts do not divide evenly",
                        self.expert
                    ));
                }
                if self.expert > self.world_size() {
                    return Err(format!(
                        "ep={} exceeds the world size of {}",
                        self.expert,
                        self.world_size()
                    ));
                }
            }
            FeedForward::Dense { .. } if self.expert > 1 => {
                return Err("expert parallelism requires an MoE model".into());
            }
            _ => {}
        }
        Ok(())
    }

    /// KV-cache bytes one rank stores per token.
    ///
    /// Tensor parallelism shards KV heads and pipeline parallelism shards
    /// layers, so both divide the per-token cost.
    pub fn kv_bytes_per_token_per_rank(&self, model: &ModelConfig) -> u64 {
        model.kv_bytes_per_token() / (self.tensor * self.pipeline) as u64
    }

    /// Weight bytes one rank holds.
    ///
    /// Attention and dense FFN shard by `tensor`; MoE experts shard by
    /// `expert` instead, while the router is replicated. Layers then divide by
    /// `pipeline`. Embeddings are assumed sharded across the tensor group.
    pub fn weight_bytes_per_rank(&self, model: &ModelConfig) -> u64 {
        let h = model.hidden_size as u64;
        let a = &model.attention;
        let attn = h * a.q_dim() as u64 + 2 * h * a.kv_dim() as u64 + a.q_dim() as u64 * h;

        let (sharded_by_tp, sharded_by_ep, replicated) = match model.ffn {
            FeedForward::Dense { intermediate_size } => {
                (attn + 3 * h * intermediate_size as u64, 0, 0)
            }
            FeedForward::Moe {
                num_experts,
                expert_intermediate_size,
                shared_experts,
                ..
            } => {
                let per_expert = 3 * h * expert_intermediate_size as u64;
                (
                    attn + per_expert * shared_experts as u64,
                    per_expert * num_experts as u64,
                    h * num_experts as u64, // router
                )
            }
        };

        let per_layer =
            sharded_by_tp / self.tensor as u64 + sharded_by_ep / self.expert as u64 + replicated;
        let layers_here = (model.num_layers / self.pipeline) as u64;

        let embeddings =
            if model.tie_word_embeddings { 1 } else { 2 } * model.vocab_size as u64 * h
                / self.tensor as u64;

        let params = per_layer * layers_here + embeddings;
        model.weights.bytes_for(params as usize) as u64
    }

    /// Collectives one token of decode triggers per layer, as a rough guide to
    /// which fabric a plan needs.
    pub fn describe_collectives(&self, model: &ModelConfig) -> Vec<&'static str> {
        let mut ops = Vec::new();
        if self.tensor > 1 {
            ops.push("all-reduce after attention and FFN (tensor parallel)");
        }
        if self.pipeline > 1 {
            ops.push("point-to-point activation handoff between stages");
        }
        if self.expert > 1 && model.ffn.is_moe() {
            ops.push("all-to-all dispatch and combine around the experts");
        }
        if ops.is_empty() {
            ops.push("none: single-rank execution");
        }
        ops
    }
}
