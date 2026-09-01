//! Transformer geometry for dense and Mixture-of-Experts models.
//!
//! Everything the runtime needs to size memory comes from here: how many bytes
//! a token of KV cache costs, how many bytes the weights occupy under a given
//! quantization scheme, and how both divide across a parallelism plan.
//!
//! Presets are convenience shapes. The canonical path is
//! [`ModelConfig::from_hf_config`], which reads the model's own `config.json`,
//! because a preset that drifts from the real checkpoint would silently
//! mis-size the cache.

use serde::{Deserialize, Serialize};

use crate::dtype::{DType, WeightFormat};

/// Grouped-query attention shape. `num_kv_heads == num_q_heads` is multi-head
/// attention; `num_kv_heads == 1` is multi-query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionConfig {
    /// Query heads sharing each KV head.
    pub fn gqa_ratio(&self) -> usize {
        self.num_q_heads / self.num_kv_heads.max(1)
    }

    pub fn q_dim(&self) -> usize {
        self.num_q_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// The feed-forward block: one dense MLP, or a routed set of experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedForward {
    Dense {
        intermediate_size: usize,
    },
    Moe {
        num_experts: usize,
        /// Experts each token is routed to.
        experts_per_token: usize,
        expert_intermediate_size: usize,
        /// Experts every token passes through regardless of routing.
        shared_experts: usize,
    },
}

impl FeedForward {
    /// Weights in this block, counting every expert.
    fn params(&self, hidden: usize) -> u64 {
        // Gated MLP: gate, up and down projections.
        const GATED_MATRICES: u64 = 3;
        match *self {
            FeedForward::Dense { intermediate_size } => {
                GATED_MATRICES * hidden as u64 * intermediate_size as u64
            }
            FeedForward::Moe {
                num_experts,
                expert_intermediate_size,
                shared_experts,
                ..
            } => {
                let per_expert = GATED_MATRICES * hidden as u64 * expert_intermediate_size as u64;
                let router = hidden as u64 * num_experts as u64;
                per_expert * (num_experts + shared_experts) as u64 + router
            }
        }
    }

    /// Weights actually touched by one token. For MoE this is the routed
    /// subset, which is why an MoE model's compute cost tracks its active
    /// parameters while its memory cost tracks its total parameters.
    fn active_params(&self, hidden: usize) -> u64 {
        const GATED_MATRICES: u64 = 3;
        match *self {
            FeedForward::Dense { intermediate_size } => {
                GATED_MATRICES * hidden as u64 * intermediate_size as u64
            }
            FeedForward::Moe {
                num_experts,
                experts_per_token,
                expert_intermediate_size,
                shared_experts,
            } => {
                let per_expert = GATED_MATRICES * hidden as u64 * expert_intermediate_size as u64;
                let router = hidden as u64 * num_experts as u64;
                per_expert * (experts_per_token + shared_experts) as u64 + router
            }
        }
    }

    pub fn is_moe(&self) -> bool {
        matches!(self, FeedForward::Moe { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub attention: AttentionConfig,
    pub ffn: FeedForward,
    pub vocab_size: usize,
    /// Embedding and output projection share weights.
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub weights: WeightFormat,
    /// Format the KV cache is stored in. Quantizing KV buys context length at
    /// a quality cost that has to be measured, not assumed.
    pub kv_dtype: DType,
}

impl ModelConfig {
    /// Weights per transformer layer, attention plus feed-forward.
    pub fn params_per_layer(&self) -> u64 {
        let h = self.hidden_size as u64;
        let a = &self.attention;
        let attn = h * a.q_dim() as u64      // q_proj
            + h * a.kv_dim() as u64          // k_proj
            + h * a.kv_dim() as u64          // v_proj
            + a.q_dim() as u64 * h; // o_proj
        attn + self.ffn.params(self.hidden_size)
    }

    fn embedding_params(&self) -> u64 {
        let one = self.vocab_size as u64 * self.hidden_size as u64;
        if self.tie_word_embeddings {
            one
        } else {
            2 * one
        }
    }

    /// Every weight in the checkpoint.
    pub fn total_params(&self) -> u64 {
        self.params_per_layer() * self.num_layers as u64 + self.embedding_params()
    }

    /// Weights one token's forward pass touches. Equal to `total_params` for a
    /// dense model; far smaller for MoE.
    pub fn active_params(&self) -> u64 {
        let h = self.hidden_size as u64;
        let a = &self.attention;
        let attn = h * a.q_dim() as u64
            + 2 * h * a.kv_dim() as u64
            + a.q_dim() as u64 * h;
        (attn + self.ffn.active_params(self.hidden_size)) * self.num_layers as u64
            + self.embedding_params()
    }

    /// Bytes the weights occupy under the configured format.
    pub fn weight_bytes(&self) -> u64 {
        self.weights.bytes_for(self.total_params() as usize) as u64
    }

    /// KV-cache bytes for one token, across all layers, on one GPU holding the
    /// whole model.
    ///
    /// Two tensors — K and V — per layer, each `num_kv_heads * head_dim`
    /// elements. This is the number that decides how much context a deployment
    /// can hold, and it is why grouped-query attention matters at scale.
    pub fn kv_bytes_per_token(&self) -> u64 {
        2 * self.num_layers as u64
            * self.kv_dtype.bytes_for(self.attention.kv_dim()) as u64
    }

    /// FLOPs for one token of decode, counting a multiply-add as two.
    pub fn decode_flops_per_token(&self) -> u64 {
        2 * self.active_params()
    }

    /// Validate the geometry itself, independent of any parallelism plan.
    pub fn validate(&self) -> Result<(), String> {
        if self.attention.num_kv_heads == 0 || self.attention.num_q_heads == 0 {
            return Err("attention must have at least one query and KV head".into());
        }
        if self.attention.num_q_heads % self.attention.num_kv_heads != 0 {
            return Err(format!(
                "{}: {} query heads do not divide evenly into {} KV heads",
                self.name, self.attention.num_q_heads, self.attention.num_kv_heads
            ));
        }
        if let FeedForward::Moe {
            num_experts,
            experts_per_token,
            ..
        } = self.ffn
        {
            if experts_per_token == 0 || experts_per_token > num_experts {
                return Err(format!(
                    "{}: routes {experts_per_token} of {num_experts} experts",
                    self.name
                ));
            }
        }
        Ok(())
    }

    /// Read geometry from a Hugging Face `config.json`.
    ///
    /// Prefer this over a preset. The checkpoint is the source of truth, and a
    /// preset that has drifted from it mis-sizes the KV cache silently.
    pub fn from_hf_config(json: &str, weights: WeightFormat, kv_dtype: DType) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let get_usize = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("config.json is missing `{k}`"))
        };

        let hidden_size = get_usize("hidden_size")?;
        let num_q_heads = get_usize("num_attention_heads")?;
        let num_kv_heads = v
            .get("num_key_value_heads")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(num_q_heads);
        let head_dim = v
            .get("head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| hidden_size / num_q_heads.max(1));

        // MoE checkpoints disagree on field names; accept the common spellings.
        let num_experts = ["num_local_experts", "num_experts", "n_routed_experts"]
            .iter()
            .find_map(|k| v.get(*k).and_then(|x| x.as_u64()))
            .map(|x| x as usize);
        let experts_per_token = ["num_experts_per_tok", "num_experts_per_token", "topk"]
            .iter()
            .find_map(|k| v.get(*k).and_then(|x| x.as_u64()))
            .map(|x| x as usize);

        let ffn = match (num_experts, experts_per_token) {
            (Some(num_experts), Some(experts_per_token)) if num_experts > 1 => {
                let expert_intermediate_size = ["moe_intermediate_size", "intermediate_size"]
                    .iter()
                    .find_map(|k| v.get(*k).and_then(|x| x.as_u64()))
                    .map(|x| x as usize)
                    .ok_or("config.json is missing an expert intermediate size")?;
                FeedForward::Moe {
                    num_experts,
                    experts_per_token,
                    expert_intermediate_size,
                    shared_experts: v
                        .get("n_shared_experts")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0) as usize,
                }
            }
            _ => FeedForward::Dense {
                intermediate_size: get_usize("intermediate_size")?,
            },
        };

        let cfg = ModelConfig {
            name: v
                .get("_name_or_path")
                .and_then(|x| x.as_str())
                .unwrap_or("unnamed")
                .to_string(),
            num_layers: get_usize("num_hidden_layers")?,
            hidden_size,
            attention: AttentionConfig {
                num_q_heads,
                num_kv_heads,
                head_dim,
            },
            ffn,
            vocab_size: get_usize("vocab_size")?,
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            max_position_embeddings: v
                .get("max_position_embeddings")
                .and_then(|x| x.as_u64())
                .unwrap_or(4096) as usize,
            weights,
            kv_dtype,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}
