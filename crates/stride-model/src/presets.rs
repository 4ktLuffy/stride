//! Reference model shapes and accelerator profiles.
//!
//! Model presets are *shapes*, kept here so the planner can be exercised
//! without a checkpoint on disk. They are not a substitute for
//! [`ModelConfig::from_hf_config`] — the tests in this crate check each preset
//! reproduces its published parameter count, which catches a typo but not a
//! checkpoint that has since changed.
//!
//! GPU profiles carry vendor peak figures. Peak numbers are ceilings, not
//! expectations: real kernels reach a fraction of them. They are used here only
//! for capacity planning arithmetic, and every field should be replaced with a
//! value measured on your own fleet before it informs a scheduling decision.

use crate::config::{AttentionConfig, FeedForward, ModelConfig};
use crate::dtype::{DType, WeightFormat};

fn llama_like(
    name: &str,
    num_layers: usize,
    hidden_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    intermediate_size: usize,
) -> ModelConfig {
    ModelConfig {
        name: name.to_string(),
        num_layers,
        hidden_size,
        attention: AttentionConfig {
            num_q_heads,
            num_kv_heads,
            head_dim: 128,
        },
        ffn: FeedForward::Dense { intermediate_size },
        vocab_size: 128_256,
        tie_word_embeddings: false,
        max_position_embeddings: 131_072,
        weights: WeightFormat::dense(DType::BF16),
        kv_dtype: DType::BF16,
    }
}

fn mixtral_like(
    name: &str,
    num_layers: usize,
    hidden_size: usize,
    num_q_heads: usize,
    expert_intermediate_size: usize,
) -> ModelConfig {
    ModelConfig {
        name: name.to_string(),
        num_layers,
        hidden_size,
        attention: AttentionConfig {
            num_q_heads,
            num_kv_heads: 8,
            head_dim: 128,
        },
        ffn: FeedForward::Moe {
            num_experts: 8,
            experts_per_token: 2,
            expert_intermediate_size,
            shared_experts: 0,
        },
        vocab_size: 32_000,
        tie_word_embeddings: false,
        max_position_embeddings: 65_536,
        weights: WeightFormat::dense(DType::BF16),
        kv_dtype: DType::BF16,
    }
}

pub fn llama3_8b() -> ModelConfig {
    llama_like("llama-3.1-8b", 32, 4096, 32, 8, 14_336)
}

pub fn llama3_70b() -> ModelConfig {
    llama_like("llama-3.1-70b", 80, 8192, 64, 8, 28_672)
}

pub fn llama3_405b() -> ModelConfig {
    llama_like("llama-3.1-405b", 126, 16_384, 128, 8, 53_248)
}

pub fn mixtral_8x7b() -> ModelConfig {
    mixtral_like("mixtral-8x7b", 32, 4096, 32, 14_336)
}

pub fn mixtral_8x22b() -> ModelConfig {
    mixtral_like("mixtral-8x22b", 56, 6144, 48, 16_384)
}

pub fn all() -> Vec<ModelConfig> {
    vec![
        llama3_8b(),
        llama3_70b(),
        llama3_405b(),
        mixtral_8x7b(),
        mixtral_8x22b(),
    ]
}

/// One accelerator's capacity, as published by the vendor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuProfile {
    pub name: &'static str,
    pub memory_bytes: u64,
    /// Peak HBM bandwidth. Decode is bandwidth-bound, so this sets the ceiling
    /// on tokens per second far more often than the FLOP number does.
    pub hbm_bandwidth_bytes_per_s: u64,
    /// Dense BF16 tensor-core peak, without sparsity.
    pub bf16_flops_per_s: u64,
    /// Whether ranks on this part are expected to be NVLink-connected.
    pub nvlink: bool,
}

pub const A100_80GB: GpuProfile = GpuProfile {
    name: "A100-SXM-80GB",
    memory_bytes: 80 * 1024 * 1024 * 1024,
    hbm_bandwidth_bytes_per_s: 2_039_000_000_000,
    bf16_flops_per_s: 312_000_000_000_000,
    nvlink: true,
};

pub const H100_80GB: GpuProfile = GpuProfile {
    name: "H100-SXM-80GB",
    memory_bytes: 80 * 1024 * 1024 * 1024,
    hbm_bandwidth_bytes_per_s: 3_350_000_000_000,
    bf16_flops_per_s: 989_000_000_000_000,
    nvlink: true,
};

pub const H200_141GB: GpuProfile = GpuProfile {
    name: "H200-SXM-141GB",
    memory_bytes: 141 * 1000 * 1000 * 1000,
    hbm_bandwidth_bytes_per_s: 4_800_000_000_000,
    bf16_flops_per_s: 989_000_000_000_000,
    nvlink: true,
};

pub const L40S_48GB: GpuProfile = GpuProfile {
    name: "L40S-48GB",
    memory_bytes: 48 * 1024 * 1024 * 1024,
    hbm_bandwidth_bytes_per_s: 864_000_000_000,
    bf16_flops_per_s: 362_000_000_000_000,
    nvlink: false,
};
