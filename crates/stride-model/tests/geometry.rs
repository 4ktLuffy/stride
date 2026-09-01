//! Checks the geometry arithmetic against independently known quantities.
//!
//! Every capacity decision the runtime makes is derived from these formulas. A
//! self-consistent but wrong parameter count would mis-size the KV cache on
//! every deployment, so each preset is checked against its published parameter
//! count rather than against another part of this crate.

use stride_model::config::FeedForward;
use stride_model::{presets, DType, MemoryPlan, ModelConfig, ParallelConfig, PlanError, WeightFormat};

/// Assert `got` is within `pct` percent of `want`.
fn near(got: u64, want: f64, pct: f64, what: &str) {
    let got = got as f64;
    let err = (got - want).abs() / want * 100.0;
    assert!(
        err <= pct,
        "{what}: got {:.3}B, expected {:.3}B ({err:.2}% off, tolerance {pct}%)",
        got / 1e9,
        want / 1e9
    );
}

#[test]
fn dense_parameter_counts_match_published_sizes() {
    near(presets::llama3_8b().total_params(), 8.03e9, 1.0, "llama-3.1-8b");
    near(presets::llama3_70b().total_params(), 70.6e9, 1.0, "llama-3.1-70b");
    near(presets::llama3_405b().total_params(), 405.8e9, 1.0, "llama-3.1-405b");
}

#[test]
fn moe_total_and_active_counts_match_published_sizes() {
    let m = presets::mixtral_8x7b();
    near(m.total_params(), 46.7e9, 1.0, "mixtral-8x7b total");
    near(m.active_params(), 12.9e9, 2.0, "mixtral-8x7b active");

    let m = presets::mixtral_8x22b();
    near(m.total_params(), 141.0e9, 1.0, "mixtral-8x22b total");
    near(m.active_params(), 39.0e9, 2.0, "mixtral-8x22b active");
}

#[test]
fn dense_models_activate_every_parameter() {
    for m in [presets::llama3_8b(), presets::llama3_70b()] {
        assert_eq!(
            m.total_params(),
            m.active_params(),
            "{}: a dense model has no inactive weights",
            m.name
        );
    }
}

#[test]
fn moe_separates_memory_cost_from_compute_cost() {
    let m = presets::mixtral_8x22b();
    let ratio = m.total_params() as f64 / m.active_params() as f64;
    assert!(
        ratio > 3.0,
        "8 experts routing 2 should activate roughly a quarter of the FFN, got {ratio:.2}x"
    );
    assert!(m.ffn.is_moe());
}

#[test]
fn kv_bytes_per_token_matches_hand_calculation() {
    // 2 tensors (K and V) x layers x kv_heads x head_dim x 2 bytes for BF16.
    let cases = [
        (presets::llama3_8b(), 2 * 32 * 8 * 128 * 2),
        (presets::llama3_70b(), 2 * 80 * 8 * 128 * 2),
        (presets::llama3_405b(), 2 * 126 * 8 * 128 * 2),
    ];
    for (m, want) in cases {
        assert_eq!(m.kv_bytes_per_token(), want as u64, "{}", m.name);
    }
    // 70B: 320 KiB per token. A single 128k-token sequence is 40 GiB of KV.
    assert_eq!(presets::llama3_70b().kv_bytes_per_token(), 320 * 1024);
}

#[test]
fn quantizing_the_kv_cache_scales_its_cost_directly() {
    let mut m = presets::llama3_70b();
    let bf16 = m.kv_bytes_per_token();
    m.kv_dtype = DType::F8E4M3;
    assert_eq!(m.kv_bytes_per_token(), bf16 / 2, "FP8 KV halves the cost");
}

#[test]
fn tensor_parallelism_divides_the_kv_cost_per_rank() {
    let m = presets::llama3_70b();
    let whole = m.kv_bytes_per_token();
    for tp in [1, 2, 4, 8] {
        let p = ParallelConfig::tp(tp);
        assert_eq!(p.kv_bytes_per_token_per_rank(&m), whole / tp as u64);
    }
}

// --- parallelism validation -------------------------------------------------

#[test]
fn rejects_a_tensor_degree_the_kv_heads_cannot_divide() {
    let m = presets::llama3_70b(); // 8 KV heads
    assert!(ParallelConfig::tp(8).validate(&m).is_ok());
    let err = ParallelConfig::tp(16).validate(&m).unwrap_err();
    assert!(err.contains("KV heads"), "unhelpful error: {err}");
}

#[test]
fn rejects_a_pipeline_degree_the_layers_cannot_divide() {
    let m = presets::llama3_70b(); // 80 layers
    assert!(ParallelConfig::tp(2).with_pipeline(4).validate(&m).is_ok());
    let err = ParallelConfig::tp(2)
        .with_pipeline(3)
        .validate(&m)
        .unwrap_err();
    assert!(err.contains("layers"), "unhelpful error: {err}");
}

#[test]
fn rejects_expert_parallelism_on_a_dense_model() {
    let m = presets::llama3_70b();
    assert!(ParallelConfig::tp(4).with_expert(2).validate(&m).is_err());

    let m = presets::mixtral_8x22b();
    assert!(ParallelConfig::tp(8).with_expert(8).validate(&m).is_ok());
    assert!(
        ParallelConfig::tp(8).with_expert(3).validate(&m).is_err(),
        "8 experts do not divide by 3"
    );
}

#[test]
fn expert_parallelism_shards_expert_weights_off_each_rank() {
    let m = presets::mixtral_8x22b();
    let tp_only = ParallelConfig::tp(8).weight_bytes_per_rank(&m);
    let tp_and_ep = ParallelConfig::tp(8).with_expert(8).weight_bytes_per_rank(&m);
    assert!(
        tp_and_ep < tp_only,
        "ep=8 must move expert weights off each rank: {tp_and_ep} vs {tp_only}"
    );
}

#[test]
fn collectives_are_reported_for_each_plan_dimension() {
    let m = presets::mixtral_8x22b();
    let ops = ParallelConfig::tp(4).with_pipeline(2).with_expert(8).describe_collectives(&m);
    assert_eq!(ops.len(), 3, "tp, pp and ep each add traffic: {ops:?}");
    assert!(ParallelConfig::default().describe_collectives(&m)[0].contains("single-rank"));
}

// --- capacity planning ------------------------------------------------------

#[test]
fn seventy_b_fits_on_four_h100s_with_room_for_context() {
    let m = presets::llama3_70b();
    let plan = MemoryPlan::new(&m, ParallelConfig::tp(4), presets::H100_80GB, 16, 0.10).unwrap();

    // 70.6B params in BF16, quartered: roughly 33 GiB of weights per rank.
    assert!(
        (30.0..40.0).contains(&plan.weight_gib()),
        "unexpected weight footprint: {:.1} GiB",
        plan.weight_gib()
    );
    assert!(plan.max_context() > 100_000, "should hold a long context");
    assert!(plan.concurrent_sequences(4096) > 8);
    assert!(
        plan.concurrent_sequences(4096) > plan.concurrent_sequences(32_768),
        "longer contexts must reduce concurrency"
    );
}

/// The planner has to *refuse* the impossible, not just approve the possible.
/// 405B in BF16 is roughly 756 GiB of weights; eight 80 GiB cards hold 640 GiB.
#[test]
fn negative_control_405b_in_bf16_does_not_fit_on_eight_h100s() {
    let m = presets::llama3_405b();
    let err = MemoryPlan::new(&m, ParallelConfig::tp(8), presets::H100_80GB, 16, 0.10).unwrap_err();
    match err {
        PlanError::WeightsDoNotFit { weight_gib, .. } => {
            assert!(
                weight_gib > 80.0,
                "should report the per-rank overflow, got {weight_gib:.1} GiB"
            );
        }
        other => panic!("expected a capacity refusal, got {other:?}"),
    }
}

/// And the same model in FP8 must then fit — otherwise the refusal above could
/// be caused by anything, and quantization would look pointless.
#[test]
fn the_same_405b_fits_on_eight_h100s_once_weights_are_fp8() {
    let mut m = presets::llama3_405b();
    m.weights = WeightFormat::w8_per_tensor(DType::F8E4M3);

    let plan = MemoryPlan::new(&m, ParallelConfig::tp(8), presets::H100_80GB, 16, 0.10).unwrap();
    assert!(
        plan.weight_gib() < 80.0,
        "FP8 weights must fit a rank: {:.1} GiB",
        plan.weight_gib()
    );
    assert!(
        plan.max_context() > 100_000,
        "and leave usable KV: {} tokens",
        plan.max_context()
    );
}

#[test]
fn four_bit_weights_cost_more_than_four_bits_once_scales_are_counted() {
    let mut m = presets::llama3_70b();
    m.weights = WeightFormat::w4_g128();
    let bits = m.weights.effective_bits(m.total_params() as usize);
    assert!(
        bits > 4.0,
        "group scales and zero points are not free, got {bits:.3} bits"
    );
    assert!(bits < 4.5, "but they should stay under 4.5 bits, got {bits:.3}");
}

// --- config.json parsing ----------------------------------------------------

#[test]
fn reads_geometry_from_a_hugging_face_config() {
    let json = r#"{
        "_name_or_path": "meta-llama/Meta-Llama-3.1-70B",
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": false
    }"#;
    let cfg = ModelConfig::from_hf_config(json, WeightFormat::dense(DType::BF16), DType::BF16)
        .expect("valid config");

    let preset = presets::llama3_70b();
    assert_eq!(
        cfg.total_params(),
        preset.total_params(),
        "the parsed checkpoint and the preset must agree"
    );
    assert_eq!(cfg.kv_bytes_per_token(), preset.kv_bytes_per_token());
}

#[test]
fn detects_a_mixture_of_experts_checkpoint() {
    let json = r#"{
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 32000
    }"#;
    let cfg = ModelConfig::from_hf_config(json, WeightFormat::dense(DType::BF16), DType::BF16)
        .expect("valid config");
    assert!(cfg.ffn.is_moe());
    assert!(matches!(
        cfg.ffn,
        FeedForward::Moe { num_experts: 8, experts_per_token: 2, .. }
    ));
    assert!(cfg.total_params() > 3 * cfg.active_params());
}

#[test]
fn a_malformed_config_is_an_error_not_a_default() {
    let err = ModelConfig::from_hf_config(
        r#"{"hidden_size": 4096}"#,
        WeightFormat::dense(DType::BF16),
        DType::BF16,
    );
    assert!(err.is_err(), "missing fields must not silently default");
}

#[test]
fn every_preset_is_internally_consistent() {
    for m in presets::all() {
        m.validate().unwrap_or_else(|e| panic!("{}: {e}", m.name));
        assert!(m.active_params() <= m.total_params(), "{}", m.name);
        assert!(m.kv_bytes_per_token() > 0, "{}", m.name);
    }
}
