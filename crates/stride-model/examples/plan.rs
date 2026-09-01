//! Print capacity plans for the reference models across accelerator profiles.
//!
//!     cargo run -p stride-model --example plan
//!
//! Every number here is arithmetic over the model's own geometry and the
//! vendor's published card specification. Nothing was measured on hardware.

use stride_model::{
    memory::decode_bandwidth_ceiling_tokens_per_s, presets, DType, MemoryPlan, ModelConfig,
    ParallelConfig, WeightFormat,
};

fn row(model: &ModelConfig, parallel: ParallelConfig, gpu: presets::GpuProfile) {
    let label = format!(
        "{} [{}] tp={} pp={} ep={} on {}",
        model.name,
        match model.weights.dtype {
            DType::BF16 => "bf16",
            DType::F8E4M3 => "fp8",
            DType::I4 => "int4-g128",
            other => {
                println!("  {other:?}");
                "other"
            }
        },
        parallel.tensor,
        parallel.pipeline,
        parallel.expert,
        gpu.name
    );
    println!("\n{label}");
    println!("{}", "-".repeat(label.len()));

    match MemoryPlan::new(model, parallel, gpu, 16, 0.10) {
        Ok(plan) => {
            println!("  weights   {:>8.1} GiB/rank", plan.weight_gib());
            println!(
                "  kv cache  {:>8.1} GiB/rank  ({} blocks x 16 tokens)",
                plan.kv_gib(),
                plan.num_blocks
            );
            println!(
                "  context   {:>8} tokens total, {} concurrent @4k, {} @32k",
                plan.max_context(),
                plan.concurrent_sequences(4096),
                plan.concurrent_sequences(32_768),
            );
            let ceiling = decode_bandwidth_ceiling_tokens_per_s(model, parallel, gpu);
            println!(
                "  decode    {:>8.0} tok/s bandwidth ceiling (upper bound, not a measurement)",
                ceiling
            );
            for op in parallel.describe_collectives(model) {
                println!("  fabric    {op}");
            }
        }
        Err(e) => println!("  REFUSED: {e}"),
    }
}

fn main() {
    println!("Stride capacity planner");
    println!("=======================");
    println!(
        "Total / active parameters, KV cost per token, and what survives on a rank.\n\
         Derived from model geometry and published card specs. Not measured."
    );

    for m in presets::all() {
        println!(
            "\n{:<16} {:>7.1}B total  {:>7.1}B active  {:>6.0} KiB KV/token",
            m.name,
            m.total_params() as f64 / 1e9,
            m.active_params() as f64 / 1e9,
            m.kv_bytes_per_token() as f64 / 1024.0,
        );
    }

    println!("\n\nDeployment plans");
    println!("================");

    row(
        &presets::llama3_8b(),
        ParallelConfig::tp(1),
        presets::L40S_48GB,
    );
    row(
        &presets::llama3_70b(),
        ParallelConfig::tp(4),
        presets::H100_80GB,
    );
    row(
        &presets::llama3_405b(),
        ParallelConfig::tp(8),
        presets::H100_80GB,
    );

    let mut fp8_405b = presets::llama3_405b();
    fp8_405b.weights = WeightFormat::w8_per_tensor(DType::F8E4M3);
    row(&fp8_405b, ParallelConfig::tp(8), presets::H100_80GB);

    row(
        &presets::mixtral_8x22b(),
        ParallelConfig::tp(8).with_expert(8),
        presets::H100_80GB,
    );
}
