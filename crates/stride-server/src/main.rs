//! Stride server: an OpenAI-compatible endpoint over the Stride runtime.
//!
//! The KV cache is sized from the model's own geometry and the target
//! accelerator rather than from a hand-picked constant, so a deployment that
//! cannot fit is refused at startup with the arithmetic that refused it.

mod api;
mod openai;

use std::net::SocketAddr;

use clap::{Parser, ValueEnum};
use stride_backend::{ByteTokenizer, HfTokenizer, RemoteExecutor, SimulatedExecutor, Tokenizer};
use stride_engine::{Engine, EngineConfig};
use stride_kvcache::KvCacheConfig;
use stride_model::{presets, DType, MemoryPlan, ModelConfig, ParallelConfig, WeightFormat};
use stride_sched::SchedulerConfig;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelArg {
    Llama3_8b,
    Llama3_70b,
    Llama3_405b,
    Mixtral8x7b,
    Mixtral8x22b,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum GpuArg {
    A100,
    H100,
    H200,
    L40s,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum WeightArg {
    Bf16,
    Fp8,
    Int4,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum KvArg {
    Bf16,
    Fp8,
}

#[derive(Parser, Debug)]
#[command(
    name = "stride",
    about = "OpenAI-compatible serving for large language models",
    long_about = "Sizes its KV cache from model geometry and the target accelerator, \
                  and refuses a deployment that cannot fit rather than failing under load."
)]
struct Args {
    /// Built-in model shape to serve.
    #[arg(long, value_enum, default_value = "llama3-8b")]
    model: ModelArg,

    /// Path to a Hugging Face config.json, overriding --model.
    #[arg(long)]
    model_config: Option<String>,

    /// Accelerator the deployment is planned against.
    #[arg(long, value_enum, default_value = "h100")]
    gpu: GpuArg,

    /// Tensor parallel degree.
    #[arg(long, default_value_t = 1)]
    tp: usize,
    /// Pipeline parallel degree.
    #[arg(long, default_value_t = 1)]
    pp: usize,
    /// Expert parallel degree. MoE models only.
    #[arg(long, default_value_t = 1)]
    ep: usize,

    #[arg(long, value_enum, default_value = "bf16")]
    weights: WeightArg,
    #[arg(long, value_enum, default_value = "bf16")]
    kv_cache: KvArg,

    /// Tokens per KV block.
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    /// Override the planned block count. Normally derived from device memory.
    #[arg(long)]
    num_blocks: Option<usize>,
    /// Share of post-weight memory reserved for activations and workspace.
    #[arg(long, default_value_t = 0.10)]
    activation_reserve: f64,

    /// Token budget per forward pass.
    #[arg(long, default_value_t = 2048)]
    max_batch_tokens: usize,
    /// Sequence budget per forward pass.
    #[arg(long, default_value_t = 256)]
    max_batch_seqs: usize,
    /// Longest prompt plus completion accepted.
    #[arg(long, default_value_t = 8192)]
    max_model_len: usize,
    /// Requests accepted before the server returns 429.
    #[arg(long, default_value_t = 1024)]
    max_queued: usize,

    /// Pace the simulated backend to its modelled latency instead of running
    /// as fast as the host allows.
    #[arg(long)]
    realtime: bool,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Print the capacity plan and exit without serving.
    #[arg(long)]
    dry_run: bool,

    /// Address of a `stride-worker` process holding the real model, as
    /// `host:port`. Without this the analytic simulator is used and responses
    /// are not model output.
    #[arg(long)]
    worker: Option<String>,

    /// Checkpoint directory or tokenizer.json. Required with --worker: serving
    /// a real model through the byte tokenizer would produce fluent nonsense.
    #[arg(long)]
    tokenizer: Option<String>,

    /// Seconds to wait for the worker to accept a connection.
    #[arg(long, default_value_t = 30)]
    worker_timeout: u64,
}

impl Args {
    fn gpu(&self) -> presets::GpuProfile {
        match self.gpu {
            GpuArg::A100 => presets::A100_80GB,
            GpuArg::H100 => presets::H100_80GB,
            GpuArg::H200 => presets::H200_141GB,
            GpuArg::L40s => presets::L40S_48GB,
        }
    }

    fn weight_format(&self) -> WeightFormat {
        match self.weights {
            WeightArg::Bf16 => WeightFormat::dense(DType::BF16),
            WeightArg::Fp8 => WeightFormat::w8_per_tensor(DType::F8E4M3),
            WeightArg::Int4 => WeightFormat::w4_g128(),
        }
    }

    fn kv_dtype(&self) -> DType {
        match self.kv_cache {
            KvArg::Bf16 => DType::BF16,
            KvArg::Fp8 => DType::F8E4M3,
        }
    }

    fn parallel(&self) -> ParallelConfig {
        ParallelConfig {
            tensor: self.tp,
            pipeline: self.pp,
            expert: self.ep,
        }
    }

    fn model_config(&self) -> Result<ModelConfig, String> {
        let mut cfg = match &self.model_config {
            Some(path) => {
                let json = std::fs::read_to_string(path)
                    .map_err(|e| format!("cannot read {path}: {e}"))?;
                ModelConfig::from_hf_config(&json, self.weight_format(), self.kv_dtype())?
            }
            None => {
                let mut c = match self.model {
                    ModelArg::Llama3_8b => presets::llama3_8b(),
                    ModelArg::Llama3_70b => presets::llama3_70b(),
                    ModelArg::Llama3_405b => presets::llama3_405b(),
                    ModelArg::Mixtral8x7b => presets::mixtral_8x7b(),
                    ModelArg::Mixtral8x22b => presets::mixtral_8x22b(),
                };
                c.weights = self.weight_format();
                c.kv_dtype = self.kv_dtype();
                c
            }
        };
        cfg.max_position_embeddings = cfg.max_position_embeddings.max(self.max_model_len);
        cfg.validate()?;
        Ok(cfg)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "stride=info,tower_http=warn".into()),
        )
        .init();

    let args = Args::parse();
    let model = args.model_config()?;
    let parallel = args.parallel();
    let gpu = args.gpu();

    let plan = MemoryPlan::new(
        &model,
        parallel,
        gpu,
        args.block_size,
        args.activation_reserve,
    )?;
    println!("{}\n", plan.summary());

    if args.dry_run {
        return Ok(());
    }

    let num_blocks = args.num_blocks.unwrap_or(plan.num_blocks);
    let engine_config = |blocks: usize| EngineConfig {
        scheduler: SchedulerConfig {
            max_batch_seqs: args.max_batch_seqs,
            max_batch_tokens: args.max_batch_tokens,
            max_model_len: args.max_model_len,
            watermark: 0.01,
            chunked_prefill: true,
        },
        cache: KvCacheConfig {
            num_blocks: blocks,
            block_size: args.block_size,
        },
        max_queued_requests: args.max_queued,
        realtime: args.realtime,
        ..Default::default()
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    match &args.worker {
        Some(endpoint) => {
            let tokenizer_path = args.tokenizer.as_ref().ok_or(
                "--tokenizer is required with --worker: serving a real model through \
                 the byte tokenizer would produce fluent nonsense rather than an error",
            )?;
            let tokenizer = HfTokenizer::from_path(tokenizer_path)?;

            tracing::info!("connecting to worker at {endpoint}");
            let executor = RemoteExecutor::connect(
                endpoint.as_str(),
                model.clone(),
                std::time::Duration::from_secs(args.worker_timeout),
            )?;
            let info = executor.info().clone();
            tracing::info!(
                "worker ready on {} as {}: tp={}, {} blocks x {} tokens, vocab {}",
                info.device,
                info.dtype,
                info.tensor_parallel_size,
                info.num_blocks,
                info.block_size,
                info.vocab_size
            );
            if info.tensor_parallel_size != args.tp {
                return Err(format!(
                    "tensor parallel mismatch: this server planned tp={} but the \
                     worker is sharded across {} ranks. The capacity plan and the \
                     KV cache sizing would both be wrong. Relaunch the worker with \
                     `torchrun --nproc_per_node={}`, or pass --tp {}.",
                    args.tp, info.tensor_parallel_size, args.tp, info.tensor_parallel_size
                )
                .into());
            }
            if info.vocab_size != tokenizer.vocab_size() {
                tracing::warn!(
                    "tokenizer vocabulary is {} but the worker reports {}; \
                     ids near the boundary may not round-trip",
                    tokenizer.vocab_size(),
                    info.vocab_size
                );
            }
            if info.block_size != args.block_size {
                return Err(format!(
                    "block size mismatch: the server plans {} tokens per block, \
                     the worker allocated {}. Restart one of them to agree.",
                    args.block_size, info.block_size
                )
                .into());
            }

            // The worker owns the blocks, so its count is authoritative.
            let engine = Engine::new(engine_config(info.num_blocks), executor, tokenizer);
            serve(engine.spawn(), model.name.clone(), false, addr).await?;
        }
        None => {
            let tokenizer = ByteTokenizer;
            let executor =
                SimulatedExecutor::new(model.clone(), parallel, gpu, tokenizer.vocab_size());
            let engine = Engine::new(engine_config(num_blocks), executor, tokenizer);

            tracing::warn!(
                "no --worker given: serving the analytic SIMULATOR. Responses are not \
                 model output and every latency figure is an estimate."
            );
            serve(engine.spawn(), model.name.clone(), true, addr).await?;
        }
    }

    Ok(())
}

async fn serve(
    engine: stride_engine::EngineHandle,
    model_name: String,
    simulated: bool,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = api::AppState {
        engine,
        model_name,
        simulated,
    };
    let app = api::router(state.clone())
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await?;

    state.engine.shutdown().await;
    Ok(())
}
