//! The interface between the scheduler and model execution.
//!
//! The scheduler decides *what* runs; an [`Executor`] runs it. Keeping that
//! boundary narrow is what lets the same scheduling and memory code drive a
//! real GPU backend, a remote worker, or the analytic simulator in this module.

use stride_core::{SequenceId, TokenId};
use stride_kvcache::BlockId;
use stride_model::{presets::GpuProfile, ModelConfig, ParallelConfig};

/// One sequence's contribution to a forward pass.
#[derive(Debug, Clone)]
pub struct SequenceWork<'a> {
    pub seq: SequenceId,
    /// Tokens whose KV entries this pass computes. One token for a decode
    /// step; a chunk of the prompt during prefill.
    pub tokens: &'a [TokenId],
    /// Position of the first of those tokens in the sequence.
    pub position: usize,
    /// Physical blocks backing this sequence's KV cache, in logical order.
    pub blocks: &'a [BlockId],
    /// Whether this pass reaches the sequence's final token and therefore has
    /// to produce logits. False for every prefill chunk but the last.
    pub needs_logits: bool,
}

/// Everything one forward pass executes.
#[derive(Debug, Default)]
pub struct ForwardPass<'a> {
    pub work: Vec<SequenceWork<'a>>,
}

impl ForwardPass<'_> {
    pub fn num_tokens(&self) -> usize {
        self.work.iter().map(|w| w.tokens.len()).sum()
    }

    pub fn num_sequences(&self) -> usize {
        self.work.len()
    }

    /// Tokens belonging to prefill chunks rather than single-token decodes.
    pub fn num_prefill_tokens(&self) -> usize {
        self.work
            .iter()
            .filter(|w| w.tokens.len() > 1)
            .map(|w| w.tokens.len())
            .sum()
    }
}

/// Logits for one sequence's next token.
#[derive(Debug, Clone)]
pub struct SequenceLogits {
    pub seq: SequenceId,
    pub logits: Vec<f32>,
}

/// What a completed pass cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassCost {
    /// Microseconds the pass took, or is modelled to have taken.
    pub duration_us: u64,
    /// True when the figure is a model rather than a measurement.
    pub estimated: bool,
}

#[derive(Debug)]
pub struct PassResult {
    pub logits: Vec<SequenceLogits>,
    pub cost: PassCost,
}

pub trait Executor: Send {
    fn model(&self) -> &ModelConfig;

    /// Size of the distribution this executor produces.
    ///
    /// May be smaller than the model's vocabulary when the executor is a
    /// stand-in; the sampler and tokenizer follow this number, not the config.
    fn vocab_size(&self) -> usize;

    fn forward(&mut self, pass: &ForwardPass) -> PassResult;
}

/// Efficiency assumptions for the analytic model.
///
/// These are the fraction of a card's published peak that a well-written kernel
/// is assumed to reach. They are stated here rather than buried in the
/// arithmetic because they are the least defensible part of any estimate, and
/// the first thing that should be replaced with a measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EfficiencyModel {
    /// Share of peak FLOPs reached during compute-bound prefill.
    pub compute: f64,
    /// Share of peak HBM bandwidth reached during memory-bound decode.
    pub bandwidth: f64,
    /// Fixed per-pass overhead: launch latency, scheduling, synchronisation.
    pub fixed_overhead_us: u64,
}

impl Default for EfficiencyModel {
    fn default() -> Self {
        Self {
            compute: 0.70,
            bandwidth: 0.80,
            fixed_overhead_us: 150,
        }
    }
}

/// An executor that models timing instead of running a model.
///
/// It exists so the whole serving path — admission, batching, paging, prefix
/// reuse, streaming, cancellation — can be exercised end to end without a GPU.
/// Its timings are a roofline estimate, never a measurement, and every
/// `PassCost` it returns is flagged `estimated: true` so nothing downstream can
/// mistake one for the other.
///
/// Its logits are a deterministic function of the context, which makes the
/// whole pipeline reproducible: the same prompt and seed must yield the same
/// completion whether or not the prefix cache was warm.
#[derive(Debug)]
pub struct SimulatedExecutor {
    model: ModelConfig,
    parallel: ParallelConfig,
    gpu: GpuProfile,
    efficiency: EfficiencyModel,
    vocab: usize,
    passes: u64,
}

impl SimulatedExecutor {
    pub fn new(
        model: ModelConfig,
        parallel: ParallelConfig,
        gpu: GpuProfile,
        vocab: usize,
    ) -> Self {
        assert!(vocab > 1, "need a distribution to sample from");
        Self {
            model,
            parallel,
            gpu,
            efficiency: EfficiencyModel::default(),
            vocab,
            passes: 0,
        }
    }

    pub fn with_efficiency(mut self, efficiency: EfficiencyModel) -> Self {
        self.efficiency = efficiency;
        self
    }

    pub fn passes(&self) -> u64 {
        self.passes
    }

    /// Roofline estimate for one pass: the slower of the compute and memory
    /// paths, plus fixed overhead.
    ///
    /// Prefill is compute-bound because it runs many tokens through the weights
    /// at once. Decode is memory-bound because it reads the whole active weight
    /// set to produce a single token per sequence. A mixed batch pays both.
    pub fn estimate_us(&self, pass: &ForwardPass) -> u64 {
        let tokens = pass.num_tokens() as f64;
        if tokens == 0.0 {
            return 0.0 as u64;
        }
        let ranks = self.parallel.world_size() as f64;

        let flops = 2.0 * self.model.active_params() as f64 * tokens;
        let compute_s =
            flops / (self.gpu.bf16_flops_per_s as f64 * self.efficiency.compute * ranks);

        // Weights are read once per pass regardless of how many tokens ride along.
        let weight_bytes = self.parallel.weight_bytes_per_rank(&self.model) as f64;
        let memory_s =
            weight_bytes / (self.gpu.hbm_bandwidth_bytes_per_s as f64 * self.efficiency.bandwidth);

        let seconds = compute_s.max(memory_s);
        (seconds * 1e6) as u64 + self.efficiency.fixed_overhead_us
    }

    /// Deterministic pseudo-logits for a context.
    ///
    /// Biased toward printable ASCII so a demo produces something legible
    /// rather than arbitrary bytes. It carries no linguistic meaning — this
    /// stands in for a model, it does not approximate one.
    fn pseudo_logits(&self, work: &SequenceWork) -> Vec<f32> {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in work.tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= (work.position as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

        let mut rng = crate::sampler::Rng::seed(h);
        let mut logits = vec![-8.0f32; self.vocab];
        // A small legible alphabet, plus a space that is twice as likely.
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz ";
        for &c in ALPHABET {
            if let Some(slot) = logits.get_mut(c as usize) {
                *slot = rng.next_f32() * 4.0 + if c == b' ' { 1.0 } else { 0.0 };
            }
        }
        logits
    }
}

impl Executor for SimulatedExecutor {
    fn model(&self) -> &ModelConfig {
        &self.model
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn forward(&mut self, pass: &ForwardPass) -> PassResult {
        self.passes += 1;
        let logits = pass
            .work
            .iter()
            .filter(|w| w.needs_logits)
            .map(|w| SequenceLogits {
                seq: w.seq,
                logits: self.pseudo_logits(w),
            })
            .collect();
        PassResult {
            logits,
            cost: PassCost {
                duration_us: self.estimate_us(pass),
                estimated: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stride_model::presets;

    fn executor() -> SimulatedExecutor {
        SimulatedExecutor::new(
            presets::llama3_70b(),
            ParallelConfig::tp(4),
            presets::H100_80GB,
            258,
        )
    }

    fn work<'a>(seq: u64, tokens: &'a [TokenId], blocks: &'a [BlockId]) -> SequenceWork<'a> {
        SequenceWork {
            seq: SequenceId(seq),
            tokens,
            position: 0,
            blocks,
            needs_logits: true,
        }
    }

    #[test]
    fn a_decode_pass_is_memory_bound_and_a_long_prefill_is_not() {
        let e = executor();
        let one = [1u32];
        let decode = ForwardPass {
            work: vec![work(1, &one, &[])],
        };
        let long: Vec<TokenId> = (0..8192).collect();
        let prefill = ForwardPass {
            work: vec![work(2, &long, &[])],
        };

        let (d, p) = (e.estimate_us(&decode), e.estimate_us(&prefill));
        assert!(d > 0 && p > 0);
        assert!(
            p > d * 4,
            "8192 tokens of prefill should cost far more than one decode: {p} vs {d}"
        );
    }

    #[test]
    fn batching_decodes_costs_little_more_than_one() {
        // The weights are read once per pass, so extra sequences are close to
        // free until the batch becomes compute-bound. This is the entire
        // economic argument for continuous batching.
        let e = executor();
        let t = [1u32];
        let single = ForwardPass {
            work: vec![work(1, &t, &[])],
        };
        let batched = ForwardPass {
            work: (0..64).map(|i| work(i, &t, &[])).collect(),
        };
        let (a, b) = (e.estimate_us(&single), e.estimate_us(&batched));
        assert!(
            b < a * 2,
            "64 batched decodes should not cost 64x one decode: {b} vs {a}"
        );
    }

    #[test]
    fn logits_are_deterministic_for_the_same_context() {
        let mut e = executor();
        let tokens = [7u32, 8, 9];
        let pass = || ForwardPass {
            work: vec![work(1, &tokens, &[])],
        };
        let a = e.forward(&pass());
        let b = e.forward(&pass());
        assert_eq!(a.logits[0].logits, b.logits[0].logits);
        assert!(
            a.cost.estimated,
            "simulated cost must be flagged as estimated"
        );
    }

    #[test]
    fn different_contexts_give_different_logits() {
        let mut e = executor();
        let (x, y) = ([1u32, 2, 3], [1u32, 2, 4]);
        let a = e.forward(&ForwardPass {
            work: vec![work(1, &x, &[])],
        });
        let b = e.forward(&ForwardPass {
            work: vec![work(1, &y, &[])],
        });
        assert_ne!(a.logits[0].logits, b.logits[0].logits);
    }

    #[test]
    fn only_sequences_that_need_logits_receive_them() {
        let mut e = executor();
        let t = [1u32, 2, 3];
        let pass = ForwardPass {
            work: vec![
                SequenceWork {
                    needs_logits: false,
                    ..work(1, &t, &[])
                },
                SequenceWork {
                    needs_logits: true,
                    ..work(2, &t, &[])
                },
            ],
        };
        let r = e.forward(&pass);
        assert_eq!(r.logits.len(), 1, "mid-prompt chunks produce no logits");
        assert_eq!(r.logits[0].seq, SequenceId(2));
    }

    #[test]
    fn more_ranks_make_a_compute_bound_pass_faster() {
        let long: Vec<TokenId> = (0..16384).collect();
        let build = |tp| {
            SimulatedExecutor::new(
                presets::llama3_70b(),
                ParallelConfig::tp(tp),
                presets::H100_80GB,
                258,
            )
        };
        let pass = ForwardPass {
            work: vec![work(1, &long, &[])],
        };
        assert!(
            build(8).estimate_us(&pass) < build(2).estimate_us(&pass),
            "tensor parallelism should shorten a compute-bound pass"
        );
    }
}
