//! Token sampling: temperature, top-k and nucleus filtering over logits.
//!
//! Sampling is separated from execution so it can be tested against synthetic
//! logits, where the correct answer is known exactly. A sampler validated only
//! through a real model is validated against nothing in particular.
//!
//! Every draw comes from an explicitly seeded generator. A request that does
//! not supply a seed is given one and told what it was, so any generation can
//! be reproduced from its response alone.

use stride_core::{SamplingParams, TokenId};

/// Small deterministic generator.
///
/// xorshift64* seeded through splitmix64, so a caller passing `seed: 0` or a
/// run of sequential seeds still gets well-separated streams.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn seed(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self {
            state: (z ^ (z >> 31)).max(1),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits give exactly the mantissa precision of an f32.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Applies sampling parameters to a logits vector and draws one token.
#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
    /// Suppressed until `min_tokens` have been produced.
    eos: TokenId,
}

impl Sampler {
    pub fn new(params: SamplingParams, eos: TokenId, fallback_seed: u64) -> Self {
        let seed = params.seed.unwrap_or(fallback_seed);
        Self {
            params,
            rng: Rng::seed(seed),
            eos,
        }
    }

    /// The seed actually in use, so it can be reported back to the client.
    pub fn effective_seed(&self) -> u64 {
        self.params.seed.unwrap_or(0)
    }

    /// Draw one token. `generated` is how many tokens this sequence has already
    /// produced, which governs whether EOS is allowed yet.
    pub fn sample(&mut self, logits: &[f32], generated: usize) -> TokenId {
        assert!(
            !logits.is_empty(),
            "cannot sample from an empty distribution"
        );
        let mut work: Vec<f32> = logits.to_vec();

        // Forbid stopping before the caller's floor.
        if generated < self.params.min_tokens {
            if let Some(slot) = work.get_mut(self.eos as usize) {
                *slot = f32::NEG_INFINITY;
            }
        }

        if self.params.is_greedy() {
            return argmax(&work);
        }

        // Temperature first: it changes the ranking's sharpness, not its order,
        // so filtering afterwards operates on the intended distribution.
        let t = self.params.temperature;
        if t > 0.0 && (t - 1.0).abs() > f32::EPSILON {
            for l in &mut work {
                *l /= t;
            }
        }

        let mut probs = softmax(&work);

        // Rank once; both filters need the same ordering.
        let mut order: Vec<u32> = (0..probs.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            probs[b as usize]
                .partial_cmp(&probs[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = order.len();
        if let Some(k) = self.params.top_k {
            keep = keep.min(k as usize);
        }
        if self.params.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut nucleus = 0;
            for (rank, &idx) in order.iter().enumerate() {
                cumulative += probs[idx as usize];
                nucleus = rank + 1;
                if cumulative >= self.params.top_p {
                    break;
                }
            }
            // Always keep at least one token, whatever top_p asks for.
            keep = keep.min(nucleus.max(1));
        }

        for &idx in &order[keep..] {
            probs[idx as usize] = 0.0;
        }

        let total: f32 = order[..keep].iter().map(|&i| probs[i as usize]).sum();
        if total <= 0.0 || !total.is_finite() {
            return order[0];
        }

        let mut target = self.rng.next_f32() * total;
        for &idx in &order[..keep] {
            target -= probs[idx as usize];
            if target <= 0.0 {
                return idx;
            }
        }
        order[keep - 1]
    }

    /// True if this token ends the sequence.
    pub fn is_stop(&self, token: TokenId) -> bool {
        if token == self.eos && !self.params.ignore_eos {
            return true;
        }
        self.params.stop_tokens.contains(&token)
    }
}

fn argmax(logits: &[f32]) -> TokenId {
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as TokenId
}

/// Numerically stable softmax: subtract the maximum before exponentiating, or
/// large logits overflow to infinity and the result is all NaN.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Every logit was -inf; fall back to uniform over the vocabulary.
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    let mut out: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for p in &mut out {
            *p /= sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: TokenId = 256;

    fn params(f: impl FnOnce(&mut SamplingParams)) -> SamplingParams {
        let mut p = SamplingParams {
            seed: Some(42),
            ..Default::default()
        };
        f(&mut p);
        p
    }

    #[test]
    fn greedy_always_takes_the_argmax() {
        let logits = vec![0.1, 5.0, 0.3, 4.9];
        let mut s = Sampler::new(SamplingParams::greedy(16), EOS, 0);
        for _ in 0..20 {
            assert_eq!(s.sample(&logits, 0), 1);
        }
    }

    #[test]
    fn top_k_of_one_is_greedy() {
        let logits = vec![0.1, 5.0, 0.3, 4.9];
        let mut s = Sampler::new(params(|p| p.top_k = Some(1)), EOS, 0);
        for _ in 0..20 {
            assert_eq!(s.sample(&logits, 0), 1);
        }
    }

    #[test]
    fn top_k_never_draws_outside_the_top_k() {
        // Ranks: 1 (5.0), 3 (4.0), then 0 and 2.
        let logits = vec![1.0, 5.0, 0.5, 4.0];
        let mut s = Sampler::new(
            params(|p| {
                p.top_k = Some(2);
                p.temperature = 2.0;
            }),
            EOS,
            0,
        );
        for _ in 0..500 {
            let t = s.sample(&logits, 0);
            assert!(t == 1 || t == 3, "sampled {t}, which is outside the top 2");
        }
    }

    #[test]
    fn nucleus_sampling_restricts_the_support() {
        // One token holds almost all the mass.
        let logits = vec![10.0, 0.0, 0.0, 0.0];
        let mut s = Sampler::new(params(|p| p.top_p = 0.5), EOS, 0);
        for _ in 0..500 {
            assert_eq!(s.sample(&logits, 0), 0, "top_p=0.5 leaves only the peak");
        }
    }

    #[test]
    fn top_p_always_keeps_at_least_one_token() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let mut s = Sampler::new(params(|p| p.top_p = 0.01), EOS, 0);
        let t = s.sample(&logits, 0);
        assert!((t as usize) < 4, "must still return a valid token, got {t}");
    }

    #[test]
    fn the_same_seed_reproduces_the_same_draws() {
        let logits = vec![1.0, 1.2, 0.9, 1.1, 1.05];
        let draw = |seed| {
            let mut s = Sampler::new(
                params(|p| {
                    p.seed = Some(seed);
                    p.temperature = 1.5;
                }),
                EOS,
                0,
            );
            (0..64).map(|_| s.sample(&logits, 0)).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7), "a seeded run must be reproducible");
        assert_ne!(draw(7), draw(8), "different seeds must diverge");
    }

    #[test]
    fn sampling_follows_the_distribution() {
        // Token 0 should win roughly e^2 : 1 against each of the others.
        let logits = vec![2.0, 0.0, 0.0, 0.0];
        let mut s = Sampler::new(params(|p| p.seed = Some(1)), EOS, 0);
        let n = 20_000;
        let hits = (0..n).filter(|_| s.sample(&logits, 0) == 0).count();
        let want = 2.0f64.exp() / (2.0f64.exp() + 3.0);
        let got = hits as f64 / n as f64;
        assert!(
            (got - want).abs() < 0.02,
            "expected token 0 about {want:.3} of the time, saw {got:.3}"
        );
    }

    #[test]
    fn min_tokens_suppresses_early_stopping() {
        // EOS is by far the most likely token.
        let mut logits = vec![0.0; 300];
        logits[EOS as usize] = 20.0;

        let mut s = Sampler::new(
            params(|p| {
                p.min_tokens = 5;
                p.temperature = 0.0;
            }),
            EOS,
            0,
        );
        for generated in 0..5 {
            assert_ne!(
                s.sample(&logits, generated),
                EOS,
                "EOS must be forbidden before min_tokens"
            );
        }
        assert_eq!(
            s.sample(&logits, 5),
            EOS,
            "and allowed once the floor is met"
        );
    }

    #[test]
    fn softmax_survives_extreme_logits() {
        let logits = vec![1e30, 1.0, -1e30];
        let p = softmax(&logits);
        assert!(p.iter().all(|x| x.is_finite()), "overflowed: {p:?}");
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[0] > 0.99);
    }

    #[test]
    fn an_all_forbidden_distribution_still_returns_a_token() {
        let logits = vec![f32::NEG_INFINITY; 8];
        let mut s = Sampler::new(params(|p| p.temperature = 1.0), EOS, 0);
        let t = s.sample(&logits, 0);
        assert!((t as usize) < 8, "must not panic or return garbage");
    }

    #[test]
    fn stop_tokens_are_recognised() {
        let s = Sampler::new(params(|p| p.stop_tokens = vec![13, 99]), EOS, 0);
        assert!(s.is_stop(EOS));
        assert!(s.is_stop(13));
        assert!(s.is_stop(99));
        assert!(!s.is_stop(5));

        let s = Sampler::new(params(|p| p.ignore_eos = true), EOS, 0);
        assert!(!s.is_stop(EOS), "ignore_eos must let generation continue");
    }
}
