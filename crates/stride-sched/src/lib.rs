//! Continuous-batching scheduler with deadline-ranked admission.
//!
//! Every step the scheduler answers one question: given a fixed token budget
//! and a finite block pool, which sequences run next? Three rules shape the
//! answer.
//!
//! **Decodes come first.** A sequence that is already generating holds blocks
//! and has a client watching tokens arrive. Displacing it to start new work
//! trades one client's stall for another's, and the stalled one has already
//! been paid for in memory.
//!
//! **Admission is ranked by slack, not by class.** A `Background` request that
//! has waited ten minutes outranks a `Interactive` request that arrived this
//! millisecond. Static priority starves the low classes under sustained load;
//! ranking by time-to-deadline degrades gracefully instead.
//!
//! **Prefill is chunked into whatever budget is left.** A 4000-token prompt
//! does not get to monopolise a step. It is split across steps so running
//! sequences keep emitting tokens while it is processed, which is what keeps
//! inter-token latency flat when a long prompt arrives mid-stream.

use std::collections::HashMap;

use stride_core::{
    Error, FinishReason, Request, RequestId, Result, Sequence, SequenceId, SequenceState, Tick,
    TokenId,
};
use stride_kvcache::{BlockId, KvCache, KvCacheConfig};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchedulerConfig {
    /// Largest number of sequences in one forward pass.
    pub max_batch_seqs: usize,
    /// Token budget per step, across prefill and decode.
    pub max_batch_tokens: usize,
    /// Longest prompt + output the model supports.
    pub max_model_len: usize,
    /// Fraction of the block pool held back so running decodes can always grow
    /// by one block. Without it, admitting new work can deadlock generation.
    pub watermark: f64,
    /// Split long prompts across steps instead of running them whole.
    pub chunked_prefill: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_seqs: 256,
            max_batch_tokens: 2048,
            max_model_len: 8192,
            watermark: 0.01,
            chunked_prefill: true,
        }
    }
}

/// A slice of one prompt to run this step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillChunk {
    pub seq: SequenceId,
    /// Offset into the sequence's tokens where this chunk begins.
    pub start: usize,
    pub len: usize,
    /// Prompt tokens this sequence skipped thanks to the prefix cache.
    pub cached: usize,
}

/// What the runtime should execute this step.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScheduledBatch {
    pub prefill: Vec<PrefillChunk>,
    pub decode: Vec<SequenceId>,
    /// Sequences evicted this step. Their tokens are kept; their blocks are not.
    pub preempted: Vec<SequenceId>,
}

impl ScheduledBatch {
    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }

    /// Total tokens in the forward pass.
    pub fn num_tokens(&self) -> usize {
        self.decode.len() + self.prefill.iter().map(|c| c.len).sum::<usize>()
    }

    pub fn num_seqs(&self) -> usize {
        self.decode.len() + self.prefill.len()
    }
}

/// One sequence that reached a terminal state.
#[derive(Debug, Clone)]
pub struct Finished {
    pub seq: SequenceId,
    pub request: RequestId,
    pub reason: FinishReason,
    pub output: Vec<TokenId>,
    pub ttft_us: Option<Tick>,
    pub mean_itl_us: Option<f64>,
    pub cached_prefix_len: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SchedulerMetrics {
    pub admitted: u64,
    pub finished: u64,
    pub preemptions: u64,
    /// Sequences whose first token missed the class TTFT budget.
    pub ttft_deadline_misses: u64,
    pub steps: u64,
    /// Context tokens run through the model, prompts and recomputation alike.
    pub prefill_tokens_computed: u64,
    /// Context tokens skipped because the prefix cache already held them.
    pub prefill_tokens_reused: u64,
}

impl SchedulerMetrics {
    /// Share of context tokens served from the prefix cache rather than run.
    pub fn prefill_reuse_rate(&self) -> f64 {
        let total = self.prefill_tokens_computed + self.prefill_tokens_reused;
        if total == 0 {
            return 0.0;
        }
        self.prefill_tokens_reused as f64 / total as f64
    }
}

pub struct Scheduler {
    cfg: SchedulerConfig,
    cache: KvCache,
    seqs: HashMap<SequenceId, Sequence>,
    tables: HashMap<SequenceId, Vec<BlockId>>,
    waiting: Vec<SequenceId>,
    running: Vec<SequenceId>,
    metrics: SchedulerMetrics,
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig, cache: KvCacheConfig) -> Self {
        Self {
            cfg,
            cache: KvCache::new(cache),
            seqs: HashMap::new(),
            tables: HashMap::new(),
            waiting: Vec::new(),
            running: Vec::new(),
            metrics: SchedulerMetrics::default(),
        }
    }

    pub fn metrics(&self) -> SchedulerMetrics {
        self.metrics
    }

    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn sequence(&self, id: SequenceId) -> Option<&Sequence> {
        self.seqs.get(&id)
    }

    /// Blocks held back from admission so decodes can always grow.
    fn reserved_blocks(&self) -> usize {
        ((self.cache.capacity() as f64 * self.cfg.watermark).ceil() as usize).max(1)
    }

    /// Accept a request and queue one sequence per requested completion.
    pub fn admit(&mut self, request: Request) -> Result<Vec<SequenceId>> {
        request.params.validate()?;
        if request.prompt_len() >= self.cfg.max_model_len {
            return Err(Error::PromptTooLong {
                got: request.prompt_len(),
                max: self.cfg.max_model_len,
            });
        }
        if request.prompt.is_empty() {
            return Err(Error::InvalidSampling("prompt must not be empty".into()));
        }

        let mut ids = Vec::with_capacity(request.n);
        for _ in 0..request.n.max(1) {
            let seq = Sequence::new(
                request.id,
                request.tenant.clone(),
                request.class,
                request.prompt.clone(),
                request.params.max_tokens,
                request.arrived_at,
            );
            let id = seq.id;
            self.seqs.insert(id, seq);
            self.tables.insert(id, Vec::new());
            self.waiting.push(id);
            ids.push(id);
            self.metrics.admitted += 1;
        }
        Ok(ids)
    }

    /// Microseconds until this sequence breaches its latency budget. Negative
    /// once breached. Lower is more urgent.
    fn slack_us(&self, id: SequenceId, now: Tick) -> i64 {
        let Some(seq) = self.seqs.get(&id) else {
            return i64::MAX;
        };
        let deadline = match seq.last_token_at {
            // Generating: measured against the next-token budget.
            Some(last) => last.saturating_add(seq.class.itl_budget_us()),
            // Not yet generating: measured against the first-token budget.
            None => seq.arrived_at.saturating_add(seq.class.ttft_budget_us()),
        };
        deadline as i64 - now as i64
    }

    /// Grow a sequence's block table to cover `num_tokens`, returning false if
    /// the pool cannot satisfy it right now.
    fn ensure_blocks(&mut self, id: SequenceId, num_tokens: usize, reserve: usize) -> bool {
        let have = self.tables.get(&id).map_or(0, |t| t.len());
        let need = self.cache.blocks_for(num_tokens);
        if need <= have {
            return true;
        }
        let want = need - have;
        if self.cache.num_allocatable() < want + reserve {
            return false;
        }
        match self.cache.allocate(want) {
            Ok(mut blocks) => {
                self.tables.entry(id).or_default().append(&mut blocks);
                true
            }
            Err(_) => false,
        }
    }

    /// Release a running sequence's blocks and return it to the queue.
    fn preempt(&mut self, id: SequenceId) {
        if let Some(table) = self.tables.get_mut(&id) {
            let blocks = std::mem::take(table);
            self.cache.release(&blocks);
        }
        if let Some(seq) = self.seqs.get_mut(&id) {
            seq.swap_out();
        }
        self.running.retain(|&r| r != id);
        self.waiting.push(id);
        self.metrics.preemptions += 1;
    }

    /// Pick the running sequence that should yield its memory.
    ///
    /// The victim is the one with the most slack — the sequence least likely to
    /// miss a deadline while it waits. Ties go to the sequence with the fewest
    /// generated tokens, because that is the least work to recompute.
    fn preemption_victim(&self, now: Tick, protect: SequenceId) -> Option<SequenceId> {
        self.running
            .iter()
            .copied()
            .filter(|&id| id != protect)
            .max_by_key(|&id| {
                let lost = self.seqs.get(&id).map_or(0, |s| s.output_len());
                (self.slack_us(id, now), std::cmp::Reverse(lost))
            })
    }

    /// Choose the work for one forward pass.
    ///
    /// Order matters. Decodes are placed first because they already hold
    /// memory and have a client watching. In-flight prefills come next, so a
    /// prompt that has consumed blocks is finished rather than left resident
    /// and idle. New admissions get whatever budget is left.
    pub fn step(&mut self, now: Tick) -> ScheduledBatch {
        self.metrics.steps += 1;
        let mut batch = ScheduledBatch::default();
        let reserve = self.reserved_blocks();
        let running = self.running.clone();

        // 1. Sequences whose context is fully computed generate one token.
        for id in running.iter().copied() {
            if batch.decode.len() >= self.cfg.max_batch_seqs
                || batch.num_tokens() >= self.cfg.max_batch_tokens
            {
                break;
            }
            let Some(seq) = self.seqs.get(&id) else {
                continue;
            };
            if seq.state != SequenceState::Decoding {
                continue;
            }
            let need_tokens = seq.total_len() + 1;

            // Make room by evicting the least urgent peer, never this sequence.
            // Decodes may draw on the watermark reserve; that is what it is for.
            while !self.ensure_blocks(id, need_tokens, 0) {
                match self.preemption_victim(now, id) {
                    Some(victim) => {
                        self.preempt(victim);
                        batch.preempted.push(victim);
                    }
                    None => break,
                }
            }
            if self.tables.get(&id).map_or(0, |t| t.len()) >= self.cache.blocks_for(need_tokens) {
                batch.decode.push(id);
            }
        }

        // 2. Continue prefills already holding blocks.
        for id in running.iter().copied() {
            let budget_left = self.cfg.max_batch_tokens.saturating_sub(batch.num_tokens());
            if budget_left == 0 || batch.num_seqs() >= self.cfg.max_batch_seqs {
                break;
            }
            if self.seqs.get(&id).map(|s| s.state) != Some(SequenceState::Prefilling) {
                continue;
            }
            if let Some(chunk) = self.try_prefill(id, budget_left, reserve) {
                batch.prefill.push(chunk);
            }
        }

        // 3. Admit new work, most urgent first.
        //
        // Key first: slack_us borrows self, so it cannot run inside a sort
        // closure that already holds a mutable borrow of `waiting`.
        let mut ranked: Vec<(i64, SequenceId)> = self
            .waiting
            .iter()
            .map(|&id| (self.slack_us(id, now), id))
            .collect();
        ranked.sort_unstable();

        let mut deferred = Vec::new();
        for (_, id) in ranked {
            let budget_left = self.cfg.max_batch_tokens.saturating_sub(batch.num_tokens());
            if budget_left == 0 || batch.num_seqs() >= self.cfg.max_batch_seqs {
                deferred.push(id);
                continue;
            }
            match self.try_prefill(id, budget_left, reserve) {
                Some(chunk) => batch.prefill.push(chunk),
                None => deferred.push(id),
            }
        }
        self.waiting = deferred;

        debug_assert!(batch.num_tokens() <= self.cfg.max_batch_tokens);
        #[cfg(debug_assertions)]
        self.cache.assert_invariants();
        batch
    }

    /// Try to schedule one prefill chunk.
    ///
    /// Works on the whole computed context, not just the prompt. A sequence
    /// that was preempted mid-generation has to recompute the tokens it had
    /// already produced, and those tokens are part of the context that the
    /// prefix cache may well still hold.
    fn try_prefill(
        &mut self,
        id: SequenceId,
        budget: usize,
        reserve: usize,
    ) -> Option<PrefillChunk> {
        let (tenant, context, fresh) = {
            let seq = self.seqs.get(&id)?;
            (
                seq.tenant.clone(),
                seq.tokens().to_vec(),
                seq.computed_len() == 0,
            )
        };

        // A first attempt consults the prefix cache; a resumed chunk does not,
        // because its earlier blocks are already in the table.
        if fresh {
            let m = self.cache.acquire_prefix(&tenant, &context);
            if !m.blocks.is_empty() {
                self.tables.insert(id, m.blocks.clone());
                if let Some(seq) = self.seqs.get_mut(&id) {
                    seq.adopt_cached_prefix(m.num_tokens);
                }
                self.metrics.prefill_tokens_reused += m.num_tokens as u64;
            }
        }

        let seq = self.seqs.get(&id)?;
        let start = seq.computed_len();
        let remaining = seq.total_len().saturating_sub(start);
        if remaining == 0 {
            // Unreachable: acquire_prefix withholds the final block precisely so
            // there is always at least one token left to run.
            return None;
        }
        let len = if self.cfg.chunked_prefill {
            remaining.min(budget)
        } else if remaining <= budget {
            remaining
        } else {
            return None;
        };
        if len == 0 {
            return None;
        }

        if !self.ensure_blocks(id, start + len, reserve) {
            return None;
        }

        let cached = {
            let seq = self.seqs.get_mut(&id)?;
            seq.advance_computed(len);
            seq.state = if seq.computed_len() >= seq.total_len() {
                SequenceState::Decoding
            } else {
                SequenceState::Prefilling
            };
            seq.cached_prefix_len
        };
        if !self.running.contains(&id) {
            self.running.push(id);
        }
        self.metrics.prefill_tokens_computed += len as u64;

        Some(PrefillChunk {
            seq: id,
            start,
            len,
            cached,
        })
    }

    /// Feed generated tokens back in. Returns the sequences that finished.
    pub fn on_tokens(&mut self, outputs: &[(SequenceId, TokenId)], now: Tick) -> Vec<Finished> {
        let mut done = Vec::new();
        for &(id, token) in outputs {
            let (finished, publish) = {
                let Some(seq) = self.seqs.get_mut(&id) else {
                    continue;
                };
                if seq.state != SequenceState::Decoding {
                    continue;
                }
                let first = seq.first_token_at.is_none();
                seq.push_token(token, now);
                if first {
                    let missed = seq
                        .ttft_us()
                        .is_some_and(|t| t > seq.class.ttft_budget_us());
                    if missed {
                        self.metrics.ttft_deadline_misses += 1;
                    }
                }
                let reason = if seq.is_at_limit() {
                    Some(FinishReason::Length)
                } else {
                    None
                };
                (reason, first)
            };
            let _ = publish;
            if let Some(reason) = finished {
                if let Some(f) = self.retire(id, reason) {
                    done.push(f);
                }
            }
        }
        done
    }

    /// Stop a sequence on an emitted stop token.
    pub fn stop(&mut self, id: SequenceId) -> Option<Finished> {
        self.retire(id, FinishReason::Stop)
    }

    /// Cancel every sequence belonging to a request.
    pub fn cancel(&mut self, request: RequestId) -> Vec<Finished> {
        let ids: Vec<_> = self
            .seqs
            .iter()
            .filter(|(_, s)| s.request == request && !s.state.is_finished())
            .map(|(&id, _)| id)
            .collect();
        ids.into_iter()
            .filter_map(|id| self.retire(id, FinishReason::Cancelled))
            .collect()
    }

    fn retire(&mut self, id: SequenceId, reason: FinishReason) -> Option<Finished> {
        let seq = self.seqs.get_mut(&id)?;
        if seq.state.is_finished() {
            return None;
        }
        seq.finish(reason);

        let out = Finished {
            seq: id,
            request: seq.request,
            reason,
            output: seq.output().to_vec(),
            ttft_us: seq.ttft_us(),
            mean_itl_us: seq.mean_itl_us(),
            cached_prefix_len: seq.cached_prefix_len,
        };
        let (tenant, tokens) = (seq.tenant.clone(), seq.tokens().to_vec());

        // Publish the finished sequence's full blocks before releasing them, so
        // its prompt and output stay reusable by the next matching request.
        if let Some(table) = self.tables.remove(&id) {
            self.cache.publish(&tenant, &tokens, &table);
            self.cache.release(&table);
        }
        self.running.retain(|&r| r != id);
        self.waiting.retain(|&r| r != id);
        self.metrics.finished += 1;
        Some(out)
    }
}
