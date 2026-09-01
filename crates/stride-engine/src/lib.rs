//! The serving loop: admission, batching, execution, sampling and streaming.
//!
//! The engine owns the scheduler and the executor and runs them in a single
//! task. Requests arrive over a channel and tokens leave over per-request
//! channels, so the HTTP layer never touches scheduler state and there is no
//! lock on the generation path.
//!
//! One loop, one owner. Continuous batching means every step's composition
//! depends on the last step's outcome, so splitting the loop across tasks would
//! buy concurrency the algorithm cannot use and cost a synchronisation point
//! per step.

mod stream;

pub use stream::{StreamEvent, TokenStream, Usage};

use std::collections::HashMap;

use stride_backend::{Executor, ForwardPass, IncrementalDecoder, Sampler, SequenceWork, Tokenizer};
use stride_core::{
    Error, FinishReason, Request, RequestId, Result, SamplingParams, SequenceId, ServiceClass,
    Tick, TokenId,
};
use stride_kvcache::{BlockId, KvCacheConfig};
use stride_sched::{Scheduler, SchedulerConfig, SchedulerMetrics};
use tokio::sync::{mpsc, oneshot};

/// A generation request as it arrives from the API layer.
#[derive(Debug)]
pub struct GenerationRequest {
    pub tenant: String,
    pub prompt: String,
    pub params: SamplingParams,
    pub class: ServiceClass,
}

#[derive(Debug)]
enum Command {
    Generate {
        request: Box<GenerationRequest>,
        reply: oneshot::Sender<Result<(RequestId, TokenStream)>>,
    },
    Cancel(RequestId),
    Metrics(oneshot::Sender<EngineMetrics>),
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngineMetrics {
    pub scheduler: SchedulerMetrics,
    pub forward_passes: u64,
    /// Sum of modelled or measured pass durations.
    pub busy_us: u64,
    /// True while the engine is backed by the analytic simulator rather than a
    /// real model, so every latency figure it reports is an estimate.
    pub estimated: bool,
    pub kv_blocks_total: usize,
    pub kv_blocks_live: usize,
    pub kv_blocks_cached: usize,
    pub kv_block_hit_rate: f64,
    pub queued: usize,
    pub running: usize,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub scheduler: SchedulerConfig,
    pub cache: KvCacheConfig,
    /// Requests accepted before the API layer is told to back off.
    pub max_queued_requests: usize,
    /// Sleep to match modelled pass durations, so a simulated deployment
    /// streams at a realistic rate instead of instantly.
    pub realtime: bool,
    /// Seed used when a request does not supply one.
    pub base_seed: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            cache: KvCacheConfig::default(),
            max_queued_requests: 1024,
            realtime: false,
            base_seed: 0x5721_9A3C_11FF_0042,
        }
    }
}

/// Handle to a running engine. Cheap to clone; every clone shares the loop.
#[derive(Debug, Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<Command>,
    capacity: usize,
}

impl EngineHandle {
    /// Submit a request and receive a stream of its tokens.
    ///
    /// Returns [`Error::Backpressure`] rather than blocking when the queue is
    /// full: a serving front end has to shed load, not absorb it silently.
    pub async fn generate(&self, request: GenerationRequest) -> Result<(RequestId, TokenStream)> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .try_send(Command::Generate {
                request: Box::new(request),
                reply,
            })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => Error::Backpressure {
                    queued: self.capacity,
                },
                mpsc::error::TrySendError::Closed(_) => Error::EngineStopped,
            })?;
        rx.await.map_err(|_| Error::EngineStopped)?
    }

    /// Ask the engine to abandon a request. Safe to call after it finished.
    pub async fn cancel(&self, id: RequestId) {
        let _ = self.tx.send(Command::Cancel(id)).await;
    }

    pub async fn metrics(&self) -> Result<EngineMetrics> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Metrics(tx))
            .await
            .map_err(|_| Error::EngineStopped)?;
        rx.await.map_err(|_| Error::EngineStopped)
    }

    pub async fn shutdown(&self) {
        let _ = self.tx.send(Command::Shutdown).await;
    }
}

/// Per-sequence state the engine keeps alongside the scheduler's.
struct Active {
    request: RequestId,
    sampler: Sampler,
    decoder: IncrementalDecoder,
    sink: mpsc::Sender<StreamEvent>,
    prompt_tokens: usize,
    emitted: usize,
}

pub struct Engine<E: Executor, T: Tokenizer> {
    scheduler: Scheduler,
    executor: E,
    tokenizer: T,
    config: EngineConfig,
    active: HashMap<SequenceId, Active>,
    /// Sequences belonging to each request, for cancellation.
    by_request: HashMap<RequestId, Vec<SequenceId>>,
    now: Tick,
    forward_passes: u64,
    busy_us: u64,
    estimated: bool,
}

impl<E: Executor + 'static, T: Tokenizer + 'static> Engine<E, T> {
    pub fn new(config: EngineConfig, executor: E, tokenizer: T) -> Self {
        Self {
            scheduler: Scheduler::new(config.scheduler, config.cache),
            executor,
            tokenizer,
            config,
            active: HashMap::new(),
            by_request: HashMap::new(),
            now: 0,
            forward_passes: 0,
            busy_us: 0,
            estimated: false,
        }
    }

    /// Start the loop on the current runtime and return a handle to it.
    pub fn spawn(mut self) -> EngineHandle {
        let capacity = self.config.max_queued_requests;
        let (tx, mut rx) = mpsc::channel(capacity);
        tokio::spawn(async move {
            loop {
                // Drain every pending command before stepping, so a burst of
                // arrivals is batched into one step rather than one step each.
                let mut stop = false;
                loop {
                    match rx.try_recv() {
                        Ok(cmd) => {
                            if self.handle(cmd) {
                                stop = true;
                            }
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            stop = true;
                            break;
                        }
                    }
                }
                if stop {
                    break;
                }

                if self.scheduler.num_running() == 0 && self.scheduler.num_waiting() == 0 {
                    // Nothing to do: block until something arrives instead of
                    // spinning through empty steps.
                    match rx.recv().await {
                        Some(cmd) => {
                            if self.handle(cmd) {
                                break;
                            }
                        }
                        None => break,
                    }
                    continue;
                }

                self.step().await;
            }
            tracing::info!("engine loop stopped");
        });

        EngineHandle { tx, capacity }
    }

    /// Returns true if the engine should stop.
    fn handle(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::Generate { request, reply } => {
                let _ = reply.send(self.admit(*request));
                false
            }
            Command::Cancel(id) => {
                for finished in self.scheduler.cancel(id) {
                    self.retire(finished.seq, FinishReason::Cancelled);
                }
                false
            }
            Command::Metrics(reply) => {
                let _ = reply.send(self.metrics());
                false
            }
            Command::Shutdown => true,
        }
    }

    fn metrics(&self) -> EngineMetrics {
        let kv = self.scheduler.cache().stats();
        EngineMetrics {
            scheduler: self.scheduler.metrics(),
            forward_passes: self.forward_passes,
            busy_us: self.busy_us,
            estimated: self.estimated,
            kv_blocks_total: kv.capacity,
            kv_blocks_live: kv.live,
            kv_blocks_cached: kv.cached,
            kv_block_hit_rate: kv.block_hit_rate,
            queued: self.scheduler.num_waiting(),
            running: self.scheduler.num_running(),
        }
    }

    fn admit(&mut self, req: GenerationRequest) -> Result<(RequestId, TokenStream)> {
        let prompt = self.tokenizer.encode(&req.prompt);
        let prompt_tokens = prompt.len();

        let request = Request::new(req.tenant, prompt, self.now)
            .with_params(req.params.clone())
            .with_class(req.class);
        let request_id = request.id;

        let seq_ids = self.scheduler.admit(request)?;

        // One channel per request; every sequence of an `n > 1` request shares
        // it and tags its events with a sequence index.
        let (sink, stream) = mpsc::channel(256);
        for (index, &seq) in seq_ids.iter().enumerate() {
            let seed = self
                .config
                .base_seed
                .wrapping_add(request_id.raw())
                .wrapping_add(index as u64);
            self.active.insert(
                seq,
                Active {
                    request: request_id,
                    sampler: Sampler::new(req.params.clone(), self.tokenizer.eos(), seed),
                    decoder: IncrementalDecoder::new(),
                    sink: sink.clone(),
                    prompt_tokens,
                    emitted: 0,
                },
            );
        }
        self.by_request.insert(request_id, seq_ids);
        Ok((request_id, TokenStream::new(stream)))
    }

    async fn step(&mut self) {
        let batch = self.scheduler.step(self.now);
        if batch.is_empty() {
            // Everything is blocked on memory. Advance the clock so deadline
            // ranking still makes progress rather than deadlocking on a tie.
            self.now += 1_000;
            return;
        }

        for seq in &batch.preempted {
            tracing::debug!(?seq, "preempted under memory pressure");
        }

        // Materialise the pass, since the executor borrows token slices while
        // the scheduler needs to stay mutable afterwards.
        struct Owned {
            seq: SequenceId,
            tokens: Vec<TokenId>,
            position: usize,
            blocks: Vec<BlockId>,
            needs_logits: bool,
        }
        let mut owned: Vec<Owned> = Vec::with_capacity(batch.num_seqs());

        for chunk in &batch.prefill {
            let Some(s) = self.scheduler.sequence(chunk.seq) else {
                continue;
            };
            let end = (chunk.start + chunk.len).min(s.total_len());
            owned.push(Owned {
                seq: chunk.seq,
                tokens: s.tokens()[chunk.start..end].to_vec(),
                position: chunk.start,
                blocks: self
                    .scheduler
                    .block_table(chunk.seq)
                    .unwrap_or(&[])
                    .to_vec(),
                // Only the chunk that reaches the end of the context produces
                // the logits a token is sampled from.
                needs_logits: end >= s.total_len(),
            });
        }
        for &seq in &batch.decode {
            let Some(s) = self.scheduler.sequence(seq) else {
                continue;
            };
            let Some(&last) = s.tokens().last() else {
                continue;
            };
            owned.push(Owned {
                seq,
                tokens: vec![last],
                position: s.total_len().saturating_sub(1),
                blocks: self.scheduler.block_table(seq).unwrap_or(&[]).to_vec(),
                needs_logits: true,
            });
        }

        let pass = ForwardPass {
            work: owned
                .iter()
                .map(|o| SequenceWork {
                    seq: o.seq,
                    tokens: &o.tokens,
                    position: o.position,
                    blocks: &o.blocks,
                    needs_logits: o.needs_logits,
                })
                .collect(),
        };

        let result = self.executor.forward(&pass);
        self.forward_passes += 1;
        self.busy_us += result.cost.duration_us;
        self.estimated = result.cost.estimated;

        if self.config.realtime && result.cost.duration_us > 0 {
            tokio::time::sleep(std::time::Duration::from_micros(result.cost.duration_us)).await;
        }
        self.now += result.cost.duration_us.max(1);

        // Sample one token per sequence that produced logits.
        let mut sampled: Vec<(SequenceId, TokenId)> = Vec::with_capacity(result.logits.len());
        let mut stopped: Vec<SequenceId> = Vec::new();
        for item in &result.logits {
            let Some(a) = self.active.get_mut(&item.seq) else {
                continue;
            };
            let generated = self
                .scheduler
                .sequence(item.seq)
                .map_or(0, |s| s.output_len());
            let token = a.sampler.sample(&item.logits, generated);
            if a.sampler.is_stop(token) {
                stopped.push(item.seq);
            } else {
                sampled.push((item.seq, token));
            }
        }

        // Feed accepted tokens back before handling stops, so a sequence that
        // stops this step still reports the tokens it produced earlier.
        let finished = self.scheduler.on_tokens(&sampled, self.now);

        for (seq, token) in &sampled {
            self.emit(*seq, *token).await;
        }
        for seq in stopped {
            self.scheduler.stop(seq);
            self.retire(seq, FinishReason::Stop);
        }
        for f in finished {
            self.retire(f.seq, f.reason);
        }
    }

    async fn emit(&mut self, seq: SequenceId, token: TokenId) {
        let tokenizer = &self.tokenizer;
        let Some(a) = self.active.get_mut(&seq) else {
            return;
        };
        let text = a.decoder.push(tokenizer, token);
        a.emitted += 1;
        let event = StreamEvent::Token {
            index: a.emitted - 1,
            token,
            text,
        };
        // A closed receiver means the client hung up. Nothing to do here; the
        // API layer issues the cancellation.
        let _ = a.sink.send(event).await;
    }

    fn retire(&mut self, seq: SequenceId, reason: FinishReason) {
        let Some(mut a) = self.active.remove(&seq) else {
            return;
        };
        let tail = a.decoder.finish();
        let usage = Usage {
            prompt_tokens: a.prompt_tokens,
            completion_tokens: a.emitted,
            cached_prompt_tokens: self
                .scheduler
                .sequence(seq)
                .map_or(0, |s| s.cached_prefix_len),
        };
        let _ = a.sink.try_send(StreamEvent::Done {
            reason,
            trailing_text: tail,
            usage,
        });
        if let Some(seqs) = self.by_request.get_mut(&a.request) {
            seqs.retain(|&s| s != seq);
            if seqs.is_empty() {
                self.by_request.remove(&a.request);
            }
        }
    }
}
