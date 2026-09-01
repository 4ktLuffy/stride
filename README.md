# Stride

A serving runtime for large language models, written in Rust.

Stride is built around the parts of LLM inference that decide whether a
deployment works at scale: how KV cache memory is paged and shared, how
requests are batched against a latency budget, and whether a model of a given
shape fits on the hardware you have. It targets large dense and
Mixture-of-Experts transformers — 70B and 405B dense, 8x22B MoE — rather than
small models that fit comfortably on one card.

**Status: in development, and validated on hardware.** The full path — control
plane, GPU worker, Triton kernels and tensor parallelism over NCCL — has been
exercised on Llama-3.1-8B and 70B across 1x and 2x L40S 48 GB and 4x H100 80 GB.

| Checked on hardware | Result |
|---|---|
| Paged attention against the PyTorch reference, to 64 pages | max error 7.4e-6 |
| Chunked prefill vs continuous, ten 512-token chunks | equivalent, max logit error 7.8e-6 |
| Tensor parallelism, 8B, tp=1 vs tp=2 | equivalent, min cosine 0.99999964 |
| Tensor parallelism, 70B, tp=4 | equivalent, min cosine 0.9999988 |
| Preemption under sustained load | 386 preemptions, 386 resumptions, no starvation or deadlock |
| Prefix reuse under memory pressure | shared blocks stay resident, no stale hits |

Those figures were measured by a second engineer on their own hardware and are
reproduced as reported; they have not been independently confirmed here. No
throughput number is claimed anywhere in this repository.

**[HANDOFF.md](HANDOFF.md) has the component-by-component detail**, including
what is still unbuilt and the order to bring the runtime up on a new machine.

---

## Why these parts first

Most of what makes large-model serving hard is not the matrix multiply.

A 70B model in BF16 costs **320 KiB of KV cache per token**. One 128k-token
conversation is 40 GiB — half an H100 — before a second user arrives. Whether a
cluster serves 10 concurrent users or 400 is decided by how that cache is
allocated, shared and reclaimed, not by how fast the attention kernel runs.

So Stride starts there: paged allocation, content-addressed prefix sharing, and
a scheduler that knows what memory pressure costs.

---

## What is implemented

### `stride-kvcache` — paged allocation with prefix reuse

KV state lives in fixed-size blocks, so a sequence's cache need not be
contiguous and a long context costs exactly the blocks it uses.

Blocks are **content-addressed**: each block's hash covers its tokens chained
onto the hash of the block before it, so a hash identifies an entire prefix
path rather than 16 tokens in isolation. Two requests sharing a system prompt
share those blocks by reference instead of recomputing them.

The chain is seeded with the tenant id. Cross-tenant sharing is therefore
impossible by construction — byte-identical content under two tenants produces
two different addresses, so the lookup simply misses. There is no isolation
check to forget to write.

Every block is in exactly one of three states — live, cached, or free — and a
cached block is deliberately kept off the free list. If it were on it, an
allocation could hand it out while the index still pointed at it, and a later
cache hit would read another sequence's attention state. The partition makes
that class of bug unrepresentable.

The three counts are maintained by the structures that own them and never
derived from one another, so `assert_invariants` — which runs after every
scheduler step in debug builds — can actually fail. An earlier version computed
`live` as `capacity - free - cached`, which made the partition identity
algebraically true: the check ran constantly and could not detect anything. It
was replaced after a 70B run surfaced a cache-accounting drift that no test had
caught, and the corrected check immediately found a second instance of the same
class.

### `stride-sched` — continuous batching against deadlines

Each step the scheduler decides what runs, under a fixed token budget and a
finite block pool. Three rules:

**Decodes come first.** A sequence already generating holds memory and has a
client watching tokens arrive. Displacing it trades one client's stall for
another's, and the stalled one has already been paid for in memory.

**Admission is ranked by slack, not by class.** A `Background` request that has
waited two minutes outranks an `Interactive` request that arrived this
millisecond. Static priority starves low classes under sustained load; ranking
by time-to-deadline degrades gracefully instead. Both directions are tested.

**Prefill is chunked into whatever budget remains.** A 4000-token prompt does
not get to monopolise a step. It is split across steps so running sequences
keep emitting tokens while it is processed — which is what keeps inter-token
latency flat when a long prompt lands mid-stream.

Under memory pressure the scheduler preempts the sequence with the most slack,
releasing its blocks while keeping its tokens. The victim resumes later by
recomputing — and often finds its own context still resident in the prefix
cache.

### `stride-model` — geometry, quantization and capacity planning

Serving a large model is an arithmetic problem before it is a performance
problem. Weights, activations and KV cache compete for the same HBM, and the
KV cache is whatever survives the other two.

This crate does that subtraction explicitly, so a deployment fails at planning
time instead of with an out-of-memory error mid-request. It covers dense and
MoE geometry, GQA, sub-byte weight formats with their group-scale overhead, and
tensor / pipeline / expert parallelism with hard divisibility validation.

```
$ cargo run -p stride-model --example plan

llama-3.1-8b         8.0B total      8.0B active     128 KiB KV/token
llama-3.1-70b       70.6B total     70.6B active     320 KiB KV/token
llama-3.1-405b     405.8B total    405.8B active     504 KiB KV/token
mixtral-8x7b        46.7B total     12.9B active     128 KiB KV/token
mixtral-8x22b      140.6B total     39.2B active     224 KiB KV/token

llama-3.1-70b [bf16] tp=4 on H100-SXM-80GB
  weights       32.9 GiB/rank
  kv cache      42.4 GiB/rank  (34760 blocks x 16 tokens)
  context     556160 tokens total, 135 concurrent @4k, 16 @32k

llama-3.1-405b [bf16] tp=8 on H100-SXM-80GB
  REFUSED: weights need 94.5 GiB per rank but H100-SXM-80GB has 80.0 GiB.
           Raise tensor or pipeline parallelism, or quantize the weights

llama-3.1-405b [fp8] tp=8 on H100-SXM-80GB
  weights       47.2 GiB/rank
  kv cache      29.5 GiB/rank  (30664 blocks x 16 tokens)
  context     490624 tokens total, 119 concurrent @4k, 14 @32k
```

The refusal is tested as deliberately as the success. A planner that only ever
approves is not validating anything.

---

## Running it

No GPU, simulated backend:

```bash
cargo test --workspace
cargo run --release -p stride-server --bin stride -- --model llama3-8b --gpu l40s
```

With a real model, two processes — see [HANDOFF.md](HANDOFF.md):

```bash
stride-worker --model /path/to/checkpoint --port 9000
stride --worker 127.0.0.1:9000 --tokenizer /path/to/checkpoint --model llama3-8b
```

Or `docker compose -f docker/docker-compose.yml up` with `MODEL` set.

---

## Layout

| Crate | What it does |
|---|---|
| `stride-core` | Requests, sequences, sampling, service classes, logical clock |
| `stride-kvcache` | Paged blocks, content-addressed prefix reuse, LRU eviction |
| `stride-model` | Dense/MoE geometry, quantization formats, parallelism, capacity planning |
| `stride-sched` | Continuous batching, chunked prefill, deadline ranking, preemption |
| `stride-backend` | Tokenizers, sampling, the executor interface, the simulator, the worker client |
| `stride-engine` | The async serving loop and token streaming |
| `stride-server` | OpenAI-compatible HTTP API |

| Python | What it does |
|---|---|
| `stride_worker` | PyTorch execution against the paged cache, over TCP |
| `stride_worker.kernels` | Triton RMSNorm, paged attention, W4A16 GEMM |
| `stride_worker.autotune` | The correctness gate, Pareto search, negative controls |

The runtime never reads the wall clock. Every component takes the current tick
as an argument, which is what makes a scheduling run replayable from a recorded
trace.

Two processes, one model: the Rust control plane owns every scheduling and
allocation decision and never touches the device; the Python worker owns the
weights and the forward pass and makes no decisions. One scheduler, one
allocator, and the side that can leak memory is the side with the test suite.

---

### Tensor parallelism

One rank per GPU under `torchrun`, with the standard Megatron split: q/k/v and
the MLP gate/up projections column-parallel, attention output and MLP down
row-parallel, two all-reduces per layer, and a vocabulary-parallel output
projection gathered once per pass. KV heads are sharded, so each rank stores
only its own.

```bash
torchrun --nproc_per_node=4 -m stride_worker.worker --model /path/to/checkpoint
stride --worker 127.0.0.1:9000 --tokenizer /path/to/checkpoint --model llama3-70b --tp 4
```

Rank 0 owns the socket. The control plane addresses one worker and is unaware
there is more than one GPU — the sharding lives entirely below that seam.

The sharding *math* is verified on CPU: each test splits a real attention block
or MLP by hand, performs the collective by summing the shards directly, and
requires the result to match the unsharded computation. The NCCL path itself is
verified on hardware — 8B at tp=1 against tp=2, and 70B at tp=4 — by comparing
per-layer hidden states rather than generated text.

That distinction cost a day of someone's time and is worth stating plainly.
Splitting a matmul changes the order of a floating-point reduction, addition is
not associative, and greedy decoding takes an argmax — so at any near-tie a
difference far below bf16 precision flips one token and every token after it
diverges. Correct tensor parallelism produces different text. So does broken
tensor parallelism. `stride-diagnose` compares logits and hidden states, and
reports the *first* divergent layer, because error compounds and the last layer
says nothing about where the fault is.

---

## Not built

- Pipeline and expert parallelism. Validated by the planner, not implemented,
  so MoE models replicate every expert on every rank.
- Multi-node. One machine's GPUs only.
- Disaggregated prefill/decode, speculative decoding, LoRA, structured output.
- A real chat template. Messages are flattened as `role: content`.

---

## License

Apache-2.0. See [LICENSE](LICENSE).
