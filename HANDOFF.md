# Handoff

Everything you need to run Stride on real hardware, and an honest account of
what has and has not been executed.

---

## Status of every component

| Component | Built | Executed | Where |
|---|---|---|---|
| `stride-core` — requests, sequences, sampling params | yes | yes | CI |
| `stride-kvcache` — paged blocks, prefix reuse, eviction | yes | yes | CI |
| `stride-model` — geometry, quantization, capacity planning | yes | yes | CI |
| `stride-sched` — batching, chunked prefill, preemption | yes | yes | CI |
| `stride-backend` — tokenizer, sampler, executor interface | yes | yes | CI |
| `stride-engine` — the async serving loop | yes | yes | end to end, simulated |
| `stride-server` — OpenAI-compatible API | yes | yes | end to end, simulated |
| `RemoteExecutor` — Rust client for the GPU worker | yes | **no** | needs a worker |
| `stride_worker` — PyTorch execution, paged cache | yes | **no** | needs a GPU |
| Tensor parallelism — sharding math | yes | yes | CI, on CPU |
| Tensor parallelism — NCCL collectives | yes | **no** | needs 2+ GPUs |
| Triton kernels — RMSNorm, paged attention, W4A16 GEMM | yes | **no** | needs a GPU |
| Autotuner — correctness gate and negative controls | yes | yes | CI, on CPU |
| Autotuner — Pareto search over kernel configs | yes | **no** | needs Triton |

**Nothing marked "no" above has ever run.** They were written on a Mac
with no CUDA device. The Rust side compiles and the Python side parses; neither
is evidence that either works. Assume the first run finds bugs, and read the
"what will break first" section before you start.

No performance number appears anywhere in this repository, because none has
been measured.

---

## What hardware you actually need

**NVIDIA GPUs, not CPUs.** This is worth stating plainly because it is easy to
assume otherwise: the Rust control plane runs on CPU and is not the bottleneck,
so a fast CPU buys nothing here. Everything that needs testing — the forward
pass, the Triton kernels, the autotuner's timings — needs CUDA.

The worker does accept `--device cpu`, and that is genuinely useful for checking
correctness on a small model without booking a GPU. It is not useful for a large
one: a 70B forward pass on CPU is minutes per token, and `triton_available()`
hard-gates on `torch.cuda.is_available()`, so no kernel in `kernels/` will run at
all. CPU proves the model code is right. It cannot prove anything about speed.

### Checkpoint compatibility — check this first

The worker implements Llama-family geometry only:

```python
SUPPORTED_ARCHITECTURES = {"LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM"}
```

It reads `architectures` from `config.json` and **refuses to load anything
else**, deliberately — running a different architecture through this code would
produce wrong activations silently rather than an error. So Llama 3.x, Mistral,
Mixtral, Qwen2/2.5 and their finetunes work. DeepSeek-V3 (multi-head latent
attention), Gemma, Phi, GPT-OSS and Command-R do not, and adding one is real
work in `model.py`, not a config change.

Check before booking anything:

```bash
python -c "import json;print(json.load(open('/path/to/checkpoint/config.json'))['architectures'])"
```

### Tiers

Capacity figures below come from `stride --dry-run`, which is arithmetic over
the model geometry and the card's published specification. They tell you what
fits, not how fast it runs.

| What you have | What you can test |
|---|---|
| No GPU | Rust suite, simulator end to end, the gate self-check, the sharding math. All green in CI. |
| **1x 24-48 GB** (L40S, A6000, 4090) | **Start here.** 8B end to end for real, all three Triton kernels, the full autotuner. Everything most likely to be broken is broken here, and it is the cheapest place to find it. |
| 1x 80 GB (A100/H100) | 8B at long context, or 70B quantised to INT4 (34.9 GiB of weights, ~133k tokens of KV). |
| 2x 80 GB | 70B in FP8 across two ranks: 32.9 GiB of weights each, ~278k tokens of KV, 67 concurrent at 4k. |
| 4x 80 GB | 70B in BF16: 32.9 GiB each, ~556k tokens of KV, 135 concurrent at 4k. |
| 8x 80 GB | 405B in FP8: 47.2 GiB each, ~490k tokens of KV, 119 concurrent at 4k. |

### Tensor parallelism

Launch one rank per GPU under `torchrun`:

```bash
torchrun --nproc_per_node=4 -m stride_worker.worker \
    --model /path/to/Llama-3.1-70B-Instruct --port 9000
```

Rank 0 owns the socket; the control plane still addresses a single worker and is
unaware there is more than one GPU. Pass a matching `--tp` to the server — a
mismatch is refused at startup, because the capacity plan and the KV sizing
would both be wrong.

The split is the standard Megatron one: q/k/v and the MLP gate/up projections
are column-parallel, the attention output and MLP down projections are
row-parallel, and two all-reduces per layer put the partial sums back together.
The output projection is vocabulary-parallel with a single gather per pass. KV
heads are sharded, so each rank stores only its own — which is why a given block
count costs less per rank as the degree rises.

**The limit is KV heads.** Llama-3.1-70B has 8, so tensor parallelism works up
to 8 ranks and is refused above that with a message saying so. Going further
would mean replicating KV heads, which is not implemented.

Still single-node: `--nproc_per_node` covers the GPUs in one machine. Pipeline
parallelism across machines is validated by the planner but not implemented.

### Recommendation

Still start with one cheap card. The sharding math is verified on CPU, but no
NCCL code has ever run, so the single-GPU path failing would tell you the same
thing for a fraction of the cost. Once 8B works on one card, move straight to
the multi-GPU node.

---

## What to run, in order

Each step is a checkpoint. If one fails, stop there — the later steps depend on
it and will fail confusingly.

### 1. The Rust side, no GPU (2 minutes)

```bash
cargo test --workspace
cargo run -p stride-model --example plan
```

Should be all green. If not, that is a bug on our side, not your environment.

### 2. The serving path against the simulator, no GPU (5 minutes)

```bash
cargo run --release -p stride-server --bin stride -- --model llama3-8b --gpu l40s --port 8000
```

In another shell:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

The text will be gibberish — that is the simulator, and it is labelled as such
in `/health` and in the `stride_backend_estimated` metric. What you are checking
is that admission, batching, paging, streaming and usage accounting all work.

Send the same long system prompt twice and watch `stride_cached_prompt_tokens`
go from 0 to nearly the whole prompt. That is the prefix cache working.

### 3. The correctness gate, CPU only, needs PyTorch (5 minutes)

This one already runs green in CI on every push, so it should pass first try.
Run it anyway on your machine — it is the check everything else rests on.

```bash
cd python && pip install -e '.[dev]' && stride-autotune verify
```

This is the most important check in the repository and it needs no GPU. It runs
eight deliberately broken RMSNorm implementations through the gate — a missing
epsilon, a wrong reduction axis, an FP16 accumulator, an off-by-one tail mask,
an injected NaN, a race-condition stand-in, a 0.5% scale error — and requires
the gate to reject every one, while the reference passes against itself.

**If any control is ACCEPTED, stop.** The gate is broken, and nothing it
approves afterwards means anything.

### 4. The worker, one GPU (30 minutes, first real hardware)

```bash
pip install torch safetensors transformers  # CUDA build
stride-worker --model /path/to/Llama-3.1-8B-Instruct --port 9000
```

Then point the control plane at it:

```bash
cargo run --release -p stride-server --bin stride -- \
  --model llama3-8b --gpu <your card> \
  --worker 127.0.0.1:9000 \
  --tokenizer /path/to/Llama-3.1-8B-Instruct \
  --port 8000
```

`/health` should now report `"backend": "device"` and
`stride_backend_estimated` should be `0`. **Now the output should be real
text.** If it is fluent but wrong, suspect the tokenizer or the RoPE scaling
before you suspect attention.

### 5. Multi-GPU, two or more cards

```bash
torchrun --nproc_per_node=2 -m stride_worker.worker \
    --model /path/to/Llama-3.1-8B-Instruct --port 9000

cargo run --release -p stride-server --bin stride -- \
  --model llama3-8b --tp 2 --gpu <your card> \
  --worker 127.0.0.1:9000 --tokenizer /path/to/Llama-3.1-8B-Instruct
```

Watch `nvidia-smi` during a run: every card should show utilisation. If only
one does, the followers are stuck in `receive_work` and never got the broadcast.

**Do not compare generated text between tp=1 and tp=2.** An earlier version of
this document told you to, and that instruction was wrong. Splitting a matmul
changes the order of a floating-point reduction, floating-point addition is not
associative, and greedy decoding takes an argmax — so at any near-tie, a
difference far below bf16 precision flips one token, and every token after it
diverges. Correct tensor parallelism produces different text. So does broken
tensor parallelism. The comparison cannot tell them apart.

Compare hidden states instead:

```bash
python -m stride_worker.diagnose.cli tp-dump --model CKPT --out tp1.pt
torchrun --nproc_per_node=2 -m stride_worker.diagnose.cli tp-dump --model CKPT --out tp2.pt
python -m stride_worker.diagnose.cli tp-compare tp1.pt tp2.pt
```

This reports the **first** layer at which the two runs stop agreeing, which is
the only informative one — error compounds, so by the last layer everything
disagrees and tells you nothing about where the fault is. It also states
whether the final logits differ by more than reduction-order noise, and says so
explicitly when they do not.

### 6. Diagnostics when something looks wrong

```bash
stride-diagnose prefill   --model CKPT --tokens 5000 --chunk 512,1024,2048
stride-diagnose attention --contexts 16,32,64,256,1024,4096
```

`prefill` compares a continuous prefill against chunked prefills of the same
prompt, on logits rather than text. `attention` sweeps the paged-attention
kernel against the PyTorch reference across growing contexts, so the answer is
a boundary — "agrees to 4 pages, diverges at 5" — rather than a bare fail, and
asserts the kernel refuses prefill-shaped input.

### 7. Kernel autotuning, one GPU

```bash
stride-autotune rmsnorm --device cuda --dtype bfloat16 --out rmsnorm.json --verbose
```

Explores 75 configurations, gates each against the PyTorch reference before
timing it, and reports a Pareto front over latency and peak memory. The report
records the device, driver, library versions, seed and shapes, so a number in it
can be reproduced or disputed later.

---

## Validated on hardware

Second run, 1x and 2x L40S 48 GB and 4x H100 80 GB, Llama-3.1-8B and 70B BF16.
All three originally suspected GPU bugs are resolved: one was real, two were
artefacts of comparing generated text.

| | |
|---|---|
| Paged attention, decode contract | verified against the reference to 64 pages, max error 7.4e-6 |
| Paged attention, prefill-shaped input | refused, as it must be |
| Chunked prefill equivalence | equivalent, max logit error 7.8e-6 across ten 512-token chunks |
| Tensor parallelism, 8B tp=1 vs tp=2 | equivalent, min cosine 0.99999964, zero decisive disagreements |
| Tensor parallelism, 70B tp=4 | equivalent, min cosine 0.9999988 |
| Preemption under load | 386 preemptions, 386 resumptions, no starvation, no deadlock |
| Prefix reuse under pressure | shared blocks stay resident, no stale hits |

The diagnostics were also checked in the failing direction on hardware: an
injected structural change was caught and localised to the layer it was
injected at; noise at 1e-5 passed. Green in both directions, not just green.

**One caveat on the 70B result.** It compared tp=4 against a reference that
reconstructs through the same partition path, so a fault in the sharding would
appear on both sides and cancel. The 8B tp=1 vs tp=2 comparison is the one that
independently validates the sharding code; the 70B run confirms it survives at
scale and at tp=4. A stronger 70B check available on the same node is **tp=2 vs
tp=4** — 70B in BF16 is about 70 GiB per rank at tp=2, which fits an 80 GiB
card, and the two configurations shard at different boundaries with different
collective sizes.

Throughput figures from that run (~2.05k aggregate tok/s at 128 concurrent on
4x H100) are the tester's measurements, reproduced here as reported. They have
not been independently confirmed.

---

## Findings from the first hardware run

Reported against 1x and 2x L40S 48 GB with Llama-3.1-8B-Instruct. Passing:
the Rust control plane, the simulator, prefix caching and tenant isolation, the
autotuner gate (8/8 controls rejected), CUDA init, weight loading, generation,
and the RMSNorm and W4A16 kernels.

**1. Paged attention is decode-only, and did not say so. Fixed.**

Measured cosine against the PyTorch reference: ~0.9999 for a single query at
the end of a short context, ~0.997 with more KV pages, worse under chunked
prefill.

That gradient is the whole diagnosis. The kernel takes `q` of shape
`(num_seqs, num_q_heads, head_dim)` — one query token per sequence — and
contains no `query_start` and no causal mask. For decode that is correct: the
single query sits at the end and may attend to everything. Give it several
query positions and every query attends to its own future, so the error grows
with the number of keys that should have been masked. Not arithmetic drift; a
missing capability being silently misused.

The kernel now raises `NotADecodeStep` rather than computing. A prefill-capable
variant needs a query-position argument and a mask, and is not written.

**2. Chunked prefill and tensor parallelism: not yet established either way.**

Both were reported as generated text differing from a reference run. That
comparison cannot settle it, for the reason given in step 5 — and the
instruction to make it was mine. `stride-diagnose prefill` and the `tp-dump` /
`tp-compare` pair ask the question properly, on logits and hidden states.

Two outcomes are possible and they need different responses. If the diagnostics
report equivalence, there is no bug and the differing text was decoding
amplifying noise. If they report divergence, `tp-compare` names the first bad
layer and `prefill` names the chunk size at which agreement breaks.

**3. 70B was correctly not benchmarked.** Publishing throughput for a runtime
whose numerics are unresolved would produce a number that is precise and
meaningless. The right call.

---

## What will break first

Ranked by how likely I think each is, having written but not run any of it.

1. **The Triton kernels.** Written blind. The paged attention kernel's online
   softmax rescaling and the W4A16 nibble unpacking are the two places where a
   sign or an index error is most likely, and both produce plausible-looking
   output rather than a crash. The gate exists precisely for this — run
   step 3 before you trust step 5.
2. **RoPE scaling on long contexts.** `build_rope_cache` implements Llama-3
   frequency scaling from the config. Short prompts will look fine either way;
   quality degrades past the original context length if it is wrong.
3. **The chunked-prefill causal mask.** `paged_attention` takes `query_start` so
   a resumed chunk masks against absolute positions. Get this wrong and a long
   prompt produces subtly worse output than a short one — compare a 5000-token
   prompt against the same prompt run through `transformers` directly.
4. **The wire protocol under load.** Framing is length-prefixed and tested by
   inspection only. A truncated frame should raise, not hang; if the server
   stalls with no error, look here.
5. **KV cache sizing.** The worker sizes blocks from free device memory after
   the weights land. If you OOM during a long-context run, lower
   `--activation-reserve` expectations by raising it (it reserves *more*), or
   set `--num-blocks` explicitly.

---

## Architecture, briefly

Two processes:

- **Control plane (Rust).** Owns scheduling, KV block allocation, prefix cache,
  admission, streaming and the HTTP API. Never touches the device.
- **Worker (Python).** Owns the weights, the KV storage and the forward pass.
  Makes no decisions.

One scheduler, one allocator. The split exists so there is exactly one place a
memory bug can live, and it is the side with the test suite.

They speak a length-prefixed protocol over TCP: JSON for control, raw
little-endian float32 for logits. The control plane sends one pass at a time —
continuous batching composes each step from the last step's result, so there is
nothing to pipeline.

---

## Things I would want reviewed

- The three-state block partition in `stride-kvcache` (live / cached / free) and
  whether `KvCache::release` can ever leak. It is asserted after every scheduler
  step in debug builds, but only against the invariant I thought to write.
- Preemption victim selection in `stride-sched`. It picks the sequence with the
  most deadline slack; under sustained pressure I have not proven this cannot
  livelock a single unlucky sequence.
- Whether `acquire_prefix` withholding the final block is the right call. It
  costs one recomputed block per warm request. vLLM matches to the token instead
  and does copy-on-write; that is strictly better and not implemented here.

---

## Not built

Named so nobody goes looking:

- Pipeline and expert parallelism. `ParallelConfig` validates both and the
  planner sizes memory for them, but only tensor parallelism is implemented.
  MoE models therefore run with every expert replicated on every rank.
- Multi-node. `torchrun --nproc_per_node` covers one machine's GPUs; nothing
  handles a second host.
- Disaggregated prefill/decode, speculative decoding, structured output,
  LoRA adapters.
- `n > 1` completions. The scheduler supports several sequences per request;
  the API rejects it rather than silently returning one.
- A real chat template. Messages are flattened as `role: content`. Fine for
  smoke tests, wrong for instruction-tuned models — wire up the checkpoint's
  Jinja template before judging output quality.
