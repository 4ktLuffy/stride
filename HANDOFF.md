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
| No GPU | Rust suite, simulator end to end, the gate self-check. All already green in CI. |
| **1x 24-48 GB** (L40S, A6000, 4090) | **The most valuable tier.** 8B model end to end for real, all three Triton kernels, the full autotuner. Everything most likely to be broken is broken here. |
| 1x 80 GB (A100/H100) | 8B at long context, or 70B quantised to INT4 (34.9 GiB weights, ~133k tokens of KV). |
| 2x 80 GB | 70B in FP8 with tensor parallelism — except TP needs NCCL, which **is not written**. Use this tier for one-GPU long-context work instead. |
| 8x 80 GB | What the planner is designed for, and what the runtime cannot yet use. Single-GPU only until NCCL lands. |

**The honest ceiling: the worker is single-GPU.** `ParallelConfig` validates
TP/PP/EP plans and the planner sizes memory for them, but no collective
communication code exists. An eight-card node runs this on one card and idles
the other seven.

So the largest model that can actually be *served* today is whatever fits on one
GPU: 8B in BF16 on 24 GB, or 70B in INT4 on 80 GB. Anything bigger is a planning
exercise until NCCL is implemented, and that is the single highest-value thing
to build next.

### Recommendation

One L40S or A6000 for a few hours. That exercises every unverified line of code
in this repository, and it costs less than a lunch. Booking an 8xH100 node
before the single-GPU path works would spend real money to idle seven cards
while finding the same bugs.

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

### 5. Kernel autotuning, one GPU

```bash
stride-autotune rmsnorm --device cuda --dtype bfloat16 --out rmsnorm.json --verbose
```

Explores 75 configurations, gates each against the PyTorch reference before
timing it, and reports a Pareto front over latency and peak memory. The report
records the device, driver, library versions, seed and shapes, so a number in it
can be reproduced or disputed later.

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

- Distributed execution. `ParallelConfig` validates TP/PP/EP plans and the
  planner sizes memory for them, but no NCCL code exists — the worker is
  single-GPU.
- Disaggregated prefill/decode, speculative decoding, structured output,
  LoRA adapters, multi-node anything.
- `n > 1` completions. The scheduler supports several sequences per request;
  the API rejects it rather than silently returning one.
- A real chat template. Messages are flattened as `role: content`. Fine for
  smoke tests, wrong for instruction-tuned models — wire up the checkpoint's
  Jinja template before judging output quality.
