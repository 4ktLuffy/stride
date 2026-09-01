"""Llama-family model execution against the paged KV cache.

The forward pass is written by hand rather than delegated to
``transformers``, because a served batch is *ragged*: several sequences at
different positions, some prefilling a chunk of prompt and some decoding a
single token, each with its own scattered block table. A padded batch API
cannot express that without wasting most of the pass on padding.

Tokens from every sequence are concatenated into one flat batch for the
projections and the MLP — those are position-independent — and attention is
then run per sequence, which is where the block tables matter. This is the
same shape as a varlen kernel, expressed in PyTorch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .cache import CacheSpec, PagedKVCache
from .layers import apply_rope, build_rope_cache, paged_attention, rms_norm, swiglu_mlp
from .protocol import SequenceWork

SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
}


@dataclass
class LayerWeights:
    input_norm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor
    post_norm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    #: Some checkpoints carry attention biases; most Llama-family ones do not.
    q_bias: torch.Tensor | None = None
    k_bias: torch.Tensor | None = None
    v_bias: torch.Tensor | None = None


def _load_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    """Load safetensors weights, sharded or not."""
    from safetensors.torch import load_file

    index = path / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(mapping.values()))
    else:
        shards = [p.name for p in sorted(path.glob("*.safetensors"))]
    if not shards:
        raise FileNotFoundError(f"no .safetensors weights under {path}")

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(path / shard), device=device))
    return state


class StrideModel:
    """A Llama-family decoder running against paged KV storage."""

    def __init__(
        self,
        config: dict,
        state: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype

        self.num_layers = int(config["num_hidden_layers"])
        self.hidden_size = int(config["hidden_size"])
        self.num_q_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config.get("num_key_value_heads", self.num_q_heads))
        self.head_dim = int(config.get("head_dim", self.hidden_size // self.num_q_heads))
        self.vocab_size = int(config["vocab_size"])
        self.eps = float(config.get("rms_norm_eps", 1e-5))
        self.scale = self.head_dim**-0.5

        self.embed = state["model.embed_tokens.weight"].to(device=device, dtype=dtype)
        self.final_norm = state["model.norm.weight"].to(device=device, dtype=dtype)
        # Tied embeddings are common on smaller checkpoints and the lm_head
        # tensor is simply absent; reusing the embedding matrix is correct.
        lm_head = state.get("lm_head.weight")
        self.lm_head = (
            self.embed if lm_head is None else lm_head.to(device=device, dtype=dtype)
        )

        self.layers: list[LayerWeights] = []
        for i in range(self.num_layers):
            p = f"model.layers.{i}"

            def get(name: str, required: bool = True):
                key = f"{p}.{name}"
                t = state.get(key)
                if t is None:
                    if required:
                        raise KeyError(f"checkpoint is missing {key}")
                    return None
                return t.to(device=device, dtype=dtype)

            self.layers.append(
                LayerWeights(
                    input_norm=get("input_layernorm.weight"),
                    q_proj=get("self_attn.q_proj.weight"),
                    k_proj=get("self_attn.k_proj.weight"),
                    v_proj=get("self_attn.v_proj.weight"),
                    o_proj=get("self_attn.o_proj.weight"),
                    post_norm=get("post_attention_layernorm.weight"),
                    gate_proj=get("mlp.gate_proj.weight"),
                    up_proj=get("mlp.up_proj.weight"),
                    down_proj=get("mlp.down_proj.weight"),
                    q_bias=get("self_attn.q_proj.bias", required=False),
                    k_bias=get("self_attn.k_proj.bias", required=False),
                    v_bias=get("self_attn.v_proj.bias", required=False),
                )
            )

        max_pos = int(config.get("max_position_embeddings", 8192))
        self.cos, self.sin = build_rope_cache(
            self.head_dim,
            max_pos,
            float(config.get("rope_theta", 10000.0)),
            device,
            config.get("rope_scaling"),
        )

    @classmethod
    def from_pretrained(
        cls, path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16
    ) -> "StrideModel":
        root = Path(path)
        config = json.loads((root / "config.json").read_text())

        architectures = config.get("architectures") or []
        if architectures and not any(a in SUPPORTED_ARCHITECTURES for a in architectures):
            raise ValueError(
                f"{architectures} is not supported. This worker implements "
                f"Llama-family geometry only: {sorted(SUPPORTED_ARCHITECTURES)}. "
                "Running it anyway would produce wrong activations silently."
            )

        state = _load_state_dict(root, device="cpu")
        return cls(config, state, torch.device(device), dtype)

    def cache_spec(self, num_blocks: int, block_size: int) -> CacheSpec:
        return CacheSpec(
            num_layers=self.num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def forward(
        self, work: list[SequenceWork], cache: PagedKVCache
    ) -> tuple[list[int], torch.Tensor, int]:
        """Run one pass.

        Returns the sequence ids that produced logits, a
        ``(len(ids), vocab_size)`` tensor, and the measured duration in
        microseconds.
        """
        if not work:
            return [], torch.empty((0, self.vocab_size), device=self.device), 0

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        flat_tokens: list[int] = []
        positions: list[int] = []
        spans: list[tuple[int, int]] = []  # (start, length) into the flat batch
        for w in work:
            spans.append((len(flat_tokens), len(w.tokens)))
            flat_tokens.extend(w.tokens)
            positions.extend(range(w.position, w.position + len(w.tokens)))

        token_ids = torch.tensor(flat_tokens, device=self.device, dtype=torch.long)
        position_ids = torch.tensor(positions, device=self.device, dtype=torch.long)
        x = self.embed[token_ids]

        for layer_index, layer in enumerate(self.layers):
            residual = x
            h = rms_norm(x, layer.input_norm, self.eps)

            q = torch.nn.functional.linear(h, layer.q_proj, layer.q_bias)
            k = torch.nn.functional.linear(h, layer.k_proj, layer.k_bias)
            v = torch.nn.functional.linear(h, layer.v_proj, layer.v_bias)

            q = q.view(-1, self.num_q_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            q, k = apply_rope(q, k, self.cos, self.sin, position_ids)

            # Publish this pass's KV before attending, so a decode step can see
            # the token it just produced.
            for w, (start, length) in zip(work, spans):
                if length:
                    cache.write(
                        layer_index,
                        w.blocks,
                        w.position,
                        k[start : start + length],
                        v[start : start + length],
                    )

            attn = torch.empty_like(q)
            for w, (start, length) in zip(work, spans):
                if not length:
                    continue
                attn[start : start + length] = paged_attention(
                    q[start : start + length],
                    cache,
                    layer_index,
                    w.blocks,
                    context_len=w.position + length,
                    query_start=w.position,
                    scale=self.scale,
                )

            x = residual + torch.nn.functional.linear(
                attn.reshape(-1, self.num_q_heads * self.head_dim), layer.o_proj
            )
            x = x + swiglu_mlp(
                rms_norm(x, layer.post_norm, self.eps),
                layer.gate_proj,
                layer.up_proj,
                layer.down_proj,
            )

        x = rms_norm(x, self.final_norm, self.eps)

        # Only the final token of each sequence that asked for logits is run
        # through the vocabulary projection. On a 128k vocabulary that is the
        # difference between projecting a handful of rows and thousands.
        wanted: list[int] = []
        rows: list[int] = []
        for w, (start, length) in zip(work, spans):
            if w.needs_logits and length:
                wanted.append(w.seq)
                rows.append(start + length - 1)

        if not wanted:
            logits = torch.empty((0, self.vocab_size), device=self.device)
        else:
            logits = torch.nn.functional.linear(x[rows], self.lm_head).float()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        duration_us = int((time.perf_counter() - started) * 1e6)
        return wanted, logits, duration_us
