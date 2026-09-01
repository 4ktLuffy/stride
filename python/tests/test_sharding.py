"""The tensor-parallel sharding math, verified without any GPUs.

NCCL cannot be exercised on one CPU, but the part most likely to be wrong is
not the collective — it is *which* weights get split along *which* axis, and
where the sum has to happen. That is pure tensor algebra, and it can be checked
exactly.

Each test shards a real block by hand, performs the collective by summing or
concatenating the shards directly, and requires the result to match the
unsharded computation. If the Megatron split is wrong, these fail here rather
than on a rented eight-card node.
"""

from __future__ import annotations

import pytest
import torch

from stride_worker.distributed import shard_column, shard_row, validate_plan
from stride_worker.layers import swiglu_mlp


def linear(x, w):
    return torch.nn.functional.linear(x, w)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_column_parallel_concatenates_to_the_whole_output(tp):
    """y = Wx with W split along its output dimension: no communication."""
    torch.manual_seed(0)
    w = torch.randn(64, 32, dtype=torch.float64)
    x = torch.randn(5, 32, dtype=torch.float64)

    whole = linear(x, w)
    parts = [linear(x, shard_column(w, r, tp)) for r in range(tp)]
    assert torch.allclose(torch.cat(parts, dim=-1), whole, atol=1e-12)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_row_parallel_sums_to_the_whole_output(tp):
    """y = Wx with W split along its reduction axis: partials need an all-reduce."""
    torch.manual_seed(0)
    w = torch.randn(32, 64, dtype=torch.float64)
    x = torch.randn(5, 64, dtype=torch.float64)

    whole = linear(x, w)
    total = torch.zeros_like(whole)
    for r in range(tp):
        # Each rank holds the matching slice of the activation.
        x_shard = x.chunk(tp, dim=-1)[r]
        total += linear(x_shard, shard_row(w, r, tp))
    assert torch.allclose(total, whole, atol=1e-12)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_a_whole_attention_block_shards_correctly(tp):
    """q/k/v column-parallel, o row-parallel, one all-reduce at the end."""
    torch.manual_seed(0)
    hidden, n_heads, head_dim = 64, 8, 8
    q_w = torch.randn(n_heads * head_dim, hidden, dtype=torch.float64)
    k_w = torch.randn(n_heads * head_dim, hidden, dtype=torch.float64)
    v_w = torch.randn(n_heads * head_dim, hidden, dtype=torch.float64)
    o_w = torch.randn(hidden, n_heads * head_dim, dtype=torch.float64)
    x = torch.randn(6, hidden, dtype=torch.float64)

    def attend(q, k, v, heads):
        q = q.view(-1, heads, head_dim).transpose(0, 1)
        k = k.view(-1, heads, head_dim).transpose(0, 1)
        v = v.view(-1, heads, head_dim).transpose(0, 1)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(0, 1).reshape(-1, heads * head_dim)

    whole = linear(attend(linear(x, q_w), linear(x, k_w), linear(x, v_w), n_heads), o_w)

    # Sharded: each rank attends over its own heads, then the partial outputs
    # are summed. Heads are independent, which is exactly why this works.
    total = torch.zeros_like(whole)
    for r in range(tp):
        local = attend(
            linear(x, shard_column(q_w, r, tp)),
            linear(x, shard_column(k_w, r, tp)),
            linear(x, shard_column(v_w, r, tp)),
            n_heads // tp,
        )
        total += linear(local, shard_row(o_w, r, tp))

    assert torch.allclose(total, whole, atol=1e-10), (
        "sharded attention diverged from the unsharded result; the split axis "
        "or the reduction is wrong"
    )


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_the_swiglu_mlp_shards_correctly(tp):
    """gate and up column-parallel, down row-parallel, one all-reduce."""
    torch.manual_seed(0)
    hidden, inter = 32, 128
    gate = torch.randn(inter, hidden, dtype=torch.float64)
    up = torch.randn(inter, hidden, dtype=torch.float64)
    down = torch.randn(hidden, inter, dtype=torch.float64)
    x = torch.randn(4, hidden, dtype=torch.float64)

    whole = swiglu_mlp(x, gate, up, down)

    total = torch.zeros_like(whole)
    for r in range(tp):
        total += swiglu_mlp(
            x,
            shard_column(gate, r, tp),
            shard_column(up, r, tp),
            shard_row(down, r, tp),
        )

    assert torch.allclose(total, whole, atol=1e-10), (
        "the gating is elementwise over the intermediate dimension, so splitting "
        "gate and up the same way must preserve it"
    )


@pytest.mark.parametrize("tp", [2, 4])
def test_vocabulary_parallel_logits_reassemble_in_rank_order(tp):
    """The gather must restore the original vocabulary indexing."""
    torch.manual_seed(0)
    vocab, hidden = 256, 32
    lm_head = torch.randn(vocab, hidden, dtype=torch.float64)
    x = torch.randn(3, hidden, dtype=torch.float64)

    whole = linear(x, lm_head)
    parts = [linear(x, shard_column(lm_head, r, tp)) for r in range(tp)]
    gathered = torch.cat(parts, dim=-1)

    assert torch.allclose(gathered, whole, atol=1e-12)
    assert int(whole.argmax(-1)[0]) == int(gathered.argmax(-1)[0]), (
        "argmax must land on the same token id, or sampling picks a different word"
    )


def test_an_indivisible_split_is_refused_not_rounded():
    w = torch.randn(10, 8)
    with pytest.raises(ValueError, match="output dimension"):
        shard_column(w, 0, 4)
    with pytest.raises(ValueError, match="reduction dimension"):
        shard_row(w, 0, 3)


def test_plan_validation_rejects_geometry_it_cannot_shard():
    # Llama-3.1-70B: 64 query heads, 8 KV heads.
    validate_plan(64, 8, 8)
    validate_plan(64, 8, 1)

    with pytest.raises(ValueError, match="KV heads"):
        validate_plan(64, 8, 16)
    with pytest.raises(ValueError, match="query heads"):
        validate_plan(6, 6, 4)


def test_the_kv_head_limit_is_explained_not_just_refused():
    """A refusal has to say what degree *would* work, or it wastes an hour."""
    with pytest.raises(ValueError) as e:
        validate_plan(64, 8, 16)
    assert "up to 8 ranks" in str(e.value)
