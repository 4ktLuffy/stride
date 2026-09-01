"""The equivalence metrics themselves.

These decide whether a reported difference is a bug or floating-point noise, so
they need to be right in both directions: noise must pass, and a real defect
must not. Both directions are tested, because a comparison that always says
"equivalent" is as useless as a gate that always accepts.
"""

from __future__ import annotations

import torch

from stride_worker.diagnose.equivalence import (
    compare_logits,
    first_divergence,
)


def logits(rows=64, vocab=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, vocab, generator=g) * 4.0


def test_identical_runs_are_equivalent():
    a = logits()
    result = compare_logits(a, a.clone())
    ok, why = result.verdict(torch.float32)
    assert ok, why
    assert result.argmax_match_rate == 1.0
    assert result.cosine > 0.9999999


def test_reduction_order_noise_is_not_reported_as_a_bug():
    """The load-bearing case: TP changes the order of a float reduction.

    Differences at this scale are what correct tensor parallelism produces. If
    the comparison flagged them, every correct run would look broken.
    """
    a = logits()
    g = torch.Generator().manual_seed(1)
    b = a + torch.randn(a.shape, generator=g) * 1e-5

    result = compare_logits(a, b)
    ok, why = result.verdict(torch.bfloat16)
    assert ok, f"noise at 1e-5 must not be called a defect: {why}"


def test_a_structurally_different_computation_is_caught():
    a = logits()
    g = torch.Generator().manual_seed(2)
    b = a + torch.randn(a.shape, generator=g) * 2.0

    result = compare_logits(a, b)
    ok, why = result.verdict(torch.bfloat16)
    assert not ok
    assert "cosine" in why or "argmax" in why


def test_disagreement_at_a_near_tie_is_tolerated():
    """A coin-flip between two equally-ranked tokens is not a defect."""
    a = torch.zeros(32, 16)
    a[:, 0] = 5.0
    a[:, 1] = 5.0  # exact tie
    b = a.clone()
    b[:, 1] += 1e-6  # the tie breaks the other way

    result = compare_logits(a, b)
    assert result.argmax_match_rate < 1.0, "the argmax should differ"
    assert result.worst_disagreement_gap < 1e-3, "and the gap should be tiny"
    ok, why = result.verdict(torch.bfloat16)
    assert ok, why


def test_disagreement_at_a_decisive_margin_is_a_defect():
    """Ranking a different token first when the reference was confident."""
    a = torch.zeros(32, 16)
    a[:, 0] = 10.0  # unambiguous winner
    b = a.clone()
    b[:, 0] = 0.0
    b[:, 3] = 10.0  # a completely different token wins

    result = compare_logits(a, b)
    ok, why = result.verdict(torch.bfloat16)
    assert not ok, why


def test_first_divergence_names_the_earliest_bad_layer():
    """Error compounds, so only the first divergent layer is informative."""
    g = torch.Generator().manual_seed(3)
    ref = [torch.randn(8, 32, generator=g) for _ in range(12)]
    cand = [t.clone() for t in ref]

    # Layer 5 goes wrong, and everything after it inherits the damage.
    for i in range(5, 12):
        cand[i] = cand[i] + torch.randn(8, 32, generator=g) * 3.0

    found = first_divergence(ref, cand)
    assert found is not None
    assert found.layer == 5, f"reported layer {found.layer}, not the first bad one"


def test_no_divergence_when_every_layer_agrees():
    g = torch.Generator().manual_seed(4)
    ref = [torch.randn(8, 32, generator=g) for _ in range(6)]
    cand = [t + torch.randn(t.shape, generator=g) * 1e-7 for t in ref]
    assert first_divergence(ref, cand) is None


def test_a_shape_change_is_a_divergence_not_a_crash():
    ref = [torch.randn(8, 32) for _ in range(3)]
    cand = list(ref)
    cand[1] = torch.randn(8, 16)
    found = first_divergence(ref, cand)
    assert found is not None and found.layer == 1
