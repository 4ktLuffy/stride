"""Diagnostics for the GPU path.

    stride-diagnose prefill    --model CKPT --tokens 5000 --chunk 2048
    stride-diagnose attention  --contexts 16,64,256,1024,4096
    stride-diagnose tp-dump    --model CKPT --out tp1.pt
    stride-diagnose tp-compare tp1.pt tp2.pt

Each compares logits or hidden states on identical input. None of them compares
generated text, which cannot distinguish a wrong kernel from a different
floating-point reduction order.
"""

from __future__ import annotations

import argparse
import sys

import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _load(args):
    from ..model import StrideModel
    from .. import distributed

    ctx = distributed.init(args.device)
    model = StrideModel.from_pretrained(args.model, args.device, DTYPES[args.dtype], ctx)
    from ..cache import PagedKVCache

    blocks = args.blocks or ((args.tokens + 15) // 16 + 8)
    cache = PagedKVCache(model.cache_spec(blocks, 16))
    return model, cache, ctx


def _tokens(args, model) -> list[int]:
    """Deterministic pseudo-token prompt, so two runs compare like for like."""
    generator = torch.Generator().manual_seed(args.seed)
    return torch.randint(
        1, min(model.vocab_size, 30000), (args.tokens,), generator=generator
    ).tolist()


def cmd_prefill(args) -> int:
    from .equivalence import report
    from .prefill import sweep

    model, cache, _ = _load(args)
    tokens = _tokens(args, model)
    sizes = [int(s) for s in args.chunk.split(",")]

    print(f"prompt: {len(tokens)} tokens, chunk sizes {sizes}\n")
    results = sweep(model, cache, tokens, sizes)

    worst_ok = True
    for size, agreement in results.items():
        ok, _ = agreement.verdict(model.dtype)
        worst_ok &= ok
        print(report(f"chunk={size}", agreement, model.dtype))
        print()

    if worst_ok:
        print(
            "Chunked prefill matches the continuous run. If generated text still\n"
            "differs between them, that is decoding amplifying reduction-order\n"
            "noise at a near-tie, not a masking bug."
        )
    else:
        print(
            "Chunked prefill does not match. The mask or the position arithmetic\n"
            "at a chunk boundary is wrong; see paged_attention's query_start."
        )
    return 0 if worst_ok else 1


def cmd_attention(args) -> int:
    from .attention import assert_refuses_prefill, format_sweep, sweep_decode

    ok, message = assert_refuses_prefill(args.device)
    print(f"prefill-shaped input: {'REFUSED' if ok else 'ACCEPTED (BUG)'}")
    print(f"  {message}\n")

    contexts = [int(c) for c in args.contexts.split(",")]
    points = sweep_decode(contexts, device=args.device, dtype=DTYPES[args.dtype])
    print(format_sweep(points))
    return 0 if ok and all(p.ok for p in points) else 1


def cmd_tp_dump(args) -> int:
    from .tp import dump

    model, cache, ctx = _load(args)
    tokens = _tokens(args, model)
    dump(model, cache, tokens, args.out)
    if ctx.is_leader:
        print(f"wrote {args.out} (tp={ctx.world_size}, {len(tokens)} tokens)")
    return 0


def cmd_tp_compare(args) -> int:
    from .tp import compare

    ok, text = compare(args.reference, args.candidate)
    print(text)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    common.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))

    model_args = argparse.ArgumentParser(add_help=False, parents=[common])
    model_args.add_argument("--model", required=True, help="checkpoint directory")
    model_args.add_argument("--tokens", type=int, default=5000)
    model_args.add_argument("--blocks", type=int, default=None)
    model_args.add_argument("--seed", type=int, default=0)

    parser = argparse.ArgumentParser(prog="stride-diagnose", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prefill", parents=[model_args], help="chunked vs continuous prefill")
    p.add_argument("--chunk", default="512,1024,2048")
    p.set_defaults(func=cmd_prefill)

    a = sub.add_parser("attention", parents=[common], help="paged attention kernel vs reference")
    a.add_argument("--contexts", default="16,32,64,256,1024,4096")
    a.set_defaults(func=cmd_attention)

    d = sub.add_parser("tp-dump", parents=[model_args], help="capture per-layer states")
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_tp_dump)

    c = sub.add_parser("tp-compare", parents=[common], help="compare two tp dumps")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.set_defaults(func=cmd_tp_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
