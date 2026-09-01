"""TCP worker: owns the device, the weights and the KV blocks.

The Rust control plane makes every scheduling and allocation decision and
sends this process only what to execute. That split is deliberate — one
scheduler, one allocator, one place where a memory bug can live — and it is
why Python never sits on the latency-critical path for anything except the
forward pass itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import socketserver
import sys
import threading

import torch

from .cache import PagedKVCache
from .model import StrideModel
from .protocol import (
    ForwardRequest,
    ProtocolError,
    encode_error,
    encode_forward_response,
    read_frame,
    write_frame,
)

log = logging.getLogger("stride.worker")

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class WorkerState:
    """Model and cache, shared by every connection.

    A lock serialises forward passes. The device cannot usefully run two at
    once anyway, and the control plane sends one at a time; the lock is here so
    that a second client connecting cannot corrupt cache state.
    """

    def __init__(self, model: StrideModel, cache: PagedKVCache):
        self.model = model
        self.cache = cache
        self.lock = threading.Lock()
        self.passes = 0

    def info(self) -> dict:
        spec = self.cache.spec
        return {
            "type": "info",
            "vocab_size": self.model.vocab_size,
            "num_layers": self.model.num_layers,
            "hidden_size": self.model.hidden_size,
            "num_q_heads": self.model.num_q_heads,
            "num_kv_heads": self.model.num_kv_heads,
            "head_dim": self.model.head_dim,
            "num_blocks": spec.num_blocks,
            "block_size": spec.block_size,
            "kv_cache_bytes": spec.total_bytes,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype).replace("torch.", ""),
            "passes": self.passes,
        }


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        peer = self.client_address
        log.info("control plane connected from %s", peer)
        state: WorkerState = self.server.state  # type: ignore[attr-defined]

        try:
            while True:
                try:
                    frame = read_frame(self.request)
                except ProtocolError:
                    break
                if not frame:
                    break

                try:
                    message = json.loads(frame.decode("utf-8"))
                except json.JSONDecodeError as e:
                    write_frame(self.request, encode_error(f"malformed request: {e}", "bad_request"))
                    continue

                kind = message.get("type")
                try:
                    if kind == "info":
                        write_frame(
                            self.request, _json_frame(state.info())
                        )
                    elif kind == "reset":
                        with state.lock:
                            state.cache.zero_()
                        write_frame(self.request, _json_frame({"type": "reset_ok"}))
                    elif kind == "forward":
                        request = ForwardRequest.from_json(message)
                        with state.lock:
                            seqs, logits, duration_us = state.model.forward(
                                request.work, state.cache
                            )
                            state.passes += 1
                        write_frame(
                            self.request,
                            encode_forward_response(
                                seqs,
                                logits.detach().to("cpu", torch.float32).numpy(),
                                duration_us,
                                state.model.vocab_size,
                            ),
                        )
                    else:
                        write_frame(
                            self.request,
                            encode_error(f"unknown request type {kind!r}", "bad_request"),
                        )
                except torch.cuda.OutOfMemoryError as e:
                    # Report rather than die: the control plane can preempt and
                    # retry, and killing the worker would drop every other
                    # sequence's KV state along with it.
                    torch.cuda.empty_cache()
                    log.error("out of memory during forward: %s", e)
                    write_frame(self.request, encode_error(str(e), "out_of_memory"))
                except Exception as e:  # noqa: BLE001 - reported to the caller
                    log.exception("forward failed")
                    write_frame(self.request, encode_error(str(e), "worker_error"))
        finally:
            log.info("control plane %s disconnected", peer)


def _json_frame(payload: dict) -> bytes:
    from .protocol import HEADER

    body = json.dumps(payload).encode("utf-8")
    return HEADER.pack(len(body)) + body


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def autosize_blocks(
    model: StrideModel, block_size: int, reserve_fraction: float
) -> int:
    """Choose a block count from what the device actually has free.

    Measured after the weights are resident, so it accounts for the real
    footprint rather than a predicted one. ``reserve_fraction`` is held back for
    activations, workspace and fragmentation.
    """
    if model.device.type != "cuda":
        return 4096

    free_bytes, _total = torch.cuda.mem_get_info(model.device)
    usable = int(free_bytes * (1.0 - reserve_fraction))
    per_block = (
        2
        * block_size
        * model.num_kv_heads
        * model.head_dim
        * torch.empty((), dtype=model.dtype).element_size()
        * model.num_layers
    )
    blocks = usable // per_block
    if blocks < 1:
        raise RuntimeError(
            f"no room for even one KV block: {free_bytes / 2**30:.1f} GiB free, "
            f"{per_block / 2**20:.1f} MiB needed per block"
        )
    return int(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stride-worker",
        description="GPU execution worker for the Stride runtime.",
    )
    parser.add_argument("--model", required=True, help="path to a Hugging Face checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="override the block count; by default sized from free device memory",
    )
    parser.add_argument(
        "--activation-reserve",
        type=float,
        default=0.15,
        help="share of free memory held back for activations and workspace",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.error("CUDA is not available. Pass --device cpu to run without a GPU.")
        return 2

    log.info("loading %s onto %s as %s", args.model, args.device, args.dtype)
    model = StrideModel.from_pretrained(args.model, args.device, DTYPES[args.dtype])

    num_blocks = args.num_blocks or autosize_blocks(
        model, args.block_size, args.activation_reserve
    )
    spec = model.cache_spec(num_blocks, args.block_size)
    log.info(
        "KV cache: %d blocks x %d tokens = %.1f GiB",
        num_blocks,
        args.block_size,
        spec.total_bytes / 2**30,
    )
    cache = PagedKVCache(spec)

    server = Server((args.host, args.port), Handler)
    server.state = WorkerState(model, cache)  # type: ignore[attr-defined]
    log.info("listening on %s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
