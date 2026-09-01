"""Wire format between the Rust control plane and this worker.

One TCP connection, request/response, no pipelining. The control plane runs a
single scheduling loop, so a second in-flight pass would have nothing to
schedule against and would only complicate cancellation.

Each message is a 4-byte big-endian length followed by that many bytes. A
request is UTF-8 JSON. A response is UTF-8 JSON, and when it carries logits
those follow immediately as raw little-endian float32 — JSON-encoding a
128k-wide distribution per sequence would cost more than the forward pass.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

HEADER = struct.Struct("!I")
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes, or raise. A short read is a truncated message."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ProtocolError(f"connection closed with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> bytes:
    (length,) = HEADER.unpack(_recv_exactly(sock, HEADER.size))
    if length > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"frame of {length} bytes exceeds the {MAX_MESSAGE_BYTES} cap")
    return _recv_exactly(sock, length) if length else b""


def write_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(HEADER.pack(len(payload)) + payload)


@dataclass
class SequenceWork:
    """One sequence's contribution to a forward pass."""

    seq: int
    tokens: list[int]
    #: Index of the first of `tokens` within the sequence.
    position: int
    #: Physical KV blocks backing this sequence, in logical order.
    blocks: list[int]
    #: Whether this pass reaches the sequence's final token and must produce
    #: logits. False for every prefill chunk but the last.
    needs_logits: bool

    @staticmethod
    def from_json(d: dict[str, Any]) -> "SequenceWork":
        return SequenceWork(
            seq=int(d["seq"]),
            tokens=[int(t) for t in d["tokens"]],
            position=int(d["position"]),
            blocks=[int(b) for b in d["blocks"]],
            needs_logits=bool(d["needs_logits"]),
        )


@dataclass
class ForwardRequest:
    work: list[SequenceWork] = field(default_factory=list)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "ForwardRequest":
        return ForwardRequest(work=[SequenceWork.from_json(w) for w in d.get("work", [])])

    @property
    def num_tokens(self) -> int:
        return sum(len(w.tokens) for w in self.work)


def encode_forward_response(
    seq_ids: list[int], logits: np.ndarray, duration_us: int, vocab_size: int
) -> bytes:
    """Pack logits for the sequences that needed them.

    `logits` must be (len(seq_ids), vocab_size) float32. The array is sent raw
    and the header states its shape, so the reader never has to infer it.
    """
    if len(seq_ids) == 0:
        body = np.empty(0, dtype=np.float32)
    else:
        if logits.shape != (len(seq_ids), vocab_size):
            raise ProtocolError(
                f"logits shape {logits.shape} does not match "
                f"({len(seq_ids)}, {vocab_size})"
            )
        body = np.ascontiguousarray(logits, dtype="<f4")

    header = json.dumps(
        {
            "type": "forward_ok",
            "seqs": seq_ids,
            "vocab_size": vocab_size,
            # Measured on the device, not modelled. The control plane reports
            # it as a real duration rather than an estimate.
            "duration_us": int(duration_us),
            "estimated": False,
        }
    ).encode("utf-8")

    return HEADER.pack(len(header)) + header + body.tobytes()


def encode_error(message: str, kind: str = "worker_error") -> bytes:
    header = json.dumps({"type": "error", "kind": kind, "message": message}).encode("utf-8")
    return HEADER.pack(len(header)) + header


def decode_response(payload: bytes) -> tuple[dict[str, Any], np.ndarray]:
    """Split a response frame into its JSON header and logits array."""
    if len(payload) < HEADER.size:
        raise ProtocolError("response is shorter than its own header")
    (header_len,) = HEADER.unpack(payload[: HEADER.size])
    start = HEADER.size
    header = json.loads(payload[start : start + header_len].decode("utf-8"))
    body = payload[start + header_len :]

    if header.get("type") != "forward_ok":
        return header, np.empty(0, dtype=np.float32)

    n, vocab = len(header["seqs"]), int(header["vocab_size"])
    if n == 0:
        return header, np.empty((0, vocab), dtype=np.float32)

    expected = n * vocab * 4
    if len(body) != expected:
        raise ProtocolError(f"expected {expected} bytes of logits, got {len(body)}")
    return header, np.frombuffer(body, dtype="<f4").reshape(n, vocab)
