"""Wire-format round trips.

The protocol is the seam between two languages, so a framing mistake shows up
as a hang or as silently misread logits rather than as an exception. These
tests pin the layout from the Python side; the Rust client parses the same
bytes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from stride_worker.protocol import (
    ForwardRequest,
    ProtocolError,
    decode_response,
    encode_error,
    encode_forward_response,
)


def test_forward_response_round_trips():
    logits = np.random.RandomState(0).randn(3, 128).astype(np.float32)
    frame = encode_forward_response([7, 8, 9], logits, duration_us=1234, vocab_size=128)
    header, got = decode_response(frame)

    assert header["type"] == "forward_ok"
    assert header["seqs"] == [7, 8, 9]
    assert header["duration_us"] == 1234
    assert header["estimated"] is False, "a measured duration must not be flagged estimated"
    np.testing.assert_array_equal(got, logits)


def test_empty_response_is_valid():
    frame = encode_forward_response([], np.empty((0, 32), np.float32), 0, 32)
    header, got = decode_response(frame)
    assert header["seqs"] == []
    assert got.shape == (0, 32)


def test_shape_mismatch_is_rejected_at_encode_time():
    with pytest.raises(ProtocolError):
        encode_forward_response([1, 2], np.zeros((3, 16), np.float32), 0, 16)


def test_truncated_logits_are_detected_not_reinterpreted():
    logits = np.zeros((2, 64), np.float32)
    frame = encode_forward_response([1, 2], logits, 0, 64)
    with pytest.raises(ProtocolError):
        decode_response(frame[:-8])


def test_error_frames_carry_their_kind():
    header, body = decode_response(encode_error("out of memory", "out_of_memory"))
    assert header["type"] == "error"
    assert header["kind"] == "out_of_memory"
    assert body.size == 0


def test_forward_request_parses_work_items():
    req = ForwardRequest.from_json(
        json.loads(
            '{"type":"forward","work":[{"seq":1,"tokens":[1,2,3],'
            '"position":0,"blocks":[4,5],"needs_logits":true}]}'
        )
    )
    assert req.num_tokens == 3
    assert req.work[0].blocks == [4, 5]
    assert req.work[0].needs_logits is True
