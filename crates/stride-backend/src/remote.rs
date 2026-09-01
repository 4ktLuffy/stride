//! Executor backed by the Python GPU worker.
//!
//! The wire format is deliberately plain: a 4-byte big-endian length, then a
//! UTF-8 JSON message. A forward response carries a JSON header followed by
//! raw little-endian float32 logits, because JSON-encoding a 128k-wide
//! distribution per sequence would cost more than the forward pass it
//! describes.
//!
//! Calls are synchronous and blocking. The engine issues exactly one forward
//! pass at a time — continuous batching composes each step from the last
//! step's outcome, so there is nothing to pipeline — and the pass is the
//! dominant cost, so the engine should own a runtime thread rather than share
//! one.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use serde::Serialize;
use stride_model::ModelConfig;

use crate::executor::{Executor, ForwardPass, PassCost, PassResult, SequenceLogits};

const MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum RemoteError {
    #[error("worker transport failed: {0}")]
    Io(#[from] std::io::Error),

    #[error("worker sent a malformed message: {0}")]
    Protocol(String),

    #[error("worker reported {kind}: {message}")]
    Worker { kind: String, message: String },
}

/// Geometry the worker reports at connection time.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerInfo {
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    /// Ranks the worker is sharded across. The control plane addresses one
    /// worker regardless; this is reported so a mismatch with the planned
    /// degree is caught rather than silently ignored.
    pub tensor_parallel_size: usize,
    pub device: String,
    pub dtype: String,
}

#[derive(Serialize)]
struct WireWork<'a> {
    seq: u64,
    tokens: &'a [u32],
    position: usize,
    blocks: Vec<u32>,
    needs_logits: bool,
}

#[derive(Serialize)]
struct WireForward<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    work: Vec<WireWork<'a>>,
}

pub struct RemoteExecutor {
    stream: TcpStream,
    model: ModelConfig,
    info: WorkerInfo,
}

impl RemoteExecutor {
    /// Connect and read the worker's geometry.
    ///
    /// The reported geometry is checked against the model the control plane
    /// thinks it is serving. A mismatch means the two sides disagree about the
    /// checkpoint, which would corrupt every KV write, so it is a hard error at
    /// startup rather than a warning.
    pub fn connect(
        addr: impl ToSocketAddrs,
        model: ModelConfig,
        timeout: Duration,
    ) -> Result<Self, RemoteError> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| RemoteError::Protocol("no address resolved".into()))?;
        let stream = TcpStream::connect_timeout(&addr, timeout)?;
        stream.set_nodelay(true)?;

        let mut me = Self {
            stream,
            model,
            info: WorkerInfo {
                vocab_size: 0,
                num_layers: 0,
                num_kv_heads: 0,
                head_dim: 0,
                num_blocks: 0,
                block_size: 0,
                tensor_parallel_size: 1,
                device: String::new(),
                dtype: String::new(),
            },
        };
        me.info = me.query_info()?;
        me.check_geometry()?;
        Ok(me)
    }

    pub fn info(&self) -> &WorkerInfo {
        &self.info
    }

    fn check_geometry(&self) -> Result<(), RemoteError> {
        let a = &self.model.attention;
        let mismatch = |what: &str, ours: usize, theirs: usize| {
            RemoteError::Protocol(format!(
                "worker and control plane disagree on {what}: {ours} here, {theirs} there. \
                 They are not serving the same checkpoint."
            ))
        };
        if self.info.num_layers != self.model.num_layers {
            return Err(mismatch(
                "layer count",
                self.model.num_layers,
                self.info.num_layers,
            ));
        }
        if self.info.num_kv_heads != a.num_kv_heads {
            return Err(mismatch(
                "KV head count",
                a.num_kv_heads,
                self.info.num_kv_heads,
            ));
        }
        if self.info.head_dim != a.head_dim {
            return Err(mismatch("head dimension", a.head_dim, self.info.head_dim));
        }
        Ok(())
    }

    fn write_frame(&mut self, payload: &[u8]) -> Result<(), RemoteError> {
        self.stream
            .write_all(&(payload.len() as u32).to_be_bytes())?;
        self.stream.write_all(payload)?;
        self.stream.flush()?;
        Ok(())
    }

    fn read_frame(&mut self) -> Result<Vec<u8>, RemoteError> {
        let mut len = [0u8; 4];
        self.stream.read_exact(&mut len)?;
        let len = u32::from_be_bytes(len) as usize;
        if len > MAX_FRAME_BYTES {
            return Err(RemoteError::Protocol(format!(
                "frame of {len} bytes exceeds the {MAX_FRAME_BYTES} cap"
            )));
        }
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Split a response frame into its JSON header and the raw logits after it.
    fn split_response(frame: &[u8]) -> Result<(serde_json::Value, &[u8]), RemoteError> {
        if frame.len() < 4 {
            return Err(RemoteError::Protocol(
                "response shorter than its header".into(),
            ));
        }
        let header_len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let start: usize = 4;
        let end = start
            .checked_add(header_len)
            .filter(|&e| e <= frame.len())
            .ok_or_else(|| RemoteError::Protocol("header length runs past the frame".into()))?;

        let header: serde_json::Value = serde_json::from_slice(&frame[start..end])
            .map_err(|e| RemoteError::Protocol(e.to_string()))?;

        if header.get("type").and_then(|t| t.as_str()) == Some("error") {
            return Err(RemoteError::Worker {
                kind: header
                    .get("kind")
                    .and_then(|k| k.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
                message: header
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("(no message)")
                    .to_string(),
            });
        }
        Ok((header, &frame[end..]))
    }

    fn query_info(&mut self) -> Result<WorkerInfo, RemoteError> {
        self.write_frame(br#"{"type":"info"}"#)?;
        let frame = self.read_frame()?;
        let (header, _) = Self::split_response(&frame)?;
        let field = |name: &str| -> Result<usize, RemoteError> {
            header
                .get(name)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| RemoteError::Protocol(format!("info is missing `{name}`")))
        };
        Ok(WorkerInfo {
            vocab_size: field("vocab_size")?,
            num_layers: field("num_layers")?,
            num_kv_heads: field("num_kv_heads")?,
            head_dim: field("head_dim")?,
            num_blocks: field("num_blocks")?,
            block_size: field("block_size")?,
            // Older workers predate sharding and simply omit this.
            tensor_parallel_size: header
                .get("tensor_parallel_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize,
            device: header
                .get("device")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            dtype: header
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
        })
    }

    fn forward_inner(&mut self, pass: &ForwardPass) -> Result<PassResult, RemoteError> {
        let message = WireForward {
            kind: "forward",
            work: pass
                .work
                .iter()
                .map(|w| WireWork {
                    seq: w.seq.raw(),
                    tokens: w.tokens,
                    position: w.position,
                    blocks: w.blocks.iter().map(|b| b.0).collect(),
                    needs_logits: w.needs_logits,
                })
                .collect(),
        };
        let payload =
            serde_json::to_vec(&message).map_err(|e| RemoteError::Protocol(e.to_string()))?;
        self.write_frame(&payload)?;

        let frame = self.read_frame()?;
        let (header, body) = Self::split_response(&frame)?;

        let seqs: Vec<u64> = header
            .get("seqs")
            .and_then(|v| v.as_array())
            .ok_or_else(|| RemoteError::Protocol("response is missing `seqs`".into()))?
            .iter()
            .filter_map(|v| v.as_u64())
            .collect();
        let vocab = header
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| RemoteError::Protocol("response is missing `vocab_size`".into()))?
            as usize;

        let expected = seqs.len() * vocab * 4;
        if body.len() != expected {
            return Err(RemoteError::Protocol(format!(
                "expected {expected} bytes of logits for {} sequences, got {}",
                seqs.len(),
                body.len()
            )));
        }

        let logits = seqs
            .iter()
            .enumerate()
            .map(|(i, &seq)| {
                let start = i * vocab * 4;
                let values = body[start..start + vocab * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                SequenceLogits {
                    seq: stride_core::SequenceId(seq),
                    logits: values,
                }
            })
            .collect();

        Ok(PassResult {
            logits,
            cost: PassCost {
                duration_us: header
                    .get("duration_us")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                // The worker times the pass on the device. This is measured.
                estimated: false,
            },
        })
    }
}

impl Executor for RemoteExecutor {
    fn model(&self) -> &ModelConfig {
        &self.model
    }

    fn vocab_size(&self) -> usize {
        self.info.vocab_size
    }

    fn forward(&mut self, pass: &ForwardPass) -> PassResult {
        match self.forward_inner(pass) {
            Ok(result) => result,
            Err(e) => {
                // Returning no logits stalls the affected sequences rather than
                // killing the loop. The engine keeps serving everything else,
                // and the error is visible in the logs and in /metrics.
                tracing::error!(error = %e, "forward pass failed on the worker");
                PassResult {
                    logits: Vec::new(),
                    cost: PassCost {
                        duration_us: 0,
                        estimated: false,
                    },
                }
            }
        }
    }
}
