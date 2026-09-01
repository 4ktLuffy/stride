//! HTTP routes.
//!
//! The handlers do three things and no more: translate a request into engine
//! terms, relay the token stream, and map errors onto status codes. All
//! scheduling and memory decisions belong to the engine, so nothing here holds
//! a lock or touches runtime state.

use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use stride_core::{Error, FinishReason};
use stride_engine::{EngineHandle, GenerationRequest, StreamEvent, TokenStream, Usage};

use crate::openai::*;

#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub model_name: String,
    /// True when the engine is backed by the analytic simulator. Surfaced on
    /// /health and /v1/models so a client is never misled about whether it is
    /// talking to a real model.
    pub simulated: bool,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn request_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

/// Maps engine errors onto HTTP semantics.
struct ApiError(Error);

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind, code) = match &self.0 {
            // Backpressure is the one error a well-behaved client should retry,
            // so it must be a 429 and not a generic 503.
            Error::Backpressure { .. } => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                Some("queue_full"),
            ),
            Error::InvalidSampling(_) | Error::PromptTooLong { .. } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("invalid_parameters"),
            ),
            Error::OutOfBlocks { .. } => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                Some("kv_cache_exhausted"),
            ),
            Error::EngineStopped => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                Some("engine_stopped"),
            ),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", None),
        };
        let body = ApiErrorBody {
            error: ApiErrorDetail {
                message: self.0.to_string(),
                r#type: kind,
                code,
            },
        };
        (status, Json(body)).into_response()
    }
}

async fn health(State(s): State<AppState>) -> Response {
    match s.engine.metrics().await {
        Ok(m) => Json(serde_json::json!({
            "status": "ok",
            "model": s.model_name,
            "backend": if s.simulated { "simulated" } else { "device" },
            "queued": m.queued,
            "running": m.running,
            "kv_blocks_total": m.kv_blocks_total,
            "kv_blocks_live": m.kv_blocks_live,
        }))
        .into_response(),
        Err(e) => ApiError(e).into_response(),
    }
}

/// Prometheus text exposition.
async fn metrics(State(s): State<AppState>) -> Response {
    let m = match s.engine.metrics().await {
        Ok(m) => m,
        Err(e) => return ApiError(e).into_response(),
    };
    let sched = m.scheduler;
    let mut out = String::new();

    let mut gauge = |name: &str, help: &str, value: f64| {
        out.push_str(&format!(
            "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n"
        ));
    };
    gauge(
        "stride_requests_queued",
        "Requests waiting for admission.",
        m.queued as f64,
    );
    gauge(
        "stride_sequences_running",
        "Sequences holding KV blocks.",
        m.running as f64,
    );
    gauge(
        "stride_kv_blocks_total",
        "KV blocks in the pool.",
        m.kv_blocks_total as f64,
    );
    gauge(
        "stride_kv_blocks_live",
        "KV blocks held by live sequences.",
        m.kv_blocks_live as f64,
    );
    gauge(
        "stride_kv_blocks_cached",
        "Unreferenced but reusable KV blocks.",
        m.kv_blocks_cached as f64,
    );
    gauge(
        "stride_kv_block_hit_rate",
        "Share of block lookups served from cache.",
        m.kv_block_hit_rate,
    );
    gauge(
        "stride_prefill_reuse_rate",
        "Share of context tokens skipped via prefix reuse.",
        sched.prefill_reuse_rate(),
    );
    gauge(
        "stride_backend_estimated",
        "1 when latency figures come from the analytic model rather than hardware.",
        if m.estimated { 1.0 } else { 0.0 },
    );

    let mut counter = |name: &str, help: &str, value: f64| {
        out.push_str(&format!(
            "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"
        ));
    };
    counter(
        "stride_requests_admitted_total",
        "Sequences admitted.",
        sched.admitted as f64,
    );
    counter(
        "stride_requests_finished_total",
        "Sequences finished.",
        sched.finished as f64,
    );
    counter(
        "stride_preemptions_total",
        "Sequences evicted under memory pressure.",
        sched.preemptions as f64,
    );
    counter(
        "stride_ttft_deadline_misses_total",
        "First tokens later than the class budget.",
        sched.ttft_deadline_misses as f64,
    );
    counter(
        "stride_scheduler_steps_total",
        "Scheduler steps.",
        sched.steps as f64,
    );
    counter(
        "stride_forward_passes_total",
        "Forward passes executed.",
        m.forward_passes as f64,
    );
    counter(
        "stride_prefill_tokens_computed_total",
        "Context tokens run through the model.",
        sched.prefill_tokens_computed as f64,
    );
    counter(
        "stride_prefill_tokens_reused_total",
        "Context tokens served from the prefix cache.",
        sched.prefill_tokens_reused as f64,
    );
    counter(
        "stride_execution_busy_microseconds_total",
        "Time spent in forward passes.",
        m.busy_us as f64,
    );

    ([("content-type", "text/plain; version=0.0.4")], out).into_response()
}

async fn list_models(State(s): State<AppState>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: s.model_name.clone(),
            object: "model",
            owned_by: if s.simulated {
                "stride-simulated"
            } else {
                "stride"
            },
        }],
    })
}

fn usage_body(u: Usage) -> UsageBody {
    UsageBody {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens(),
        stride_cached_prompt_tokens: u.cached_prompt_tokens,
    }
}

/// Relay a token stream as server-sent events.
fn sse_from(
    stream: TokenStream,
    id: String,
    model: String,
    chat: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    struct S {
        stream: TokenStream,
        id: String,
        model: String,
        chat: bool,
        created: u64,
        sent_role: bool,
        done: bool,
    }

    let state = S {
        stream,
        id,
        model,
        chat,
        created: now_secs(),
        sent_role: false,
        done: false,
    };

    let body = futures::stream::unfold(state, |mut s| async move {
        if s.done {
            return None;
        }
        let event = match s.stream.next().await {
            Some(e) => e,
            None => {
                // The engine dropped the stream without a Done event; close the
                // SSE stream cleanly rather than hanging the client.
                s.done = true;
                return Some((Ok(Event::default().data("[DONE]")), s));
            }
        };

        let payload = match event {
            StreamEvent::Token { text, .. } => {
                if text.is_empty() {
                    // A partial multi-byte character: nothing to show yet.
                    return Some((Ok(Event::default().comment("partial")), s));
                }
                let role = if s.chat && !s.sent_role {
                    s.sent_role = true;
                    Some("assistant")
                } else {
                    None
                };
                if s.chat {
                    serde_json::to_string(&ChatCompletionChunk {
                        id: s.id.clone(),
                        object: "chat.completion.chunk",
                        created: s.created,
                        model: s.model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    })
                } else {
                    serde_json::to_string(&serde_json::json!({
                        "id": s.id,
                        "object": "text_completion",
                        "created": s.created,
                        "model": s.model,
                        "choices": [{"index": 0, "text": text, "finish_reason": null}],
                    }))
                }
            }
            StreamEvent::Done {
                reason,
                trailing_text,
                usage,
            } => {
                s.done = true;
                let finish = finish_reason_str(reason);
                if s.chat {
                    serde_json::to_string(&ChatCompletionChunk {
                        id: s.id.clone(),
                        object: "chat.completion.chunk",
                        created: s.created,
                        model: s.model.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: if trailing_text.is_empty() {
                                    None
                                } else {
                                    Some(trailing_text)
                                },
                            },
                            finish_reason: Some(finish),
                        }],
                        usage: Some(usage_body(usage)),
                    })
                } else {
                    serde_json::to_string(&serde_json::json!({
                        "id": s.id,
                        "object": "text_completion",
                        "created": s.created,
                        "model": s.model,
                        "choices": [{"index": 0, "text": trailing_text, "finish_reason": finish}],
                        "usage": usage_body(usage),
                    }))
                }
            }
        };

        let event = match payload {
            Ok(json) => Event::default().data(json),
            Err(e) => Event::default().data(format!("{{\"error\":\"{e}\"}}")),
        };
        Some((Ok(event), s))
    });

    Sse::new(body).keep_alive(KeepAlive::default())
}

async fn chat_completions(
    State(s): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let params = req.sampling();
    params.validate()?;
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError(Error::InvalidSampling(
            "only n = 1 is implemented".into(),
        )));
    }

    let gen = GenerationRequest {
        tenant: "default".to_string(),
        prompt: req.to_prompt(),
        params,
        class: parse_service_class(req.service_class.as_deref()),
    };
    let (_, stream) = s.engine.generate(gen).await?;
    let id = request_id("chatcmpl");
    let model = if req.model.is_empty() {
        s.model_name.clone()
    } else {
        req.model.clone()
    };

    if req.stream {
        return Ok(sse_from(stream, id, model, true).into_response());
    }

    let (text, usage, reason) = stream.collect().await;
    Ok(Json(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: now_secs(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content: text,
            },
            finish_reason: finish_reason_str(reason.unwrap_or(FinishReason::Stop)),
        }],
        usage: usage_body(usage),
    })
    .into_response())
}

async fn completions(
    State(s): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let params = req.sampling();
    params.validate()?;

    let gen = GenerationRequest {
        tenant: "default".to_string(),
        prompt: req.prompt.clone(),
        params,
        class: parse_service_class(req.service_class.as_deref()),
    };
    let (_, stream) = s.engine.generate(gen).await?;
    let id = request_id("cmpl");
    let model = if req.model.is_empty() {
        s.model_name.clone()
    } else {
        req.model.clone()
    };

    if req.stream {
        return Ok(sse_from(stream, id, model, false).into_response());
    }

    let (text, usage, reason) = stream.collect().await;
    Ok(Json(CompletionResponse {
        id,
        object: "text_completion",
        created: now_secs(),
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason: finish_reason_str(reason.unwrap_or(FinishReason::Stop)),
        }],
        usage: usage_body(usage),
    })
    .into_response())
}
