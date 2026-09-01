//! Request and response types for the OpenAI-compatible surface.
//!
//! Field names and shapes follow the OpenAI API so existing clients work
//! unmodified. Where Stride reports something the API has no field for —
//! notably how much of a prompt was served from the KV cache — it is added
//! under a namespaced extension rather than by overloading a standard field.

use serde::{Deserialize, Serialize};
use stride_core::{FinishReason, SamplingParams, ServiceClass};

fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_max_tokens() -> usize {
    256
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub min_tokens: usize,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    /// Completions to return. Only `n = 1` is implemented; a larger value is
    /// rejected rather than silently returning one choice.
    #[serde(default)]
    pub n: Option<usize>,
    /// Stride extension: which latency class to admit this request under.
    #[serde(default, rename = "stride_service_class")]
    pub service_class: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, rename = "stride_service_class")]
    pub service_class: Option<String>,
}

pub fn parse_service_class(name: Option<&str>) -> ServiceClass {
    match name.map(str::to_ascii_lowercase).as_deref() {
        Some("batch") => ServiceClass::Batch,
        Some("background") => ServiceClass::Background,
        _ => ServiceClass::Interactive,
    }
}

impl ChatCompletionRequest {
    /// Flatten messages into a prompt.
    ///
    /// A real deployment applies the model's own chat template; this is the
    /// neutral fallback used when no template is loaded, and it is kept
    /// deliberately simple so it is obvious which is in play.
    pub fn to_prompt(&self) -> String {
        let mut out = String::new();
        for m in &self.messages {
            out.push_str(&m.role);
            out.push_str(": ");
            out.push_str(&m.content);
            out.push('\n');
        }
        out.push_str("assistant: ");
        out
    }

    pub fn sampling(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_tokens: self.max_tokens,
            min_tokens: self.min_tokens,
            stop_tokens: Vec::new(),
            seed: self.seed,
            ignore_eos: false,
        }
    }
}

impl CompletionRequest {
    pub fn sampling(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_tokens: self.max_tokens,
            min_tokens: 0,
            stop_tokens: Vec::new(),
            seed: self.seed,
            ignore_eos: false,
        }
    }
}

pub fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
        FinishReason::Cancelled => "cancelled",
        // Not an OpenAI value. A preempted sequence that never recovered did
        // not stop for a model reason, and saying "stop" would misreport it.
        FinishReason::Preempted => "preempted",
    }
}

#[derive(Debug, Serialize)]
pub struct UsageBody {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Stride extension: prompt tokens served from the prefix cache.
    pub stride_cached_prompt_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatResponseMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChatResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageBody,
}

#[derive(Debug, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageBody>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageBody,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetail {
    pub message: String,
    pub r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
}
