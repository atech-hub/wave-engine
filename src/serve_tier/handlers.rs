//! HTTP request handlers — connects api_types, prompt, and inference.
//! Zero forward pass code — all computation goes through common/.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use axum::response::sse::{Event, Sse};
use axum::http::StatusCode;
use tokio_stream::wrappers::ReceiverStream;

use super::api_types::*;
use super::inference::{self, GenerationConfig};
use super::prompt;
use super::server::AppState;

/// POST /v1/chat/completions
pub async fn handle_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, StatusCode> {
    if req.messages.is_empty() {
        let err = ErrorResponse {
            error: ErrorDetail {
                message: "messages array must not be empty".to_string(),
                r#type: "invalid_request_error".to_string(),
            },
        };
        return Ok((StatusCode::BAD_REQUEST, Json(err)).into_response());
    }

    let prompt_tokens = prompt::format_chat(&req.messages, &state.vocab);
    if prompt_tokens.is_empty() {
        let err = ErrorResponse {
            error: ErrorDetail {
                message: "prompt encoded to zero tokens".to_string(),
                r#type: "invalid_request_error".to_string(),
            },
        };
        return Ok((StatusCode::BAD_REQUEST, Json(err)).into_response());
    }

    let block_size = state.dims.block_size;
    let prompt_tokens = if prompt_tokens.len() > block_size {
        prompt_tokens[prompt_tokens.len() - block_size..].to_vec()
    } else {
        prompt_tokens
    };

    let config = GenerationConfig {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
    };

    let model_name = state.model_name.clone();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

    let prompt_len = prompt_tokens.len();

    if req.stream {
        Ok(handle_streaming(state, prompt_tokens, config, model_name, request_id, created).await)
    } else {
        Ok(handle_non_streaming(state, prompt_tokens, config, model_name, request_id, created, prompt_len).await)
    }
}

async fn handle_non_streaming(
    state: Arc<AppState>,
    prompt_tokens: Vec<usize>,
    config: GenerationConfig,
    model_name: String,
    request_id: String,
    created: u64,
    prompt_len: usize,
) -> Response {
    let model = state.model.clone();
    let vocab = state.vocab.clone();
    let dims = state.dims;
    let stencil = state.stencil.clone();

    // Build memory offsets if memory is loaded
    let mem_offsets = state.memory.as_ref().map(|m| {
        let mem = m.lock().unwrap();
        crate::common::wave_memory::build_offsets(&mem)
    });
    let mem_slices: Option<Vec<(&[f32], &[f32])>> = mem_offsets.as_ref().map(|o| o.as_slices());
    // Clone for move into spawn_blocking
    let mem_for_gen: Option<Vec<(Vec<f32>, Vec<f32>)>> = mem_offsets.as_ref().map(|o| {
        o.offsets.iter().map(|(r, s)| (r.clone(), s.clone())).collect()
    });

    let result = tokio::task::spawn_blocking(move || {
        let mem_refs: Option<Vec<(&[f32], &[f32])>> = mem_for_gen.as_ref().map(|v| {
            v.iter().map(|(r, s)| (r.as_slice(), s.as_slice())).collect()
        });
        inference::generate(&model, &prompt_tokens, &config, &vocab, dims, &stencil,
            mem_refs.as_deref())
    })
    .await
    .unwrap();

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        created,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: result.text,
            },
            finish_reason: "length".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: result.tokens.len(),
            total_tokens: prompt_len + result.tokens.len(),
        },
    };

    // Accumulate memory after generation (non-streaming saves immediately)
    if let (Some(mem_lock), Some(path)) = (&state.memory, &state.memory_path) {
        if let Ok(mut mem) = mem_lock.lock() {
            // Simple accumulation: increment conversation count
            // Full ODE state extraction would need a separate forward pass
            mem.n_convos += 1;
            crate::common::wave_memory::save(path, &mem);
        }
    }

    Json(response).into_response()
}

async fn handle_streaming(
    state: Arc<AppState>,
    prompt_tokens: Vec<usize>,
    config: GenerationConfig,
    model_name: String,
    request_id: String,
    created: u64,
) -> Response {
    let model = state.model.clone();
    let vocab = state.vocab.clone();
    let dims = state.dims;
    let stencil = state.stencil.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    // Send initial chunk with role
    let initial_chunk = ChatCompletionChunk {
        id: request_id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model_name.clone(),
        choices: vec![DeltaChoice {
            index: 0,
            delta: Delta { role: Some("assistant".to_string()), content: None },
            finish_reason: None,
        }],
    };
    let _ = tx.send(Ok(Event::default()
        .data(serde_json::to_string(&initial_chunk).unwrap()))).await;

    let tx_clone = tx.clone();
    let req_id = request_id;
    let mn = model_name;

    // Build memory offsets for streaming
    let mem_for_stream: Option<Vec<(Vec<f32>, Vec<f32>)>> = state.memory.as_ref().map(|m| {
        let mem = m.lock().unwrap();
        let offsets = crate::common::wave_memory::build_offsets(&mem);
        offsets.offsets.iter().map(|(r, s)| (r.clone(), s.clone())).collect()
    });

    tokio::task::spawn_blocking(move || {
        let mem_refs: Option<Vec<(&[f32], &[f32])>> = mem_for_stream.as_ref().map(|v| {
            v.iter().map(|(r, s)| (r.as_slice(), s.as_slice())).collect()
        });
        inference::generate_streaming(&model, &prompt_tokens, &config, &vocab, dims, &stencil, mem_refs.as_deref(), |event| {
            let chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: mn.clone(),
                choices: vec![DeltaChoice {
                    index: 0,
                    delta: Delta { role: None, content: Some(event.text) },
                    finish_reason: if event.done { Some("length".to_string()) } else { None },
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap();
            tx_clone.blocking_send(Ok(Event::default().data(json))).is_ok()
        });
        let _ = tx_clone.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    Sse::new(ReceiverStream::new(rx)).into_response()
}

/// GET /v1/models
pub async fn handle_models(
    State(state): State<Arc<AppState>>,
) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "wave-engine".to_string(),
        }],
    })
}

/// GET /health
pub async fn handle_health(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_name.clone(),
        vocab_size: state.vocab.vocab_size,
        n_embd: state.dims.n_embd,
        n_layers: state.model.blocks.len(),
    })
}
