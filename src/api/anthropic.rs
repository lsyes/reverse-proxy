use crate::conversation::ConversationIds;
use crate::error::{anthropic_error, ApiError, ApiResult};
use crate::inband::{TextToolCall, TextToolCallScanner};
use crate::sanitize::sanitize_chat_body;
use crate::sse::{sse_event, SseBuffer};
use crate::state::{now_ms, SharedState};
use crate::store::RequestRecord;
use crate::tokens::estimate_chat_tokens;
use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::time::Instant;

pub async fn messages(
    State(state): State<SharedState>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Response {
    let Json(body) = match payload {
        Ok(payload) => payload,
        Err(e) => {
            let status = e.status();
            return anthropic_error(ApiError::new(
                status,
                "invalid_request_error",
                format!("invalid JSON body: {}", e.body_text()),
            ));
        }
    };
    let started = Instant::now();
    let requested = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let model = state.resolve_model(requested.as_deref()).await;
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let conversation = ConversationIds::from_request(&headers, &body);
    match messages_impl(state.clone(), body, model.clone(), conversation).await {
        Ok(r) => r,
        Err(e) => {
            state
                .record_request(RequestRecord {
                    at: now_ms(),
                    model,
                    account_id: String::new(),
                    account_name: String::new(),
                    stream,
                    ok: false,
                    status: e.status.as_u16(),
                    latency_ms: started.elapsed().as_millis() as u64,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    error: Some(e.message.clone()),
                })
                .await;
            anthropic_error(e)
        }
    }
}

/// Anthropic's `count_tokens` endpoint. WorkBuddy does not expose a tokenizer,
/// so this returns a conservative local estimate.
pub async fn count_tokens(payload: Result<Json<Value>, JsonRejection>) -> Response {
    let Json(body) = match payload {
        Ok(payload) => payload,
        Err(e) => {
            let status = e.status();
            return anthropic_error(ApiError::new(
                status,
                "invalid_request_error",
                format!("invalid JSON body: {}", e.body_text()),
            ));
        }
    };
    match count_tokens_impl(body) {
        Ok(input_tokens) => Json(json!({"input_tokens": input_tokens})).into_response(),
        Err(e) => anthropic_error(e),
    }
}

fn count_tokens_impl(body: Value) -> ApiResult<u64> {
    let (openai_body, _) = anthropic_to_openai(&body)?;
    Ok(estimate_chat_tokens(&openai_body))
}

async fn messages_impl(
    state: SharedState,
    body: Value,
    model: String,
    conversation: ConversationIds,
) -> ApiResult<Response> {
    let client_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let emit_thinking = thinking_requested(&body);
    let (openai_body, max_tokens) = anthropic_to_openai(&body)?;
    let mut openai_body = openai_body;
    {
        let uobj = openai_body
            .as_object_mut()
            .ok_or_else(|| ApiError::bad_request("invalid body"))?;
        uobj.insert("model".into(), Value::String(model.clone()));
        uobj.insert("stream".into(), Value::Bool(true));
    }
    // Same gateway compatibility pass as the OpenAI route: a foreign agent's
    // system prompt reaches the upstream fastest through this endpoint too.
    let sanitized = sanitize_chat_body(&mut openai_body);
    if !sanitized.is_noop() {
        tracing::debug!(
            scrubbed_strings = sanitized.scrubbed_strings,
            stripped_reasoning = sanitized.stripped_reasoning,
            model = %model,
            "rewrote anthropic-derived request for upstream channel compatibility"
        );
    }
    let started = Instant::now();
    let (account, resp) = state
        .chat_with_rotation(&openai_body, &model, &conversation)
        .await?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let (msg, code) = crate::upstream::parse_upstream_error(status, &text);
        return Err(ApiError::upstream(status, msg, code));
    }
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();

    if client_stream {
        if !content_type.contains("text/event-stream") {
            let text = resp.text().await.unwrap_or_default();
            return Err(ApiError::upstream(
                StatusCode::BAD_GATEWAY,
                format!(
                    "upstream returned non-SSE body: {}",
                    crate::upstream::truncate(&text, 300)
                ),
                None,
            ));
        }
        let body = anthropic_stream_body(
            state.clone(),
            account,
            resp.bytes_stream(),
            model,
            max_tokens,
            emit_thinking,
            started,
        );
        let mut res = Response::new(body);
        res.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );
        res.headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        res.headers_mut()
            .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
        res.headers_mut()
            .insert("x-accel-buffering", HeaderValue::from_static("no"));
        return Ok(res);
    }

    let text = resp.text().await.unwrap_or_default();
    let mut agg = crate::api::openai::ChatAggregate::default();
    if content_type.contains("text/event-stream") {
        let mut upstream_error = None;
        for line in text.lines() {
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                break;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if let Some(message) = crate::api::openai::stream_error_message(&v) {
                    upstream_error = Some(message);
                    break;
                }
                agg.apply(&v);
            }
        }
        if let Some(message) = upstream_error {
            return Err(ApiError::upstream(StatusCode::BAD_GATEWAY, message, None));
        }
    } else if let Ok(v) = serde_json::from_str::<Value>(&text) {
        // Defensive: tolerate gateways that ignore `stream: true`.
        if v.get("choices").is_some() {
            agg.apply(&v);
        } else {
            let (msg, code) = crate::upstream::parse_upstream_error(StatusCode::BAD_GATEWAY, &text);
            return Err(ApiError::upstream(StatusCode::BAD_GATEWAY, msg, code));
        }
    } else {
        let (msg, code) = crate::upstream::parse_upstream_error(StatusCode::BAD_GATEWAY, &text);
        return Err(ApiError::upstream(StatusCode::BAD_GATEWAY, msg, code));
    }
    // The gateway hides tool invocations in `content` as DSML text; a standard
    // Anthropic client only reads `tool_use` blocks, so recover them first.
    agg.recover_inband_tool_calls();
    let usage = agg.usage.clone();
    let out = to_anthropic_response(&agg, &model, max_tokens, emit_thinking);
    let latency = started.elapsed().as_millis() as u64;
    state
        .record_request(RequestRecord {
            at: now_ms(),
            model: model.clone(),
            account_id: account.id.clone(),
            account_name: account.name.clone(),
            stream: false,
            ok: true,
            status: 200,
            latency_ms: latency,
            prompt_tokens: usage
                .as_ref()
                .and_then(|u| u.get("prompt_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            completion_tokens: usage
                .as_ref()
                .and_then(|u| u.get("completion_tokens"))
                .and_then(Value::as_u64)
                .unwrap_or(0),
            error: None,
        })
        .await;
    Ok(Json(out).into_response())
}

fn anthropic_stream_body(
    state: SharedState,
    account: crate::store::Account,
    upstream: impl futures_util::Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
    model: String,
    _max_tokens: i64,
    emit_thinking: bool,
    started: Instant,
) -> Body {
    let stream = async_stream::stream! {
        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let start = json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        });
        yield Ok::<Bytes, std::io::Error>(Bytes::from(sse_event("message_start", &start.to_string())));

        let mut parser = SseBuffer::new();
        let mut upstream = Box::pin(upstream);
        let mut text_open = false;
        let mut text_seen = false;
        let mut text_block_index: i64 = 0;
        let mut thinking_open = false;
        let mut thinking_block_index: i64 = 0;
        let mut thinking_signature = String::new();
        let mut next_index: i64 = 0;
        let mut pending_tools: BTreeMap<usize, PendingToolCall> = BTreeMap::new();
        let mut recovered: Vec<TextToolCall> = Vec::new();
        // The gateway does not emit structured tool calls: the invocation
        // arrives as DSML text inside `content`. An Anthropic client only acts
        // on `tool_use` blocks, so scan the text and emit the calls; otherwise
        // the markup leaks into the visible answer and the tool never runs.
        let mut scanner = TextToolCallScanner::new();
        let mut finish_reason: Option<String> = None;
        let mut usage: Option<Value> = None;
        let mut failed: Option<String> = None;
        let mut done = false;

        'stream: while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => {
                    for data in parser.push(&bytes) {
                        if data == "[DONE]" {
                            done = true;
                            break;
                        }
                        let Ok(v) = serde_json::from_str::<Value>(&data) else { continue };
                        if let Some(message) = crate::api::openai::stream_error_message(&v) {
                            failed = Some(message.clone());
                            let ev = json!({"type": "error", "error": {"type": "api_error", "message": message}});
                            yield Ok(Bytes::from(sse_event("error", &ev.to_string())));
                            break 'stream;
                        }
                        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                            usage = Some(u.clone());
                        }
                        let Some(choice) = v.get("choices").and_then(|c| c.get(0)) else { continue };
                        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
                            if !fr.is_empty() {
                                finish_reason = Some(fr.to_string());
                            }
                        }
                        let Some(delta) = choice.get("delta") else { continue };

                        // Thinking must be a separate content block and must not
                        // be interleaved with text: Anthropic clients dispatch on
                        // block order. Only emit it when requested — an
                        // unrequested block is leaked reasoning, and clients that
                        // cannot replay it demote it to visible text.
                        if emit_thinking {
                            if let Some(r) = delta
                                .get("reasoning_content")
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                            {
                                if !thinking_open && !text_open {
                                    thinking_open = true;
                                    thinking_block_index = next_index;
                                    next_index += 1;
                                    thinking_signature = new_thinking_signature();
                                    let ev = json!({
                                        "type": "content_block_start",
                                        "index": thinking_block_index,
                                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                                    });
                                    yield Ok(Bytes::from(sse_event("content_block_start", &ev.to_string())));
                                }
                                if thinking_open {
                                    let ev = json!({
                                        "type": "content_block_delta",
                                        "index": thinking_block_index,
                                        "delta": {"type": "thinking_delta", "thinking": r}
                                    });
                                    yield Ok(Bytes::from(sse_event("content_block_delta", &ev.to_string())));
                                }
                            }
                        }

                        if let Some(t) = delta.get("content").and_then(Value::as_str) {
                            if !t.is_empty() {
                                let (visible, calls) = scanner.push(t);
                                recovered.extend(calls);
                                if visible.is_empty() {
                                    continue;
                                }
                                // First visible text ends the thinking phase.
                                if thinking_open {
                                    yield Ok(Bytes::from(sse_event(
                                        "content_block_delta",
                                        &json!({
                                            "type": "content_block_delta",
                                            "index": thinking_block_index,
                                            "delta": {"type": "signature_delta", "signature": thinking_signature}
                                        })
                                        .to_string(),
                                    )));
                                    let ev = json!({
                                        "type": "content_block_stop",
                                        "index": thinking_block_index
                                    });
                                    yield Ok(Bytes::from(sse_event("content_block_stop", &ev.to_string())));
                                    thinking_open = false;
                                }
                                if !text_open {
                                    text_open = true;
                                    text_seen = true;
                                    text_block_index = next_index;
                                    next_index += 1;
                                    let ev = json!({
                                        "type": "content_block_start",
                                        "index": text_block_index,
                                        "content_block": {"type": "text", "text": ""}
                                    });
                                    yield Ok(Bytes::from(sse_event("content_block_start", &ev.to_string())));
                                }
                                let ev = json!({
                                    "type": "content_block_delta",
                                    "index": text_block_index,
                                    "delta": {"type": "text_delta", "text": visible}
                                });
                                yield Ok(Bytes::from(sse_event("content_block_delta", &ev.to_string())));
                            }
                        }

                        // Structured calls are accumulated, not streamed: a name
                        // or argument JSON can span chunks, and Anthropic wants
                        // one complete tool_use block per call.
                        if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                            for (fallback_idx, tc) in tcs.iter().enumerate() {
                                let uidx = tc
                                    .get("index")
                                    .and_then(Value::as_u64)
                                    .map(|v| v as usize)
                                    .unwrap_or(fallback_idx);
                                let pending = pending_tools.entry(uidx).or_default();
                                if let Some(id) = tc.get("id").and_then(Value::as_str) {
                                    if !id.is_empty() {
                                        pending.id = Some(id.to_string());
                                    }
                                }
                                if let Some(name) = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(Value::as_str)
                                {
                                    pending.name.push_str(name);
                                }
                                if let Some(args) = tc
                                    .get("function")
                                    .and_then(|f| f.get("arguments"))
                                    .and_then(Value::as_str)
                                {
                                    pending.arguments.push_str(args);
                                }
                            }
                        }
                    }
                    if done { break; }
                }
                Err(e) => {
                    failed = Some(e.to_string());
                    let ev = json!({"type": "error", "error": {"type": "api_error", "message": format!("upstream stream error: {e}")}});
                    yield Ok(Bytes::from(sse_event("error", &ev.to_string())));
                    break 'stream;
                }
            }
        }

        let output_tokens = usage
            .as_ref()
            .and_then(|u| u.get("completion_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let input_tokens = usage
            .as_ref()
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);

        // Flush the scanner before finalizing: the last delta may still hold a
        // partial marker, or a truncated block may still contain a full call.
        let (tail_text, tail_calls) = scanner.finish();
        recovered.extend(tail_calls);

        // Do not emit a normal message_end after a stream error: Anthropic
        // clients expect the error event to terminate the message.
        if failed.is_none() {
            if !tail_text.is_empty() {
                if thinking_open {
                    yield Ok(Bytes::from(sse_event(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": thinking_block_index,
                            "delta": {"type": "signature_delta", "signature": thinking_signature}
                        })
                        .to_string(),
                    )));
                    let ev = json!({"type": "content_block_stop", "index": thinking_block_index});
                    yield Ok(Bytes::from(sse_event("content_block_stop", &ev.to_string())));
                    thinking_open = false;
                }
                if !text_open {
                    text_open = true;
                    text_seen = true;
                    text_block_index = next_index;
                    next_index += 1;
                    let ev = json!({
                        "type": "content_block_start",
                        "index": text_block_index,
                        "content_block": {"type": "text", "text": ""}
                    });
                    yield Ok(Bytes::from(sse_event("content_block_start", &ev.to_string())));
                }
                let ev = json!({
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": tail_text}
                });
                yield Ok(Bytes::from(sse_event("content_block_delta", &ev.to_string())));
            }

            // Close thinking and text before opening tool blocks. Anthropic
            // content blocks are sequential: a block started while another is
            // still open is rejected by strict clients.
            if thinking_open {
                yield Ok(Bytes::from(sse_event(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": thinking_block_index,
                        "delta": {"type": "signature_delta", "signature": thinking_signature}
                    })
                    .to_string(),
                )));
                let ev = json!({"type": "content_block_stop", "index": thinking_block_index});
                yield Ok(Bytes::from(sse_event("content_block_stop", &ev.to_string())));
            }
            if text_open {
                let ev = json!({"type": "content_block_stop", "index": text_block_index});
                yield Ok(Bytes::from(sse_event("content_block_stop", &ev.to_string())));
            }

            let calls = collect_tool_calls(&pending_tools, &recovered);
            for call in &calls {
                let bi = next_index;
                next_index += 1;
                let ev = json!({
                    "type": "content_block_start",
                    "index": bi,
                    "content_block": {"type": "tool_use", "id": call.id, "name": call.name, "input": {}}
                });
                yield Ok(Bytes::from(sse_event("content_block_start", &ev.to_string())));
                if !call.arguments.is_empty() {
                    let delta = json!({
                        "type": "content_block_delta",
                        "index": bi,
                        "delta": {"type": "input_json_delta", "partial_json": call.arguments}
                    });
                    yield Ok(Bytes::from(sse_event("content_block_delta", &delta.to_string())));
                }
                let ev = json!({"type": "content_block_stop", "index": bi});
                yield Ok(Bytes::from(sse_event("content_block_stop", &ev.to_string())));
            }

            // Anthropic wants at least one content block. A thinking-only turn
            // still gets a trailing empty text block, after the thinking block.
            if !text_seen && calls.is_empty() {
                let bi = next_index;
                let ev = json!({"type": "content_block_start", "index": bi, "content_block": {"type": "text", "text": ""}});
                yield Ok(Bytes::from(sse_event("content_block_start", &ev.to_string())));
                let ev = json!({"type": "content_block_stop", "index": bi});
                yield Ok(Bytes::from(sse_event("content_block_stop", &ev.to_string())));
            }

            let stop_reason = map_stop_reason(finish_reason.as_deref(), !calls.is_empty());
            let ev = json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}
            });
            yield Ok(Bytes::from(sse_event("message_delta", &ev.to_string())));
            yield Ok(Bytes::from(sse_event("message_stop", "{\"type\":\"message_stop\"}")));
        }

        let latency = started.elapsed().as_millis() as u64;
        state.record_request(RequestRecord {
            at: now_ms(),
            model: model.clone(),
            account_id: account.id.clone(),
            account_name: account.name.clone(),
            stream: true,
            ok: failed.is_none(),
            status: 200,
            latency_ms: latency,
            prompt_tokens: input_tokens,
            completion_tokens: output_tokens,
            error: failed,
        }).await;
    };
    Body::from_stream(stream)
}

#[derive(Default)]
struct PendingToolCall {
    id: Option<String>,
    name: String,
    arguments: String,
}

struct FinalToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Merge structured and recovered calls into the blocks to emit.
///
/// The upstream normally uses only one channel; de-duplicating by
/// `(name, arguments)` keeps a gateway that uses both from executing a call
/// twice.
fn collect_tool_calls(
    pending: &BTreeMap<usize, PendingToolCall>,
    recovered: &[TextToolCall],
) -> Vec<FinalToolCall> {
    let mut out = Vec::new();
    let mut seen: BTreeMap<(String, String), ()> = BTreeMap::new();
    for call in recovered {
        push_tool_call(
            &mut out,
            &mut seen,
            &call.name,
            &call.arguments.to_string(),
            None,
        );
    }
    for pending in pending.values() {
        push_tool_call(
            &mut out,
            &mut seen,
            &pending.name,
            &pending.arguments,
            pending.id.clone(),
        );
    }
    out
}

fn push_tool_call(
    out: &mut Vec<FinalToolCall>,
    seen: &mut BTreeMap<(String, String), ()>,
    name: &str,
    arguments: &str,
    id: Option<String>,
) {
    let name = name.trim();
    if name.is_empty() {
        return;
    }
    let key = (name.to_string(), arguments.to_string());
    if seen.insert(key, ()).is_some() {
        return;
    }
    let id = id
        .filter(|id| !id.trim().is_empty())
        .unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4().simple()));
    out.push(FinalToolCall {
        id,
        name: name.to_string(),
        arguments: arguments.to_string(),
    });
}

#[cfg(test)]
async fn test_anthropic_stream(chunks: Vec<Bytes>) -> String {
    test_anthropic_stream_with(chunks, false).await
}

#[cfg(test)]
async fn test_anthropic_stream_with(chunks: Vec<Bytes>, emit_thinking: bool) -> String {
    use futures_util::StreamExt as _;

    let data_dir =
        std::env::temp_dir().join(format!("wb-anthropic-stream-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&data_dir).expect("create test data dir");
    let state = SharedState::new(
        crate::state::AppState::new(
            crate::config::Config::default(),
            data_dir.clone(),
            "wbadmin_test".into(),
        )
        .await
        .expect("state"),
    );
    let account = crate::store::Account::new("test".into(), "token".into());
    let upstream = futures_util::stream::iter(chunks.into_iter().map(Ok));
    let body = anthropic_stream_body(
        state,
        account,
        upstream,
        "auto".into(),
        64,
        emit_thinking,
        Instant::now(),
    );
    let mut body = body.into_data_stream();
    let mut out = String::new();
    while let Some(Ok(chunk)) = body.next().await {
        out.push_str(&String::from_utf8_lossy(&chunk));
    }
    drop(body);
    let _ = std::fs::remove_dir_all(&data_dir);
    out
}

pub fn anthropic_to_openai(body: &Value) -> ApiResult<(Value, i64)> {
    let obj = body
        .as_object()
        .ok_or_else(|| ApiError::bad_request("body must be an object"))?;
    let mut messages: Vec<Value> = Vec::new();

    if let Some(system) = obj.get("system") {
        let text = blocks_to_text(system);
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }

    let msgs = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::bad_request("missing messages"))?;
    for m in msgs {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        match role {
            "assistant" => {
                let mut text = String::new();
                let mut tool_calls = Vec::new();
                if let Some(arr) = content.as_array() {
                    for b in arr {
                        match b.get("type").and_then(Value::as_str).unwrap_or("") {
                            "text" => {
                                if let Some(t) = b.get("text").and_then(Value::as_str) {
                                    text.push_str(t);
                                }
                            }
                            "tool_use" => {
                                let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                                let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                                let input = b.get("input").cloned().unwrap_or_else(|| json!({}));
                                tool_calls.push(json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": input.to_string()}
                                }));
                            }
                            _ => {}
                        }
                    }
                } else if let Some(t) = content.as_str() {
                    text.push_str(t);
                }
                let mut msg = Map::new();
                msg.insert("role".into(), json!("assistant"));
                if text.is_empty() && !tool_calls.is_empty() {
                    msg.insert("content".into(), Value::Null);
                } else {
                    msg.insert("content".into(), Value::String(text));
                }
                if !tool_calls.is_empty() {
                    msg.insert("tool_calls".into(), Value::Array(tool_calls));
                }
                messages.push(Value::Object(msg));
            }
            "user" => {
                if let Some(arr) = content.as_array() {
                    let mut parts: Vec<Value> = Vec::new();
                    for b in arr {
                        match b.get("type").and_then(Value::as_str).unwrap_or("") {
                            "tool_result" => {
                                let tool_id =
                                    b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                                let text = blocks_to_text(b.get("content").unwrap_or(&Value::Null));
                                messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": text,
                                }));
                            }
                            "text" => {
                                if let Some(t) = b.get("text").and_then(Value::as_str) {
                                    parts.push(json!({"type": "text", "text": t}));
                                }
                            }
                            "image" => {
                                if let Some(url) = image_url(b) {
                                    parts.push(
                                        json!({"type": "image_url", "image_url": {"url": url}}),
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                    if !parts.is_empty() {
                        let only_text = parts
                            .iter()
                            .all(|p| p.get("type").and_then(Value::as_str) == Some("text"));
                        if only_text {
                            let text = parts
                                .iter()
                                .filter_map(|p| p.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n");
                            messages.push(json!({"role": "user", "content": text}));
                        } else {
                            messages.push(json!({"role": "user", "content": parts}));
                        }
                    }
                } else {
                    messages.push(json!({"role": "user", "content": content}));
                }
            }
            _ => {
                messages.push(json!({"role": role, "content": content}));
            }
        }
    }

    let max_tokens = obj
        .get("max_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(4096);
    let mut out = Map::new();
    out.insert(
        "model".into(),
        obj.get("model").cloned().unwrap_or(Value::Null),
    );
    out.insert("messages".into(), Value::Array(messages));
    out.insert("max_tokens".into(), json!(max_tokens));
    out.insert("stream".into(), Value::Bool(true));
    for key in ["temperature", "top_p"] {
        if let Some(v) = obj.get(key) {
            out.insert(key.into(), v.clone());
        }
    }
    if let Some(stop) = obj.get("stop_sequences") {
        out.insert("stop".into(), stop.clone());
    }
    // Anthropic spells the thinking budget as `thinking.budget_tokens`; the chat
    // upstream only takes a level. Without this hop a Claude Code caller that
    // asked for extended thinking gets the upstream default instead, which is
    // exactly the "thinking mode is on but nothing is thought" trap.
    if let Some(effort) = reasoning_effort_from_thinking(obj) {
        out.insert("reasoning_effort".into(), Value::String(effort.to_string()));
    }
    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        let converted: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let name = t.get("name").and_then(Value::as_str)?;
                Some(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": t.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": t.get("input_schema").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}})),
                    }
                }))
            })
            .collect();
        if !converted.is_empty() {
            out.insert("tools".into(), Value::Array(converted));
        }
    }
    if let Some(tc) = obj.get("tool_choice") {
        let t = tc.get("type").and_then(Value::as_str).unwrap_or("auto");
        out.insert(
            "tool_choice".into(),
            match t {
                "auto" => json!("auto"),
                "any" => json!("required"),
                "none" => json!("none"),
                "tool" => json!({
                    "type": "function",
                    "function": {"name": tc.get("name").and_then(Value::as_str).unwrap_or("")}
                }),
                _ => json!("auto"),
            },
        );
    }
    Ok((Value::Object(out), max_tokens))
}

/// Map Anthropic's `thinking` request block onto a chat reasoning level.
///
/// Anthropic asks for a token budget, chat asks for a named level, so the
/// budget is bucketed. `{"type": "disabled"}` means "do not think" and is left
/// as an unset field: the upstream treats an absent `reasoning_effort` as its
/// own default, and every model in its catalogue thinks by default.
fn reasoning_effort_from_thinking(obj: &Map<String, Value>) -> Option<&'static str> {
    let thinking = obj.get("thinking")?;
    let kind = thinking
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("enabled");
    // Adaptive thinking has no explicit budget: the model decides. Asking the
    // upstream for the highest level is the closest equivalent; treating it as
    // "not enabled" silently downgrades the caller to the upstream default.
    if kind == "adaptive" {
        return Some("high");
    }
    if kind != "enabled" {
        return None;
    }
    let budget = thinking
        .get("budget_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    Some(match budget {
        i64::MIN..=4095 => "low",
        4096..=16383 => "medium",
        16384..=i64::MAX => "high",
    })
}

/// Whether the caller asked for a thinking content block.
///
/// Anthropic only returns `thinking` blocks when the request enables them.
/// Surfacing the upstream's default reasoning to a caller that did not ask is
/// both a protocol violation and a chain-of-thought leak.
fn thinking_requested(body: &Value) -> bool {
    body.get("thinking")
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| matches!(kind, "enabled" | "adaptive"))
}

/// Opaque, non-empty signature for a synthesized thinking block.
///
/// Anthropic clients never verify it cryptographically; they only echo it back
/// when replaying the assistant turn, and this proxy ignores replayed thinking
/// blocks. What matters is that it is non-empty: compatible clients (pi-ai
/// among them) convert unsigned thinking into visible text on replay, which is
/// exactly the long-session chain-of-thought leak this guard prevents.
fn new_thinking_signature() -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .encode(format!("wbproxy:thinking:{}", uuid::Uuid::new_v4()))
}

fn blocks_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|b| {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    Some(t.to_string())
                } else {
                    b.get("content").map(blocks_to_text)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn image_url(block: &Value) -> Option<String> {
    let src = block.get("source")?;
    match src.get("type").and_then(Value::as_str)? {
        "base64" => {
            let media = src.get("media_type").and_then(Value::as_str)?;
            let data = src.get("data").and_then(Value::as_str)?;
            Some(format!("data:{media};base64,{data}"))
        }
        "url" => src
            .get("url")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        _ => None,
    }
}

pub fn to_anthropic_response(
    agg: &crate::api::openai::ChatAggregate,
    model: &str,
    _max_tokens: i64,
    emit_thinking: bool,
) -> Value {
    let mut content = Vec::new();
    if emit_thinking && !agg.reasoning.is_empty() {
        // Emit a non-empty signature: clients only echo it back, and an
        // unsigned thinking block is demoted to visible assistant text by
        // compatible runtimes (pi-ai, several gateways), which leaks the chain
        // of thought into the replayed conversation.
        content.push(json!({
            "type": "thinking",
            "thinking": agg.reasoning,
            "signature": new_thinking_signature(),
        }));
    }
    if !agg.content.is_empty() {
        content.push(json!({"type": "text", "text": agg.content}));
    }
    for tc in agg.tool_calls.values() {
        if tc.name.is_empty() {
            continue;
        }
        let input: Value =
            serde_json::from_str(&tc.arguments).unwrap_or_else(|_| json!({"_raw": tc.arguments}));
        content.push(json!({
            "type": "tool_use",
            "id": tc.id.clone().unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4().simple())),
            "name": tc.name.clone(),
            "input": input,
        }));
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    let stop_reason = map_stop_reason(agg.finish_reason.as_deref(), agg.has_tool_calls());
    let usage = agg.usage.as_ref();
    json!({
        "id": agg.id.clone().map(|i| format!("msg_{}", i.replace("cmb-", ""))).unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4().simple())),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": usage.and_then(|u| u.get("prompt_tokens")).and_then(Value::as_u64).unwrap_or(0),
            "output_tokens": usage.and_then(|u| u.get("completion_tokens")).and_then(Value::as_u64).unwrap_or(0),
        }
    })
}

fn map_stop_reason(finish: Option<&str>, has_tools: bool) -> &'static str {
    match finish {
        Some("length") => "max_tokens",
        Some("tool_calls") | Some("function_call") => "tool_use",
        Some("content_filter") => "refusal",
        Some("stop") => {
            if has_tools {
                "tool_use"
            } else {
                "end_turn"
            }
        }
        _ => {
            if has_tools {
                "tool_use"
            } else {
                "end_turn"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_system_and_basic_messages() {
        let body = json!({
            "model": "auto",
            "max_tokens": 128,
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
            ]
        });
        let (out, max_tokens) = anthropic_to_openai(&body).expect("convert");
        assert_eq!(max_tokens, 128);
        assert_eq!(out["stream"], true);
        let messages = out["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be concise.");
        assert_eq!(messages[1]["content"], "hello");
        assert_eq!(messages[2]["content"], "hi");
    }

    #[test]
    fn converts_images_and_tool_results() {
        let body = json!({
            "model": "auto",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text", "text": "result"}]}
                ]
            }]
        });
        let (out, _) = anthropic_to_openai(&body).expect("convert");
        let messages = out["messages"].as_array().expect("messages");
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["content"], "result");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(
            messages[1]["content"][0]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn converts_tool_definitions_and_choice() {
        let body = json!({
            "model": "auto",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        });
        let (out, _) = anthropic_to_openai(&body).expect("convert");
        assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            out["tool_choice"],
            json!({"type": "function", "function": {"name": "get_weather"}})
        );
    }

    #[test]
    fn thinking_budget_maps_to_a_chat_reasoning_level() {
        for (budget, expected) in [
            (1024, "low"),
            (4095, "low"),
            (4096, "medium"),
            (8000, "medium"),
            (16384, "high"),
            (64000, "high"),
        ] {
            let body = json!({
                "model": "auto",
                "max_tokens": 64000,
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "messages": [{"role": "user", "content": "hi"}]
            });
            let (out, _) = anthropic_to_openai(&body).expect("convert");
            assert_eq!(out["reasoning_effort"], expected, "budget {budget}");
        }
    }

    #[test]
    fn adaptive_thinking_still_asks_for_reasoning() {
        let body = json!({
            "model": "auto",
            "max_tokens": 64000,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (out, _) = anthropic_to_openai(&body).expect("convert");
        assert_eq!(out["reasoning_effort"], "high");
    }

    #[test]
    fn disabled_thinking_sends_no_reasoning_level() {
        let body = json!({
            "model": "auto",
            "max_tokens": 64,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (out, _) = anthropic_to_openai(&body).expect("convert");
        assert!(out.get("reasoning_effort").is_none());
    }

    #[test]
    fn thinking_becomes_the_first_content_block() {
        let agg = crate::api::openai::ChatAggregate {
            reasoning: "先看调用链".into(),
            content: "改好了".into(),
            ..Default::default()
        };
        let out = to_anthropic_response(&agg, "deepseek-v4.1-flash", 1024, true);
        let blocks = out["content"].as_array().expect("content");
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "先看调用链");
        let signature = blocks[0]["signature"].as_str().expect("signature");
        assert!(
            !signature.is_empty(),
            "an unsigned thinking block is demoted to text"
        );
        assert_eq!(blocks[1]["type"], "text");
    }

    #[test]
    fn thinking_is_not_returned_when_the_caller_did_not_ask() {
        let agg = crate::api::openai::ChatAggregate {
            reasoning: "chain of thought".into(),
            content: "answer".into(),
            ..Default::default()
        };
        let out = to_anthropic_response(&agg, "auto", 1024, false);
        let blocks = out["content"].as_array().expect("content");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "text");
    }

    #[test]
    fn rejects_missing_messages() {
        let err = anthropic_to_openai(&json!({"model": "auto"})).expect_err("should fail");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn converts_tool_calls_in_response() {
        let mut agg = crate::api::openai::ChatAggregate::default();
        agg.apply(&json!({
            "id": "cmb-1",
            "choices": [{
                "delta": {
                    "content": "checking",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-1",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Beijing\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8}
        }));
        let out = to_anthropic_response(&agg, "auto", 64, false);
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["content"][0]["type"], "text");
        assert_eq!(out["content"][1]["type"], "tool_use");
        assert_eq!(out["content"][1]["input"]["city"], "Beijing");
        assert_eq!(out["usage"]["input_tokens"], 12);
    }

    #[test]
    fn maps_stop_reasons() {
        assert_eq!(map_stop_reason(Some("length"), false), "max_tokens");
        assert_eq!(map_stop_reason(Some("tool_calls"), false), "tool_use");
        assert_eq!(map_stop_reason(Some("content_filter"), false), "refusal");
        assert_eq!(map_stop_reason(Some("stop"), false), "end_turn");
        assert_eq!(map_stop_reason(Some("stop"), true), "tool_use");
    }

    #[tokio::test]
    async fn stream_combines_split_tool_names_and_arguments() {
        let chunks = vec![
            Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"function\":{\"name\":\"get_\",\"arguments\":\"{\\\"ci\"}}]}}]}\n\n",
            ),
            Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"weather\",\"arguments\":\"ty\\\":\\\"Beijing\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n",
            ),
        ];
        let out = test_anthropic_stream(chunks).await;

        assert!(out.contains(r#""name":"get_weather""#), "{out}");
        assert_eq!(out.matches(r#""type":"tool_use""#).count(), 1, "{out}");
        assert!(
            out.contains(r#""partial_json":"{\"city\":\"Beijing\"}""#),
            "{out}"
        );
        assert!(out.contains(r#""stop_reason":"tool_use""#), "{out}");
        assert!(out.contains(r#""type":"message_stop""#), "{out}");
    }

    fn sse_frame(delta: Value, finish_reason: Value) -> Bytes {
        let frame = json!({
            "choices": [{"delta": delta, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        });
        Bytes::from(format!("data: {frame}\n\n"))
    }

    fn dsml_read_call() -> String {
        let bar = "\u{FF5C}\u{FF5C}";
        format!(
            "let me look\n<{bar}DSML{bar} calls>\n<{bar}DSML{bar} invoke name=\"read\">\n<{bar}DSML{bar} parameter name=\"path\" string=\"true\">/etc/hostname</{bar}DSML{bar} parameter>\n</{bar}DSML{bar} invoke>\n</{bar}DSML{bar} calls>"
        )
    }

    #[test]
    fn non_streaming_recovers_inband_calls_before_building_the_response() {
        let markup = dsml_read_call();
        let markup = markup.split_once('\n').expect("dsml has a line break").1;
        let mut agg = crate::api::openai::ChatAggregate::default();
        agg.apply(&json!({
            "choices": [{"delta": {"content": markup}, "finish_reason": "stop"}]
        }));
        assert!(agg.recover_inband_tool_calls());
        let out = to_anthropic_response(&agg, "auto", 64, false);
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["content"][0]["type"], "tool_use");
        assert_eq!(out["content"][0]["name"], "read");
        assert_eq!(out["content"][0]["input"]["path"], "/etc/hostname");
        assert!(!out.to_string().contains('\u{FF5C}'));
    }

    #[tokio::test]
    async fn stream_recovers_inband_tool_calls_instead_of_leaking_markup() {
        let chunks = vec![sse_frame(
            json!({"content": dsml_read_call()}),
            json!("stop"),
        )];
        let out = test_anthropic_stream(chunks).await;
        assert!(out.contains(r#""type":"tool_use""#), "{out}");
        assert!(out.contains(r#""name":"read""#), "{out}");
        assert!(
            out.contains(r#""partial_json":"{\"path\":\"/etc/hostname\"}""#),
            "{out}"
        );
        assert!(out.contains(r#""stop_reason":"tool_use""#), "{out}");
        assert!(out.contains("let me look"), "{out}");
        assert!(!out.contains('\u{FF5C}'), "raw DSML markup leaked: {out}");
    }

    #[tokio::test]
    async fn stream_emits_signed_thinking_only_when_requested() {
        let chunks = vec![
            sse_frame(
                json!({"reasoning_content": "consider the call chain"}),
                Value::Null,
            ),
            sse_frame(json!({"content": "done"}), json!("stop")),
        ];
        let with = test_anthropic_stream_with(chunks.clone(), true).await;
        assert!(with.contains(r#""type":"thinking""#), "{with}");
        assert!(with.contains(r#""type":"signature_delta""#), "{with}");
        assert!(with.contains(r#""type":"thinking_delta""#), "{with}");
        let signature_at = with.find(r#""type":"signature_delta""#).unwrap();
        let stop_at = with
            .find(r#""index":0,"type":"content_block_stop""#)
            .unwrap_or_else(|| panic!("thinking block must close: {with}"));
        assert!(signature_at < stop_at, "signature must precede block stop");

        let without = test_anthropic_stream_with(chunks, false).await;
        assert!(!without.contains(r#""type":"thinking""#), "{without}");
        assert!(without.contains(r#""type":"text""#), "{without}");
    }

    #[tokio::test]
    async fn stream_closes_thinking_before_a_recovered_tool_block() {
        // Pure markup: no visible prose precedes the call, so a correct stream
        // has thinking(0) closed before tool_use(1) starts.
        let markup = dsml_read_call();
        let markup = markup.split_once('\n').expect("dsml has a line break").1;
        let chunks = vec![
            sse_frame(json!({"reasoning_content": "plan"}), Value::Null),
            sse_frame(json!({"content": markup}), json!("stop")),
        ];
        let out = test_anthropic_stream_with(chunks, true).await;
        let thinking_stop = out
            .find(r#""index":0,"type":"content_block_stop""#)
            .unwrap_or_else(|| panic!("thinking must close: {out}"));
        let tool_start = out
            .find(r#""index":1,"type":"content_block_start""#)
            .unwrap_or_else(|| panic!("tool block must follow thinking: {out}"));
        assert!(
            thinking_stop < tool_start,
            "content blocks must not interleave: {out}"
        );
        assert!(out.contains(r#""type":"tool_use""#), "{out}");
        assert!(out.contains(r#""stop_reason":"tool_use""#), "{out}");
        assert!(!out.contains(r#""type":"text""#), "{out}");
    }
}
