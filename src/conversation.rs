//! Correlation ids for one upstream chat request.
//!
//! The copilot gateway joins the steps of a conversation by the
//! `X-Conversation-*` header family, and the official client reuses a single
//! `X-Conversation-ID` for the lifetime of one chat while rotating the
//! request/message ids on every turn.
//!
//! The proxy originally derived `X-Conversation-ID` from the *account* id, so
//! every chat served by one account — the main agent, its subagents, unrelated
//! clients — was appended to the same upstream conversation. Once the gateway
//! uses that id to correlate a conversation, the reasoning of one chat can
//! resurface in another, and the shared history keeps growing for as long as
//! the process runs.
//!
//! Identity therefore comes from the client, in this order:
//!
//! 1. its own `X-Conversation-ID` / `X-Conversation-Request-ID` /
//!    `X-Conversation-Message-ID` headers, forwarded verbatim;
//! 2. an explicit session key (session header or body field), hashed into a
//!    stable UUID so the turns of one client session stay together;
//! 3. otherwise a fresh UUID *per request*.
//!
//! Message text is deliberately not used as an identity: two unrelated chats
//! that happen to open with the same user message must not be joined.

use axum::http::HeaderMap;
use serde_json::Value;
use std::fmt::Write as _;

/// Headers through which clients announce their session.
///
/// `x-client-request-id` is a session id for the OpenAI Responses/Codex
/// clients we target (they set it once per conversation), not a per-request id.
const SESSION_HEADERS: &[&str] = &[
    "x-session-id",
    "x-claude-code-session-id",
    "x-opencode-session",
    "x-session-affinity",
    "x-client-request-id",
    "session_id",
    "session-id",
];

/// Body fields that carry an explicit session identity.
const SESSION_FIELDS: &[&str] = &["conversation_id", "session_id", "prompt_cache_key"];

/// The three ids attached to one upstream call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationIds {
    /// Stable for every turn of the same client conversation.
    pub conversation_id: String,
    /// Fresh for each upstream request; retries reuse it.
    pub request_id: String,
    /// Id of the message itself; mirrors `X-Request-ID`.
    pub message_id: String,
}

impl ConversationIds {
    /// Resolve ids from the inbound request.
    pub fn from_request(headers: &HeaderMap, body: &Value) -> Self {
        let request_id =
            header_value(headers, "x-conversation-request-id").unwrap_or_else(fresh_uuid);
        let message_id = header_value(headers, "x-conversation-message-id")
            .unwrap_or_else(|| request_id.clone());
        let conversation_id = header_value(headers, "x-conversation-id")
            .or_else(|| session_key(headers, body).map(|key| stable_uuid(&key)))
            .unwrap_or_else(fresh_uuid);
        Self {
            conversation_id,
            request_id,
            message_id,
        }
    }

    /// Ids for an internal request that has no client behind it (account test).
    pub fn fresh() -> Self {
        let request_id = fresh_uuid();
        Self {
            conversation_id: fresh_uuid(),
            message_id: request_id.clone(),
            request_id,
        }
    }
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    let value = headers.get(name)?.to_str().ok()?.trim();
    (!value.is_empty()).then(|| value.to_string())
}

/// The client's explicit session identity, if it announced one.
fn session_key(headers: &HeaderMap, body: &Value) -> Option<String> {
    for name in SESSION_HEADERS {
        if let Some(value) = header_value(headers, name) {
            return Some(value);
        }
    }
    for field in SESSION_FIELDS {
        if let Some(value) = body
            .get(field)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(value.to_string());
        }
    }
    None
}

fn fresh_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Deterministic UUID-shaped id derived from a client session key.
///
/// The value is an opaque correlation id, not a capability or a secret, so a
/// non-cryptographic hash is enough. Two independently salted SipHash passes
/// fill the 128 bits: a single FNV pass was tried first, but its high bits
/// barely move for keys that differ only in the last byte (`sess-1` and
/// `sess-2` shared their whole UUID prefix). The version/variant bits are set
/// so the gateway's shape check (if any) sees an RFC-4122 UUID.
fn stable_uuid(key: &str) -> String {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&salted_hash(key, 0x9e37_79b9_7f4a_7c15).to_be_bytes());
    bytes[8..].copy_from_slice(&salted_hash(key, 0xc2b2_ae3d_27d4_eb4f).to_be_bytes());
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    let mut hex = String::with_capacity(32);
    for byte in bytes {
        let _ = write!(hex, "{byte:02x}");
    }
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

fn salted_hash(key: &str, salt: u64) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    salt.hash(&mut hasher);
    key.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                axum::http::HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    #[test]
    fn client_conversation_ids_are_forwarded_verbatim() {
        let ids = ConversationIds::from_request(
            &headers(&[
                ("x-conversation-id", "conv-1"),
                ("X-Conversation-Request-ID", "req-1"),
                ("x-conversation-message-id", "msg-1"),
            ]),
            &json!({}),
        );
        assert_eq!(ids.conversation_id, "conv-1");
        assert_eq!(ids.request_id, "req-1");
        assert_eq!(ids.message_id, "msg-1");
    }

    #[test]
    fn same_session_header_joins_turns() {
        let body = json!({"messages": [{"role": "user", "content": "one"}]});
        let first = ConversationIds::from_request(&headers(&[("x-session-id", "s1")]), &body);
        let second = ConversationIds::from_request(
            &headers(&[("X-Session-ID", "s1")]),
            &json!({"messages": [{"role": "user", "content": "two"}]}),
        );
        assert_eq!(first.conversation_id, second.conversation_id);
        assert_ne!(first.request_id, second.request_id);
        assert!(
            first.conversation_id.len() == 36,
            "{}",
            first.conversation_id
        );
    }

    #[test]
    fn body_session_fields_join_turns() {
        let first = ConversationIds::from_request(
            &HeaderMap::new(),
            &json!({"prompt_cache_key": "sess-1"}),
        );
        let second = ConversationIds::from_request(
            &HeaderMap::new(),
            &json!({"prompt_cache_key": "sess-1"}),
        );
        assert_eq!(first.conversation_id, second.conversation_id);

        let other =
            ConversationIds::from_request(&HeaderMap::new(), &json!({"conversation_id": "sess-1"}));
        assert_eq!(first.conversation_id, other.conversation_id);
    }

    #[test]
    fn no_session_identity_means_one_conversation_per_request() {
        let body = json!({"messages": [{"role": "user", "content": "same opening"}]});
        let a = ConversationIds::from_request(&HeaderMap::new(), &body);
        let b = ConversationIds::from_request(&HeaderMap::new(), &body);
        assert_ne!(a.conversation_id, b.conversation_id);
        assert_ne!(a.request_id, b.request_id);
    }

    #[test]
    fn different_sessions_do_not_share_a_conversation() {
        let a = ConversationIds::from_request(&headers(&[("x-session-id", "a")]), &json!({}));
        let b = ConversationIds::from_request(&headers(&[("x-session-id", "b")]), &json!({}));
        assert_ne!(a.conversation_id, b.conversation_id);
    }

    #[test]
    fn blank_client_ids_count_as_absent() {
        let ids = ConversationIds::from_request(
            &headers(&[
                ("x-conversation-id", "   "),
                ("x-conversation-request-id", ""),
            ]),
            &json!({}),
        );
        assert!(!ids.conversation_id.trim().is_empty());
        assert!(!ids.request_id.trim().is_empty());
        assert!(
            ids.conversation_id.contains('-'),
            "expected uuid shape: {}",
            ids.conversation_id
        );
    }

    #[test]
    fn stable_uuid_is_deterministic_and_conversation_scoped() {
        assert_eq!(stable_uuid("s"), stable_uuid("s"));
        assert_ne!(stable_uuid("a"), stable_uuid("b"));
        assert_eq!(stable_uuid("s").len(), 36);
        // Similar keys must not share an id prefix: the gateway may group on
        // the whole value, but a weak prefix is a smell and makes ids look
        // suspiciously correlated in logs.
        assert_ne!(
            stable_uuid("sess-1").split('-').next(),
            stable_uuid("sess-2").split('-').next()
        );
    }
}
