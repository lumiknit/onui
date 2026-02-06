use anyhow::Result;
use std::{future::Future, pin::Pin};

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub lua_code: Option<String>,
    pub lua_timeout_sec: Option<u64>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            lua_code: None,
            lua_timeout_sec: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            lua_code: None,
            lua_timeout_sec: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(
        content: Option<String>,
        lua_code: Option<String>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            lua_code,
            lua_timeout_sec: None,
            tool_call_id,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            lua_code: None,
            lua_timeout_sec: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

pub trait LLMClient: Send + Sync {
    fn chat<'a>(
        &'a self,
        history: &'a [ChatMessage],
    ) -> Pin<Box<dyn Future<Output = Result<ChatMessage>> + Send + 'a>>;
}
