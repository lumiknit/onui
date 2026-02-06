use anyhow::Result;
use async_trait::async_trait;

#[async_trait(?Send)]
pub trait LLMEventHandler {
    /// Called when a new chunk of assistant message is received.
    /// Note that the message may be incomplete and streaming.
    async fn on_assistant_chunk(&mut self, msg: &str) -> Result<()>;

    /// Called when tool call lua is requested by the LLM.
    async fn on_lua_call(&mut self, code: &str) -> Result<String>;
}

/// LLMClient is an interface for sending messages to the LLM.
/// The implementor should have LLMEventHandler when constructed,
/// and after each send, the implementor should call the appropriate
/// LLMEventHandler methods when events occur.
#[async_trait(?Send)]
pub trait LLMClient {
    /// Asynchronously send a message to the LLM.
    /// The response will be passed by LLM event handler.
    async fn send_user_msg(&mut self, message: &str);

    /// Asynchronously send a message to the LLM.
    /// The response will be passed by LLM event handler.
    async fn send_lua_result(&mut self, message: &str);

    /// Asynchronously send a message to the LLM indicating Lua execution was rejected.
    /// The response will be passed by LLM event handler.
    async fn send_lua_rejected(&mut self, message: &str);

    /// Asynchronously check if the LLM is idle (not processing any request).
    async fn is_idle(&self) -> bool;

    /// Asynchronously cancel the current LLM operation.
    async fn cancel(&mut self) -> Result<()>;
}
