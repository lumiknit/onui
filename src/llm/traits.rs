use anyhow::Result;
use async_trait::async_trait;

/// Status represents the current status of the LLM client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Idle,
    WaitForLuaResult,
    Generating,
}

#[async_trait(?Send)]
pub trait LLMEventHandler: Send + Sync {
    /// Called when a new chunk of assistant message is received.
    /// Note that the message may be incomplete and streaming.
    async fn on_assistant_chunk(&mut self, msg: &str) -> Result<()>;

    /// Called when tool call lua is requested by the LLM.
    async fn on_lua_call(&mut self, id: &str, code: &str, timeout_sec: Option<u64>) -> Result<()>;

    /// Called when the LLM has finished generating the response.
    async fn on_llm_finished(&mut self) -> Result<()>;
}

/// LLMClient is an interface for sending messages to the LLM.
/// The implementor should have LLMEventHandler when constructed,
/// and after each send, the implementor should call the appropriate
/// LLMEventHandler methods when events occur.
#[async_trait(?Send)]
pub trait LLMClient: Send + Sync {
    /// If running return Generating.
    /// Otherwise, wait for user's input, but,
    /// - If some lua call is pending returns WaitForLuaResult.
    /// - Otherwise returns Idle.
    async fn get_status(&self) -> Status;

    /// Asynchronously send a message to the LLM.
    /// The response will be passed by LLM event handler.
    async fn send_user_msg(&mut self, message: &str) -> Result<()>;

    /// Asynchronously send lua execution results to the LLM.
    /// The results is a list of (id, output) tuples.
    async fn send_lua_results(&mut self, results: &[(String, String)]) -> Result<()>;
}

pub type BoxedLLMClient = Box<dyn LLMClient + Send>;
