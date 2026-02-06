/// mod io is the IO interaction module for User or other systems.
pub mod cli;

use anyhow::Result;
use tokio::sync::mpsc;

pub enum UserMsg {
    Exit,
    Cancel,
    Input(String),
}

/// IO is an interface for handling LLM or User's interactions.
pub trait IO {
    /// Trigger when IO starts.
    /// This is an helper to setup IO resources.
    fn open(&mut self) -> Result<()>;

    /// Trigger when IO ends.
    /// This is an helper to cleanup IO resources.
    fn close(&mut self) -> Result<()>;

    /// Get a channel receiver to listen for user messages.
    fn input_channel(&mut self) -> mpsc::Receiver<UserMsg>;

    /// Handle to display a message from the system to the user.
    /// System message is complete lines.
    async fn msg_system(&mut self, message: &str) -> Result<()>;

    /// Handle to display a message from the AI assistant to the user.
    /// Assistant message may be streaming. At the end of message, empty string is sent.
    async fn msg_assistant(&mut self, message: &str) -> Result<()>;

    /// Handle to display a message from the Lua code to the user.
    /// User may need to approve or reject the code execution, hence returns a bool.
    async fn msg_lua(&mut self, code: &str) -> Result<bool>;

    /// Handle to display the result of Lua code execution to the user.
    /// Output is complete lines.
    async fn msg_lua_result(&mut self, output: &str) -> Result<()>;

    /// Handle to display LLM stopped. (Answer finished, etc.)
    /// You may show a prompt or do nothing.
    async fn llm_stopped(&mut self) -> Result<()>;
}
