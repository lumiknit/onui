use crate::config::Config;
use crate::io::{IO, UserMsg};
use crate::llm::{ChatMessage, LLMClient};
use crate::lua::{LuaExecution, LuaRuntime};
use anyhow::{Context, Result};

pub struct Agent<L, R, I> {
    llm: L,
    lua: R,
    io: I,
    history: Vec<ChatMessage>,
}

enum CommandResult {
    Exit,
    Handled,
    NotCommand,
}

impl<L, R, I> Agent<L, R, I>
where
    L: LLMClient,
    R: LuaRuntime,
    I: IO,
{
    pub fn new(llm: L, lua: R, io: I, system_prompt: Option<String>) -> Self {
        let mut history = Vec::new();
        if let Some(prompt) = system_prompt {
            history.push(ChatMessage::system(prompt));
        }

        Self {
            llm,
            lua,
            io,
            history,
        }
    }

    pub fn reset_vm(&mut self) -> Result<()> {
        self.lua.reset()
    }

    pub fn new_session(&mut self) {
        let system_prompt = self
            .history
            .iter()
            .find(|msg| msg.role == "system")
            .and_then(|msg| msg.content.clone());

        self.history.clear();
        if let Some(prompt) = system_prompt {
            self.history.push(ChatMessage::system(prompt));
        }
    }

    pub async fn run(&mut self, config: &Config) -> Result<()> {
        self.io
            .msg_system(&format!(
                "Running onui with config: {}\n- LLM: {}\n- cwd: {}",
                config
                    .config_path
                    .clone()
                    .map_or("N/A".to_string(), |p| p.display().to_string()),
                config.default_llm,
                config.workspace_dir().display()
            ))
            .await?;
        self.io.llm_stopped().await?;

        let mut input_rx = self.io.input_channel();
        while let Some(message) = input_rx.recv().await {
            match message {
                UserMsg::Exit => {
                    self.io.msg_system("Goodbye.").await?;
                    return Ok(());
                }
                UserMsg::Cancel => {
                    self.io
                        .msg_system("Cancelled. 한번 더 누르면 종료됩니다")
                        .await?;
                    self.io.llm_stopped().await?;
                }
                UserMsg::Input(line) => {
                    match self.handle_command(&line).await? {
                        CommandResult::Exit => return Ok(()),
                        CommandResult::Handled => {
                            self.io.llm_stopped().await?;
                            continue;
                        }
                        CommandResult::NotCommand => {}
                    }

                    let response = self.handle_user_input(&line).await?;
                    if let Some(content) = response.content.as_deref() {
                        if !content.is_empty() {
                            self.io.msg_assistant(content).await?;
                        }
                    }
                    self.io.msg_assistant("").await?;
                    self.io.llm_stopped().await?;
                }
            }
        }

        Ok(())
    }

    async fn handle_command(&mut self, line: &str) -> Result<CommandResult> {
        let trimmed = line.trim();
        if !trimmed.starts_with('/') {
            return Ok(CommandResult::NotCommand);
        }

        let normalized = trimmed.to_ascii_lowercase();
        match normalized.as_str() {
            "/exit" | "/quit" => {
                self.io.msg_system("Goodbye.").await?;
                return Ok(CommandResult::Exit);
            }
            "/help" => {
                self.io
                    .msg_system("Commands: /help, /new, /reset-vm, /exit")
                    .await?;
            }
            "/new" => {
                self.new_session();
                self.io.msg_system("New conversation started.").await?;
            }
            "/reset-vm" => {
                self.reset_vm()?;
                self.io.msg_system("Lua VM reset.").await?;
            }
            _ => {
                self.io
                    .msg_system(&format!("Unknown command: {}", line))
                    .await?;
            }
        }

        Ok(CommandResult::Handled)
    }

    pub async fn handle_user_input(&mut self, input: &str) -> Result<ChatMessage> {
        self.history.push(ChatMessage::user(input));

        loop {
            let response = self.llm.chat(&self.history).await?;
            let lua_code = response.lua_code.clone();
            let tool_call_id = response.tool_call_id.clone();
            self.history.push(response.clone());

            if let Some(code) = lua_code {
                self.io.msg_system(&format!("lua 실행:\n{}", code)).await?;

                let approved = self.io.msg_lua(&code).await?;
                if !approved {
                    let tool_call_id = tool_call_id.unwrap_or_else(|| "call_lua".to_string());
                    let tool_output = "Lua execution rejected by user.".to_string();
                    self.history
                        .push(ChatMessage::tool(tool_call_id, tool_output.clone()));
                    self.io.msg_system("Lua execution rejected.").await?;
                    continue;
                }

                let execution = self
                    .lua
                    .execute_script(&code, response.lua_timeout_sec)
                    .context("lua script execution failed")?;
                let tool_output = self.render_tool_output(&execution);
                self.io.msg_lua_result(&tool_output).await?;

                let tool_call_id = tool_call_id.unwrap_or_else(|| "call_lua".to_string());
                self.history
                    .push(ChatMessage::tool(tool_call_id, tool_output));
                continue;
            }

            return Ok(response);
        }
    }

    fn render_tool_output(&self, execution: &LuaExecution) -> String {
        let mut output = execution.stdout.clone();
        if !execution.returns.is_empty() {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(&execution.returns.join("\t"));
            output.push('\n');
        }
        if let Some(ref error) = execution.error {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(error);
        }
        output
    }
}
