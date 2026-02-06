use crate::config::Config;
use crate::io::{IO, UserMsg};
use crate::llm::{LLMClient, LLMEventHandler};
use crate::lua::{LuaExecution, LuaRuntime};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct AgentResources<R, I> {
    lua: R,
    io: I,
}

impl<R, I> AgentResources<R, I> {
    pub fn new(lua: R, io: I) -> Self {
        Self { lua, io }
    }
}

pub struct AgentHandler<R, I> {
    resources: Arc<Mutex<AgentResources<R, I>>>,
}

impl<R, I> AgentHandler<R, I> {
    pub fn new(resources: Arc<Mutex<AgentResources<R, I>>>) -> Self {
        Self { resources }
    }
}

enum CommandResult {
    Exit,
    Handled,
    NotCommand,
}

pub struct Agent<R, I> {
    llm: Box<dyn LLMClient>,
    resources: Arc<Mutex<AgentResources<R, I>>>,
    config: Config,
}

#[async_trait(?Send)]
impl<R, I> LLMEventHandler for AgentHandler<R, I>
where
    R: LuaRuntime,
    I: IO,
{
    async fn on_assistant_chunk(&mut self, msg: &str) -> Result<()> {
        let mut guard = self.resources.lock().await;
        guard.io.msg_assistant(msg).await?;
        if msg.is_empty() {
            guard.io.llm_stopped().await?;
        }
        Ok(())
    }

    async fn on_lua_call(&mut self, code: &str) -> Result<String> {
        let mut guard = self.resources.lock().await;

        let approved = guard.io.msg_lua(code).await?;
        if !approved {
            guard.io.msg_system("Lua execution rejected.").await?;
            return Ok("Lua execution rejected by user.".to_string());
        }

        let execution = guard
            .lua
            .execute_script(code, None)
            .context("lua script execution failed")?;
        let tool_output = render_tool_output(&execution);
        guard.io.msg_lua_result(&tool_output).await?;
        Ok(tool_output)
    }
}

impl<R, I> Agent<R, I>
where
    R: LuaRuntime,
    I: IO,
{
    pub fn new(
        config: &Config,
        llm: Box<dyn LLMClient>,
        resources: Arc<Mutex<AgentResources<R, I>>>,
    ) -> Self {
        Self {
            llm,
            resources,
            config: config.clone(),
        }
    }

    async fn pre_run(&mut self) -> Result<()> {
        let mut guard = self.resources.lock().await;
        guard.io.open()?;
        Ok(())
    }

    async fn post_run(&mut self) -> Result<()> {
        let mut guard = self.resources.lock().await;
        guard.io.close()?;
        Ok(())
    }

    async fn show_status(&mut self) -> Result<()> {
        let mut guard = self.resources.lock().await;
        guard
            .io
            .msg_system(&format!(
                "Running onui with config: {}\n- LLM: {}\n- cwd: {}",
                self.config
                    .config_path
                    .clone()
                    .map_or("N/A".to_string(), |p| p.display().to_string()),
                self.config.default_llm,
                self.config.workspace_dir().display()
            ))
            .await?;
        Ok(())
    }

    async fn send_llm_stopped(&mut self) -> Result<()> {
        let mut guard = self.resources.lock().await;
        guard.io.llm_stopped().await?;
        Ok(())
    }

    async fn mainloop(&mut self) -> Result<()> {
        let mut input_rx = {
            let mut guard = self.resources.lock().await;
            guard.io.input_channel()
        };
        while let Some(message) = input_rx.recv().await {
            match message {
                UserMsg::Exit => {
                    let mut guard = self.resources.lock().await;
                    guard.io.msg_system("Goodbye.").await?;
                    return Ok(());
                }
                UserMsg::Cancel => {
                    self.llm.cancel().await?;
                    let mut guard = self.resources.lock().await;
                    guard
                        .io
                        .msg_system("Cancelled. 한번 더 누르면 종료됩니다")
                        .await?;
                    guard.io.llm_stopped().await?;
                }
                UserMsg::Input(line) => {
                    match self.handle_command(&line).await? {
                        CommandResult::Exit => return Ok(()),
                        CommandResult::Handled => {
                            let mut guard = self.resources.lock().await;
                            guard.io.llm_stopped().await?;
                            continue;
                        }
                        CommandResult::NotCommand => {}
                    }

                    self.handle_user_input(&line).await?;
                }
            }
        }
        Ok(())
    }

    pub async fn run(&mut self) -> Result<()> {
        self.pre_run().await?;
        self.show_status().await?;
        self.send_llm_stopped().await?;
        self.mainloop().await?;
        self.post_run().await?;
        Ok(())
    }

    async fn handle_command(&mut self, line: &str) -> Result<CommandResult> {
        let trimmed = line.trim();
        if !trimmed.starts_with('/') {
            return Ok(CommandResult::NotCommand);
        }

        let normalized = trimmed.to_ascii_lowercase();
        let mut guard = self.resources.lock().await;
        match normalized.as_str() {
            "/exit" | "/quit" => {
                guard.io.msg_system("Goodbye.").await?;
                return Ok(CommandResult::Exit);
            }
            "/cancel" => {
                drop(guard);
                self.llm.cancel().await?;
                let mut guard = self.resources.lock().await;
                guard
                    .io
                    .msg_system("Cancelled. 한번 더 누르면 종료됩니다")
                    .await?;
                return Ok(CommandResult::Handled);
            }
            "/help" => {
                guard
                    .io
                    .msg_system("Commands: /help, /new, /reset-vm, /cancel, /exit")
                    .await?;
            }
            "/new" => {
                guard.io.msg_system("New conversation started.").await?;
            }
            "/reset-vm" => {
                guard.lua.reset()?;
                guard.io.msg_system("Lua VM reset.").await?;
            }
            _ => {
                guard
                    .io
                    .msg_system(&format!("Unknown command: {}", line))
                    .await?;
            }
        }

        Ok(CommandResult::Handled)
    }

    async fn handle_user_input(&mut self, input: &str) -> Result<()> {
        self.llm.send_user_msg(input).await;
        Ok(())
    }
}

fn render_tool_output(execution: &LuaExecution) -> String {
    let mut items = vec![execution.stdout.trim().to_string()];
    if !execution.returns.is_empty() {
        for (idx, ret) in execution.returns.iter().enumerate() {
            items.push(format!("** Ret[{}]: {}", idx + 1, ret).trim().to_string());
        }
    }
    if let Some(ref error) = execution.error {
        items.push(format!("** Err: {}", error).trim().to_string());
    }
    items.join("\n").trim().to_string()
}
