use crate::config::Config;
use crate::io::{self, IO, IOChan, Input, Output};
use crate::llm::{LLMClient, LLMEventHandler, Status};
use crate::lua::{self, LuaRuntime};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

enum ApprovalTarget {
    All,
    One(String),
}

struct PendingLua {
    id: String,
    code: String,
    timeout_sec: u64,
    approved: bool,
    output: Option<String>,
}

pub struct AgentResources<R: lua::LuaRuntime> {
    pending_lua: Vec<PendingLua>,
    determined_lua: Vec<PendingLua>,

    lua: R,

    llm_task: Option<JoinHandle<Result<()>>>,
}

impl<R: lua::LuaRuntime> AgentResources<R> {
    pub fn new(lua: R) -> Self {
        Self {
            pending_lua: Vec::new(),
            determined_lua: Vec::new(),
            lua,
            llm_task: None,
        }
    }

    pub fn has_pending_lua(&self) -> bool {
        !self.pending_lua.is_empty()
    }

    pub fn get_lua_targets(&self, target: ApprovalTarget) -> Vec<String> {
        match target {
            ApprovalTarget::All => self.pending_lua.iter().map(|p| p.id.clone()).collect(),
            ApprovalTarget::One(id) => {
                if self.pending_lua.iter().any(|p| p.id == id) {
                    vec![id]
                } else {
                    vec![]
                }
            }
        }
    }

    pub fn pending_lua_index(&self, id: &str) -> Option<usize> {
        self.pending_lua.iter().position(|p| p.id == id)
    }

    pub fn pending_lua_code(&self, id: &str) -> Option<String> {
        self.pending_lua
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.code.clone())
    }

    pub fn determine_lua(&mut self, id: &str, approve: bool, output: String) -> Result<()> {
        let index = self
            .pending_lua_index(id)
            .ok_or_else(|| anyhow!("No pending lua with id {}", id))?;
        let mut pending = self.pending_lua.remove(index);
        pending.approved = approve;
        pending.output = Some(output);
        self.determined_lua.push(pending);
        Ok(())
    }

    pub fn cancel_llm(&mut self) {
        if let Some(handle) = self.llm_task.take() {
            handle.abort();
        }
    }
}

pub struct AgentHandler<R: lua::LuaRuntime> {
    resources: Arc<Mutex<AgentResources<R>>>,
    output_tx: mpsc::Sender<Output>,
}

impl<R: lua::LuaRuntime> AgentHandler<R> {
    pub fn new(resources: Arc<Mutex<AgentResources<R>>>, output_tx: mpsc::Sender<Output>) -> Self {
        Self {
            resources,
            output_tx,
        }
    }
}

#[async_trait(?Send)]
impl<R: lua::LuaRuntime> LLMEventHandler for AgentHandler<R> {
    async fn on_assistant_chunk(&mut self, msg: &str) -> Result<()> {
        send_output(&self.output_tx, Output::AssistantMsg(msg.to_string())).await?;
        Ok(())
    }

    async fn on_lua_call(&mut self, id: &str, code: &str, timeout_sec: Option<u64>) -> Result<()> {
        {
            let mut guard = self.resources.lock().await;
            guard.pending_lua.push(PendingLua {
                id: id.to_string(),
                code: code.to_string(),
                timeout_sec: timeout_sec.unwrap_or(10),
                approved: false,
                output: None,
            });
        }

        send_output(
            &self.output_tx,
            Output::LuaCode {
                id: id.to_string(),
                code: code.to_string(),
            },
        )
        .await?;
        Ok(())
    }

    async fn on_llm_finished(&mut self) -> Result<()> {
        send_output(&self.output_tx, Output::AssistantMsg(String::new())).await?;
        Ok(())
    }
}

enum CommandResult {
    Exit,
    Handled,
    NotCommand,
}

pub struct Agent<R: lua::LuaRuntime, I> {
    io: I,
    config: Config,
    resources: Arc<Mutex<AgentResources<R>>>,

    llm: Arc<Mutex<Box<dyn LLMClient>>>,

    input_rx: mpsc::Receiver<Input>,
    output_tx: mpsc::Sender<Output>,
}

impl<R, I> Agent<R, I>
where
    R: LuaRuntime,
    I: IO,
{
    pub fn new(
        config: &Config,
        llm: Box<dyn LLMClient>,
        resources: Arc<Mutex<AgentResources<R>>>,
        io: I,
        io_chan: IOChan,
    ) -> Self {
        Self {
            resources,
            io,
            config: config.clone(),
            llm: Arc::new(Mutex::new(llm)),
            output_tx: io_chan.output_tx,
            input_rx: io_chan.input_rx,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        self.pre_run().await?;
        self.show_status().await?;
        self.main_loop().await?;
        self.post_run().await?;
        Ok(())
    }

    async fn pre_run(&mut self) -> Result<()> {
        Ok(())
    }

    async fn post_run(&mut self) -> Result<()> {
        self.io.close()?;
        Ok(())
    }

    async fn show_status(&mut self) -> Result<()> {
        send_output(
            &self.output_tx,
            Output::SystemMsg(format!(
                "Running onui with config: {}\n- LLM: {}\n- cwd: {}",
                self.config
                    .config_path
                    .clone()
                    .map_or("N/A".to_string(), |p| p.display().to_string()),
                self.config.default_llm,
                self.config.workspace_dir().display()
            )),
        )
        .await?;
        Ok(())
    }

    async fn main_loop(&mut self) -> Result<()> {
        loop {
            tokio::select! {
                Some(input) = self.input_rx.recv() => {
                    if self.handle_input(input).await? {
                        break
                    }
                }
                else => {
                    break;
                }
            }
        }
        Ok(())
    }

    async fn has_pending_lua(&self) -> Result<bool> {
        let guard = self.resources.lock().await;
        Ok(guard.has_pending_lua())
    }

    async fn handle_input(&mut self, input: Input) -> Result<bool> {
        match input {
            Input::Text(line) => {
                if self.has_pending_lua().await? {
                    self.handle_text_for_lua(&line).await?;
                } else {
                    self.handle_user_input(&line).await?;
                }
                Ok(false)
            }
            Input::Command { cmd, arg, details } => match self.handle_command(cmd, &arg).await? {
                CommandResult::Exit => Ok(true),
                _ => Ok(false),
            },
        }
    }

    async fn handle_text_for_lua(&mut self, line: &str) -> Result<()> {
        let token = line
            .trim()
            .splitn(2, char::is_whitespace)
            .next()
            .unwrap_or("");
        match token.to_ascii_lowercase().as_str() {
            "y" | "yes" | "approve" | "ok" => self.approve_lua(ApprovalTarget::All).await,
            "n" | "no" | "reject" => self.reject_lua(ApprovalTarget::All).await,
            _ => Ok(()),
        }
    }

    async fn approve_lua(&mut self, target: ApprovalTarget) -> Result<()> {
        let targets = {
            let guard = self.resources.lock().await;
            guard.get_lua_targets(target)
        };
        for id in targets {
            let mut guard = self.resources.lock().await;

            let code = guard
                .pending_lua_code(&id)
                .ok_or_else(|| anyhow!("No pending lua with id {}", id))?;
            let output = match guard.lua.execute_script(&code, None) {
                Ok(result) => result.to_string(),
                Err(err) => format!("Lua execution error: {}", err),
            };
            guard.determine_lua(&id, true, output)?;
        }
        Ok(())
    }

    async fn reject_lua(&mut self, target: ApprovalTarget) -> Result<()> {
        let targets = {
            let guard = self.resources.lock().await;
            guard.get_lua_targets(target)
        };
        for id in targets {
            let mut guard = self.resources.lock().await;
            guard.determine_lua(&id, false, "Reject by user.".to_string())?;
        }
        Ok(())
    }

    async fn handle_text_as_user_message(&mut self, line: &str) -> Result<()> {
        self.handle_user_input(line).await?;
        Ok(())
    }

    async fn handle_command(&mut self, cmd: io::Command, arg: &str) -> Result<CommandResult> {
        match cmd {
            io::Command::Exit => {
                send_output(&self.output_tx, Output::SystemMsg("Goodbye.".to_string())).await?;
                return Ok(CommandResult::Exit);
            }
            io::Command::Cancel => {
                let mut guard = self.resources.lock().await;
                guard.cancel_llm();
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("Cancelled. One more time to exit.".to_string()),
                )
                .await?;
                return Ok(CommandResult::Handled);
            }
            io::Command::Help => {
                send_output(
                    &self.output_tx,
                    Output::SystemMsg(
                        "Commands: /help, /new, /reset-vm, /cancel, /exit".to_string(),
                    ),
                )
                .await?;
            }
            io::Command::Approve => {
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("New conversation started.".to_string()),
                )
                .await?;
            }
            _ => {
                let suffix = if arg.is_empty() {
                    "".to_string()
                } else {
                    format!(" {}", arg)
                };
                send_output(
                    &self.output_tx,
                    Output::SystemMsg(format!("Unknown command: /{:?}{}", cmd, suffix)),
                )
                .await?;
            }
        }

        Ok(CommandResult::Handled)
    }

    async fn handle_user_input(&self, input: &str) -> Result<()> {
        // Spawn a task
        let task = {
            let llm: Arc<Mutex<Box<dyn LLMClient + 'static>>> = self.llm.clone();
            let input = input.to_string();
            tokio::task::spawn_local(async move {
                let mut llm = llm.lock().await;
                llm.send_user_msg(&input).await;
                Ok(())
            })
        };
        self.resources.lock().await.llm_task = Some(task);
        Ok(())
    }
}

// Helpers

async fn send_output(output_tx: &mpsc::Sender<Output>, output: Output) -> Result<()> {
    output_tx
        .send(output)
        .await
        .map_err(|err| anyhow!("output channel closed: {}", err))
}
