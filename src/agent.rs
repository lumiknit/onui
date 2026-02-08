use crate::config::Config;
use crate::io::{self, IO, IOChan, Input, Output};
use crate::llm::{BoxedLLMClient, LLMClient, LLMEventHandler, Status};
use crate::lua::{self, LuaRuntime};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

enum ApprovalTarget {
    All,
}

struct PendingLua {
    id: String,
    code: String,
    timeout_sec: u64,
    approved: bool,
    output: Option<String>,
}

pub struct AgentResources {
    pending_lua: Vec<PendingLua>,
    determined_lua: Vec<PendingLua>,
}

impl AgentResources {
    pub fn new() -> Self {
        Self {
            pending_lua: Vec::new(),
            determined_lua: Vec::new(),
        }
    }

    pub fn has_pending_lua(&self) -> bool {
        !self.pending_lua.is_empty()
    }

    fn get_lua_targets(&self, target: ApprovalTarget) -> Vec<String> {
        match target {
            ApprovalTarget::All => self.pending_lua.iter().map(|p| p.id.clone()).collect(),
        }
    }

    pub fn pending_lua_index(&self, id: &str) -> Option<usize> {
        self.pending_lua.iter().position(|p| p.id == id)
    }

    fn pending_lua_job(&self, id: &str) -> Option<(String, u64)> {
        self.pending_lua
            .iter()
            .find(|p| p.id == id)
            .map(|p| (p.code.clone(), p.timeout_sec))
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
}

pub struct AgentHandler {
    resources: Arc<Mutex<AgentResources>>,
    output_tx: mpsc::Sender<Output>,
}

impl AgentHandler {
    pub fn new(resources: Arc<Mutex<AgentResources>>, output_tx: mpsc::Sender<Output>) -> Self {
        Self {
            resources,
            output_tx,
        }
    }
}

#[async_trait(?Send)]
impl LLMEventHandler for AgentHandler {
    async fn on_assistant_chunk(&self, msg: &str) -> Result<()> {
        send_output(&self.output_tx, Output::AssistantMsg(msg.to_string())).await?;
        Ok(())
    }

    async fn on_lua_call(&self, id: &str, code: &str, timeout_sec: Option<u64>) -> Result<()> {
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

    async fn on_llm_finished(&self) -> Result<()> {
        self.output_tx.send(Output::InputReady).await?;
        Ok(())
    }
}

enum CommandResult {
    Exit,
    Handled,
}

pub struct Agent<R: lua::LuaRuntime, I> {
    running: bool,

    io: I,
    config: Config,
    resources: Arc<Mutex<AgentResources>>,

    llm: Arc<Mutex<BoxedLLMClient>>,

    lua: R,

    input_rx: mpsc::Receiver<Input>,
    signal_rx: mpsc::Receiver<io::Signal>,
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
        lua: R,
        resources: Arc<Mutex<AgentResources>>,
        io: I,
        io_chan: IOChan,
    ) -> Self {
        Self {
            running: false,
            resources,
            io,
            config: config.clone(),
            llm: Arc::new(Mutex::new(llm)),
            lua: lua,
            output_tx: io_chan.output_tx,
            input_rx: io_chan.input_rx,
            signal_rx: io_chan.signal_rx,
        }
    }

    pub async fn run(&mut self) -> Result<()> {
        self.show_status().await?;
        self.pre_run().await?;
        self.main_loop().await?;
        self.post_run().await?;
        Ok(())
    }

    async fn pre_run(&mut self) -> Result<()> {
        self.running = true;
        self.output_tx.send(Output::InputReady).await?;
        Ok(())
    }

    async fn post_run(&mut self) -> Result<()> {
        self.output_tx
            .send(Output::SystemMsg("Agent stopped.".to_string()))
            .await?;
        self.io.close()?;
        Ok(())
    }

    async fn show_status(&mut self) -> Result<()> {
        let llm_status = {
            let llm = self.llm.lock().await;
            llm.get_status()
        };
        let llm_status_text = match llm_status {
            Status::Idle => "idle",
            Status::WaitForLuaResult => "waiting for lua",
            Status::Generating => "generating",
        };
        let pending_lua = {
            let guard = self.resources.lock().await;
            guard.pending_lua.len()
        };
        let llm_model = {
            let llm = self.llm.lock().await;
            llm.get_model_name()
        };
        let (token_used, token_limit) = {
            let llm = self.llm.lock().await;
            llm.context_size()
        };
        let msg = format!(
            "[onui Status]\n\
            - LLM: {}\n\
            - Model: {}\n\
            - LLM Status: {}\n\
            - Token Usage: {}/{}\n\
            - cwd: {}\n\
            - Pending Lua scripts: {}",
            self.config.default_llm,
            llm_model,
            llm_status_text,
            token_used,
            token_limit,
            self.config.workspace_dir().display(),
            pending_lua
        );
        send_output(&self.output_tx, Output::SystemMsg(msg)).await?;
        Ok(())
    }

    async fn main_loop(&mut self) -> Result<()> {
        while self.running {
            tokio::select! {
                Some(signal) = self.signal_rx.recv() => {
                    send_output(&self.output_tx, Output::SystemMsg(format!("Received signal '{:?}'", signal))).await?;
                    if signal == io::Signal::Exit {
                        self.running = false;
                        break;
                    }
                }
                Some(input) = self.input_rx.recv() => {
                    self.handle_input(input).await?;
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
            Input::Command { cmd, arg, .. } => match self.handle_command(cmd, &arg).await? {
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
            let output = {
                let (code, timeout_sec) = guard
                    .pending_lua_job(&id)
                    .ok_or_else(|| anyhow!("No pending lua with id {}", id))?;
                let result = match self.lua.execute_script(&code, Some(timeout_sec)) {
                    Ok(exec) => exec.to_string(),
                    Err(err) => format!("Lua execution error: {}", err),
                };
                guard.determine_lua(&id, true, result.clone())?;
                result
            };
            send_output(
                &self.output_tx,
                Output::LuaResult {
                    id: id.clone(),
                    output: output.clone(),
                },
            )
            .await?;
        }
        // Check and send
        {
            let guard = self.resources.lock().await;
            if !guard.has_pending_lua() {
                let results = guard
                    .determined_lua
                    .iter()
                    .filter_map(|p| {
                        if p.approved {
                            Some((p.id.clone(), p.output.clone().unwrap_or_default()))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<(String, String)>>();
                let mut llm = self.llm.lock().await;
                llm.send_lua_results(&results).await?;
            }
        }
        Ok(())
    }

    async fn reject_lua(&mut self, target: ApprovalTarget) -> Result<()> {
        let targets = {
            let guard = self.resources.lock().await;
            guard.get_lua_targets(target)
        };
        let mut results = Vec::new();
        for id in targets {
            let mut guard = self.resources.lock().await;
            let output = "Reject by user.".to_string();
            guard.determine_lua(&id, false, output.clone())?;
            results.push((id.clone(), output));
        }
        if !results.is_empty() {
            let mut llm = self.llm.lock().await;
            llm.send_lua_results(&results).await?;
        }
        Ok(())
    }

    async fn handle_command(&mut self, cmd: io::Command, _arg: &str) -> Result<CommandResult> {
        match cmd {
            io::Command::Exit => {
                send_output(&self.output_tx, Output::SystemMsg("Goodbye.".to_string())).await?;
                return Ok(CommandResult::Exit);
            }
            io::Command::Cancel => {
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
                        "Commands: /help, /status, /reset-vm, /cancel, /exit, /approve, /reject"
                            .to_string(),
                    ),
                )
                .await?;
            }
            io::Command::Status => {
                self.show_status().await?;
            }
            io::Command::ResetVM => {
                {
                    let mut guard = self.resources.lock().await;
                    self.lua.reset()?;
                    guard.pending_lua.clear();
                    guard.determined_lua.clear();
                }
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("Lua VM reset.".to_string()),
                )
                .await?;
            }
            io::Command::Approve => {
                if self.has_pending_lua().await? {
                    self.approve_lua(ApprovalTarget::All).await?;
                    send_output(
                        &self.output_tx,
                        Output::SystemMsg("Approved pending Lua scripts.".to_string()),
                    )
                    .await?;
                } else {
                    send_output(
                        &self.output_tx,
                        Output::SystemMsg("No pending Lua scripts to approve.".to_string()),
                    )
                    .await?;
                }
            }
            io::Command::Reject => {
                if self.has_pending_lua().await? {
                    self.reject_lua(ApprovalTarget::All).await?;
                    send_output(
                        &self.output_tx,
                        Output::SystemMsg("Rejected pending Lua scripts.".to_string()),
                    )
                    .await?;
                } else {
                    send_output(
                        &self.output_tx,
                        Output::SystemMsg("No pending Lua scripts to reject.".to_string()),
                    )
                    .await?;
                }
            }
            io::Command::ApproveAlways => {
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("Always-approve mode is not supported yet.".to_string()),
                )
                .await?;
            }
            io::Command::Compact => {
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("Conversation compaction is not implemented.".to_string()),
                )
                .await?;
            }
        }

        Ok(CommandResult::Handled)
    }

    async fn handle_user_input(&mut self, input: &str) -> Result<()> {
        let llm: Arc<Mutex<BoxedLLMClient>> = self.llm.clone();
        let input = input.to_string();
        let output_tx = self.output_tx.clone();
        let mut llm = llm.lock().await;

        tokio::select! {
            _t = self.signal_rx.recv() => {
                Ok(())
            }
            send_result = llm.send_user_msg(&input) => {
                send_result.map_err(|err| {
                    let _ = output_tx
                        .try_send(Output::SystemMsg(format!(
                            "Failed to send message to LLM: {}",
                            err
                        )));
                    err
                })
            }
        }
    }
}

// Helpers

async fn send_output(output_tx: &mpsc::Sender<Output>, output: Output) -> Result<()> {
    output_tx
        .send(output)
        .await
        .map_err(|err| anyhow!("output channel closed: {}", err))
}
