use crate::config::Config;
use crate::io::{IO, IOChan, Input, Output, Signal};
use crate::llm::{LLMClient, LLMEventHandler, LuaResult, Status};
use crate::lua::{LuaExecution, LuaRuntime};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

struct PendingLua {
    id: String,
    code: String,
    approved: Option<bool>,
}

pub struct AgentResources<R> {
    lua: R,
    lua_wait_list: HashMap<String, PendingLua>,
}

impl<R> AgentResources<R> {
    pub fn new(lua: R) -> Self {
        Self {
            lua,
            lua_wait_list: HashMap::new(),
        }
    }
}

pub struct AgentHandler<R> {
    resources: Arc<Mutex<AgentResources<R>>>,
    output_tx: mpsc::Sender<Output>,
}

impl<R> AgentHandler<R> {
    pub fn new(resources: Arc<Mutex<AgentResources<R>>>, output_tx: mpsc::Sender<Output>) -> Self {
        Self {
            resources,
            output_tx,
        }
    }
}

enum CommandResult {
    Exit,
    Handled,
    NotCommand,
}

enum ApprovalTarget {
    All,
    One(String),
}

pub struct Agent<R, I> {
    llm: Box<dyn LLMClient>,
    resources: Arc<Mutex<AgentResources<R>>>,
    io: I,
    config: Config,
    output_tx: mpsc::Sender<Output>,
    input_rx: Option<mpsc::Receiver<Input>>,
    signal_rx: Option<mpsc::Receiver<Signal>>,
}

#[async_trait(?Send)]
impl<R> LLMEventHandler for AgentHandler<R>
where
    R: LuaRuntime,
{
    async fn on_assistant_chunk(&mut self, msg: &str) -> Result<()> {
        send_output(&self.output_tx, Output::AssistantMsg(msg.to_string())).await?;
        Ok(())
    }

    async fn on_lua_call(&mut self, id: &str, code: &str) -> Result<()> {
        {
            let mut guard = self.resources.lock().await;
            guard
                .lua_wait_list
                .entry(id.to_string())
                .or_insert(PendingLua {
                    id: id.to_string(),
                    code: code.to_string(),
                    approved: None,
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
            llm,
            resources,
            io,
            config: config.clone(),
            output_tx: io_chan.output_tx,
            input_rx: Some(io_chan.input_rx),
            signal_rx: Some(io_chan.signal_rx),
        }
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

    async fn send_llm_stopped(&mut self) -> Result<()> {
        send_output(&self.output_tx, Output::AssistantMsg(String::new())).await?;
        Ok(())
    }

    async fn main_loop(&mut self) -> Result<()> {
        let mut input_rx = self.input_rx.take().expect("input channel not initialized");
        let mut signal_rx = self
            .signal_rx
            .take()
            .expect("signal channel not initialized");

        loop {
            tokio::select! {
                Some(signal) = signal_rx.recv() => {
                    if self.handle_signal(signal).await? {
                        break;
                    }
                }
                Some(input) = input_rx.recv() => {
                    if self.handle_input(input).await? {
                        break;
                    }
                }
                else => {
                    break;
                }
            }
        }
        Ok(())
    }

    pub async fn run(&mut self) -> Result<()> {
        self.pre_run().await?;
        self.show_status().await?;
        self.send_llm_stopped().await?;
        self.main_loop().await?;
        self.post_run().await?;
        Ok(())
    }

    async fn handle_signal(&mut self, signal: Signal) -> Result<bool> {
        match signal {
            Signal::Exit => {
                self.reject_all_pending().await?;
                send_output(&self.output_tx, Output::SystemMsg("Goodbye.".to_string())).await?;
                return Ok(true);
            }
            Signal::Cancel => {
                self.llm.cancel().await?;
                self.reject_all_pending().await?;
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("Cancelled. One more time to exit.".to_string()),
                )
                .await?;
                self.send_llm_stopped().await?;
            }
        }
        Ok(false)
    }

    async fn handle_input(&mut self, input: Input) -> Result<bool> {
        match input {
            Input::Command { name, rest } => {
                let normalized = name.to_ascii_lowercase();
                match self.handle_command(&normalized, &rest).await? {
                    CommandResult::Exit => return Ok(true),
                    CommandResult::Handled => {
                        self.send_llm_stopped().await?;
                    }
                    CommandResult::NotCommand => {}
                }
            }
            Input::Text(line) => {
                if self.handle_lua_input(&line).await? {
                    return Ok(false);
                }

                if matches!(self.llm.get_status().await, Status::Generating) {
                    return Ok(false);
                }

                if self.has_pending_lua().await {
                    self.handle_lua_input(&line).await?;
                    return Ok(false);
                }

                if matches!(self.llm.get_status().await, Status::Idle) {
                    self.handle_user_input(&line).await?;
                }
            }
        }

        Ok(false)
    }

    async fn handle_command(&mut self, name: &str, rest: &str) -> Result<CommandResult> {
        match name {
            "exit" | "quit" => {
                send_output(&self.output_tx, Output::SystemMsg("Goodbye.".to_string())).await?;
                return Ok(CommandResult::Exit);
            }
            "cancel" => {
                self.llm.cancel().await?;
                self.reject_all_pending().await?;
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("Cancelled. One more time to exit.".to_string()),
                )
                .await?;
                return Ok(CommandResult::Handled);
            }
            "help" => {
                send_output(
                    &self.output_tx,
                    Output::SystemMsg(
                        "Commands: /help, /new, /reset-vm, /cancel, /exit".to_string(),
                    ),
                )
                .await?;
            }
            "new" => {
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("New conversation started.".to_string()),
                )
                .await?;
            }
            "reset-vm" => {
                let mut guard = self.resources.lock().await;
                guard.lua.reset()?;
                send_output(
                    &self.output_tx,
                    Output::SystemMsg("Lua VM reset.".to_string()),
                )
                .await?;
            }
            _ => {
                let suffix = if rest.is_empty() {
                    "".to_string()
                } else {
                    format!(" {}", rest)
                };
                send_output(
                    &self.output_tx,
                    Output::SystemMsg(format!("Unknown command: /{}{}", name, suffix)),
                )
                .await?;
            }
        }

        Ok(CommandResult::Handled)
    }

    async fn handle_user_input(&mut self, input: &str) -> Result<()> {
        self.llm.send_user_msg(input).await;
        Ok(())
    }

    async fn handle_lua_input(&mut self, input: &str) -> Result<bool> {
        let (decision, target) = match self.parse_lua_approval(input).await? {
            Some(result) => result,
            None => return Ok(false),
        };

        self.apply_lua_approval(decision, target).await?;
        Ok(true)
    }

    async fn parse_lua_approval(&mut self, input: &str) -> Result<Option<(bool, ApprovalTarget)>> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        let ids = self.pending_lua_ids().await;
        if ids.is_empty() {
            return Ok(None);
        }

        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.is_empty() {
            return Ok(None);
        }

        if tokens.len() == 1 {
            if let Some(decision) = parse_decision(tokens[0]) {
                return Ok(Some((decision, ApprovalTarget::All)));
            }
            if ids.iter().any(|id| id == tokens[0]) {
                return Ok(Some((true, ApprovalTarget::One(tokens[0].to_string()))));
            }
        } else if tokens.len() >= 2 {
            if let Some(decision) = parse_decision(tokens[0]) {
                if tokens[1] == "all" {
                    return Ok(Some((decision, ApprovalTarget::All)));
                }
                if ids.iter().any(|id| id == tokens[1]) {
                    return Ok(Some((decision, ApprovalTarget::One(tokens[1].to_string()))));
                }
            }
            if let Some(decision) = parse_decision(tokens[1]) {
                if tokens[0] == "all" {
                    return Ok(Some((decision, ApprovalTarget::All)));
                }
                if ids.iter().any(|id| id == tokens[0]) {
                    return Ok(Some((decision, ApprovalTarget::One(tokens[0].to_string()))));
                }
            }
        }

        send_output(
            &self.output_tx,
            Output::SystemMsg("Lua approval input not recognized.".to_string()),
        )
        .await?;
        Ok(None)
    }

    async fn apply_lua_approval(&mut self, decision: bool, target: ApprovalTarget) -> Result<()> {
        let mut pending = Vec::new();
        let mut guard = self.resources.lock().await;

        match target {
            ApprovalTarget::All => {
                for item in guard.lua_wait_list.values_mut() {
                    item.approved = Some(decision);
                }
            }
            ApprovalTarget::One(id) => {
                if let Some(item) = guard.lua_wait_list.get_mut(&id) {
                    item.approved = Some(decision);
                } else {
                    return Ok(());
                }
            }
        }

        let all_decided = !guard.lua_wait_list.is_empty()
            && guard
                .lua_wait_list
                .values()
                .all(|item| item.approved.is_some());

        if all_decided {
            pending = guard.lua_wait_list.drain().map(|(_, item)| item).collect();
        }
        drop(guard);

        if !pending.is_empty() {
            self.finish_lua_batch(pending).await?;
        }

        Ok(())
    }

    async fn finish_lua_batch(&mut self, pending: Vec<PendingLua>) -> Result<()> {
        let mut outputs = Vec::new();
        let mut results = Vec::new();

        {
            let guard = self.resources.lock().await;
            for item in pending {
                let (approved, output) = if item.approved == Some(true) {
                    let execution = guard
                        .lua
                        .execute_script(&item.code, None)
                        .context("lua script execution failed")?;
                    (true, render_tool_output(&execution))
                } else {
                    (false, "Lua execution rejected by user.".to_string())
                };

                outputs.push((item.id.clone(), output.clone()));
                results.push(LuaResult {
                    id: item.id,
                    approved,
                    output,
                });
            }
        }

        for (id, output) in outputs {
            send_output(&self.output_tx, Output::LuaResult { id, output }).await?;
        }

        self.llm.send_lua_result(results).await;
        Ok(())
    }

    async fn reject_all_pending(&mut self) -> Result<()> {
        let pending = {
            let mut guard = self.resources.lock().await;
            if guard.lua_wait_list.is_empty() {
                return Ok(());
            }
            guard.lua_wait_list.values_mut().for_each(|item| {
                item.approved = Some(false);
            });
            guard
                .lua_wait_list
                .drain()
                .map(|(_, item)| item)
                .collect::<Vec<_>>()
        };

        self.finish_lua_batch(pending).await?;
        Ok(())
    }

    async fn has_pending_lua(&self) -> bool {
        let guard = self.resources.lock().await;
        !guard.lua_wait_list.is_empty()
    }

    async fn pending_lua_ids(&self) -> Vec<String> {
        let guard = self.resources.lock().await;
        guard.lua_wait_list.keys().cloned().collect()
    }
}

fn parse_decision(token: &str) -> Option<bool> {
    match token.to_ascii_lowercase().as_str() {
        "y" | "yes" | "approve" | "ok" => Some(true),
        "n" | "no" | "reject" => Some(false),
        _ => None,
    }
}

async fn send_output(output_tx: &mpsc::Sender<Output>, output: Output) -> Result<()> {
    output_tx
        .send(output)
        .await
        .map_err(|err| anyhow!("output channel closed: {}", err))
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
