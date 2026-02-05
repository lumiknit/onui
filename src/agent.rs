use crate::llm_openai::{ChatMessage, LLMLuaClient};
use crate::lua::LuaVM;
use anyhow::{Context, Result};

pub struct Agent {
    llm: LLMLuaClient,
    lua: LuaVM,
    history: Vec<ChatMessage>,
}

impl Agent {
    pub fn new(llm: LLMLuaClient, lua: LuaVM, system_prompt: Option<String>) -> Self {
        let mut history = Vec::new();
        if let Some(prompt) = system_prompt {
            history.push(ChatMessage::system(prompt));
        }

        Self { llm, lua, history }
    }

    pub fn reset_vm(&mut self) {
        self.lua = LuaVM::new().expect("Failed to create new LuaVM");
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

    pub async fn handle_user_input(&mut self, input: &str) -> Result<ChatMessage> {
        self.history.push(ChatMessage::user(input));

        loop {
            let response = self.llm.chat(&self.history).await?;
            let lua_code = response.lua_code.clone();
            let tool_call_id = response.tool_call_id.clone();
            self.history.push(response.clone());

            if let Some(code) = lua_code {
                self.print_system(&format!("lua 실행:\n{}", code));
                let execution = self
                    .lua
                    .execute_script(&code, response.lua_timeout_sec)
                    .context("lua script execution failed")?;
                let tool_output =
                    self.render_tool_output(&execution.stdout, execution.error, &execution.returns);
                self.print_system(&format!("lua 결과:\n{}", tool_output));

                let tool_call_id = tool_call_id.unwrap_or_else(|| "call_lua".to_string());
                self.history
                    .push(ChatMessage::tool(tool_call_id, tool_output));
                continue;
            }

            return Ok(response);
        }
    }

    fn print_system(&self, message: &str) {
        if message.is_empty() {
            println!("* ");
            return;
        }

        for line in message.lines() {
            println!("* {}", line);
        }
    }

    fn render_tool_output(
        &self,
        stdout: &str,
        error: Option<String>,
        returns: &[String],
    ) -> String {
        let mut output = stdout.to_string();
        if !returns.is_empty() {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(&returns.join("\t"));
            output.push('\n');
        }
        if let Some(error) = error {
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(&error);
        }
        output
    }
}
