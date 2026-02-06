use super::traits::{LLMClient, LLMEventHandler};
use crate::{config::LLMOpenAIConfig, consts::DEFAULT_SYSTEM_PROMPT, llm::traits::Status};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
struct ChatMessage {
    role: String,
    content: Option<String>,
    lua_calls: Vec<LuaCall>,
    tool_call_id: Option<String>,
}

#[derive(Clone, Debug)]
struct LuaCall {
    id: String,
    code: String,
    timeout_sec: Option<u64>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            lua_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            lua_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant(content: Option<String>, lua_calls: Vec<LuaCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            lua_calls,
            tool_call_id: None,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            lua_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    reasoning_effort: Option<String>,
    handler: Box<dyn LLMEventHandler>,
    history: Vec<ChatMessage>,
    in_flight: bool,

    status: Status,
}

impl OpenAIClient {
    pub fn new(config: &LLMOpenAIConfig, handler: Box<dyn LLMEventHandler>) -> Result<Self> {
        let api_key = config
            .get_api_key()
            .ok_or_else(|| anyhow!("OPENAI_API_KEY is not configured"))?;
        let base_url = config
            .get_base_url()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        let model = config
            .model
            .clone()
            .unwrap_or_else(|| "gpt-5-nano".to_string());

        let mut history = Vec::new();
        history.push(ChatMessage::system(DEFAULT_SYSTEM_PROMPT));
        Ok(Self {
            client: Client::new(),
            api_key,
            base_url,
            model,
            reasoning_effort: config.reasoning_effort.clone(),
            handler,
            history,
            in_flight: false,
            status: Status::Idle,
        })
    }

    async fn chat(&self, history: &[ChatMessage]) -> Result<ChatMessage> {
        Self::chat_with(
            &self.client,
            &self.api_key,
            &self.base_url,
            &self.model,
            self.reasoning_effort.clone(),
            history,
        )
        .await
    }

    async fn run_chat_loop(&mut self) -> Result<()> {
        loop {
            if self.history.is_empty() {
                return Ok(());
            }

            let response = self.chat(&self.history).await?;
            self.history.push(response.clone());

            if !response.lua_calls.is_empty() {
                self.status = Status::WaitForLuaResult;
                for call in response.lua_calls.iter() {
                    if let Err(err) = self.handler.on_lua_call(&call.id, &call.code).await {
                        eprintln!("LLM lua handler failed: {err}");
                    }
                }
                break;
            }

            if let Some(content) = response.content.as_deref() {
                if !content.is_empty() {
                    if let Err(err) = self.handler.on_assistant_chunk(content).await {
                        eprintln!("LLM event handler failed: {err}");
                    }
                }
            }

            let _ = self.handler.on_assistant_chunk("").await;

            self.status = Status::Idle;

            break;
        }

        Ok(())
    }

    async fn chat_with(
        client: &Client,
        api_key: &str,
        base_url: &str,
        model: &str,
        reasoning_effort: Option<String>,
        history: &[ChatMessage],
    ) -> Result<ChatMessage> {
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        let payload = OpenAIChatRequest::from_history(history, model, reasoning_effort)?;

        let response = client
            .post(url)
            .bearer_auth(api_key)
            .json(&payload)
            .send()
            .await
            .context("failed to call OpenAI chat completions")?;

        let status = response.status();
        let body_text = response
            .text()
            .await
            .context("failed to read OpenAI response body")?;

        if !status.is_success() {
            return Err(anyhow!(
                "OpenAI chat completions returned error: status={} body={}",
                status,
                body_text
            ));
        }

        let body: OpenAIChatResponse =
            serde_json::from_str(&body_text).context("failed to parse OpenAI response")?;

        let choice = body
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("OpenAI response missing choices"))?;

        let mut lua_calls = Vec::new();
        if let Some(calls) = choice.message.tool_calls.as_ref() {
            for call in calls {
                if call.function.name != "lua" {
                    continue;
                }
                let args: Value = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| Value::Object(Default::default()));
                if let Some(code) = args.get("code").and_then(|value| value.as_str()) {
                    let timeout_sec = args.get("timeout_sec").and_then(parse_timeout);
                    lua_calls.push(LuaCall {
                        id: call.id.clone(),
                        code: code.to_string(),
                        timeout_sec,
                    });
                }
            }
        }

        Ok(ChatMessage::assistant(choice.message.content, lua_calls))
    }
}

#[async_trait(?Send)]
impl LLMClient for OpenAIClient {
    async fn get_status(&self) -> Status {
        self.status.clone()
    }

    async fn send_user_msg(&mut self, message: &str) {
        if self.in_flight {
            eprintln!("LLM request ignored: client is busy");
            return;
        }

        self.history.push(ChatMessage::user(message));
        self.in_flight = true;
        self.status = Status::Generating;
        if let Err(err) = self.run_chat_loop().await {
            eprintln!("LLM processing failed: {err}");
            let _ = self.handler.on_assistant_chunk("").await;
        }
        self.in_flight = false;
        self.status = Status::Idle;
    }

    async fn send_lua_result(&mut self, results: Vec<super::traits::LuaResult>) {
        if self.in_flight {
            eprintln!("LLM request ignored: client is busy");
            return;
        }

        for result in results {
            self.history
                .push(ChatMessage::tool(result.id, result.output));
        }
        self.in_flight = true;
        self.status = Status::Generating;
        if let Err(err) = self.run_chat_loop().await {
            eprintln!("LLM processing failed: {err}");
            let _ = self.handler.on_assistant_chunk("").await;
        }
        self.in_flight = false;
        self.status = Status::Idle;
    }

    async fn is_idle(&self) -> bool {
        matches!(self.status, Status::Idle)
    }

    async fn cancel(&mut self) -> Result<()> {
        self.in_flight = false;
        self.status = Status::Idle;
        Ok(())
    }
}

#[derive(Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    tools: Vec<OpenAITool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

impl OpenAIChatRequest {
    fn from_history(
        history: &[ChatMessage],
        model: &str,
        reasoning_effort: Option<String>,
    ) -> Result<Self> {
        let messages = history
            .iter()
            .map(OpenAIMessage::from_chat)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            model: model.to_string(),
            messages,
            tools: vec![OpenAITool::lua_tool()],
            reasoning_effort,
        })
    }
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OpenAIToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl OpenAIMessage {
    fn from_chat(message: &ChatMessage) -> Result<Self> {
        let mut tool_calls = Vec::new();
        if message.role == "assistant" {
            for call in message.lua_calls.iter() {
                tool_calls.push(OpenAIToolCall::lua_call(
                    call.id.clone(),
                    &call.code,
                    call.timeout_sec,
                )?);
            }
        }

        if message.role == "tool" && message.tool_call_id.is_none() {
            return Err(anyhow!("tool message missing tool_call_id"));
        }

        Ok(Self {
            role: message.role.clone(),
            content: message.content.clone(),
            tool_calls,
            tool_call_id: message.tool_call_id.clone(),
        })
    }
}

#[derive(Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    kind: String,
    function: OpenAIToolFunction,
}

impl OpenAITool {
    fn lua_tool() -> Self {
        Self {
            kind: "function".to_string(),
            function: OpenAIToolFunction {
                name: "lua".to_string(),
                description: "Execute a Lua script.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Lua source code to execute."
                        },
                        "timeout_sec": {
                            "type": "integer",
                            "description": "Timeout in seconds."
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": false
                }),
            },
        }
    }
}

#[derive(Serialize)]
struct OpenAIToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Serialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: OpenAIFunction,
}

impl OpenAIToolCall {
    fn lua_call(id: String, code: &str, timeout_sec: Option<u64>) -> Result<Self> {
        let mut args = json!({ "code": code });
        if let Some(timeout) = timeout_sec {
            args["timeout_sec"] = json!(timeout);
        }
        Ok(Self {
            id,
            kind: "function".to_string(),
            function: OpenAIFunction {
                name: "lua".to_string(),
                arguments: serde_json::to_string(&args)
                    .context("failed to serialize lua arguments")?,
            },
        })
    }
}

fn parse_timeout(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse::<u64>().ok(),
        _ => None,
    }
}

#[derive(Serialize)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
}

#[derive(Deserialize)]
struct OpenAIResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIResponseToolCall>>,
}

#[derive(Deserialize)]
struct OpenAIResponseToolCall {
    id: String,
    #[serde(rename = "type")]
    _kind: String,
    function: OpenAIResponseFunction,
}

#[derive(Deserialize)]
struct OpenAIResponseFunction {
    name: String,
    arguments: String,
}
