use crate::config::LLMOpenAIConfig;
use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub lua_code: Option<String>,
    pub lua_timeout_sec: Option<u64>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            lua_code: None,
            lua_timeout_sec: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            lua_code: None,
            lua_timeout_sec: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(
        content: Option<String>,
        lua_code: Option<String>,
        tool_call_id: Option<String>,
    ) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            lua_code,
            lua_timeout_sec: None,
            tool_call_id,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            lua_code: None,
            lua_timeout_sec: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

pub struct LLMLuaClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    reasoning_effort: Option<String>,
}

impl LLMLuaClient {
    pub fn new(config: &LLMOpenAIConfig) -> Result<Self> {
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

        Ok(Self {
            client: Client::new(),
            api_key,
            base_url,
            model,
            reasoning_effort: config.reasoning_effort.clone(),
        })
    }

    pub async fn chat(&self, history: &[ChatMessage]) -> Result<ChatMessage> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let payload =
            OpenAIChatRequest::from_history(history, &self.model, self.reasoning_effort.clone())?;

        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
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

        let mut lua_code = None;
        let mut lua_timeout_sec = None;
        let mut tool_call_id = None;
        if let Some(call) = choice
            .message
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
        {
            if call.function.name == "lua" {
                let args: Value = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| Value::Object(Default::default()));
                if let Some(code) = args.get("code").and_then(|value| value.as_str()) {
                    lua_code = Some(code.to_string());
                    tool_call_id = Some(call.id.clone());
                }
                if let Some(timeout_value) = args.get("timeout_sec") {
                    lua_timeout_sec = parse_timeout(timeout_value);
                }
            }
        }

        let mut message = ChatMessage::assistant(choice.message.content, lua_code, tool_call_id);
        message.lua_timeout_sec = lua_timeout_sec;
        Ok(message)
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
            if let Some(code) = message.lua_code.as_deref() {
                let call_id = message
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(|| "call_lua".to_string());
                tool_calls.push(OpenAIToolCall::lua_call(
                    call_id,
                    code,
                    message.lua_timeout_sec,
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
