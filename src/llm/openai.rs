use std::sync::atomic::AtomicBool;

use super::traits::{LLMClient, LLMEventHandler};
use crate::{config::LLMOpenAIConfig, consts::DEFAULT_SYSTEM_PROMPT, llm::traits::Status};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// Structs

// OpenAI Tool Definition

#[derive(Serialize)]
struct OpenAIToolFunction {
    name: String,
    description: String,
    parameters: Value,
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

// Message for history

#[derive(Serialize, Deserialize, Clone)]
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
    fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.to_string()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.to_string()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize, Deserialize, Clone)]
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

// Chat Request

#[derive(Serialize)]
struct OpenAIChatRequest<'a> {
    model: String,
    messages: &'a Vec<OpenAIMessage>,
    tools: Vec<OpenAITool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
}

// Chat Response

#[derive(Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    reasoning_effort: Option<String>,
    handler: Box<dyn LLMEventHandler>,
    history: Vec<OpenAIMessage>,

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
        history.push(OpenAIMessage::system(DEFAULT_SYSTEM_PROMPT));
        Ok(Self {
            client: Client::new(),
            api_key,
            base_url,
            model,
            reasoning_effort: config.reasoning_effort.clone(),
            handler,
            history,
            status: Status::Idle,
        })
    }

    async fn chat_completion(&mut self, history: &Vec<OpenAIMessage>) -> Result<OpenAIMessage> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let payload = OpenAIChatRequest {
            model: self.model.to_string(),
            messages: history,
            tools: vec![OpenAITool::lua_tool()],
            reasoning_effort: self.reasoning_effort.clone(),
        };

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

        let mut lua_calls = Vec::new();
        for call in &choice.message.tool_calls {
            if call.function.name == "lua" {
                lua_calls.push(call.clone());
            }
            let args: Value = serde_json::from_str(&call.function.arguments)
                .unwrap_or_else(|_| Value::Object(Default::default()));
            if let Some(code) = args.get("code").and_then(|value| value.as_str()) {
                let timeout_sec = args.get("timeout_sec").and_then(parse_timeout);
                self.handler
                    .on_lua_call(&call.id, code, timeout_sec)
                    .await?;
            }
        }

        Ok(choice.message)
    }

    async fn chat(&mut self, new_messages: &[OpenAIMessage]) -> Result<OpenAIMessage> {
        let mut new_history = self.history.clone();
        for msg in new_messages {
            new_history.push(msg.clone());
        }
        let response_msg = self.chat_completion(&new_history).await?;
        // Update history with new messages and response.
        for msg in new_messages {
            self.history.push(msg.clone());
        }
        self.history.push(response_msg.clone());
        Ok(response_msg)
    }
}

#[async_trait(?Send)]
impl LLMClient for OpenAIClient {
    async fn get_status(&self) -> Status {
        self.status.clone()
    }

    async fn send_user_msg(&mut self, message: &str) -> Result<()> {
        let new_msgs = vec![OpenAIMessage::user(message)];
        self.chat(&new_msgs).await?;
        Ok(())
    }

    async fn send_lua_results(&mut self, results: &[(String, String)]) -> Result<()> {
        let mut new_msgs = Vec::new();
        for (id, output) in results {
            let content = format!("Lua execution result:\n{}", output);
            let msg = OpenAIMessage {
                role: "tool".to_string(),
                content: Some(content),
                tool_calls: Vec::new(),
                tool_call_id: Some(id.clone()),
            };
            new_msgs.push(msg);
        }
        self.chat(&new_msgs).await?;
        Ok(())
    }
}
