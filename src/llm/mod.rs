pub mod lua_call;
pub mod openai;
pub mod traits;

pub use openai::OpenAIClient;
pub use traits::{LLMClient, LLMEventHandler, LuaResult, Status};

use crate::config::LLMConfig;

pub fn instantiate(
    config: &LLMConfig,
    handler: Box<dyn LLMEventHandler>,
) -> anyhow::Result<Box<dyn LLMClient>> {
    match config {
        LLMConfig::OpenAI(openai_cfg) => {
            let llm = OpenAIClient::new(&openai_cfg, handler)?;
            Ok(Box::new(llm) as Box<dyn LLMClient>)
        } // Future LLM providers can be added here.
    }
}
