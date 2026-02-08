pub mod openai;
pub mod traits;

pub use openai::OpenAIClient;
pub use traits::{DynLLMClient, LLMClient, LLMEventHandler};

use crate::config::LLMConfig;

pub fn instantiate(
    config: &LLMConfig,
    handler: Box<dyn LLMEventHandler + Send>,
) -> anyhow::Result<DynLLMClient> {
    match config {
        LLMConfig::OpenAI(openai_cfg) => {
            let llm = OpenAIClient::new(&openai_cfg, handler)?;
            Ok(Box::new(llm) as DynLLMClient)
        } // Future LLM providers can be added here.
    }
}
