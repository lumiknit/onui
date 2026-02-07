pub mod openai;
pub mod traits;

pub use openai::OpenAIClient;
pub use traits::{BoxedLLMClient, LLMClient, LLMEventHandler, Status};

use crate::config::LLMConfig;

pub fn instantiate(
    config: &LLMConfig,
    handler: Box<dyn LLMEventHandler + Send>,
) -> anyhow::Result<BoxedLLMClient> {
    match config {
        LLMConfig::OpenAI(openai_cfg) => {
            let llm = OpenAIClient::new(&openai_cfg, handler)?;
            Ok(Box::new(llm) as BoxedLLMClient)
        } // Future LLM providers can be added here.
    }
}
