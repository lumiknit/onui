mod agent;
mod config;
mod consts;
mod io;
mod llm;
mod lua;

use agent::Agent;
use anyhow::Context;
use consts::DEFAULT_SYSTEM_PROMPT;
use io::cli::CliIO;
use llm::LLMLuaClient;
use lua::LuaVM;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = config::load_from_cli().context("loading configuration")?;

    let llm_config = config.llm.get(&config.default_llm).unwrap_or_else(|| {
        panic!(
            "Default LLM '{}' not found in configuration",
            config.default_llm
        )
    });

    let (llm, system_prompt) = match llm_config {
        config::LLMConfig::OpenAI(openai_cfg) => (
            LLMLuaClient::new(openai_cfg), // Future LLM providers can have their validation here.
            openai_cfg.system_prompt.clone(),
        ),
    };

    let llm = llm.context("building LLM client")?;

    let lua = LuaVM::new().context("creating Lua VM")?;
    let prompt = system_prompt.unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
    let io = CliIO::new(config.pipe);
    let mut agent = Agent::new(llm, lua, io, Some(prompt));

    agent.run(&config).await?;

    Ok(())
}
