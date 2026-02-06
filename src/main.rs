mod agent;
mod cli;
mod config;
mod consts;
mod io;
mod llm_openai;
mod lua;

use agent::Agent;
use anyhow::Context;
use clap::Parser;
use cli::{Cli, CliArgs};
use consts::DEFAULT_SYSTEM_PROMPT;
use llm_openai::LLMLuaClient;
use lua::LuaVM;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();
    let config =
        config::load_from_file_list(&args.config_path()).context("loading configuration")?;

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
    let mut agent = Agent::new(llm, lua, Some(prompt));
    let mut cli = Cli::new(args, config);

    cli.run(&mut agent).await?;

    Ok(())
}
