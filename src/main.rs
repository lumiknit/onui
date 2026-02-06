mod agent;
mod config;
mod consts;
mod io;
mod llm;
mod lua;

use agent::{Agent, AgentHandler, AgentResources};
use anyhow::Context;
use io::cli::CliIO;
use lua::LuaVM;
use std::{process::exit, sync::Arc};
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = config::load_from_cli().context("loading configuration")?;

    let llm_config = config.llm.get(&config.default_llm).unwrap_or_else(|| {
        panic!(
            "Default LLM '{}' not found in configuration",
            config.default_llm
        )
    });

    let lua = LuaVM::new().context("creating Lua VM")?;
    let io = CliIO::new();
    let resources = Arc::new(Mutex::new(AgentResources::new(lua, io)));
    let handler = Box::new(AgentHandler::new(resources.clone()));

    let llm = llm::instantiate(&llm_config, handler).context("instantiating LLM client")?;
    let mut agent = Agent::new(&config, llm, resources);

    agent.run().await?;

    // Exit, even all tasks are not finished yet.
    println!("Exiting...");
    exit(0);
}
