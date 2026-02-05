use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize, Debug, Default)]
pub struct Config {
    pub config_path: Option<PathBuf>,

    pub default_llm: String,
    pub llm: HashMap<String, LLMConfig>,
}

/// LLM configuration for each provider defined under `[llm.*]`.
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LLMConfig {
    OpenAI(LLMOpenAIConfig),
    // Future LLM providers can be added here.
}

#[derive(Deserialize, Debug)]
pub struct LLMOpenAIConfig {
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub base_url_env: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub system_prompt: Option<String>,
}

impl LLMOpenAIConfig {
    pub fn get_api_key(&self) -> Option<String> {
        if let Some(ref key) = self.api_key {
            Some(key.clone())
        } else if let Some(ref env_var) = self.api_key_env {
            std::env::var(env_var).ok()
        } else {
            None
        }
    }

    pub fn get_base_url(&self) -> Option<String> {
        if let Some(ref url) = self.base_url {
            Some(url.clone())
        } else if let Some(ref env_var) = self.base_url_env {
            std::env::var(env_var).ok()
        } else {
            None
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if !self.llm.contains_key(&self.default_llm) {
            anyhow::bail!(
                "Default LLM '{}' is not defined in the configuration",
                self.default_llm
            );
        }
        Ok(())
    }
}

pub fn load_from_file(path: &Path) -> Result<Config> {
    let content =
        fs::read_to_string(path).with_context(|| format!("read error: {}", path.display()))?;
    let config: Config = toml::from_str(&content)
        .map_err(|err| anyhow::anyhow!("parse error in {}: {}", path.display(), err))?;

    config.validate()?;
    Ok(config)
}

pub fn load_from_file_list(paths: &Vec<PathBuf>) -> Result<Config> {
    for path in paths {
        match load_from_file(path) {
            Ok(mut config) => {
                config.config_path = Some(path.clone());
                return Ok(config);
            }
            Err(e) => {
                println!(
                    "Warning: Could not load config from {}: {}. Trying next path...",
                    path.display(),
                    e
                );
            }
        }
    }
    anyhow::bail!("No valid configuration file found in the provided paths.");
}
