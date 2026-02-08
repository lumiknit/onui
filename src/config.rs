use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

#[derive(Clone, Deserialize, Debug, Default)]
pub struct Config {
    pub config_path: Option<PathBuf>,

    #[serde(skip)]
    pub path: Option<PathBuf>,

    pub default_llm: String,
    pub llm: HashMap<String, LLMConfig>,
}

/// Parses command line options for `onui`.
#[derive(Parser, Debug)]
#[command(author, version, about = "Minimal onui runner")]
pub struct CliArgs {
    /// Path to the configuration file.
    #[arg(short = 'c', long)]
    pub config: Option<PathBuf>,

    /// Run in pipe-friendly mode instead of interactive.
    #[arg(long)]
    pub pipe: bool,

    /// Base directory to run from.
    pub path: Option<PathBuf>,
}

impl CliArgs {
    /// Returns the config file path if provided.
    /// Paths are:
    /// - Command line argument (1st)
    /// - Environment variable `ONUI_CONFIG` (2nd)
    /// - Current path: `./.onui/config.toml` (3rd)
    /// - Home directory: `~/.onui/config.toml` (4th)
    pub fn config_path(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = Vec::new();

        if let Some(ref path) = self.config {
            paths.push(path.to_path_buf());
        }

        if let Ok(env_path) = env::var("ONUI_CONFIG") {
            paths.push(PathBuf::from(env_path));
        }

        paths.push(PathBuf::from(".").join(".onui").join("config.toml"));

        if let Some(home_dir) = env::home_dir() {
            let home_config = home_dir.join(".onui").join("config.toml");
            paths.push(home_config);
        }

        paths
    }
}

/// LLM configuration for each provider defined under `[llm.*]`.
#[derive(Clone, Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LLMConfig {
    OpenAI(LLMOpenAIConfig),
    // Future LLM providers can be added here.
}

#[derive(Clone, Deserialize, Debug)]
pub struct LLMOpenAIConfig {
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub base_url_env: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub system_prompt: Option<String>,
    pub stream: Option<bool>, // Default is true
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

    pub fn workspace_dir(&self) -> PathBuf {
        self.path
            .clone()
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
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

pub fn load_from_cli() -> Result<Config> {
    let args = CliArgs::parse();
    let mut config = load_from_file_list(&args.config_path())?;
    config.path = args.path;
    Ok(config)
}
