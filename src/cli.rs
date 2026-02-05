use crate::{agent::Agent, config::Config};
use anyhow::{Context, Result};
use clap::Parser;
use std::{
    env,
    io::{self, Write},
    path::PathBuf,
};

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

/// Drives the REPL/command loop.
pub struct Cli {
    args: CliArgs,
    config: Config,
}

impl Cli {
    pub fn new(args: CliArgs, config: Config) -> Self {
        Self { args, config }
    }

    pub async fn run(&mut self, agent: &mut Agent) -> Result<()> {
        self.print_system(&format!(
            "Running onui with config: {} (LLM: {}, cwd: {})",
            self.config
                .config_path
                .clone()
                .map_or("N/A".to_string(), |p| p.display().to_string()),
            self.config.default_llm,
            self.workspace_dir().display()
        ));

        loop {
            let input = self.read_input()?;
            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            if input.starts_with('/') {
                if self.handle_command(input, agent)? {
                    break;
                }
                continue;
            }

            let response = agent.handle_user_input(input).await?;
            if let Some(content) = response.content.as_deref() {
                if !content.is_empty() {
                    self.print_ai(content);
                }
            }
        }

        Ok(())
    }

    fn read_input(&self) -> Result<String> {
        if !self.args.pipe {
            print!("> ");
            io::stdout().flush()?;
        }

        let mut buffer = String::new();
        io::stdin()
            .read_line(&mut buffer)
            .context("failed to read stdin")?;
        Ok(buffer)
    }

    fn handle_command(&self, line: &str, agent: &mut Agent) -> Result<bool> {
        let normalized = line.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "/exit" | "/quit" => {
                self.print_system("Goodbye.");
                return Ok(true);
            }
            "/help" => {
                self.print_system("Commands: /help, /new, /reset-vm, /exit");
            }
            "/new" => {
                agent.new_session();
                self.print_system("New conversation started.");
            }
            "/reset-vm" => {
                agent.reset_vm();
                self.print_system("Lua VM reset.");
            }
            _ => {
                self.print_system(&format!("Unknown command: {}", line));
            }
        }

        Ok(false)
    }

    fn workspace_dir(&self) -> PathBuf {
        self.args
            .path
            .clone()
            .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }

    fn print_system(&self, message: &str) {
        if message.is_empty() {
            println!("* ");
            return;
        }

        for line in message.lines() {
            println!("* {}", line);
        }
    }

    fn print_ai(&self, message: &str) {
        if message.is_empty() {
            println!("  ");
            return;
        }

        for line in message.lines() {
            println!("  {}", line);
        }
    }
}
