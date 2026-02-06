use std::sync::Arc;
use tokio::io::{self, AsyncBufReadExt};
use tokio::sync::Mutex;

/// CliIO is an implementation of IO, which is for command line interface.
pub struct CliIO {
    input_history: Arc<Mutex<Vec<String>>>,
}

use anyhow::Result;
use tokio::sync::mpsc;

impl CliIO {
    /// Create a new CliIO instance.
    pub fn new() -> Self {
        CliIO {
            input_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a snapshot of user inputs collected so far.
    pub async fn input_history(&self) -> Vec<String> {
        self.input_history.lock().await.clone()
    }
}

impl super::IO for CliIO {
    fn input_channel(&mut self) -> mpsc::Receiver<super::UserMsg> {
        let (tx, rx) = mpsc::channel(32);
        let history = self.input_history.clone();

        let stdin_tx = tx.clone();
        tokio::spawn(async move {
            let stdin = io::stdin();
            let reader = io::BufReader::new(stdin);
            let mut lines = reader.lines();

            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        let trimmed = line.trim().to_string();
                        let trimmed_str = trimmed.as_str();
                        if trimmed_str.is_empty() {
                            continue;
                        }

                        {
                            let mut lock = history.lock().await;
                            lock.push(line.clone());
                        }

                        let msg = match trimmed_str {
                            "/exit" | "/quit" => super::UserMsg::Exit,
                            "/cancel" => super::UserMsg::Cancel,
                            _ => super::UserMsg::Input(line),
                        };

                        if stdin_tx.send(msg).await.is_err() {
                            break;
                        }

                        if matches!(trimmed_str, "/exit" | "/quit") {
                            break;
                        }
                    }
                    Ok(None) => {
                        let _ = stdin_tx.send(super::UserMsg::Exit).await;
                        break;
                    }
                    Err(_) => {
                        let _ = stdin_tx.send(super::UserMsg::Exit).await;
                        break;
                    }
                }
            }
        });

        let ctrl_tx = tx.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                let _ = ctrl_tx.send(super::UserMsg::Cancel).await;
            }
        });

        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};

            let term_tx = tx.clone();
            tokio::spawn(async move {
                if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
                    sigterm.recv().await;
                    let _ = term_tx.send(super::UserMsg::Exit).await;
                }
            });

            let hup_tx = tx.clone();
            tokio::spawn(async move {
                if let Ok(mut sighup) = signal(SignalKind::hangup()) {
                    sighup.recv().await;
                    let _ = hup_tx.send(super::UserMsg::Exit).await;
                }
            });

            let quit_tx = tx;
            tokio::spawn(async move {
                if let Ok(mut sigquit) = signal(SignalKind::quit()) {
                    sigquit.recv().await;
                    let _ = quit_tx.send(super::UserMsg::Exit).await;
                }
            });
        }

        rx
    }

    async fn msg_system(&mut self, message: &str) -> Result<()> {
        for line in message.lines() {
            println!("* {}", line);
        }
        Ok(())
    }

    async fn msg_assistant(&mut self, message: &str) -> Result<()> {
        if message.is_empty() {
            println!();
        } else {
            print!("{}", message);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
        Ok(())
    }

    async fn msg_lua(&mut self, code: &str) -> Result<bool> {
        println!("Lua code to execute:\n{}", code);
        println!("Approve execution? (y/n): ");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let approved = matches!(input.trim().to_lowercase().as_str(), "y" | "yes");
        Ok(approved)
    }

    async fn msg_lua_result(&mut self, output: &str) -> Result<()> {
        for line in output.lines() {
            println!("> {}", line);
        }
        Ok(())
    }
}
