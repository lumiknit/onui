use anyhow::Result;
use std::io::{Write, stdin, stdout};
use std::sync::Arc;
use tokio::io::{self, AsyncBufReadExt};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

/// CliIO is an implementation of IO, which is for command line interface.
pub struct CliIO {
    input_history: Arc<Mutex<Vec<String>>>,

    in_chan_sender: mpsc::Sender<super::UserMsg>,
    in_chan_receiver: Option<mpsc::Receiver<super::UserMsg>>,

    input_task: Option<JoinHandle<()>>,
    signal_tasks: Vec<JoinHandle<()>>,
}

impl CliIO {
    /// Create a new CliIO instance.
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(32);

        CliIO {
            input_history: Arc::new(Mutex::new(Vec::new())),
            in_chan_sender: tx,
            in_chan_receiver: Some(rx),
            input_task: None,
            signal_tasks: Vec::new(),
        }
    }
}

impl super::IO for CliIO {
    fn open(&mut self) -> Result<()> {
        if self.input_task.is_some() {
            return Ok(());
        }

        let history = self.input_history.clone();
        let stdin_tx = self.in_chan_sender.clone();
        self.input_task = Some(tokio::spawn(async move {
            let i = io::stdin();
            let reader = io::BufReader::new(i);
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

                        if stdin_tx.send(super::UserMsg::Input(line)).await.is_err() {
                            break;
                        }
                    }
                    _ => {
                        let _ = stdin_tx.send(super::UserMsg::Exit).await;
                        break;
                    }
                }
            }
        }));

        let ctrl_tx = self.in_chan_sender.clone();
        self.signal_tasks.push(tokio::spawn(async move {
            let mut count = 0u8;
            loop {
                if tokio::signal::ctrl_c().await.is_err() {
                    break;
                }
                count = count.saturating_add(1);
                let msg = if count >= 2 {
                    super::UserMsg::Exit
                } else {
                    super::UserMsg::Cancel
                };
                if ctrl_tx.send(msg).await.is_err() {
                    break;
                }
                if count >= 2 {
                    break;
                }
            }
        }));

        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};

            let term_tx = self.in_chan_sender.clone();
            self.signal_tasks.push(tokio::spawn(async move {
                if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
                    sigterm.recv().await;
                    let _ = term_tx.send(super::UserMsg::Exit).await;
                }
            }));

            let hup_tx = self.in_chan_sender.clone();
            self.signal_tasks.push(tokio::spawn(async move {
                if let Ok(mut sighup) = signal(SignalKind::hangup()) {
                    sighup.recv().await;
                    let _ = hup_tx.send(super::UserMsg::Exit).await;
                }
            }));

            let quit_tx = self.in_chan_sender.clone();
            self.signal_tasks.push(tokio::spawn(async move {
                if let Ok(mut sigquit) = signal(SignalKind::quit()) {
                    sigquit.recv().await;
                    let _ = quit_tx.send(super::UserMsg::Exit).await;
                }
            }));
        }

        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        if let Some(handle) = self.input_task.take() {
            handle.abort();
        }

        for handle in self.signal_tasks.drain(..) {
            handle.abort();
        }

        Ok(())
    }

    fn input_channel(&mut self) -> mpsc::Receiver<super::UserMsg> {
        self.in_chan_receiver
            .take()
            .expect("input_channel called more than once")
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
            std::io::stdout().flush().unwrap();
        }
        Ok(())
    }

    async fn msg_lua(&mut self, code: &str) -> Result<bool> {
        println!("------[lua]------");
        for line in code.lines() {
            println!("    {}", line);
        }
        println!("------[end]------");
        print!("* Approve execution? (y/n) > ");
        stdout().flush().unwrap();
        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();
        let approved = matches!(input.trim().to_lowercase().as_str(), "y" | "yes");
        Ok(approved)
    }

    async fn msg_lua_result(&mut self, output: &str) -> Result<()> {
        for line in output.lines() {
            println!("--> {}", line);
        }
        Ok(())
    }

    async fn llm_stopped(&mut self) -> Result<()> {
        print!("> ");
        stdout().flush().unwrap();
        Ok(())
    }
}
