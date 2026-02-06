use anyhow::{Result, anyhow};
use std::io::{Write, stdout};
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use tokio::io::{self, AsyncBufReadExt};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::io::msg::Action;

use super::{IOChan, Output, Signal};

/// CliIO is an implementation of IO, which is for command line interface.
pub struct CliIO {
    async_tasks: Vec<JoinHandle<()>>,
    sigint_cnt: Arc<AtomicU8>,
}

impl CliIO {
    /// Create a new CliIO instance.
    pub fn new() -> Self {
        CliIO {
            async_tasks: Vec::new(),
            sigint_cnt: Arc::new(AtomicU8::new(0)),
        }
    }

    pub fn running(&self) -> bool {
        !self.async_tasks.is_empty()
    }

    pub fn abort_all_tasks(&mut self) {
        for handle in self.async_tasks.drain(..) {
            handle.abort();
        }
    }
}

impl super::IO for CliIO {
    fn open(&mut self) -> Result<IOChan> {
        if self.running() {
            return Err(anyhow!("CliIO is already open"));
        }

        let (signal_tx, signal_rx) = mpsc::channel(32);
        let (input_tx, input_rx) = mpsc::channel(32);
        let (output_tx, output_rx) = mpsc::channel(32);

        let signal_tx_input = signal_tx.clone();
        let sigint_cnt = self.sigint_cnt.clone();
        self.async_tasks.push(tokio::spawn(async move {
            let i = io::stdin();
            let reader = io::BufReader::new(i);
            let mut lines = reader.lines();

            let mut buf = String::new();

            loop {
                if let Ok(Some(line)) = lines.next_line().await {
                    sigint_cnt.store(0, std::sync::atomic::Ordering::SeqCst);
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    buf += &line;

                    if buf.ends_with('\\') {
                        // Continue to read next line.
                        buf.pop();
                        buf.push('\n');
                        continue;
                    }

                    match Action::from_raw(buf.as_str()) {
                        Action::Signal(s) => {
                            let _ = signal_tx_input.send(s).await;
                        }
                        Action::Input(i) => {
                            let _ = input_tx.send(i).await;
                        }
                    }
                }
            }
        }));

        let ctrl_tx = signal_tx.clone();
        let sigint_cnt = self.sigint_cnt.clone();
        self.async_tasks.push(tokio::spawn(async move {
            loop {
                if tokio::signal::ctrl_c().await.is_err() {
                    break;
                }
                sigint_cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let msg = if sigint_cnt.load(std::sync::atomic::Ordering::SeqCst) >= 2 {
                    Signal::Exit
                } else {
                    Signal::Cancel
                };
                if ctrl_tx.send(msg).await.is_err() {
                    break;
                }
                if sigint_cnt.load(std::sync::atomic::Ordering::SeqCst) >= 2 {
                    break;
                }
            }
        }));

        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};

            let term_tx = signal_tx.clone();
            self.async_tasks.push(tokio::spawn(async move {
                if let Ok(mut sigterm) = signal(SignalKind::terminate()) {
                    sigterm.recv().await;
                    let _ = term_tx.send(Signal::Exit).await;
                }
            }));

            let hup_tx = signal_tx.clone();
            self.async_tasks.push(tokio::spawn(async move {
                if let Ok(mut sighup) = signal(SignalKind::hangup()) {
                    sighup.recv().await;
                    let _ = hup_tx.send(Signal::Exit).await;
                }
            }));

            let quit_tx = signal_tx.clone();
            self.async_tasks.push(tokio::spawn(async move {
                if let Ok(mut sigquit) = signal(SignalKind::quit()) {
                    sigquit.recv().await;
                    let _ = quit_tx.send(Signal::Exit).await;
                }
            }));
        }

        self.async_tasks.push(tokio::spawn(async move {
            let mut output_rx = output_rx;
            while let Some(output) = output_rx.recv().await {
                match output {
                    Output::SystemMsg(message) => {
                        for line in message.lines() {
                            println!("* {}", line);
                        }
                    }
                    Output::AssistantMsg(message) => {
                        if message.is_empty() {
                            println!();
                            print!("> ");
                            let _ = stdout().flush();
                        } else {
                            print!("{}", message);
                            let _ = stdout().flush();
                        }
                    }
                    Output::LuaCode { id, code } => {
                        println!("------[lua]------");
                        println!("    [id] {}", id);
                        for line in code.lines() {
                            println!("    {}", line);
                        }
                        println!("------[end]------");
                        print!("* Approve execution? (y/n) > ");
                        let _ = stdout().flush();
                    }
                    Output::LuaResult { id, output } => {
                        println!("--> [id] {}", id);
                        for line in output.lines() {
                            println!("--> {}", line);
                        }
                    }
                }
            }
        }));

        Ok(IOChan {
            signal_rx,
            input_rx,
            output_tx,
        })
    }

    fn close(&mut self) -> Result<()> {
        self.abort_all_tasks();
        Ok(())
    }
}
