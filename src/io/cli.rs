use anyhow::{Result, anyhow};
use std::io::{Write, stdout};
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use tokio::io::{self, AsyncBufReadExt};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::io::{Input, Signal};

use super::{IOChan, Output};

const CHANNEL_BUFFER_SIZE: usize = 32;

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

        let (input_tx, input_rx) = mpsc::channel(CHANNEL_BUFFER_SIZE);
        let (signal_tx, signal_rx) = mpsc::channel(CHANNEL_BUFFER_SIZE);
        let (output_tx, output_rx) = mpsc::channel(CHANNEL_BUFFER_SIZE);

        {
            let input_tx = input_tx.clone();
            let signal_tx = signal_tx.clone();
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

                        let i = Input::from_raw(buf.as_str());
                        buf.clear();
                        if let Some(signal) = i.as_signal() {
                            let _ = signal_tx.send(signal).await;
                        } else {
                            let _ = input_tx.send(i).await;
                        }
                    }
                }
            }));
        }

        {
            let ctrl_tx = signal_tx.clone();
            let sigint_cnt = self.sigint_cnt.clone();
            self.async_tasks.push(tokio::spawn(async move {
                loop {
                    if tokio::signal::ctrl_c().await.is_err() {
                        break;
                    }
                    sigint_cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let sig = if sigint_cnt.load(std::sync::atomic::Ordering::SeqCst) >= 2 {
                        Signal::Exit
                    } else {
                        Signal::Cancel
                    };
                    if ctrl_tx.send(sig).await.is_err() {
                        break;
                    }
                    if sigint_cnt.load(std::sync::atomic::Ordering::SeqCst) >= 2 {
                        break;
                    }
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
                        print!("{}", message);
                        let _ = stdout().flush();
                    }
                    Output::LuaCode { id, code } => {
                        println!("---[LUA:{}]---", id);
                        for line in code.lines() {
                            println!("    {}", line);
                        }
                        println!("---[END:{}]---", id);
                        print!("* Approve execution? (Yes/No/Always)");
                        let _ = stdout().flush();
                    }
                    Output::LuaResult { id, output } => {
                        println!("-->[RESULT:{}]---", id);
                        for line in output.lines() {
                            println!("    {}", line);
                        }
                        println!("-->[END RESULT:{}]---", id);
                    }
                    Output::InputReady => {
                        print!("\n> ");
                        let _ = stdout().flush();
                    }
                }
            }
        }));

        Ok(IOChan {
            input_rx,
            signal_rx,
            output_tx,
        })
    }

    fn close(&mut self) -> Result<()> {
        self.abort_all_tasks();
        Ok(())
    }
}
