/// mod io is the IO interaction module for User or other systems.
pub mod cli;
pub mod msg;

use anyhow::Result;
use tokio::sync::mpsc;

pub use msg::{Command, Input, Output};

pub struct IOChan {
    pub input_rx: mpsc::Receiver<Input>,
    pub output_tx: mpsc::Sender<Output>,
}

/// IO is an abstraction of background loop,
/// which handles input/output interactions via mpsc channels,
/// between User/Client and System.
///
/// When agent system starts, it'll 'open' the IO, and communicate via channels.
/// When agent system ends, it'll 'close' the IO, and cleanup resources.
pub trait IO {
    /// Trigger when IO starts.
    /// This is an helper to setup IO resources.
    fn open(&mut self) -> Result<IOChan>;

    /// Trigger when IO ends.
    /// This is an helper to cleanup IO resources.
    fn close(&mut self) -> Result<()>;
}
