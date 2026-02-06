#[derive(Debug, Clone)]
pub enum LuaCallStatus {
    WaitingReview,
    Executing,
    Approved,
    Rejected,
}

/// LuaCall is a struct to manage lua code execution requests.
pub struct Item {
    pub id: String,
    pub status: LuaCallStatus,

    /// Output from the execution, or rejected reason.
    pub output: String,
}

pub struct Manager {
    pub unhandled: Vec<Item>,
    pub handled: Vec<Item>,
}

impl Manager {
    pub fn new() -> Self {
        Self {
            unhandled: Vec::new(),
            handled: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.unhandled.clear();
        self.handled.clear();
    }

    pub fn all_ready(&self) -> bool {
        self.unhandled.is_empty()
    }

    pub fn insert(&mut self, id: &str, code: &str) {
        let call = Item {
            id: id.to_string(),
            status: LuaCallStatus::WaitingReview,
            output: code.to_string(),
        };
        self.unhandled.push(call);
    }

    fn pop(&mut self, id: &str) -> anyhow::Result<Item> {
        if let Some(pos) = self
            .unhandled
            .iter()
            .position(|call| call.id == id && call.status == LuaCallStatus::WaitingReview)
        {
            Ok(self.unhandled.remove(pos))
        } else {
            Err(anyhow::anyhow!("Call with id {} not found", id))
        }
    }

    /// Just move to handled with executing status.
    pub fn approve(&mut self, id: &str) -> Result<()> {
        let mut call = self.pop(id)?;
        call.status = LuaCallStatus::Executing;
        self.handled.push(call);
        Ok(())
    }

    pub fn executed(&mut self, id: &str, output: &str) -> Result<()> {
        let mut call = self.pop(id)?;
        call.status = LuaCallStatus::Approved;
        call.output = output.to_string();
        self.handled.push(call);
        Ok(())
    }

    pub fn reject(&mut self, id: &str, reason: &str) -> Result<()> {
        let mut call = self.pop(id)?;
        call.status = LuaCallStatus::Rejected;
        call.output = reason.to_string();
        self.handled.push(call);
        Ok(())
    }
}
