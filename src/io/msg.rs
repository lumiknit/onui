#[derive(Debug, Clone)]
pub enum Command {
    Exit,
    Cancel,
    Help,
    Status,
    ResetVM,
    Compact,

    Approve,
    Reject,
    ApproveAlways,
}

static CMD_NAMES: phf::Map<&'static str, Command> = phf::phf_map! {
    "exit" => Command::Exit,
    "quit" => Command::Exit,
    "q" => Command::Exit,
    "cancel" => Command::Cancel,
    "c" => Command::Cancel,
    "stop" => Command::Cancel,
    "help" => Command::Help,
    "status" => Command::Status,
    "resetvm" => Command::ResetVM,
    "compact" => Command::Compact,
    "approve" => Command::Approve,
    "a" => Command::Approve,
    "reject" => Command::Reject,
    "r" => Command::Reject,
    "always" => Command::ApproveAlways,
};

impl Command {
    pub fn from_name(name: &str) -> Option<Command> {
        CMD_NAMES.get(name.to_lowercase().as_str()).cloned()
    }
}

/// Input is common input of user.
///
/// Command looks like "/<cmd> [<arg>]\n<details...>"
pub enum Input {
    Text(String), // normal text input.
    Command {
        cmd: Command,
        arg: String,
        details: String,
    },
}

impl Input {
    pub fn from_raw(original: &str) -> Self {
        let trimmed = original.trim();
        if let Some(stripped) = trimmed.strip_prefix('/') {
            let trimmed = stripped.trim();
            if !trimmed.is_empty() {
                // Find next whitespace to split command name and rest.
                let mut parts = trimmed.splitn(2, char::is_whitespace);
                let name_raw = parts.next().unwrap_or("");
                if let Some(cmd) = Command::from_name(name_raw) {
                    let mut rest = parts.next().unwrap_or("").splitn(2, '\n');
                    let arg = rest.next().unwrap_or("").trim().to_string();
                    let details = rest.next().unwrap_or("").trim().to_string();
                    return Self::Command {
                        cmd: cmd,
                        arg: arg,
                        details: details,
                    };
                }
            }
        }

        Self::Text(original.to_string())
    }
}

/// Output is from the system to the user.
pub enum Output {
    SystemMsg(String),                        // system message, complete lines.
    AssistantMsg(String),                     // assistant message, may be streaming.
    LuaCode { id: String, code: String },     // lua code to be approved by user.
    LuaResult { id: String, output: String }, // lua execution result, complete lines.
}
