/// Signal is some notification or non-input message from the IO system.
/// This may include interrupt, kill, etc.
pub enum Signal {
    Cancel, // e.g. Ctrl-C, stop current task, but not exit at all.
    Exit,   // exit the program.
}

/// Input is common input of user.
pub enum Input {
    Text(String), // normal text input.
    Command { name: String, rest: String },
}

impl Input {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Input::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn is_command(&self, cmd_name: &str) -> bool {
        match self {
            Input::Command { name, .. } if name == cmd_name => true,
            _ => false,
        }
    }
}

pub enum Action {
    Signal(Signal),
    Input(Input),
}

impl Action {
    pub fn from_raw(original: &str) -> Action {
        let trimmed = original.trim();
        if let Some(stripped) = trimmed.strip_prefix('/') {
            let trimmed = stripped.trim();
            if !trimmed.is_empty() {
                // Find next whitespace to split command name and rest.
                let mut parts = trimmed.splitn(2, char::is_whitespace);
                let name = parts.next().unwrap_or("");
                let rest = parts.next().unwrap_or("");
                match name {
                    "exit" | "quit" => return Action::Signal(Signal::Exit),
                    "cancel" | "stop" => return Action::Signal(Signal::Cancel),
                    _ => {
                        return Action::Input(Input::Command {
                            name: name.to_string(),
                            rest: rest.to_string(),
                        });
                    }
                }
            }
        }

        Action::Input(Input::Text(original.to_string()))
    }
}

/// Output is from the system to the user.
pub enum Output {
    SystemMsg(String),                        // system message, complete lines.
    AssistantMsg(String),                     // assistant message, may be streaming.
    LuaCode { id: String, code: String },     // lua code to be approved by user.
    LuaResult { id: String, output: String }, // lua execution result, complete lines.
}
