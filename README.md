# onui

**오누이(onui) [o.nu.i]**

Minimalist's AI Agent in a Single Executable.

## Usage

### Build and Installation

onui build with Rust.

- `cargo build -r` or `cargo install` to install the binary.

### Configuration

Your configuration file should be in TOML format.
The possible configuration path is:

- Passed by `--config` argument.
- Env `ONUI_CONFIG`
- `./.onui/config.toml`
- `~/.onui/config.toml`

```toml
default_llm = "my_open_ai"

[llm.my_open_ai]
type = "openai"
base_url = "https://api.openai.com/v1"
api_key = "sk-example"
# or, api_key_env = "OPENAI_API_KEY"
model = "gpt-5-nano"
```

### Usage

onui is a common chat-based agent. Please type your commands after the prompt.

### Commands

If your prompt starts with a slash `/`, it is treated as a command.
Type `/help` to see the list of available commands.
