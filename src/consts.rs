pub const PLATFORM: &str = if cfg!(target_os = "windows") {
    "Windows"
} else if cfg!(target_os = "linux") {
    "Linux"
} else if cfg!(target_os = "macos") {
    "macOS"
} else {
    "Unknown"
};

pub const DEFAULT_SYSTEM_PROMPT: &str = const_format::formatcp!(
    r#"
# Grand Rules
- Platform: {platform}
- 'lua' tools are available, to execute code and get stdout & return values.
  - Use `print` for intermediate output.
  - Do not use `os.exit`. Instead of `os.execute`, use `io.popen` because you can capture outputs.
- You may run code step-by-step. After execution, analyze the result, and continue to answer or run code.
- You may create utility script file in .onui/*.lua. You may load by 'load'.
"#,
    platform = PLATFORM
);
