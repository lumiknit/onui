use anyhow::{Result, anyhow};
use mlua::{HookTriggers, Lua, MultiValue, Value, Variadic, VmState};
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

/// Result of executing Lua code.
pub struct LuaExecution {
    pub stdout: String,
    pub error: Option<String>,
    pub returns: Vec<String>,
}

fn value_to_string(lua: &Lua, value: &Value) -> Result<String, mlua::Error> {
    match value {
        Value::String(text) => Ok(text.to_str()?.to_string()),
        _ => match lua.coerce_string(value.clone())? {
            Some(text) => Ok(text.to_str()?.to_string()),
            None => Ok(format!("{:?}", value)),
        },
    }
}

fn map_lua_error(error: mlua::Error) -> anyhow::Error {
    anyhow!(error.to_string())
}

/// Wraps a single embedded LuaVM instance.
pub struct LuaVM {
    /// The underlying Lua instance.
    lua: Lua,

    /// Captured standard output from the last execution.
    out_buffer: Rc<RefCell<String>>,
}

pub trait LuaRuntime {
    fn execute_script(&self, script: &str, timeout_sec: Option<u64>) -> Result<LuaExecution>;
    fn reset(&mut self) -> Result<()>;
}

impl LuaVM {
    /// Initialize a new Lua instance with the built-in functions.
    fn setup_functions(&mut self) -> Result<()> {
        // Print function to capture stdout
        let out_buffer = Rc::clone(&self.out_buffer);
        let print_fn = self
            .lua
            .create_function(move |lua, args: Variadic<Value>| {
                let mut buffer = out_buffer.borrow_mut();
                for (idx, value) in args.iter().enumerate() {
                    if idx > 0 {
                        buffer.push('\t');
                    }
                    let text = value_to_string(lua, value)?;
                    buffer.push_str(&text);
                }
                buffer.push('\n');
                Ok(())
            })
            .expect("Failed to create print function");
        let globals = self.lua.globals();
        globals
            .set("print", print_fn)
            .expect("Failed to set print function");

        let out_buffer = Rc::clone(&self.out_buffer);
        let write_fn = self
            .lua
            .create_function(move |lua, args: Variadic<Value>| {
                let mut buffer = out_buffer.borrow_mut();
                for value in args.iter() {
                    let text = value_to_string(lua, value)?;
                    buffer.push_str(&text);
                }
                Ok(())
            })
            .expect("Failed to create io.write function");

        if let Ok(io_table) = globals.get::<mlua::Table>("io") {
            io_table
                .set("stdin", Value::Nil)
                .expect("Failed to disable io.stdin");
            io_table
                .set("stdout", Value::Nil)
                .expect("Failed to disable io.stdout");
            io_table
                .set("stderr", Value::Nil)
                .expect("Failed to disable io.stderr");
            io_table
                .set("write", write_fn)
                .expect("Failed to override io.write");
        }

        if let Ok(os_table) = globals.get::<mlua::Table>("os") {
            os_table
                .set("exit", Value::Nil)
                .expect("Failed to disable os.exit");
            os_table
                .set("execute", Value::Nil)
                .expect("Failed to disable os.execute");
        }
        Ok(())
    }

    /// Create a new Lua virtual machine.
    pub fn new() -> Result<Self> {
        let lua = Lua::new();
        let mut s = Self {
            lua,
            out_buffer: Rc::new(RefCell::new(String::new())),
        };
        s.setup_functions()?;
        Ok(s)
    }

    /// Execute the provided Lua code string.
    pub fn execute_script(&self, script: &str, timeout_sec: Option<u64>) -> Result<LuaExecution> {
        // Clear previous output
        self.out_buffer.borrow_mut().clear();

        if let Some(seconds) = timeout_sec {
            let start = Instant::now();
            let timeout = Duration::from_secs(seconds);
            self.lua
                .set_hook(
                    HookTriggers::new().every_nth_instruction(10_000),
                    move |_lua, _debug| {
                        if start.elapsed() > timeout {
                            Err(mlua::Error::RuntimeError(
                                "Lua execution timed out".to_string(),
                            ))
                        } else {
                            Ok(VmState::Continue)
                        }
                    },
                )
                .map_err(map_lua_error)?;
        }

        let exec_result: Result<MultiValue, mlua::Error> =
            self.lua.load(script).set_name("onui-agent").eval();

        self.lua
            .set_hook(HookTriggers::new(), |_lua, _debug| Ok(VmState::Continue))
            .map_err(map_lua_error)?;

        let stdout = self.out_buffer.borrow().clone();
        match exec_result {
            Ok(values) => {
                let returns = values
                    .iter()
                    .map(|value| value_to_string(&self.lua, value))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(map_lua_error)?;
                Ok(LuaExecution {
                    stdout,
                    error: None,
                    returns,
                })
            }
            Err(err) => Ok(LuaExecution {
                stdout,
                error: Some(format!("Lua execution failed: {err}")),
                returns: Vec::new(),
            }),
        }
    }
}

impl LuaRuntime for LuaVM {
    fn execute_script(&self, script: &str, timeout_sec: Option<u64>) -> Result<LuaExecution> {
        LuaVM::execute_script(self, script, timeout_sec)
    }

    fn reset(&mut self) -> Result<()> {
        *self = LuaVM::new()?;
        Ok(())
    }
}
