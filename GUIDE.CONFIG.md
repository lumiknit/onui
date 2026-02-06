# Configuration Guide

## LLM

### OpenAI Chat Completion

```toml
[llm.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"
```

### Gemini, OpenAI compatible API

```toml
[llm.gemini]
type = "openai"
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key = "YOUR_API_KEY_HERE"
model = "gemini-2.5-flash"
```
