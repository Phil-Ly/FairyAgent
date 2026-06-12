# Mini Agent Harness

This project is a minimal local agent harness prototype built with LangChain and LangGraph. It keeps the first version's small scope while letting LangChain handle model/tool abstractions and LangGraph handle the agent loop.

It intentionally avoids databases, web services, MCP, multi-agent orchestration, planners, long-term memory, async queues, and complex tracing.

## Install

This project expects Python 3.12 and `uv`.

```bash
uv sync
```

The default install supports core development and local examples. Real model provider usage through OpenAI-compatible APIs is optional:

```bash
uv sync --extra openai-compatible
```

## Configure

```bash
cp .env.example .env
```

For local examples, no API key is required. For the product CLI, configure an OpenAI-compatible endpoint in `.env`:

```env
MODEL_PROVIDER=openai_compatible
MODEL_NAME=deepseek-chat
MODEL_BASE_URL=https://api.deepseek.com
MODEL_API_KEY=your_model_api_key_here
MODEL_TEMPERATURE=0
MODEL_TIMEOUT_SECONDS=60
MAX_STEPS=8
```

`MODEL_API_KEY` is preferred. `OPENAI_API_KEY` remains as a legacy fallback, but new provider configs should use `MODEL_API_KEY`.

Example OpenAI-compatible endpoints:

- DeepSeek: `https://api.deepseek.com`
- Qwen compatible mode: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Moonshot: `https://api.moonshot.cn/v1`

Never commit `.env` or real API keys. `.env.example` should contain placeholders only.

## Run

Run with the configured model provider after installing the optional compatible-provider extra and setting `MODEL_API_KEY`:

```bash
uv run python -m mini_agent.cli run "What is 2 + 3 * 4?"
```

The legacy prompt-only form is still supported:

```bash
uv run python -m mini_agent.cli "What is 2 + 3 * 4?"
```

You can also use the installed script after `uv sync`:

```bash
uv run mini-agent run "What is 12 * (3 + 4)?"
```

Start a provider-backed multi-turn chat session:

```bash
uv run python -m mini_agent.cli chat
```

Inside chat, use `/exit`, `/clear`, `/history`, `/tools`, and `/doctor`.
Tool calls are printed as `tool-call`, tool results as `tool-result`, and
final model replies as `assistant`.

Run a local example without an API key:

```bash
uv run python examples/local_run.py "What is 2 + 3 * 4?"
```

Start a local example chat session without an API key:

```bash
uv run python examples/local_chat.py
```

The local examples live outside `src/mini_agent`, so the product runtime stays provider-backed and free of test or example model logic.

List bundled tools:

```bash
uv run python -m mini_agent.cli tools
```

Check local deployment readiness:

```bash
uv run python -m mini_agent.cli doctor
```

`doctor` reports Python, uv, core LangChain dependencies, provider configuration, key presence, and bundled tools. It never prints the API key value.

## Test

```bash
uv run pytest
```

Run the optional real provider smoke test only when credentials and the compatible-provider extra are available:

```bash
RUN_MODEL_INTEGRATION_TESTS=1 uv run pytest tests/test_integration_model_provider.py
```

Real provider integration should stay optional until a project or deployment environment has stable credentials. Keep default tests on fake models so local development does not depend on network access, API quota, or model nondeterminism.

## Quality Checks

```bash
uv run pytest
uv run pytest --cov=mini_agent
uv run ruff check .
uv run pyright
```

If `uv` is not installed, install it first and rerun `uv sync`. The source tree also supports direct local verification with a Python 3.12 virtual environment.

## Structure

```text
src/mini_agent/
  config.py   # environment configuration
  models.py   # ChatOpenAI factory
  memory.py   # in-run LangChain message memory
  tools.py    # LangChain calculator and echo tools
  diagnostics.py    # runtime readiness and tool display helpers
  chat_session.py   # line-oriented chat command handling
  agent.py    # LangGraph model/tool loop
  cli.py      # run, chat, tools, and doctor commands

examples/
  local_model.py  # deterministic local example model
  local_run.py    # one-turn local example
  local_chat.py   # multi-turn local example
```
