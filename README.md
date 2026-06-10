# Mini Agent Harness

This project is a minimal local agent harness prototype built with LangChain and LangGraph. It keeps the first version's small scope while letting LangChain handle model/tool abstractions and LangGraph handle the agent loop.

It intentionally avoids databases, web services, MCP, multi-agent orchestration, planners, long-term memory, async queues, and complex tracing.

## Install

```bash
uv sync
```

## Configure

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Optional values:

- `MODEL_NAME`, default: `gpt-4.1-mini`
- `MAX_STEPS`, default: `8`

## Run

Run with the configured OpenAI model:

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

Run a deterministic local smoke demo without an API key:

```bash
uv run python -m mini_agent.cli demo "What is 2 + 3 * 4?"
```

List bundled tools:

```bash
uv run python -m mini_agent.cli tools
```

Check local deployment readiness:

```bash
uv run python -m mini_agent.cli doctor
```

## Test

```bash
uv run pytest
```

## Structure

```text
src/mini_agent/
  config.py   # environment configuration
  models.py   # ChatOpenAI factory
  memory.py   # in-run LangChain message memory
  tools.py    # LangChain calculator and echo tools
  demo.py     # deterministic local demo model
  agent.py    # LangGraph model/tool loop
  cli.py      # command line entrypoint
```
