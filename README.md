# Mini Agent Harness

This project is a minimal local agent harness prototype. It only includes four core parts: an agent loop, short-term memory, a local tool registry, and a model client.

It intentionally avoids LangChain, LangGraph, databases, web services, MCP, multi-agent orchestration, planners, long-term memory, async queues, and complex tracing.

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

```bash
uv run python -m mini_agent.cli "What is 2 + 3 * 4?"
```

You can also use the installed script:

```bash
uv run mini-agent "What is 12 * (3 + 4)?"
```

## Test

```bash
uv run pytest
```

## Structure

```text
src/mini_agent/
  config.py   # environment configuration
  models.py   # ModelClient protocol and OpenAI client
  memory.py   # in-run short-term memory
  tools.py    # tool data structure, registry, calculator, echo
  agent.py    # model/tool loop
  cli.py      # command line entrypoint
```
