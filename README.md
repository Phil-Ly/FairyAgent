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
  models.py   # ChatOpenAI factory
  memory.py   # in-run LangChain message memory
  tools.py    # LangChain calculator and echo tools
  agent.py    # LangGraph model/tool loop
  cli.py      # command line entrypoint
```
