# AgentLoop Harness

This project is a minimal local agent harness prototype built with LangChain and LangGraph. It keeps the first version's small scope while letting LangChain handle model/tool abstractions and LangGraph handle the agent loop.

It intentionally avoids databases, web services, MCP, multi-agent orchestration, planners, long-term memory, async queues, and complex tracing.

## Install

This project expects Python 3.12 and `uv`.

```bash
uv sync
```

The default install supports core development and local examples. Install the
extra matching the model protocol you want to use:

```bash
uv sync --extra openai-compatible
uv sync --extra anthropic
uv sync --extra gemini
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

`MODEL_API_KEY` is an explicit override for every provider. Without it, each
provider uses its official environment variables:

- OpenAI-compatible: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GOOGLE_API_KEY`, then `GEMINI_API_KEY`

Native Anthropic configuration:

```env
MODEL_PROVIDER=anthropic
MODEL_NAME=claude-sonnet-4-6
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Native Gemini configuration:

```env
MODEL_PROVIDER=gemini
MODEL_NAME=gemini-2.5-flash
GOOGLE_API_KEY=your_google_api_key_here
```

`MODEL_BASE_URL` can override the selected provider's default endpoint. Omit
`MODEL_TEMPERATURE` to let the provider integration choose its model-specific
default.

Example OpenAI-compatible endpoints:

- DeepSeek: `https://api.deepseek.com`
- Qwen compatible mode: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Moonshot: `https://api.moonshot.cn/v1`

Never commit `.env` or real API keys. `.env.example` should contain placeholders only.

## Run

Run after installing the matching provider extra and configuring an accepted
API key:

```bash
uv run python -m agentloop.cli run "What is 2 + 3 * 4?"
```

The legacy prompt-only form is still supported:

```bash
uv run python -m agentloop.cli "What is 2 + 3 * 4?"
```

You can also use the installed script after `uv sync`:

```bash
uv run agentloop run "What is 12 * (3 + 4)?"
```

Start a provider-backed multi-turn chat session:

```bash
uv run python -m agentloop.cli chat
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

The local examples live outside `src/agentloop`, so the product runtime stays provider-backed and free of test or example model logic.

List bundled tools:

```bash
uv run python -m agentloop.cli tools
```

Bundled tools are managed through the local tool runtime:

- `calculator`: safely evaluates simple arithmetic.
- `echo`: returns text unchanged.
- `list_files`: lists direct children of a workspace directory.
- `read_file`: reads a UTF-8 text file inside the workspace.
- `search_text`: searches UTF-8 text files inside the workspace.
- `project_tree`: shows a depth-limited workspace directory tree.

Tool calls return normalized runtime metadata such as status, duration,
risk level, and read-only state. The default bundled tools are low-risk
and read-only. Local file tools reject paths outside the current workspace.
Tools marked high-risk or requiring confirmation are not executed silently:
the runtime returns a confirmation result, and the agent emits an
`intervention_request`. If the same tool fails 3 consecutive times in a run,
the agent stops automatic retry and asks for user intervention.

Check local deployment readiness:

```bash
uv run python -m agentloop.cli doctor
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
uv run pytest --cov=agentloop
uv run ruff check .
uv run pyright
```

If `uv` is not installed, install it first and rerun `uv sync`. The source tree also supports direct local verification with a Python 3.12 virtual environment.

## Structure

```text
src/agentloop/
  config.py   # environment configuration
  models.py   # ChatOpenAI factory
  memory.py   # in-run LangChain message memory
  tool_runtime.py  # tool metadata and normalized tool results
  tools.py    # bundled low-risk LangChain tools
  diagnostics.py    # runtime readiness and tool display helpers
  chat_session.py   # line-oriented chat command handling
  agent.py    # LangGraph model/tool loop
  cli.py      # run, chat, tools, and doctor commands

examples/
  local_model.py  # deterministic local example model
  local_run.py    # one-turn local example
  local_chat.py   # multi-turn local example
```
