# Agent Productization Roadmap

## Current Baseline

The project is currently a minimal local agent harness. It already has a LangGraph-based model/tool loop, OpenAI model construction through LangChain, short-term in-process message memory, two built-in tools (`calculator` and `echo`), a single-prompt CLI entrypoint, and unit tests for the core loop, memory, and tools.

This is a solid prototype baseline, but it is not yet a product-grade agent. The next work should avoid adding many visible features too early. The more scientific path is to first make the core runtime reliable, then improve interaction, then expand tools, then add persistence, observability, service APIs, and finally stronger autonomy.

## Iteration Principles

1. Stabilize the core before expanding capabilities.
   If the model/tool loop cannot handle errors, unknown tools, bad arguments, and configuration mistakes predictably, later features will be difficult to debug.

2. Prove behavior locally before building a service.
   A multi-turn CLI is cheaper to build and test than a Web/API product surface. Use it to validate agent behavior before adding server and UI layers.

3. Build a tool runtime before adding many tools.
   The important product capability is not just having more tools, but having tools that are typed, observable, permissioned, and recoverable when they fail.

4. Add observability before advanced memory and planning.
   Long-term memory and planning can make behavior harder to explain. Add run logs, traces, and evaluation cases first so regressions are visible.

5. Treat safety as part of the runtime, not as an afterthought.
   File access, shell execution, network calls, and credential use need permission checks, user confirmation, and audit records from the beginning.

## Recommended Iteration Direction

### Stage 1: Core Runtime Hardening

Goal: make the existing minimal agent loop reliable and predictable.

Do first:

- Normalize model, tool, and config errors into clear agent-level failures.
- Handle unknown tool calls without crashing the process.
- Return tool failures as structured messages the model can reason about.
- Add tests for error paths, not only successful tool calls.
- Add dependency locking and basic quality checks such as linting, formatting, and type checks.

Done when:

- The existing tests pass.
- Common failure modes have tests.
- A bad tool call or bad expression does not crash the agent unexpectedly.
- A new developer can install, test, and run the project from the documentation.

Defer:

- Web UI.
- Long-term memory.
- Large tool libraries.
- Multi-agent orchestration.

Likely stack:

- Python 3.12
- LangGraph
- LangChain
- pytest
- ruff
- pyright or mypy
- uv lockfile or an equivalent dependency lock

### Stage 2: Local Multi-Turn Interaction

Goal: make the agent usable as a local assistant, not only a one-shot command.

Do next:

- Add an interactive CLI chat mode.
- Keep memory across turns inside the same session.
- Add local commands such as `/clear`, `/exit`, and `/history`.
- Add streaming output if the model client supports it cleanly.
- Make configuration easy to override from CLI flags.

Done when:

- A user can have a multi-turn terminal conversation.
- The agent can use previous turns in the same session.
- The user can reset or inspect the current session.
- CLI behavior is covered by tests where practical.

Defer:

- Persistent history across process restarts.
- Browser UI.
- Background task execution.

Likely stack:

- argparse or Typer
- Rich for terminal rendering
- LangChain streaming callbacks
- pytest CLI tests

### Stage 3: Tool Runtime and Safe Tool Expansion

Goal: turn tools from demo functions into a controlled capability layer.

Do after local interaction works:

- Introduce a clear tool registry or tool catalog.
- Define tool metadata: name, description, argument schema, timeout, risk level, and permission requirement.
- Add structured tool results: success, failure, denied, and needs-confirmation.
- Add tool execution logging.
- Add a small set of practical local tools, starting with safe read-only tools such as file listing, file reading, and text search.
- Require explicit confirmation for tools that write files, call the network, or execute commands.

Done when:

- New tools can be added without changing the agent loop.
- Tool failures are visible and recoverable.
- Tool calls have enough metadata for logging, UI display, and permission checks.
- The default tool set remains small and safe.

Defer:

- Broad shell access.
- External service integrations.
- Plugin marketplace behavior.

Likely stack:

- LangChain `BaseTool`
- Pydantic schemas
- pathlib
- subprocess wrappers for controlled search commands
- pytest with `tmp_path`

### Stage 4: Observability and Evaluation

Goal: make agent behavior inspectable before adding complex memory and planning.

Do before long-term memory:

- Add structured run logs for each user request.
- Record model calls, tool calls, routing decisions, latency, and final outcome.
- Add a stable JSONL trace format.
- Create a small evaluation suite for expected tool use, no-tool cases, error recovery, and max-step behavior.
- Track regressions with repeatable local commands.

Done when:

- A failed run can be debugged from logs without stepping through the code.
- Core behavior has repeatable eval cases.
- Changes to prompts, tools, or graph routing can be compared against previous behavior.

Defer:

- Heavy observability platforms.
- Complex dashboards.
- Automated quality scoring that is not yet tied to product behavior.

Likely stack:

- Python logging or structlog
- JSONL traces
- pytest parameterized evals
- Optional OpenTelemetry later

### Stage 5: Persistent Sessions and Memory

Goal: allow the agent to resume context across runs while keeping memory behavior explainable.

Do after observability exists:

- Persist sessions, messages, tool calls, and run metadata.
- Support session resume from CLI.
- Add context trimming for long conversations.
- Add summary memory for older turns.
- Add a simple long-term memory interface for explicit user preferences and durable facts.
- Keep memory provenance visible so the agent can explain why a memory was used.

Done when:

- A user can resume a prior session.
- Long conversations do not exceed context limits blindly.
- Memory insertion, retrieval, and deletion are testable.
- The system distinguishes conversation history, summaries, and long-term memory.

Defer:

- Complex vector memory until there is a clear need.
- Automatic memory of every conversation detail.
- Cross-user memory sharing.

Likely stack:

- SQLite first
- SQLModel or SQLAlchemy
- Alembic if schema changes become frequent
- Optional vector store later, such as sqlite-vec, Chroma, or LanceDB

### Stage 6: Service API and Product Surface

Goal: expose the agent through stable interfaces that a UI or external client can use.

Do after CLI behavior is stable:

- Add an HTTP API for sessions and messages.
- Add streaming events through SSE or WebSocket.
- Expose run and tool-call status.
- Build a minimal Web UI only after the API shape is stable.
- Keep CLI and API using the same runtime code path.

Done when:

- A client can create a session, send a message, stream events, and inspect history.
- The Web UI shows message flow and tool execution state.
- API tests cover normal and failure paths.

Defer:

- Complex user management.
- Multi-tenant deployment.
- Advanced frontend workflows.

Likely stack:

- FastAPI
- SSE or WebSocket
- SQLite or PostgreSQL
- React and TypeScript when UI work starts
- httpx for API tests

### Stage 7: Safety, Permissions, and Audit

Goal: make powerful tools safe enough for real use.

Do before enabling risky tools by default:

- Classify tools by risk level.
- Add per-session or per-user tool allowlists.
- Require confirmation for write, network, shell, and credential-sensitive actions.
- Restrict filesystem tools to allowed workspace roots.
- Redact secrets from logs and traces.
- Persist audit events for denied, confirmed, and executed high-risk actions.

Done when:

- Risky tools cannot run silently.
- Tool permissions are enforced in code, not just documented.
- Audit logs show who approved what and what happened.
- Secret values do not appear in normal logs or traces.

Defer:

- Full enterprise policy engines.
- Organization-level RBAC.
- Remote sandbox clusters.

Likely stack:

- Pydantic policy models
- pathlib path validation
- local sandboxing first
- audit tables in the existing database
- optional OPA or a stronger policy engine later

### Stage 8: Task-Oriented Autonomy

Goal: move from chat-with-tools to managed task execution.

Do only after tools, memory, and observability are stable:

- Add a task/run/step state model.
- Let the agent create a plan for multi-step tasks.
- Execute plans step by step with checkpoints.
- Support retry, cancellation, and human approval at key steps.
- Keep the first implementation single-agent before introducing multi-agent patterns.

Done when:

- A long task can pause, resume, fail clearly, or ask for approval.
- Each step has visible status and trace history.
- Planning improves task execution without hiding what the agent is doing.

Defer:

- Multi-agent orchestration.
- Autonomous background execution without user-visible controls.
- Complex schedulers.

Likely stack:

- LangGraph state machines
- persistent checkpoints
- database-backed task state
- SSE or WebSocket event streaming

## Practical Priority

The recommended near-term order is:

1. Core runtime hardening.
2. Multi-turn CLI.
3. Tool runtime and safe local tools.
4. Observability and evaluation.
5. Persistent sessions and memory.
6. Service API and UI.
7. Safety, permissions, and audit.
8. Task-oriented autonomy.

This order keeps the project grounded. It improves the agent's reliability before increasing its power, validates behavior locally before adding product surfaces, and adds observability before introducing memory and planning features that can otherwise make failures difficult to diagnose.
