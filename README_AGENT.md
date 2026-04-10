# Agent Runtime Rules

## Source of Truth
- The external memory system on disk is the only trusted project truth across runs.
- Agent context is temporary and must not be treated as durable memory.
- Project history must be recoverable from the filesystem and Git, not from conversation state.

## Mandatory Read Order
- Before doing any work, a Coding Agent must read `memory/tasks/TODO.md`, `memory/tasks/DONE.md`, and `memory/status/PROGRESS.md`.

## Execution Rules
- Each round must complete exactly one smallest executable and verifiable task.
- Do not start a second task in the same round.
- Do not overwrite existing task state without explicit confirmation that re-initialization is required.
- Initializer Agent runs once and must not perform business development.

## Persistence Rules
- Every round must write a persistent log entry under `memory/logs/` for attempts, results, and errors.
- Every round must end with a Git commit so the repository returns to a clean, recoverable state.
- Every handoff must assume context will expire; status, decisions, failures, and the next step must be written to disk before the agent stops.
