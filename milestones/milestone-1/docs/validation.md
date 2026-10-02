# Milestone 1 validation report

Validation date: 2026-10-02. Windows, Python 3.12.14. Each milestone used its own editable
installation and locked dependencies. No real model calls or model-weight downloads occurred.

## Implemented

Bounded routing-only baseline B3, separate teacher capability, visible episode-local history,
schema-2 routing events, separate weak/teacher accounting, deterministic configurations and
scientific-invariant tests. Milestone 0 remains unchanged in its own folder. No new dependencies.

## Routing flow

Start weak. A configured consecutive-execution-failure threshold or finite window without
verified goal success ends the attempt. Run the configured number of bounded weak recovery
windows. After another failure, make teacher escalation eligible. If available, the teacher
chooses one action per call for the remaining episode. Every decision consumes the same total
step cap. Success, environment termination/truncation, no available actions or total exhaustion
stop further calls. Transport errors end the run explicitly.

## Teacher isolation

`TeacherAccess` is AVAILABLE or WITHDRAWN. The capability privately holds the provider only while
available; withdrawal drops the reference. The runner checks availability before each new call
and the capability checks it again. Blocked calls are explicit events and consume zero teacher
calls. Tests cover initial withdrawal, direct capability invocation and withdrawal during takeover.

Requests contain visible observation, task, advertised actions and bounded visible feedback.
Hidden-state and verifier-evidence canaries remain in audit traces and never enter teacher
requests. Reusing a runner/provider starts the next episode with empty history. Prior traces
are never read by the runtime. This gate does not cancel an already dispatched remote request
or sandbox arbitrary Python; scheduled withdrawal experiments belong to a later milestone.

## Budget accounting

Each invocation has a role, provider/model identity, usage provenance and measured latency.
Weak and teacher call/token/latency groups reconcile with generic invocation events. Unknown
tokens remain null; known subtotals remain inspectable. Failed calls count with unknown usage
and measured latency. Environment actions count actual dispatch attempts. Teacher-call fraction
is null when total calls are zero. Synthetic fixture tokens are never interpreted as API cost.

## Tests run

The following are the exact PowerShell commands used for final core checks. From
`C:\Users\ngede\Documents\Codex\2026-10-02\g\outputs\MOSAIC\milestones\milestone-1`:

```powershell
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-1-venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider --basetemp 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\pytest-m1-final-20261002b'
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-1-venv\Scripts\python.exe' -m ruff check mosaic tests
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-1-venv\Scripts\python.exe' -m ruff format --check mosaic tests
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-1-venv\Scripts\python.exe' -m mypy mosaic tests
```

Results: **41 passed, 2 skipped in 8.11 seconds**; lint passed; 26 files already formatted;
strict type checking passed for 26 source files. The skipped tests are the optional Alem tests.
The suite covers all nine requested invariants and provider-error, step-cap, configuration,
hidden-context and runtime-withdrawal cases.

From the preserved `milestones\milestone-0` folder:

```powershell
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-0-final-venv\Scripts\python.exe' -m pytest -q -p no:cacheprovider --basetemp 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\pytest-m0-relocated-20261002a'
```

Result: **26 passed, 1 skipped in 4.89 seconds**. The original suite also passed before changes.

Real Alem checks used the separately installed pinned dependency in the Milestone 1 directory:

```powershell
$env:MOSAIC_TEST_ALEM='1'
$env:JAX_PLATFORM_NAME='cpu'
$env:MPLCONFIGDIR='C:\Users\ngede\Documents\Codex\2026-10-02\g\work\m1-matplotlib'
& 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\milestone-1-alem-venv\Scripts\python.exe' -m pytest -q -m alem -p no:cacheprovider --basetemp 'C:\Users\ngede\Documents\Codex\2026-10-02\g\work\pytest-m1-alem-20261002a'
```

Result: **2 passed, 40 deselected in 92.63 seconds**. This ran before the additional dynamic
withdrawal test was added; that test is covered by the final core checks above. Dedicated fresh
workspace temporary directories avoided the host's inaccessible default temporary test directory.

Portable equivalents after `uv sync --locked`: `uv run --locked python -m pytest -q`,
`uv run --locked ruff check mosaic tests`, `uv run --locked ruff format --check mosaic tests`,
and `uv run --locked python -m mypy mosaic tests`. Set the Alem environment variables and use
`uv sync --locked --extra alem --python 3.12` for optional tests.

## Example run

`configs/routing.yaml`, seed 42: two weak waits exhaust the one-turn attempt and one-turn recovery.
The teacher then executes the original 16-action pump repair. The verified result is success:
**2 weak calls, 16 teacher calls, 18 actions, 1 escalation, 0 blocked escalations**.
Teacher-call fraction is 16/18. The weak-success case uses 16 weak calls and zero teacher calls;
recovery success uses 17 weak calls and zero teacher calls. Initial withdrawal blocks escalation
after two weak calls with zero teacher calls. These counts are asserted in integration tests.

## Trace

The CLI writes a fresh `events.jsonl` and `summary.json` under the selected output directory.
`WeakAttemptStarted`, `WeakRecoveryStarted`, `TeacherEscalationEligible` (including objective
reason/counters), `TeacherEscalated`, role-tagged `ModelCalled` / `ModelResponded`, and
`TaskSucceeded` reconstruct the routing example. `TeacherInvocationBlocked` explains withdrawal.
The first event records configuration, source fingerprints, implementations and dependencies.

## Alem status

The unchanged adapter uses official Alem 0.2.1 at commit
`14d412e5ee961f9c43d6ce92ee05fee9cd1efc5e`. Tests executed its original one-Noop smoke and
the new two-weak/one-teacher Noop routing smoke on Windows CPU with JAX 0.4.38. Expected
environment-verified failure in both cases proves integration without claiming task intelligence.

## Real model status

Ollama was unavailable. No real provider was exercised. The async provider protocol is sufficient
for an external adapter, but configuration and the CLI currently select only fake fixtures. A
real adapter must preserve explicit model identity, visible context and truthful unknown usage.

## Scientific concerns

Scripted policies do not establish real weak/strong capability or savings. A finite success
deadline can escalate while the weak agent is making legitimate progress. Register and sweep
these limits for later studies. Match prompts, history, seeds, tasks and inference budgets in
comparisons. The decision cap does not enforce lifetime token spending. Teacher takeover can
consume most remaining calls. No persistent learning or retention claim is supported.

## Next recommended step

Milestone 2: Teacher → Candidate Skill → Weak-Agent Validation → Shared PlayBook, after review
of this baseline. It has not been implemented.
