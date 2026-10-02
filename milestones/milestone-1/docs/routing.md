# Routing Only: baseline B3

Milestone 1 implements controlled weak-to-teacher routing without persistent learning.
The teacher remains available unless the experiment explicitly starts with withdrawn access.
No skill, memory store, teaching explanation, reflection, learned router or UI is implemented.

## Exact policy

Start with the weak provider. Each weak decision consumes one turn, including rejected or
malformed proposals. The environment remains authoritative about action acceptance and goal
success. Two independent limits can end a weak window:

- A configurable consecutive-execution-failure threshold, counting parser rejections and
  environment-rejected actions. An accepted action clears that consecutive-failure counter.
- Exhausting the configured number of decisions without environment-verified goal success.

Neither signal is model confidence. An unsuccessful individual action is allowed while the
window still has capacity. There is no invented domain-independent progress metric.

When the initial window ends, begin a bounded weak recovery window if a configured recovery
remains. Reset phase counters, retain only the configured bounded visible episode history,
and continue using the same weak model. When all recovery windows are exhausted, escalation
becomes eligible. Default recovery-attempt count is one.

If the teacher is available, hand control of the remaining episode to it. It proposes one
advertised action per call; the same parser and environment perform execution and verification.
Teacher failures cannot cause further escalation or unbounded retries. Every decision still
consumes the shared `max_steps` limit. Goal success, environment termination/truncation, no
available actions, or exhausted total steps stop execution before another teacher call.
Transport errors end the run with an explicit error rather than pretending they are task failures.

The `routing.yaml` fixture has one initial weak turn and one recovery turn. Both choose an
advertised `wait`; the verifier remains false at both finite deadlines. The teacher then follows
the existing pump fixture's 16-action solution. This is exhausted bounded progress, not a claim
that Alem or the fake simulator treats every unsuccessful action as invalid. A separate test
checks actual environment-rejected weak actions and escalation by the failure threshold.

## Teacher isolation

`TeacherCapability` holds the teacher provider privately. `withdraw()` drops that reference and
sets access to WITHDRAWN; there is no runtime re-enable method. `generate()` checks access
itself. The runner also checks access before reserving/logging an invocation. A blocked eligible
escalation emits `TeacherInvocationBlocked`, stops the episode and consumes no teacher call.
Initial runtime availability must match the saved configuration.

This enforces prevention of new invocations through the experiment runtime. It is not a sandbox
against arbitrary malicious Python code or cancellation of an already dispatched remote request.
Scheduled hard withdrawal during an experiment is not implemented in this milestone.

## Observable teacher context and no learning

Weak and teacher requests use the same action-selection prompt and contain the current visible
observation, task ID, advertised actions, sampling settings, step and a bounded tuple of visible
attempt feedback. Feedback contains proposed action, acceptance/rejection, reason and visible
post-action observation. Neither raw snapshots nor verifier evidence enter requests.

The runner constructs history, phase counters and accounting locally inside each `run()` call.
The stateless fake policies use only that request. Reusing the same runner starts with empty
history and reset environment state. Runtime code never reads prior trace files. Raw teacher
responses persist only for scientific audit, not as operational knowledge.

## Trace and accounting

Schema version 2 adds `WeakAttemptStarted`, `WeakRecoveryStarted`,
`TeacherEscalationEligible`, `TeacherEscalated`, `TeacherInvocationBlocked`, and
`ModelCallFailed`. Generic `ModelCalled` / `ModelResponded` events represent invocation start /
finish with explicit model role, provider/model identity, usage and measured latency. Action
events also identify the proposing role. No redundant teacher invocation events are needed.

The budget contains aggregate dimensions plus `weak` and `teacher` groups. Each group includes
calls, input/output tokens, known subtotals, unknown-call counts, synthetic-call counts and
model-call wall time. Failed calls retain unknown usage in their own role and record elapsed
time. Blocked invocations are not dispatched and do not become model calls.

The summary includes task, outcome, total calls, teacher-call fraction (null for zero calls),
escalations, blocked escalations, action count and total runner wall time. Role wall time is
time inside provider calls, not separately allocated environment execution time.

## Scientific limitations

All provided policies are fixtures, not real weak/strong language models. Synthetic token counts
cannot establish monetary savings. A teacher takeover may consume most remaining inference;
report its full usage rather than counting only the transition. Fixed deadlines may escalate
during legitimate multi-step progress, so pre-register and sweep those limits for later studies.

Future B0/B3 comparisons must match episode history, prompts, tasks, seeds, action limits and
inference budgets. Current `max_steps` is a decision cap, not enforced lifetime token spending.
The Alem smoke intentionally uses Noop for both providers and measures integration only.
No retention or knowledge-acquisition claim is supported by this milestone.
