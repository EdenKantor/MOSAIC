# Research scope

Research question: under a fixed lifetime inference budget, can weak agents convert temporary
access to a strong teacher into validated shared procedures that preserve useful capability
after the teacher becomes completely disabled?

The hypothesis is that some procedures become **Durable Capability Capital**: assets whose
cumulative useful reuse exceeds acquisition and maintenance cost. The hypothesis can fail.

Milestone 2's **engineering scaffold is complete; its scientific hypothesis is unvalidated**.
It implements verified teacher acquisition, deterministic extraction, independent weak-only
validation, promotion and scoped reuse. Deterministic fixtures demonstrate the plumbing,
validation gates, persistence and accounting. They do not demonstrate useful transfer,
real-model learning, economic value or retention after a global withdrawal intervention.

The default pilot acquires on seed 42, validates on seed 19 and reuses on seed 20, with a fresh
agent ID for reuse. All three have initial part location 7 and matching selected starting
conditions. Separate seeds are bookkeeping independence here; this remains seen-condition
reuse, not held-out generalization. A new seed alone does not define a transfer split.

## Prerequisites before a withdrawal study

The roadmap is M0 complete → M1 complete → M2 engineering scaffold complete → M1.25 measurement
and provider hardening → M1.5 real-model calibration → M1.75 protocol freeze → M2-R
real-trajectory skill semantics → M3 hard teacher withdrawal → M4 economics. The first three
completion labels describe engineering work; M2 has not established scientifically useful
skills. [The root roadmap](../../../README.md) records the sequence.

Calibrate real weak and teacher behavior before selecting skill semantics. Then freeze task
definitions, prompts, splits, budgets, baselines and outcome rules. The current compiler stores
only the teacher suffix and records its conditions at teacher takeover, but retrieval occurs
only at reset and validation also resets. Waiting fixtures mask that mismatch because their
prefix changes only excluded step metadata. Real weak progress may leave a takeover state
from which the suffix works but a reset state from which it cannot activate or succeed.

Two redesign options are documented, without implementation: a complete accepted reset-to-goal
workflow under initial conditions (preferred for Experiment 1), or dynamically retrieved
mid-subgoal teacher suffixes with defensible validation starting states. See [skill semantics](skills.md).
Calibration must precede choosing between them; current fixture success cannot resolve the
problem.

## Withdrawal and task categories

A later predefined boundary must make teacher access impossible for solving, repair,
compilation, validation, retrieval, summarization, planning and curation. The current compiler,
validator and retriever have no teacher dependency, and weak-only reuse receives no teacher.
The tested `TeacherCapability.withdraw()` remains an inference-gate primitive. A scheduled,
complete withdrawal experiment is future Milestone 3 after the calibration, protocol and
semantics prerequisites; it is not a result of this milestone.

Evaluate seen tasks, near transfer, composition and structural novelty separately. The exact
visible conditions currently refuse a changed part location. That refusal is a coverage limit,
not successful near transfer. No composition or structural-novelty capability is claimed.

## Baseline status

| Baseline | Condition | Status |
| --- | --- | --- |
| B0 | Weak agents without persistent procedures | Weak-only runner; scripted fixture |
| B1 | Extra weak inference matched to memory overhead | Planned |
| B2 | Teacher only | Planned |
| B3 | Weak-to-teacher routing without learning | Preserved routing configurations; scripted fixture |
| B4 | Static teacher-derived memory | Planned comparative baseline |
| B5 | Shared procedures with fixed promotion rules | Engineering scaffold; scripted seen-condition demo, scientifically unvalidated |
| B6 | Validated procedures and a cost-aware investment gate | Planned |

No comparative baseline study has been completed. Successful episodes with the
`follow_skill` fixture do not establish that a smaller language model benefits from a procedure.

## Measurement and falsification

Report acquisition, validation and reuse costs together. Traces support role-separated calls,
raw token dimensions and unknown usage, model latency, actions, compilation/retrieval counts
and local time, injected skill bytes, verification and complete-procedure use. Synthetic
fixture tokens cannot estimate API cost or compare tokenizers.

A deterministic compiler consumes local compute even with zero compiler model calls.
Every validation and reuse action consumes weak inference. Lower teacher reliance during reuse
does not establish net savings: acquisition and validation must be repaid over an explicit
reuse horizon. No avoided-cost, break-even or retention metric is invented without a matched
comparison and underlying data.

The finite router can escalate legitimate progress before goal success. Exact-condition
retrieval can reject useful procedures. Fixed all-pass validation covers only registered
seeds and does not establish robustness under distribution shift. Independent fixture seeds
with the same initial part location remain deliberate plumbing evidence, not a held-out
transfer evaluation. A teacher suffix valid at takeover is not necessarily valid at reset.

The hypothesis is weakened or falsified if performance vanishes after withdrawal, fails to
beat matched-budget baselines, relies on hidden teacher access or evaluation leakage, harms
weak execution, or cannot repay memory overhead under realistic reuse frequency. Register
tasks, seeds, conditions, budgets and transfer splits before interpreting real-model results.
The current decision cap does not enforce a lifetime token budget.

Core CI covers M0/M1/M2 on Python 3.12 without Alem or ordinary real-provider calls. It checks
deterministic engineering behavior rather than the scientific hypothesis.

The next recommended step is to **run real-model calibration** after measurement/provider
hardening.
