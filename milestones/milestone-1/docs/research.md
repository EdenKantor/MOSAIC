# Research scope

Research question: under a fixed lifetime inference budget, can weak agents convert temporary
access to a strong teacher into validated shared procedures that preserve useful capability
after the teacher is completely disabled?

The hypothesis is that some teacher-assisted trajectories can become **Durable Capability
Capital**: procedural assets whose cumulative useful reuse exceeds their acquisition and
maintenance costs. The hypothesis can fail.

Milestone 1 implements the B3 routing-only control using scripted weak and teacher providers.
It preserves the weak-only path and adds bounded attempt/recovery windows, runtime teacher
withdrawal gates and role-separated accounting. It has no skills, population, persistent learning,
promotion policy or scheduled withdrawal experiment. The fixture cannot establish language-model
capability, retained learning, or comparative inference savings.

## Eventual experimental intervention

At a predefined boundary, teacher access must become impossible in the runtime. Withdrawal
must cover task solving, skill repair, compilation, validation, retrieval, summarization,
planning and curation. Prompt instructions alone are insufficient enforcement.

Evaluate seen tasks, near transfer, skill composition and structural novelty separately.
Structural novelty failure is expected to be informative; retained skills need not solve it.

## Baseline status

| Baseline | Condition | Status |
| --- | --- | --- |
| B0 | Weak agents without persistent procedures | Weak-only runner; scripted fixture |
| B1 | Weak agents with inference matched to memory-system overhead | Planned |
| B2 | Teacher only | Planned |
| B3 | Weak-to-teacher routing without learning | Implemented; scripted fixture |
| B4 | Static teacher-derived memory | Planned |
| B5 | Shared procedures with fixed promotion rules | Planned |
| B6 | Validated procedures and a cost-aware Skill Investment Gate | Planned |

## Measurement and falsification

Current observations support task success/failure, call counts, action counts, token subtotals,
unknown usage, teacher-call fraction, escalation counts and elapsed time by role. Fake token
counts are synthetic fixture units. No retention,
skill value or cost-savings metrics are calculated without the underlying experimental data.

Later compare post-/pre-withdrawal success within each transfer category, teacher calls divided
by all model calls, complete inference cost per successful task, negative transfer, and skill
acquisition/maintenance economics. Keep models and tokenizers identifiable; do not collapse
raw dimensions into a universal cost scalar.

The current router treats a finite goal-success deadline as bounded failure; an accepted action
can still make legitimate progress without meeting that deadline. Register and sweep these limits
before interpreting routing quality. Match prompts, visible history, seeds, tasks and inference
budgets across later B0/B3 comparisons. The current decision cap does not enforce a token budget.

The hypothesis is weakened or falsified if retained performance vanishes after withdrawal,
does not beat matched-budget baselines, depends on hidden teacher access or evaluation leakage,
or fails to repay memory overhead across realistic reuse frequencies. Pre-register budgets,
tasks and evaluation splits before interpreting results. Use repeated seeds and appropriate
uncertainty estimates for later stochastic model experiments.
