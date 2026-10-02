# MOSAIC

Reproducible research on durable agent capability under a fixed lifetime inference budget.

Each milestone lives in a separate, independently runnable folder and receives its own Git
commit. Earlier milestone folders are preserved when later milestones are implemented.

| Folder | Scope | Status |
| --- | --- | --- |
| [milestones/milestone-0](milestones/milestone-0/README.md) | Deterministic single-agent vertical slice | Implemented and tested |
| [milestones/milestone-1](milestones/milestone-1/README.md) | Bounded teacher escalation, routing-only baseline B3 | Implemented and tested |
| [milestones/milestone-2](milestones/milestone-2/README.md) | Candidate procedures, weak validation and shared PlayBook | Implemented and tested |

Run the validated-procedure experiment from its directory:

```sh
cd milestones/milestone-2
uv sync --locked
uv run --locked python -m mosaic.experiments.learn --config configs/skill-pilot.yaml --output-dir runs/skill-pilot
```

Each folder has its own package, lockfile, configurations, tests, documentation and run outputs.
Do not combine the milestone packages into one Python environment.
