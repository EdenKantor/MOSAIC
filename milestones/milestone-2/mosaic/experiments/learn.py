import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from mosaic.core.config import ExperimentConfig, ModelConfig, load_learning_config
from mosaic.environments.base import EnvironmentAdapter
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.learning import LearningRunner
from mosaic.providers.base import ModelProvider
from mosaic.providers.fake import FakeModelProvider


def environment_factory(config: ExperimentConfig) -> EnvironmentAdapter:
    if config.environment.name == "fake":
        return FakeEnvironment(config.agent.agent_id)
    from mosaic.environments.alem.adapter import AlemEnvironment

    return AlemEnvironment(config.agent.agent_id, config.max_steps)


def provider_factory(config: ModelConfig) -> ModelProvider:
    return FakeModelProvider(config.policy, config.failed_calls)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire, validate and share one bounded procedure"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_learning_config(args.config)
        output = args.output_dir or Path("runs") / config.experiment_id / str(uuid4())
        result = asyncio.run(
            LearningRunner(environment_factory, provider_factory, provider_factory).run(
                config, output
            )
        )
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print(f"Cannot start skill experiment: {error}", file=sys.stderr)
        return 2
    print(f"{result.summary.status}: {result.summary.reason}")
    print(f"Skill: {result.summary.skill_id}; promoted: {result.summary.promoted}")
    print(
        f"All phases: weak calls {result.summary.budget.weak.model_calls}; "
        f"teacher calls {result.summary.budget.teacher.model_calls}; "
        f"actions {result.summary.budget.environment_actions}"
    )
    print("Fake token counts are synthetic. Compilation is a measured local operation.")
    print(f"Trace: {result.trace_path.resolve()}")
    print(f"Summary: {result.summary_path.resolve()}")
    print(f"PlayBook: {result.playbook_path.resolve()}")
    return {"succeeded": 0, "failed": 1, "error": 2}[result.summary.status]


if __name__ == "__main__":
    raise SystemExit(main())
