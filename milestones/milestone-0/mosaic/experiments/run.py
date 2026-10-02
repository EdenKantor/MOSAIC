import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from mosaic.core.config import load_config
from mosaic.environments.base import EnvironmentAdapter
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.runner import ExperimentRunner
from mosaic.providers.fake import FakeModelProvider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one reproducible MOSAIC episode")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        environment: EnvironmentAdapter
        if config.environment.name == "fake":
            environment = FakeEnvironment(config.agent.agent_id)
        else:
            from mosaic.environments.alem.adapter import AlemEnvironment

            environment = AlemEnvironment(config.agent.agent_id, config.max_steps)
        provider = FakeModelProvider(config.model.policy)
        output = args.output_dir or Path("runs") / config.experiment_id / str(uuid4())
        result = asyncio.run(ExperimentRunner(environment, provider).run(config, output))
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print(f"Cannot start experiment: {error}", file=sys.stderr)
        return 2
    print(f"{result.summary.status}: {result.summary.reason}")
    print(
        f"Model calls: {result.summary.budget.model_calls}; "
        f"actions: {result.summary.budget.environment_actions}"
    )
    print(f"Trace: {result.trace_path.resolve()}")
    print(f"Summary: {result.summary_path.resolve()}")
    return {"succeeded": 0, "failed": 1, "error": 2}[result.summary.status]


if __name__ == "__main__":
    raise SystemExit(main())
