import asyncio
import os
from pathlib import Path

import pytest

from mosaic.core.config import load_config
from mosaic.core.events import ActionExecuted, EnvironmentChanged, read_events
from mosaic.experiments.runner import ExperimentRunner
from mosaic.providers.fake import FakeModelProvider


@pytest.mark.alem
@pytest.mark.skipif(os.environ.get("MOSAIC_TEST_ALEM") != "1", reason="Optional Alem CPU smoke")
def test_actual_alem_uses_same_runner(tmp_path: Path) -> None:
    from mosaic.environments.alem.adapter import AlemEnvironment

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "alem-smoke.yaml")
    environment = AlemEnvironment(config.agent.agent_id, config.max_steps)
    result = asyncio.run(
        ExperimentRunner(environment, FakeModelProvider("first_available")).run(
            config, tmp_path / "alem"
        )
    )
    assert result.summary.status == "failed", result.summary.reason
    assert result.summary.budget.model_calls == result.summary.budget.environment_actions == 1
    assert not environment.verify("collect_wood").succeeded
    events = read_events(result.trace_path)
    executed = [e.payload for e in events if isinstance(e.payload, ActionExecuted)]
    assert len(executed) == 1 and executed[0].action.name == "Noop"
    assert any(isinstance(e.payload, EnvironmentChanged) for e in events)
