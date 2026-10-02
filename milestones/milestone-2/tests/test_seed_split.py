import asyncio
from pathlib import Path

from mosaic.core.config import load_config, load_learning_config
from mosaic.core.events import ExperimentStarted, read_events
from mosaic.environments.fake import FakeEnvironment
from mosaic.experiments.learn import environment_factory, provider_factory
from mosaic.experiments.learning import LearningRunner
from mosaic.skills.models import CandidateSkill, project_observation

PROJECT = Path(__file__).resolve().parents[1]


def test_default_phase_seeds_are_disjoint_but_starting_conditions_are_seen() -> None:
    config = load_learning_config(PROJECT / "configs" / "skill-pilot.yaml")
    acquisition = {config.acquisition.seed}
    validation = set(config.validation.seeds)
    reuse = set(config.reuse.seeds)
    assert acquisition == {42} and validation == {19} and reuse == {20}
    assert acquisition.isdisjoint(validation | reuse) and validation.isdisjoint(reuse)
    standalone = load_config(PROJECT / "configs" / "skill-reuse.yaml")
    assert standalone.seed in reuse
    assert standalone.model == config.acquisition.model
    assert standalone.teacher is None and standalone.routing is None

    visible_conditions = []
    for seed in (config.acquisition.seed, *config.validation.seeds, *config.reuse.seeds):
        environment = FakeEnvironment(config.acquisition.agent.agent_id)
        observation = environment.reset(seed, config.acquisition.task)
        visible_conditions.append(
            project_observation(observation.text, config.compilation.observation_fields)
        )
    # Different random seeds produce the same task conditions in this tiny fixture.
    # Holding out seed IDs consequently demonstrates direct reuse, not generalization.
    assert len(set(visible_conditions)) == 1


def test_default_pipeline_records_each_phase_seed_and_source_provenance(tmp_path: Path) -> None:
    config = load_learning_config(PROJECT / "configs" / "skill-pilot.yaml")
    output = tmp_path / "learn"
    result = asyncio.run(
        LearningRunner(environment_factory, provider_factory, provider_factory).run(config, output)
    )
    assert result.summary.status == "succeeded" and result.summary.budget_complete
    assert result.summary.validation is not None
    assert [trial.seed for trial in result.summary.validation.trials] == [19]
    candidate = CandidateSkill.model_validate_json((output / "candidate.json").read_text())
    assert candidate.provenance.source_seed == 42
    paths = (
        output / "acquisition" / "events.jsonl",
        output / "validation" / "19" / "events.jsonl",
        output / "reuse" / "20" / "events.jsonl",
    )
    configs = [
        next(
            event.payload.config
            for event in read_events(path)
            if isinstance(event.payload, ExperimentStarted)
        )
        for path in paths
    ]
    assert [phase.seed for phase in configs] == [42, 19, 20]
    assert configs[0].teacher is not None
    assert all(phase.teacher is None and phase.routing is None for phase in configs[1:])
    assert result.summary.budget.weak.model_calls == 34
    assert result.summary.budget.teacher.model_calls == 16
