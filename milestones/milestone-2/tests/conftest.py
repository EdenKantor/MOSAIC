from pathlib import Path

import pytest

from mosaic.core.config import ExperimentConfig, load_config


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "pilot.yaml")


@pytest.fixture
def routing_config() -> ExperimentConfig:
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "routing.yaml")
