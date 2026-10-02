import re
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_core_ci_checks_every_completed_milestone_without_optional_environment_install() -> None:
    workflow = cast(
        dict[str, Any], yaml.safe_load((ROOT / ".github" / "workflows" / "core.yml").read_text())
    )
    assert set(workflow["on"]) == {"push", "pull_request"}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["core"]
    assert job["strategy"]["matrix"]["milestone"] == [
        "milestone-0",
        "milestone-1",
        "milestone-1.25",
        "milestone-2",
    ]
    assert job["strategy"]["fail-fast"] is False
    assert job["defaults"]["run"]["working-directory"] == "milestones/${{ matrix.milestone }}"
    for milestone in job["strategy"]["matrix"]["milestone"]:
        assert (ROOT / "milestones" / milestone / "pyproject.toml").is_file()
        assert (ROOT / "milestones" / milestone / "uv.lock").is_file()
    actions = [step for step in job["steps"] if "uses" in step]
    assert {step["uses"].split("@", 1)[0] for step in actions} == {
        "actions/checkout",
        "actions/setup-python",
        "astral-sh/setup-uv",
    }
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"]) for step in actions)
    checkout = next(step for step in actions if step["uses"].startswith("actions/checkout@"))
    assert checkout["with"]["persist-credentials"] is False
    python = next(step for step in actions if step["uses"].startswith("actions/setup-python@"))
    assert python["with"]["python-version"] == "3.12"

    commands = [step["run"] for step in job["steps"] if "run" in step]
    sync = next(command for command in commands if command.startswith("uv sync "))
    assert "--locked" in sync and "--all-extras" not in sync and "--extra" not in sync
    assert all(
        "jax" not in command.lower() and "alem" not in command.lower() for command in commands[:-1]
    )
    assert any("ruff format --check mosaic tests" in command for command in commands)
    assert any("ruff check mosaic tests" in command for command in commands)
    assert any("mypy mosaic tests" in command for command in commands)
    tests = next(command for command in commands if "pytest" in command)
    assert '-m "not alem"' in tests
    assert all(
        "||" not in command and "continue-on-error" not in step
        for step in job["steps"]
        for command in [step.get("run", "")]
    )
