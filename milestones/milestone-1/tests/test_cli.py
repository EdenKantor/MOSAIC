import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_cli_success_failure_and_bad_config(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    pilot = root / "configs" / "pilot.yaml"
    success = subprocess.run(
        [
            sys.executable,
            "-m",
            "mosaic.experiments.run",
            "--config",
            str(pilot),
            "--output-dir",
            str(tmp_path / "success"),
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert success.returncode == 0, success.stderr
    assert (tmp_path / "success" / "events.jsonl").exists()
    assert json.loads((tmp_path / "success" / "summary.json").read_text())["status"] == "succeeded"
    short = yaml.safe_load(pilot.read_text())
    short["max_steps"] = 1
    short_path = tmp_path / "short.yaml"
    short_path.write_text(yaml.safe_dump(short))
    failure = subprocess.run(
        [
            sys.executable,
            "-m",
            "mosaic.experiments.run",
            "--config",
            str(short_path),
            "--output-dir",
            str(tmp_path / "failure"),
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert failure.returncode == 1, failure.stderr
    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "mosaic.experiments.run",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert invalid.returncode == 2
