import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from mosaic.core.config import TeacherAccess, load_calibration_config, load_config
from mosaic.core.models import ModelRequest, ModelResponse
from mosaic.core.teacher import TeacherCapability, TeacherUnavailable
from mosaic.experiments.calibration import CalibrationRunner, calibration_environment, preflight
from mosaic.experiments.runner import ExperimentRunner
from mosaic.providers.factory import provider_factory
from mosaic.providers.http import ProviderError, redact


class _UnbackedTeacher:
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise TeacherUnavailable("Teacher access is withdrawn")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a MOSAIC episode or independent paired calibration"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--calibration-config", type=Path)
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-pilot-reviewed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.calibration_config:
            calibration = load_calibration_config(args.calibration_config)
            if args.preflight:
                readiness = preflight(calibration)
                for check in readiness.checks:
                    print(f"{'passed' if check.passed else 'failed'}: {check.name}: {check.detail}")
                print("Preflight made zero remote inference calls and zero Groq network requests.")
                return 0 if readiness.passed else 2
            if calibration.stage == "expanded" and not args.confirm_pilot_reviewed:
                print(
                    "Expanded calibration requires --confirm-pilot-reviewed after pilot review.",
                    file=sys.stderr,
                )
                return 2
            output = args.output_dir or Path("runs") / calibration.calibration_id / str(uuid4())
            report = asyncio.run(
                CalibrationRunner().run(
                    calibration, output, confirm_pilot_reviewed=args.confirm_pilot_reviewed
                )
            )
            print(
                f"{report.status}: {len(report.pairs)} independent pairs; "
                f"{report.excluded_pairs} excluded"
            )
            print(
                f"Calls: {report.total_budget.model_calls}; "
                f"actions: {report.total_budget.environment_actions}"
            )
            print(
                "Fake-only synthetic fixture."
                if report.fixture_only
                else (
                    "Provider-reported usage; missing dimensions "
                    "and provider prices remain unknown."
                )
            )
            print(f"Report: {(output / 'report.md').resolve()}")
            print(f"Summary: {(output / 'summary.json').resolve()}")
            return 0 if report.status == "completed" else 2
        if args.preflight or args.confirm_pilot_reviewed:
            raise ValueError("Calibration flags require --calibration-config")
        config = load_config(args.config)
        environment = calibration_environment(config)
        provider = provider_factory(config.model)
        teacher = None
        if config.teacher:
            teacher = TeacherCapability(
                _UnbackedTeacher()
                if config.teacher.access == TeacherAccess.WITHDRAWN
                else provider_factory(config.teacher.model),
                config.teacher.access,
            )
        output = args.output_dir or Path("runs") / config.experiment_id / str(uuid4())
        result = asyncio.run(ExperimentRunner(environment, provider, teacher).run(config, output))
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        detail = (
            redact(str(error))
            if isinstance(error, ProviderError)
            else "Configuration or runtime setup failed"
        )
        if isinstance(error, ValidationError):
            detail = "Configuration validation failed"
        print(f"Cannot start experiment: {detail}", file=sys.stderr)
        return 2
    print(f"{result.summary.status}: {result.summary.reason}")
    print(
        f"Weak calls: {result.summary.budget.weak.model_calls}; "
        f"teacher calls: {result.summary.budget.teacher.model_calls}; "
        f"actions: {result.summary.budget.environment_actions}"
    )
    print(
        f"Escalations: {result.summary.escalations}; blocked: {result.summary.blocked_escalations}"
    )
    print(
        "Fake-provider token counts are synthetic fixture units."
        if config.model.provider == "fake"
        else "Provider-reported token counts; missing dimensions remain unknown."
    )
    print(f"Trace: {result.trace_path.resolve()}")
    print(f"Summary: {result.summary_path.resolve()}")
    return {"succeeded": 0, "failed": 1, "error": 2}[result.summary.status]


if __name__ == "__main__":
    raise SystemExit(main())
