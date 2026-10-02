import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mosaic.core.budget import BudgetLedger
from mosaic.core.models import Action, Observation
from mosaic.skills import models
from mosaic.skills.models import (
    CandidateSkill,
    SkillProvenance,
    SkillScope,
    ValidationReport,
    ValidationTrial,
    VerifiedSkill,
    candidate_content_sha256,
    project_observation,
)
from mosaic.skills.playbook import PlayBook


def candidate(*, action: str = "move_east", episode: str = "source-1") -> CandidateSkill:
    scope = SkillScope(environment="fake", revision="1", task="repair_pump")
    fields = ("position", "has_part")
    initiation = '{"has_part":false,"position":0}'
    procedure = (Action(name=action),)
    tools = (action,)
    digest = models.candidate_content_sha256(
        scope, fields, initiation, procedure, tools, scope.task
    )
    return CandidateSkill(
        skill_id=f"skill-{digest[:16]}",
        scope=scope,
        observation_fields=fields,
        initiation_json=initiation,
        procedure=procedure,
        allowed_tools=tools,
        termination_goal=scope.task,
        provenance=SkillProvenance(
            source_episode=episode,
            source_agent="source-agent",
            source_seed=42,
            source_provider="fake",
            source_model="teacher-fixture",
            trace_sha256="a" * 64,
            config_sha256="b" * 64,
            mosaic_commit=None,
        ),
        content_sha256=digest,
    )


def report(skill: CandidateSkill, **changes: Any) -> ValidationReport:
    trial = ValidationTrial(
        seed=43,
        config_sha256="c" * 64,
        trace_sha256="d" * 64,
        provider="fake",
        model="weak-fixture",
        status="succeeded",
        reason="environment_verified",
        skill_completed=True,
        budget=BudgetLedger().totals(0),
    )
    data = {
        "skill_id": skill.skill_id,
        "version": skill.version,
        "content_sha256": skill.content_sha256,
        "trials": [trial.model_dump(mode="json")],
    }
    return ValidationReport.model_validate_json(json.dumps({**data, **changes}))


def changed(skill: CandidateSkill, **changes: Any) -> CandidateSkill:
    return CandidateSkill.model_validate_json(
        json.dumps({**skill.model_dump(mode="json"), **changes})
    )


def test_visible_projection_is_canonical_and_ignores_only_undeclared_fields() -> None:
    assert project_observation(
        '{"steps":99,"position":0,"has_part":false}', ("position", "has_part")
    ) == ('{"has_part":false,"position":0}')
    skill = candidate()
    observation = Observation(
        agent_id="another-agent", text='{"has_part":false,"position":0,"steps":9}'
    )
    assert skill.matches(skill.scope, observation)
    assert not skill.matches(
        skill.scope, observation.model_copy(update={"text": '{"has_part":true,"position":0}'})
    )


@pytest.mark.parametrize(
    "text,fields",
    [
        ("[]", ("position",)),
        ("{}", ("position",)),
        ('{"position":0,"position":1}', ("position",)),
        ('{"position":NaN}', ("position",)),
        ('{"position":Infinity}', ("position",)),
        ('{"position":0}', ()),
        ('{"position":0}', ("position", "position")),
        ('{"position":0}', (" ",)),
    ],
)
def test_invalid_projection_fails_closed(text: str, fields: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        project_observation(text, fields)
    assert not candidate().matches(candidate().scope, Observation(agent_id="agent", text=text))


@pytest.mark.parametrize(
    "change",
    [
        {"content_sha256": "f" * 64},
        {"skill_id": "skill-" + "0" * 16},
        {"initiation_json": '{ "has_part": false, "position": 0 }'},
        {"initiation_json": '{"extra":1,"has_part":false,"position":0}'},
        {"observation_fields": ["position", "position"]},
        {"observation_fields": []},
        {"procedure": []},
        {"procedure": [{"name": "wait"}] * 257},
        {"allowed_tools": ["move_east", "move_east"]},
        {"allowed_tools": ["repair"]},
        {"termination_goal": "different_task"},
    ],
)
def test_candidate_integrity_and_bounds(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        changed(candidate(), **change)


def test_content_identity_excludes_provenance() -> None:
    first, second = candidate(), candidate(episode="source-2")
    assert first.provenance != second.provenance
    assert first.skill_id == second.skill_id
    assert first.content_sha256 == second.content_sha256
    assert first.content_sha256 == candidate_content_sha256(
        first.scope,
        first.observation_fields,
        first.initiation_json,
        first.procedure,
        first.allowed_tools,
        first.termination_goal,
    )


@pytest.mark.parametrize(
    "status,completed", [("failed", True), ("error", True), ("succeeded", False)]
)
def test_failed_or_incomplete_validation_cannot_promote(
    tmp_path: Path, status: str, completed: bool
) -> None:
    skill = candidate()
    trial = report(skill).trials[0].model_dump(mode="json")
    trial.update(status=status, skill_completed=completed)
    validation = report(skill, trials=[trial])
    assert not validation.passed
    book = PlayBook(tmp_path / "playbook.json")
    with pytest.raises(ValidationError):
        book.promote(skill, validation)
    assert book.entries() == () and book.fingerprint() is None


def test_validation_rejects_teacher_calls_duplicate_seeds_and_missing_trials() -> None:
    skill = candidate()
    trial = report(skill).trials[0].model_dump(mode="json")
    ledger = BudgetLedger()
    ledger.model_called("teacher")
    with pytest.raises(ValueError, match="teacher"):
        ValidationTrial.model_validate_json(
            json.dumps({**trial, "budget": ledger.totals(1).model_dump(mode="json")})
        )
    with pytest.raises(ValueError, match="unique"):
        report(skill, trials=[trial, trial])
    with pytest.raises(ValueError):
        report(skill, trials=[])


@pytest.mark.parametrize(
    "change", [{"skill_id": "skill-" + "0" * 16}, {"version": 2}, {"content_sha256": "f" * 64}]
)
def test_report_must_match_the_candidate(change: dict[str, Any]) -> None:
    skill = candidate()
    with pytest.raises(ValueError, match="match"):
        VerifiedSkill(candidate=skill, validation=report(skill, **change))


def test_store_reopens_and_retrieves_for_another_agent_only_in_matching_scope(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shared" / "playbook.json"
    skill = candidate()
    assert PlayBook(path).fingerprint() is None
    assert PlayBook(path).promote(skill, report(skill))
    book = PlayBook(path)
    assert book.entries() == (VerifiedSkill(candidate=skill, validation=report(skill)),)
    observation = Observation(agent_id="different-agent", text=skill.initiation_json)
    assert book.retrieve(skill.scope, observation) == book.entries()[0]
    for field in ("task", "environment", "revision"):
        scope = skill.scope.model_copy(update={field: "different"})
        assert book.retrieve(scope, observation) is None
    assert book.retrieve(skill.scope, observation.model_copy(update={"text": "bad JSON"})) is None
    assert book.fingerprint() is not None


def test_identical_reacquisition_preserves_first_evidence(tmp_path: Path) -> None:
    book = PlayBook(tmp_path / "playbook.json")
    first, second = candidate(), candidate(episode="source-2")
    assert book.promote(first, report(first))
    fingerprint = book.fingerprint()
    assert not book.promote(second, report(second))
    assert book.entries()[0].candidate.provenance.source_episode == "source-1"
    assert book.fingerprint() == fingerprint


def test_retrieval_order_is_deterministic(tmp_path: Path) -> None:
    book = PlayBook(tmp_path / "playbook.json")
    first, second = candidate(), candidate(action="wait")
    for skill in sorted((first, second), key=lambda item: item.skill_id, reverse=True):
        book.promote(skill, report(skill))
    retrieved = book.retrieve(
        first.scope, Observation(agent_id="agent", text=first.initiation_json)
    )
    assert retrieved is not None
    assert retrieved.candidate.skill_id == min(first.skill_id, second.skill_id)


def test_truncated_identity_collision_cannot_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = models.candidate_content_sha256

    def collision(
        scope: SkillScope,
        fields: tuple[str, ...],
        initiation: str,
        procedure: tuple[Action, ...],
        tools: tuple[str, ...],
        goal: str,
    ) -> str:
        return "0" * 16 + original(scope, fields, initiation, procedure, tools, goal)[16:]

    monkeypatch.setattr(models, "candidate_content_sha256", collision)
    book = PlayBook(tmp_path / "playbook.json")
    first, second = candidate(), candidate(action="wait")
    assert first.skill_id == second.skill_id and first.content_sha256 != second.content_sha256
    book.promote(first, report(first))
    fingerprint = book.fingerprint()
    with pytest.raises(ValueError, match="collision"):
        book.promote(second, report(second))
    assert book.fingerprint() == fingerprint


@pytest.mark.parametrize(
    "contents",
    [
        "not JSON",
        "{}",
        '{"schema_version":2,"entries":[]}',
        '{"schema_version":1,"schema_version":1,"entries":[]}',
        '{"schema_version":1,"entries":[],"unexpected":true}',
    ],
)
def test_corrupt_document_is_not_silently_replaced(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "playbook.json"
    path.write_text(contents, encoding="utf-8")
    book = PlayBook(path)
    with pytest.raises(ValueError):
        book.entries()
    with pytest.raises(ValueError):
        book.promote(candidate(), report(candidate()))
    assert path.read_text(encoding="utf-8") == contents


def test_corrupt_evidence_digest_and_duplicate_records_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "playbook.json"
    skill = candidate()
    book = PlayBook(path)
    book.promote(skill, report(skill))
    original = path.read_text(encoding="utf-8")
    for mutation in ("digest", "validation", "duplicate"):
        document = json.loads(original)
        if mutation == "digest":
            document["entries"][0]["candidate"]["content_sha256"] = "f" * 64
        elif mutation == "validation":
            document["entries"][0]["validation"]["trials"][0]["status"] = "failed"
        else:
            document["entries"].append(document["entries"][0])
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError):
            book.entries()


def test_invalid_model_copy_is_revalidated_before_persistence(tmp_path: Path) -> None:
    skill = candidate()
    broken = skill.model_copy(update={"procedure": ()})
    book = PlayBook(tmp_path / "playbook.json")
    with pytest.raises(ValueError):
        book.promote(broken, report(broken))
    assert book.fingerprint() is None


def test_atomic_write_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = PlayBook(tmp_path / "playbook.json")
    first, second = candidate(), candidate(action="wait")
    book.promote(first, report(first))
    fingerprint = book.fingerprint()

    def fail_replace(source: object, target: object) -> None:
        raise OSError("Injected replacement failure")

    monkeypatch.setattr("mosaic.skills.playbook.os.replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        book.promote(second, report(second))
    assert book.fingerprint() == fingerprint
    assert len(book.entries()) == 1
    assert not tuple(tmp_path.glob(".playbook-*.tmp"))
