"""Unit tests for the Run/Experiment aggregates (invariants, no I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.domain.entities import Experiment, Run
from experiments.domain.exceptions import StateError, ValidationError
from experiments.domain.ids import ArtifactKey, ExperimentID, ParamKey, RunID, StorageKey
from experiments.domain.models import Artifact, mint
from experiments.domain.status import ExperimentStatus, RunStatus
from tests.conftest import RecordingSink


def _run(sink: RecordingSink) -> Run:
    return Run(run_id=RunID("r1"), experiment_id=ExperimentID("e1"), sink=sink)


def _artifact(key: str = "model") -> Artifact:
    return mint(
        Artifact,
        key=ArtifactKey(key),
        storage_key=StorageKey("deadbeef"),
        path=Path("/tmp/model"),
        size_bytes=4,
    )


def test_log_metric_auto_increments_per_key(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    run.log_metric("loss", 0.5)
    run.log_metric("loss", 0.4)
    run.log_metric("acc", 0.9)
    steps = [m[1].step for m in recording_sink.metrics]
    assert steps == [0, 1, 0]


def test_log_metric_honors_explicit_step(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    run.log_metric("loss", 0.5, step=42)
    assert recording_sink.metrics[0][1].step == 42


def test_log_metric_invalid_key_raises(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    with pytest.raises(ValidationError):
        run.log_metric("bad key", 1.0)


def test_write_once_param_idempotent(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    run.log_param("lr", 0.1)
    run.log_param("lr", 0.1)
    assert len(recording_sink.params) == 1


def test_write_once_param_conflict_raises(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    run.log_param("lr", 0.1)
    with pytest.raises(ValidationError):
        run.log_param("lr", 0.2)


def test_log_param_invalid_key_raises(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    with pytest.raises(ValidationError):
        run.log_param("../x", 1)


def test_set_tag_overwrites(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    run.set_tag("phase", "warmup")
    run.set_tag("phase", "train")
    assert [t[1].value for t in recording_sink.tags] == ["warmup", "train"]


def test_set_tag_invalid_key_raises(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    with pytest.raises(ValidationError):
        run.set_tag("../x", "v")


def test_log_artifact_records_and_forwards(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    art = _artifact()
    run.log_artifact(art, b"data")
    assert recording_sink.artifacts[0][1] is art
    assert recording_sink.artifacts[0][2] == b"data"


def test_finish_is_idempotent_and_guards_subsequent_writes(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    run.log_metric("loss", 0.5)
    run.finish()
    run.finish()
    assert recording_sink.statuses[0][1] is RunStatus.COMPLETED
    with pytest.raises(StateError):
        run.log_metric("loss", 0.5)


def test_finish_killed_via_explicit_status(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    run.finish(RunStatus.KILLED)
    assert recording_sink.statuses[0][1] is RunStatus.KILLED


def test_finish_illegal_transition_raises(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    with pytest.raises(StateError):
        run.finish(RunStatus.RUNNING)


def test_run_equality_by_id(recording_sink: RecordingSink) -> None:
    a = _run(recording_sink)
    b = Run(run_id=RunID("r1"), experiment_id=ExperimentID("other"), sink=recording_sink)
    assert a == b
    assert hash(a) == hash(b)


def test_experiment_complete_and_fail() -> None:
    exp = Experiment(experiment_id=ExperimentID("e1"), name="n")
    assert exp.status is ExperimentStatus.RUNNING
    exp.complete()
    assert exp.status is ExperimentStatus.COMPLETED
    with pytest.raises(StateError):
        exp.fail()


def test_experiment_fail_transition() -> None:
    exp = Experiment(experiment_id=ExperimentID("e1"), name="n")
    exp.fail()
    assert exp.status is ExperimentStatus.FAILED


def test_run_properties_expose_state(recording_sink: RecordingSink) -> None:
    run = _run(recording_sink)
    assert run.id == RunID("r1")
    assert run.experiment_id == ExperimentID("e1")
    assert run.status is RunStatus.RUNNING
    assert dict(run.params) == {}
    run.log_param("lr", 0.1)
    assert ParamKey("lr") in run.params


def test_experiment_properties() -> None:
    exp = Experiment(experiment_id=ExperimentID("e1"), name="resnet")
    assert exp.id == ExperimentID("e1")
    assert exp.name == "resnet"
    assert exp.status is ExperimentStatus.RUNNING
