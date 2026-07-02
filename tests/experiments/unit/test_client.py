"""Unit tests for the Client / contexts / _Sink driving layer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from experiments.adapters import InMemoryArtifactStore, InMemoryBackend
from experiments.client import (
    BufferPolicy,
    Client,
    _build_artifact,
    _Sink,
    _terminal_run_status,
    close_default_client,
    default_client,
)
from experiments.domain.exceptions import ValidationError
from experiments.domain.ids import ExperimentID, MetricKey, RunID
from experiments.domain.models import GitInfo, Metric, RunMetadata, mint
from experiments.domain.status import RunStatus

if TYPE_CHECKING:
    pass


class _CountingBackend(InMemoryBackend):
    def __init__(self) -> None:
        super().__init__()
        self.open_count = 0

    def open(self) -> None:
        self.open_count += 1


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _make_run(backend: InMemoryBackend, rid: str = "r1", exp_id: str = "e1") -> RunID:
    backend.create_run(mint(RunMetadata, id=RunID(rid), experiment_id=ExperimentID(exp_id)))
    return RunID(rid)


def test_from_store_preserves_concrete_type() -> None:
    backend = InMemoryBackend()
    client = Client.from_store(backend)
    assert client.metadata is backend
    assert client.artifacts is backend


def test_identity_guard_opens_composite_once() -> None:
    backend = _CountingBackend()
    Client.from_store(backend)
    assert backend.open_count == 1


def test_two_store_client_opens_both() -> None:
    meta = _CountingBackend()
    art = InMemoryArtifactStore()
    Client(meta, art)
    assert meta.open_count == 1


def test_close_is_idempotent() -> None:
    client = Client(_CountingBackend(), InMemoryArtifactStore())
    client.close()
    client.close()


def test_client_context_manager_closes_on_exit() -> None:
    backend = _CountingBackend()
    with Client.from_store(backend) as client:
        assert client.metadata is backend


def test_default_client_is_memoized(isolated_default: Path) -> None:
    first = default_client()
    second = default_client()
    assert first is second
    close_default_client()
    third = default_client()
    assert third is not first


@pytest.mark.parametrize(
    ("exc_type", "expected"),
    [
        (None, RunStatus.COMPLETED),
        (RuntimeError, RunStatus.FAILED),
        (KeyboardInterrupt, RunStatus.KILLED),
        (SystemExit, RunStatus.KILLED),
    ],
)
def test_terminal_run_status(exc_type: type[BaseException] | None, expected: RunStatus) -> None:
    assert _terminal_run_status(exc_type) is expected


def test_sink_write_through_when_no_buffer() -> None:
    backend = InMemoryBackend()
    client = Client.from_store(backend)
    rid = _make_run(backend)
    sink = _Sink(client.metadata, client.artifacts, None)
    sink.append_metric(rid, mint(Metric, key=MetricKey("loss"), value=1.0))
    assert len(client.metadata.get_metric_history(rid, MetricKey("loss"))) == 1


def test_sink_flushes_at_max_records() -> None:
    backend = InMemoryBackend()
    client = Client.from_store(backend)
    rid = _make_run(backend)
    clock = _FakeClock()
    sink = _Sink(client.metadata, client.artifacts, BufferPolicy(max_records=2, max_interval_s=1000), clock=clock)
    sink.append_metric(rid, mint(Metric, key=MetricKey("loss"), value=1.0))
    assert len(client.metadata.get_metric_history(rid, MetricKey("loss"))) == 0
    sink.append_metric(rid, mint(Metric, key=MetricKey("loss"), value=2.0))
    assert len(client.metadata.get_metric_history(rid, MetricKey("loss"))) == 2


def test_sink_flushes_on_interval() -> None:
    backend = InMemoryBackend()
    client = Client.from_store(backend)
    rid = _make_run(backend)
    clock = _FakeClock()
    sink = _Sink(client.metadata, client.artifacts, BufferPolicy(max_records=1000, max_interval_s=5.0), clock=clock)
    sink.append_metric(rid, mint(Metric, key=MetricKey("loss"), value=1.0))
    clock.advance(10.0)
    sink.append_metric(rid, mint(Metric, key=MetricKey("loss"), value=2.0))
    assert len(client.metadata.get_metric_history(rid, MetricKey("loss"))) == 2


def test_sink_flushes_before_status_change() -> None:
    backend = InMemoryBackend()
    client = Client.from_store(backend)
    rid = _make_run(backend)
    clock = _FakeClock()
    sink = _Sink(client.metadata, client.artifacts, BufferPolicy(max_records=1000, max_interval_s=1000), clock=clock)
    sink.append_metric(rid, mint(Metric, key=MetricKey("loss"), value=1.0))
    sink.set_run_status(rid, RunStatus.COMPLETED)
    assert len(client.metadata.get_metric_history(rid, MetricKey("loss"))) == 1


def test_experiment_context_run_clean_completes(in_memory_client: Client) -> None:
    with in_memory_client.experiment("e") as exp, exp.run("r") as run:
        run.log_metric("loss", 0.5)
    assert in_memory_client.metadata.get_run(RunID("r")).status is RunStatus.COMPLETED


def test_experiment_context_run_exception_fails(in_memory_client: Client) -> None:
    with pytest.raises(RuntimeError, match="boom"), in_memory_client.experiment("e") as exp, exp.run("r"):
        raise RuntimeError("boom")
    assert in_memory_client.metadata.get_run(RunID("r")).status is RunStatus.FAILED


def test_experiment_context_run_keyboard_interrupt_killed(in_memory_client: Client) -> None:
    with pytest.raises(KeyboardInterrupt), in_memory_client.experiment("e") as exp, exp.run("r"):
        raise KeyboardInterrupt
    assert in_memory_client.metadata.get_run(RunID("r")).status is RunStatus.KILLED


def test_experiment_context_run_invalid_params_fail(in_memory_client: Client) -> None:
    with pytest.raises(ValidationError), in_memory_client.experiment("e") as exp:
        with exp.run("r", params={"../x": 1}):  # type: ignore[dict-item]
            pass
    assert in_memory_client.metadata.get_run(RunID("r")).status is RunStatus.FAILED


def test_build_artifact_single_construction_path(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    store = InMemoryArtifactStore()
    art = _build_artifact(source, b"weights", store, None)
    assert art.key == "model.bin"
    assert art.size_bytes == 7
    assert art.storage_key == store.new_storage_key(b"weights")


def test_experiment_records_provenance(in_memory_client: Client) -> None:
    with in_memory_client.experiment(
        "e",
        description="desc",
        git_info=GitInfo(commit="abc"),
        custom_metadata={"team": "ml"},
    ):
        pass
    meta = in_memory_client.metadata.get_experiment(ExperimentID("e"))
    assert meta.description == "desc"
    assert meta.git_info is not None and meta.git_info.commit == "abc"
    assert meta.custom_metadata == {"team": "ml"}


def test_run_context_log_artifact_round_trip(tmp_path: Path, in_memory_client: Client) -> None:
    model_file = tmp_path / "m.bin"
    model_file.write_bytes(b"weights")
    with in_memory_client.experiment("e") as exp, exp.run("r") as runctx:
        artifact = runctx.log_artifact(str(model_file))
    assert artifact.size_bytes == 7
    assert in_memory_client.artifacts.open_artifact(RunID("r"), artifact.key) == b"weights"
