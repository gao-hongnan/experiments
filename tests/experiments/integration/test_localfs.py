"""Integration tests for the local-filesystem backend, incl. the durability /
concurrency / cache regression coverage for the defects this merge fixed."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from experiments import Client
from experiments.adapters.localfs import LocalFileSystemBackend
from experiments.domain.exceptions import NotFoundError, StorageError
from experiments.domain.ids import ArtifactKey, ExperimentID, MetricKey, ParamKey, RunID
from experiments.domain.models import Artifact, ExperimentMetadata, Metric, Param, RunMetadata, Tag, mint
from experiments.domain.status import RunStatus

_RUN_FILE = "run.json"


@pytest.fixture
def backend(tmp_path: Path) -> LocalFileSystemBackend:
    b = LocalFileSystemBackend(tmp_path)
    b.open()
    return b


def _seed(backend: LocalFileSystemBackend, eid: str = "e1", rid: str = "r1") -> None:
    backend.create_experiment(mint(ExperimentMetadata, id=ExperimentID(eid), name=eid))
    backend.create_run(mint(RunMetadata, id=RunID(rid), experiment_id=ExperimentID(eid)))


def test_crud_round_trip(backend: LocalFileSystemBackend) -> None:
    _seed(backend)
    backend.set_param(RunID("r1"), mint(Param, key=ParamKey("lr"), value=0.1))
    backend.set_tag(RunID("r1"), Tag(key="phase", value="warmup"))
    backend.append_metric(RunID("r1"), mint(Metric, key=MetricKey("loss"), value=0.5, step=0))
    backend.append_metric(RunID("r1"), mint(Metric, key=MetricKey("loss"), value=0.4, step=1))
    assert backend.get_params(RunID("r1"))[ParamKey("lr")].value == 0.1
    hist = backend.get_metric_history(RunID("r1"), MetricKey("loss"))
    assert [m.value for m in hist] == [0.5, 0.4]
    assert MetricKey("loss") in set(backend.list_metric_keys(RunID("r1")))


def test_list_experiments_and_runs(backend: LocalFileSystemBackend) -> None:
    _seed(backend, "e1", "r1")
    _seed(backend, "e2", "r2")
    assert len(backend.list_experiments()) == 2
    assert len(backend.list_runs(ExperimentID("e1"))) == 1


def test_not_found_errors(backend: LocalFileSystemBackend) -> None:
    _seed(backend)
    with pytest.raises(NotFoundError):
        backend.get_experiment(ExperimentID("missing"))
    with pytest.raises(NotFoundError):
        backend.get_run(RunID("missing"))


def test_corrupt_run_metadata_raises_storage_error(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    _seed(backend)
    run_json = tmp_path / "experiments" / "e1" / "runs" / "r1" / _RUN_FILE
    run_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(StorageError):
        backend.get_run(RunID("r1"))


def test_run_dir_cache_hit_after_create(backend: LocalFileSystemBackend) -> None:
    _seed(backend)
    assert RunID("r1") in backend._run_dirs
    assert backend.get_run(RunID("r1")).id == "r1"


def test_run_dir_cache_miss_falls_back_to_rglob(tmp_path: Path) -> None:
    first = LocalFileSystemBackend(tmp_path)
    first.open()
    _seed(first)
    second = LocalFileSystemBackend(tmp_path)
    second.open()
    assert second.get_run(RunID("r1")).id == "r1"
    assert RunID("r1") in second._run_dirs


def test_concurrent_tags_do_not_lose_updates(backend: LocalFileSystemBackend) -> None:
    """system-2 regression: per-run RLock serializes the read-modify-write so
    concurrent tag writes to the same run are all persisted."""
    _seed(backend)

    def write_tag(key: str) -> None:
        backend.set_tag(RunID("r1"), Tag(key=key, value="v"))

    threads = [threading.Thread(target=write_tag, args=(f"k{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    keys = {tag.key for tag in backend.get_run(RunID("r1")).tags}
    assert keys == {f"k{i}" for i in range(8)}


def test_concurrent_params_do_not_lose_updates(backend: LocalFileSystemBackend) -> None:
    _seed(backend)

    def write_param(name: str) -> None:
        backend.set_param(RunID("r1"), mint(Param, key=ParamKey(name), value=1))

    threads = [threading.Thread(target=write_param, args=(f"p{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    keys = set(backend.get_params(RunID("r1")).keys())
    assert keys == {ParamKey(f"p{i}") for i in range(8)}


def test_resilient_metric_reader_skips_torn_trailing_line(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    """system-4 regression: a torn trailing line must not erase the whole series."""
    _seed(backend)
    backend.append_metric(RunID("r1"), mint(Metric, key=MetricKey("loss"), value=0.5, step=0))
    backend.append_metric(RunID("r1"), mint(Metric, key=MetricKey("loss"), value=0.4, step=1))
    path = tmp_path / "experiments" / "e1" / "runs" / "r1" / "metrics" / "loss.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{truncated line")
    hist = backend.get_metric_history(RunID("r1"), MetricKey("loss"))
    assert [m.value for m in hist] == [0.5, 0.4]


def test_atomic_writes_leave_no_temp_files(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    _seed(backend)
    backend.set_param(RunID("r1"), mint(Param, key=ParamKey("lr"), value=0.1))
    backend.set_tag(RunID("r1"), Tag(key="k", value="v"))
    assert list(tmp_path.rglob("*.tmp")) == []


def test_reopen_same_root_reads_back_state(tmp_path: Path) -> None:
    client = Client.from_store(LocalFileSystemBackend(tmp_path))
    with client.experiment("e") as exp, exp.run("r", params={"lr": 0.1}) as run:
        run.log_metric("loss", 0.5)
    reopened = Client.from_store(LocalFileSystemBackend(tmp_path))
    assert reopened.metadata.get_run(RunID("r")).status is RunStatus.COMPLETED
    assert reopened.metadata.get_metric_history(RunID("r"), MetricKey("loss"))[0].value == 0.5
    assert reopened.metadata.get_params(RunID("r"))[ParamKey("lr")].value == 0.1


def test_corrupt_experiment_metadata_raises(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    _seed(backend)
    (tmp_path / "experiments" / "e1" / "experiment.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(StorageError):
        backend.get_experiment(ExperimentID("e1"))


def test_list_experiments_corrupt_raises(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    _seed(backend, "e1", "r1")
    _seed(backend, "e2", "r2")
    (tmp_path / "experiments" / "e2" / "experiment.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(StorageError):
        backend.list_experiments()


def test_list_runs_corrupt_raises(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    _seed(backend)
    (tmp_path / "experiments" / "e1" / "runs" / "r1" / "run.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(StorageError):
        backend.list_runs(ExperimentID("e1"))


def test_corrupt_params_raises(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    _seed(backend)
    (tmp_path / "experiments" / "e1" / "runs" / "r1" / "params.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(StorageError):
        backend.get_params(RunID("r1"))


def test_list_metric_keys_corrupt_raises(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    _seed(backend)
    backend.append_metric(RunID("r1"), mint(Metric, key=MetricKey("loss"), value=0.5))
    (tmp_path / "experiments" / "e1" / "runs" / "r1" / "metrics" / "loss.jsonl").write_text("{bad\n", encoding="utf-8")
    with pytest.raises(StorageError):
        backend.list_metric_keys(RunID("r1"))


def test_corrupt_artifact_index_raises(backend: LocalFileSystemBackend, tmp_path: Path) -> None:
    _seed(backend)
    art = mint(
        Artifact,
        key=ArtifactKey("model"),
        storage_key=backend.new_storage_key(b"abc"),
        path="/tmp/m",
    )
    backend.put_artifact(RunID("r1"), art, b"abc")
    (tmp_path / "experiments" / "e1" / "runs" / "r1" / "artifacts" / "index.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(StorageError):
        backend.list_artifacts(RunID("r1"))


def test_open_artifact_missing_raises(backend: LocalFileSystemBackend) -> None:
    _seed(backend)
    with pytest.raises(NotFoundError):
        backend.open_artifact(RunID("r1"), ArtifactKey("nope"))


def test_empty_returns_for_fresh_run(backend: LocalFileSystemBackend) -> None:
    _seed(backend)
    assert backend.get_params(RunID("r1")) == {}
    assert backend.get_metric_history(RunID("r1"), MetricKey("loss")) == ()
    assert backend.list_metric_keys(RunID("r1")) == ()
    assert backend.list_artifacts(RunID("r1")) == ()


def test_open_failure_raises_storage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("cannot mkdir")

    monkeypatch.setattr(Path, "mkdir", boom)
    backend = LocalFileSystemBackend(tmp_path)
    with pytest.raises(StorageError):
        backend.open()


def test_metadata_write_failure_raises_storage_error(
    backend: LocalFileSystemBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(backend)

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(StorageError):
        backend.set_param(RunID("r1"), mint(Param, key=ParamKey("lr"), value=0.1))


def test_blob_write_failure_raises_storage_error(
    backend: LocalFileSystemBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(backend)
    art = mint(
        Artifact,
        key=ArtifactKey("model"),
        storage_key=backend.new_storage_key(b"abc"),
        path="/tmp/m",
    )

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(StorageError):
        backend.put_artifact(RunID("r1"), art, b"abc")
