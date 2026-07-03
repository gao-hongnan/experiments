"""Integration tests for the in-memory backend and the two-store client seam."""

from __future__ import annotations

import pytest

from experiments import Client
from experiments.adapters import InMemoryArtifactStore, InMemoryBackend
from experiments.domain.exceptions import NotFoundError
from experiments.domain.ids import ArtifactKey, ExperimentID, MetricKey, ParamKey, RunID
from experiments.domain.models import (
    Artifact,
    ExperimentMetadata,
    Metric,
    Param,
    RunMetadata,
    Tag,
    mint,
)


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


def _seed(backend: InMemoryBackend, eid: str = "e1", rid: str = "r1") -> None:
    backend.create_experiment(mint(ExperimentMetadata, id=ExperimentID(eid), name=eid))
    backend.create_run(mint(RunMetadata, id=RunID(rid), experiment_id=ExperimentID(eid)))


def test_experiment_crud(backend: InMemoryBackend) -> None:
    _seed(backend)
    assert backend.get_experiment(ExperimentID("e1")).name == "e1"
    assert len(backend.list_experiments()) == 1
    with pytest.raises(NotFoundError):
        backend.get_experiment(ExperimentID("missing"))


def test_run_crud(backend: InMemoryBackend) -> None:
    _seed(backend)
    assert backend.get_run(RunID("r1")).experiment_id == "e1"
    assert len(backend.list_runs(ExperimentID("e1"))) == 1
    with pytest.raises(NotFoundError):
        backend.get_run(RunID("missing"))


def test_params_round_trip(backend: InMemoryBackend) -> None:
    _seed(backend)
    backend.set_param(RunID("r1"), mint(Param, key=ParamKey("lr"), value=0.1))
    backend.set_param(RunID("r1"), mint(Param, key=ParamKey("epochs"), value=10))
    params = backend.get_params(RunID("r1"))
    assert params[ParamKey("lr")].value == 0.1
    assert params[ParamKey("epochs")].value == 10
    assert params[ParamKey("lr")].value != 1


def test_tags_overwrite(backend: InMemoryBackend) -> None:
    _seed(backend)
    backend.set_tag(RunID("r1"), Tag(key="phase", value="warmup"))
    backend.set_tag(RunID("r1"), Tag(key="phase", value="train"))
    tags = {t.key: t.value for t in backend.get_run(RunID("r1")).tags}
    assert tags == {"phase": "train"}


def test_metrics_append_history_and_keys(backend: InMemoryBackend) -> None:
    _seed(backend)
    for step, value in enumerate((0.5, 0.4, 0.3)):
        backend.append_metric(RunID("r1"), mint(Metric, key=MetricKey("loss"), value=value, step=step))
    backend.append_metric(RunID("r1"), mint(Metric, key=MetricKey("acc"), value=0.9, step=0))
    hist = backend.get_metric_history(RunID("r1"), MetricKey("loss"))
    assert [m.value for m in hist] == [0.5, 0.4, 0.3]
    assert set(backend.list_metric_keys(RunID("r1"))) == {MetricKey("loss"), MetricKey("acc")}


def test_artifacts_content_addressed(backend: InMemoryBackend) -> None:
    _seed(backend)
    art = mint(Artifact, key=ArtifactKey("model"), storage_key=backend.new_storage_key(b"abc"), path="/tmp/m")
    backend.put_artifact(RunID("r1"), art, b"abc")
    assert backend.open_artifact(RunID("r1"), ArtifactKey("model")) == b"abc"
    assert [a.key for a in backend.list_artifacts(RunID("r1"))] == [ArtifactKey("model")]
    with pytest.raises(NotFoundError):
        backend.open_artifact(RunID("r1"), ArtifactKey("missing"))


def test_two_store_client_seam() -> None:
    meta = InMemoryBackend()
    art = InMemoryArtifactStore()
    client = Client(meta, art)
    with client.experiment("e") as exp, exp.run("r") as run:
        run.log_metric("loss", 0.5)
        run.set_tag("t", "v")
    assert meta.get_metric_history(RunID("r"), MetricKey("loss"))[0].value == 0.5


def test_in_memory_artifact_store_crud() -> None:
    store = InMemoryArtifactStore()
    art = mint(
        Artifact,
        key=ArtifactKey("model"),
        storage_key=store.new_storage_key(b"abc"),
        path="/tmp/m",
    )
    store.put_artifact(RunID("r1"), art, b"abc")
    assert store.open_artifact(RunID("r1"), ArtifactKey("model")) == b"abc"
    assert [a.key for a in store.list_artifacts(RunID("r1"))] == [ArtifactKey("model")]
    with pytest.raises(NotFoundError):
        store.open_artifact(RunID("r1"), ArtifactKey("missing"))
    with pytest.raises(NotFoundError):
        store.open_artifact(RunID("other"), ArtifactKey("model"))


def test_in_memory_backend_list_and_empty_returns(backend: InMemoryBackend) -> None:
    _seed(backend, "e1", "r1")
    _seed(backend, "e2", "r2")
    assert len(backend.list_experiments()) == 2
    assert len(backend.list_runs(ExperimentID("e1"))) == 1
    assert backend.get_params(RunID("r1")) == {}
    assert backend.get_metric_history(RunID("r1"), MetricKey("loss")) == ()
    assert backend.list_metric_keys(RunID("r1")) == ()
    assert backend.list_artifacts(RunID("r1")) == ()
    with pytest.raises(NotFoundError):
        backend.open_artifact(RunID("r1"), ArtifactKey("nope"))
