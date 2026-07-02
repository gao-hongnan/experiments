"""End-to-end tests: object API and ambient API over real backends."""

from __future__ import annotations

from pathlib import Path

from experiments import Client
from experiments.adapters import InMemoryBackend
from experiments.adapters.localfs import LocalFileSystemBackend
from experiments.domain.ids import ExperimentID, MetricKey, RunID
from experiments.domain.status import RunStatus
from experiments.tracker import experiment, log_metric, run, set_tag


def test_object_api_full_round_trip_on_localfs(tmp_path: Path) -> None:
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"weights")
    client = Client.from_store(LocalFileSystemBackend(tmp_path))
    with client.experiment("resnet") as exp, exp.run("baseline", params={"lr": 0.1}) as runctx:
        runctx.log_metric("train/loss", 0.5)
        runctx.log_metric("train/loss", 0.4)
        runctx.set_tag("phase", "warmup")
        artifact = runctx.log_artifact(str(model_file))

    reopened = Client.from_store(LocalFileSystemBackend(tmp_path))
    assert reopened.metadata.get_experiment(ExperimentID("resnet")).name == "resnet"
    assert reopened.metadata.get_run(RunID("baseline")).status is RunStatus.COMPLETED
    hist = reopened.metadata.get_metric_history(RunID("baseline"), MetricKey("train/loss"))
    assert [m.value for m in hist] == [0.5, 0.4]
    assert reopened.metadata.get_params(RunID("baseline"))[MetricKey("lr").__class__("lr")].value == 0.1  # type: ignore[union-attr]
    assert reopened.artifacts.open_artifact(RunID("baseline"), artifact.key) == b"weights"


def test_object_api_on_memory_backend() -> None:
    client = Client.from_store(InMemoryBackend())
    with client.experiment("e") as exp:
        for lr in (0.01, 0.1):
            with exp.run(f"lr_{lr}", params={"lr": lr}) as runctx:
                runctx.log_metric("loss", 1.0 - lr)
    runs = client.metadata.list_runs(ExperimentID("e"))
    assert {r.name for r in runs} == {"lr_0.01", "lr_0.1"}


def test_ambient_api_round_trip_on_localfs(tmp_path: Path) -> None:
    client = Client.from_store(LocalFileSystemBackend(tmp_path))
    with experiment("e", client=client), run("r", params={"lr": 0.1}):
        log_metric("loss", 0.5)
        set_tag("phase", "warmup")

    reopened = Client.from_store(LocalFileSystemBackend(tmp_path))
    assert reopened.metadata.get_metric_history(RunID("r"), MetricKey("loss"))[0].value == 0.5
    tags = {t.key: t.value for t in reopened.metadata.get_run(RunID("r")).tags}
    assert tags == {"phase": "warmup"}


def test_failed_run_is_persisted_as_failed(tmp_path: Path) -> None:
    client = Client.from_store(LocalFileSystemBackend(tmp_path))
    try:
        with client.experiment("e") as exp, exp.run("r"):
            raise RuntimeError("training blew up")
    except RuntimeError:
        pass
    assert client.metadata.get_run(RunID("r")).status is RunStatus.FAILED
