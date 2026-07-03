"""Unit tests for the ambient tracker API and its zero-config lifecycle fixes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import experiments.client as client_module
from experiments.client import ExperimentContext, default_client
from experiments.domain.exceptions import (
    NoActiveClientError,
    NoActiveExperimentError,
    NoActiveRunError,
)
from experiments.domain.ids import ExperimentID, MetricKey, RunID
from experiments.tracker import (
    active_experiment,
    active_run,
    experiment,
    finish,
    log_artifact,
    log_metric,
    log_params,
    run,
    set_tag,
)


def _spy_exit(monkeypatch: pytest.MonkeyPatch, seen: dict[str, Any]) -> None:
    """Record the exc_type ExperimentContext.__exit__ receives, then call through."""
    original = ExperimentContext.__exit__

    def spy(self: ExperimentContext, exc_type: object, exc_value: object, tb: object) -> None:
        seen["exc_type"] = exc_type
        return original(self, exc_type, exc_value, tb)  # type: ignore[arg-type]

    monkeypatch.setattr(ExperimentContext, "__exit__", spy)


def test_ambient_verbs_route_to_active_run(in_memory_client: object) -> None:
    with experiment("e", client=in_memory_client):  # type: ignore[arg-type]
        with run("r", params={"lr": 0.1}):
            log_metric("loss", 0.5)
            set_tag("phase", "warmup")
            assert active_run() is not None
        assert active_run() is None
    assert active_experiment() is None
    meta = in_memory_client.metadata.get_experiment(ExperimentID("e"))  # type: ignore[attr-defined]
    assert meta.name == "e"
    hist = in_memory_client.metadata.get_metric_history(RunID("r"), MetricKey("loss"))  # type: ignore[attr-defined]
    assert len(hist) == 1 and hist[0].value == 0.5


def test_experiment_nesting_restores_parent(in_memory_client: object) -> None:
    with experiment("outer", client=in_memory_client) as outer:  # type: ignore[arg-type]
        assert active_experiment() is outer
        with experiment("inner", client=in_memory_client) as inner:  # type: ignore[arg-type]
            assert active_experiment() is inner
        assert active_experiment() is outer
    assert active_experiment() is None


def test_log_metric_without_run_raises_no_active_run() -> None:
    finish()
    with pytest.raises(NoActiveRunError):
        log_metric("loss", 0.5)


def test_log_params_without_run_raises_no_active_run() -> None:
    finish()
    with pytest.raises(NoActiveRunError):
        log_params({"lr": 0.1})


def test_run_missing_experiment_raise() -> None:
    finish()
    with pytest.raises(NoActiveExperimentError), run("r", missing_experiment="raise"):
        pass


def test_log_artifact_without_client_raises_no_active_client(in_memory_client: object, tmp_path: Path) -> None:
    from experiments.tracker import _ambient_client

    with experiment("e", client=in_memory_client):  # type: ignore[arg-type]
        with run("r"):
            token = _ambient_client.set(None)
            try:
                artifact_file = tmp_path / "m.bin"
                artifact_file.write_bytes(b"x")
                with pytest.raises(NoActiveClientError):
                    log_artifact(str(artifact_file))
            finally:
                _ambient_client.reset(token)


def test_zero_config_run_clean_completes_default(isolated_default: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    _spy_exit(monkeypatch, seen)
    with run("r"):
        log_metric("loss", 0.5)
    assert seen["exc_type"] is None
    assert active_run() is None
    assert active_experiment() is None


def test_zero_config_run_block_exception_forwards_to_experiment_exit(
    isolated_default: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """software-3: the transient Default experiment must be FAILED (real exc
    forwarded to __exit__), not always COMPLETED."""
    seen: dict[str, Any] = {}
    _spy_exit(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="boom"), run("r"):
        raise RuntimeError("boom")
    assert seen["exc_type"] is RuntimeError


def test_zero_config_run_entry_failure_resets_contextvars(
    isolated_default: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """system-3: a failure during run()-entry (create_run raises) must still
    reset the ambient contextvars, not leak them."""
    client = default_client()

    def boom(meta: object) -> None:
        raise RuntimeError("create_run failed")

    monkeypatch.setattr(client.metadata, "create_run", boom)
    with pytest.raises(RuntimeError, match="create_run failed"), run("r"):
        pass
    assert active_run() is None
    assert active_experiment() is None


def test_finish_closes_default_client(isolated_default: Path) -> None:
    with experiment("e"), run("r"):
        log_metric("loss", 0.5)
    finish()
    assert client_module._default_client is None


def test_finish_resets_ambient_context(in_memory_client: object) -> None:
    with experiment("e", client=in_memory_client):  # type: ignore[arg-type]
        with run("r"):
            log_metric("loss", 0.5)
        assert active_run() is None
    assert active_experiment() is None
    finish()
    assert active_run() is None


def test_ambient_log_artifact_round_trip(in_memory_client: object, tmp_path: Path) -> None:
    model_file = tmp_path / "m.bin"
    model_file.write_bytes(b"abc")
    with experiment("e", client=in_memory_client):  # type: ignore[arg-type]
        with run("r"):
            log_metric("loss", 0.5)
            artifact = log_artifact(str(model_file))
    assert artifact.size_bytes == 3
    assert in_memory_client.artifacts.open_artifact(RunID("r"), artifact.key) == b"abc"  # type: ignore[attr-defined]


def test_active_accessors_none_when_idle() -> None:
    finish()
    assert active_run() is None
    assert active_experiment() is None
