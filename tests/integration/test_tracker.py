"""Integration tests for experiment tracker module."""

from __future__ import annotations

from typing import Any

import pytest

from experiments.tracker import (
    experiment,
    finish,
    get_experiment,
    get_run,
    is_active,
    run,
)


class SimpleStorage:
    """Simple storage for integration testing."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def initialize(self, run_id: str, experiment_id: str) -> None:
        self.state["run_id"] = run_id
        self.state["exp_id"] = experiment_id
        self.state["initialized"] = True

    def finalize(self, status: str) -> None:
        self.state["status"] = status
        self.state["finalized"] = True


class TestTracker:
    """Test global tracker functions."""

    def test_experiment_context(self) -> None:
        storage = SimpleStorage()

        with experiment("test_exp", storage=storage) as exp:
            assert get_experiment() is exp
            assert is_active()

        assert get_experiment() is None
        assert not is_active()

    def test_run_context(self) -> None:
        storage = SimpleStorage()

        with experiment("test_exp", storage=storage):
            with run("test_run") as r:
                assert get_run() is r
                assert storage.state["initialized"]
                assert storage.state["run_id"] == "test_run"

            assert storage.state["finalized"]
            assert storage.state["status"] == "completed"

    def test_finish_clears_context(self) -> None:
        storage = SimpleStorage()

        with experiment("test_exp", storage=storage):
            assert is_active()
            finish()
            assert not is_active()
            assert get_experiment() is None

    def test_nested_experiments(self) -> None:
        storage1 = SimpleStorage()
        storage2 = SimpleStorage()

        with experiment("exp1", storage=storage1) as exp1:
            assert get_experiment() is exp1

            # Nested experiment clears the outer context
            with experiment("exp2", storage=storage2) as exp2:
                assert get_experiment() is exp2
                with run("run2") as r2:
                    assert get_run() is r2

            # After inner experiment, context is cleared (not restored)
            assert get_experiment() is None

    def test_run_without_experiment_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No active experiment"):
            with run("test"):
                pass

    def test_multiple_runs_in_experiment(self) -> None:
        storage = SimpleStorage()

        with experiment("multi_run", storage=storage):
            for i in range(3):
                with run(f"run_{i}") as r:
                    assert r.id == f"run_{i}"
                    assert storage.state["run_id"] == f"run_{i}"
