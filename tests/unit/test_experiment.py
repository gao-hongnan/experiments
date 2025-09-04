"""Unit tests for Experiment and Run classes."""

from __future__ import annotations

from experiments.experiment import Experiment, Run
from experiments.types import ExperimentID, RunID


class MockStorage:
    """Mock storage implementation for testing."""

    def __init__(self) -> None:
        self.initialized = False
        self.finalized = False
        self.run_id: str | None = None
        self.experiment_id: str | None = None
        self.status: str | None = None
        self.metrics: dict[str, float] = {}

    def initialize(self, run_id: str, experiment_id: str) -> None:
        self.initialized = True
        self.run_id = run_id
        self.experiment_id = experiment_id

    def finalize(self, status: str) -> None:
        self.finalized = True
        self.status = status

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = value


class TestRun:
    """Test Run class functionality."""

    def test_run_creation(self) -> None:
        storage = MockStorage()
        run_id = RunID("test_run")
        exp_id = ExperimentID("test_exp")

        run = Run(run_id=run_id, storage=storage, experiment_id=exp_id)

        assert run.id == run_id
        assert run.experiment_id == exp_id
        assert run.storage is storage
        assert "id" in run.metadata
        assert "experiment_id" in run.metadata

    def test_run_context_manager(self) -> None:
        storage = MockStorage()
        run_id = RunID("test_run")
        exp_id = ExperimentID("test_exp")

        with Run(run_id=run_id, storage=storage, experiment_id=exp_id) as run:
            assert storage.initialized
            assert storage.run_id == str(run_id)
            assert storage.experiment_id == str(exp_id)
            run.storage.log_metric("loss", 0.5)

        assert storage.finalized
        assert storage.status == "completed"
        assert storage.metrics["loss"] == 0.5


class TestExperiment:
    """Test Experiment class functionality."""

    def test_experiment_creation(self) -> None:
        storage = MockStorage()
        exp = Experiment("test", storage=storage)

        assert exp.name == "test"
        assert exp._storage is storage
        assert "name" in exp.metadata
        assert "id" in exp.metadata

    def test_experiment_with_custom_id(self) -> None:
        storage = MockStorage()
        exp = Experiment("test", storage=storage, id="custom_id")

        assert exp.id == ExperimentID("custom_id")

    def test_experiment_run_creation(self) -> None:
        storage = MockStorage()
        exp = Experiment("test", storage=storage)

        with exp.run("test_run") as run:
            assert run.id == RunID("test_run")
            assert run.experiment_id == exp.id
            assert storage.initialized

        assert storage.finalized
        assert storage.status == "completed"
