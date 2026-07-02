"""Unit tests for the status state machines (exhaustive match arms)."""

from __future__ import annotations

import pytest

from experiments.domain.status import ExperimentStatus, RunStatus


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RunStatus.RUNNING, False),
        (RunStatus.COMPLETED, True),
        (RunStatus.FAILED, True),
        (RunStatus.KILLED, True),
    ],
)
def test_run_status_is_terminal(status: RunStatus, expected: bool) -> None:
    assert status.is_terminal is expected


def test_run_status_can_transition_to() -> None:
    assert RunStatus.RUNNING.can_transition_to(RunStatus.COMPLETED)
    assert RunStatus.RUNNING.can_transition_to(RunStatus.FAILED)
    assert RunStatus.RUNNING.can_transition_to(RunStatus.KILLED)
    assert RunStatus.RUNNING.can_transition_to(RunStatus.RUNNING) is False
    for terminal in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.KILLED):
        for target in RunStatus:
            assert terminal.can_transition_to(target) is False


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ExperimentStatus.RUNNING, False),
        (ExperimentStatus.COMPLETED, True),
        (ExperimentStatus.FAILED, True),
    ],
)
def test_experiment_status_is_terminal(status: ExperimentStatus, expected: bool) -> None:
    assert status.is_terminal is expected


def test_experiment_status_can_transition_to() -> None:
    assert ExperimentStatus.RUNNING.can_transition_to(ExperimentStatus.COMPLETED)
    assert ExperimentStatus.RUNNING.can_transition_to(ExperimentStatus.FAILED)
    assert ExperimentStatus.RUNNING.can_transition_to(ExperimentStatus.RUNNING) is False
    for terminal in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
        for target in ExperimentStatus:
            assert terminal.can_transition_to(target) is False


def test_str_enum_values_round_trip() -> None:
    assert RunStatus("completed") is RunStatus.COMPLETED
    assert ExperimentStatus("failed") is ExperimentStatus.FAILED
