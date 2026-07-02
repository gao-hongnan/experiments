"""Unit tests for the structured exception hierarchy."""

from __future__ import annotations

import pytest

from experiments.domain.exceptions import (
    ExperimentError,
    NoActiveClientError,
    NoActiveContextError,
    NoActiveExperimentError,
    NoActiveRunError,
    NotFoundError,
    StateError,
    StorageError,
    ValidationError,
)
from experiments.domain.status import ExperimentStatus, RunStatus


def test_experiment_error_immutable_context() -> None:
    err = ExperimentError("boom", {"a": 1})
    assert err.message == "boom"
    assert err.context == {"a": 1}
    assert "boom" in repr(err)
    with pytest.raises(TypeError):
        err.context["a"] = 2  # type: ignore[index]


def test_experiment_error_default_context_is_empty() -> None:
    assert ExperimentError("x").context == {}


def test_validation_error_carries_field_and_value() -> None:
    err = ValidationError(field="lr", value=0.1, message="out of range")
    assert err.field == "lr"
    assert err.value == 0.1
    assert "lr" in err.message


def test_storage_error_builds_context() -> None:
    err = StorageError("disk full", operation="write", path="/p")
    assert err.operation == "write"
    assert err.path == "/p"
    assert err.context == {"operation": "write", "path": "/p"}


def test_storage_error_without_optionals() -> None:
    err = StorageError("oops")
    assert err.operation is None
    assert err.path is None


def test_not_found_error() -> None:
    err = NotFoundError("run", "r1")
    assert err.resource_type == "run"
    assert err.identifier == "r1"
    assert "r1" in err.message


def test_state_error_preserves_typed_current_state() -> None:
    err = StateError(current_state=RunStatus.RUNNING, action="log", allowed_states=[RunStatus.RUNNING])
    assert err.current_state is RunStatus.RUNNING
    assert err.action == "log"
    assert err.allowed_states == (RunStatus.RUNNING,)


def test_state_error_without_allowed_states() -> None:
    err = StateError(current_state=ExperimentStatus.RUNNING, action="x")
    assert err.allowed_states is None


def test_state_error_accepts_experiment_status_too() -> None:
    err = StateError(current_state=ExperimentStatus.FAILED, action="restart")
    assert err.current_state is ExperimentStatus.FAILED


@pytest.mark.parametrize(
    ("exc", "resource"),
    [
        (NoActiveRunError("log"), "run"),
        (NoActiveExperimentError("start"), "experiment"),
        (NoActiveClientError("store"), "client"),
    ],
)
def test_no_active_errors_carry_resource(exc: NoActiveContextError, resource: str) -> None:
    assert exc.resource == resource
    assert isinstance(exc, NoActiveContextError)
    assert isinstance(exc, ExperimentError)
    assert not isinstance(exc, StateError)
