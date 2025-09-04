from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from experiments.types import ExperimentStatus, RunStatus

T = TypeVar("T")

ResourceType = Literal["experiment", "run", "artifact", "metric", "storage"]
StorageOperation = Literal["read", "write", "delete", "list", "exists"]


class ExperimentError(Exception):
    """Base exception for experiment-related errors with immutable context."""

    __slots__ = ("_message", "_context")

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self._message: Final[str] = message
        self._context: Final[MappingProxyType[str, Any]] = MappingProxyType(context or {})
        super().__init__(message)

    @property
    def message(self) -> str:
        return self._message

    @property
    def context(self) -> Mapping[str, Any]:
        return self._context

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._message!r}, context={dict(self._context)!r})"


class ValidationError(ExperimentError, Generic[T]):
    """Validation error with strongly typed field and value."""

    __slots__ = ("_field", "_value")

    def __init__(self, field: str, value: T, message: str) -> None:
        self._field: Final[str] = field
        self._value: Final[T] = value
        super().__init__(
            f"Validation failed for '{field}': {message}",
            {"field": field, "value": value},
        )

    @property
    def field(self) -> str:
        return self._field

    @property
    def value(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return f"ValidationError(field={self._field!r}, value={self._value!r}, message={self.message!r})"


class StorageError(ExperimentError):
    """Storage operation error with optional operation context."""

    __slots__ = ("_operation", "_path")

    def __init__(
        self,
        message: str,
        *,
        operation: StorageOperation | None = None,
        path: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self._operation: Final[StorageOperation | None] = operation
        self._path: Final[str | None] = path

        full_context = context or {}
        if operation:
            full_context["operation"] = operation
        if path:
            full_context["path"] = path

        super().__init__(message, full_context)

    @property
    def operation(self) -> StorageOperation | None:
        return self._operation

    @property
    def path(self) -> str | None:
        return self._path


class NotFoundError(ExperimentError):
    """Resource not found error with constrained resource types."""

    __slots__ = ("_resource_type", "_identifier")

    def __init__(self, resource_type: ResourceType, identifier: str) -> None:
        self._resource_type: Final[ResourceType] = resource_type
        self._identifier: Final[str] = identifier
        super().__init__(
            f"{resource_type} not found: {identifier}",
            {"resource_type": resource_type, "identifier": identifier},
        )

    @property
    def resource_type(self) -> ResourceType:
        return self._resource_type

    @property
    def identifier(self) -> str:
        return self._identifier

    def __repr__(self) -> str:
        return f"NotFoundError(resource_type={self._resource_type!r}, identifier={self._identifier!r})"


class StateError(ExperimentError):
    """State transition error with type-safe state handling."""

    __slots__ = ("_current_state", "_action", "_allowed_states")

    def __init__(
        self,
        current_state: str | ExperimentStatus | RunStatus,
        action: str,
        *,
        allowed_states: list[str | ExperimentStatus | RunStatus] | None = None,
    ) -> None:
        self._current_state: Final[str] = str(current_state)
        self._action: Final[str] = action
        self._allowed_states: Final[tuple[str, ...] | None] = (
            tuple(str(s) for s in allowed_states) if allowed_states else None
        )

        context: dict[str, Any] = {
            "current_state": self._current_state,
            "action": action,
        }
        if allowed_states:
            context["allowed_states"] = self._allowed_states

        message = f"Cannot {action} in {current_state} state"
        if allowed_states:
            message += f" (allowed states: {', '.join(str(s) for s in allowed_states)})"

        super().__init__(message, context)

    @property
    def current_state(self) -> str:
        return self._current_state

    @property
    def action(self) -> str:
        return self._action

    @property
    def allowed_states(self) -> tuple[str, ...] | None:
        return self._allowed_states

    def __repr__(self) -> str:
        return (
            f"StateError(current_state={self._current_state!r}, "
            f"action={self._action!r}, allowed_states={self._allowed_states!r})"
        )
