from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal

from experiments.domain.status import ExperimentStatus, RunStatus

type ResourceType = Literal["experiment", "run", "artifact", "metric", "storage"]
type StorageOperation = Literal["read", "write", "delete", "list", "exists"]


class ExperimentError(Exception):
    """Base error with an immutable structured context."""

    __slots__ = ("_message", "_context")

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        self._message: Final[str] = message
        self._context: Final[MappingProxyType[str, object]] = MappingProxyType(context or {})
        super().__init__(message)

    @property
    def message(self) -> str:
        return self._message

    @property
    def context(self) -> Mapping[str, object]:
        return self._context

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._message!r}, context={dict(self._context)!r})"


class ValidationError[T](ExperimentError):
    """A validation failure carrying the offending field and value."""

    __slots__ = ("_field", "_value")

    def __init__(self, *, field: str, value: T, message: str) -> None:
        self._field: Final[str] = field
        self._value: Final[T] = value
        super().__init__(
            f"Validation failed for {field!r}: {message}",
            {"field": field, "value": value},
        )

    @property
    def field(self) -> str:
        return self._field

    @property
    def value(self) -> T:
        return self._value


class StorageError(ExperimentError):
    """An infrastructure failure from a backend adapter."""

    __slots__ = ("_operation", "_path")

    def __init__(
        self,
        message: str,
        *,
        operation: StorageOperation | None = None,
        path: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        self._operation: Final[StorageOperation | None] = operation
        self._path: Final[str | None] = path
        full: dict[str, object] = dict(context or {})
        if operation is not None:
            full["operation"] = operation
        if path is not None:
            full["path"] = path
        super().__init__(message, full)

    @property
    def operation(self) -> StorageOperation | None:
        return self._operation

    @property
    def path(self) -> str | None:
        return self._path


class NotFoundError(ExperimentError):
    """A whole-resource lookup that found nothing."""

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


class StateError[StatusT: (RunStatus, ExperimentStatus)](ExperimentError):
    """An illegal action or transition for the current status."""

    __slots__ = ("_current_state", "_action", "_allowed_states")

    def __init__(
        self,
        *,
        current_state: StatusT,
        action: str,
        allowed_states: Sequence[StatusT] | None = None,
    ) -> None:
        self._current_state: Final[StatusT] = current_state
        self._action: Final[str] = action
        self._allowed_states: Final[tuple[StatusT, ...] | None] = (
            tuple(allowed_states) if allowed_states is not None else None
        )
        context: dict[str, object] = {"current_state": current_state, "action": action}
        message = f"Cannot {action} in {current_state} state"
        if self._allowed_states is not None:
            context["allowed_states"] = self._allowed_states
            allowed = ", ".join(str(s) for s in self._allowed_states)
            message += f" (allowed: {allowed})"
        super().__init__(message, context)

    @property
    def current_state(self) -> StatusT:
        return self._current_state

    @property
    def action(self) -> str:
        return self._action

    @property
    def allowed_states(self) -> tuple[StatusT, ...] | None:
        return self._allowed_states


class NoActiveContextError(ExperimentError):
    """An ambient verb was called with no active run/client/experiment.

    Distinct from :class:`StateError` (an illegal transition of an *existing*
    aggregate): this means there is no aggregate at all. ``resource`` names what
    was missing so callers can branch on it instead of overloading a status enum.
    """

    __slots__ = ("_resource",)

    def __init__(self, resource: str, action: str) -> None:
        self._resource: Final[str] = resource
        super().__init__(
            f"No active {resource}: cannot {action}",
            {"resource": resource, "action": action},
        )

    @property
    def resource(self) -> str:
        return self._resource


class NoActiveRunError(NoActiveContextError):
    """No run context is active for an ambient logging verb."""

    __slots__ = ()

    def __init__(self, action: str) -> None:
        super().__init__("run", action)


class NoActiveExperimentError(NoActiveContextError):
    """No experiment context is active."""

    __slots__ = ()

    def __init__(self, action: str) -> None:
        super().__init__("experiment", action)


class NoActiveClientError(NoActiveContextError):
    """No client is active for an ambient verb that needs a backend."""

    __slots__ = ()

    def __init__(self, action: str) -> None:
        super().__init__("client", action)
