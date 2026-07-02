from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol

from experiments.domain.exceptions import StateError, ValidationError
from experiments.domain.ids import (
    ArtifactKey,
    ExperimentID,
    MetricKey,
    ParamKey,
    ParamValue,
    RunID,
    is_valid_metric_key,
    is_valid_param_key,
    is_valid_tag_key,
)
from experiments.domain.models import Artifact, Metric, Param, Tag, mint
from experiments.domain.status import ExperimentStatus, RunStatus


class PersistenceSink(Protocol):
    """The narrow write contract `Run` forwards through. Any `Backend`, or a
    metadata+artifact store pair, satisfies it structurally."""

    def append_metric(self, run_id: RunID, point: Metric) -> None: ...
    def set_param(self, run_id: RunID, param: Param) -> None: ...
    def put_artifact(self, run_id: RunID, artifact: Artifact, data: bytes) -> None: ...
    def set_tag(self, run_id: RunID, tag: Tag) -> None: ...
    def set_run_status(self, run_id: RunID, status: RunStatus, ended_at: datetime | None = None) -> None: ...


class _MetricLog:
    """Mutable hot-path accumulator for one metric key. O(1) append."""

    __slots__ = ("key", "_points")

    def __init__(self, key: MetricKey) -> None:
        self.key = key
        self._points: list[Metric] = []

    def append(self, point: Metric) -> None:
        self._points.append(point)

    @property
    def last_step(self) -> int | None:
        return self._points[-1].step if self._points else None


class Run:
    """The aggregate root. Owns params/metrics/tags/status invariants. Persistence-ignorant."""

    def __init__(
        self,
        *,
        run_id: RunID,
        experiment_id: ExperimentID,
        sink: PersistenceSink,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._id = run_id
        self._experiment_id = experiment_id
        self._sink = sink
        self._clock = clock
        self._status = RunStatus.RUNNING
        self._ended_at: datetime | None = None
        self._params: dict[ParamKey, Param] = {}
        self._metrics: dict[MetricKey, _MetricLog] = {}
        self._tags: dict[str, Tag] = {}
        self._artifacts: dict[ArtifactKey, Artifact] = {}

    @property
    def id(self) -> RunID:
        return self._id

    @property
    def experiment_id(self) -> ExperimentID:
        return self._experiment_id

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def params(self) -> Mapping[ParamKey, Param]:
        return dict(self._params)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Run) and other._id == self._id

    def __hash__(self) -> int:
        return hash(self._id)

    def _guard_running(self, action: str) -> None:
        if self._status.is_terminal:
            raise StateError(
                current_state=self._status,
                action=action,
                allowed_states=[RunStatus.RUNNING],
            )

    def _next_step(self, key: MetricKey) -> int:
        log = self._metrics.get(key)
        last = log.last_step if log is not None else None
        return 0 if last is None else last + 1

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        self._guard_running("log_metric")
        if not is_valid_metric_key(key):
            raise ValidationError(field=key, value=key, message="invalid metric key")
        mkey = MetricKey(key)
        next_step = step if step is not None else self._next_step(mkey)
        point = mint(Metric, key=mkey, value=value, step=next_step, timestamp=self._clock())
        self._metrics.setdefault(mkey, _MetricLog(mkey)).append(point)
        self._sink.append_metric(self._id, point)

    def log_param(self, key: str, value: ParamValue) -> None:
        self._guard_running("log_param")
        if not is_valid_param_key(key):
            raise ValidationError(field=key, value=key, message="invalid param key")
        pkey = ParamKey(key)
        existing = self._params.get(pkey)
        if existing is not None:
            if existing.value != value:
                raise ValidationError(
                    field=key,
                    value=value,
                    message=f"param already set to {existing.value!r}",
                )
            return
        param = mint(Param, key=pkey, value=value)
        self._params[pkey] = param
        self._sink.set_param(self._id, param)

    def set_tag(self, key: str, value: str) -> None:
        self._guard_running("set_tag")
        if not is_valid_tag_key(key):
            raise ValidationError(field=key, value=key, message="invalid tag key")
        tag = Tag(key=key, value=value)
        self._tags[key] = tag
        self._sink.set_tag(self._id, tag)

    def log_artifact(self, artifact: Artifact, data: bytes) -> None:
        """Record an already-built, content-addressed artifact. File I/O happens
        in the driving layer (RunContext); the domain stays persistence-ignorant."""
        self._guard_running("log_artifact")
        self._artifacts[artifact.key] = artifact
        self._sink.put_artifact(self._id, artifact, data)

    def finish(self, status: RunStatus | None = None) -> None:
        if self._status.is_terminal:
            return
        target = status if status is not None else RunStatus.COMPLETED
        if not self._status.can_transition_to(target):
            raise StateError(current_state=self._status, action=f"transition to {target}")
        self._status = target
        self._ended_at = self._clock()
        self._sink.set_run_status(self._id, target, self._ended_at)


class Experiment:
    """A small aggregate referencing runs by id only."""

    def __init__(self, *, experiment_id: ExperimentID, name: str) -> None:
        self._id = experiment_id
        self._name = name
        self._status = ExperimentStatus.RUNNING

    @property
    def id(self) -> ExperimentID:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> ExperimentStatus:
        return self._status

    def _transition(self, target: ExperimentStatus, action: str) -> None:
        if not self._status.can_transition_to(target):
            raise StateError(current_state=self._status, action=action)
        self._status = target

    def complete(self) -> None:
        self._transition(ExperimentStatus.COMPLETED, "complete")

    def fail(self) -> None:
        self._transition(ExperimentStatus.FAILED, "fail")
