import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from experiments.domain.exceptions import NotFoundError
from experiments.domain.ids import (
    ArtifactKey,
    ExperimentID,
    MetricKey,
    ParamKey,
    RunID,
    StorageKey,
)
from experiments.domain.models import (
    Artifact,
    ExperimentMetadata,
    Metric,
    Param,
    RunMetadata,
    Tag,
)
from experiments.domain.status import RunStatus


@dataclass
class _RunState:
    meta: RunMetadata
    params: dict[ParamKey, Param] = field(default_factory=dict)
    tags: dict[str, Tag] = field(default_factory=dict)
    metrics: dict[MetricKey, list[Metric]] = field(default_factory=dict)
    artifacts: dict[ArtifactKey, tuple[Artifact, bytes]] = field(default_factory=dict)


class InMemoryBackend:
    """Dict-backed backend for tests and ephemeral use. Stores frozen models directly."""

    def __init__(self) -> None:
        self._experiments: dict[ExperimentID, ExperimentMetadata] = {}
        self._runs: dict[RunID, _RunState] = {}

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _run(self, run_id: RunID) -> _RunState:
        state = self._runs.get(run_id)
        if state is None:
            raise NotFoundError("run", run_id)
        return state

    def create_experiment(self, meta: ExperimentMetadata) -> None:
        self._experiments[meta.id] = meta

    def get_experiment(self, experiment_id: ExperimentID) -> ExperimentMetadata:
        meta = self._experiments.get(experiment_id)
        if meta is None:
            raise NotFoundError("experiment", experiment_id)
        return meta

    def list_experiments(self) -> Sequence[ExperimentMetadata]:
        return tuple(self._experiments.values())

    def create_run(self, meta: RunMetadata) -> None:
        self._runs[meta.id] = _RunState(meta=meta)

    def get_run(self, run_id: RunID) -> RunMetadata:
        return self._run(run_id).meta

    def list_runs(self, experiment_id: ExperimentID) -> Sequence[RunMetadata]:
        return tuple(s.meta for s in self._runs.values() if s.meta.experiment_id == experiment_id)

    def set_run_status(self, run_id: RunID, status: RunStatus, ended_at: datetime | None = None) -> None:
        state = self._run(run_id)
        updates: dict[str, object] = {"status": status}
        if ended_at is not None:
            updates["ended_at"] = ended_at
        state.meta = state.meta.model_copy(update=updates)

    def set_param(self, run_id: RunID, param: Param) -> None:
        self._run(run_id).params[param.key] = param

    def get_params(self, run_id: RunID) -> Mapping[ParamKey, Param]:
        return dict(self._run(run_id).params)

    def set_tag(self, run_id: RunID, tag: Tag) -> None:
        state = self._run(run_id)
        state.tags[tag.key] = tag
        state.meta = state.meta.model_copy(update={"tags": list(state.tags.values())})

    def append_metric(self, run_id: RunID, point: Metric) -> None:
        self._run(run_id).metrics.setdefault(point.key, []).append(point)

    def get_metric_history(self, run_id: RunID, key: MetricKey) -> Sequence[Metric]:
        return tuple(self._run(run_id).metrics.get(key, ()))

    def list_metric_keys(self, run_id: RunID) -> Sequence[MetricKey]:
        return tuple(self._run(run_id).metrics.keys())

    def new_storage_key(self, data: bytes) -> StorageKey:
        return StorageKey(hashlib.sha256(data).hexdigest())

    def put_artifact(self, run_id: RunID, artifact: Artifact, data: bytes) -> None:
        self._run(run_id).artifacts[artifact.key] = (artifact, data)

    def open_artifact(self, run_id: RunID, key: ArtifactKey) -> bytes:
        entry = self._run(run_id).artifacts.get(key)
        if entry is None:
            raise NotFoundError("artifact", key)
        return entry[1]

    def list_artifacts(self, run_id: RunID) -> Sequence[Artifact]:
        return tuple(a for a, _ in self._run(run_id).artifacts.values())


class InMemoryArtifactStore:
    """ArtifactStore-only backend (no metadata plane).

    Pairs with any ``MetadataStore`` via the two-param ``Client`` to prove blobs
    and metadata have different homes — the S3-shaped seam. A future
    ``S3ArtifactStore(bucket, prefix)`` is the same shape (``Lifecycle`` + these 4
    methods, deriving the blob key from ``run_id`` per call).
    """

    def __init__(self) -> None:
        self._blobs: dict[RunID, dict[ArtifactKey, tuple[Artifact, bytes]]] = {}

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def new_storage_key(self, data: bytes) -> StorageKey:
        return StorageKey(hashlib.sha256(data).hexdigest())

    def put_artifact(self, run_id: RunID, artifact: Artifact, data: bytes) -> None:
        self._blobs.setdefault(run_id, {})[artifact.key] = (artifact, data)

    def open_artifact(self, run_id: RunID, key: ArtifactKey) -> bytes:
        entry = self._blobs.get(run_id, {}).get(key)
        if entry is None:
            raise NotFoundError("artifact", key)
        return entry[1]

    def list_artifacts(self, run_id: RunID) -> Sequence[Artifact]:
        return tuple(a for a, _ in self._blobs.get(run_id, {}).values())
