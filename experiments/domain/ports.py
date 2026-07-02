from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from experiments.domain.ids import ArtifactKey, ExperimentID, MetricKey, ParamKey, RunID, StorageKey
from experiments.domain.models import Artifact, ExperimentMetadata, Metric, Param, RunMetadata, Tag
from experiments.domain.status import RunStatus


class Lifecycle(Protocol):
    """Shared open/close lifecycle for the driven stores.

    ``close`` MUST be idempotent: the client may close a composite store passed
    as both planes. ``open`` is called once per *distinct* store — the client
    skips the second ``open`` when the artifact store is the same object as the
    metadata store — but idempotency is still recommended for adapters reused
    across clients. Both reference adapters are idempotent (in-memory no-op;
    local-fs ``mkdir(exist_ok=True)``).
    """

    def open(self) -> None: ...
    def close(self) -> None: ...


class MetadataStore(Lifecycle, Protocol):
    """The queryable plane (~ MLflow ``AbstractStore``): experiments, runs,
    params, tags, and append-only metric time-series.

    Concurrency: one instance MUST be safe for concurrent access to *different*
    runs (each run's files/keys are independent). Concurrent writes targeting
    the *same* ``run_id`` are NOT serialized by this port — params, tags and
    run-status are read-modify-write at the adapter — so callers that share one
    run across threads or processes MUST provide their own serialization
    (a per-run lock or single-writer discipline).
    """

    def create_experiment(self, meta: ExperimentMetadata) -> None: ...
    def get_experiment(self, experiment_id: ExperimentID) -> ExperimentMetadata: ...
    def list_experiments(self) -> Sequence[ExperimentMetadata]: ...
    def create_run(self, meta: RunMetadata) -> None: ...
    def get_run(self, run_id: RunID) -> RunMetadata: ...
    def list_runs(self, experiment_id: ExperimentID) -> Sequence[RunMetadata]: ...
    def set_run_status(self, run_id: RunID, status: RunStatus, ended_at: datetime | None = None) -> None: ...
    def set_param(self, run_id: RunID, param: Param) -> None: ...
    def get_params(self, run_id: RunID) -> Mapping[ParamKey, Param]: ...
    def set_tag(self, run_id: RunID, tag: Tag) -> None: ...
    def append_metric(self, run_id: RunID, point: Metric) -> None: ...
    def get_metric_history(self, run_id: RunID, key: MetricKey) -> Sequence[Metric]: ...
    def list_metric_keys(self, run_id: RunID) -> Sequence[MetricKey]: ...


class ArtifactStore(Lifecycle, Protocol):
    """The bytes plane (~ MLflow ``ArtifactRepository``). Run-scoped: ``run_id``
    is an explicit per-call argument, not a constructor argument, matching a
    single shared instance. ``new_storage_key`` lives here because it
    content-addresses bytes; a future store may choose its own scheme.

    Concurrency: safe for concurrent access to *different* runs; concurrent
    writes to the *same* ``run_id`` (``put_artifact``) are NOT serialized.
    """

    def new_storage_key(self, data: bytes) -> StorageKey: ...
    def put_artifact(self, run_id: RunID, artifact: Artifact, data: bytes) -> None: ...
    def open_artifact(self, run_id: RunID, key: ArtifactKey) -> bytes: ...
    def list_artifacts(self, run_id: RunID) -> Sequence[Artifact]: ...


class Backend(MetadataStore, ArtifactStore, Protocol):
    """Composite port: one object that is both planes (one home, one ``./mlruns``
    tree). Retained so existing ``backend: Backend`` annotations and the
    conformance fixture keep checking unchanged. ``LocalFileSystemBackend`` /
    ``InMemoryBackend`` satisfy it structurally with no code change.
    """
