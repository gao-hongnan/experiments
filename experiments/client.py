import mimetypes
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from experiments.domain.entities import Experiment, Run
from experiments.domain.ids import ArtifactKey, ExperimentID, FilePath, ParamValue, RunID, StorageKey
from experiments.domain.models import Artifact, ExperimentMetadata, GitInfo, Metric, Param, RunMetadata, Tag, mint
from experiments.domain.ports import ArtifactStore, Backend, MetadataStore
from experiments.domain.status import RunStatus


@dataclass(frozen=True, slots=True)
class BufferPolicy:
    """Client-side metric buffering policy for hot logging loops.

    Buffers metric points in the client and flushes them to the metadata store
    in batches instead of one write per ``log_metric``. Buffering is purely
    client-side and never enters the ``MetadataStore`` port; params, tags,
    artifacts, and status changes always write through immediately. A buffer flushes when
    ``max_records`` points accumulate, when ``max_interval_s`` seconds have
    elapsed since the last flush (checked on the next ``log_metric``), or
    unconditionally when the run finishes.

    Attributes
    ----------
    max_records
        Flush once this many metric points are buffered.
    max_interval_s
        Flush when at least this many seconds have elapsed since the last flush,
        evaluated on the next metric log. Flushing is synchronous with no
        background thread, so a long idle gap does not flush on its own; the run
        finishing still flushes the tail.

    Notes
    -----
    Buffered metric points live only in process memory. They are NOT durable
    across a hard process exit (OOM-kill, ``SIGKILL``, a segfault in native
    code, or power loss) before the next flush — nothing flushes them on such
    exits. ``buffer=None`` (the default, write-through) is the crash-safe
    choice; params, tags, artifacts, and status changes always write through
    immediately regardless of this policy.
    """

    max_records: int = 100
    max_interval_s: float = 5.0


class _Sink:
    """The ``PersistenceSink`` a ``Run`` forwards through, fanning writes to two
    driven ports: metadata-plane writes to the ``MetadataStore``, artifact bytes
    to the ``ArtifactStore``. Owns client-side metric buffering (metadata plane
    only). In the common composite case ``metadata is artifacts`` (same object).
    """

    def __init__(
        self,
        metadata: MetadataStore,
        artifacts: ArtifactStore,
        buffer: BufferPolicy | None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._metadata = metadata
        self._artifacts = artifacts
        self._buffer = buffer
        self._clock = clock
        self._pending: list[tuple[RunID, Metric]] = []
        self._last_flush = clock()

    def append_metric(self, run_id: RunID, point: Metric) -> None:
        if self._buffer is None:
            self._metadata.append_metric(run_id, point)
            return
        self._pending.append((run_id, point))
        elapsed = self._clock() - self._last_flush
        if len(self._pending) >= self._buffer.max_records or elapsed >= self._buffer.max_interval_s:
            self.flush()

    def set_param(self, run_id: RunID, param: Param) -> None:
        self._metadata.set_param(run_id, param)

    def set_tag(self, run_id: RunID, tag: Tag) -> None:
        self._metadata.set_tag(run_id, tag)

    def set_run_status(self, run_id: RunID, status: RunStatus, ended_at: datetime | None = None) -> None:
        self.flush()
        self._metadata.set_run_status(run_id, status, ended_at)

    def put_artifact(self, run_id: RunID, artifact: Artifact, data: bytes) -> None:
        self._artifacts.put_artifact(run_id, artifact, data)

    def new_storage_key(self, data: bytes) -> StorageKey:
        return self._artifacts.new_storage_key(data)

    def flush(self) -> None:
        for run_id, point in self._pending:
            self._metadata.append_metric(run_id, point)
        self._pending.clear()
        self._last_flush = self._clock()


def _terminal_run_status(exc_type: type[BaseException] | None) -> RunStatus:
    if exc_type is None:
        return RunStatus.COMPLETED
    if issubclass(exc_type, KeyboardInterrupt | SystemExit):
        return RunStatus.KILLED
    return RunStatus.FAILED


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _build_artifact(source: Path, data: bytes, store: ArtifactStore, name: str | None) -> Artifact:
    """Read+hash+mint an ``Artifact`` from a local file via the artifact store.

    Single construction path shared by the object API (``RunContext.log_artifact``)
    and the ambient API (``tracker.log_artifact``) so the two cannot diverge.
    """
    return mint(
        Artifact,
        key=ArtifactKey(name or source.name),
        storage_key=store.new_storage_key(data),
        path=source,
        size_bytes=len(data),
        mime_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream",
    )


class RunContext[MetaT: MetadataStore, ArtT: ArtifactStore]:
    """An active run's handle: the logging verbs plus typed store accessors.

    Obtained from :meth:`ExperimentContext.run` as a context manager; the run is
    finished automatically on context exit. The logging verbs delegate to the
    underlying :class:`~experiments.domain.entities.Run` aggregate.
    """

    def __init__(self, run: Run, metadata: MetaT, artifacts: ArtT, sink: _Sink) -> None:
        self._run = run
        self._metadata = metadata
        self._artifacts = artifacts
        self._sink = sink

    @property
    def run(self) -> Run:
        """The underlying ``Run`` aggregate that every verb delegates to."""
        return self._run

    @property
    def metadata(self) -> MetaT:
        """The metadata store, concrete type preserved for store-specific calls."""
        return self._metadata

    @property
    def artifacts(self) -> ArtT:
        """The artifact store, concrete type preserved for store-specific calls."""
        return self._artifacts

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Record one metric point on the active run's time-series for ``key``.

        Parameters
        ----------
        key
            Metric name. May contain ``/`` (e.g. ``"train/loss"``); rejected if
            it contains path-traversal segments.
        value
            Observed value. ``NaN`` and infinities are recorded as-is.
        step
            Series position. When omitted, the next integer for this ``key`` is
            used (per-key auto-increment); explicit out-of-order steps are
            accepted.

        Raises
        ------
        StateError
            If the run has already finished.
        ValidationError
            If ``key`` is not a valid metric key.
        """
        self._run.log_metric(key, value, step=step)

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        """Record write-once configuration parameters on the run.

        Parameters
        ----------
        params
            Mapping of parameter name to value. Values keep their type
            (``bool``/``int``/``float``/``str``) on round-trip.

        Raises
        ------
        StateError
            If the run has already finished.
        ValidationError
            If a key is re-logged with a different value.
        """
        for key, value in params.items():
            self._run.log_param(key, value)

    def set_tag(self, key: str, value: str) -> None:
        """Attach or overwrite a key/value tag on the run.

        Raises
        ------
        StateError
            If the run has already finished.
        """
        self._run.set_tag(key, value)

    def log_artifact(self, path: FilePath, *, name: str | None = None) -> Artifact:
        """Read a file and store it as a content-addressed run artifact.

        The file bytes are read and hashed here (driving side); the persisted
        artifact handle is content-addressed so identical bytes are stored once.

        Parameters
        ----------
        path
            Path to the local file to store.
        name
            Artifact key to store under. Defaults to the file's basename.

        Returns
        -------
        Artifact
            The stored artifact handle (key, content hash, size, MIME type).

        Raises
        ------
        StateError
            If the run has already finished.
        """
        source = Path(path)
        data = source.read_bytes()
        artifact = _build_artifact(source, data, self._artifacts, name)
        self._run.log_artifact(artifact, data)
        return artifact

    def finish(self, status: RunStatus | None = None) -> None:
        """Transition the run to a terminal status, flushing buffered metrics.

        Parameters
        ----------
        status
            Terminal status to record. Defaults to ``COMPLETED``.

        Notes
        -----
        Idempotent: once the run is terminal, further calls are no-ops. The run
        context manager calls this automatically on exit, so explicit calls are
        only needed to set a non-default status early.
        """
        self._run.finish(status)


class ExperimentContext[MetaT: MetadataStore, ArtT: ArtifactStore]:
    """An active experiment's handle and factory for its runs."""

    def __init__(self, experiment: Experiment, metadata: MetaT, artifacts: ArtT) -> None:
        self._experiment = experiment
        self._metadata = metadata
        self._artifacts = artifacts

    @property
    def id(self) -> ExperimentID:
        """The experiment's identifier."""
        return self._experiment.id

    @property
    def metadata(self) -> MetaT:
        """The metadata store, concrete type preserved."""
        return self._metadata

    @property
    def artifacts(self) -> ArtT:
        """The artifact store, concrete type preserved."""
        return self._artifacts

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._experiment.complete()
        else:
            self._experiment.fail()

    @contextmanager
    def run(
        self,
        name: str | None = None,
        *,
        params: Mapping[str, ParamValue] | None = None,
        buffer: BufferPolicy | None = None,
    ) -> Generator[RunContext[MetaT, ArtT]]:
        """Open a run as a context manager and finish it on exit.

        The run's terminal status is derived from how the block exits: clean
        exit yields ``COMPLETED``; ``KeyboardInterrupt``/``SystemExit`` yields
        ``KILLED`` (and re-raises); any other exception yields ``FAILED``.

        Parameters
        ----------
        name
            Run identifier. A unique ``run_<hex>`` id is generated when omitted.
        params
            Initial write-once parameters logged before the block runs. Invalid
            params finish the run as ``FAILED`` rather than leaving it open.
        buffer
            Opt-in client-side metric buffering. Write-through when omitted.

        Yields
        ------
        RunContext[MetaT, ArtT]
            Handle exposing the logging verbs and the typed stores.

        Examples
        --------
        >>> from experiments import Client
        >>> from experiments.adapters import InMemoryBackend
        >>> client = Client.from_store(InMemoryBackend())
        >>> with client.experiment("demo") as exp:
        ...     with exp.run("r", params={"lr": 0.1}) as run:
        ...         run.log_metric("loss", 0.5)
        """
        run_id = RunID(name or _new_id("run"))
        self._metadata.create_run(mint(RunMetadata, id=run_id, experiment_id=self._experiment.id, name=name or run_id))
        sink = _Sink(self._metadata, self._artifacts, buffer)
        run = Run(run_id=run_id, experiment_id=self._experiment.id, sink=sink)
        ctx = RunContext(run, self._metadata, self._artifacts, sink)
        try:
            if params:
                ctx.log_params(params)
            yield ctx
        except BaseException as exc:
            run.finish(_terminal_run_status(type(exc)))
            raise
        else:
            run.finish(RunStatus.COMPLETED)


class Client[MetaT: MetadataStore, ArtT: ArtifactStore]:
    """The opinionated tracking client bound to a metadata store and an artifact
    store. Carries both concrete store types through ``.metadata`` / ``.artifacts``
    and the run handles, with zero casts.

    For the common one-home case, use :meth:`from_store` with a single composite
    backend (e.g. ``Client.from_store(LocalFileSystemBackend(root))``). Use it as
    a context manager (``with Client.from_store(backend) as client:``) to ensure
    the stores are closed.

    Examples
    --------
    >>> from experiments.adapters import InMemoryBackend
    >>> client = Client.from_store(InMemoryBackend())
    >>> with client.experiment("resnet50") as exp:
    ...     with exp.run("baseline") as run:
    ...         run.log_metric("loss", 0.5)
    """

    def __init__(self, metadata: MetaT, artifacts: ArtT) -> None:
        self._metadata = metadata
        self._artifacts = artifacts
        self._closed = False
        self._metadata.open()
        if id(artifacts) != id(metadata):
            self._artifacts.open()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close both stores. Idempotent; safe to call twice (manual + CM exit)."""
        if self._closed:
            return
        self._metadata.close()
        self._artifacts.close()
        self._closed = True

    @property
    def metadata(self) -> MetaT:
        """The metadata store, concrete type preserved."""
        return self._metadata

    @property
    def artifacts(self) -> ArtT:
        """The artifact store, concrete type preserved."""
        return self._artifacts

    @classmethod
    def from_store[S: Backend](cls, store: S) -> Client[S, S]:
        """Build a client where one composite backend serves both planes.

        The concrete type of ``store`` is preserved on both ``.metadata`` and
        ``.artifacts``.
        """
        return Client[S, S](store, store)

    def experiment(
        self,
        name: str,
        *,
        experiment_id: str | None = None,
        description: str = "",
        git_info: GitInfo | None = None,
        custom_metadata: dict[str, object] | None = None,
    ) -> ExperimentContext[MetaT, ArtT]:
        """Create an experiment and return its context manager.

        Parameters
        ----------
        name
            Human-readable experiment name.
        experiment_id
            Explicit identifier. Defaults to ``name``, else a generated id.
        description
            Optional free-text description stored with the experiment.
        git_info
            Optional structured git provenance stored with the experiment.
        custom_metadata
            Optional free-form metadata (JSON-ish values) stored with the
            experiment.

        Returns
        -------
        ExperimentContext[MetaT, ArtT]
            Context manager that creates runs and records the experiment's
            terminal status on exit.
        """
        exp_id = ExperimentID(experiment_id or name or _new_id("exp"))
        self._metadata.create_experiment(
            mint(
                ExperimentMetadata,
                id=exp_id,
                name=name,
                description=description,
                git_info=git_info,
                custom_metadata=custom_metadata or {},
            )
        )
        return ExperimentContext(Experiment(experiment_id=exp_id, name=name), self._metadata, self._artifacts)


_default_client: Client[Backend, Backend] | None = None


def default_client() -> Client[Backend, Backend]:
    """Return the zero-config client backed by a local-filesystem store at ``./mlruns``.

    Memoized per process so the ambient API does not re-instantiate or ``mkdir``
    a backend on every call. The single ``LocalFileSystemBackend`` serves both
    planes; the on-disk tree is identical to a single-port client. Use
    :func:`experiments.tracker.finish` to close and forget it between jobs.
    """
    global _default_client
    if _default_client is None:
        from experiments.adapters.localfs import LocalFileSystemBackend

        backend: Backend = LocalFileSystemBackend(root=Path("./mlruns"))
        _default_client = Client.from_store(backend)
    return _default_client


def close_default_client() -> None:
    """Close and forget the memoized default client, if any.

    Used by :func:`experiments.tracker.finish` to release the ambient backend
    between jobs; tests use it to reset memoization. Closing is idempotent.
    """
    global _default_client
    if _default_client is not None:
        _default_client.close()
        _default_client = None
