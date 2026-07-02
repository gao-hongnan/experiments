import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Literal

from experiments.client import (
    BufferPolicy,
    Client,
    ExperimentContext,
    RunContext,
    _build_artifact,
    close_default_client,
    default_client,
)
from experiments.domain.entities import Run
from experiments.domain.exceptions import (
    NoActiveClientError,
    NoActiveExperimentError,
    NoActiveRunError,
)
from experiments.domain.ids import FilePath, ParamValue
from experiments.domain.models import Artifact, GitInfo
from experiments.domain.ports import Backend

type MissingExperimentPolicy = Literal["create_default", "raise"]
"""How :func:`run` behaves with no active experiment: auto-create a "Default"
one (the zero-config path) or raise :class:`NoActiveExperimentError`."""

_active_experiment: ContextVar[ExperimentContext[Backend, Backend] | None] = ContextVar(
    "experiments_active_experiment", default=None
)
_active_run: ContextVar[Run | None] = ContextVar("experiments_active_run", default=None)
_ambient_client: ContextVar[Client[Backend, Backend] | None] = ContextVar("experiments_ambient_client", default=None)


def _require_run() -> Run:
    run = _active_run.get()
    if run is None:
        raise NoActiveRunError("log without an active run")
    return run


@contextmanager
def experiment(
    name: str,
    *,
    client: Client[Backend, Backend] | None = None,
    description: str = "",
    git_info: GitInfo | None = None,
    custom_metadata: dict[str, object] | None = None,
) -> Generator[ExperimentContext[Backend, Backend]]:
    """Open an ambient experiment so module-level verbs target it.

    Sets the experiment (and its client) as the active context for the duration
    of the block, restoring any previously-active experiment on exit
    (experiments nest correctly). The experiment is marked ``COMPLETED`` on
    clean exit and ``FAILED`` if the block raises.

    Parameters
    ----------
    name
        Human-readable experiment name.
    client
        Client whose backend stores this experiment. Defaults to the zero-config
        local-filesystem client at ``./mlruns``.
    description
        Optional free-text description stored with the experiment.
    git_info
        Optional structured git provenance stored with the experiment.
    custom_metadata
        Optional free-form metadata (JSON-ish values) stored with the experiment.

    Yields
    ------
    ExperimentContext[Backend, Backend]
        The active experiment handle; also reachable via :func:`active_experiment`.

    Examples
    --------
    >>> import experiments as ex
    >>> with ex.experiment("resnet50"):
    ...     with ex.run("baseline"):
    ...         ex.log_metric("loss", 0.5)
    """
    active_client = client if client is not None else default_client()
    client_token: Token[Client[Backend, Backend] | None] = _ambient_client.set(active_client)
    try:
        ctx = active_client.experiment(
            name, description=description, git_info=git_info, custom_metadata=custom_metadata
        )
        exp_token: Token[ExperimentContext[Backend, Backend] | None] = _active_experiment.set(ctx)
        try:
            with ctx:
                yield ctx
        finally:
            _active_experiment.reset(exp_token)
    finally:
        _ambient_client.reset(client_token)


@contextmanager
def run(
    name: str | None = None,
    *,
    params: Mapping[str, ParamValue] | None = None,
    buffer: BufferPolicy | None = None,
    missing_experiment: MissingExperimentPolicy = "create_default",
) -> Generator[RunContext[Backend, Backend]]:
    """Open an ambient run so module-level verbs target it.

    Sets the run as the active context for the block and finishes it on exit
    (``COMPLETED`` clean, ``KILLED`` on ``KeyboardInterrupt``/``SystemExit``,
    ``FAILED`` otherwise). With no active experiment, the default policy
    auto-creates a transient "Default" experiment (and a zero-config client);
    pass ``missing_experiment="raise"`` to require one instead.

    Parameters
    ----------
    name
        Run identifier. A unique ``run_<hex>`` id is generated when omitted.
    params
        Initial write-once parameters logged before the block runs.
    buffer
        Opt-in client-side metric buffering. Write-through when omitted.
    missing_experiment
        ``"create_default"`` (default) to auto-create an experiment when none is
        active, or ``"raise"`` to raise :class:`NoActiveExperimentError`.

    Yields
    ------
    RunContext[Backend, Backend]
        The active run handle; also reachable via :func:`active_run`.

    Raises
    ------
    NoActiveExperimentError
        If no experiment is active and ``missing_experiment="raise"``.

    Examples
    --------
    >>> import experiments as ex
    >>> with ex.run("quick") as run:
    ...     run.log_metric("loss", 0.5)
    """
    exp = _active_experiment.get()
    own_experiment: ExperimentContext[Backend, Backend] | None = None
    exp_token: Token[ExperimentContext[Backend, Backend] | None] | None = None
    client_token: Token[Client[Backend, Backend] | None] | None = None
    if exp is None:
        if missing_experiment == "raise":
            raise NoActiveExperimentError("run() with no active experiment")
        client = default_client()
        client_token = _ambient_client.set(client)
        exp = client.experiment("Default")
        exp.__enter__()
        exp_token = _active_experiment.set(exp)
        own_experiment = exp
    try:
        with exp.run(name, params=params, buffer=buffer) as ctx:
            run_token: Token[Run | None] = _active_run.set(ctx.run)
            try:
                yield ctx
            finally:
                _active_run.reset(run_token)
    finally:
        if own_experiment is not None and exp_token is not None and client_token is not None:
            exc_type, exc_value, tb = sys.exc_info()
            own_experiment.__exit__(exc_type, exc_value, tb)
            _active_experiment.reset(exp_token)
            _ambient_client.reset(client_token)


def log_metric(key: str, value: float, *, step: int | None = None) -> None:
    """Record a metric point on the ambient active run.

    Parameters
    ----------
    key
        Metric name; may contain ``/`` (e.g. ``"train/loss"``).
    value
        Observed value. ``NaN`` and infinities are recorded as-is.
    step
        Series position; per-key auto-increment when omitted.

    Raises
    ------
    NoActiveRunError
        If no run is active.
    StateError
        If the active run has finished.
    ValidationError
        If ``key`` is not a valid metric key.
    """
    _require_run().log_metric(key, value, step=step)


def log_params(params: Mapping[str, ParamValue]) -> None:
    """Record write-once parameters on the ambient active run.

    Raises
    ------
    NoActiveRunError
        If no run is active.
    StateError
        If the active run has finished.
    ValidationError
        If a key is re-logged with a different value.
    """
    run_obj = _require_run()
    for key, value in params.items():
        run_obj.log_param(key, value)


def set_tag(key: str, value: str) -> None:
    """Attach or overwrite a key/value tag on the ambient active run.

    Raises
    ------
    NoActiveRunError
        If no run is active.
    StateError
        If the active run has finished.
    """
    _require_run().set_tag(key, value)


def log_artifact(path: FilePath, *, name: str | None = None) -> Artifact:
    """Store a local file as a content-addressed artifact on the active run.

    Parameters
    ----------
    path
        Path to the local file to store.
    name
        Artifact key. Defaults to the file's basename.

    Returns
    -------
    Artifact
        The stored artifact handle.

    Raises
    ------
    NoActiveRunError
        If no run is active.
    NoActiveClientError
        If no client is active.
    StateError
        If the active run has finished.
    """
    run_obj = _require_run()
    client = _ambient_client.get()
    if client is None:
        raise NoActiveClientError("log_artifact with no active client")
    source = Path(path)
    data = source.read_bytes()
    artifact = _build_artifact(source, data, client.artifacts, name)
    run_obj.log_artifact(artifact, data)
    return artifact


def active_run() -> Run | None:
    """Return the ambient active run, or ``None`` if none is active.

    Returns
    -------
    Run | None
        The active run aggregate. Backend access is intentionally not exposed
        here; use the object API (:class:`~experiments.client.RunContext`) for a
        typed backend.
    """
    return _active_run.get()


def active_experiment() -> ExperimentContext[Backend, Backend] | None:
    """Return the ambient active experiment handle, or ``None`` if none is active."""
    return _active_experiment.get()


def finish() -> None:
    """Tear down all ambient context, clearing the active run, experiment, and
    client, and closing the memoized zero-config default client (if one was
    created) so long-lived processes release the backend between jobs.

    Notes
    -----
    This is an explicit reset (e.g. between independent training scripts in one
    process), distinct from context-manager exit which restores the parent
    scope. Calling it inside an active ``with`` block abandons that block's
    context. Only the process-wide default client is closed here; a client you
    constructed explicitly remains yours to close.
    """
    _active_run.set(None)
    _active_experiment.set(None)
    _ambient_client.set(None)
    close_default_client()
