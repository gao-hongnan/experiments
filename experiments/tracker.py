from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator, TypeVar, cast

from experiments.experiment import Experiment, Run
from experiments.protocols import Storage

StorageT = TypeVar("StorageT", bound=Storage)

# NOTE: Context variables for async safety (better than thread-local)
# We use Storage as the type parameter since it's the bound for StorageT
_active_experiment: contextvars.ContextVar[Experiment[Storage] | None] = contextvars.ContextVar(
    "_active_experiment", default=None
)
_active_run: contextvars.ContextVar[Run[Storage] | None] = contextvars.ContextVar("_active_run", default=None)


@contextmanager
def experiment(
    name: str,
    storage: StorageT,
    id: str | None = None,
    **kwargs: Any,
) -> Iterator[Experiment[StorageT]]:
    """Context manager for creating and managing experiments.

    Parameters
    ----------
    name : str
        Experiment name (required)
    storage : Storage
        Storage backend (required - users must provide their own storage implementation)
    id : str | None
        Optional experiment ID (auto-generated if not provided)
    **kwargs : Any
        Additional arguments passed to Experiment constructor

    Yields
    ------
    Experiment[StorageT]
        The experiment instance with preserved storage type

    Examples
    --------
    Single-run experiment:
    >>> class MyStorage:
    ...     def initialize(self, run_id: str, experiment_id: str) -> None: ...
    ...     def finalize(self, status: str) -> None: ...
    ...     def log_metric(self, key: str, value: float) -> None: ...
    >>>
    >>> storage = MyStorage()
    >>> with experiment("training", storage=storage) as exp:
    ...     with run() as r:
    ...         r.storage.log_metric("loss", 0.5)

    Multi-run experiment:
    >>> with experiment("hyperparam_search", storage=storage) as exp:
    ...     for lr in [0.001, 0.01]:
    ...         with run(f"lr_{lr}") as r:
    ...             r.storage.log_metric("loss", 0.5)
    """
    finish()

    exp = Experiment(
        name=name,
        storage=storage,
        id=id,
        **kwargs,
    )

    # NOTE: Cast to base Storage type when storing in context var (intentional type erasure)
    _active_experiment.set(cast(Experiment[Storage], exp))

    try:
        yield exp
    finally:
        _active_experiment.set(None)


@contextmanager
def run(name: str | None = None, **metadata: Any) -> Iterator[Run[Storage]]:
    """Context manager for creating and managing runs within the active experiment.

    Parameters
    ----------
    name : str | None
        Optional name for the run
    **metadata : Any
        Additional metadata to store with the run

    Yields
    ------
    Run[Storage]
        The run instance

    Examples
    --------
    >>> with experiment("training") as exp:
    ...     with run("epoch_1", learning_rate=0.001) as r:
    ...         r.storage.log_metric("loss", 0.5)
    ...         print(r.metadata["learning_rate"])  # Access metadata
    """
    exp = _active_experiment.get()
    if not exp:
        raise RuntimeError("No active experiment. Use experiment() context manager first.")

    # NOTE: Use the experiment's run() context manager
    with exp.run(name or "", **metadata) as r:
        # NOTE: Cast to base Storage type when storing in context var (intentional type erasure)
        _active_run.set(cast(Run[Storage], r))
        try:
            yield r
        finally:
            _active_run.set(None)


def finish() -> None:
    """Clean up active run and experiment."""
    _active_run.set(None)
    _active_experiment.set(None)


def get_experiment() -> Experiment[Storage] | None:
    """Get the active experiment (for advanced usage).

    Returns
    -------
    Experiment[Storage] | None
        The active experiment or None if no experiment is active

    Examples
    --------
    >>> exp = get_experiment()
    >>> if exp:
    ...     print(f"Active experiment: {exp.id}")
    """
    return _active_experiment.get()


def get_run() -> Run[Storage] | None:
    """Get the active run (for advanced usage).

    Returns
    -------
    Run[Storage] | None
        The active run or None if no run is active

    Examples
    --------
    >>> r = get_run()
    >>> if r:
    ...     print(f"Active run: {r.id}")
    """
    return _active_run.get()


def is_active() -> bool:
    """Check if an experiment is active.

    Returns
    -------
    bool
        True if an experiment is currently active

    Examples
    --------
    >>> if is_active():
    ...     print("Experiment is active")
    """
    return _active_experiment.get() is not None
