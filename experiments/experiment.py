"""Experiment tracking with pure lifecycle management.

This module implements the simplest possible design where Run and Experiment
are pure lifecycle managers. Storage handles ALL operations.

Design Philosophy:
    - Run/Experiment ONLY manage lifecycle (context management)
    - Storage handles ALL I/O operations
    - Users access storage directly via run.storage
    - Storage can have ANY methods it wants
    - No artificial separation between experiment and run storage

Examples
--------
>>> # Storage can have ANY methods
>>> class MyStorage:
...     def initialize(self, run_id: str, experiment_id: str) -> None:
...         # Setup for run
...         pass
...     def finalize(self, status: str) -> None:
...         # Cleanup for run
...         pass
...     # Add ANY methods you want!
...     def log_metric(self, key: str, value: float) -> None: ...
...     def save_model(self, model: Any) -> None: ...
...     def custom_operation(self) -> None: ...
>>>
>>> storage = MyStorage()
>>> exp = Experiment("test", storage=storage)
>>>
>>> with exp.run("training") as run:
...     # Direct access to ALL storage methods!
...     run.storage.log_metric("loss", 0.5)
...     run.storage.save_model(model)
...     run.storage.custom_operation()
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Generic, Iterator, Self, TypeVar

from experiments.protocols import Storage
from experiments.types import ExperimentID, RunID

StorageT = TypeVar("StorageT", bound=Storage)


class Run(Generic[StorageT]):
    """A run is just a lifecycle manager.

    The Run class is a context manager that calls storage.initialize() on enter
    and storage.finalize() on exit. All actual operations are done
    through storage, which users access directly.

    Parameters
    ----------
    run_id : RunID
        Unique identifier for the run
    storage : Storage
        Storage backend that handles ALL operations
    experiment_id : ExperimentID
        Parent experiment identifier

    Attributes
    ----------
    storage : StorageT
        Direct access to storage - can have any methods!

    Examples
    --------
    >>> with exp.run("training") as run:
    ...     # Storage can have ANY methods - you're not limited!
    ...     run.storage.log_metric("loss", 0.5)
    ...     run.storage.save_checkpoint(model, optimizer)
    ...     run.storage.whatever_method_you_defined()
    """

    def __init__(
        self,
        run_id: RunID,
        storage: StorageT,
        experiment_id: ExperimentID,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = run_id
        self.experiment_id = experiment_id
        self.storage = storage
        self._completed = False

        self.metadata = {
            **(metadata or {}),
            "name": metadata.get("name", str(self.id)) if metadata else str(self.id),
            "id": str(self.id),
            "experiment_id": str(self.experiment_id),
        }

    def __enter__(self) -> Self:
        """Enter context - initialize storage."""
        self.storage.initialize(run_id=str(self.id), experiment_id=str(self.experiment_id))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit context - finalize storage."""
        status = "failed" if exc_type else "completed"
        if not self._completed:
            self.storage.finalize(status=status)
            self._completed = True


class Experiment(Generic[StorageT]):
    """An experiment is just a lifecycle manager.

    The Experiment class creates Run instances and passes them the storage.
    That's its only job. All actual operations are handled by storage.

    Parameters
    ----------
    name : str
        Name of the experiment
    storage : StorageT
        Storage backend for ALL operations
    id : str | None, optional
        Unique identifier (auto-generated if not provided)

    Examples
    --------
    >>> storage = MyCustomStorage()
    >>> exp = Experiment("training", storage=storage)
    >>>
    >>> with exp.run() as run:
    ...     run.storage.do_anything()
    """

    def __init__(
        self,
        name: str,
        storage: StorageT,
        id: str | None = None,
        **metadata: Any,
    ) -> None:
        self.id = ExperimentID(id) if id else ExperimentID(f"{name}_{uuid.uuid4().hex[:8]}")
        self.name = name
        self._storage = storage
        self._completed = False

        self.metadata = {
            **(metadata or {}),
            "name": name,
            "id": str(self.id),
        }

    @contextmanager
    def run(self, name: str = "", **metadata: Any) -> Iterator[Run[StorageT]]:
        """Create and manage a run.

        Parameters
        ----------
        name : str, optional
            Name for the run (auto-generated if not provided)
        **metadata : Any
            Additional metadata to store with the run

        Yields
        ------
        Run[StorageT]
            Run instance with direct storage access

        Examples
        --------
        >>> with exp.run("epoch_1", learning_rate=0.001) as run:
        ...     run.storage.log_metric("loss", 0.5)
        ...     print(run.metadata["learning_rate"])
        """
        run_id = RunID(name) if name else RunID(f"run_{uuid.uuid4().hex[:8]}")

        run_metadata = {
            **metadata,
            "name": name if name else str(run_id),
            "id": str(run_id),
        }

        run_obj = Run(
            run_id=run_id,
            storage=self._storage,
            experiment_id=self.id,
            metadata=run_metadata,
        )
        with run_obj as run:
            yield run

    def __enter__(self) -> Self:
        """Enter experiment context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit experiment context."""
        self._completed = True
