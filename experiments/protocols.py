from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    def initialize(self, run_id: str, experiment_id: str) -> None:
        """Initialize storage for a run.

        Called when entering the run context. Storage can set up directories,
        connections, or any other initialization needed.

        Parameters
        ----------
        run_id : str
            Unique identifier for the run
        experiment_id : str
            Parent experiment identifier
        """
        ...

    def finalize(self, status: str) -> None:
        """Finalize storage when run completes.

        Called when exiting the run context. Storage can flush buffers,
        close connections, update metadata, etc.

        Parameters
        ----------
        status : str
            Final run status (completed/failed/killed)
        """
        ...
