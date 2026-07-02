"""Shared pytest fixtures for the experiments test suite."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from experiments import Client
from experiments.adapters import InMemoryBackend
from experiments.adapters.localfs import LocalFileSystemBackend
from experiments.client import close_default_client
from experiments.domain.ids import RunID
from experiments.domain.models import Artifact, Metric, Param, Tag
from experiments.domain.status import RunStatus


class RecordingSink:
    """A recording ``PersistenceSink`` for unit-testing the ``Run`` aggregate.

    Captures every forwarded write so entity tests can assert what the aggregate
    emitted without any I/O. Satisfies ``PersistenceSink`` structurally.
    """

    def __init__(self) -> None:
        self.metrics: list[tuple[RunID, Metric]] = []
        self.params: list[tuple[RunID, Param]] = []
        self.tags: list[tuple[RunID, Tag]] = []
        self.artifacts: list[tuple[RunID, Artifact, bytes]] = []
        self.statuses: list[tuple[RunID, RunStatus, datetime | None]] = []

    def append_metric(self, run_id: RunID, point: Metric) -> None:
        self.metrics.append((run_id, point))

    def set_param(self, run_id: RunID, param: Param) -> None:
        self.params.append((run_id, param))

    def put_artifact(self, run_id: RunID, artifact: Artifact, data: bytes) -> None:
        self.artifacts.append((run_id, artifact, data))

    def set_tag(self, run_id: RunID, tag: Tag) -> None:
        self.tags.append((run_id, tag))

    def set_run_status(self, run_id: RunID, status: RunStatus, ended_at: datetime | None = None) -> None:
        self.statuses.append((run_id, status, ended_at))


@pytest.fixture
def recording_sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def in_memory_client() -> Client:
    """A client over an in-memory backend (fast, no FS)."""
    return Client.from_store(InMemoryBackend())


@pytest.fixture
def localfs_client(tmp_path: Path) -> Client:
    """A client over a local-fs backend rooted at a per-test tmp dir."""
    return Client.from_store(LocalFileSystemBackend(tmp_path))


@pytest.fixture
def isolated_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run with cwd in tmp and the memoized default client reset before/after,
    so zero-config ``default_client()`` tests neither pollute ``./mlruns`` nor
    leak the singleton across tests."""
    close_default_client()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    close_default_client()
