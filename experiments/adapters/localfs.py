import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final
from urllib.parse import quote

from experiments.domain.exceptions import NotFoundError, StorageError
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

_EXPERIMENT_FILE: Final[str] = "experiment.json"
_RUN_FILE: Final[str] = "run.json"


def _safe_filename(key: str) -> str:
    return quote(key, safe="")


class LocalFileSystemBackend:
    """JSON-tree backend. One experiment.json per experiment, one run.json per run,
    one append-only <key>.jsonl per metric, content-addressed artifact blobs.

    Concurrency: a per-run reentrant lock serializes read-modify-write paths
    (params/tags/status/artifact-index) within one process, honoring the
    ``MetadataStore`` contract for *different* run_ids without contention.
    Cross-process safety on a shared ``./mlruns`` tree is NOT provided (would
    need ``fcntl.flock``); the reference backend is single-process-safe.
    Durability: metadata and blob writes go through temp-file + ``fsync`` +
    atomic ``os.replace`` (with parent-directory fsync), so a crash between
    writes cannot leave a torn metadata file. Appended metric lines are flushed
    + fsync'd per call.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._run_dirs: dict[RunID, Path] = {}
        self._locks: dict[RunID, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def open(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError("cannot create root", operation="write", path=str(self._root)) from exc

    def close(self) -> None:
        return None

    def _lock_for(self, run_id: RunID) -> threading.RLock:
        with self._locks_guard:
            lock = self._locks.get(run_id)
            if lock is None:
                lock = threading.RLock()
                self._locks[run_id] = lock
            return lock

    def _exp_dir(self, experiment_id: ExperimentID) -> Path:
        return self._root / "experiments" / _safe_filename(experiment_id)

    def _find_run_dir(self, run_id: RunID) -> Path:
        cached = self._run_dirs.get(run_id)
        if cached is not None and (cached / _RUN_FILE).exists():
            return cached
        safe = _safe_filename(run_id)
        for run_json in self._root.rglob(f"runs/{safe}/{_RUN_FILE}"):
            run_dir = run_json.parent
            self._run_dirs[run_id] = run_dir
            return run_dir
        raise NotFoundError("run", run_id)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        try:
            dir_fd = os.open(str(path), os.O_RDONLY)
        except OSError:  # pragma: no cover - sandbox disallows opening a dir fd
            return
        try:
            os.fsync(dir_fd)
        except OSError:  # pragma: no cover - filesystem cannot fsync a dir fd
            pass
        finally:
            os.close(dir_fd)

    def _write_json_atomic(self, path: Path, payload: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
                self._fsync_dir(path.parent)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError as exc:
            raise StorageError("write failed", operation="write", path=str(path)) from exc

    def _write_bytes_atomic(self, path: Path, data: bytes) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
                self._fsync_dir(path.parent)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except OSError as exc:
            raise StorageError("write failed", operation="write", path=str(path)) from exc

    def create_experiment(self, meta: ExperimentMetadata) -> None:
        self._write_json_atomic(self._exp_dir(meta.id) / _EXPERIMENT_FILE, meta.model_dump_json())

    def get_experiment(self, experiment_id: ExperimentID) -> ExperimentMetadata:
        path = self._exp_dir(experiment_id) / _EXPERIMENT_FILE
        if not path.exists():
            raise NotFoundError("experiment", experiment_id)
        try:
            return ExperimentMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StorageError("corrupt experiment metadata", operation="read", path=str(path)) from exc

    def list_experiments(self) -> Sequence[ExperimentMetadata]:
        base = self._root / "experiments"
        if not base.exists():
            return ()
        out: list[ExperimentMetadata] = []
        for exp_json in base.rglob(_EXPERIMENT_FILE):
            try:
                out.append(ExperimentMetadata.model_validate_json(exp_json.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                raise StorageError("corrupt experiment metadata", operation="read", path=str(exp_json)) from exc
        return tuple(out)

    def create_run(self, meta: RunMetadata) -> None:
        run_dir = self._exp_dir(meta.experiment_id) / "runs" / _safe_filename(meta.id)
        self._run_dirs[meta.id] = run_dir
        self._write_json_atomic(run_dir / _RUN_FILE, meta.model_dump_json())

    def get_run(self, run_id: RunID) -> RunMetadata:
        path = self._find_run_dir(run_id) / _RUN_FILE
        try:
            return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StorageError("corrupt run metadata", operation="read", path=str(path)) from exc

    def list_runs(self, experiment_id: ExperimentID) -> Sequence[RunMetadata]:
        base = self._exp_dir(experiment_id) / "runs"
        if not base.exists():
            return ()
        out: list[RunMetadata] = []
        for run_json in base.rglob(_RUN_FILE):
            try:
                out.append(RunMetadata.model_validate_json(run_json.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                raise StorageError("corrupt run metadata", operation="read", path=str(run_json)) from exc
        return tuple(out)

    def _update_run(self, run_id: RunID, **updates: object) -> None:
        with self._lock_for(run_id):
            meta = self.get_run(run_id)
            new_meta = meta.model_copy(update=updates)
            self._write_json_atomic(self._find_run_dir(run_id) / _RUN_FILE, new_meta.model_dump_json())

    def set_run_status(self, run_id: RunID, status: RunStatus, ended_at: datetime | None = None) -> None:
        updates: dict[str, object] = {"status": status}
        if ended_at is not None:
            updates["ended_at"] = ended_at
        self._update_run(run_id, **updates)

    def set_param(self, run_id: RunID, param: Param) -> None:
        with self._lock_for(run_id):
            run_dir = self._find_run_dir(run_id)
            params = dict(self.get_params(run_id))
            params[param.key] = param
            payload = json.dumps({k: p.model_dump(mode="json") for k, p in params.items()})
            self._write_json_atomic(run_dir / "params.json", payload)

    def get_params(self, run_id: RunID) -> Mapping[ParamKey, Param]:
        path = self._find_run_dir(run_id) / "params.json"
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StorageError("corrupt params", operation="read", path=str(path)) from exc
        return {ParamKey(k): Param.model_validate(v) for k, v in raw.items()}

    def set_tag(self, run_id: RunID, tag: Tag) -> None:
        with self._lock_for(run_id):
            meta = self.get_run(run_id)
            tags = [t for t in meta.tags if t.key != tag.key]
            tags.append(tag)
            new_meta = meta.model_copy(update={"tags": tags})
            self._write_json_atomic(self._find_run_dir(run_id) / _RUN_FILE, new_meta.model_dump_json())

    def append_metric(self, run_id: RunID, point: Metric) -> None:
        path = self._find_run_dir(run_id) / "metrics" / f"{_safe_filename(point.key)}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(point.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise StorageError("metric append failed", operation="write", path=str(path)) from exc

    def get_metric_history(self, run_id: RunID, key: MetricKey) -> Sequence[Metric]:
        path = self._find_run_dir(run_id) / "metrics" / f"{_safe_filename(key)}.jsonl"
        if not path.exists():
            return ()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise StorageError("corrupt metric log", operation="read", path=str(path)) from exc
        points: list[Metric] = []
        for line in lines:
            if not line:
                continue
            try:
                points.append(Metric.model_validate_json(line))
            except ValueError:
                continue
        return tuple(points)

    def list_metric_keys(self, run_id: RunID) -> Sequence[MetricKey]:
        metrics_dir = self._find_run_dir(run_id) / "metrics"
        if not metrics_dir.exists():
            return ()
        out: list[MetricKey] = []
        try:
            for jsonl in metrics_dir.glob("*.jsonl"):
                first = jsonl.read_text(encoding="utf-8").splitlines()[:1]
                if first:
                    out.append(Metric.model_validate_json(first[0]).key)
        except (OSError, ValueError) as exc:
            raise StorageError("corrupt metric log", operation="read", path=str(metrics_dir)) from exc
        return tuple(out)

    def new_storage_key(self, data: bytes) -> StorageKey:
        return StorageKey(hashlib.sha256(data).hexdigest())

    def put_artifact(self, run_id: RunID, artifact: Artifact, data: bytes) -> None:
        run_dir = self._find_run_dir(run_id)
        blob = run_dir / "artifacts" / "blobs" / artifact.storage_key
        self._write_bytes_atomic(blob, data)
        with self._lock_for(run_id):
            index = self._read_artifact_index(run_dir)
            index[artifact.key] = artifact
            payload = json.dumps({k: a.model_dump(mode="json") for k, a in index.items()})
            self._write_json_atomic(run_dir / "artifacts" / "index.json", payload)

    def _read_artifact_index(self, run_dir: Path) -> dict[ArtifactKey, Artifact]:
        path = run_dir / "artifacts" / "index.json"
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StorageError("corrupt artifact index", operation="read", path=str(path)) from exc
        return {ArtifactKey(k): Artifact.model_validate(v) for k, v in raw.items()}

    def open_artifact(self, run_id: RunID, key: ArtifactKey) -> bytes:
        run_dir = self._find_run_dir(run_id)
        index = self._read_artifact_index(run_dir)
        artifact = index.get(key)
        if artifact is None:
            raise NotFoundError("artifact", key)
        blob = run_dir / "artifacts" / "blobs" / artifact.storage_key
        try:
            return blob.read_bytes()
        except OSError as exc:
            raise StorageError("artifact read failed", operation="read", path=str(blob)) from exc

    def list_artifacts(self, run_id: RunID) -> Sequence[Artifact]:
        return tuple(self._read_artifact_index(self._find_run_dir(run_id)).values())
