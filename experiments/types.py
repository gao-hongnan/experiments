from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, NewType, TypeAlias, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, field_validator

EXPERIMENT_METADATA_FILE: Final[str] = "experiment.json"
RUN_METADATA_FILE: Final[str] = "run.json"

ExperimentID = NewType("ExperimentID", str)
RunID = NewType("RunID", str)
MetricKey = NewType("MetricKey", str)
ArtifactKey = NewType("ArtifactKey", str)
StorageKey = NewType("StorageKey", str)

FilePath: TypeAlias = Path | str


class ExperimentStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


def is_valid_experiment_id(value: str) -> TypeGuard[ExperimentID]:
    return bool(value and not value.startswith("/") and ".." not in value)


def is_valid_run_id(value: str) -> TypeGuard[RunID]:
    return bool(value and not value.startswith("/") and ".." not in value)


def is_valid_metric_key(value: str) -> TypeGuard[MetricKey]:
    return bool(
        value and value.replace("_", "").replace("-", "").replace(".", "").isalnum()
    )


class ExperimentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ExperimentID
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: ExperimentStatus = ExperimentStatus.RUNNING
    git_info: dict[str, Any] = Field(default_factory=dict)
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: ExperimentID) -> ExperimentID:
        if not is_valid_experiment_id(v):
            raise ValueError(f"Invalid experiment ID: {v}")
        return v


class RunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: RunID
    experiment_id: ExperimentID
    name: str = ""
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: RunID) -> RunID:
        if not is_valid_run_id(v):
            raise ValueError(f"Invalid run ID: {v}")
        return v


class Metric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: MetricKey
    value: float
    step: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: MetricKey) -> MetricKey:
        if not is_valid_metric_key(v):
            raise ValueError(f"Invalid metric key: {v}")
        return v


class Artifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: ArtifactKey
    storage_key: StorageKey
    path: Path
    size_bytes: int = 0
    mime_type: str = "application/octet-stream"
    created_at: datetime = Field(default_factory=datetime.now)
