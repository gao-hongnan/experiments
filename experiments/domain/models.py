from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError as _PydanticValidationError

from experiments.domain.exceptions import ValidationError
from experiments.domain.ids import (
    ArtifactKey,
    ExperimentID,
    MetricKey,
    ParamKey,
    ParamValue,
    RunID,
    StorageKey,
    is_valid_artifact_key,
    is_valid_experiment_id,
    is_valid_metric_key,
    is_valid_param_key,
    is_valid_run_id,
    is_valid_tag_key,
)
from experiments.domain.status import ExperimentStatus, RunStatus


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Param(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: ParamKey
    value: ParamValue

    @field_validator("key")
    @classmethod
    def _check_key(cls: type[Self], value: ParamKey) -> ParamKey:
        if not is_valid_param_key(value):
            raise ValueError(f"invalid param key: {value!r}")
        return value


class Tag(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    value: str

    @field_validator("key")
    @classmethod
    def _check_key(cls: type[Self], value: str) -> str:
        if not is_valid_tag_key(value):
            raise ValueError(f"invalid tag key: {value!r}")
        return value


class Metric(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: MetricKey
    value: float
    step: Annotated[int, Field(ge=0)] = 0
    timestamp: datetime = Field(default_factory=_utc_now)

    @field_validator("key")
    @classmethod
    def _check_key(cls: type[Self], value: MetricKey) -> MetricKey:
        if not is_valid_metric_key(value):
            raise ValueError(f"invalid metric key: {value!r}")
        return value


class Artifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: ArtifactKey
    storage_key: StorageKey
    path: Path
    size_bytes: Annotated[int, Field(ge=0)] = 0
    mime_type: str = "application/octet-stream"
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("key")
    @classmethod
    def _check_key(cls: type[Self], value: ArtifactKey) -> ArtifactKey:
        if not is_valid_artifact_key(value):
            raise ValueError(f"invalid artifact key: {value!r}")
        return value


class GitInfo(BaseModel):
    """Structured git provenance captured at experiment creation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str = ""
    branch: str = ""
    dirty: bool = False
    remote_url: str = ""


class ExperimentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ExperimentID
    name: str
    description: str = ""
    tags: list[Tag] = Field(default_factory=list)
    git_info: GitInfo | None = None
    custom_metadata: dict[str, object] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.RUNNING
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("id")
    @classmethod
    def _check_id(cls: type[Self], value: ExperimentID) -> ExperimentID:
        if not is_valid_experiment_id(value):
            raise ValueError(f"invalid experiment id: {value!r}")
        return value


class RunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: RunID
    experiment_id: ExperimentID
    name: str = ""
    status: RunStatus = RunStatus.RUNNING
    tags: list[Tag] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utc_now)
    ended_at: datetime | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls: type[Self], value: RunID) -> RunID:
        if not is_valid_run_id(value):
            raise ValueError(f"invalid run id: {value!r}")
        return value

    @field_validator("experiment_id")
    @classmethod
    def _check_experiment_id(cls: type[Self], value: ExperimentID) -> ExperimentID:
        if not is_valid_experiment_id(value):
            raise ValueError(f"invalid experiment id: {value!r}")
        return value


def mint[ModelT: BaseModel](model_cls: type[ModelT], /, **fields: object) -> ModelT:
    """Construct a domain model, converting pydantic's ValidationError into the
    project's ValidationError so callers catch one error type at the boundary."""
    try:
        return model_cls(**fields)
    except _PydanticValidationError as exc:
        errors = exc.errors()
        loc = errors[0]["loc"] if errors else ()
        field = str(loc[0]) if loc else model_cls.__name__
        raise ValidationError(field=field, value=fields.get(field), message=str(exc)) from exc
