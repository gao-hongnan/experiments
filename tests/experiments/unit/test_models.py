"""Unit tests for the pydantic domain models and the mint() boundary converter."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from experiments.domain.exceptions import ValidationError
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
    GitInfo,
    Metric,
    Param,
    RunMetadata,
    Tag,
    mint,
)


def test_mint_converts_pydantic_error_to_project_error() -> None:
    with pytest.raises(ValidationError) as info:
        mint(Metric, key=MetricKey("bad key"), value=1.0)
    assert info.value.field == "key"


def test_param_key_validator_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        mint(Param, key=ParamKey("../x"), value=1)


def test_tag_key_validator_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        mint(Tag, key="../x", value="v")


def test_metric_key_validator_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        mint(Metric, key=MetricKey("bad key"), value=1.0)


def test_metric_step_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        mint(Metric, key=MetricKey("loss"), value=1.0, step=-1)


def test_artifact_key_validator_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        mint(
            Artifact,
            key=ArtifactKey("../x"),
            storage_key=StorageKey("deadbeef"),
            path="/tmp/x",
        )


def test_experiment_metadata_accepts_provenance() -> None:
    meta = mint(
        ExperimentMetadata,
        id=ExperimentID("e1"),
        name="resnet",
        git_info=GitInfo(commit="abc123", branch="main", dirty=True),
        custom_metadata={"team": "ml", "n": 3},
    )
    assert meta.git_info is not None
    assert meta.git_info.commit == "abc123"
    assert meta.git_info.dirty is True
    assert meta.custom_metadata == {"team": "ml", "n": 3}


def test_experiment_metadata_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        mint(ExperimentMetadata, id=ExperimentID("e1"), name="n", unknown="x")  # type: ignore[call-arg]


def test_experiment_metadata_rejects_bad_id() -> None:
    with pytest.raises(ValidationError):
        mint(ExperimentMetadata, id=ExperimentID("../x"), name="n")


def test_run_metadata_validates_ids() -> None:
    with pytest.raises(ValidationError):
        mint(RunMetadata, id=RunID("../x"), experiment_id=ExperimentID("e1"))
    with pytest.raises(ValidationError):
        mint(RunMetadata, id=RunID("r1"), experiment_id=ExperimentID("../x"))


def test_models_are_frozen() -> None:
    metric = mint(Metric, key=MetricKey("loss"), value=1.0)
    with pytest.raises(PydanticValidationError):
        metric.value = 2.0  # type: ignore[misc]


def test_metric_round_trips_json() -> None:
    metric = mint(Metric, key=MetricKey("train/loss"), value=1.5, step=3)
    again = Metric.model_validate_json(metric.model_dump_json())
    assert again == metric


def test_experiment_metadata_round_trips_json_with_provenance() -> None:
    meta = mint(
        ExperimentMetadata,
        id=ExperimentID("e1"),
        name="n",
        git_info=GitInfo(commit="c"),
        custom_metadata={"k": "v"},
    )
    again = ExperimentMetadata.model_validate_json(meta.model_dump_json())
    assert again == meta
    assert again.git_info is not None
    assert again.git_info.commit == "c"


def test_param_value_keeps_bool_type() -> None:
    param = mint(Param, key=ParamKey("flag"), value=True)
    assert param.value is True
    dumped = Param.model_validate_json(param.model_dump_json())
    assert dumped.value is True
