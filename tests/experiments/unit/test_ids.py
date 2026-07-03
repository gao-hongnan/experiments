"""Unit tests for the id/key validity predicates."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from experiments.domain.ids import (
    ArtifactKey,
    ExperimentID,
    MetricKey,
    ParamKey,
    RunID,
    is_valid_artifact_key,
    is_valid_experiment_id,
    is_valid_metric_key,
    is_valid_param_key,
    is_valid_run_id,
    is_valid_tag_key,
)

SAFE_PREDICATES = [
    is_valid_experiment_id,
    is_valid_run_id,
    is_valid_param_key,
    is_valid_artifact_key,
    is_valid_tag_key,
]


@pytest.mark.parametrize("pred", SAFE_PREDICATES)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("valid", True),
        ("train/loss", True),
        ("a-b_c.d", True),
        ("with space", True),
        ("", False),
        ("/leading", False),
        ("../escape", False),
        ("a/../b", False),
    ],
)
def test_safe_segment_predicates(pred: Callable[[str], bool], value: str, expected: bool) -> None:
    assert pred(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("loss", True),
        ("train/loss", True),
        ("a-b", True),
        ("a.b", True),
        ("a_b", True),
        ("", False),
        ("/loss", False),
        ("../x", False),
        ("with space", False),
        ("a b", False),
    ],
)
def test_metric_key_alnum_constraint(value: str, expected: bool) -> None:
    assert is_valid_metric_key(value) is expected


def test_newtypes_round_trip_as_underlying_str() -> None:
    assert ExperimentID("e") == "e"
    assert MetricKey("loss") == "loss"
    assert ParamKey("lr") == "lr"
    assert ArtifactKey("model") == "model"
    assert RunID("r1") == "r1"
