from pathlib import Path
from typing import NewType, TypeIs

ExperimentID = NewType("ExperimentID", str)
RunID = NewType("RunID", str)
MetricKey = NewType("MetricKey", str)
ArtifactKey = NewType("ArtifactKey", str)
StorageKey = NewType("StorageKey", str)
ParamKey = NewType("ParamKey", str)

type ParamValue = bool | int | float | str
type FilePath = Path | str


def _is_safe_segment(value: str) -> bool:
    return bool(value) and not value.startswith("/") and ".." not in value


def is_valid_experiment_id(value: str) -> TypeIs[ExperimentID]:
    return _is_safe_segment(value)


def is_valid_run_id(value: str) -> TypeIs[RunID]:
    return _is_safe_segment(value)


def is_valid_metric_key(value: str) -> TypeIs[MetricKey]:
    if not _is_safe_segment(value):
        return False
    stripped = value.replace("_", "").replace("-", "").replace(".", "").replace("/", "")
    return stripped.isalnum()


def is_valid_param_key(value: str) -> TypeIs[ParamKey]:
    return _is_safe_segment(value)


def is_valid_artifact_key(value: str) -> TypeIs[ArtifactKey]:
    return _is_safe_segment(value)


def is_valid_tag_key(value: str) -> bool:
    return _is_safe_segment(value)
