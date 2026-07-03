from enum import StrEnum
from typing import assert_never


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"

    @property
    def is_terminal(self) -> bool:
        match self:
            case RunStatus.RUNNING:
                return False
            case RunStatus.COMPLETED | RunStatus.FAILED | RunStatus.KILLED:
                return True
            case _:  # pragma: no cover - exhaustive match; unreachable while the enum is closed
                assert_never(self)  # pragma: no cover - exhaustive match; unreachable while the enum is closed

    def can_transition_to(self, target: RunStatus) -> bool:
        match self:
            case RunStatus.RUNNING:
                return target.is_terminal
            case RunStatus.COMPLETED | RunStatus.FAILED | RunStatus.KILLED:
                return False
            case _:  # pragma: no cover - exhaustive match; unreachable while the enum is closed
                assert_never(self)  # pragma: no cover - exhaustive match; unreachable while the enum is closed


class ExperimentStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        match self:
            case ExperimentStatus.RUNNING:
                return False
            case ExperimentStatus.COMPLETED | ExperimentStatus.FAILED:
                return True
            case _:  # pragma: no cover - exhaustive match; unreachable while the enum is closed
                assert_never(self)  # pragma: no cover - exhaustive match; unreachable while the enum is closed

    def can_transition_to(self, target: ExperimentStatus) -> bool:
        match self:
            case ExperimentStatus.RUNNING:
                return target.is_terminal
            case ExperimentStatus.COMPLETED | ExperimentStatus.FAILED:
                return False
            case _:  # pragma: no cover - exhaustive match; unreachable while the enum is closed
                assert_never(self)  # pragma: no cover - exhaustive match; unreachable while the enum is closed
