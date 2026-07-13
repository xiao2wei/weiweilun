"""Feasibility-first validation selection for frozen controller settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ValidationLimits:
    timeout_rate_limit: float
    failure_rate_limit: float
    coverage_rate_minimum: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class ValidationCandidate:
    config_id: str
    hard_invariant_failures: int
    timeout_rate_ucb: float
    failure_rate_ucb: float
    coverage_rate_lcb: float
    mean_task_cost: float
    mean_failure_rate: float
    mean_vehicle_power_w: float
    mean_rsu_power_w: float
    model_complexity: float

    def __post_init__(self) -> None:
        if not self.config_id:
            raise ValueError("config_id is required")
        if (
            isinstance(self.hard_invariant_failures, bool)
            or self.hard_invariant_failures < 0
        ):
            raise ValueError("hard_invariant_failures must be a nonnegative integer")
        for name in (
            "timeout_rate_ucb",
            "failure_rate_ucb",
            "coverage_rate_lcb",
            "mean_failure_rate",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in (
            "mean_task_cost",
            "mean_vehicle_power_w",
            "mean_rsu_power_w",
            "model_complexity",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class SelectionReport:
    selected_config_id: str | None
    feasible_config_ids: tuple[str, ...]
    rejected: dict[str, tuple[str, ...]]
    ordered_feasible: tuple[str, ...]
    rule: str = (
        "hard/long-term-confidence feasibility, then cost, failure rate, "
        "total power, complexity, config_id"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_feasible_validation_candidate(
    candidates: Iterable[ValidationCandidate], limits: ValidationLimits
) -> SelectionReport:
    """Select frozen hyperparameters using Scheme 1's feasibility-first rule."""

    rows = list(candidates)
    ids = [row.config_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("validation candidate config_id values must be unique")
    rejected: dict[str, tuple[str, ...]] = {}
    feasible: list[ValidationCandidate] = []
    for row in rows:
        reasons: list[str] = []
        if row.hard_invariant_failures:
            reasons.append("HARD_INVARIANT_FAILURE")
        if row.timeout_rate_ucb > limits.timeout_rate_limit:
            reasons.append("TIMEOUT_RATE_UCB")
        if row.failure_rate_ucb > limits.failure_rate_limit:
            reasons.append("FAILURE_RATE_UCB")
        if row.coverage_rate_lcb < limits.coverage_rate_minimum:
            reasons.append("COVERAGE_RATE_LCB")
        if reasons:
            rejected[row.config_id] = tuple(reasons)
        else:
            feasible.append(row)
    ordered = sorted(
        feasible,
        key=lambda row: (
            row.mean_task_cost,
            row.mean_failure_rate,
            row.mean_vehicle_power_w + row.mean_rsu_power_w,
            row.model_complexity,
            row.config_id,
        ),
    )
    ordered_ids = tuple(row.config_id for row in ordered)
    return SelectionReport(
        selected_config_id=None if not ordered else ordered[0].config_id,
        feasible_config_ids=tuple(sorted(row.config_id for row in feasible)),
        rejected=dict(sorted(rejected.items())),
        ordered_feasible=ordered_ids,
    )


__all__ = [
    "SelectionReport",
    "ValidationCandidate",
    "ValidationLimits",
    "select_feasible_validation_candidate",
]
