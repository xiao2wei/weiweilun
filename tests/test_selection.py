from __future__ import annotations

import pytest

from privacy_edge_sim.selection import (
    ValidationCandidate,
    ValidationLimits,
    select_feasible_validation_candidate,
)


def _candidate(config_id: str, **changes: float | int) -> ValidationCandidate:
    values: dict[str, float | int | str] = {
        "config_id": config_id,
        "hard_invariant_failures": 0,
        "timeout_rate_ucb": 0.05,
        "failure_rate_ucb": 0.08,
        "coverage_rate_lcb": 0.90,
        "mean_task_cost": 1.0,
        "mean_failure_rate": 0.04,
        "mean_vehicle_power_w": 8.0,
        "mean_rsu_power_w": 30.0,
        "model_complexity": 1.0,
    }
    values.update(changes)
    return ValidationCandidate(**values)  # type: ignore[arg-type]


def test_feasibility_precedes_lower_cost() -> None:
    report = select_feasible_validation_candidate(
        [
            _candidate("unsafe-cheap", mean_task_cost=0.1, hard_invariant_failures=1),
            _candidate("feasible", mean_task_cost=2.0),
        ],
        ValidationLimits(0.1, 0.1, 0.8),
    )
    assert report.selected_config_id == "feasible"
    assert report.rejected["unsafe-cheap"] == ("HARD_INVARIANT_FAILURE",)


def test_confidence_bounds_filter_then_stable_tie_breaks() -> None:
    report = select_feasible_validation_candidate(
        [
            _candidate("coverage-bad", coverage_rate_lcb=0.79),
            _candidate("timeout-bad", timeout_rate_ucb=0.11),
            _candidate("failure-bad", failure_rate_ucb=0.101),
            _candidate("power-high", mean_vehicle_power_w=9.0),
            _candidate("power-low", mean_vehicle_power_w=7.0),
        ],
        ValidationLimits(0.1, 0.1, 0.8),
    )
    assert report.selected_config_id == "power-low"
    assert report.ordered_feasible == ("power-low", "power-high")
    assert report.rejected == {
        "coverage-bad": ("COVERAGE_RATE_LCB",),
        "failure-bad": ("FAILURE_RATE_UCB",),
        "timeout-bad": ("TIMEOUT_RATE_UCB",),
    }


def test_no_feasible_candidate_is_explicit() -> None:
    report = select_feasible_validation_candidate(
        [_candidate("bad", failure_rate_ucb=0.5)],
        ValidationLimits(0.1, 0.1, 0.8),
    )
    assert report.selected_config_id is None
    assert report.feasible_config_ids == ()


def test_duplicate_candidate_ids_rejected() -> None:
    with pytest.raises(ValueError):
        select_feasible_validation_candidate(
            [_candidate("same"), _candidate("same")],
            ValidationLimits(0.1, 0.1, 0.8),
        )
