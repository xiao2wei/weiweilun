from __future__ import annotations

import math

import pytest

from privacy_edge_sim.certificates import finite_scenario_ratio_certificate


def test_finite_scenario_certificate_matches_scheme_formula() -> None:
    cert = finite_scenario_ratio_certificate(
        action_count=3,
        scenario_count=100_000,
        confidence_error=0.05,
        numerator_abs_bound=4.0,
        duration_lower_bound_s=0.2,
        duration_upper_bound_s=1.0,
    )
    concentration = math.sqrt(math.log(4.0 * 3 / 0.05) / (2.0 * 100_000))
    eta_n = 4.0 * concentration
    eta_d = concentration
    expected = eta_n / (0.2 - eta_d) + 4.0 * eta_d / (0.2 * (0.2 - eta_d))
    assert cert.valid
    assert cert.eta_n == pytest.approx(eta_n)
    assert cert.eta_d_s == pytest.approx(eta_d)
    assert cert.uniform_ratio_error == pytest.approx(expected)
    assert cert.empirical_argmin_gap_bound == pytest.approx(2.0 * expected)


def test_certificate_with_unseparated_denominator_withdraws_performance_bound() -> None:
    cert = finite_scenario_ratio_certificate(
        action_count=8,
        scenario_count=4,
        confidence_error=0.05,
        numerator_abs_bound=10.0,
        duration_lower_bound_s=0.01,
        duration_upper_bound_s=2.0,
    )
    assert not cert.valid
    assert cert.uniform_ratio_error is None
    assert cert.empirical_argmin_gap_bound is None
    assert cert.reason == "DENOMINATOR_CONCENTRATION_NOT_SEPARATED"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_count": 0},
        {"scenario_count": 0},
        {"confidence_error": 0.0},
        {"confidence_error": 1.0},
        {"numerator_abs_bound": 0.0},
        {"duration_lower_bound_s": 0.0},
        {"duration_lower_bound_s": 2.0, "duration_upper_bound_s": 1.0},
    ],
)
def test_certificate_rejects_invalid_physical_bounds(kwargs: dict[str, float]) -> None:
    values: dict[str, float | int] = {
        "action_count": 2,
        "scenario_count": 32,
        "confidence_error": 0.05,
        "numerator_abs_bound": 4.0,
        "duration_lower_bound_s": 0.1,
        "duration_upper_bound_s": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        finite_scenario_ratio_certificate(**values)  # type: ignore[arg-type]
