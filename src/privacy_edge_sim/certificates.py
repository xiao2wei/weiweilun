"""Auditable finite-scenario error certificates for ESL-SMPC.

The certificate implemented here is the finite-action Hoeffding/union-bound
construction specified in Scheme 1, section 8.7.  It is deliberately kept
separate from policy code so validation and reporting can recompute it without
trusting a controller's cached score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class ScenarioErrorCertificate:
    """Simultaneous ratio-estimation error bound for a finite action set."""

    action_count: int
    scenario_count: int
    confidence_error: float
    numerator_abs_bound: float
    duration_lower_bound_s: float
    duration_upper_bound_s: float
    eta_n: float
    eta_d_s: float
    uniform_ratio_error: float | None
    empirical_argmin_gap_bound: float | None
    valid: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_scenario_ratio_certificate(
    *,
    action_count: int,
    scenario_count: int,
    confidence_error: float,
    numerator_abs_bound: float,
    duration_lower_bound_s: float,
    duration_upper_bound_s: float,
) -> ScenarioErrorCertificate:
    """Return the scheme's simultaneous finite-scenario ratio certificate.

    ``numerator_abs_bound`` is :math:`B_N`; durations must be certified inside
    ``[duration_lower_bound_s, duration_upper_bound_s]``.  A non-positive or
    non-finite bound is rejected instead of silently producing a performance
    claim.  When the denominator concentration error is not smaller than the
    duration lower bound, the hard-safety properties remain usable but no
    finite performance certificate is returned.
    """

    if isinstance(action_count, bool) or action_count < 1:
        raise ValueError("action_count must be a positive integer")
    if isinstance(scenario_count, bool) or scenario_count < 1:
        raise ValueError("scenario_count must be a positive integer")
    values = {
        "confidence_error": confidence_error,
        "numerator_abs_bound": numerator_abs_bound,
        "duration_lower_bound_s": duration_lower_bound_s,
        "duration_upper_bound_s": duration_upper_bound_s,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < confidence_error < 1.0:
        raise ValueError("confidence_error must be in (0, 1)")
    if numerator_abs_bound <= 0.0:
        raise ValueError("numerator_abs_bound must be positive")
    if duration_lower_bound_s <= 0.0:
        raise ValueError("duration_lower_bound_s must be positive")
    if duration_upper_bound_s < duration_lower_bound_s:
        raise ValueError(
            "duration_upper_bound_s must be at least duration_lower_bound_s"
        )

    log_term = math.log(4.0 * action_count / confidence_error)
    concentration = math.sqrt(log_term / (2.0 * scenario_count))
    eta_n = numerator_abs_bound * concentration
    eta_d = duration_upper_bound_s * concentration
    if eta_d >= duration_lower_bound_s:
        return ScenarioErrorCertificate(
            action_count=action_count,
            scenario_count=scenario_count,
            confidence_error=confidence_error,
            numerator_abs_bound=numerator_abs_bound,
            duration_lower_bound_s=duration_lower_bound_s,
            duration_upper_bound_s=duration_upper_bound_s,
            eta_n=eta_n,
            eta_d_s=eta_d,
            uniform_ratio_error=None,
            empirical_argmin_gap_bound=None,
            valid=False,
            reason="DENOMINATOR_CONCENTRATION_NOT_SEPARATED",
        )

    denominator_margin = duration_lower_bound_s - eta_d
    ratio_error = eta_n / denominator_margin + (
        numerator_abs_bound * eta_d / (duration_lower_bound_s * denominator_margin)
    )
    return ScenarioErrorCertificate(
        action_count=action_count,
        scenario_count=scenario_count,
        confidence_error=confidence_error,
        numerator_abs_bound=numerator_abs_bound,
        duration_lower_bound_s=duration_lower_bound_s,
        duration_upper_bound_s=duration_upper_bound_s,
        eta_n=eta_n,
        eta_d_s=eta_d,
        uniform_ratio_error=ratio_error,
        empirical_argmin_gap_bound=2.0 * ratio_error,
        valid=True,
        reason=None,
    )


__all__ = [
    "ScenarioErrorCertificate",
    "finite_scenario_ratio_certificate",
]
