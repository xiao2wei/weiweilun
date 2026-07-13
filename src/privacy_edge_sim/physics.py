"""Deterministic continuous-time service integrals used by tests and runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .errors import TraceValidationError


@dataclass(frozen=True, slots=True)
class ServiceSegment:
    start_s: float
    end_s: float
    service_rate_per_s: float
    side_a_power_w: float = 0.0
    side_b_power_w: float = 0.0
    available: bool = True
    permanent_loss_at_end: bool = False

    def __post_init__(self) -> None:
        values = (
            self.start_s,
            self.end_s,
            self.service_rate_per_s,
            self.side_a_power_w,
            self.side_b_power_w,
        )
        if any(not math.isfinite(v) for v in values):
            raise TraceValidationError(
                "TRACE_NONFINITE", "service segment contains non-finite quantity"
            )
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise TraceValidationError(
                "TRACE_SEGMENT_TIME", "service segment must have positive duration"
            )
        if (
            self.service_rate_per_s < 0
            or self.side_a_power_w < 0
            or self.side_b_power_w < 0
        ):
            raise TraceValidationError(
                "TRACE_NEGATIVE_PHYSICS", "service and power must be nonnegative"
            )


@dataclass(frozen=True, slots=True)
class ServiceIntegral:
    delivered: float
    side_a_energy_j: float
    side_b_energy_j: float
    end_s: float
    completed: bool


def validate_segments(
    segments: Iterable[ServiceSegment], *, require_contiguous: bool = True
) -> tuple[ServiceSegment, ...]:
    rows = tuple(sorted(segments, key=lambda row: (row.start_s, row.end_s)))
    if not rows:
        raise TraceValidationError(
            "TRACE_SEGMENTS_EMPTY", "at least one service segment is required"
        )
    for previous, current in zip(rows, rows[1:]):
        if current.start_s < previous.end_s - 1e-12:
            raise TraceValidationError(
                "TRACE_SEGMENT_OVERLAP", "service segments overlap"
            )
        if require_contiguous and not math.isclose(
            current.start_s, previous.end_s, rel_tol=0.0, abs_tol=1e-12
        ):
            raise TraceValidationError(
                "TRACE_SEGMENT_GAP",
                "service segments must explicitly cover zero-service gaps",
            )
    return rows


def integrate_until_complete(
    amount: float,
    *,
    start_s: float,
    segments: Iterable[ServiceSegment],
    stop_s: float | None = None,
) -> ServiceIntegral:
    """Integrate exact service and paired energy, stopping inside a segment."""

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("amount must be finite and positive")
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("start_s must be finite and nonnegative")
    if stop_s is not None and (not math.isfinite(stop_s) or stop_s < start_s):
        raise ValueError("stop_s must be finite and no earlier than start_s")
    rows = validate_segments(segments, require_contiguous=False)
    remaining = amount
    delivered = 0.0
    energy_a = 0.0
    energy_b = 0.0
    current_time = start_s
    for row in rows:
        if row.end_s <= current_time:
            continue
        if row.start_s > current_time:
            gap_end = row.start_s if stop_s is None else min(row.start_s, stop_s)
            current_time = gap_end
            if stop_s is not None and current_time >= stop_s:
                break
        interval_end = row.end_s if stop_s is None else min(row.end_s, stop_s)
        if interval_end <= current_time:
            continue
        duration = interval_end - current_time
        rate = row.service_rate_per_s if row.available else 0.0
        if rate > 0 and rate * duration >= remaining - 1e-12:
            used = max(0.0, remaining / rate)
            energy_a += row.side_a_power_w * used
            energy_b += row.side_b_power_w * used
            current_time += used
            delivered += remaining
            remaining = 0.0
            return ServiceIntegral(delivered, energy_a, energy_b, current_time, True)
        segment_delivery = rate * duration
        delivered += segment_delivery
        remaining = max(0.0, remaining - segment_delivery)
        energy_a += row.side_a_power_w * duration
        energy_b += row.side_b_power_w * duration
        current_time = interval_end
        if stop_s is not None and current_time >= stop_s:
            break
        if row.permanent_loss_at_end and remaining > 0:
            break
    return ServiceIntegral(
        delivered, energy_a, energy_b, current_time, remaining <= 1e-12
    )


def completion_time(
    amount: float, *, start_s: float, segments: Iterable[ServiceSegment]
) -> float | None:
    result = integrate_until_complete(amount, start_s=start_s, segments=segments)
    return result.end_s if result.completed else None
