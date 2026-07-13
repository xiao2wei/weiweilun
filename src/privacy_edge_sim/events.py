"""Stable compound-event queue for continuous simulated time."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .enums import EventKind, EventPriority
from .errors import InvariantViolation


def _same_representable_instant(left_s: float, right_s: float) -> bool:
    """Treat only floating-point roundoff variants as one physical instant.

    Continuous-time expressions such as ``0.1 + 0.2`` and a trace boundary
    written as ``0.3`` may differ by one IEEE-754 ulp.  They are the same
    mathematical event time and must share the compound-event priority order.
    Eight ulps covers short arithmetic chains without merging physically
    distinguishable timestamps at the current floating-point resolution.
    """

    if left_s == right_s:
        return True
    if not math.isfinite(left_s) or not math.isfinite(right_s):
        return False
    tolerance = 8.0 * max(math.ulp(left_s), math.ulp(right_s))
    return abs(left_s - right_s) <= tolerance


def _strict_future_instant(now_s: float, positive_delta_s: float) -> float:
    """Add a positive duration without rounding a completion back to ``now_s``.

    A residual service interval can be smaller than half an IEEE-754 ulp at a
    large absolute timestamp.  Advancing by one representable instant is the
    smallest continuous-time step that can preserve the required strict-future
    completion semantics.
    """

    if not math.isfinite(now_s) or now_s < 0:
        raise ValueError("current time must be finite and nonnegative")
    if not math.isfinite(positive_delta_s) or positive_delta_s <= 0:
        raise ValueError("future duration must be finite and positive")
    finish = now_s + positive_delta_s
    if not math.isfinite(finish):
        raise ValueError("future instant must be finite")
    return finish if finish > now_s else math.nextafter(now_s, math.inf)


def priority_for(kind: EventKind) -> EventPriority:
    if kind in {EventKind.COMPUTE_COMPLETE, EventKind.TRANSFER_COMPLETE}:
        return EventPriority.COMPLETION
    if kind in {
        EventKind.DEVICE_FAULT,
        EventKind.LINK_CHANGE,
        EventKind.MODEL_VERSION,
        EventKind.PROFILE_VERSION,
        EventKind.THERMAL_CHANGE,
        EventKind.RSU_SNAPSHOT,
        EventKind.BATTERY_GUARD,
    }:
        return EventPriority.FAULT_LINK_VERSION_THERMAL
    if kind is EventKind.DEADLINE:
        return EventPriority.DEADLINE
    if kind is EventKind.ARRIVAL:
        return EventPriority.ARRIVAL
    return EventPriority.DISPATCH_DECISION


@dataclass(order=True, frozen=True, slots=True)
class Event:
    time_s: float
    priority: int
    seq: int
    kind: EventKind = field(compare=False)
    task_id: str | None = field(default=None, compare=False)
    object_id: str | None = field(default=None, compare=False)
    version_token: int | None = field(default=None, compare=False)
    payload: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("event time must be finite and nonnegative")


class EventQueue:
    """A min heap that always pops all atomic events at one timestamp."""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._seq = itertools.count()

    def __len__(self) -> int:
        return len(self._heap)

    def push(
        self,
        time_s: float,
        kind: EventKind,
        *,
        task_id: str | None = None,
        object_id: str | None = None,
        version_token: int | None = None,
        payload: Any = None,
        priority: EventPriority | int | None = None,
    ) -> Event:
        event = Event(
            float(time_s),
            int(priority if priority is not None else priority_for(kind)),
            next(self._seq),
            kind,
            task_id,
            object_id,
            version_token,
            payload,
        )
        heapq.heappush(self._heap, event)
        return event

    def peek_time(self) -> float:
        if not self._heap:
            raise IndexError("empty event queue")
        return self._heap[0].time_s

    def pop_compound(self, *, current_time_s: float) -> tuple[float, list[Event]]:
        if not self._heap:
            raise IndexError("empty event queue")
        first_time_s = self._heap[0].time_s
        if first_time_s < current_time_s and not _same_representable_instant(
            first_time_s, current_time_s
        ):
            raise InvariantViolation(
                "TIME_REGRESSION",
                "event queue attempted to move simulation clock backward",
                current_time_s=current_time_s,
                event_time_s=first_time_s,
            )
        batch: list[Event] = []
        while self._heap and _same_representable_instant(
            self._heap[0].time_s, first_time_s
        ):
            batch.append(heapq.heappop(self._heap))
        # Advance to the latest representational variant so a completion that
        # rounded one ulp above a trace deadline has received its full service
        # before completion-priority handlers execute.
        time_s = max(current_time_s, *(event.time_s for event in batch))
        batch.sort(key=lambda e: (e.priority, e.seq))
        return time_s, batch

    def extend(self, events: Iterable[Event]) -> None:
        for event in events:
            heapq.heappush(self._heap, event)

    def cancel_task(self, task_id: str) -> tuple[Event, ...]:
        """Remove every future event owned by ``task_id`` from the heap.

        A compound batch has already been removed from the heap before its
        handlers execute, so this method intentionally covers only future
        timestamps.  Same-timestamp events that remain in the active compound
        batch are harmless: terminal states are absorbing and their handlers
        perform the usual stale/terminal checks.
        """

        cancelled = tuple(event for event in self._heap if event.task_id == task_id)
        if not cancelled:
            return ()
        self._heap = [event for event in self._heap if event.task_id != task_id]
        heapq.heapify(self._heap)
        return tuple(sorted(cancelled))

    def pending_for_task(self, task_id: str) -> tuple[Event, ...]:
        """Return a stable read-only view of future events for one task."""

        return tuple(sorted(event for event in self._heap if event.task_id == task_id))

    def snapshot(self) -> tuple[tuple[float, int, int, str, str | None], ...]:
        return tuple(
            sorted(
                (e.time_s, e.priority, e.seq, e.kind.value, e.task_id)
                for e in self._heap
            )
        )
