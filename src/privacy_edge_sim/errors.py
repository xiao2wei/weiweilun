"""Structured failures used at validation and run time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: str
    message: str
    context: dict[str, Any]


class ResearchSimError(RuntimeError):
    """Base error carrying a stable machine-readable reason code."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.detail = ErrorDetail(code, message, dict(context))


class ConfigError(ResearchSimError):
    pass


class ProfileValidationError(ResearchSimError):
    pass


class AdapterValidationError(ResearchSimError):
    """A frozen executable adapter is incompatible with its profile."""


class TraceValidationError(ResearchSimError):
    pass


class EvidenceValidationError(ResearchSimError):
    """A frozen study-evidence document failed its trust-chain checks."""


class UnsupportedCondition(ResearchSimError):
    pass


class TransitionError(ResearchSimError):
    pass


class PacketConstructionError(ResearchSimError):
    pass


class InvariantViolation(ResearchSimError):
    """A fatal run invariant failure; simulations must stop immediately."""
