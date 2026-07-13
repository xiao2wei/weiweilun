"""Privacy-safe anonymous-image edge inference research simulator."""

from .config import SimulationConfig, load_config
from .errors import (
    AdapterValidationError,
    ConfigError,
    InvariantViolation,
    UnsupportedCondition,
)
from .packets import AnonFERRequest, RawImageHandle

__all__ = [
    "AnonFERRequest",
    "AdapterValidationError",
    "ConfigError",
    "InvariantViolation",
    "RawImageHandle",
    "SimulationConfig",
    "UnsupportedCondition",
    "load_config",
]

__version__ = "0.1.0"
