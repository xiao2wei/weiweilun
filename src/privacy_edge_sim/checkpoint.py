"""Deterministic replay checkpoints for interruption recovery.

The trusted vehicle-domain raw/aligned capabilities are intentionally not
serializable.  A checkpoint therefore records a cryptographically bound event
prefix, not a pickle of live simulator objects.  Resume deterministically
replays the frozen trace to that stable compound-event boundary, verifies the
prefix digest, and then continues.  This provides exact logical recovery while
preserving the no-raw-serialization invariant; it does not claim to avoid the
CPU cost of replaying the verified prefix.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .profiles import canonical_json_bytes


CHECKPOINT_SCHEMA_VERSION = "replay-checkpoint/1.0"


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint identity cannot contain non-finite values")
        return value
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def configuration_sha256(config: Any) -> str:
    """Return a stable hash of the fully resolved immutable configuration."""

    return hashlib.sha256(canonical_json_bytes(_plain(config))).hexdigest()


def checkpoint_identity(
    *,
    config: Any,
    profile_hash: str,
    evaluation_trace_hash: str,
    scenario_trace_hash: str,
    protocol_version: str,
    policy: str,
    code_version: str,
) -> dict[str, Any]:
    return {
        "configuration_sha256": configuration_sha256(config),
        "profile_hash": profile_hash,
        "evaluation_trace_hash": evaluation_trace_hash,
        "scenario_trace_hash": scenario_trace_hash,
        "protocol_version": protocol_version,
        "policy": policy,
        "code_version": code_version,
    }


def _self_hash(document: Mapping[str, Any]) -> str:
    material = dict(document)
    material.pop("checkpoint_sha256", None)
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def write_replay_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    compound_events: int,
    clock_s: float,
    prefix_sha256: str,
    complete: bool,
) -> Path:
    """Atomically replace one replay checkpoint after a stable macro-event."""

    if compound_events < 0 or not math.isfinite(clock_s) or clock_s < 0:
        raise ValueError("invalid replay checkpoint boundary")
    if len(prefix_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in prefix_sha256
    ):
        raise ValueError("prefix_sha256 must be a lowercase SHA-256 digest")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": _plain(identity),
        "compound_events": compound_events,
        "clock_s": clock_s,
        "prefix_sha256": prefix_sha256,
        "complete": bool(complete),
        "recovery_mode": "deterministic_replay",
        "contains_raw_or_aligned_payload": False,
        "checkpoint_sha256": "",
    }
    document["checkpoint_sha256"] = _self_hash(document)
    payload = (
        json.dumps(
            document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        )
        + "\n"
    )
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        if os.name != "nt":  # Windows cannot fsync directory handles portably.
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_replay_checkpoint(
    path: str | Path, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    target = Path(path).resolve()
    document = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("checkpoint root must be an object")
    if document.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported replay checkpoint schema")
    expected_hash = document.get("checkpoint_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _self_hash(document):
        raise ValueError("replay checkpoint self-hash mismatch")
    if document.get("identity") != _plain(expected_identity):
        raise ValueError("replay checkpoint run identity mismatch")
    if document.get("contains_raw_or_aligned_payload") is not False:
        raise ValueError("checkpoint violates raw/aligned non-serialization boundary")
    count = document.get("compound_events")
    clock_s = document.get("clock_s")
    digest = document.get("prefix_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("invalid checkpoint compound event count")
    if (
        isinstance(clock_s, bool)
        or not isinstance(clock_s, (int, float))
        or not math.isfinite(float(clock_s))
        or float(clock_s) < 0
    ):
        raise ValueError("invalid checkpoint clock")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("invalid checkpoint prefix digest")
    return document


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "checkpoint_identity",
    "configuration_sha256",
    "load_replay_checkpoint",
    "write_replay_checkpoint",
]
