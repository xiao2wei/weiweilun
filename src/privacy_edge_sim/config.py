"""Strict JSON configuration schema with explicit SI units and source tags."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ConfigError


ALLOWED_SOURCE_TYPES = {
    "measured",
    "public_specification",
    "literature_range",
    "engineering_assumption",
    "stress_test_boundary",
}
ALLOWED_POLICIES = {
    "all_local",
    "fixed_safe_lowest_link_cost",
    "fixed_safe_shortest_visible_queue",
    "safe_greedy",
    "safe_lyapunov_h1",
    "esl_smpc",
    "safe_one_shot",
}
ALLOWED_ROLLOUT_POLICIES = {
    "all_local",
    "fixed_safe_lowest_link_cost",
    "fixed_safe_shortest_visible_queue",
    "safe_greedy",
}


def _finite(
    name: str, value: Any, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ConfigError(
            "CONFIG_NONFINITE",
            f"{name} must be a finite number",
            field=name,
            value=value,
        )
    result = float(value)
    if positive and result <= 0:
        raise ConfigError(
            "CONFIG_NONPOSITIVE", f"{name} must be > 0", field=name, value=value
        )
    if nonnegative and result < 0:
        raise ConfigError(
            "CONFIG_NEGATIVE", f"{name} must be >= 0", field=name, value=value
        )
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(
            "CONFIG_INTEGER_RANGE",
            f"{name} must be an integer >= 1",
            field=name,
            value=value,
        )
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(
            "CONFIG_INTEGER_RANGE",
            f"{name} must be an integer >= 0",
            field=name,
            value=value,
        )
    return value


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            "CONFIG_STRING",
            f"{name} must be a non-empty string",
            field=name,
            value=value,
        )
    return value


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(
            "CONFIG_BOOLEAN", f"{name} must be a boolean", field=name, value=value
        )
    return value


def _strict_object(
    name: str,
    value: Any,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError("CONFIG_OBJECT", f"{name} must be an object", field=name)
    allowed = required | (optional or set())
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing:
        raise ConfigError(
            "CONFIG_FIELD_MISSING",
            f"{name} is missing required fields",
            fields=sorted(missing),
        )
    if unknown:
        raise ConfigError(
            "CONFIG_FIELD_UNKNOWN",
            f"{name} contains unknown fields",
            fields=sorted(unknown),
        )
    return value


@dataclass(frozen=True, slots=True)
class VehicleConfig:
    vehicle_id: str
    device_type: str
    battery_capacity_j: float
    initial_battery_j: float
    memory_capacity_bytes: int
    accelerator_descriptors: int
    cpu_descriptors: int
    encoder_descriptors: int
    idle_power_w: float
    hold_power_w: float
    average_power_budget_w: float


@dataclass(frozen=True, slots=True)
class RSUConfig:
    rsu_id: str
    descriptor_capacity: int
    vram_capacity_bytes: int
    workload_capacity_gpu_s: float
    gpu_servers: int
    idle_power_w: float
    hold_power_w: float
    cached_models: tuple[str, ...]
    average_power_budget_w: float


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    policy: str
    horizon_events: int
    scenarios: int
    lyapunov_v: float
    controller_overhead_s: float
    controller_energy_j: float
    rollout_policy: str
    physical_queue_weight: float
    vehicle_resource_theta: Mapping[str, float]
    rsu_resource_theta: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    risk_threshold: float
    confidence_error: float
    quality_miscoverage: float
    min_subjects: int
    min_emission_lcb: float


@dataclass(frozen=True, slots=True)
class CostConfig:
    latency_scale_s: float
    vehicle_energy_scale_j: float
    rsu_energy_scale_j: float
    utility_scale: float
    failure_loss: float
    weights: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class LongTermConfig:
    timeout_rate_limit: float
    failure_rate_limit: float
    coverage_rate_minimum: float


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    schema_version: str
    protocol_version: str
    profile_path: Path
    trace_path: Path
    scenario_trace_path: Path
    evidence_path: Path | None
    max_snapshot_age_s: float
    rsu_snapshot_period_s: float
    rsu_telemetry_delay_s: float
    rsu_telemetry_quantum_work_s: float
    rsu_telemetry_drop_every: int
    uplink_pause_limit_s: float
    downlink_pause_limit_s: float
    metadata_bits: int
    vehicles: tuple[VehicleConfig, ...]
    rsus: tuple[RSUConfig, ...]
    controller: ControllerConfig
    privacy: PrivacyConfig
    cost: CostConfig
    long_term: LongTermConfig
    seeds: Mapping[str, int]
    parameter_sources: Mapping[str, str]
    output_parquet: bool

    def validate(self) -> None:
        if not self.schema_version or not self.protocol_version:
            raise ConfigError(
                "CONFIG_VERSION_MISSING", "schema and protocol versions are required"
            )
        if (
            len({v.vehicle_id for v in self.vehicles}) != len(self.vehicles)
            or not self.vehicles
        ):
            raise ConfigError(
                "CONFIG_VEHICLE_IDS", "vehicle IDs must be unique and non-empty"
            )
        if len({r.rsu_id for r in self.rsus}) != len(self.rsus) or not self.rsus:
            raise ConfigError("CONFIG_RSU_IDS", "RSU IDs must be unique and non-empty")
        if self.metadata_bits < 0:
            raise ConfigError("CONFIG_METADATA_BITS", "metadata_bits must be >= 0")
        for v in self.vehicles:
            if v.initial_battery_j > v.battery_capacity_j:
                raise ConfigError(
                    "CONFIG_BATTERY_CONFLICT",
                    "initial battery exceeds capacity",
                    vehicle=v.vehicle_id,
                )
        required_seeds = {
            "environment",
            "arrivals",
            "mobility",
            "wireless",
            "vehicle",
            "rsu",
            "fault",
            "scenario",
        }
        if set(self.seeds) != required_seeds:
            raise ConfigError(
                "CONFIG_SEEDS",
                "seeds must contain exactly the independent registered streams",
                expected=sorted(required_seeds),
                actual=sorted(self.seeds),
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.seeds.values()
        ):
            raise ConfigError(
                "CONFIG_SEED_RANGE", "all registered seeds must be nonnegative integers"
            )
        if self.controller.policy not in ALLOWED_POLICIES:
            raise ConfigError(
                "CONFIG_POLICY",
                "controller.policy is not registered",
                policy=self.controller.policy,
            )
        if self.controller.rollout_policy not in ALLOWED_ROLLOUT_POLICIES:
            raise ConfigError(
                "CONFIG_ROLLOUT_POLICY",
                "controller.rollout_policy is not a supported recourse selector",
                policy=self.controller.rollout_policy,
            )
        bad_sources = {
            k: v
            for k, v in self.parameter_sources.items()
            if v not in ALLOWED_SOURCE_TYPES
        }
        if bad_sources:
            raise ConfigError(
                "CONFIG_SOURCE_TYPE",
                "unknown parameter source category",
                invalid=bad_sources,
            )


def _mapping(d: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(d))


def load_config(path: str | Path) -> SimulationConfig:
    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(
            "CONFIG_READ",
            "cannot read strict UTF-8 JSON config",
            path=str(config_path),
            error=str(exc),
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigError("CONFIG_ROOT", "config root must be an object")
    root_required = {
        "schema_version",
        "protocol_version",
        "profile_path",
        "trace_path",
        "scenario_trace_path",
        "max_snapshot_age_s",
        "rsu_snapshot_period_s",
        "uplink_pause_limit_s",
        "downlink_pause_limit_s",
        "metadata_bits",
        "vehicles",
        "rsus",
        "controller",
        "privacy",
        "cost",
        "long_term",
        "seeds",
        "parameter_sources",
    }
    _strict_object(
        "config",
        raw,
        root_required,
        {
            "evidence_path",
            "output_parquet",
            "rsu_telemetry_delay_s",
            "rsu_telemetry_quantum_work_s",
            "rsu_telemetry_drop_every",
        },
    )
    if not isinstance(raw["vehicles"], list) or not raw["vehicles"]:
        raise ConfigError("CONFIG_VEHICLES", "vehicles must be a non-empty array")
    if not isinstance(raw["rsus"], list) or not raw["rsus"]:
        raise ConfigError("CONFIG_RSUS", "rsus must be a non-empty array")
    vehicle_required = {
        "vehicle_id",
        "device_type",
        "battery_capacity_j",
        "initial_battery_j",
        "memory_capacity_bytes",
        "accelerator_descriptors",
        "cpu_descriptors",
        "encoder_descriptors",
        "idle_power_w",
        "hold_power_w",
        "average_power_budget_w",
    }
    rsu_required = {
        "rsu_id",
        "descriptor_capacity",
        "vram_capacity_bytes",
        "workload_capacity_gpu_s",
        "gpu_servers",
        "idle_power_w",
        "hold_power_w",
        "cached_models",
        "average_power_budget_w",
    }
    vehicle_rows = [
        _strict_object(f"vehicles[{i}]", row, vehicle_required)
        for i, row in enumerate(raw["vehicles"])
    ]
    rsu_rows = [
        _strict_object(f"rsus[{i}]", row, rsu_required)
        for i, row in enumerate(raw["rsus"])
    ]
    for index, row in enumerate(rsu_rows):
        if not isinstance(row["cached_models"], list):
            raise ConfigError(
                "CONFIG_MODEL_CACHE",
                "cached_models must be an array",
                field=f"rsus[{index}].cached_models",
            )
    ctl = _strict_object(
        "controller",
        raw["controller"],
        {
            "policy",
            "horizon_events",
            "scenarios",
            "lyapunov_v",
            "controller_overhead_s",
            "controller_energy_j",
            "rollout_policy",
        },
        {
            "physical_queue_weight",
            "vehicle_resource_theta",
            "rsu_resource_theta",
        },
    )
    p = _strict_object(
        "privacy",
        raw["privacy"],
        {
            "risk_threshold",
            "confidence_error",
            "quality_miscoverage",
            "min_subjects",
            "min_emission_lcb",
        },
    )
    c = _strict_object(
        "cost",
        raw["cost"],
        {
            "latency_scale_s",
            "vehicle_energy_scale_j",
            "rsu_energy_scale_j",
            "utility_scale",
            "failure_loss",
            "weights",
        },
    )
    long_raw = _strict_object(
        "long_term",
        raw["long_term"],
        {"timeout_rate_limit", "failure_rate_limit", "coverage_rate_minimum"},
    )
    if not isinstance(c["weights"], dict) or not c["weights"]:
        raise ConfigError("CONFIG_WEIGHTS", "cost.weights must be a non-empty object")
    if not isinstance(raw["seeds"], dict) or not isinstance(
        raw["parameter_sources"], dict
    ):
        raise ConfigError(
            "CONFIG_MAPPING", "seeds and parameter_sources must be objects"
        )
    base = config_path.parent
    try:
        vehicles = tuple(
            VehicleConfig(
                vehicle_id=_string("vehicle_id", v["vehicle_id"]),
                device_type=_string("device_type", v["device_type"]),
                battery_capacity_j=_finite(
                    "battery_capacity_j", v["battery_capacity_j"], positive=True
                ),
                initial_battery_j=_finite(
                    "initial_battery_j", v["initial_battery_j"], nonnegative=True
                ),
                memory_capacity_bytes=_positive_int(
                    "memory_capacity_bytes", v["memory_capacity_bytes"]
                ),
                accelerator_descriptors=_positive_int(
                    "accelerator_descriptors", v["accelerator_descriptors"]
                ),
                cpu_descriptors=_positive_int("cpu_descriptors", v["cpu_descriptors"]),
                encoder_descriptors=_positive_int(
                    "encoder_descriptors", v["encoder_descriptors"]
                ),
                idle_power_w=_finite(
                    "vehicle.idle_power_w", v["idle_power_w"], nonnegative=True
                ),
                hold_power_w=_finite(
                    "vehicle.hold_power_w", v["hold_power_w"], nonnegative=True
                ),
                average_power_budget_w=_finite(
                    "vehicle.average_power_budget_w",
                    v["average_power_budget_w"],
                    positive=True,
                ),
            )
            for v in vehicle_rows
        )
        rsus = tuple(
            RSUConfig(
                rsu_id=_string("rsu_id", r["rsu_id"]),
                descriptor_capacity=_positive_int(
                    "descriptor_capacity", r["descriptor_capacity"]
                ),
                vram_capacity_bytes=_positive_int(
                    "vram_capacity_bytes", r["vram_capacity_bytes"]
                ),
                workload_capacity_gpu_s=_finite(
                    "workload_capacity_gpu_s",
                    r["workload_capacity_gpu_s"],
                    positive=True,
                ),
                gpu_servers=_positive_int("gpu_servers", r["gpu_servers"]),
                idle_power_w=_finite(
                    "rsu.idle_power_w", r["idle_power_w"], nonnegative=True
                ),
                hold_power_w=_finite(
                    "rsu.hold_power_w", r["hold_power_w"], nonnegative=True
                ),
                cached_models=tuple(
                    _string("cached_models[]", x) for x in r["cached_models"]
                ),
                average_power_budget_w=_finite(
                    "rsu.average_power_budget_w",
                    r["average_power_budget_w"],
                    positive=True,
                ),
            )
            for r in rsu_rows
        )
        theta_defaults = {
            "physical_queue_weight": 1.0,
            "vehicle_resource_theta": {},
            "rsu_resource_theta": {},
        }
        theta_rows: dict[str, Mapping[str, float]] = {}
        for family, allowed_resources in (
            ("vehicle_resource_theta", {"accelerator", "cpu", "encoder"}),
            ("rsu_resource_theta", {"ingress", "gpu"}),
        ):
            raw_theta = ctl.get(family, theta_defaults[family])
            if not isinstance(raw_theta, dict):
                raise ConfigError(
                    "CONFIG_MAPPING", f"controller.{family} must be an object"
                )
            unknown = set(raw_theta) - allowed_resources
            if unknown:
                raise ConfigError(
                    "CONFIG_FIELD_UNKNOWN",
                    f"controller.{family} contains unknown resources",
                    fields=sorted(unknown),
                )
            theta_rows[family] = _mapping(
                {
                    _string(f"controller.{family} key", key): _finite(
                        f"controller.{family}.{key}", value, nonnegative=True
                    )
                    for key, value in raw_theta.items()
                }
            )
        controller = ControllerConfig(
            policy=_string("controller.policy", ctl["policy"]),
            horizon_events=_positive_int("horizon_events", ctl["horizon_events"]),
            scenarios=_positive_int("scenarios", ctl["scenarios"]),
            lyapunov_v=_finite("lyapunov_v", ctl["lyapunov_v"], positive=True),
            controller_overhead_s=_finite(
                "controller_overhead_s", ctl["controller_overhead_s"], nonnegative=True
            ),
            controller_energy_j=_finite(
                "controller_energy_j", ctl["controller_energy_j"], nonnegative=True
            ),
            rollout_policy=_string("controller.rollout_policy", ctl["rollout_policy"]),
            physical_queue_weight=_finite(
                "controller.physical_queue_weight",
                ctl.get(
                    "physical_queue_weight",
                    theta_defaults["physical_queue_weight"],
                ),
                nonnegative=True,
            ),
            vehicle_resource_theta=theta_rows["vehicle_resource_theta"],
            rsu_resource_theta=theta_rows["rsu_resource_theta"],
        )
        privacy = PrivacyConfig(
            risk_threshold=_finite(
                "risk_threshold", p["risk_threshold"], nonnegative=True
            ),
            confidence_error=_finite(
                "confidence_error", p["confidence_error"], positive=True
            ),
            quality_miscoverage=_finite(
                "quality_miscoverage", p["quality_miscoverage"], positive=True
            ),
            min_subjects=_positive_int("min_subjects", p["min_subjects"]),
            min_emission_lcb=_finite(
                "min_emission_lcb", p["min_emission_lcb"], nonnegative=True
            ),
        )
        for name, value in {
            "risk_threshold": privacy.risk_threshold,
            "confidence_error": privacy.confidence_error,
            "quality_miscoverage": privacy.quality_miscoverage,
            "min_emission_lcb": privacy.min_emission_lcb,
        }.items():
            if value > 1:
                raise ConfigError(
                    "CONFIG_PROBABILITY",
                    f"{name} must lie in [0,1]",
                    field=name,
                    value=value,
                )
        weights = {
            _string("cost.weights key", k): _finite(
                f"cost.weights.{k}", v, nonnegative=True
            )
            for k, v in c["weights"].items()
        }
        cost = CostConfig(
            latency_scale_s=_finite(
                "latency_scale_s", c["latency_scale_s"], positive=True
            ),
            vehicle_energy_scale_j=_finite(
                "vehicle_energy_scale_j", c["vehicle_energy_scale_j"], positive=True
            ),
            rsu_energy_scale_j=_finite(
                "rsu_energy_scale_j", c["rsu_energy_scale_j"], positive=True
            ),
            utility_scale=_finite("utility_scale", c["utility_scale"], positive=True),
            failure_loss=_finite("failure_loss", c["failure_loss"], nonnegative=True),
            weights=_mapping(weights),
        )
        long_term = LongTermConfig(
            timeout_rate_limit=_finite(
                "timeout_rate_limit", long_raw["timeout_rate_limit"], nonnegative=True
            ),
            failure_rate_limit=_finite(
                "failure_rate_limit", long_raw["failure_rate_limit"], nonnegative=True
            ),
            coverage_rate_minimum=_finite(
                "coverage_rate_minimum",
                long_raw["coverage_rate_minimum"],
                nonnegative=True,
            ),
        )
        for name, value in {
            "timeout_rate_limit": long_term.timeout_rate_limit,
            "failure_rate_limit": long_term.failure_rate_limit,
            "coverage_rate_minimum": long_term.coverage_rate_minimum,
        }.items():
            if value > 1:
                raise ConfigError(
                    "CONFIG_PROBABILITY",
                    f"{name} must lie in [0,1]",
                    field=name,
                    value=value,
                )
        seeds = {
            _string("seed key", key): _nonnegative_int(f"seeds.{key}", value)
            for key, value in raw["seeds"].items()
        }
        sources = {
            _string("parameter source key", key): _string(
                f"parameter_sources.{key}", value
            )
            for key, value in raw["parameter_sources"].items()
        }
        cfg = SimulationConfig(
            schema_version=_string("schema_version", raw["schema_version"]),
            protocol_version=_string("protocol_version", raw["protocol_version"]),
            profile_path=(
                base / _string("profile_path", raw["profile_path"])
            ).resolve(),
            trace_path=(base / _string("trace_path", raw["trace_path"])).resolve(),
            scenario_trace_path=(
                base / _string("scenario_trace_path", raw["scenario_trace_path"])
            ).resolve(),
            evidence_path=(
                None
                if raw.get("evidence_path") is None
                else (base / _string("evidence_path", raw["evidence_path"])).resolve()
            ),
            max_snapshot_age_s=_finite(
                "max_snapshot_age_s", raw["max_snapshot_age_s"], nonnegative=True
            ),
            rsu_snapshot_period_s=_finite(
                "rsu_snapshot_period_s",
                raw["rsu_snapshot_period_s"],
                positive=True,
            ),
            rsu_telemetry_delay_s=_finite(
                "rsu_telemetry_delay_s",
                raw.get("rsu_telemetry_delay_s", 0.0),
                nonnegative=True,
            ),
            rsu_telemetry_quantum_work_s=_finite(
                "rsu_telemetry_quantum_work_s",
                raw.get("rsu_telemetry_quantum_work_s", 0.0),
                nonnegative=True,
            ),
            rsu_telemetry_drop_every=_nonnegative_int(
                "rsu_telemetry_drop_every",
                raw.get("rsu_telemetry_drop_every", 0),
            ),
            uplink_pause_limit_s=_finite(
                "uplink_pause_limit_s", raw["uplink_pause_limit_s"], nonnegative=True
            ),
            downlink_pause_limit_s=_finite(
                "downlink_pause_limit_s",
                raw["downlink_pause_limit_s"],
                nonnegative=True,
            ),
            metadata_bits=_nonnegative_int("metadata_bits", raw["metadata_bits"]),
            vehicles=vehicles,
            rsus=rsus,
            controller=controller,
            privacy=privacy,
            cost=cost,
            long_term=long_term,
            seeds=_mapping(seeds),
            parameter_sources=_mapping(sources),
            output_parquet=_boolean("output_parquet", raw.get("output_parquet", False)),
        )
    except KeyError as exc:
        raise ConfigError(
            "CONFIG_FIELD_MISSING", "required config field is missing", field=str(exc)
        ) from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(
            "CONFIG_FIELD_TYPE", "invalid config field type", error=str(exc)
        ) from exc
    cfg.validate()
    return cfg
