"""Frozen, causal action bounds used by every hard mask and controller.

The provider exposes only aggregate support-cell statistics.  It never returns
the task's selected test row, future wireless samples, labels, identities or
realized FER loss.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean
import math
from typing import Any, Iterable, Mapping

from .config import SimulationConfig
from .enums import ActionKind
from .profiles import FrozenProfileBundle
from .safety import Action, Observation
from .traces import DeviceContext, ScenarioLibrary


@dataclass(frozen=True, slots=True)
class RuntimeAdmissionBounds:
    """Profile-pinned RSU reservation certificate for one READY action.

    The certificate proves support using only the identity-free
    training/validation :class:`ScenarioLibrary`.  Its resource quantities are
    deployment bounds frozen in the profile; no evaluation row or hidden true
    quality cell contributes a numeric reservation value.
    """

    descriptor_count: int
    max_vram_bytes: int
    max_ingress_work_s: float
    max_ingress_energy_j: float
    max_gpu_work_s: float
    max_gpu_energy_j: float
    max_result_size_bits: int
    rsu_id: str
    model_id: str
    model_hash: str
    pipeline_id: str
    protocol_version: str
    profile_hash: str
    candidate_quality_bins: tuple[str, ...]
    scenario_trace_version: str
    scenario_trace_hash: str
    scenario_split_role: str


def encode_context(context: DeviceContext) -> str:
    return f"{context.thermal_state}|{context.power_mode}|{context.memory_pressure}"


def decode_context(value: str | DeviceContext | Mapping[str, Any]) -> DeviceContext:
    if isinstance(value, DeviceContext):
        return value
    if isinstance(value, Mapping):
        return DeviceContext.from_value(value)
    parts = str(value).split("|")
    if len(parts) != 3:
        return DeviceContext(str(value or "nominal"), "nominal", "normal")
    return DeviceContext(parts[0], parts[1], parts[2])


class FrozenEstimateProvider:
    """Conservative aggregate bounds over exact frozen trace support cells."""

    def __init__(
        self,
        profile: FrozenProfileBundle,
        scenario: ScenarioLibrary,
        config: SimulationConfig,
        *,
        requires_evaluation_pair: bool = False,
    ) -> None:
        if not isinstance(scenario, ScenarioLibrary):
            raise TypeError(
                "scenario estimates require an identity-free ScenarioLibrary"
            )
        self.profile = profile
        # Store only the identity-free scenario rows.  In particular, do not
        # retain the source TraceBundle: the provider is reachable through a
        # policy's HardMaskEngine and therefore must not offer a path to
        # subject clusters, fixture keys, arrivals, or future environment
        # events.  Evaluation-pair authorization is supplied per call as one
        # boolean capability; no evaluation artifact, support index, FER
        # outcome or trace object is retained in this policy-reachable object.
        self.scenario_profile_hash = scenario.profile_hash
        self.scenario_protocol_version = scenario.protocol_version
        self.scenario_trace_version = scenario.trace_version
        self.scenario_trace_hash = scenario.trace_hash
        self.scenario_split_role = scenario.split_role
        self.anon_rows = tuple(scenario.anon_rows)
        self.local_rows = tuple(scenario.local_rows)
        self.edge_rows = tuple(scenario.edge_rows)
        self.requires_evaluation_pair = bool(requires_evaluation_pair)
        self.failure_loss = float(config.cost.failure_loss)
        self.metadata_bits = int(config.metadata_bits)
        self.cost_weights = dict(config.cost.weights)
        self.latency_scale_s = float(config.cost.latency_scale_s)
        self.vehicle_energy_scale_j = float(config.cost.vehicle_energy_scale_j)
        self.rsu_energy_scale_j = float(config.cost.rsu_energy_scale_j)
        self.controller_overhead_s = float(config.controller.controller_overhead_s)
        # Retain only the identity-free distribution of the next sanitized RAW
        # decision time.  The provider deliberately does not retain scenario
        # windows or a cursor that could expose evaluation-future truth.
        self._next_external_raw_offsets_s = tuple(
            min(
                (
                    float(task.arrival_offset_s) + float(task.prep_work_s)
                    for task in environment.future_tasks
                    if task.complete_support
                ),
                default=math.inf,
            )
            for environment in scenario.environment_scenarios
        )

    @staticmethod
    def _observable_unidentified_ready_offset_s(
        observation: Observation,
    ) -> float:
        """Frozen causal proxy for an unidentified in-flight vehicle job.

        Policy observations intentionally omit job type.  A non-empty visible
        pool may therefore contain another anonymization transaction whose
        completion creates READY.  Residual work divided by the visible job
        count is the preregistered conditional mean proxy; treating every such
        completion as a macro decision is conservative about interval length.
        """

        resources = observation.vehicle.get("resources", {})
        if not isinstance(resources, Mapping):
            return math.inf
        candidates: list[float] = []
        for row in resources.values():
            if not isinstance(row, Mapping):
                continue
            residual = float(row.get("residual_work_s", 0.0) or 0.0)
            count = int(row.get("running_count", 0) or 0) + int(
                row.get("waiting_count", 0) or 0
            )
            servers = max(1, int(row.get("server_count", 1) or 1))
            if residual > 0 and count > 0:
                candidates.append(residual / min(count, servers))
        return min(candidates, default=math.inf)

    def _macro_interval_statistics(
        self, action_duration_s: float, observation: Observation
    ) -> tuple[float, float, float]:
        duration = max(0.0, float(action_duration_s))
        unidentified_ready = self._observable_unidentified_ready_offset_s(observation)
        external = self._next_external_raw_offsets_s or (math.inf,)
        next_macros = [min(offset, unidentified_ready) for offset in external]
        intervals = [min(duration, offset) for offset in next_macros]
        completed = [float(duration <= offset + 1e-12) for offset in next_macros]
        arrivals = [float(offset < duration - 1e-12) for offset in external]
        return fmean(intervals), fmean(completed), fmean(arrivals)

    def _with_macro_interval(
        self,
        values: Mapping[str, Any],
        observation: Observation,
        *,
        vehicle_resources: tuple[str, ...] = (),
        rsu_id: str | None = None,
        counts_task_terminal: bool = True,
    ) -> dict[str, Any]:
        row = dict(values)
        duration = float(row.get("expected_duration_s", 0.0) or 0.0)
        interval, completion_probability, expected_arrivals = (
            self._macro_interval_statistics(duration, observation)
        )
        fraction = 1.0 if duration <= 1e-12 else min(1.0, interval / duration)
        row["expected_next_macro_interval_s"] = interval
        terminal_probability = completion_probability if counts_task_terminal else 0.0
        failure = max(0.0, float(row.get("expected_failure", 0.0) or 0.0))
        row["action_completed_before_macro_probability"] = completion_probability
        row["interval_completion_probability"] = terminal_probability * (1.0 - failure)
        row["interval_expected_arrivals"] = expected_arrivals
        row["interval_expected_vehicle_energy_j"] = max(
            0.0, float(row.get("expected_vehicle_energy_j", 0.0) or 0.0) * fraction
        )
        row["interval_expected_rsu_energy_j"] = max(
            0.0, float(row.get("expected_rsu_energy_j", 0.0) or 0.0) * fraction
        )
        row["interval_expected_loss"] = max(
            0.0,
            float(row.get("expected_loss", 0.0) or 0.0) * terminal_probability,
        )
        row["interval_expected_failure"] = max(
            0.0,
            failure * terminal_probability,
        )
        if vehicle_resources:
            resources = observation.vehicle.get("resources", {})
            services: dict[str, float] = {}
            for vehicle_resource in vehicle_resources:
                resource_row = (
                    resources.get(vehicle_resource, {})
                    if isinstance(resources, Mapping)
                    else {}
                )
                servers = (
                    max(1, int(resource_row.get("server_count", 1) or 1))
                    if isinstance(resource_row, Mapping)
                    else 1
                )
                services[vehicle_resource] = interval * servers
            row["vehicle_service_s"] = services
        if rsu_id is not None:
            rsu = observation.rsus.get(rsu_id, {})
            gpu_servers = (
                max(1, int(rsu.get("gpu_servers", 1) or 1))
                if isinstance(rsu, Mapping)
                else 1
            )
            row["rsu_ingress_service_s"] = interval
            row["rsu_gpu_service_s"] = interval * gpu_servers
        return row

    def _profile_ok(self, profile_hash: str | None) -> bool:
        return (
            profile_hash in {None, self.profile.profile_hash}
            and self.scenario_profile_hash == self.profile.profile_hash
        )

    @staticmethod
    def _ctx(value: Any) -> DeviceContext:
        return decode_context(value)

    def has_anon_support(self, **kwargs: Any) -> bool:
        if not self._profile_ok(kwargs.get("profile_hash")):
            return False
        pipeline_id = str(kwargs.get("pipeline_id"))
        bins = tuple(kwargs.get("quality_bins") or ())
        device_type = str(kwargs.get("device_type"))
        context = self._ctx(kwargs.get("device_context"))
        covered = {
            row.quality_bin
            for row in self.anon_rows
            if row.pipeline_id == pipeline_id
            and row.quality_bin in bins
            and row.device_type == device_type
            and row.context == context
        }
        return bool(bins) and covered == set(bins)

    supports_anon = has_anon_support

    def has_local_support(self, **kwargs: Any) -> bool:
        if not self._profile_ok(kwargs.get("profile_hash")):
            return False
        model_id = str(kwargs.get("model_id"))
        bins = tuple(kwargs.get("quality_bins") or ())
        device = str(kwargs.get("device_type"))
        ctx = self._ctx(kwargs.get("device_context"))
        for quality_bin in bins:
            if not any(
                row.model_id == model_id
                and row.quality_bin == quality_bin
                and row.device_type == device
                and row.context == ctx
                for row in self.local_rows
            ):
                return False
        return bool(bins)

    supports_local = has_local_support

    def has_edge_support(self, **kwargs: Any) -> bool:
        if not self._profile_ok(kwargs.get("profile_hash")):
            return False
        artifact = kwargs.get("artifact_token")
        pipeline = kwargs.get("pipeline_id")
        if not artifact or not pipeline:
            return False
        bins = tuple(kwargs.get("quality_bins") or ())
        rsu_context_raw = kwargs.get("rsu_context")
        rsu_context = None if rsu_context_raw is None else self._ctx(rsu_context_raw)
        scenario_rows = [
            row
            for row in self.edge_rows
            if row.rsu_id == str(kwargs.get("rsu_id"))
            and row.model_id == str(kwargs.get("model_id"))
            and row.pipeline_id == str(pipeline)
            and row.quality_bin in bins
            and (rsu_context is None or row.context == rsu_context)
        ]
        vehicle_context = self._ctx(kwargs.get("device_context"))
        device_type = str(kwargs.get("device_type"))
        formed_by_quality = {
            quality_bin: {
                row.artifact_token
                for row in self.anon_rows
                if row.pipeline_id == str(pipeline)
                and row.quality_bin == quality_bin
                and row.device_type == device_type
                and row.context == vehicle_context
                and row.formed_packet
                and row.artifact_token
            }
            for quality_bin in bins
        }
        scenario_supported = bool(bins) and all(
            artifacts
            and any(
                row.quality_bin == quality_bin and row.artifact_token in artifacts
                for row in scenario_rows
            )
            for quality_bin, artifacts in formed_by_quality.items()
        )
        if not scenario_supported:
            return False
        # Scenario-local artifact tokens are created only by the sanitized
        # training/validation ScenarioLibrary.  They cannot be joined to an
        # evaluation artifact and are valid solely inside an isolated MPC
        # branch, where requiring evaluation capability would incorrectly
        # delete every READY recourse action.
        if any(
            getattr(row, "artifact_token", None) == str(artifact)
            for row in scenario_rows
        ):
            return True
        if not self.requires_evaluation_pair:
            return True
        return kwargs.get("evaluation_pair_supported") is True

    supports_edge = has_edge_support

    def runtime_admission_bounds(
        self,
        *,
        action: Action,
        observation: Observation,
        evaluation_pair_supported: bool = False,
        rsu_context: DeviceContext | Mapping[str, Any] | str | None = None,
    ) -> RuntimeAdmissionBounds | None:
        """Certify one atomic RSU request without evaluation-future values.

        Support must exist in every conformal quality candidate cell.  Within
        each cell a formed scenario packet from the selected pipeline must have
        a paired edge measurement for the requested RSU/model/context.  The
        evaluation artifact is checked only as an opaque capability tuple; its
        work, memory, energy, FER and hidden quality are not retained here.

        Returned reservation quantities come exclusively from the frozen edge
        model deployment bounds in the profile.  Scenario rows are used to
        prove paired support and are defensively checked not to exceed those
        preregistered bounds.
        """

        if action.kind is not ActionKind.EDGE:
            return None
        rsu_id = action.rsu_id or ""
        model_id = action.edge_model_id or ""
        pipeline_id = observation.selected_pipeline or ""
        bins = tuple(
            dict.fromkeys(str(value) for value in observation.conformal_quality_bins)
        )
        model = self.profile.edge_models.get(model_id)
        pipeline = self.profile.pipelines.get(pipeline_id)
        if not rsu_id or model is None or pipeline is None or not bins:
            return None
        if (
            rsu_id not in model.supported_rsus
            or (
                model.supported_pipelines
                and pipeline_id not in model.supported_pipelines
            )
            or model.protocol_version != self.profile.protocol_version
            or pipeline.protocol_version != self.profile.protocol_version
            or self.scenario_profile_hash != self.profile.profile_hash
            or self.scenario_protocol_version != self.profile.protocol_version
        ):
            return None

        versions = observation.versions
        if (
            versions.get("profile_hash") != self.profile.profile_hash
            or versions.get("protocol_version") != self.profile.protocol_version
        ):
            return None
        edge_versions = versions.get("edge_models", {})
        if not isinstance(edge_versions, Mapping):
            return None
        active_model = edge_versions.get(model_id)
        if not isinstance(active_model, Mapping) or (
            active_model.get("model_hash") != model.model_hash
            or active_model.get("protocol_version", model.protocol_version)
            != model.protocol_version
        ):
            return None

        encoded = observation.encoded_evidence
        if (
            observation.artifact_token is None
            or encoded.get("artifact_token") != observation.artifact_token
            or encoded.get("pipeline_id") != pipeline_id
            or encoded.get("pipeline_hash") != pipeline.pipeline_hash
            or encoded.get("guard_hash") != pipeline.guard_hash
            or encoded.get("encoder_hash") != pipeline.encoder_hash
            or encoded.get("profile_hash") != self.profile.profile_hash
            or tuple(encoded.get("quality_bins", ())) != bins
        ):
            return None

        current_rsu_context = (
            self._ctx(
                observation.rsus.get(rsu_id, {}).get("device_context")
                if isinstance(observation.rsus.get(rsu_id), Mapping)
                else ""
            )
            if rsu_context is None
            else self._ctx(rsu_context)
        )
        if not self.has_edge_support(
            rsu_id=rsu_id,
            model_id=model_id,
            pipeline_id=pipeline_id,
            artifact_token=observation.artifact_token,
            evaluation_pair_supported=evaluation_pair_supported,
            quality_bins=bins,
            profile_hash=self.profile.profile_hash,
            device_type=observation.device_type,
            device_context=observation.device_context,
            rsu_context=current_rsu_context,
        ):
            return None

        # Use a detached observation only to select the current RSU context;
        # _edge_rows still requires paired scenario artifacts in every quality
        # cell and never reads an evaluation trace row.
        rsu_rows = {
            key: (
                {**dict(value), "device_context": encode_context(current_rsu_context)}
                if key == rsu_id and isinstance(value, Mapping)
                else value
            )
            for key, value in observation.rsus.items()
        }
        support_observation = observation
        if rsu_rows != dict(observation.rsus):
            support_observation = replace(observation, rsus=rsu_rows)
        rows = self._edge_rows(action, support_observation)
        if not rows or {str(row.quality_bin) for row in rows} != set(bins):
            return None
        if any(row.model_hash != model.model_hash for row in rows):
            return None

        raw_bounds = model.deployment_resource_bounds
        required = {
            "max_vram_bytes",
            "max_ingress_work_s",
            "max_ingress_energy_j",
            "max_gpu_work_s",
            "max_gpu_energy_j",
            "max_result_size_bits",
        }
        if not isinstance(raw_bounds, Mapping) or not required.issubset(raw_bounds):
            return None
        try:
            max_vram_bytes = int(raw_bounds["max_vram_bytes"])
            max_result_size_bits = int(raw_bounds["max_result_size_bits"])
            max_ingress_work_s = float(raw_bounds["max_ingress_work_s"])
            max_ingress_energy_j = float(raw_bounds["max_ingress_energy_j"])
            max_gpu_work_s = float(raw_bounds["max_gpu_work_s"])
            max_gpu_energy_j = float(raw_bounds["max_gpu_energy_j"])
        except (TypeError, ValueError, OverflowError):
            return None
        quantities = (
            max_ingress_work_s,
            max_ingress_energy_j,
            max_gpu_work_s,
            max_gpu_energy_j,
        )
        if (
            max_vram_bytes < 1
            or max_result_size_bits < 1
            or any(not math.isfinite(value) or value <= 0 for value in quantities)
        ):
            return None
        if any(
            int(row.vram_bytes) > max_vram_bytes
            or float(row.ingress_work_s) > max_ingress_work_s + 1e-12
            or float(row.ingress_energy_j) > max_ingress_energy_j + 1e-12
            or float(row.gpu_work_s) > max_gpu_work_s + 1e-12
            or float(row.gpu_energy_j) > max_gpu_energy_j + 1e-12
            or int(row.result_size_bits) > max_result_size_bits
            for row in rows
        ):
            return None
        return RuntimeAdmissionBounds(
            descriptor_count=1,
            max_vram_bytes=max_vram_bytes,
            max_ingress_work_s=max_ingress_work_s,
            max_ingress_energy_j=max_ingress_energy_j,
            max_gpu_work_s=max_gpu_work_s,
            max_gpu_energy_j=max_gpu_energy_j,
            max_result_size_bits=max_result_size_bits,
            rsu_id=rsu_id,
            model_id=model.model_id,
            model_hash=model.model_hash,
            pipeline_id=pipeline_id,
            protocol_version=self.profile.protocol_version,
            profile_hash=self.profile.profile_hash,
            candidate_quality_bins=bins,
            scenario_trace_version=self.scenario_trace_version,
            scenario_trace_hash=self.scenario_trace_hash,
            scenario_split_role=self.scenario_split_role,
        )

    def _local_rows(self, action: Action, obs: Observation) -> list[Any]:
        ctx = decode_context(obs.device_context)
        rows = [
            row
            for row in self.local_rows
            if row.model_id == action.local_model_id
            and row.device_type == obs.device_type
            and row.quality_bin in obs.conformal_quality_bins
            and row.context == ctx
        ]
        if {row.quality_bin for row in rows} != set(obs.conformal_quality_bins):
            return []
        return rows

    def _anon_rows(self, action: Action, obs: Observation) -> list[Any]:
        ctx = decode_context(obs.device_context)
        rows = [
            row
            for row in self.anon_rows
            if row.pipeline_id == action.pipeline_id
            and row.device_type == obs.device_type
            and row.quality_bin in obs.conformal_quality_bins
            and row.context == ctx
        ]
        if {row.quality_bin for row in rows} != set(obs.conformal_quality_bins):
            return []
        return rows

    def _edge_rows(self, action: Action, obs: Observation) -> list[Any]:
        rsu = obs.rsus.get(action.rsu_id or "", {})
        rsu_context_raw = (
            rsu.get("device_context") if isinstance(rsu, Mapping) else None
        )
        rsu_context = (
            None if rsu_context_raw is None else decode_context(rsu_context_raw)
        )
        vehicle_context = decode_context(obs.device_context)
        formed_by_quality = {
            quality_bin: {
                row.artifact_token
                for row in self.anon_rows
                if row.pipeline_id == obs.selected_pipeline
                and row.quality_bin == quality_bin
                and row.device_type == obs.device_type
                and row.context == vehicle_context
                and row.formed_packet
                and row.artifact_token
            }
            for quality_bin in obs.conformal_quality_bins
        }
        rows = [
            row
            for row in self.edge_rows
            if row.rsu_id == action.rsu_id
            and row.model_id == action.edge_model_id
            and row.pipeline_id == obs.selected_pipeline
            and row.quality_bin in obs.conformal_quality_bins
            and row.artifact_token in formed_by_quality.get(row.quality_bin, set())
            and (rsu_context is None or row.context == rsu_context)
        ]
        if not all(formed_by_quality.values()) or {
            row.quality_bin for row in rows
        } != set(obs.conformal_quality_bins):
            return []
        return rows

    @staticmethod
    def _valid_losses(rows: Iterable[Any]) -> list[float]:
        return [
            float(row.fer_loss)
            for row in rows
            if getattr(row, "fer_loss", None) is not None
            and not bool(getattr(row, "failed", False))
            and not bool(getattr(row, "ingress_failed", False))
        ]

    def action_bounds(
        self, *, action: Action, observation: Observation
    ) -> Mapping[str, Any]:
        if action.kind is ActionKind.FAIL:
            return self._with_macro_interval(
                {
                    "optimistic_duration_s": 0.0,
                    "expected_duration_s": self.controller_overhead_s,
                    "vehicle_energy_upper_j": 0.0,
                    "vehicle_memory_upper_bytes": 0,
                    "expected_loss": self.failure_loss,
                    "expected_failure": 1.0,
                },
                observation,
            )
        if action.kind is ActionKind.LOCAL:
            rows = self._local_rows(action, observation)
            if not rows:
                return {}
            model = self.profile.local_models.get(action.local_model_id or "")
            if model is None:
                return {}
            resource_bounds = model.deployment_resource_bounds
            losses = self._valid_losses(rows)
            expected_duration = fmean(row.service_work_s for row in rows)
            expected_energy = fmean(row.dynamic_energy_j for row in rows)
            failure = fmean(float(row.failed) for row in rows)
            loss = fmean(losses) if losses else self.failure_loss
            return self._with_macro_interval(
                {
                    "optimistic_duration_s": min(row.service_work_s for row in rows),
                    "expected_duration_s": expected_duration,
                    "vehicle_energy_upper_j": float(
                        resource_bounds["max_dynamic_energy_j"]
                    ),
                    "expected_vehicle_energy_j": expected_energy,
                    "vehicle_memory_upper_bytes": int(
                        resource_bounds["max_memory_bytes"]
                    ),
                    "descriptor_tokens": {"accelerator": 1},
                    "expected_loss": loss,
                    "expected_failure": failure,
                    "physical_workload_s": expected_duration,
                },
                observation,
                vehicle_resources=("accelerator",),
            )
        if action.kind is ActionKind.PIPE:
            rows = self._anon_rows(action, observation)
            if not rows:
                return {}
            pipeline = self.profile.pipelines.get(action.pipeline_id or "")
            if pipeline is None:
                return {}
            resource_bounds = pipeline.deployment_resource_bounds
            expected_duration = fmean(row.total_work_s for row in rows)
            expected_energy = fmean(row.total_energy_j for row in rows)
            formed = fmean(float(row.formed_packet) for row in rows)
            memory = int(resource_bounds["max_peak_memory_bytes"])
            if pipeline.fallback_local_model:
                fallback = self.profile.local_models.get(pipeline.fallback_local_model)
                if fallback is None:
                    return {}
                memory = max(
                    memory,
                    int(fallback.deployment_resource_bounds["max_memory_bytes"]),
                )
            energy_upper = pipeline.max_attempts * sum(
                float(resource_bounds[key])
                for key in (
                    "max_anon_energy_j",
                    "max_guard_energy_j",
                    "max_encode_energy_j",
                )
            )
            work_by_resource = {
                "accelerator": fmean(
                    sum(float(attempt.anon_work_s) for attempt in row.attempts)
                    for row in rows
                ),
                "cpu": fmean(
                    sum(float(attempt.guard_work_s or 0.0) for attempt in row.attempts)
                    for row in rows
                ),
                "encoder": fmean(
                    sum(float(attempt.encode_work_s or 0.0) for attempt in row.attempts)
                    for row in rows
                ),
            }
            optimistic = min(row.total_work_s for row in rows)
            return self._with_macro_interval(
                {
                    "optimistic_duration_s": optimistic,
                    "expected_duration_s": expected_duration,
                    "vehicle_energy_upper_j": energy_upper,
                    "expected_vehicle_energy_j": expected_energy,
                    "vehicle_memory_upper_bytes": memory,
                    "descriptor_tokens": {"accelerator": 1, "cpu": 1, "encoder": 1},
                    "expected_loss": 1.0 - formed,
                    "expected_failure": 1.0 - formed,
                    "physical_workload_s": expected_duration,
                    "vehicle_work_s": work_by_resource,
                },
                observation,
                vehicle_resources=("accelerator", "cpu", "encoder"),
                counts_task_terminal=False,
            )
        rows = self._edge_rows(action, observation)
        if not rows:
            return {}
        edge_model = self.profile.edge_models.get(action.edge_model_id or "")
        if edge_model is None:
            return {}
        edge_resource_bounds = edge_model.deployment_resource_bounds
        link = observation.links.get(action.rsu_id or "", {})
        if not isinstance(link, Mapping):
            link = {}
        ul_rate = float(link.get("ul_goodput_bps", 0.0) or 0.0)
        dl_rate = float(link.get("dl_goodput_bps", 0.0) or 0.0)
        ul_bits = float(observation.remaining_bits.get("uplink", 0.0) or 0.0)
        if ul_bits <= 0 and observation.encoded_size_bytes:
            ul_bits = observation.encoded_size_bytes * 8 + self.metadata_bits
        compute = [
            row.ingress_work_s + (0.0 if row.ingress_failed else row.gpu_work_s)
            for row in rows
        ]
        results = [
            float(row.result_size_bits)
            if not row.ingress_failed and not row.failed
            else 0.0
            for row in rows
        ]
        link_available = ul_rate > 0 and dl_rate > 0
        ul_duration = ul_bits / ul_rate if link_available else None
        expected_dl_duration = fmean(results) / dl_rate if link_available else None
        optimistic_wireless = (
            ul_bits / ul_rate + min(results) / dl_rate if link_available else None
        )
        expected_duration = (
            fmean(compute) + ul_duration + expected_dl_duration
            if ul_duration is not None and expected_dl_duration is not None
            else None
        )
        compute_rsu_energy = fmean(
            row.ingress_energy_j + (0.0 if row.ingress_failed else row.gpu_energy_j)
            for row in rows
        )
        expected_vehicle_energy = (
            ul_duration * float(link.get("ul_transmitter_power_w", 0.0) or 0.0)
            + expected_dl_duration * float(link.get("dl_receiver_power_w", 0.0) or 0.0)
            if ul_duration is not None and expected_dl_duration is not None
            else None
        )
        expected_rsu_energy = (
            compute_rsu_energy
            + ul_duration * float(link.get("ul_receiver_power_w", 0.0) or 0.0)
            + expected_dl_duration
            * float(link.get("dl_transmitter_power_w", 0.0) or 0.0)
            if ul_duration is not None and expected_dl_duration is not None
            else compute_rsu_energy
        )
        vehicle_energy_upper = (
            (ul_bits / ul_rate) * float(link.get("ul_transmitter_power_w", 0.0) or 0.0)
            + (max(results) / dl_rate)
            * float(link.get("dl_receiver_power_w", 0.0) or 0.0)
            if link_available
            else None
        )
        losses = self._valid_losses(rows)
        failure = fmean(float(row.ingress_failed or row.failed) for row in rows)
        loss = fmean(losses) if losses else self.failure_loss
        return self._with_macro_interval(
            {
                "optimistic_duration_s": (
                    min(compute) + optimistic_wireless
                    if optimistic_wireless is not None
                    else None
                ),
                "expected_duration_s": expected_duration,
                "vehicle_energy_upper_j": (
                    max(
                        float(link.get("uplink_start_energy_j", 0.001)),
                        vehicle_energy_upper,
                    )
                    if vehicle_energy_upper is not None
                    else None
                ),
                "expected_vehicle_energy_j": expected_vehicle_energy,
                "vehicle_memory_upper_bytes": 0,
                "rsu_descriptor_count": 1,
                "rsu_vram_upper_bytes": int(edge_resource_bounds["max_vram_bytes"]),
                "rsu_work_upper_gpu_s": float(edge_resource_bounds["max_gpu_work_s"]),
                "expected_rsu_energy_j": expected_rsu_energy,
                "expected_loss": loss,
                "expected_failure": failure,
                "physical_workload_s": fmean(
                    0.0 if row.ingress_failed else row.gpu_work_s for row in rows
                ),
                "visible_queue_s": float(
                    observation.rsus.get(action.rsu_id or "", {}).get(
                        "gpu_residual_work_s", 0.0
                    )
                ),
                "link_cost": (1.0 / ul_rate if link_available else None),
            },
            observation,
            rsu_id=action.rsu_id,
        )

    def information_ablation_bounds(
        self, *, action: Action, observation: Observation
    ) -> Mapping[str, float]:
        """Conservative READY information replacements from frozen support."""

        if action.kind is not ActionKind.EDGE or not action.rsu_id:
            return {}
        link = observation.links.get(action.rsu_id, {})
        rsu = observation.rsus.get(action.rsu_id, {})
        if not isinstance(link, Mapping) or not isinstance(rsu, Mapping):
            return {}
        ul_rate = float(link.get("ul_goodput_bps", 0.0) or 0.0)
        if ul_rate <= 0.0 or not observation.selected_pipeline:
            return {}
        rows = [
            row
            for row in self.anon_rows
            if row.pipeline_id == observation.selected_pipeline
            and row.quality_bin in observation.conformal_quality_bins
            and row.device_type == observation.device_type
            and row.context == decode_context(observation.device_context)
            and row.formed_packet
        ]
        if not rows:
            return {}
        observed_bits = float(
            observation.remaining_bits.get("uplink", 0.0)
            or ((observation.encoded_size_bytes or 0) * 8 + self.metadata_bits)
        )
        conservative_bits = float(
            max(row.final_encoded_size_bytes for row in rows) * 8 + self.metadata_bits
        )
        conservative_bits = max(observed_bits, conservative_bits)
        latency_weight = float(self.cost_weights.get("latency", 1.0))
        vehicle_weight = float(self.cost_weights.get("vehicle_energy", 1.0))
        rsu_weight = float(self.cost_weights.get("rsu_energy", 1.0))
        vehicle_power = float(link.get("ul_transmitter_power_w", 0.0) or 0.0)
        rsu_power = float(link.get("ul_receiver_power_w", 0.0) or 0.0)

        def uplink_cost(bits: float) -> float:
            duration = bits / ul_rate
            return (
                latency_weight * duration / self.latency_scale_s
                + vehicle_weight
                * duration
                * vehicle_power
                / self.vehicle_energy_scale_j
                + rsu_weight * duration * rsu_power / self.rsu_energy_scale_j
            )

        observed_queue_s = max(0.0, float(rsu.get("gpu_residual_work_s", 0.0)))
        conservative_queue_s = max(
            observed_queue_s,
            float(rsu.get("workload_capacity_gpu_s", observed_queue_s)),
        )
        return {
            "observed_output_size_cost": uplink_cost(observed_bits),
            "conservative_output_size_cost": uplink_cost(conservative_bits),
            "observed_fresh_queue_cost": latency_weight
            * observed_queue_s
            / self.latency_scale_s,
            "conservative_stale_queue_cost": latency_weight
            * conservative_queue_s
            / self.latency_scale_s,
        }


__all__ = [
    "FrozenEstimateProvider",
    "RuntimeAdmissionBounds",
    "decode_context",
    "encode_context",
]
