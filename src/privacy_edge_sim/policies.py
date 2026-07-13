"""Hard-safe two-stage baselines and ESL-SMPC controllers.

Every controller in this module obtains candidates from the same
``HardMaskEngine`` and passes its proposal through the same
``DeterministicRepair`` execution gate.  No controller receives task labels,
identity, attack truth, realized test FER or an environment trace cursor.

``SafeLyapunovPolicy`` is the H=1 drift--cost ratio controller.
``ESLSMPCPolicy`` is an empirical H>1 extension; it intentionally makes no
claim that multi-step rollout inherits every H=1 theorem.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
import time
from dataclasses import dataclass, field, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

from .certificates import finite_scenario_ratio_certificate
from .enums import ActionKind, ActionStage, TaskState, TransferDirection
from .events import _same_representable_instant, _strict_future_instant
from .profiles import deep_freeze, thaw_json
from .safety import (
    Action,
    DeterministicRepair,
    HardMaskEngine,
    MaskResult,
    Observation,
    RepairDecision,
    action_estimate,
)
from .state import SimulationState, TaskRecord
from .traces import ScenarioEnvironment, TraceBundle


_EPS_DURATION_S = 1e-9
_RSU_WORKLOAD_CAPACITY_TOLERANCE_GPU_S = 1e-12


def _finite(value: Any, default: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _weight(weights: Mapping[str, float], *names: str, default: float) -> float:
    for name in names:
        if name in weights:
            return max(0.0, _finite(weights[name], default))
    return default


def _duration(
    row: Mapping[str, Any], *, fail_default: float = _EPS_DURATION_S
) -> float:
    """Return the causal estimate of the next macro-decision interval.

    Providers may expose ``expected_next_macro_interval_s`` when they have a
    frozen event-time predictor.  ``expected_duration_s`` is the conservative
    fallback used by the bundled estimator; it represents the interval until
    the action's next decision/terminal event, never a unit MDP step.
    """

    value = _finite(
        row.get(
            "expected_next_macro_interval_s",
            row.get(
                "expected_duration_s",
                row.get(
                    "duration_s",
                    row.get("duration_lower_bound_s", row.get("optimistic_duration_s")),
                ),
            ),
        ),
        fail_default,
    )
    return max(_EPS_DURATION_S, fail_default if value <= 0 else value)


def _cost_config(mask_engine: HardMaskEngine) -> Any:
    return None if mask_engine.config is None else mask_engine.config.cost


def _task_cost_from_row(
    action: Action,
    observation: Observation,
    mask_engine: HardMaskEngine,
    row: Mapping[str, Any],
) -> float:
    config = _cost_config(mask_engine)
    weights: Mapping[str, float] = {} if config is None else config.weights
    latency_scale = 1.0 if config is None else config.latency_scale_s
    vehicle_scale = 1.0 if config is None else config.vehicle_energy_scale_j
    rsu_scale = 1.0 if config is None else config.rsu_energy_scale_j
    utility_scale = 1.0 if config is None else config.utility_scale
    failure_loss = 1e6 if config is None else config.failure_loss
    latency = _duration(row)
    vehicle_energy = max(
        0.0,
        _finite(
            row.get(
                "interval_expected_vehicle_energy_j",
                row.get("expected_vehicle_energy_j", row.get("vehicle_energy_j")),
            ),
            0.0,
        ),
    )
    rsu_energy = max(
        0.0,
        _finite(
            row.get(
                "interval_expected_rsu_energy_j",
                row.get("expected_rsu_energy_j", row.get("rsu_energy_j")),
            ),
            0.0,
        ),
    )
    utility_loss = max(
        0.0,
        _finite(
            row.get(
                "interval_expected_loss",
                row.get(
                    "expected_fer_loss", row.get("expected_loss", row.get("fer_loss"))
                ),
            ),
            0.0,
        ),
    )
    fail_probability = max(
        0.0,
        min(
            1.0,
            _finite(
                row.get(
                    "interval_expected_failure",
                    row.get("failure_probability", row.get("expected_failure")),
                ),
                0.0,
            ),
        ),
    )
    timeout_probability = max(
        0.0, min(1.0, _finite(row.get("timeout_probability"), 0.0))
    )
    if action.kind is ActionKind.FAIL and "expected_next_macro_interval_s" not in row:
        fail_probability = 1.0
        latency = (
            _EPS_DURATION_S
            if mask_engine.config is None
            else max(
                _EPS_DURATION_S,
                mask_engine.config.controller.controller_overhead_s,
            )
        )
    return (
        _weight(weights, "latency", "latency_weight", default=1.0)
        * latency
        / latency_scale
        + _weight(weights, "vehicle_energy", "vehicle_energy_weight", default=1.0)
        * vehicle_energy
        / vehicle_scale
        + _weight(weights, "rsu_energy", "rsu_energy_weight", default=1.0)
        * rsu_energy
        / rsu_scale
        + _weight(weights, "utility", "fer", "utility_weight", default=1.0)
        * utility_loss
        / utility_scale
        + _weight(weights, "failure", "failure_weight", default=1.0)
        * failure_loss
        * fail_probability
        + _weight(weights, "timeout", "timeout_weight", default=1.0)
        * failure_loss
        * timeout_probability
    )


def expected_task_cost(
    action: Action, observation: Observation, mask_engine: HardMaskEngine
) -> float:
    """Normalized expected task cost from frozen action-level trace summaries."""

    row = action_estimate(action, observation, mask_engine.trace_support)
    return _task_cost_from_row(action, observation, mask_engine, row)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    policy_name: str
    proposed: Action
    executed: Action
    scores: Mapping[Action, float]
    mask: MaskResult
    repair: RepairDecision
    controller_wallclock_s: float
    diagnostics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def audit_row(self) -> dict[str, Any]:
        return {
            "policy": self.policy_name,
            "proposed": self.proposed.to_dict(),
            "executed": self.executed.to_dict(),
            "scores": {
                action.canonical_id: value
                for action, value in sorted(self.scores.items())
            },
            "controller_wallclock_s": self.controller_wallclock_s,
            "repair": self.repair.audit_row(),
            "diagnostics": thaw_json(self.diagnostics),
        }


class Policy(Protocol):
    def decide(
        self,
        task: TaskRecord,
        observation: Observation,
        state: SimulationState | None = None,
    ) -> PolicyDecision: ...


class SafePolicy:
    """Base class that enforces the shared mask/repair path."""

    name = "safe_policy"

    def __init__(
        self,
        mask_engine: HardMaskEngine,
        repairer: DeterministicRepair | None = None,
    ) -> None:
        self.mask_engine = mask_engine
        self.repairer = repairer or DeterministicRepair(mask_engine)

    def _scores(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: MaskResult,
        state: SimulationState | None,
    ) -> Mapping[Action, float]:
        return MappingProxyType(
            {
                action: expected_task_cost(action, observation, self.mask_engine)
                for action in mask.allowed
            }
        )

    def _score_state_snapshot(self) -> Any:
        """Return mutable policy-only state touched while computing scores."""

        return None

    def _restore_score_state(self, snapshot: Any) -> None:
        """Restore policy-only diagnostic state after a pure rescore."""

    def score_current_actions(
        self,
        task: TaskRecord,
        observation: Observation,
        state: SimulationState | None = None,
        *,
        mask: MaskResult | None = None,
    ) -> Mapping[Action, float]:
        """Purely rescore every action in the current hard-safe set.

        Delayed execution may invalidate the decision-epoch argmin and may
        also make a previously removed alternative safe.  This method applies
        the policy's own H=1 drift or H>1 scenario objective to the current
        mask without calling ``decide`` and without consuming mutable random
        state.  Implementations use content-addressed local RNGs and isolated
        rollout branches; the only mutable score side effect is diagnostics,
        which is restored before returning.
        """

        current_mask = mask or self.mask_engine.enumerate(task, observation, state)
        snapshot = self._score_state_snapshot()
        try:
            raw_scores = self._scores(task, observation, current_mask, state)
        finally:
            self._restore_score_state(snapshot)
        missing = tuple(
            action for action in current_mask.allowed if action not in raw_scores
        )
        if missing:
            raise RuntimeError(
                "policy current-score API omitted hard-safe actions: "
                + ", ".join(action.canonical_id for action in missing)
            )
        return MappingProxyType(
            {action: raw_scores[action] for action in current_mask.allowed}
        )

    def _propose(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: MaskResult,
        scores: Mapping[Action, float],
    ) -> Action:
        return min(
            mask.allowed,
            key=lambda action: (scores.get(action, math.inf), action.sort_key),
        )

    def _diagnostics(self) -> Mapping[str, Any]:
        return MappingProxyType({})

    def decide(
        self,
        task: TaskRecord,
        observation: Observation,
        state: SimulationState | None = None,
    ) -> PolicyDecision:
        started = time.perf_counter()
        mask = self.mask_engine.enumerate(task, observation, state)
        scores = self._scores(task, observation, mask, state)
        proposed = self._propose(task, observation, mask, scores)
        repair = self.repairer.repair(proposed, task, observation, state, score=scores)
        elapsed = time.perf_counter() - started
        return PolicyDecision(
            self.name,
            proposed,
            repair.executed,
            MappingProxyType(dict(sorted(scores.items()))),
            mask,
            repair,
            elapsed,
            self._diagnostics(),
        )

    def choose_action(
        self,
        task: TaskRecord,
        observation: Observation,
        state: SimulationState | None = None,
    ) -> Action:
        return self.decide(task, observation, state).executed

    select_action = choose_action


def _fail(mask: MaskResult) -> Action:
    action = Action.fail(mask.stage)
    if action not in mask.allowed:
        raise RuntimeError("explicit FAIL must remain in every hard-safe action set")
    return action


def _frozen_fallback(
    task: TaskRecord, mask: MaskResult, mask_engine: HardMaskEngine
) -> Action:
    pipeline = mask_engine.profile.pipelines.get(task.selected_pipeline or "")
    fallback = None if pipeline is None else pipeline.fallback_local_model
    for action in mask.allowed:
        if action.kind is ActionKind.LOCAL and action.local_model_id == fallback:
            return action
    return _fail(mask)


def _globally_safe_fixed_pipeline(mask_engine: HardMaskEngine) -> str:
    """Choose one preregistered pipeline safe in every frozen profile cell."""

    config = mask_engine.config
    for pipeline_id, pipeline in sorted(mask_engine.profile.pipelines.items()):
        if all(
            mask_engine.profile.query_privacy(
                pipeline_id,
                mask_engine.profile.quality_bins,
                device_type,
                **(
                    {
                        "risk_threshold": config.privacy.risk_threshold,
                        "min_subjects": config.privacy.min_subjects,
                        "min_emission_lcb": config.privacy.min_emission_lcb,
                    }
                    if config is not None
                    else {}
                ),
            ).safe
            for device_type in pipeline.supported_devices
        ):
            return pipeline_id
    # A fixed-safe baseline cannot silently adapt task-by-task.  Keeping a
    # deterministic pipeline still yields conservative local fallback when
    # no pipeline is globally certified.
    return min(mask_engine.profile.pipelines)


class AllLocalPolicy(SafePolicy):
    """Use the lowest lexical hard-safe local model at both decision stages."""

    name = "all_local"

    def _propose(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: MaskResult,
        scores: Mapping[Action, float],
    ) -> Action:
        local = [action for action in mask.allowed if action.kind is ActionKind.LOCAL]
        return min(local) if local else _fail(mask)


class _FixedSafePipelinePolicy(SafePolicy):
    def __init__(
        self,
        mask_engine: HardMaskEngine,
        repairer: DeterministicRepair | None = None,
        *,
        pipeline_id: str | None = None,
        edge_model_id: str | None = None,
    ) -> None:
        super().__init__(mask_engine, repairer)
        # The baseline registration freezes one pipeline/model for the entire
        # policy instance.  Per-task re-selection would silently turn this
        # baseline into a quality-adaptive policy.
        self.pipeline_id = pipeline_id or _globally_safe_fixed_pipeline(mask_engine)
        self.edge_model_id = edge_model_id or min(mask_engine.profile.edge_models)
        if self.pipeline_id not in mask_engine.profile.pipelines:
            raise ValueError(f"unknown fixed pipeline_id: {self.pipeline_id}")
        if self.edge_model_id not in mask_engine.profile.edge_models:
            raise ValueError(f"unknown fixed edge_model_id: {self.edge_model_id}")

    def _raw_action(self, mask: MaskResult) -> Action:
        pipeline_actions = [
            action for action in mask.allowed if action.kind is ActionKind.PIPE
        ]
        pipeline_actions = [
            action
            for action in pipeline_actions
            if action.pipeline_id == self.pipeline_id
        ]
        if pipeline_actions:
            return min(pipeline_actions)
        local = [action for action in mask.allowed if action.kind is ActionKind.LOCAL]
        return min(local) if local else _fail(mask)

    def _edge_actions(self, mask: MaskResult) -> list[Action]:
        actions = [action for action in mask.allowed if action.kind is ActionKind.EDGE]
        actions = [
            action for action in actions if action.edge_model_id == self.edge_model_id
        ]
        return actions


class FixedSafeLowestLinkCostPolicy(_FixedSafePipelinePolicy):
    """Fixed safe pipeline, then the currently lowest observable link cost."""

    name = "fixed_safe_lowest_link_cost"

    @staticmethod
    def _link_cost(action: Action, observation: Observation) -> float:
        link = _mapping(observation.links.get(action.rsu_id or ""))
        for key in ("link_cost", "cost", "estimated_ul_duration_s", "distance_m"):
            value = _finite(link.get(key), math.nan)
            if math.isfinite(value):
                return value
        goodput = _finite(
            link.get(
                "ul_goodput_bps",
                link.get("goodput_bps", link.get("max_goodput_bps")),
            ),
            0.0,
        )
        return math.inf if goodput <= 0 else 1.0 / goodput

    def _propose(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: MaskResult,
        scores: Mapping[Action, float],
    ) -> Action:
        if observation.stage is ActionStage.RAW:
            return self._raw_action(mask)
        edges = self._edge_actions(mask)
        if edges:
            return min(
                edges,
                key=lambda action: (
                    self._link_cost(action, observation),
                    action.sort_key,
                ),
            )
        return _frozen_fallback(task, mask, self.mask_engine)


class FixedSafeShortestQueuePolicy(_FixedSafePipelinePolicy):
    """Fixed safe pipeline, then shortest currently visible RSU work queue."""

    name = "fixed_safe_shortest_visible_queue"

    @staticmethod
    def _queue_metric(action: Action, observation: Observation) -> tuple[float, int]:
        row = _mapping(observation.rsus.get(action.rsu_id or ""))
        work = max(0.0, _finite(row.get("ingress_residual_work_s"), 0.0)) + max(
            0.0, _finite(row.get("gpu_residual_work_s"), 0.0)
        )
        count = int(row.get("ingress_waiting", 0)) + int(row.get("gpu_waiting", 0))
        return work, count

    def _propose(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: MaskResult,
        scores: Mapping[Action, float],
    ) -> Action:
        if observation.stage is ActionStage.RAW:
            return self._raw_action(mask)
        edges = self._edge_actions(mask)
        if edges:
            return min(
                edges,
                key=lambda action: (
                    self._queue_metric(action, observation),
                    action.sort_key,
                ),
            )
        return _frozen_fallback(task, mask, self.mask_engine)


class SafeOneShotCommitmentPolicy(SafePolicy):
    """Experimental one-shot RAW commitment with safe READY repair.

    RAW freezes a pipeline, RSU and edge model using only that timestamp's
    public observation.  The RSU snapshot is explicitly *not* a reservation.
    READY attempts exactly the frozen edge action; the shared hard mask and
    deterministic fallback remain authoritative if state changed meanwhile.
    """

    name = "safe_one_shot"

    def __init__(
        self,
        mask_engine: HardMaskEngine,
        repairer: DeterministicRepair | None = None,
    ) -> None:
        super().__init__(mask_engine, repairer)
        self._commitments: dict[str, Mapping[str, Any]] = {}
        self._last_commitment_diagnostics: Mapping[str, Any] = MappingProxyType({})

    def _edge_commitment_candidates(
        self, pipeline_id: str, observation: Observation
    ) -> list[tuple[float, str, str]]:
        config = self.mask_engine.config
        active_models = _mapping(observation.versions.get("edge_models"))
        candidates: list[tuple[float, str, str]] = []
        for rsu_id, raw_rsu in sorted(observation.rsus.items()):
            rsu = _mapping(raw_rsu)
            link = _mapping(observation.links.get(rsu_id))
            if (
                not bool(link.get("connected", False))
                or bool(rsu.get("failed", True))
                or (
                    config is not None
                    and _finite(rsu.get("snapshot_age_s"), math.inf)
                    > config.max_snapshot_age_s
                )
                or int(rsu.get("descriptors", 0))
                >= int(rsu.get("descriptor_capacity", 0))
                or int(rsu.get("vram_bytes", 0))
                >= int(rsu.get("vram_capacity_bytes", 0))
                or _finite(rsu.get("reserved_work_gpu_s"), 0.0)
                >= _finite(rsu.get("workload_capacity_gpu_s"), 0.0)
            ):
                continue
            cached = _mapping(rsu.get("cached_models"))
            for model_id, model in sorted(self.mask_engine.profile.edge_models.items()):
                active = _mapping(active_models.get(model_id))
                if (
                    rsu_id not in model.supported_rsus
                    or (
                        model.supported_pipelines
                        and pipeline_id not in model.supported_pipelines
                    )
                    or cached.get(model_id) != model.model_hash
                    or (
                        active
                        and (
                            active.get("model_hash") != model.model_hash
                            or active.get("protocol_version") != model.protocol_version
                        )
                    )
                ):
                    continue
                probe = Action.edge(rsu_id, model_id)
                link_cost = FixedSafeLowestLinkCostPolicy._link_cost(probe, observation)
                if not math.isfinite(link_cost):
                    continue
                candidates.append(
                    (
                        link_cost,
                        rsu_id,
                        model_id,
                    )
                )
        return candidates

    def _propose(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: MaskResult,
        scores: Mapping[Action, float],
    ) -> Action:
        if observation.stage is ActionStage.RAW:
            greedy = min(
                mask.allowed,
                key=lambda action: (scores.get(action, math.inf), action.sort_key),
            )
            if greedy.kind is ActionKind.PIPE:
                candidates = self._edge_commitment_candidates(
                    greedy.pipeline_id or "", observation
                )
                if candidates:
                    link_cost, rsu_id, model_id = min(candidates)
                    commitment = deep_freeze(
                        {
                            "task_token": task.task_id,
                            "committed_at_s": observation.time_s,
                            "pipeline_id": greedy.pipeline_id,
                            "rsu_id": rsu_id,
                            "edge_model_id": model_id,
                            "observed_link_cost": link_cost,
                            "snapshot_age_s": _finite(
                                _mapping(observation.rsus.get(rsu_id)).get(
                                    "snapshot_age_s"
                                ),
                                math.inf,
                            ),
                            "snapshot_is_reservation": False,
                            "profile_hash": self.mask_engine.profile.profile_hash,
                            "protocol_version": self.mask_engine.profile.protocol_version,
                        }
                    )
                    self._commitments[task.task_id] = commitment
                    self._last_commitment_diagnostics = deep_freeze(
                        {"phase": "RAW_COMMIT", "commitment": commitment}
                    )
                    return greedy
            else:
                self._commitments.pop(task.task_id, None)
                self._last_commitment_diagnostics = deep_freeze(
                    {
                        "phase": "RAW_COMMIT",
                        "commitment": None,
                        "reason": "GREEDY_FIRST_ACTION_IS_NOT_PIPE",
                        "repair_action": greedy.canonical_id,
                    }
                )
                return greedy
            local = [
                action for action in mask.allowed if action.kind is ActionKind.LOCAL
            ]
            selected = min(local) if local else _fail(mask)
            self._last_commitment_diagnostics = deep_freeze(
                {
                    "phase": "RAW_COMMIT",
                    "commitment": None,
                    "reason": "NO_HARD_SAFE_COMPLETE_COMMITMENT",
                    "repair_action": selected.canonical_id,
                }
            )
            return selected

        commitment = self._commitments.pop(task.task_id, None)
        if commitment is None:
            selected = _frozen_fallback(task, mask, self.mask_engine)
            self._last_commitment_diagnostics = deep_freeze(
                {
                    "phase": "READY_ATTEMPT",
                    "commitment": None,
                    "hard_mask_allowed": False,
                    "reason": "COMMITMENT_MISSING",
                    "repair_action": selected.canonical_id,
                }
            )
            return selected
        committed = Action.edge(
            str(commitment["rsu_id"]), str(commitment["edge_model_id"])
        )
        pipeline_matches = task.selected_pipeline == commitment["pipeline_id"]
        if pipeline_matches and committed in mask.allowed:
            self._last_commitment_diagnostics = deep_freeze(
                {
                    "phase": "READY_ATTEMPT",
                    "commitment": commitment,
                    "hard_mask_allowed": True,
                    "outcome": "COMMITTED_EDGE",
                    "repair_action": None,
                }
            )
            return committed
        selected = _frozen_fallback(task, mask, self.mask_engine)
        reasons = (
            ("PIPELINE_CHANGED",)
            if not pipeline_matches
            else tuple(reason.value for reason in mask.reasons_for(committed))
        )
        self._last_commitment_diagnostics = deep_freeze(
            {
                "phase": "READY_ATTEMPT",
                "commitment": commitment,
                "hard_mask_allowed": False,
                "outcome": "DETERMINISTIC_FALLBACK",
                "reason_codes": reasons,
                "repair_action": selected.canonical_id,
            }
        )
        return selected

    def _diagnostics(self) -> Mapping[str, Any]:
        return self._last_commitment_diagnostics

    def on_task_terminal(self, task_id: str) -> None:
        """Discard non-physical commitment bookkeeping at absorption."""

        self._commitments.pop(task_id, None)


class SafeGreedyPolicy(SafePolicy):
    """Minimum immediate normalized expected cost inside the hard-safe set."""

    name = "safe_greedy"


def _quadratic_increment(queue: float, arrival: float, service: float = 0.0) -> float:
    after = max(0.0, queue + arrival - service)
    return 0.5 * (after * after - queue * queue)


def _work_mapping(value: Any, default_key: str) -> Mapping[str, float]:
    if isinstance(value, Mapping):
        return {str(key): max(0.0, _finite(item, 0.0)) for key, item in value.items()}
    scalar = _finite(value, 0.0)
    return {default_key: max(0.0, scalar)} if scalar > 0 else {}


class SafeLyapunovPolicy(SafePolicy):
    """H=1 safe Lyapunov drift--cost ratio controller.

    The denominator is the action's predicted real duration in seconds, not a
    unit MDP step.  The numerator contains long-term virtual queues and the
    quadratic physical-workload increment in addition to normalized cost.
    """

    name = "safe_lyapunov_h1"

    def __init__(
        self,
        mask_engine: HardMaskEngine,
        repairer: DeterministicRepair | None = None,
        *,
        lyapunov_v: float | None = None,
        physical_queue_weight: float | None = None,
        vehicle_resource_theta: Mapping[str, float] | None = None,
        rsu_resource_theta: Mapping[str, float] | None = None,
        scenario_source: Any = None,
        scenarios: int | None = None,
    ) -> None:
        super().__init__(mask_engine, repairer)
        configured = (
            None
            if mask_engine.config is None
            else mask_engine.config.controller.lyapunov_v
        )
        self.lyapunov_v = float(
            lyapunov_v if lyapunov_v is not None else configured if configured else 1.0
        )
        configured_physical_weight = (
            None
            if mask_engine.config is None
            else mask_engine.config.controller.physical_queue_weight
        )
        self.physical_queue_weight = float(
            physical_queue_weight
            if physical_queue_weight is not None
            else configured_physical_weight
            if configured_physical_weight is not None
            else 1.0
        )
        configured_vehicle_theta = (
            {}
            if mask_engine.config is None
            else mask_engine.config.controller.vehicle_resource_theta
        )
        configured_rsu_theta = (
            {}
            if mask_engine.config is None
            else mask_engine.config.controller.rsu_resource_theta
        )
        self.vehicle_resource_theta = MappingProxyType(
            {
                str(name): float(value)
                for name, value in (
                    configured_vehicle_theta
                    if vehicle_resource_theta is None
                    else vehicle_resource_theta
                ).items()
            }
        )
        self.rsu_resource_theta = MappingProxyType(
            {
                str(name): float(value)
                for name, value in (
                    configured_rsu_theta
                    if rsu_resource_theta is None
                    else rsu_resource_theta
                ).items()
            }
        )
        self.scenario_source = scenario_source
        self.h1_scenarios = int(
            scenarios
            if scenarios is not None
            else mask_engine.config.controller.scenarios
            if mask_engine.config is not None
            else 1
        )
        self._last_h1_diagnostics: Mapping[str, Any] = MappingProxyType({})
        if not math.isfinite(self.lyapunov_v) or self.lyapunov_v <= 0:
            raise ValueError("lyapunov_v must be finite and positive")
        if (
            not math.isfinite(self.physical_queue_weight)
            or self.physical_queue_weight < 0
        ):
            raise ValueError("physical_queue_weight must be finite and nonnegative")
        for family, values in (
            ("vehicle_resource_theta", self.vehicle_resource_theta),
            ("rsu_resource_theta", self.rsu_resource_theta),
        ):
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError(f"{family} values must be finite and nonnegative")
        if self.h1_scenarios < 1:
            raise ValueError("scenarios must be >= 1")

    def _theta(self, owner: str, resource: str) -> float:
        values = (
            self.vehicle_resource_theta
            if owner == "vehicle"
            else self.rsu_resource_theta
        )
        return values.get(resource, self.physical_queue_weight)

    def _physical_drift(
        self, action: Action, observation: Observation, row: Mapping[str, Any]
    ) -> float:
        result = 0.0
        vehicle_resources = _mapping(observation.vehicle.get("resources"))
        additions = (
            _work_mapping(
                row.get(
                    "vehicle_work_s",
                    row.get("added_vehicle_work_s", row.get("physical_workload_s")),
                ),
                "accelerator",
            )
            if action.kind in {ActionKind.LOCAL, ActionKind.PIPE}
            else {}
        )
        services = _work_mapping(row.get("vehicle_service_s"), "accelerator")
        for resource in sorted(set(additions) | set(services)):
            arrival = additions.get(resource, 0.0)
            queue = max(
                0.0,
                _finite(
                    _mapping(vehicle_resources.get(resource)).get("residual_work_s"),
                    0.0,
                ),
            )
            result += self._theta("vehicle", resource) * _quadratic_increment(
                queue, arrival, services.get(resource, 0.0)
            )

        if action.rsu_id:
            rsu = _mapping(observation.rsus.get(action.rsu_id))
            ingress_q = max(0.0, _finite(rsu.get("ingress_residual_work_s"), 0.0))
            gpu_q = max(0.0, _finite(rsu.get("gpu_residual_work_s"), 0.0))
            ingress_a = max(0.0, _finite(row.get("rsu_ingress_work_s"), 0.0))
            gpu_a = max(
                0.0,
                _finite(
                    row.get(
                        "rsu_gpu_work_s",
                        row.get(
                            "conservative_work_gpu_s", row.get("physical_workload_s")
                        ),
                    ),
                    0.0,
                ),
            )
            ingress_service = max(0.0, _finite(row.get("rsu_ingress_service_s"), 0.0))
            gpu_service = max(0.0, _finite(row.get("rsu_gpu_service_s"), 0.0))
            result += self._theta("rsu", "ingress") * _quadratic_increment(
                ingress_q, ingress_a, ingress_service
            )
            result += self._theta("rsu", "gpu") * _quadratic_increment(
                gpu_q, gpu_a, gpu_service
            )
        return result

    def _virtual_drift(
        self, action: Action, observation: Observation, row: Mapping[str, Any]
    ) -> float:
        queues = observation.virtual_queues
        explicit = row.get("virtual_queue_increments")
        if isinstance(explicit, Mapping):
            total = 0.0
            for name, increment in explicit.items():
                if name == "vehicle_power":
                    q = _finite(
                        _mapping(queues.get("vehicle_power")).get(
                            observation.vehicle_id
                        ),
                        0.0,
                    )
                elif name == "rsu_power" and action.rsu_id:
                    q = _finite(
                        _mapping(queues.get("rsu_power")).get(action.rsu_id), 0.0
                    )
                else:
                    q = _finite(queues.get(str(name)), 0.0)
                delta = _finite(increment, 0.0)
                after = max(0.0, q + delta)
                total += 0.5 * (after * after - q * q)
            return total

        vehicle_q = _finite(
            _mapping(queues.get("vehicle_power")).get(observation.vehicle_id), 0.0
        )
        rsu_q = (
            0.0
            if not action.rsu_id
            else _finite(_mapping(queues.get("rsu_power")).get(action.rsu_id), 0.0)
        )
        timeout_q = _finite(queues.get("timeout"), 0.0)
        failure_q = _finite(queues.get("failure"), 0.0)
        coverage_q = _finite(queues.get("coverage"), 0.0)
        vehicle_energy = max(
            0.0,
            _finite(
                row.get(
                    "interval_expected_vehicle_energy_j",
                    row.get("expected_vehicle_energy_j", row.get("vehicle_energy_j")),
                ),
                0.0,
            ),
        )
        rsu_energy = max(
            0.0,
            _finite(
                row.get(
                    "interval_expected_rsu_energy_j",
                    row.get("expected_rsu_energy_j", row.get("rsu_energy_j")),
                ),
                0.0,
            ),
        )
        timeout_probability = max(
            0.0, min(1.0, _finite(row.get("timeout_probability"), 0.0))
        )
        failure_probability = max(
            0.0,
            min(
                1.0,
                _finite(
                    row.get(
                        "interval_expected_failure",
                        row.get("failure_probability", row.get("expected_failure")),
                    ),
                    0.0,
                ),
            ),
        )
        completion_probability = max(
            0.0,
            min(
                1.0,
                _finite(
                    row.get(
                        "interval_completion_probability",
                        row.get("completion_probability"),
                    ),
                    1.0 - failure_probability,
                ),
            ),
        )
        if (
            action.kind is ActionKind.FAIL
            and "interval_completion_probability" not in row
        ):
            failure_probability = 1.0
            completion_probability = 0.0
        config = self.mask_engine.config
        duration = _duration(row)
        vehicle_budget = (
            0.0
            if config is None
            else _finite(
                config.vehicle_power_budgets_w.get(observation.vehicle_id), 0.0
            )
        )
        rsu_budget = (
            0.0
            if config is None or not action.rsu_id
            else _finite(config.rsu_power_budgets_w.get(action.rsu_id), 0.0)
        )
        arrivals = max(
            0.0,
            _finite(
                row.get("interval_expected_arrivals", row.get("expected_arrivals")),
                0.0,
            ),
        )
        beta_timeout = 0.0 if config is None else config.long_term.timeout_rate_limit
        beta_failure = 0.0 if config is None else config.long_term.failure_rate_limit
        beta_coverage = (
            0.0 if config is None else config.long_term.coverage_rate_minimum
        )

        def increment(queue: float, delta: float) -> float:
            after = max(0.0, queue + delta)
            return 0.5 * (after * after - queue * queue)

        return (
            increment(vehicle_q, vehicle_energy - vehicle_budget * duration)
            + increment(rsu_q, rsu_energy - rsu_budget * duration)
            + increment(timeout_q, timeout_probability - beta_timeout * arrivals)
            + increment(failure_q, failure_probability - beta_failure * arrivals)
            + increment(coverage_q, beta_coverage * arrivals - completion_probability)
        )

    def score_action(
        self,
        action: Action,
        observation: Observation,
        *,
        outcome_override: Mapping[str, Any] | None = None,
    ) -> float:
        row = dict(action_estimate(action, observation, self.mask_engine.trace_support))
        if outcome_override:
            row.update(outcome_override)
        duration = _duration(
            row,
            fail_default=(
                self.mask_engine.config.controller.controller_overhead_s
                if self.mask_engine.config is not None
                else _EPS_DURATION_S
            ),
        )
        cost = _task_cost_from_row(action, observation, self.mask_engine, row)
        numerator = (
            self._physical_drift(action, observation, row)
            + self._virtual_drift(action, observation, row)
            + self.lyapunov_v * cost
        )
        return numerator / duration

    def _scores(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: MaskResult,
        state: SimulationState | None,
    ) -> Mapping[Action, float]:
        if self.scenario_source is not None:
            engine = ESLSMPCPolicy(
                self.mask_engine,
                self.repairer,
                horizon_events=2,
                scenarios=self.h1_scenarios,
                lyapunov_v=self.lyapunov_v,
                scenario_source=self.scenario_source,
                physical_queue_weight=self.physical_queue_weight,
                vehicle_resource_theta=self.vehicle_resource_theta,
                rsu_resource_theta=self.rsu_resource_theta,
            )
            exact_scores: dict[Action, float] = {}
            incomplete_reasons: set[str] = set()
            for action in mask.allowed:
                numerators: list[float] = []
                durations: list[float] = []
                for scenario_index in range(self.h1_scenarios):
                    numerator, duration, _, diagnostics = engine._one_rollout(
                        task,
                        observation,
                        action,
                        scenario_index,
                        include_diagnostics=True,
                        stop_before_next_macro=True,
                    )
                    if not diagnostics["complete_macro_recourse"]:
                        incomplete_reasons.add(
                            str(
                                diagnostics["incomplete_reason"]
                                or "UNKNOWN_INCOMPLETE_BRANCH"
                            )
                        )
                        numerator = 1e30 * max(_EPS_DURATION_S, duration)
                    numerators.append(numerator)
                    durations.append(duration)
                exact_scores[action] = sum(numerators) / max(
                    _EPS_DURATION_S, sum(durations)
                )
            self._last_h1_diagnostics = deep_freeze(
                {
                    "h1_semantics": "exact_isolated_branch_to_next_macro_event",
                    "scenario_count": self.h1_scenarios,
                    "complete_macro_recourse": not incomplete_reasons,
                    "incomplete_reasons": tuple(sorted(incomplete_reasons)),
                }
            )
            return MappingProxyType(exact_scores)
        self._last_h1_diagnostics = deep_freeze(
            {
                "h1_semantics": "frozen_aggregate_fallback_no_scenario_source",
                "scenario_count": 0,
                "complete_macro_recourse": False,
                "incomplete_reasons": ("SCENARIO_SOURCE_MISSING",),
            }
        )
        return MappingProxyType(
            {action: self.score_action(action, observation) for action in mask.allowed}
        )

    def _diagnostics(self) -> Mapping[str, Any]:
        return self._last_h1_diagnostics

    def _score_state_snapshot(self) -> Any:
        return self._last_h1_diagnostics

    def _restore_score_state(self, snapshot: Any) -> None:
        self._last_h1_diagnostics = snapshot


@dataclass(slots=True)
class _PredictionState:
    """Sanitized mutable branch state; contains no vehicle-local handles."""

    observation: dict[str, Any]
    environment: ScenarioEnvironment | None = None
    elapsed_s: float = 0.0
    cumulative_cost: float = 0.0
    joint_row_ids: list[str] = field(default_factory=list)
    terminal: bool = False
    macro_events: int = 1
    next_environment_index: int = 0
    battery_j: float = 0.0
    slack_s: float = 0.0
    vehicle_queues: dict[str, float] = field(default_factory=dict)
    vehicle_servers: dict[str, int] = field(default_factory=dict)
    rsus: dict[str, dict[str, Any]] = field(default_factory=dict)
    public_rsus: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_telemetry: dict[tuple[str, int], dict[str, Any]] = field(
        default_factory=dict
    )
    virtual_queues: dict[str, Any] = field(default_factory=dict)
    active_faults: set[tuple[str, str, str]] = field(default_factory=set)
    telemetry_age_s: dict[str, float] = field(default_factory=dict)
    active_profile_hash: str = ""
    active_protocol_version: str = ""
    active_local_model_hashes: dict[tuple[str, str], str] = field(default_factory=dict)
    vehicle_energy_j: float = 0.0
    rsu_energy_j: dict[str, float] = field(default_factory=dict)
    vehicle_physical_energy_j: dict[str, float] = field(default_factory=dict)
    rsu_physical_energy_j: dict[str, float] = field(default_factory=dict)
    focal_decisions: int = 0
    use_future_tasks: bool = False
    complete_macro_recourse: bool = False
    incomplete_reason: str | None = None
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    scheduler_physical_lyapunov: float | None = None
    scheduler_trace: list[dict[str, Any]] = field(default_factory=list)
    common_scenario_seed: int = 0


@dataclass(frozen=True, slots=True)
class _ScenarioOutcome:
    row_id: str
    duration_s: float
    values: Mapping[str, Any]
    formed_packet: bool = False
    artifact_key: str | None = None
    terminal: bool = True


@dataclass(slots=True)
class _BranchJob:
    job_id: str
    task_token: str
    owner_type: str
    owner_id: str
    resource: str
    remaining_work_s: float
    total_work_s: float
    total_energy_j: float
    absolute_deadline_s: float
    enqueue_seq: int
    completion_kind: str
    maintenance_event: Any | None = None


@dataclass(slots=True)
class _BranchResource:
    owner_type: str
    owner_id: str
    resource: str
    server_count: int
    waiting: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _BranchTransfer:
    transfer_id: str
    task_token: str
    vehicle_id: str
    rsu_id: str
    direction: TransferDirection
    remaining_bits: float
    paused_since_s: float | None = None


@dataclass(slots=True)
class _BranchTask:
    task_token: str
    vehicle_id: str
    arrival_s: float
    deadline_s: float
    future: Any
    base_observation: Observation
    state: str = "PENDING"
    selected_pipeline: str | None = None
    artifact_key: str | None = None
    encoded_size_bytes: int | None = None
    pending_stages: list[tuple[str, float, float, str]] = field(default_factory=list)
    edge_outcome: _ScenarioOutcome | None = None
    edge_pairing_token: str | None = None
    admitted_rsu: str | None = None
    reserved_vram_bytes: int = 0
    reserved_gpu_work_s: float = 0.0
    admission_vram_upper_bytes: int = 0
    admission_gpu_work_upper_s: float = 0.0
    reservation_tokens: dict[str, int] = field(default_factory=dict)
    reserved_memory_bytes: int = 0
    last_action: Action | None = None
    last_observation: Observation | None = None
    last_outcome: _ScenarioOutcome | None = None
    pending_action: Action | None = None
    pending_stage: ActionStage | None = None
    pending_sample_seed: int | None = None
    focal: bool = False
    fixed_continuation: bool = False
    continuation_path_kind: str | None = None
    continuation_prep_failed: bool = False
    continuation_stages: list[tuple[str, float, float, str]] = field(
        default_factory=list
    )
    continuation_action_memory_bytes: int = 0
    continuation_action_tokens: dict[str, int] = field(default_factory=dict)
    continuation_control_next: str | None = None
    latent_quality_bin: str | None = None


@dataclass(slots=True)
class _BranchScheduler:
    branch: _PredictionState
    focal_vehicle_id: str
    tasks: dict[str, _BranchTask]
    resources: dict[tuple[str, str, str], _BranchResource]
    jobs: dict[str, _BranchJob]
    transfers: dict[str, _BranchTransfer]
    events: list[tuple[float, int, int, str, str]]
    vehicle_observation_rows: dict[str, dict[str, Any]]
    vehicle_battery_j: dict[str, float]
    vehicle_memory_capacity: dict[str, int]
    vehicle_memory_reserved: dict[str, int]
    descriptor_capacity: dict[str, dict[str, int]]
    descriptor_reserved: dict[str, dict[str, int]]
    macro_decisions: int = 0
    sequence: int = 0
    event_trace: list[dict[str, Any]] = field(default_factory=list)
    last_decision_time_s: float | None = None
    pending_arrivals: int = 0
    pending_timeouts: int = 0
    pending_failures: int = 0
    pending_completed: int = 0
    maintenance_job_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    maintenance_active: dict[tuple[str, str], str] = field(default_factory=dict)
    maintenance_waiting: dict[tuple[str, str], list[str]] = field(default_factory=dict)


class ESLSMPCPolicy(SafeLyapunovPolicy):
    """Finite H>1 joint-scenario rollout that executes only its first action."""

    name = "esl_smpc"

    def __init__(
        self,
        mask_engine: HardMaskEngine,
        repairer: DeterministicRepair | None = None,
        *,
        horizon_events: int | None = None,
        scenarios: int | None = None,
        lyapunov_v: float | None = None,
        scenario_seed: int | None = None,
        scenario_source: Any = None,
        rollout_policy: str | None = None,
        physical_queue_weight: float | None = None,
        vehicle_resource_theta: Mapping[str, float] | None = None,
        rsu_resource_theta: Mapping[str, float] | None = None,
        scenario_certificate_bounds: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(
            mask_engine,
            repairer,
            lyapunov_v=lyapunov_v,
            physical_queue_weight=physical_queue_weight,
            vehicle_resource_theta=vehicle_resource_theta,
            rsu_resource_theta=rsu_resource_theta,
        )
        ctl = None if mask_engine.config is None else mask_engine.config.controller
        self.horizon_events = int(
            horizon_events
            if horizon_events is not None
            else ctl.horizon_events
            if ctl
            else 2
        )
        self.scenarios = int(
            scenarios if scenarios is not None else ctl.scenarios if ctl else 8
        )
        configured_seed = (
            None
            if mask_engine.config is None
            else int(mask_engine.config.seeds["scenario"])
        )
        self.scenario_seed = int(
            scenario_seed if scenario_seed is not None else configured_seed or 0
        )
        selected_source = (
            scenario_source
            if scenario_source is not None
            else mask_engine.trace_support
        )
        if isinstance(selected_source, TraceBundle):
            raise TypeError(
                "ESL-SMPC scenario_source must be an identity-free ScenarioLibrary, not TraceBundle"
            )
        self.scenario_source = selected_source
        self.rollout_policy = rollout_policy or (
            ctl.rollout_policy if ctl else "safe_greedy"
        )
        self.rollout_pipeline_id = _globally_safe_fixed_pipeline(mask_engine)
        compatible_models = sorted(
            model.model_id
            for model in mask_engine.profile.edge_models.values()
            if not model.supported_pipelines
            or self.rollout_pipeline_id in model.supported_pipelines
        )
        self.rollout_edge_model_id = (
            compatible_models[0]
            if compatible_models
            else min(mask_engine.profile.edge_models)
        )
        self.scenario_certificate_bounds = MappingProxyType(
            {
                str(name): float(value)
                for name, value in (scenario_certificate_bounds or {}).items()
            }
        )
        if self.horizon_events < 2:
            raise ValueError(
                "ESL-SMPC is the empirical H>1 controller; horizon_events must be >= 2"
            )
        if self.scenarios < 1:
            raise ValueError("scenarios must be >= 1")
        if self.rollout_policy not in {
            "all_local",
            "fixed_safe_lowest_link_cost",
            "fixed_safe_shortest_visible_queue",
            "safe_greedy",
        }:
            raise ValueError(
                "rollout_policy is not a supported READY-stage recourse selector"
            )
        self._last_diagnostics: Mapping[str, Any] = MappingProxyType({})

    def _diagnostics(self) -> Mapping[str, Any]:
        return self._last_diagnostics

    def _score_state_snapshot(self) -> Any:
        return self._last_h1_diagnostics, self._last_diagnostics

    def _restore_score_state(self, snapshot: Any) -> None:
        self._last_h1_diagnostics, self._last_diagnostics = snapshot

    def _seed_for(self, observation: Observation, scenario_index: int) -> int:
        """Common exogenous seed shared by every candidate first action."""

        material = (
            f"{self.scenario_seed}|{observation.task_id}|{observation.time_s:.12f}|"
            f"{scenario_index}"
        ).encode("utf-8")
        return int.from_bytes(
            hashlib.sha256(material).digest()[:8], "big", signed=False
        )

    @staticmethod
    def _action_seed(common_seed: int, action: Action) -> int:
        material = f"{common_seed}|{action.canonical_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    @staticmethod
    def _conditional_decision_seed(
        branch: _PredictionState,
        task_token: str,
        stage: ActionStage,
        action: Action,
        time_s: float,
    ) -> int:
        """Traversal-independent controller-scenario substream."""

        material = (
            f"{branch.common_scenario_seed}|conditional-outcome|{task_token}|"
            f"{stage.value}|{action.canonical_id}|{time_s:.12f}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

    @staticmethod
    def _context_matches(row: Any, observation: Observation) -> bool:
        return ESLSMPCPolicy._context_value_matches(row, observation.device_context)

    @staticmethod
    def _context_value_matches(row: Any, requested: str | None) -> bool:
        context = getattr(row, "context", None)
        if context is None or not requested:
            return True
        composite = "|".join(
            str(getattr(context, name, ""))
            for name in ("thermal_state", "power_mode", "memory_pressure")
        )
        values = {
            str(getattr(context, "thermal_state", "")),
            str(getattr(context, "power_mode", "")),
            str(getattr(context, "memory_pressure", "")),
            composite,
        }
        return requested in values or (
            requested == "nominal"
            and all(value in {"nominal", "normal"} for value in composite.split("|"))
        )

    @staticmethod
    def _scenario_artifact(row: Any) -> str | None:
        """Return only a scenario-local pairing token when one is available."""

        token = getattr(row, "artifact_token", None)
        if token is not None:
            return str(token)
        value = getattr(row, "artifact_key", None)
        return None if value is None else str(value)

    @staticmethod
    def _scenario_row_id(row: Any) -> str:
        value = getattr(row, "scenario_id", None)
        if value is None:
            value = getattr(row, "row_id", "joint-scenario-row")
        return str(value)

    @staticmethod
    def _edge_context_matches(
        row: Any, observation: Observation, rsu_id: str | None
    ) -> bool:
        rsu = _mapping(observation.rsus.get(rsu_id or ""))
        requested = rsu.get("device_context")
        return ESLSMPCPolicy._context_value_matches(
            row,
            None if requested is None else str(requested),
        )

    def _trace_candidates(self, action: Action, observation: Observation) -> list[Any]:
        source = self.scenario_source
        if source is None:
            return []
        bins = set(observation.conformal_quality_bins)
        if action.kind is ActionKind.PIPE:
            return [
                row
                for row in getattr(source, "anon_rows", ())
                if row.pipeline_id == action.pipeline_id
                and row.quality_bin in bins
                and row.device_type == observation.device_type
                and self._context_matches(row, observation)
            ]
        if action.kind is ActionKind.LOCAL:
            return [
                row
                for row in getattr(source, "local_rows", ())
                if row.model_id == action.local_model_id
                and row.quality_bin in bins
                and row.device_type == observation.device_type
                and self._context_matches(row, observation)
            ]
        if action.kind is ActionKind.EDGE:
            return [
                row
                for row in getattr(source, "edge_rows", ())
                if row.rsu_id == action.rsu_id
                and row.model_id == action.edge_model_id
                and row.pipeline_id == observation.selected_pipeline
                and row.quality_bin in bins
                and self._edge_context_matches(row, observation, action.rsu_id)
            ]
        return []

    @staticmethod
    def _scenario_rows_for(future: Any, source: Any, name: str) -> tuple[Any, ...]:
        """Return only immutable scenario rows belonging to this branch task."""

        rows = getattr(future, name, None)
        if rows is None:
            rows = getattr(source, name, ())
        return tuple(rows)

    def _actual_edge_rows(
        self,
        future: Any,
        action: Action,
        observation: Observation,
        pairing_token: str | None,
        *,
        quality_bin: str | None = None,
    ) -> tuple[Any, ...]:
        """Find service measurements paired with the one realized packet.

        A realized anonymous artifact has exactly one realized quality cell; it
        cannot simultaneously carry every conformal candidate label.  The
        all-candidate requirement is enforced separately when computing the
        conservative admission envelope.
        """

        if action.kind is not ActionKind.EDGE or not pairing_token:
            return ()
        return tuple(
            row
            for row in self._scenario_rows_for(
                future, self.scenario_source, "edge_rows"
            )
            if row.rsu_id == action.rsu_id
            and row.model_id == action.edge_model_id
            and row.pipeline_id == observation.selected_pipeline
            and self._scenario_artifact(row) == pairing_token
            and (quality_bin is None or str(row.quality_bin) == quality_bin)
            and self._edge_context_matches(row, observation, action.rsu_id)
        )

    def _certified_edge_admission_bounds(
        self,
        future: Any,
        action: Action,
        observation: Observation,
    ) -> tuple[int, float] | None:
        """Return a conservative paired envelope over every quality candidate.

        Each quality cell must contain a formed packet from the selected
        pipeline and at least one edge measurement of that exact packet under
        the live RSU context.  Missing cells are unsupported; values are never
        filled from another pipeline or an unpaired marginal average.
        """

        if action.kind is not ActionKind.EDGE or not observation.selected_pipeline:
            return None
        anon_rows = self._scenario_rows_for(future, self.scenario_source, "anon_rows")
        edge_rows = self._scenario_rows_for(future, self.scenario_source, "edge_rows")
        selected: list[Any] = []
        for quality_bin in sorted(set(observation.conformal_quality_bins)):
            artifacts = {
                self._scenario_artifact(row)
                for row in anon_rows
                if row.pipeline_id == observation.selected_pipeline
                and str(row.quality_bin) == quality_bin
                and row.device_type == observation.device_type
                and self._context_matches(row, observation)
                and bool(getattr(row, "formed_packet", False))
                and self._scenario_artifact(row)
            }
            cell = [
                row
                for row in edge_rows
                if row.rsu_id == action.rsu_id
                and row.model_id == action.edge_model_id
                and row.pipeline_id == observation.selected_pipeline
                and str(row.quality_bin) == quality_bin
                and self._scenario_artifact(row) in artifacts
                and self._edge_context_matches(row, observation, action.rsu_id)
            ]
            if not artifacts or not cell:
                return None
            selected.extend(cell)
        if not selected:
            return None
        return (
            max(int(row.vram_bytes) for row in selected),
            max(float(row.gpu_work_s) for row in selected),
        )

    def _select_surrogate_pairing_token(
        self,
        branch: _PredictionState,
        task: _BranchTask,
        observation: Observation,
    ) -> str | None:
        """Map an evaluation artifact to an identity-free scenario packet.

        The real artifact key remains in the safety observation.  This token is
        used only inside the controller's scenario library to replay a paired
        numerical edge outcome for the already-realized quality cell.
        """

        quality_bin = task.latent_quality_bin
        if not quality_bin or not observation.selected_pipeline:
            return None
        rows = [
            row
            for row in self._scenario_rows_for(
                task.future, self.scenario_source, "anon_rows"
            )
            if row.pipeline_id == observation.selected_pipeline
            and str(row.quality_bin) == quality_bin
            and row.device_type == observation.device_type
            and self._context_matches(row, observation)
            and bool(getattr(row, "formed_packet", False))
            and self._scenario_artifact(row)
        ]
        if not rows:
            return None
        rows.sort(key=self._scenario_row_id)
        material = (
            f"{branch.common_scenario_seed}|surrogate-artifact|{task.task_token}|"
            f"{observation.selected_pipeline}|{quality_bin}"
        ).encode("utf-8")
        index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(rows)
        return self._scenario_artifact(rows[index])

    @staticmethod
    def _deterministic_scenario_row(
        branch: _PredictionState,
        task_token: str,
        tag: str,
        rows: Sequence[Any],
    ) -> Any | None:
        """Select a conditional row without depending on container traversal."""

        ordered = sorted(rows, key=ESLSMPCPolicy._scenario_row_id)
        if not ordered:
            return None
        material = (f"{branch.common_scenario_seed}|{tag}|{task_token}").encode("utf-8")
        index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % len(
            ordered
        )
        return ordered[index]

    @staticmethod
    def _snapshot_action(value: Mapping[str, Any]) -> Action:
        stage = ActionStage(str(value.get("stage")))
        kind = ActionKind(str(value.get("kind")))
        return Action(
            stage,
            kind,
            local_model_id=value.get("local_model_id"),
            pipeline_id=value.get("pipeline_id"),
            rsu_id=value.get("rsu_id"),
            edge_model_id=value.get("edge_model_id"),
        )

    @staticmethod
    def _row_outcome(row: Any) -> _ScenarioOutcome:
        row_id = ESLSMPCPolicy._scenario_row_id(row)
        if hasattr(row, "attempts"):
            # The complete attempts tuple is consumed as one indivisible joint
            # row.  No field is independently re-sampled.
            duration = max(
                _EPS_DURATION_S, _finite(getattr(row, "total_work_s", None), 0.0)
            )
            energy = max(0.0, _finite(getattr(row, "total_energy_j", None), 0.0))
            formed = bool(getattr(row, "formed_packet", False))
            attempts = tuple(getattr(row, "attempts", ()))
            anon_work = sum(
                max(0.0, _finite(item.anon_work_s, 0.0)) for item in attempts
            )
            guard_work = sum(
                max(0.0, _finite(item.guard_work_s, 0.0)) for item in attempts
            )
            encode_work = sum(
                max(0.0, _finite(item.encode_work_s, 0.0)) for item in attempts
            )
            values = {
                "expected_duration_s": duration,
                "vehicle_work_s": {
                    "accelerator": anon_work,
                    "cpu": guard_work,
                    "encoder": encode_work,
                },
                "anon_work_s": anon_work,
                "guard_work_s": guard_work,
                "encode_work_s": encode_work,
                "expected_vehicle_energy_j": energy,
                "failure_probability": 0.0 if formed else 1.0,
                "completion_probability": 0.0,
                # PIPE has not selected or delivered an FER result.  Utility is
                # charged exactly once by the eventual local/edge recourse.
                "expected_fer_loss": 0.0,
                "joint_attempt_count": len(getattr(row, "attempts", ())),
                "encoded_size_bytes": int(getattr(row, "final_encoded_size_bytes", 0)),
                "vehicle_memory_upper_bytes": max(
                    (int(item.peak_memory_bytes) for item in attempts), default=0
                ),
                "pipeline_stage_sequence": tuple(
                    stage
                    for item in attempts
                    for stage in (
                        (
                            "accelerator",
                            max(0.0, _finite(item.anon_work_s, 0.0)),
                            max(0.0, _finite(item.anon_energy_j, 0.0)),
                            "PIPE_STAGE",
                        ),
                        *(
                            (
                                (
                                    "cpu",
                                    max(0.0, _finite(item.guard_work_s, 0.0)),
                                    max(0.0, _finite(item.guard_energy_j, 0.0)),
                                    "PIPE_STAGE",
                                ),
                            )
                            if item.guard_work_s is not None
                            else ()
                        ),
                        *(
                            (
                                (
                                    "encoder",
                                    max(0.0, _finite(item.encode_work_s, 0.0)),
                                    max(0.0, _finite(item.encode_energy_j, 0.0)),
                                    "PIPE_STAGE",
                                ),
                            )
                            if item.encode_work_s is not None
                            else ()
                        ),
                    )
                    if stage[1] > 0
                ),
                "quality_bin": str(getattr(row, "quality_bin", "")),
            }
            return _ScenarioOutcome(
                row_id,
                duration,
                deep_freeze(values),
                formed,
                ESLSMPCPolicy._scenario_artifact(row),
                False,
            )
        if hasattr(row, "service_work_s"):
            duration = max(_EPS_DURATION_S, _finite(row.service_work_s, 0.0))
            failed = bool(getattr(row, "failed", False))
            values = {
                "expected_duration_s": duration,
                "vehicle_work_s": duration,
                "expected_vehicle_energy_j": max(
                    0.0, _finite(getattr(row, "dynamic_energy_j", 0.0), 0.0)
                ),
                "failure_probability": 1.0 if failed else 0.0,
                "completion_probability": 0.0 if failed else 1.0,
                "expected_fer_loss": max(
                    0.0, _finite(getattr(row, "fer_loss", 0.0), 0.0)
                ),
                "vehicle_memory_upper_bytes": int(getattr(row, "memory_bytes", 0)),
                "quality_bin": str(getattr(row, "quality_bin", "")),
            }
            return _ScenarioOutcome(
                row_id, duration, deep_freeze(values), terminal=True
            )
        if hasattr(row, "gpu_work_s"):
            ingress_failed = bool(getattr(row, "ingress_failed", False))
            duration = max(
                _EPS_DURATION_S,
                _finite(getattr(row, "ingress_work_s", 0.0), 0.0)
                + (
                    0.0
                    if ingress_failed
                    else _finite(getattr(row, "gpu_work_s", 0.0), 0.0)
                ),
            )
            failed = ingress_failed or bool(getattr(row, "failed", False))
            values = {
                "expected_duration_s": duration,
                "rsu_ingress_work_s": max(
                    0.0, _finite(getattr(row, "ingress_work_s", 0.0), 0.0)
                ),
                "rsu_gpu_work_s": max(
                    0.0,
                    0.0
                    if ingress_failed
                    else _finite(getattr(row, "gpu_work_s", 0.0), 0.0),
                ),
                "expected_rsu_energy_j": max(
                    0.0,
                    _finite(getattr(row, "ingress_energy_j", 0.0), 0.0)
                    + (
                        0.0
                        if ingress_failed
                        else _finite(getattr(row, "gpu_energy_j", 0.0), 0.0)
                    ),
                ),
                "rsu_ingress_energy_j": max(
                    0.0, _finite(getattr(row, "ingress_energy_j", 0.0), 0.0)
                ),
                "rsu_gpu_energy_j": max(
                    0.0,
                    0.0
                    if ingress_failed
                    else _finite(getattr(row, "gpu_energy_j", 0.0), 0.0),
                ),
                "ingress_failure": ingress_failed,
                "failure_probability": 1.0 if failed else 0.0,
                "completion_probability": 0.0 if failed else 1.0,
                "expected_fer_loss": max(
                    0.0, _finite(getattr(row, "fer_loss", 0.0), 0.0)
                ),
                "vram_bytes": int(getattr(row, "vram_bytes", 0)),
                "result_size_bits": int(getattr(row, "result_size_bits", 0)),
                "quality_bin": str(getattr(row, "quality_bin", "")),
            }
            return _ScenarioOutcome(
                row_id,
                duration,
                deep_freeze(values),
                artifact_key=ESLSMPCPolicy._scenario_artifact(row),
                terminal=True,
            )
        if isinstance(row, Mapping):
            values = dict(row)
            duration = _duration(values)
            return _ScenarioOutcome(
                str(
                    values.get(
                        "joint_row_id", values.get("row_id", "joint-scenario-row")
                    )
                ),
                duration,
                deep_freeze(values),
                bool(values.get("formed_packet", False)),
                values.get("artifact_token", values.get("artifact_key")),
                bool(values.get("terminal", True)),
            )
        raise TypeError(
            "scenario source must return a complete joint trace row or mapping"
        )

    def _sample_outcome(
        self,
        action: Action,
        observation: Observation,
        branch: _PredictionState,
        rng: random.Random,
    ) -> _ScenarioOutcome:
        source = self.scenario_source
        if source is not None:
            for name in (
                "sample_joint_scenario",
                "sample_action_outcome",
                "sample_scenario_row",
            ):
                method = getattr(source, name, None)
                if callable(method):
                    row = method(
                        action=action,
                        observation=observation,
                        prediction_state=branch,
                        rng=rng,
                    )
                    value = getattr(row, "value", row)
                    supported = getattr(row, "supported", True)
                    if not supported or value is None:
                        break
                    return self._row_outcome(value)
        rows = self._trace_candidates(action, observation)
        if rows:
            row = self._sample_quality_weighted_rows(rows, observation, rng)
            return self._row_outcome(row)
        # Deterministic frozen summary is a degenerate one-row scenario, useful
        # for engineering smoke tests.  It is never labelled measured evidence.
        estimate = action_estimate(action, observation, self.mask_engine.trace_support)
        values = dict(estimate)
        if action.kind is ActionKind.FAIL:
            values.update(
                {
                    "expected_duration_s": _EPS_DURATION_S,
                    "failure_probability": 1.0,
                    "completion_probability": 0.0,
                }
            )
        if not values:
            raise RuntimeError(
                f"no joint scenario or frozen action summary for {action.canonical_id}"
            )
        values["scenario_data_kind"] = "deterministic_frozen_summary"
        values["terminal"] = action.kind is not ActionKind.PIPE
        return self._row_outcome(values)

    def _sample_quality_weighted_rows(
        self,
        rows: Sequence[Any],
        observation: Observation,
        rng: random.Random,
    ) -> Any:
        """Sample calibrated quality first, then one paired joint row."""

        if not rows:
            raise LookupError("no paired scenario rows")
        probabilities = {
            str(name): max(0.0, float(probability))
            for name, probability in observation.quality_probabilities
            if str(name) in observation.conformal_quality_bins
        }
        supported_bins = sorted({str(row.quality_bin) for row in rows})
        weights = [probabilities.get(name, 0.0) for name in supported_bins]
        if sum(weights) <= 0:
            weights = [1.0] * len(supported_bins)
        draw = rng.random() * sum(weights)
        chosen_bin = supported_bins[-1]
        cumulative = 0.0
        for name, weight in zip(supported_bins, weights, strict=True):
            cumulative += weight
            if draw <= cumulative:
                chosen_bin = name
                break
        cell = [row for row in rows if str(row.quality_bin) == chosen_bin]
        sampler = getattr(self.scenario_source, "sample_rows", None)
        return (
            sampler(cell, rng) if callable(sampler) else cell[rng.randrange(len(cell))]
        )

    @staticmethod
    def _task_latent_quality(
        branch: _PredictionState,
        task_token: str,
        observation: Observation,
    ) -> str | None:
        bins = tuple(sorted(set(observation.conformal_quality_bins)))
        if not bins:
            return None
        probability_map = {
            str(name): max(0.0, float(value))
            for name, value in observation.quality_probabilities
            if str(name) in bins
        }
        weights = [probability_map.get(name, 0.0) for name in bins]
        if sum(weights) <= 0:
            weights = [1.0] * len(bins)
        material = (
            f"{branch.common_scenario_seed}|latent-quality|{task_token}"
        ).encode("utf-8")
        draw_rng = random.Random(
            int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        )
        draw = draw_rng.random() * sum(weights)
        cumulative = 0.0
        for name, weight in zip(bins, weights, strict=True):
            cumulative += weight
            if draw <= cumulative:
                return name
        return bins[-1]

    @staticmethod
    def _latent_sampling_observation(
        task: _BranchTask, observation: Observation
    ) -> Observation:
        quality_bin = task.latent_quality_bin
        if quality_bin is None:
            return observation
        return replace(
            observation,
            conformal_quality_bins=(quality_bin,),
            quality_probabilities=((quality_bin, 1.0),),
        )

    def _new_branch(
        self, observation: Observation, environment: ScenarioEnvironment | None
    ) -> _PredictionState:
        data = thaw_json(observation.to_dict())
        resources = _mapping(observation.vehicle.get("resources"))
        private_rsus = {
            str(rsu_id): thaw_json(row) for rsu_id, row in observation.rsus.items()
        }
        if environment is not None:
            for anchor in getattr(environment, "rsu_anchors", ()):
                rsu_id = str(getattr(anchor, "rsu_id"))
                row = private_rsus.setdefault(rsu_id, {})
                resource_rows = getattr(anchor, "resources", {})
                ingress = _mapping(resource_rows.get("ingress"))
                gpu = _mapping(resource_rows.get("gpu"))
                row.update(
                    {
                        "failed": bool(getattr(anchor, "failed", False)),
                        "permanent_failure": bool(
                            getattr(anchor, "permanent_failure", False)
                        ),
                        "descriptors": int(getattr(anchor, "descriptors_reserved", 0)),
                        "descriptor_capacity": int(
                            getattr(anchor, "descriptor_capacity", 0)
                        ),
                        "vram_bytes": int(getattr(anchor, "vram_reserved_bytes", 0)),
                        "vram_capacity_bytes": int(
                            getattr(anchor, "vram_capacity_bytes", 0)
                        ),
                        "reserved_work_gpu_s": float(
                            getattr(anchor, "workload_reserved_gpu_s", 0.0)
                        ),
                        "workload_capacity_gpu_s": float(
                            getattr(anchor, "workload_capacity_gpu_s", 0.0)
                        ),
                        "cached_models": dict(getattr(anchor, "cached_models", {})),
                        "ingress_servers": max(1, int(ingress.get("server_count", 1))),
                        "ingress_waiting": int(ingress.get("waiting_count", 0)),
                        "ingress_running": int(ingress.get("running_count", 0)),
                        "ingress_residual_work_s": max(
                            0.0, _finite(ingress.get("residual_work_s"), 0.0)
                        ),
                        "ingress_remaining_dynamic_energy_j": max(
                            0.0,
                            _finite(ingress.get("remaining_dynamic_energy_j"), 0.0),
                        ),
                        "gpu_servers": max(1, int(gpu.get("server_count", 1))),
                        "gpu_waiting": int(gpu.get("waiting_count", 0)),
                        "gpu_running": int(gpu.get("running_count", 0)),
                        "gpu_residual_work_s": max(
                            0.0, _finite(gpu.get("residual_work_s"), 0.0)
                        ),
                        "gpu_remaining_dynamic_energy_j": max(
                            0.0,
                            _finite(gpu.get("remaining_dynamic_energy_j"), 0.0),
                        ),
                    }
                )
        future_tasks = (
            () if environment is None else getattr(environment, "future_tasks", ())
        )
        # A mixed-vehicle window may still have a complete same-vehicle prefix
        # long enough to fill H.  Validate every task lazily before executing
        # it; never reject a usable prefix because a later, unused task would
        # require the not-yet-supported concurrent multi-vehicle scheduler.
        first_future = future_tasks[0] if future_tasks else None
        complete_future = bool(
            first_future is not None
            and bool(getattr(first_future, "complete_support", False))
            and getattr(first_future, "vehicle_id", None) == observation.vehicle_id
        )
        branch = _PredictionState(
            observation=data,
            environment=environment,
            battery_j=max(0.0, _finite(observation.vehicle.get("battery_j"), 0.0)),
            slack_s=max(0.0, observation.slack_s),
            vehicle_queues={
                str(name): max(0.0, _finite(_mapping(row).get("residual_work_s"), 0.0))
                for name, row in resources.items()
            },
            vehicle_servers={
                str(name): max(1, int(_mapping(row).get("server_count", 1)))
                for name, row in resources.items()
            },
            rsus=private_rsus,
            public_rsus={
                str(rsu_id): thaw_json(row) for rsu_id, row in observation.rsus.items()
            },
            virtual_queues=thaw_json(observation.virtual_queues),
            telemetry_age_s={
                str(rsu_id): max(0.0, _finite(_mapping(row).get("snapshot_age_s"), 0.0))
                for rsu_id, row in observation.rsus.items()
            },
            rsu_energy_j={str(rsu_id): 0.0 for rsu_id in private_rsus},
            vehicle_physical_energy_j={
                str(vehicle_id): 0.0
                for vehicle_id in (
                    self.mask_engine.config.vehicle_branch_parameters
                    if self.mask_engine.config is not None
                    else (observation.vehicle_id,)
                )
            },
            rsu_physical_energy_j={str(rsu_id): 0.0 for rsu_id in private_rsus},
            active_profile_hash=str(
                observation.versions.get(
                    "profile_hash", self.mask_engine.profile.profile_hash
                )
            ),
            active_protocol_version=str(
                observation.versions.get(
                    "protocol_version", self.mask_engine.profile.protocol_version
                )
            ),
            active_local_model_hashes={
                (vehicle_id, model_id): model.model_hash
                for vehicle_id in (
                    self.mask_engine.config.vehicle_branch_parameters
                    if self.mask_engine.config is not None
                    else {observation.vehicle_id: {}}
                )
                for model_id, model in self.mask_engine.profile.local_models.items()
            },
            # Explicit future arrivals and legacy aggregate background loads
            # are mutually exclusive representations of the same frozen
            # arrivals.  Select the event-heap representation before applying
            # offset-zero environment effects, including mixed-vehicle windows.
            use_future_tasks=bool(future_tasks),
            complete_macro_recourse=complete_future,
            incomplete_reason=(
                None if complete_future else "SCENARIO_FUTURE_TASK_SUPPORT_INCOMPLETE"
            ),
        )
        if environment is not None:
            for anchor in getattr(environment, "vehicle_anchors", ()):
                if bool(getattr(anchor, "failed", False)) or bool(
                    getattr(anchor, "battery_depleted", False)
                ):
                    branch.active_faults.add(
                        ("vehicle", str(getattr(anchor, "vehicle_id")), "all")
                    )
            for anchor in getattr(environment, "rsu_anchors", ()):
                if bool(getattr(anchor, "failed", False)):
                    branch.active_faults.add(
                        ("rsu", str(getattr(anchor, "rsu_id")), "all")
                    )
        for model_id, row in _mapping(observation.versions.get("local_models")).items():
            value = _mapping(row).get("model_hash") if isinstance(row, Mapping) else row
            if value is not None:
                branch.active_local_model_hashes[
                    (observation.vehicle_id, str(model_id))
                ] = str(value)
        if environment is not None:
            self._apply_environment_at(branch, 0.0, observation.vehicle_id)
            offsets = environment.macro_event_offsets
            while (
                branch.next_environment_index < len(offsets)
                and offsets[branch.next_environment_index] <= _EPS_DURATION_S
            ):
                branch.next_environment_index += 1
        return branch

    @staticmethod
    def _faulted(
        branch: _PredictionState, owner_type: str, owner_id: str, resource: str
    ) -> bool:
        return any(
            key in branch.active_faults
            for key in (
                (owner_type, owner_id, resource),
                (owner_type, owner_id, "all"),
            )
        )

    @staticmethod
    def _thermal_multiplier(
        branch: _PredictionState, owner_type: str, owner_id: str, resource: str
    ) -> float:
        row = ESLSMPCPolicy._branch_thermal_segment(
            branch, owner_type, owner_id, resource
        )
        return 1.0 if row is None else max(0.0, row.service_rate_multiplier)

    @staticmethod
    def _branch_thermal_segment(
        branch: _PredictionState,
        owner_type: str,
        owner_id: str,
        resource: str,
    ) -> Any | None:
        """Select the causal thermal row with production-equivalent precedence."""

        environment = branch.environment
        if environment is None:
            return None
        candidates = [
            row
            for row in environment.thermal
            if (
                row.owner_type == owner_type
                and row.owner_id == owner_id
                and row.resource in {resource, "all"}
                and row.start_offset_s - 1e-12
                <= branch.elapsed_s
                < row.end_offset_s - 1e-12
            )
        ]
        exact = [row for row in candidates if row.resource == resource]
        rows = exact or candidates
        if not rows:
            return None
        # The final fields give a canonical, input-order-independent tie break
        # for malformed/overlapping rows that share the same latest start.
        return max(
            rows,
            key=lambda row: (
                row.start_offset_s,
                row.end_offset_s,
                str(row.state),
                row.service_rate_multiplier,
                row.dynamic_power_multiplier,
            ),
        )

    def _drain_queues(
        self, branch: _PredictionState, dt_s: float, vehicle_id: str
    ) -> None:
        for resource, queued in tuple(branch.vehicle_queues.items()):
            if self._faulted(branch, "vehicle", vehicle_id, resource):
                continue
            rate = branch.vehicle_servers.get(resource, 1) * self._thermal_multiplier(
                branch, "vehicle", vehicle_id, resource
            )
            branch.vehicle_queues[resource] = max(0.0, queued - rate * dt_s)
        for rsu_id, row in branch.rsus.items():
            for resource, field_name, servers_field in (
                ("ingress", "ingress_residual_work_s", None),
                ("gpu", "gpu_residual_work_s", "gpu_servers"),
            ):
                if self._faulted(branch, "rsu", rsu_id, resource):
                    continue
                servers = (
                    1
                    if servers_field is None
                    else max(1, int(row.get(servers_field, 1)))
                )
                rate = servers * self._thermal_multiplier(
                    branch, "rsu", rsu_id, resource
                )
                served = min(max(0.0, _finite(row.get(field_name), 0.0)), rate * dt_s)
                row[field_name] = max(0.0, _finite(row.get(field_name), 0.0) - served)

    def _sample_public_rsu(
        self,
        branch: _PredictionState,
        rsu_id: str,
        *,
        work_quantum_s: float,
    ) -> dict[str, Any] | None:
        """Freeze the branch live RSU state without exposing it immediately."""

        live = branch.rsus.get(rsu_id)
        if live is None:
            return None
        snapshot = {
            str(key): thaw_json(value)
            for key, value in live.items()
            if key not in {"snapshot_time_s", "snapshot_age_s"}
        }
        snapshot["failed"] = self._faulted(branch, "rsu", rsu_id, "all")
        environment = branch.environment
        thermal_rows = (
            []
            if environment is None
            else [
                row
                for row in environment.thermal
                if row.owner_type == "rsu"
                and row.owner_id == rsu_id
                and row.resource == "all"
                and row.start_offset_s - 1e-12
                <= branch.elapsed_s
                < row.end_offset_s - 1e-12
            ]
        )
        thermal_state = (
            "nominal"
            if not thermal_rows
            else max(thermal_rows, key=lambda row: row.start_offset_s).state
        )
        # Match DiscreteEventSimulator._rsu_public_snapshot: device context is
        # sampled at the telemetry sampling instant, not copied from the
        # decision-epoch snapshot.
        snapshot["device_context"] = f"{thermal_state}|nominal|normal"
        if work_quantum_s > 0:
            for field_name in (
                "reserved_work_gpu_s",
                "ingress_residual_work_s",
                "gpu_residual_work_s",
            ):
                value = max(0.0, _finite(snapshot.get(field_name), 0.0))
                snapshot[field_name] = round(value / work_quantum_s) * work_quantum_s
        return snapshot

    def _apply_version_event(
        self,
        branch: _PredictionState,
        event: Any,
        *,
        defer_rsu_maintenance: bool = False,
    ) -> None:
        if event.event_type in {"MODEL_VERSION", "MODEL_CACHE"}:
            if event.target_type == "rsu" and event.target_id in branch.rsus:
                if defer_rsu_maintenance:
                    return
                model_id = event.model_id or ""
                if model_id:
                    cache = dict(
                        _mapping(branch.rsus[event.target_id].get("cached_models"))
                    )
                    if event.remove:
                        cache.pop(model_id, None)
                    elif event.new_version:
                        cache[model_id] = event.new_version
                    branch.rsus[event.target_id]["cached_models"] = cache
            elif event.target_type == "vehicle" and event.model_id:
                branch.active_local_model_hashes[(event.target_id, event.model_id)] = (
                    event.new_version or "expired"
                )
        elif event.event_type == "PROFILE_VERSION":
            branch.active_profile_hash = event.new_version or "expired"
        elif event.event_type == "PROTOCOL_VERSION":
            branch.active_protocol_version = event.new_version or "expired"

    def _apply_environment_at(
        self,
        branch: _PredictionState,
        offset_s: float,
        vehicle_id: str,
        *,
        defer_rsu_maintenance: bool = False,
    ) -> None:
        environment = branch.environment
        if environment is None:
            return
        for event in environment.faults:
            if not math.isclose(event.offset_s, offset_s, abs_tol=1e-12):
                continue
            # ScenarioFaultEvent represents DEVICE_FAULT, matching the
            # production simulator's device-wide runtime.failed semantics.
            # ``resource`` is retained as source metadata but must not turn a
            # device fault into a narrower branch-only resource failure.
            key = (event.target_type, event.target_id, "all")
            if event.event_type.endswith(("RECOVER", "END")):
                branch.active_faults.discard(key)
            else:
                branch.active_faults.add(key)
        for event in environment.versions:
            if math.isclose(event.offset_s, offset_s, abs_tol=1e-12):
                self._apply_version_event(
                    branch,
                    event,
                    defer_rsu_maintenance=defer_rsu_maintenance,
                )
        matched_loads = 0
        for load in environment.background_loads:
            if not math.isclose(load.offset_s, offset_s, abs_tol=1e-12):
                continue
            if branch.use_future_tasks:
                # Complete future tasks are scheduled explicitly below.  The
                # legacy aggregate load is the fallback representation of the
                # same arrival and must not be charged twice.
                continue
            matched_loads += 1
            if load.vehicle_id == vehicle_id:
                branch.vehicle_queues[load.vehicle_resource] = (
                    branch.vehicle_queues.get(load.vehicle_resource, 0.0)
                    + load.vehicle_work_s
                )
                self._spend_vehicle(branch, load.vehicle_energy_j, vehicle_id)
            if load.rsu_id and load.rsu_id in branch.rsus:
                row = branch.rsus[load.rsu_id]
                row["ingress_residual_work_s"] = (
                    max(0.0, _finite(row.get("ingress_residual_work_s"), 0.0))
                    + load.ingress_work_s
                )
                row["gpu_residual_work_s"] = (
                    max(0.0, _finite(row.get("gpu_residual_work_s"), 0.0))
                    + load.gpu_work_s
                )
                row["reserved_work_gpu_s"] = (
                    max(0.0, _finite(row.get("reserved_work_gpu_s"), 0.0))
                    + load.gpu_work_s
                )
                self._spend_rsu(branch, load.rsu_id, load.rsu_energy_j)
        # Offset-zero load is the scenario anchor represented by the focal
        # task whose arrival has already updated the real queue bank.  Later
        # sanitized load impulses each represent a future arrival; apply the
        # same arrival-side virtual-queue increments as VirtualQueueBank.
        if matched_loads and offset_s > _EPS_DURATION_S:
            config = self.mask_engine.config
            if config is not None:
                branch.virtual_queues["timeout"] = max(
                    0.0,
                    _finite(branch.virtual_queues.get("timeout"), 0.0)
                    - config.long_term.timeout_rate_limit * matched_loads,
                )
                branch.virtual_queues["failure"] = max(
                    0.0,
                    _finite(branch.virtual_queues.get("failure"), 0.0)
                    - config.long_term.failure_rate_limit * matched_loads,
                )
                branch.virtual_queues["coverage"] = max(
                    0.0,
                    _finite(branch.virtual_queues.get("coverage"), 0.0)
                    + config.long_term.coverage_rate_minimum * matched_loads,
                )
        # Sampling freezes branch-local live state.  Delivery is a distinct
        # causal event, so a policy never reads live admission/cache/fault
        # state through a nominally public snapshot.
        for telemetry in environment.telemetry:
            if not math.isclose(telemetry.offset_s, offset_s, abs_tol=1e-12):
                continue
            if telemetry.dropped:
                continue
            snapshot = self._sample_public_rsu(
                branch,
                telemetry.rsu_id,
                work_quantum_s=max(0.0, telemetry.work_quantum_s),
            )
            if snapshot is not None:
                branch.pending_telemetry[
                    (telemetry.rsu_id, telemetry.sample_sequence)
                ] = snapshot
        for telemetry in environment.telemetry:
            delivery_offset = (
                telemetry.offset_s
                if telemetry.delivery_offset_s is None and not telemetry.dropped
                else telemetry.delivery_offset_s
            )
            if delivery_offset is None or not math.isclose(
                delivery_offset, offset_s, abs_tol=1e-12
            ):
                continue
            snapshot = branch.pending_telemetry.pop(
                (telemetry.rsu_id, telemetry.sample_sequence), None
            )
            if snapshot is None:
                continue
            branch.public_rsus[telemetry.rsu_id] = snapshot
            branch.public_rsus[telemetry.rsu_id]["snapshot_time_s"] = telemetry.offset_s
            branch.telemetry_age_s[telemetry.rsu_id] = max(
                0.0, delivery_offset - telemetry.offset_s
            )

    def _advance_branch(
        self, branch: _PredictionState, target_s: float, vehicle_id: str
    ) -> None:
        target_s = max(branch.elapsed_s, target_s)
        environment = branch.environment
        offsets = () if environment is None else environment.macro_event_offsets
        while branch.elapsed_s < target_s - 1e-12:
            next_offset = (
                offsets[branch.next_environment_index]
                if branch.next_environment_index < len(offsets)
                else math.inf
            )
            boundary = min(target_s, next_offset)
            dt_s = max(0.0, boundary - branch.elapsed_s)
            self._drain_queues(branch, dt_s, vehicle_id)
            config = self.mask_engine.config
            if config is not None and dt_s > 0:
                vehicle_power = branch.virtual_queues.setdefault("vehicle_power", {})
                if isinstance(vehicle_power, dict):
                    for owner_id, budget_w in config.vehicle_power_budgets_w.items():
                        vehicle_power[owner_id] = max(
                            0.0,
                            _finite(vehicle_power.get(owner_id), 0.0) - budget_w * dt_s,
                        )
                rsu_power = branch.virtual_queues.setdefault("rsu_power", {})
                if isinstance(rsu_power, dict):
                    for owner_id, budget_w in config.rsu_power_budgets_w.items():
                        rsu_power[owner_id] = max(
                            0.0,
                            _finite(rsu_power.get(owner_id), 0.0) - budget_w * dt_s,
                        )
            for rsu_id in branch.telemetry_age_s:
                branch.telemetry_age_s[rsu_id] += dt_s
            branch.elapsed_s = boundary
            branch.slack_s = max(0.0, branch.slack_s - dt_s)
            if next_offset <= target_s + 1e-12:
                self._apply_environment_at(branch, next_offset, vehicle_id)
                branch.next_environment_index += 1
                branch.macro_events += 1

    def _serve_work(
        self,
        branch: _PredictionState,
        *,
        owner_type: str,
        owner_id: str,
        resource: str,
        work_s: float,
        vehicle_id: str,
    ) -> bool:
        if work_s <= 0:
            return True
        if owner_type == "vehicle":
            queue = branch.vehicle_queues
            field = resource
            servers = branch.vehicle_servers.get(resource, 1)
        else:
            row = branch.rsus.get(owner_id)
            if row is None:
                return False
            field = (
                "ingress_residual_work_s"
                if resource == "ingress"
                else "gpu_residual_work_s"
            )
            queue = row
            servers = (
                1 if resource == "ingress" else max(1, int(row.get("gpu_servers", 1)))
            )
        ahead = max(0.0, _finite(queue.get(field), 0.0))
        queue[field] = ahead + work_s
        required = ahead + work_s
        environment = branch.environment
        while required > 1e-12 and branch.slack_s > 1e-12:
            if self._faulted(branch, owner_type, owner_id, resource):
                return False
            rate = servers * self._thermal_multiplier(
                branch, owner_type, owner_id, resource
            )
            offsets = () if environment is None else environment.macro_event_offsets
            next_offset = (
                offsets[branch.next_environment_index]
                if branch.next_environment_index < len(offsets)
                else math.inf
            )
            if rate <= 0:
                if not math.isfinite(next_offset):
                    return False
                self._advance_branch(branch, next_offset, vehicle_id)
                continue
            dt_complete = required / rate
            dt_boundary = next_offset - branch.elapsed_s
            dt = min(dt_complete, dt_boundary, branch.slack_s)
            if dt <= 1e-12:
                self._advance_branch(branch, next_offset, vehicle_id)
                continue
            self._advance_branch(branch, branch.elapsed_s + dt, vehicle_id)
            required = max(0.0, required - rate * dt)
        return required <= 1e-9

    def _wireless_at(
        self,
        branch: _PredictionState,
        observation: Observation,
        rsu_id: str,
        direction: TransferDirection,
    ) -> tuple[float, float, float, str, float]:
        environment = branch.environment
        if environment is not None:
            exact = [
                row
                for row in environment.wireless
                if row.vehicle_id == observation.vehicle_id
                and row.rsu_id == rsu_id
                and row.direction is direction
                and row.start_offset_s - 1e-12
                <= branch.elapsed_s
                < row.end_offset_s - 1e-12
            ]
            if exact:
                row = min(exact, key=lambda item: (item.end_offset_s, item.vehicle_id))
                return (
                    row.goodput_bps,
                    row.transmitter_power_w,
                    row.receiver_power_w,
                    row.link_state,
                    row.end_offset_s,
                )
            # A scenario row belonging to another vehicle is not paired
            # support for this transfer.  Missing exact joint support is a
            # conservative no-service condition, never a cross-vehicle
            # interpolation.
            return (0.0, 0.0, 0.0, "missing", branch.elapsed_s + branch.slack_s)
        link = _mapping(observation.links.get(rsu_id))
        prefix = "ul" if direction is TransferDirection.UL else "dl"
        return (
            max(0.0, _finite(link.get(f"{prefix}_goodput_bps"), 0.0)),
            0.0,
            0.0,
            str(link.get(f"{prefix}_link_state", "missing")),
            branch.elapsed_s + branch.slack_s,
        )

    def _transfer_bits(
        self,
        branch: _PredictionState,
        observation: Observation,
        rsu_id: str,
        direction: TransferDirection,
        bits: float,
    ) -> bool:
        remaining = max(0.0, bits)
        while remaining > 1e-6 and branch.slack_s > 1e-12:
            if self._faulted(branch, "rsu", rsu_id, "all"):
                return False
            rate, tx_power, rx_power, state, end_s = self._wireless_at(
                branch, observation, rsu_id, direction
            )
            if state in {"permanent_loss", "handover", "missing"}:
                return False
            if end_s <= branch.elapsed_s + 1e-12:
                return False
            available = min(end_s - branch.elapsed_s, branch.slack_s)
            if rate <= 0 or state != "connected":
                self._advance_branch(
                    branch, branch.elapsed_s + available, observation.vehicle_id
                )
                continue
            dt = min(available, remaining / rate)
            self._advance_branch(branch, branch.elapsed_s + dt, observation.vehicle_id)
            delivered = rate * dt
            remaining = max(0.0, remaining - delivered)
            if direction is TransferDirection.UL:
                self._spend_vehicle(branch, tx_power * dt, observation.vehicle_id)
                self._spend_rsu(branch, rsu_id, rx_power * dt)
            else:
                self._spend_vehicle(branch, rx_power * dt, observation.vehicle_id)
                self._spend_rsu(branch, rsu_id, tx_power * dt)
        return remaining <= 1e-6 and branch.battery_j >= -1e-12

    @staticmethod
    def _spend_vehicle(
        branch: _PredictionState, energy_j: float, vehicle_id: str
    ) -> None:
        energy = max(0.0, energy_j)
        branch.battery_j -= energy
        branch.vehicle_energy_j += energy
        branch.vehicle_physical_energy_j[vehicle_id] = (
            branch.vehicle_physical_energy_j.get(vehicle_id, 0.0) + energy
        )
        queues = branch.virtual_queues.setdefault("vehicle_power", {})
        if isinstance(queues, dict):
            queues[vehicle_id] = max(0.0, _finite(queues.get(vehicle_id), 0.0) + energy)

    @staticmethod
    def _spend_rsu(branch: _PredictionState, rsu_id: str, energy_j: float) -> None:
        energy = max(0.0, energy_j)
        branch.rsu_energy_j[rsu_id] = branch.rsu_energy_j.get(rsu_id, 0.0) + energy
        branch.rsu_physical_energy_j[rsu_id] = (
            branch.rsu_physical_energy_j.get(rsu_id, 0.0) + energy
        )
        queues = branch.virtual_queues.setdefault("rsu_power", {})
        if isinstance(queues, dict):
            queues[rsu_id] = max(0.0, _finite(queues.get(rsu_id), 0.0) + energy)

    def _lyapunov(self, branch: _PredictionState) -> float:
        if branch.scheduler_physical_lyapunov is not None:
            physical = max(0.0, branch.scheduler_physical_lyapunov)
        else:
            physical = sum(
                self._theta("vehicle", resource) * max(0.0, work) ** 2
                for resource, work in branch.vehicle_queues.items()
            )
            for row in branch.rsus.values():
                physical += (
                    self._theta("rsu", "ingress")
                    * max(0.0, _finite(row.get("ingress_residual_work_s"), 0.0)) ** 2
                )
                physical += (
                    self._theta("rsu", "gpu")
                    * max(0.0, _finite(row.get("gpu_residual_work_s"), 0.0)) ** 2
                )
        virtual_values: list[float] = []
        for key, value in branch.virtual_queues.items():
            if isinstance(value, Mapping):
                virtual_values.extend(
                    max(0.0, _finite(item, 0.0)) for item in value.values()
                )
            else:
                virtual_values.append(max(0.0, _finite(value, 0.0)))
        return 0.5 * (physical + sum(value * value for value in virtual_values))

    def _execute_prediction_action(
        self,
        action: Action,
        outcome: _ScenarioOutcome,
        observation: Observation,
        branch: _PredictionState,
    ) -> _ScenarioOutcome:
        started = branch.elapsed_s
        branch.focal_decisions += 1
        branch.decision_trace.append(
            {
                "decision_index": branch.focal_decisions,
                "time_s": branch.elapsed_s,
                "task_token": observation.task_id,
                "stage": observation.stage.value,
                "action": action.canonical_id,
            }
        )
        vehicle_energy_before = branch.vehicle_energy_j
        rsu_energy_before = dict(branch.rsu_energy_j)
        values = dict(outcome.values)
        success = True
        terminal = action.kind is not ActionKind.PIPE
        formed = outcome.formed_packet

        if action.kind is ActionKind.FAIL:
            fail_duration = (
                _EPS_DURATION_S
                if self.mask_engine.config is None
                else max(
                    _EPS_DURATION_S,
                    self.mask_engine.config.controller.controller_overhead_s,
                )
            )
            self._advance_branch(
                branch,
                branch.elapsed_s + fail_duration,
                observation.vehicle_id,
            )
            success = False
        elif action.kind is ActionKind.PIPE:
            for resource, key in (
                ("accelerator", "anon_work_s"),
                ("cpu", "guard_work_s"),
                ("encoder", "encode_work_s"),
            ):
                success = success and self._serve_work(
                    branch,
                    owner_type="vehicle",
                    owner_id=observation.vehicle_id,
                    resource=resource,
                    work_s=max(0.0, _finite(values.get(key), 0.0)),
                    vehicle_id=observation.vehicle_id,
                )
                if not success:
                    break
            energy = max(0.0, _finite(values.get("expected_vehicle_energy_j"), 0.0))
            self._spend_vehicle(branch, energy, observation.vehicle_id)
            success = success and formed and branch.battery_j >= -1e-12
            # PIPE is a first-stage transaction, not a task terminal result.
            # A failed transaction is resolved by frozen READY recourse.
            terminal = False
            values["expected_fer_loss"] = 0.0
            values["completion_probability"] = 0.0
        elif action.kind is ActionKind.LOCAL:
            success = self._serve_work(
                branch,
                owner_type="vehicle",
                owner_id=observation.vehicle_id,
                resource="accelerator",
                work_s=max(0.0, _finite(values.get("vehicle_work_s"), 0.0)),
                vehicle_id=observation.vehicle_id,
            )
            self._spend_vehicle(
                branch,
                max(0.0, _finite(values.get("expected_vehicle_energy_j"), 0.0)),
                observation.vehicle_id,
            )
            success = (
                success
                and _finite(values.get("failure_probability"), 0.0) < 0.5
                and branch.battery_j >= -1e-12
            )
        elif action.kind is ActionKind.EDGE:
            rsu_id = action.rsu_id or ""
            row = branch.rsus.get(rsu_id)
            model = self.mask_engine.profile.edge_models.get(action.edge_model_id or "")
            pipeline_id = observation.selected_pipeline
            pipeline = self.mask_engine.profile.pipelines.get(pipeline_id or "")
            active_protocol = str(
                observation.versions.get(
                    "protocol_version", self.mask_engine.profile.protocol_version
                )
            )
            active_models = _mapping(observation.versions.get("edge_models"))
            active_model = _mapping(active_models.get(action.edge_model_id or ""))
            cached_models = _mapping({} if row is None else row.get("cached_models"))
            encoded = _mapping(observation.encoded_evidence)
            message_valid = (
                encoded.get("message_source_type") == "EncodedAnon"
                and encoded.get("artifact_token") == observation.artifact_token
                and encoded.get("pipeline_id") == pipeline_id
                and encoded.get("profile_hash") == self.mask_engine.profile.profile_hash
            )
            version_valid = bool(
                model is not None
                and pipeline is not None
                and active_protocol == self.mask_engine.profile.protocol_version
                and model.protocol_version == active_protocol
                and pipeline.protocol_version == active_protocol
                and rsu_id in model.supported_rsus
                and (
                    not model.supported_pipelines
                    or pipeline_id in model.supported_pipelines
                )
                and cached_models.get(model.model_id) == model.model_hash
                and (
                    not active_model
                    or (
                        active_model.get("model_hash") == model.model_hash
                        and active_model.get("protocol_version") == active_protocol
                    )
                )
            )
            packet_bytes = observation.encoded_size_bytes or int(
                values.get("encoded_size_bytes", 0)
            )
            metadata_bits = (
                0
                if self.mask_engine.config is None
                else self.mask_engine.config.metadata_bits
            )
            packet_bits = max(0, int(packet_bytes) * 8 + metadata_bits)
            success = (
                row is not None
                and packet_bits > 0
                and self._transfer_bits(
                    branch,
                    observation,
                    rsu_id,
                    TransferDirection.UL,
                    packet_bits,
                )
            )
            vram = max(0, int(values.get("vram_bytes", 0)))
            gpu_work = max(0.0, _finite(values.get("rsu_gpu_work_s"), 0.0))
            admitted = False
            if success and row is not None:
                live_cached_models = _mapping(row.get("cached_models"))
                admission_version_valid = bool(
                    version_valid
                    and branch.active_profile_hash
                    == self.mask_engine.profile.profile_hash
                    and branch.active_protocol_version
                    == self.mask_engine.profile.protocol_version
                    and model is not None
                    and live_cached_models.get(model.model_id) == model.model_hash
                )
                admitted = (
                    not self._faulted(branch, "rsu", rsu_id, "all")
                    and message_valid
                    and admission_version_valid
                    and 1 + int(row.get("descriptors", 0))
                    <= int(row.get("descriptor_capacity", 0))
                    and vram + int(row.get("vram_bytes", 0))
                    <= int(row.get("vram_capacity_bytes", 0))
                    and gpu_work + _finite(row.get("reserved_work_gpu_s"), 0.0)
                    <= _finite(row.get("workload_capacity_gpu_s"), 0.0) + 1e-12
                )
                if admitted:
                    row["descriptors"] = int(row.get("descriptors", 0)) + 1
                    row["vram_bytes"] = int(row.get("vram_bytes", 0)) + vram
                    row["reserved_work_gpu_s"] = (
                        _finite(row.get("reserved_work_gpu_s"), 0.0) + gpu_work
                    )
            success = success and admitted
            if success:
                success = self._serve_work(
                    branch,
                    owner_type="rsu",
                    owner_id=rsu_id,
                    resource="ingress",
                    work_s=max(0.0, _finite(values.get("rsu_ingress_work_s"), 0.0)),
                    vehicle_id=observation.vehicle_id,
                )
            if success:
                success = self._serve_work(
                    branch,
                    owner_type="rsu",
                    owner_id=rsu_id,
                    resource="gpu",
                    work_s=gpu_work,
                    vehicle_id=observation.vehicle_id,
                )
            if admitted:
                self._spend_rsu(
                    branch,
                    rsu_id,
                    max(0.0, _finite(values.get("expected_rsu_energy_j"), 0.0)),
                )
            success = success and _finite(values.get("failure_probability"), 0.0) < 0.5
            if success:
                success = self._transfer_bits(
                    branch,
                    observation,
                    rsu_id,
                    TransferDirection.DL,
                    max(0.0, _finite(values.get("result_size_bits"), 0.0)),
                )
                success = success and bool(
                    branch.active_profile_hash == self.mask_engine.profile.profile_hash
                    and branch.active_protocol_version
                    == self.mask_engine.profile.protocol_version
                )
            if admitted and row is not None:
                row["descriptors"] = max(0, int(row.get("descriptors", 0)) - 1)
                row["vram_bytes"] = max(0, int(row.get("vram_bytes", 0)) - vram)
                row["reserved_work_gpu_s"] = max(
                    0.0, _finite(row.get("reserved_work_gpu_s"), 0.0) - gpu_work
                )

        duration = max(_EPS_DURATION_S, branch.elapsed_s - started)
        success = success and branch.slack_s >= -1e-12
        values["expected_duration_s"] = duration
        values["failure_probability"] = (
            0.0 if action.kind is ActionKind.PIPE else 0.0 if success else 1.0
        )
        values["completion_probability"] = (
            1.0
            if success and action.kind in {ActionKind.LOCAL, ActionKind.EDGE}
            else 0.0
        )
        values["expected_vehicle_energy_j"] = max(
            0.0, branch.vehicle_energy_j - vehicle_energy_before
        )
        if action.rsu_id:
            values["expected_rsu_energy_j"] = max(
                0.0,
                branch.rsu_energy_j.get(action.rsu_id, 0.0)
                - rsu_energy_before.get(action.rsu_id, 0.0),
            )
        if terminal:
            branch.terminal = True
            if success:
                branch.virtual_queues["coverage"] = max(
                    0.0,
                    _finite(branch.virtual_queues.get("coverage"), 0.0) - 1.0,
                )
            else:
                branch.virtual_queues["failure"] = max(
                    0.0,
                    _finite(branch.virtual_queues.get("failure"), 0.0) + 1.0,
                )
                if branch.slack_s <= _EPS_DURATION_S:
                    branch.virtual_queues["timeout"] = max(
                        0.0,
                        _finite(branch.virtual_queues.get("timeout"), 0.0) + 1.0,
                    )
        return _ScenarioOutcome(
            outcome.row_id,
            duration,
            deep_freeze(values),
            formed and success,
            outcome.artifact_key,
            terminal,
        )

    def _recourse_action(
        self,
        task: TaskRecord,
        observation: Observation,
        first: Action,
        outcome: _ScenarioOutcome,
        branch: _PredictionState,
        rng: random.Random,
    ) -> tuple[Action, Observation] | None:
        def frozen_local_or_fail() -> Action:
            pipeline = self.mask_engine.profile.pipelines.get(first.pipeline_id or "")
            fallback = None if pipeline is None else pipeline.fallback_local_model
            if fallback is None:
                return Action.fail(ActionStage.READY)
            action = Action.local(ActionStage.READY, fallback)
            model = self.mask_engine.profile.local_models.get(fallback)
            if model is None or observation.device_type not in model.supported_devices:
                return Action.fail(ActionStage.READY)
            rows = self._trace_candidates(action, observation)
            if rows:
                max_memory = max(int(row.memory_bytes) for row in rows)
                max_energy = max(float(row.dynamic_energy_j) for row in rows)
                optimistic = min(float(row.service_work_s) for row in rows)
            else:
                bound = action_estimate(
                    action, observation, self.mask_engine.trace_support
                )
                max_memory = _finite(bound.get("vehicle_memory_upper_bytes"), math.inf)
                max_energy = _finite(bound.get("vehicle_energy_upper_j"), math.inf)
                optimistic = _finite(bound.get("optimistic_duration_s"), math.inf)
            if max_memory > int(observation.vehicle.get("memory_remaining_bytes", 0)):
                return Action.fail(ActionStage.READY)
            if max_energy > branch.battery_j:
                return Action.fail(ActionStage.READY)
            if optimistic > branch.slack_s + _EPS_DURATION_S:
                return Action.fail(ActionStage.READY)
            return action

        if first.kind is not ActionKind.PIPE:
            return None
        if not outcome.formed_packet or not outcome.artifact_key:
            return frozen_local_or_fail(), observation
        if self.rollout_policy == "all_local":
            return frozen_local_or_fail(), observation
        if (
            branch.active_profile_hash != self.mask_engine.profile.profile_hash
            or branch.active_protocol_version
            != self.mask_engine.profile.protocol_version
        ):
            return frozen_local_or_fail(), observation

        # Create a sanitized predicted READY observation.  It is independent
        # of the real TaskRecord and therefore cannot mutate the simulator.
        predicted_observation = replace(
            observation,
            stage=ActionStage.READY,
            selected_pipeline=first.pipeline_id,
            artifact_key=outcome.artifact_key,
            encoded_size_bytes=int(outcome.values.get("encoded_size_bytes", 0)),
        )
        candidates: list[tuple[Action, float, float, float]] = []
        for row in getattr(self.scenario_source, "edge_rows", ()):
            if (
                self._scenario_artifact(row) != outcome.artifact_key
                or row.pipeline_id != first.pipeline_id
                or row.quality_bin not in observation.conformal_quality_bins
                or not self._edge_context_matches(row, observation, row.rsu_id)
            ):
                continue
            ul_rate, _, _, ul_state, _ = self._wireless_at(
                branch, observation, row.rsu_id, TransferDirection.UL
            )
            dl_rate, _, _, dl_state, _ = self._wireless_at(
                branch, observation, row.rsu_id, TransferDirection.DL
            )
            if ul_state != "connected" or dl_state != "connected":
                continue
            model = self.mask_engine.profile.edge_models.get(row.model_id)
            if (
                model is None
                or row.rsu_id not in model.supported_rsus
                or (
                    model.supported_pipelines
                    and first.pipeline_id not in model.supported_pipelines
                )
            ):
                continue
            queue = _mapping(branch.public_rsus.get(row.rsu_id))
            if not queue or bool(queue.get("failed", True)):
                continue
            max_age = (
                self.mask_engine.config.max_snapshot_age_s
                if self.mask_engine.config is not None
                else _finite(observation.metadata.get("max_snapshot_age_s"), 0.0)
            )
            if (
                branch.telemetry_age_s.get(row.rsu_id, math.inf)
                > max_age + _EPS_DURATION_S
            ):
                continue
            cached = _mapping(queue.get("cached_models"))
            if cached.get(row.model_id) != model.model_hash:
                continue
            if 1 + int(queue.get("descriptors", 0)) > int(
                queue.get("descriptor_capacity", 0)
            ):
                continue
            if int(row.vram_bytes) + int(queue.get("vram_bytes", 0)) > int(
                queue.get("vram_capacity_bytes", 0)
            ):
                continue
            if (
                float(row.gpu_work_s) + _finite(queue.get("reserved_work_gpu_s"), 0.0)
                > _finite(queue.get("workload_capacity_gpu_s"), 0.0) + _EPS_DURATION_S
            ):
                continue
            packet_bits = int(outcome.values.get("encoded_size_bytes", 0)) * 8
            if self.mask_engine.config is not None:
                packet_bits += self.mask_engine.config.metadata_bits
            if ul_rate <= 0 or dl_rate <= 0 or packet_bits <= 0:
                continue
            optimistic_remaining = (
                packet_bits / ul_rate
                + float(row.ingress_work_s)
                + float(row.gpu_work_s)
                + float(row.result_size_bits) / dl_rate
            )
            if optimistic_remaining > branch.slack_s + _EPS_DURATION_S:
                continue
            start_energy = _finite(
                _mapping(observation.links.get(row.rsu_id)).get(
                    "uplink_start_energy_j"
                ),
                0.0,
            )
            if start_energy > branch.battery_j:
                continue
            visible_work = max(
                0.0, _finite(queue.get("ingress_residual_work_s"), 0.0)
            ) + max(0.0, _finite(queue.get("gpu_residual_work_s"), 0.0))
            link_cost = 1.0 / ul_rate + 1.0 / dl_rate
            fer_loss = max(0.0, _finite(getattr(row, "fer_loss", 0.0), 0.0))
            rsu_energy = max(
                0.0,
                _finite(getattr(row, "ingress_energy_j", 0.0), 0.0)
                + _finite(getattr(row, "gpu_energy_j", 0.0), 0.0),
            )
            greedy_cost = optimistic_remaining + fer_loss + 0.02 * rsu_energy
            candidates.append(
                (
                    Action.edge(row.rsu_id, row.model_id),
                    link_cost,
                    visible_work,
                    greedy_cost,
                )
            )
        if candidates:
            if self.rollout_policy == "fixed_safe_lowest_link_cost":
                action = min(candidates, key=lambda item: (item[1], item[0].sort_key))[
                    0
                ]
            elif self.rollout_policy == "fixed_safe_shortest_visible_queue":
                action = min(candidates, key=lambda item: (item[2], item[0].sort_key))[
                    0
                ]
            else:
                action = min(candidates, key=lambda item: (item[3], item[0].sort_key))[
                    0
                ]
            # ``Observation`` is immutable and cumbersome to reconstruct only
            # to change artifact metadata.  The row sampler below uses an
            # explicit predicted artifact override instead.
            return action, predicted_observation
        return frozen_local_or_fail(), predicted_observation

    def _sample_recourse_outcome(
        self,
        action: Action,
        observation: Observation,
        first_outcome: _ScenarioOutcome,
        branch: _PredictionState,
        rng: random.Random,
    ) -> _ScenarioOutcome:
        if action.kind is ActionKind.EDGE:
            rows = [
                row
                for row in getattr(self.scenario_source, "edge_rows", ())
                if row.rsu_id == action.rsu_id
                and row.model_id == action.edge_model_id
                and self._scenario_artifact(row) == first_outcome.artifact_key
                and row.quality_bin in observation.conformal_quality_bins
                and self._edge_context_matches(row, observation, action.rsu_id)
            ]
            if rows:
                sampler = getattr(self.scenario_source, "sample_rows", None)
                row = (
                    sampler(rows, rng)
                    if callable(sampler)
                    else rows[rng.randrange(len(rows))]
                )
                return self._row_outcome(row)
        if action.kind is ActionKind.LOCAL:
            rows = self._trace_candidates(action, observation)
            if rows:
                sampler = getattr(self.scenario_source, "sample_rows", None)
                row = (
                    sampler(rows, rng)
                    if callable(sampler)
                    else rows[rng.randrange(len(rows))]
                )
                return self._row_outcome(row)
        return self._sample_outcome(action, observation, branch, rng)

    def _configured_vehicle_observation_row(self, vehicle_id: str) -> dict[str, Any]:
        """Build one public, frozen-config vehicle row for an isolated branch.

        A future scenario vehicle has no right to inherit the focal vehicle's
        live observation.  Only the allow-listed branch parameters are used
        here; mutable simulator state and task-private fields are unavailable.
        """

        config = self.mask_engine.config
        raw = (
            None if config is None else config.vehicle_branch_parameters.get(vehicle_id)
        )
        row = _mapping(raw)
        initial_battery_j = max(0.0, _finite(row.get("initial_battery_j"), 0.0))
        memory_capacity_bytes = max(0, int(row.get("memory_capacity_bytes", 0)))
        descriptor_capacity = {
            str(name): max(0, int(value))
            for name, value in _mapping(row.get("descriptor_capacity")).items()
        }
        server_count = _mapping(row.get("server_count"))
        resources = {
            resource: {
                "server_count": max(1, int(server_count.get(resource, 1))),
                "waiting_count": 0,
                "running_count": 0,
                "residual_work_s": 0.0,
            }
            for resource in ("accelerator", "cpu", "encoder")
        }
        return {
            "battery_j": initial_battery_j,
            "battery_capacity_j": max(
                initial_battery_j,
                _finite(row.get("battery_capacity_j"), initial_battery_j),
            ),
            "memory_capacity_bytes": memory_capacity_bytes,
            "memory_reserved_bytes": 0,
            "memory_remaining_bytes": memory_capacity_bytes,
            "descriptor_capacity": descriptor_capacity,
            "descriptors_reserved": {resource: 0 for resource in descriptor_capacity},
            "descriptor_remaining": dict(descriptor_capacity),
            "resources": resources,
            "task_energy_j": 0.0,
        }

    @staticmethod
    def _anchor_future(anchor: Any) -> Any:
        """Adapt one opaque continuation to the normal scenario-task interface."""

        local_rows = tuple(getattr(anchor, "fallback_local_rows", ()))
        edge_rows = tuple(getattr(anchor, "edge_rows", ()))
        paired_rows = (*local_rows, *edge_rows)
        quality_candidates = tuple(
            sorted({str(getattr(row, "quality_bin")) for row in paired_rows})
        )
        context = next((getattr(row, "context", None) for row in paired_rows), None)
        device_type = next(
            (
                str(getattr(row, "device_type"))
                for row in paired_rows
                if getattr(row, "device_type", None) is not None
            ),
            "unsupported",
        )
        return SimpleNamespace(
            task_token=str(getattr(anchor, "task_token")),
            arrival_offset_s=0.0,
            relative_deadline_s=max(
                _EPS_DURATION_S, float(getattr(anchor, "deadline_offset_s"))
            ),
            vehicle_id=str(getattr(anchor, "vehicle_id")),
            device_type=device_type,
            context=context,
            quality_candidates=quality_candidates,
            quality_probabilities=tuple(
                (quality_bin, 1.0 / len(quality_candidates))
                for quality_bin in quality_candidates
            ),
            ood=False,
            quality_features=(),
            prep_work_s=0.0,
            prep_energy_j=0.0,
            prep_memory_bytes=0,
            prep_failed=bool(getattr(anchor, "prep_failed", False)),
            local_rows=local_rows,
            anon_rows=(),
            edge_rows=edge_rows,
            complete_support=bool(getattr(anchor, "complete_support", False)),
            support_reason=getattr(anchor, "support_reason", None),
            anchor=anchor,
        )

    @staticmethod
    def _anchor_outcome(anchor: Any) -> _ScenarioOutcome:
        edge = str(getattr(anchor, "path_kind", "")) == "edge"
        values: dict[str, Any] = {
            "failure_probability": (
                1.0 if bool(getattr(anchor, "inference_failed", False)) else 0.0
            ),
            "completion_probability": (
                0.0 if bool(getattr(anchor, "inference_failed", False)) else 1.0
            ),
            "expected_fer_loss": max(
                0.0, _finite(getattr(anchor, "fer_loss", 0.0), 0.0)
            ),
            "vehicle_work_s": max(
                0.0, _finite(getattr(anchor, "total_work_s", 0.0), 0.0)
            ),
            "expected_vehicle_energy_j": max(
                0.0, _finite(getattr(anchor, "total_energy_j", 0.0), 0.0)
            ),
            "vehicle_memory_upper_bytes": max(
                0, int(getattr(anchor, "action_memory_bytes", 0))
            ),
            "quality_bin": (
                ""
                if getattr(anchor, "realized_quality_bin", None) is None
                else str(getattr(anchor, "realized_quality_bin"))
            ),
            "ingress_failure": bool(getattr(anchor, "ingress_failed", False)),
        }
        if edge:
            values.update(
                {
                    "rsu_ingress_work_s": max(
                        0.0,
                        _finite(getattr(anchor, "ingress_total_work_s", 0.0), 0.0),
                    ),
                    "rsu_ingress_energy_j": max(
                        0.0,
                        _finite(getattr(anchor, "ingress_total_energy_j", 0.0), 0.0),
                    ),
                    "rsu_gpu_work_s": max(
                        0.0,
                        _finite(getattr(anchor, "gpu_total_work_s", 0.0), 0.0),
                    ),
                    "rsu_gpu_energy_j": max(
                        0.0,
                        _finite(getattr(anchor, "gpu_total_energy_j", 0.0), 0.0),
                    ),
                    "vram_bytes": max(0, int(getattr(anchor, "vram_bytes", 0))),
                    "result_size_bits": max(
                        1.0, _finite(getattr(anchor, "result_size_bits", 0.0), 0.0)
                    ),
                }
            )
        return _ScenarioOutcome(
            row_id=f"anchor-continuation:{getattr(anchor, 'task_token')}",
            duration_s=_EPS_DURATION_S,
            values=deep_freeze(values),
            formed_packet=edge and bool(getattr(anchor, "artifact_token", None)),
            artifact_key=getattr(anchor, "artifact_token", None),
            terminal=True,
        )

    @staticmethod
    def _anchor_action(anchor: Any) -> Action | None:
        path_kind = str(getattr(anchor, "path_kind", ""))
        model_id = getattr(anchor, "model_id", None)
        if path_kind == "edge":
            rsu_id = getattr(anchor, "rsu_id", None)
            return (
                Action.edge(str(rsu_id), str(model_id)) if rsu_id and model_id else None
            )
        if path_kind == "local" and model_id:
            stage = (
                ActionStage.READY
                if getattr(anchor, "pipeline_id", None)
                else ActionStage.RAW
            )
            return Action.local(stage, str(model_id))
        return None

    def _future_observation(
        self,
        future: Any,
        base: Observation,
        branch: _PredictionState,
        *,
        stage: ActionStage,
        selected_pipeline: str | None = None,
        artifact_key: str | None = None,
        encoded_size_bytes: int | None = None,
        vehicle_state: Mapping[str, Any] | None = None,
    ) -> Observation:
        """Build a causal observation for one sanitized scenario task."""

        future_task_token = str(getattr(future, "task_token"))
        future_vehicle_id = str(getattr(future, "vehicle_id"))
        if vehicle_state is not None:
            vehicle = thaw_json(vehicle_state)
        elif future_vehicle_id == base.vehicle_id:
            vehicle = thaw_json(base.vehicle)
            vehicle["battery_j"] = max(0.0, branch.battery_j)
            resources = _mapping(vehicle.get("resources"))
            vehicle["resources"] = {
                str(name): {
                    **thaw_json(_mapping(row)),
                    "residual_work_s": branch.vehicle_queues.get(str(name), 0.0),
                }
                for name, row in resources.items()
            }
        else:
            vehicle = self._configured_vehicle_observation_row(future_vehicle_id)
        rsus: dict[str, Any] = {}
        links: dict[str, Any] = {}
        wireless_observation = replace(base, vehicle_id=future_vehicle_id)
        for rsu_id, source in sorted(branch.public_rsus.items()):
            row = thaw_json(source)
            row["snapshot_age_s"] = branch.telemetry_age_s.get(rsu_id, math.inf)
            rsus[rsu_id] = row
            ul = self._wireless_at(
                branch, wireless_observation, rsu_id, TransferDirection.UL
            )
            dl = self._wireless_at(
                branch, wireless_observation, rsu_id, TransferDirection.DL
            )
            links[rsu_id] = {
                "connected": ul[3] == "connected" and dl[3] == "connected",
                "ul_goodput_bps": ul[0],
                "dl_goodput_bps": dl[0],
                "ul_link_state": ul[3],
                "dl_link_state": dl[3],
                "ul_transmitter_power_w": ul[1],
                "ul_receiver_power_w": ul[2],
                "dl_transmitter_power_w": dl[1],
                "dl_receiver_power_w": dl[2],
                "uplink_start_energy_j": 0.001,
            }
        context = getattr(future, "context", None)
        if context is None:
            device_context = base.device_context
        else:
            device_context = "|".join(
                str(getattr(context, name, "nominal"))
                for name in ("thermal_state", "power_mode", "memory_pressure")
            )
        features = getattr(future, "quality_features", ())
        if isinstance(features, Mapping):
            quality_features = tuple(
                float(value) for _, value in sorted(features.items())
            )
        else:
            quality_features = tuple(float(value) for value in features)
        bins = tuple(str(value) for value in getattr(future, "quality_candidates", ()))
        probabilities = tuple(
            (str(name), float(value))
            for name, value in getattr(future, "quality_probabilities", ())
        )
        if not probabilities and bins:
            probabilities = tuple((name, 1.0 / len(bins)) for name in bins)
        pipeline = self.mask_engine.profile.pipelines.get(selected_pipeline or "")
        # ScenarioLibrary artifact identifiers are already library-local opaque
        # tokens.  Keep them directly inside the isolated branch; never route
        # them through the evaluation capability registry.
        artifact_token = None if artifact_key is None else str(artifact_key)
        encoded_evidence: Mapping[str, Any] = {}
        if stage is ActionStage.READY and pipeline is not None and artifact_token:
            encoded_evidence = {
                "message_source_type": "EncodedAnon",
                "artifact_token": artifact_token,
                "pipeline_id": pipeline.pipeline_id,
                "pipeline_hash": pipeline.pipeline_hash,
                "guard_hash": pipeline.guard_hash,
                "encoder_hash": pipeline.encoder_hash,
                "profile_hash": self.mask_engine.profile.profile_hash,
                "quality_bins": bins,
                "size_bytes": encoded_size_bytes,
            }
        arrival = float(getattr(future, "arrival_offset_s"))
        deadline = arrival + float(getattr(future, "relative_deadline_s"))
        versions = thaw_json(base.versions)
        versions["profile_hash"] = branch.active_profile_hash
        versions["protocol_version"] = branch.active_protocol_version
        versions["local_models"] = {
            model_id: {
                "model_hash": branch.active_local_model_hashes.get(
                    (str(getattr(future, "vehicle_id")), model_id),
                    model.model_hash,
                ),
                "protocol_version": branch.active_protocol_version,
            }
            for model_id, model in sorted(self.mask_engine.profile.local_models.items())
        }
        return replace(
            base,
            time_s=branch.elapsed_s,
            task_id=future_task_token,
            vehicle_id=future_vehicle_id,
            stage=stage,
            task_state=TaskState.RAW if stage is ActionStage.RAW else TaskState.READY,
            arrival_time_s=arrival,
            absolute_deadline_s=deadline,
            slack_s=deadline - branch.elapsed_s,
            quality_features=quality_features,
            quality_probabilities=probabilities,
            conformal_quality_bins=bins,
            ood=bool(getattr(future, "ood", False)),
            device_type=str(getattr(future, "device_type")),
            device_context=device_context,
            selected_pipeline=selected_pipeline,
            selected_local_model=None,
            selected_rsu=None,
            selected_edge_model=None,
            artifact_token=artifact_token,
            encoded_size_bytes=encoded_size_bytes,
            encoded_evidence=deep_freeze(encoded_evidence),
            remaining_bits=deep_freeze({"uplink": 0.0, "downlink": 0.0}),
            vehicle=deep_freeze(vehicle),
            rsus=deep_freeze(rsus),
            links=deep_freeze(links),
            virtual_queues=deep_freeze(branch.virtual_queues),
            versions=deep_freeze(versions),
        )

    @staticmethod
    def _future_task_record(future: Any, observation: Observation) -> TaskRecord:
        return TaskRecord(
            task_id=observation.task_id,
            vehicle_id=observation.vehicle_id,
            arrival_time_s=observation.arrival_time_s,
            relative_deadline_s=(
                observation.absolute_deadline_s - observation.arrival_time_s
            ),
            absolute_deadline_s=observation.absolute_deadline_s,
            raw_handle=None,
            quality_features=observation.quality_features,
            quality_probabilities=observation.quality_probabilities,
            conformal_quality_bins=observation.conformal_quality_bins,
            ood=observation.ood,
            device_context=observation.device_context or "nominal",
            selected_pipeline=observation.selected_pipeline,
            artifact_key=observation.artifact_token,
            encoded_size_bytes=observation.encoded_size_bytes,
        )

    def _future_action(
        self,
        future: Any,
        observation: Observation,
        branch: _PredictionState,
    ) -> Action:
        task = self._future_task_record(future, observation)
        mask = self.mask_engine.enumerate(task, observation)
        allowed = tuple(mask.allowed)
        local = [action for action in allowed if action.kind is ActionKind.LOCAL]
        if self.rollout_policy == "all_local":
            return min(local) if local else _fail(mask)
        if observation.stage is ActionStage.RAW:
            if self.rollout_policy.startswith("fixed_safe"):
                fixed = [
                    action
                    for action in allowed
                    if action.kind is ActionKind.PIPE
                    and action.pipeline_id == self.rollout_pipeline_id
                ]
                return min(fixed) if fixed else min(local) if local else _fail(mask)
        else:
            edges = [action for action in allowed if action.kind is ActionKind.EDGE]
            if self.rollout_policy.startswith("fixed_safe"):
                edges = [
                    action
                    for action in edges
                    if action.edge_model_id == self.rollout_edge_model_id
                ]
                if edges:
                    if self.rollout_policy == "fixed_safe_lowest_link_cost":
                        return min(
                            edges,
                            key=lambda action: (
                                FixedSafeLowestLinkCostPolicy._link_cost(
                                    action, observation
                                ),
                                action.sort_key,
                            ),
                        )
                    return min(
                        edges,
                        key=lambda action: (
                            FixedSafeShortestQueuePolicy._queue_metric(
                                action, observation
                            ),
                            action.sort_key,
                        ),
                    )
                return min(local) if local else _fail(mask)
        scores = {
            action: expected_task_cost(action, observation, self.mask_engine)
            for action in allowed
        }
        return min(allowed, key=lambda action: (scores[action], action.sort_key))

    def _future_outcome(
        self,
        future: Any,
        action: Action,
        observation: Observation,
        rng: random.Random,
        *,
        pairing_token: str | None = None,
        quality_bin: str | None = None,
    ) -> _ScenarioOutcome:
        if action.kind is ActionKind.FAIL:
            return self._row_outcome(
                {
                    "scenario_id": f"{observation.task_id}:explicit-fail",
                    "expected_duration_s": _EPS_DURATION_S,
                    "failure_probability": 1.0,
                    "completion_probability": 0.0,
                }
            )
        if action.kind is ActionKind.LOCAL:
            rows = tuple(getattr(future, "local_rows", ()))
            rows = tuple(
                row
                for row in rows
                if row.model_id == action.local_model_id
                and row.quality_bin in observation.conformal_quality_bins
                and row.device_type == observation.device_type
                and self._context_matches(row, observation)
            )
        elif action.kind is ActionKind.PIPE:
            rows = tuple(getattr(future, "anon_rows", ()))
            rows = tuple(
                row
                for row in rows
                if row.pipeline_id == action.pipeline_id
                and row.quality_bin in observation.conformal_quality_bins
                and row.device_type == observation.device_type
                and self._context_matches(row, observation)
            )
        else:
            rows = self._actual_edge_rows(
                future,
                action,
                observation,
                pairing_token or observation.artifact_token,
                quality_bin=quality_bin,
            )
            if not rows:
                raise LookupError(
                    f"future task lacks an actual paired row for {action.canonical_id}"
                )
            sampler = getattr(self.scenario_source, "sample_rows", None)
            row = (
                sampler(rows, rng)
                if callable(sampler)
                else rows[rng.randrange(len(rows))]
            )
            return self._row_outcome(row)
        covered = {str(row.quality_bin) for row in rows}
        if covered != set(observation.conformal_quality_bins):
            raise LookupError(
                f"future task lacks paired rows for {action.canonical_id}"
            )
        row = self._sample_quality_weighted_rows(rows, observation, rng)
        return self._row_outcome(row)

    def _scheduler_new(
        self,
        branch: _PredictionState,
        observation: Observation,
        environment: ScenarioEnvironment,
    ) -> _BranchScheduler | None:
        config = self.mask_engine.config
        if config is None:
            branch.incomplete_reason = "BRANCH_RUNTIME_PARAMETERS_MISSING"
            return None
        future_tasks = tuple(getattr(environment, "future_tasks", ()))
        anchor_by_vehicle = {
            str(getattr(item, "vehicle_id")): item
            for item in getattr(environment, "vehicle_anchors", ())
        }
        anchor_by_rsu = {
            str(getattr(item, "rsu_id")): item
            for item in getattr(environment, "rsu_anchors", ())
        }
        if set(anchor_by_rsu) != set(branch.rsus):
            branch.incomplete_reason = "SCENARIO_RSU_ANCHOR_MISSING"
            return None
        nonfocal_future_vehicles = {
            str(getattr(item, "vehicle_id"))
            for item in future_tasks
            if str(getattr(item, "vehicle_id")) != observation.vehicle_id
        }
        nonfocal_anchor_vehicles = {
            vehicle_id
            for vehicle_id in anchor_by_vehicle
            if vehicle_id != observation.vehicle_id
        }
        nonfocal_branch_vehicles = nonfocal_future_vehicles | nonfocal_anchor_vehicles
        if any(
            not bool(getattr(item, "complete_support", False))
            or getattr(item, "vehicle_id", None) not in config.vehicle_branch_parameters
            for item in future_tasks
        ):
            branch.incomplete_reason = "SCENARIO_FUTURE_TASK_SUPPORT_INCOMPLETE"
            return None
        if any(
            not bool(getattr(item, "complete_support", False))
            for item in anchor_by_rsu.values()
        ):
            branch.incomplete_reason = "SCENARIO_RSU_ANCHOR_INCOMPLETE"
            return None
        if any(
            vehicle_id not in anchor_by_vehicle
            or not bool(
                getattr(anchor_by_vehicle[vehicle_id], "complete_support", False)
            )
            for vehicle_id in nonfocal_branch_vehicles
        ):
            branch.incomplete_reason = "SCENARIO_VEHICLE_ANCHOR_INCOMPLETE"
            return None
        resources: dict[tuple[str, str, str], _BranchResource] = {}
        vehicle_observation_rows: dict[str, dict[str, Any]] = {}
        batteries: dict[str, float] = {}
        memory_capacity: dict[str, int] = {}
        memory_reserved: dict[str, int] = {}
        descriptor_capacity: dict[str, dict[str, int]] = {}
        descriptor_reserved: dict[str, dict[str, int]] = {}
        for vehicle_id, raw in config.vehicle_branch_parameters.items():
            configured = self._configured_vehicle_observation_row(vehicle_id)
            if vehicle_id == observation.vehicle_id:
                row = thaw_json(observation.vehicle)
            else:
                anchor = anchor_by_vehicle.get(vehicle_id)
                if anchor is None:
                    # Vehicles absent from this scenario window are never
                    # initialized from deployment t=0 and therefore cannot
                    # silently participate in recourse.
                    row = configured
                else:
                    row = {
                        **configured,
                        "battery_j": float(getattr(anchor, "battery_j")),
                        "memory_capacity_bytes": int(
                            getattr(anchor, "memory_capacity_bytes")
                        ),
                        "memory_reserved_bytes": int(
                            getattr(anchor, "memory_reserved_bytes")
                        ),
                        "descriptor_capacity": dict(
                            getattr(anchor, "descriptor_capacity")
                        ),
                        "descriptors_reserved": dict(
                            getattr(anchor, "descriptors_reserved")
                        ),
                        "resources": {
                            str(name): thaw_json(resource_row)
                            for name, resource_row in getattr(
                                anchor, "resources"
                            ).items()
                        },
                    }
                    row["memory_remaining_bytes"] = max(
                        0,
                        row["memory_capacity_bytes"] - row["memory_reserved_bytes"],
                    )
                    row["descriptor_remaining"] = {
                        resource: max(
                            0,
                            int(row["descriptor_capacity"].get(resource, 0))
                            - int(row["descriptors_reserved"].get(resource, 0)),
                        )
                        for resource in row["descriptor_capacity"]
                    }
            vehicle_observation_rows[vehicle_id] = row
            batteries[vehicle_id] = max(0.0, _finite(row.get("battery_j"), 0.0))
            if vehicle_id == observation.vehicle_id:
                batteries[vehicle_id] = max(
                    0.0, _finite(observation.vehicle.get("battery_j"), 0.0)
                )
            memory_capacity[vehicle_id] = max(
                0, int(row.get("memory_capacity_bytes", 0))
            )
            memory_reserved[vehicle_id] = int(row.get("memory_reserved_bytes", 0))
            descriptor_capacity[vehicle_id] = {
                str(name): max(0, int(value))
                for name, value in _mapping(row.get("descriptor_capacity")).items()
            }
            descriptor_reserved[vehicle_id] = {
                resource: int(
                    _mapping(row.get("descriptors_reserved")).get(resource, 0)
                )
                for resource in descriptor_capacity[vehicle_id]
            }
            if (
                memory_reserved[vehicle_id] < 0
                or memory_reserved[vehicle_id] > memory_capacity[vehicle_id]
                or any(
                    count < 0
                    or count > descriptor_capacity[vehicle_id].get(resource, 0)
                    for resource, count in descriptor_reserved[vehicle_id].items()
                )
            ):
                branch.incomplete_reason = "BRANCH_VEHICLE_RESERVATION_STATE_INVALID"
                return None
            resource_rows = _mapping(row.get("resources"))
            configured_servers = _mapping(_mapping(raw).get("server_count"))
            for resource in ("accelerator", "cpu", "encoder"):
                resource_row = _mapping(resource_rows.get(resource))
                resources[("vehicle", vehicle_id, resource)] = _BranchResource(
                    "vehicle",
                    vehicle_id,
                    resource,
                    max(
                        1,
                        int(
                            resource_row.get(
                                "server_count",
                                configured_servers.get(resource, 1),
                            )
                        ),
                    ),
                )
        for rsu_id, row in branch.rsus.items():
            resources[("rsu", rsu_id, "ingress")] = _BranchResource(
                "rsu", rsu_id, "ingress", max(1, int(row.get("ingress_servers", 1)))
            )
            resources[("rsu", rsu_id, "gpu")] = _BranchResource(
                "rsu", rsu_id, "gpu", max(1, int(row.get("gpu_servers", 1)))
            )
        scheduler = _BranchScheduler(
            branch=branch,
            focal_vehicle_id=observation.vehicle_id,
            tasks={},
            resources=resources,
            jobs={},
            transfers={},
            events=[],
            vehicle_observation_rows=vehicle_observation_rows,
            vehicle_battery_j=batteries,
            vehicle_memory_capacity=memory_capacity,
            vehicle_memory_reserved=memory_reserved,
            descriptor_capacity=descriptor_capacity,
            descriptor_reserved=descriptor_reserved,
        )
        branch.use_future_tasks = True
        branch.complete_macro_recourse = True
        branch.incomplete_reason = None

        # Non-focal vehicles inherit a causal, identity-free state replayed
        # from this sampled training/validation window.  Each active job and
        # packet remains associated with an opaque scenario-local task token;
        # aggregate workload is never reset to deployment t=0.
        for vehicle_id in sorted(nonfocal_branch_vehicles):
            anchor = anchor_by_vehicle[vehicle_id]
            anchored_tasks = {
                str(getattr(item, "task_token")): item
                for item in getattr(anchor, "tasks", ())
            }
            if len(anchored_tasks) != int(
                getattr(anchor, "active_task_count", len(anchored_tasks))
            ):
                branch.incomplete_reason = "SCENARIO_ANCHOR_TASK_COUNT_MISMATCH"
                branch.complete_macro_recourse = False
                return None
            for token, anchored in sorted(anchored_tasks.items()):
                state = str(getattr(anchored, "state"))
                deadline = float(getattr(anchored, "deadline_offset_s"))
                if (
                    state
                    not in {
                        "PREP",
                        "RAW_CONTROL",
                        "COMPUTE",
                        "READY_CONTROL",
                        "UL",
                        "RSU_INGRESS",
                        "RSU_GPU",
                        "DL",
                        "LOCAL_FALLBACK",
                    }
                    or deadline <= 1e-12
                ):
                    branch.incomplete_reason = (
                        "SCENARIO_ANCHOR_TASK_CONTINUATION_UNSUPPORTED"
                    )
                    branch.complete_macro_recourse = False
                    return None
                future = self._anchor_future(anchored)
                outcome = self._anchor_outcome(anchored)
                action = self._anchor_action(anchored)
                stages = [
                    (
                        str(getattr(stage, "resource")),
                        float(getattr(stage, "work_s")),
                        float(getattr(stage, "energy_j")),
                        str(getattr(stage, "stage")),
                    )
                    for stage in getattr(anchored, "remaining_vehicle_stages", ())
                ]
                anchored_task = _BranchTask(
                    token,
                    vehicle_id,
                    0.0,
                    deadline,
                    future,
                    observation,
                    state=state,
                    selected_pipeline=getattr(anchored, "pipeline_id", None),
                    artifact_key=getattr(anchored, "artifact_token", None),
                    encoded_size_bytes=(
                        max(1, int(float(getattr(anchored, "uplink_bits", 0.0)) / 8))
                        if getattr(anchored, "path_kind", "") == "edge"
                        else None
                    ),
                    reservation_tokens={
                        str(name): int(value)
                        for name, value in getattr(
                            anchored, "descriptor_tokens", {}
                        ).items()
                    },
                    reserved_memory_bytes=int(
                        getattr(anchored, "memory_reserved_bytes", 0)
                    ),
                    last_action=action,
                    last_outcome=outcome,
                    edge_outcome=outcome
                    if getattr(anchored, "path_kind", "") == "edge"
                    else None,
                    edge_pairing_token=getattr(anchored, "artifact_token", None),
                    admitted_rsu=(
                        str(getattr(anchored, "rsu_id"))
                        if state in {"RSU_INGRESS", "RSU_GPU"}
                        and getattr(anchored, "rsu_id", None)
                        else None
                    ),
                    reserved_vram_bytes=(
                        int(getattr(anchored, "admission_vram_upper_bytes", 0))
                        if state in {"RSU_INGRESS", "RSU_GPU"}
                        else 0
                    ),
                    reserved_gpu_work_s=(
                        float(getattr(anchored, "admission_gpu_work_upper_s", 0.0))
                        if state in {"RSU_INGRESS", "RSU_GPU"}
                        else 0.0
                    ),
                    fixed_continuation=True,
                    continuation_path_kind=str(getattr(anchored, "path_kind", "")),
                    continuation_prep_failed=bool(
                        getattr(anchored, "prep_failed", False)
                    ),
                    continuation_stages=stages,
                    continuation_action_memory_bytes=int(
                        getattr(anchored, "action_memory_bytes", 0)
                    ),
                    continuation_action_tokens={
                        str(name): int(value)
                        for name, value in getattr(
                            anchored, "action_descriptor_tokens", {}
                        ).items()
                    },
                    continuation_control_next=getattr(
                        anchored, "controller_next", None
                    ),
                    admission_vram_upper_bytes=int(
                        getattr(anchored, "admission_vram_upper_bytes", 0)
                    ),
                    admission_gpu_work_upper_s=float(
                        getattr(anchored, "admission_gpu_work_upper_s", 0.0)
                    ),
                    latent_quality_bin=getattr(anchored, "realized_quality_bin", None),
                )
                scheduler.tasks[token] = anchored_task
                self._scheduler_push(scheduler, deadline, 20, "DEADLINE", token)
                if state in {"RAW_CONTROL", "READY_CONTROL"}:
                    self._scheduler_push(
                        scheduler,
                        max(
                            0.0,
                            float(getattr(anchored, "controller_remaining_s", 0.0)),
                        ),
                        40,
                        "ANCHOR_CONTROL_DONE",
                        token,
                    )

            anchor_resources = getattr(anchor, "resources", {})
            restored_job_tokens: set[str] = set()
            for resource in ("accelerator", "cpu", "encoder"):
                resource_row = _mapping(anchor_resources.get(resource))
                pool = scheduler.resources[("vehicle", vehicle_id, resource)]
                for bucket, running_bucket in (
                    ("running_jobs", True),
                    ("waiting_jobs", False),
                ):
                    jobs = tuple(resource_row.get(bucket, ()))
                    for job_row_raw in jobs:
                        job_row = _mapping(job_row_raw)
                        task_token = str(job_row.get("task_token", ""))
                        if (
                            not task_token
                            or task_token not in anchored_tasks
                            or task_token in restored_job_tokens
                        ):
                            branch.incomplete_reason = (
                                "SCENARIO_ANCHOR_JOB_ASSOCIATION_INVALID"
                            )
                            branch.complete_macro_recourse = False
                            return None
                        remaining = max(
                            0.0, _finite(job_row.get("remaining_work_s"), 0.0)
                        )
                        total = max(
                            remaining, _finite(job_row.get("total_work_s"), remaining)
                        )
                        energy = max(0.0, _finite(job_row.get("total_energy_j"), 0.0))
                        if remaining <= 0 or total <= 0:
                            branch.incomplete_reason = (
                                "SCENARIO_ANCHOR_JOB_WORK_INVALID"
                            )
                            branch.complete_macro_recourse = False
                            return None
                        scheduler.sequence += 1
                        job_id = f"branch-anchor-job:{scheduler.sequence:08d}"
                        anchor_state = str(getattr(anchored_tasks[task_token], "state"))
                        completion_kind = {
                            "PREP": "ANCHOR_PREP_DONE",
                            "COMPUTE": "ANCHOR_COMPUTE_DONE",
                            "LOCAL_FALLBACK": "ANCHOR_LOCAL_DONE",
                        }.get(anchor_state)
                        if completion_kind is None:
                            branch.incomplete_reason = (
                                "SCENARIO_ANCHOR_VEHICLE_JOB_STATE_INVALID"
                            )
                            branch.complete_macro_recourse = False
                            return None
                        scheduler.jobs[job_id] = _BranchJob(
                            job_id,
                            task_token,
                            "vehicle",
                            vehicle_id,
                            resource,
                            remaining,
                            total,
                            energy,
                            float(
                                job_row.get(
                                    "deadline_offset_s",
                                    getattr(
                                        anchored_tasks[task_token], "deadline_offset_s"
                                    ),
                                )
                            ),
                            int(job_row.get("enqueue_seq", scheduler.sequence)),
                            completion_kind,
                        )
                        restored_job_tokens.add(task_token)
                        (pool.running if running_bucket else pool.waiting).append(
                            job_id
                        )
            compute_tokens = {
                token
                for token, item in anchored_tasks.items()
                if str(getattr(item, "state")) in {"PREP", "COMPUTE", "LOCAL_FALLBACK"}
            }
            if restored_job_tokens != compute_tokens:
                branch.incomplete_reason = "SCENARIO_ANCHOR_JOB_SET_MISMATCH"
                branch.complete_macro_recourse = False
                return None

            restored_transfer_tokens: set[str] = set()
            for transfer_anchor in getattr(anchor, "transfers", ()):
                task_token = str(getattr(transfer_anchor, "task_token"))
                transfer_id = str(getattr(transfer_anchor, "transfer_token"))
                if (
                    task_token not in anchored_tasks
                    or task_token in restored_transfer_tokens
                    or transfer_id in scheduler.transfers
                ):
                    branch.incomplete_reason = (
                        "SCENARIO_ANCHOR_TRANSFER_ASSOCIATION_INVALID"
                    )
                    branch.complete_macro_recourse = False
                    return None
                status = str(getattr(transfer_anchor, "status"))
                if status not in {"connected", "temporary_outage"}:
                    branch.incomplete_reason = "SCENARIO_ANCHOR_TRANSFER_UNSUPPORTED"
                    branch.complete_macro_recourse = False
                    return None
                remaining_bits = float(getattr(transfer_anchor, "remaining_bits"))
                if remaining_bits <= 0:
                    branch.incomplete_reason = "SCENARIO_ANCHOR_TRANSFER_BITS_INVALID"
                    branch.complete_macro_recourse = False
                    return None
                scheduler.transfers[transfer_id] = _BranchTransfer(
                    transfer_id,
                    task_token,
                    vehicle_id,
                    str(getattr(transfer_anchor, "rsu_id")),
                    getattr(transfer_anchor, "direction"),
                    remaining_bits,
                    paused_since_s=(
                        -max(0.0, float(getattr(transfer_anchor, "pause_age_s")))
                        if status == "temporary_outage"
                        else None
                    ),
                )
                restored_transfer_tokens.add(task_token)
            transfer_task_tokens = {
                token
                for token, item in anchored_tasks.items()
                if str(getattr(item, "state")) in {"UL", "DL"}
            }
            if restored_transfer_tokens != transfer_task_tokens:
                branch.incomplete_reason = "SCENARIO_ANCHOR_TRANSFER_SET_MISMATCH"
                branch.complete_macro_recourse = False
                return None

        # Restore private RSU reservations and associated ingress/GPU jobs from
        # the same joint anchor.  Tasks belonging to the real focal vehicle are
        # replaced by its live observation and therefore are deliberately not
        # imported from the sampled scenario anchor.
        fixed_tasks = {
            token: task
            for token, task in scheduler.tasks.items()
            if task.fixed_continuation
        }
        for rsu_id, anchor in sorted(anchor_by_rsu.items()):
            all_anchor_admitted = {
                str(getattr(item, "task_token")): item
                for vehicle_anchor in anchor_by_vehicle.values()
                for item in getattr(vehicle_anchor, "tasks", ())
                if str(getattr(item, "rsu_id", "")) == rsu_id
                and str(getattr(item, "state", "")) in {"RSU_INGRESS", "RSU_GPU"}
            }
            anchor_job_tokens = {
                str(_mapping(job).get("task_token", ""))
                for resource in ("ingress", "gpu")
                for bucket in ("running_jobs", "waiting_jobs")
                for job in _mapping(getattr(anchor, "resources", {}).get(resource)).get(
                    bucket, ()
                )
            }
            if (
                len(all_anchor_admitted)
                != int(getattr(anchor, "active_task_count", -1))
                or anchor_job_tokens != set(all_anchor_admitted)
                or int(getattr(anchor, "descriptors_reserved", -1))
                != len(all_anchor_admitted)
                or int(getattr(anchor, "vram_reserved_bytes", -1))
                != sum(
                    int(getattr(item, "admission_vram_upper_bytes", 0))
                    for item in all_anchor_admitted.values()
                )
                or not math.isclose(
                    float(getattr(anchor, "workload_reserved_gpu_s", -1.0)),
                    sum(
                        float(getattr(item, "admission_gpu_work_upper_s", 0.0))
                        for item in all_anchor_admitted.values()
                    ),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                branch.incomplete_reason = "SCENARIO_RSU_ANCHOR_AGGREGATE_MISMATCH"
                branch.complete_macro_recourse = False
                return None
            admitted = tuple(
                task
                for task in fixed_tasks.values()
                if task.admitted_rsu == rsu_id
                and task.state in {"RSU_INGRESS", "RSU_GPU"}
            )
            live_row = branch.rsus[rsu_id]
            live_row["descriptors"] = len(admitted)
            live_row["vram_bytes"] = sum(task.reserved_vram_bytes for task in admitted)
            live_row["reserved_work_gpu_s"] = sum(
                task.reserved_gpu_work_s for task in admitted
            )
            restored_rsu_tokens: set[str] = set()
            for resource in ("ingress", "gpu"):
                resource_row = _mapping(getattr(anchor, "resources", {}).get(resource))
                pool = scheduler.resources[("rsu", rsu_id, resource)]
                for bucket, running_bucket in (
                    ("running_jobs", True),
                    ("waiting_jobs", False),
                ):
                    for job_row_raw in tuple(resource_row.get(bucket, ())):
                        job_row = _mapping(job_row_raw)
                        task_token = str(job_row.get("task_token", ""))
                        task = fixed_tasks.get(task_token)
                        if task is None:
                            continue
                        expected_state = (
                            "RSU_INGRESS" if resource == "ingress" else "RSU_GPU"
                        )
                        if (
                            task.state != expected_state
                            or task_token in restored_rsu_tokens
                        ):
                            branch.incomplete_reason = (
                                "SCENARIO_RSU_ANCHOR_JOB_ASSOCIATION_INVALID"
                            )
                            branch.complete_macro_recourse = False
                            return None
                        remaining = max(
                            0.0, _finite(job_row.get("remaining_work_s"), 0.0)
                        )
                        total = max(
                            remaining, _finite(job_row.get("total_work_s"), remaining)
                        )
                        energy = max(0.0, _finite(job_row.get("total_energy_j"), 0.0))
                        if remaining <= 0 or total <= 0:
                            branch.incomplete_reason = (
                                "SCENARIO_RSU_ANCHOR_JOB_WORK_INVALID"
                            )
                            branch.complete_macro_recourse = False
                            return None
                        scheduler.sequence += 1
                        job_id = f"branch-rsu-anchor-job:{scheduler.sequence:08d}"
                        scheduler.jobs[job_id] = _BranchJob(
                            job_id,
                            task_token,
                            "rsu",
                            rsu_id,
                            resource,
                            remaining,
                            total,
                            energy,
                            task.deadline_s,
                            int(job_row.get("enqueue_seq", scheduler.sequence)),
                            "INGRESS_DONE" if resource == "ingress" else "GPU_DONE",
                        )
                        restored_rsu_tokens.add(task_token)
                        (pool.running if running_bucket else pool.waiting).append(
                            job_id
                        )
            expected_rsu_tokens = {
                task.task_token
                for task in admitted
                if task.state in {"RSU_INGRESS", "RSU_GPU"}
            }
            if restored_rsu_tokens != expected_rsu_tokens:
                branch.incomplete_reason = "SCENARIO_RSU_ANCHOR_JOB_SET_MISMATCH"
                branch.complete_macro_recourse = False
                return None

        if not self._scheduler_restore_live_vehicle_tasks(scheduler, observation):
            return None
        for future in future_tasks:
            token = str(getattr(future, "task_token"))
            arrival = float(getattr(future, "arrival_offset_s"))
            deadline = arrival + float(getattr(future, "relative_deadline_s"))
            scheduler.tasks[token] = _BranchTask(
                token,
                str(getattr(future, "vehicle_id")),
                arrival,
                deadline,
                future,
                observation,
            )
            self._scheduler_push(scheduler, arrival, 30, "ARRIVAL", token)
            self._scheduler_push(scheduler, deadline, 20, "DEADLINE", token)
        for offset in environment.macro_event_offsets:
            self._scheduler_push(scheduler, offset, 10, "ENVIRONMENT", "")
        self._scheduler_dispatch(scheduler)
        return scheduler

    def _scheduler_restore_live_vehicle_tasks(
        self,
        scheduler: _BranchScheduler,
        observation: Observation,
    ) -> bool:
        """Restore every non-focal task already active on the real vehicle.

        The observation exposes opaque task-local associations for jobs,
        transfers and admitted RSU work.  Frozen scenario rows supply only the
        not-yet-observed conditional outcome.  Already consumed work/energy is
        never replayed, while residual dynamic energy and all later stages are
        charged normally.
        """

        branch = scheduler.branch

        def incomplete(reason: str) -> bool:
            branch.complete_macro_recourse = False
            branch.incomplete_reason = reason
            return False

        raw_tasks = tuple(
            _mapping(row) for row in observation.vehicle.get("active_tasks", ())
        )
        if int(observation.vehicle.get("active_task_count", len(raw_tasks))) != len(
            raw_tasks
        ):
            return incomplete("LIVE_ACTIVE_TASK_COUNT_MISMATCH")
        owned_memory = sum(
            max(0, int(row.get("memory_reservation_bytes", 0))) for row in raw_tasks
        )
        owned_tokens: dict[str, int] = {
            name: 0
            for name in scheduler.descriptor_capacity.get(observation.vehicle_id, {})
        }
        for row in raw_tasks:
            for name, value in _mapping(row.get("reservation_tokens")).items():
                owned_tokens[str(name)] = owned_tokens.get(str(name), 0) + max(
                    0, int(value)
                )
        if owned_memory != scheduler.vehicle_memory_reserved.get(
            observation.vehicle_id, 0
        ) or owned_tokens != scheduler.descriptor_reserved.get(
            observation.vehicle_id, {}
        ):
            return incomplete("LIVE_RESERVATION_OWNERSHIP_MISMATCH")
        other_rows = tuple(row for row in raw_tasks if not bool(row.get("is_focal")))
        tokens = [str(row.get("task_token", "")) for row in other_rows]
        if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
            return incomplete("LIVE_ACTIVE_TASK_TOKEN_INVALID")

        job_rows: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for resource in ("accelerator", "cpu", "encoder"):
            resource_row = _mapping(
                _mapping(observation.vehicle.get("resources")).get(resource)
            )
            active_jobs = tuple(
                _mapping(row) for row in resource_row.get("active_jobs", ())
            )
            if len(active_jobs) != int(resource_row.get("running_count", 0)) + int(
                resource_row.get("waiting_count", 0)
            ):
                return incomplete("LIVE_JOB_COUNT_MISMATCH")
            for row in active_jobs:
                token = str(row.get("task_token", ""))
                if not token or token in job_rows or token not in tokens:
                    return incomplete("LIVE_JOB_ASSOCIATION_INVALID")
                job_rows[token] = (resource, row)

        transfer_rows: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for rsu_id, link in sorted(observation.links.items()):
            for row_raw in _mapping(link).get("active_transfers", ()):
                row = _mapping(row_raw)
                token = str(row.get("task_token", ""))
                if not token or token in transfer_rows or token not in tokens:
                    return incomplete("LIVE_TRANSFER_ASSOCIATION_INVALID")
                transfer_rows[token] = (str(rsu_id), row)

        source = self.scenario_source
        for row in sorted(other_rows, key=lambda item: str(item.get("task_token"))):
            token = str(row["task_token"])
            if token in scheduler.tasks:
                return incomplete("LIVE_TASK_TOKEN_COLLISION")
            deadline = _finite(row.get("deadline_offset_s"), -1.0)
            if deadline <= 1e-12:
                return incomplete("LIVE_TASK_DEADLINE_INVALID")
            quality_bins = tuple(
                str(value) for value in row.get("conformal_quality_bins", ())
            )
            probabilities = tuple(
                (str(name), float(value))
                for name, value in row.get("quality_probabilities", ())
            )
            if not quality_bins:
                return incomplete("LIVE_TASK_QUALITY_SUPPORT_MISSING")
            future = SimpleNamespace(
                task_token=token,
                arrival_offset_s=0.0,
                relative_deadline_s=deadline,
                vehicle_id=observation.vehicle_id,
                device_type=observation.device_type,
                context=None,
                quality_candidates=quality_bins,
                quality_probabilities=probabilities,
                ood=bool(row.get("ood", False)),
                quality_features=tuple(row.get("quality_features", ())),
                prep_work_s=0.0,
                prep_energy_j=0.0,
                prep_memory_bytes=0,
                prep_failed=False,
                local_rows=tuple(getattr(source, "local_rows", ())),
                anon_rows=tuple(getattr(source, "anon_rows", ())),
                edge_rows=tuple(getattr(source, "edge_rows", ())),
                complete_support=True,
                support_reason=None,
            )
            live_observation = replace(
                observation,
                task_id=token,
                arrival_time_s=0.0,
                absolute_deadline_s=deadline,
                slack_s=deadline,
                quality_features=tuple(
                    float(value) for value in row.get("quality_features", ())
                ),
                quality_probabilities=probabilities,
                conformal_quality_bins=quality_bins,
                ood=bool(row.get("ood", False)),
                selected_pipeline=row.get("selected_pipeline"),
                selected_local_model=row.get("selected_local_model"),
                selected_rsu=row.get("selected_rsu"),
                selected_edge_model=row.get("selected_edge_model"),
                attempt_started_count=max(0, int(row.get("attempt_started_count", 0))),
                max_attempts=max(0, int(row.get("max_attempts", 0))),
                artifact_token=row.get("artifact_token"),
                encoded_size_bytes=row.get("encoded_size_bytes"),
                device_context=str(
                    row.get("device_context", observation.device_context)
                ),
            )
            task = _BranchTask(
                token,
                observation.vehicle_id,
                0.0,
                deadline,
                future,
                live_observation,
                selected_pipeline=row.get("selected_pipeline"),
                artifact_key=row.get("artifact_token"),
                encoded_size_bytes=(
                    None
                    if row.get("encoded_size_bytes") is None
                    else int(row["encoded_size_bytes"])
                ),
                reservation_tokens={
                    str(name): max(0, int(value))
                    for name, value in _mapping(row.get("reservation_tokens")).items()
                },
                reserved_memory_bytes=max(
                    0, int(row.get("memory_reservation_bytes", 0))
                ),
            )
            task.latent_quality_bin = self._task_latent_quality(
                branch, token, live_observation
            )
            if task.latent_quality_bin is None:
                return incomplete("LIVE_TASK_LATENT_QUALITY_MISSING")

            state = str(row.get("state", ""))
            selected_local = row.get("selected_local_model")
            selected_pipeline = row.get("selected_pipeline")
            selected_rsu = row.get("selected_rsu")
            selected_edge = row.get("selected_edge_model")
            expected_job_operation: str | None = None
            completion_kind: str | None = None

            if state == TaskState.RAW_BUF.value:
                prep_rows = [
                    item
                    for item in getattr(source, "prep_rows", ())
                    if item.device_type == observation.device_type
                    and str(item.quality_bin) == task.latent_quality_bin
                    and self._context_matches(item, live_observation)
                ]
                prep = self._deterministic_scenario_row(
                    branch, token, "live-prep", prep_rows
                )
                if prep is None:
                    return incomplete("LIVE_PREP_PAIRING_MISSING")
                future.prep_work_s = float(prep.service_work_s)
                future.prep_energy_j = float(prep.dynamic_energy_j)
                future.prep_memory_bytes = int(prep.memory_bytes)
                future.prep_failed = bool(prep.failed)
                task.state = "PENDING"
                self._scheduler_push(scheduler, 0.0, 30, "ARRIVAL", token)
            elif state in {TaskState.PREP_WAIT.value, TaskState.PREP_RUN.value}:
                task.state = "LIVE_PREP"
                expected_job_operation = "PREP"
                completion_kind = "LIVE_PREP_DONE"
                prep_rows = [
                    item
                    for item in getattr(source, "prep_rows", ())
                    if item.device_type == observation.device_type
                    and str(item.quality_bin) == task.latent_quality_bin
                    and self._context_matches(item, live_observation)
                ]
                prep = self._deterministic_scenario_row(
                    branch, token, "live-prep", prep_rows
                )
                if prep is None:
                    return incomplete("LIVE_PREP_PAIRING_MISSING")
                future.prep_failed = bool(prep.failed)
            elif state == TaskState.RAW.value:
                task.state = "RAW"
            elif state == TaskState.READY.value:
                if not selected_pipeline or not task.encoded_size_bytes:
                    return incomplete("LIVE_READY_EVIDENCE_MISSING")
                task.state = "READY"
                task.edge_pairing_token = self._select_surrogate_pairing_token(
                    branch, task, live_observation
                )
                if task.edge_pairing_token is None:
                    return incomplete("LIVE_READY_SCENARIO_ARTIFACT_MISSING")
            elif state in {TaskState.LOCAL_WAIT.value, TaskState.LOCAL_RUN.value}:
                if not selected_local:
                    return incomplete("LIVE_LOCAL_MODEL_MISSING")
                local_rows = [
                    item
                    for item in getattr(source, "local_rows", ())
                    if item.model_id == selected_local
                    and item.device_type == observation.device_type
                    and str(item.quality_bin) == task.latent_quality_bin
                    and self._context_matches(item, live_observation)
                ]
                local = self._deterministic_scenario_row(
                    branch, token, "live-local", local_rows
                )
                if local is None:
                    return incomplete("LIVE_LOCAL_PAIRING_MISSING")
                task.state = "LIVE_LOCAL"
                task.last_action = Action.local(
                    ActionStage.READY if selected_pipeline else ActionStage.RAW,
                    str(selected_local),
                )
                task.last_outcome = self._row_outcome(local)
                expected_job_operation = "LOCAL_FER"
                completion_kind = "LIVE_LOCAL_DONE"
            elif state in {
                TaskState.ANON_WAIT.value,
                TaskState.ANON_RUN.value,
                TaskState.GUARD_WAIT.value,
                TaskState.GUARD_RUN.value,
                TaskState.ENCODE_WAIT.value,
                TaskState.ENCODE_RUN.value,
            }:
                if not selected_pipeline:
                    return incomplete("LIVE_PIPELINE_ID_MISSING")
                anon_rows = [
                    item
                    for item in getattr(source, "anon_rows", ())
                    if item.pipeline_id == selected_pipeline
                    and item.device_type == observation.device_type
                    and str(item.quality_bin) == task.latent_quality_bin
                    and self._context_matches(item, live_observation)
                ]
                anon = self._deterministic_scenario_row(
                    branch, token, "live-pipeline", anon_rows
                )
                if anon is None:
                    return incomplete("LIVE_PIPELINE_PAIRING_MISSING")
                outcome = self._row_outcome(anon)
                task.state = "LIVE_PIPELINE"
                task.last_action = Action.pipeline(str(selected_pipeline))
                task.last_outcome = outcome
                task.edge_outcome = outcome
                task.edge_pairing_token = outcome.artifact_key
                task.encoded_size_bytes = int(
                    outcome.values.get("encoded_size_bytes", 0)
                )
                operation = (
                    "ANON"
                    if state in {TaskState.ANON_WAIT.value, TaskState.ANON_RUN.value}
                    else "GUARD"
                    if state in {TaskState.GUARD_WAIT.value, TaskState.GUARD_RUN.value}
                    else "ENCODE"
                )
                expected_job_operation = operation
                completion_kind = "LIVE_PIPE_STAGE_DONE"
                attempt_cursor = max(0, int(row.get("attempt_started_count", 1)) - 1)
                remaining: list[tuple[str, float, float, str]] = []
                for attempt_index, attempt in enumerate(anon.attempts):
                    stages = (
                        (
                            "ANON",
                            "accelerator",
                            attempt.anon_work_s,
                            attempt.anon_energy_j,
                        ),
                        ("GUARD", "cpu", attempt.guard_work_s, attempt.guard_energy_j),
                        (
                            "ENCODE",
                            "encoder",
                            attempt.encode_work_s,
                            attempt.encode_energy_j,
                        ),
                    )
                    for stage_name, resource, work, energy in stages:
                        if work is None or float(work) <= 0:
                            continue
                        if attempt_index < attempt_cursor or (
                            attempt_index == attempt_cursor
                            and ("ANON", "GUARD", "ENCODE").index(stage_name)
                            <= ("ANON", "GUARD", "ENCODE").index(operation)
                        ):
                            continue
                        remaining.append(
                            (resource, float(work), float(energy or 0.0), "PIPE_STAGE")
                        )
                task.pending_stages = remaining
            elif state in {TaskState.UL.value, TaskState.DL.value}:
                if not (selected_pipeline and selected_rsu and selected_edge):
                    return incomplete("LIVE_EDGE_SELECTION_MISSING")
                action = Action.edge(str(selected_rsu), str(selected_edge))
                task.edge_pairing_token = self._select_surrogate_pairing_token(
                    branch, task, live_observation
                )
                actual = self._actual_edge_rows(
                    future,
                    action,
                    self._scheduler_execution_observation(
                        scheduler, action, live_observation
                    ),
                    task.edge_pairing_token,
                    quality_bin=task.latent_quality_bin,
                )
                edge = self._deterministic_scenario_row(
                    branch, token, "live-edge", actual
                )
                if edge is None:
                    return incomplete("LIVE_EDGE_PAIRING_MISSING")
                task.state = state
                task.last_action = action
                task.last_outcome = self._row_outcome(edge)
                task.edge_outcome = task.last_outcome
            elif state in {TaskState.EDGE_WAIT.value, TaskState.EDGE_RUN.value}:
                if not (selected_pipeline and selected_rsu and selected_edge):
                    return incomplete("LIVE_EDGE_SELECTION_MISSING")
                action = Action.edge(str(selected_rsu), str(selected_edge))
                task.edge_pairing_token = self._select_surrogate_pairing_token(
                    branch, task, live_observation
                )
                execution = self._scheduler_execution_observation(
                    scheduler, action, live_observation
                )
                actual = self._actual_edge_rows(
                    future,
                    action,
                    execution,
                    task.edge_pairing_token,
                    quality_bin=task.latent_quality_bin,
                )
                edge = self._deterministic_scenario_row(
                    branch, token, "live-edge", actual
                )
                continuation = _mapping(row.get("rsu_continuation"))
                reservation = _mapping(continuation.get("reservation"))
                rsu_job = _mapping(continuation.get("job"))
                if edge is None or not reservation or not rsu_job:
                    return incomplete("LIVE_RSU_CONTINUATION_MISSING")
                task.last_action = action
                task.last_outcome = self._row_outcome(edge)
                task.edge_outcome = task.last_outcome
                task.admitted_rsu = str(selected_rsu)
                task.reserved_vram_bytes = int(reservation.get("vram_bytes", 0))
                task.reserved_gpu_work_s = float(
                    reservation.get("conservative_work_gpu_s", 0.0)
                )
                task.admission_vram_upper_bytes = task.reserved_vram_bytes
                task.admission_gpu_work_upper_s = task.reserved_gpu_work_s
                rsu_resource = (
                    "ingress"
                    if str(rsu_job.get("operation")) == "RSU_INGRESS"
                    else "gpu"
                    if str(rsu_job.get("operation")) == "EDGE_FER"
                    else ""
                )
                if not rsu_resource:
                    return incomplete("LIVE_RSU_JOB_OPERATION_INVALID")
                task.state = "INGRESS_WAIT" if rsu_resource == "ingress" else "GPU_WAIT"
                live_rsu = branch.rsus.get(str(selected_rsu))
                if live_rsu is None:
                    return incomplete("LIVE_RSU_MISSING")
                live_rsu["descriptors"] = int(live_rsu.get("descriptors", 0)) + int(
                    reservation.get("descriptor_count", 0)
                )
                live_rsu["vram_bytes"] = (
                    int(live_rsu.get("vram_bytes", 0)) + task.reserved_vram_bytes
                )
                live_rsu["reserved_work_gpu_s"] = (
                    _finite(live_rsu.get("reserved_work_gpu_s"), 0.0)
                    + task.reserved_gpu_work_s
                )
                if (
                    int(live_rsu["descriptors"])
                    > int(live_rsu.get("descriptor_capacity", 0))
                    or int(live_rsu["vram_bytes"])
                    > int(live_rsu.get("vram_capacity_bytes", 0))
                    or float(live_rsu["reserved_work_gpu_s"])
                    > _finite(live_rsu.get("workload_capacity_gpu_s"), 0.0) + 1e-12
                ):
                    return incomplete("LIVE_RSU_RESERVATION_OVER_CAPACITY")
                scheduler.sequence += 1
                job_id = f"branch-live-rsu:{scheduler.sequence:08d}"
                residual = _finite(rsu_job.get("residual_work_s"), -1.0)
                remaining_energy = _finite(
                    rsu_job.get("remaining_nominal_dynamic_energy_j"), -1.0
                )
                if residual <= 0 or remaining_energy < 0:
                    return incomplete("LIVE_RSU_JOB_RESIDUAL_INVALID")
                scheduler.jobs[job_id] = _BranchJob(
                    job_id,
                    token,
                    "rsu",
                    str(selected_rsu),
                    rsu_resource,
                    residual,
                    residual,
                    remaining_energy,
                    deadline,
                    int(rsu_job.get("enqueue_seq", scheduler.sequence)),
                    "INGRESS_DONE" if rsu_resource == "ingress" else "GPU_DONE",
                )
                pool = scheduler.resources[("rsu", str(selected_rsu), rsu_resource)]
                (
                    pool.running if rsu_job.get("status") == "RUNNING" else pool.waiting
                ).append(job_id)
            else:
                return incomplete("LIVE_TASK_STATE_UNSUPPORTED")

            pending = _mapping(row.get("pending_decision"))
            if pending:
                if state not in {TaskState.RAW.value, TaskState.READY.value}:
                    return incomplete("LIVE_PENDING_DECISION_STATE_INVALID")
                if pending.get("controller_energy_already_charged") is not True:
                    return incomplete("LIVE_PENDING_DECISION_ENERGY_AMBIGUOUS")
                proposed = self._snapshot_action(_mapping(pending.get("proposed")))
                expected_stage = (
                    ActionStage.RAW
                    if state == TaskState.RAW.value
                    else ActionStage.READY
                )
                remaining_overhead = _finite(pending.get("remaining_overhead_s"), -1.0)
                if proposed.stage is not expected_stage or remaining_overhead < 0:
                    return incomplete("LIVE_PENDING_DECISION_INVALID")
                task.pending_action = proposed
                task.pending_stage = expected_stage
                task.pending_sample_seed = self._conditional_decision_seed(
                    branch, token, expected_stage, proposed, 0.0
                )
                task.state = "DECISION_WAIT"
                self._scheduler_push(
                    scheduler,
                    remaining_overhead,
                    40,
                    "DECISION_COMMIT",
                    token,
                )

            scheduler.tasks[token] = task
            self._scheduler_push(scheduler, deadline, 20, "DEADLINE", token)

            if expected_job_operation is not None:
                associated = job_rows.get(token)
                if associated is None:
                    return incomplete("LIVE_VEHICLE_JOB_MISSING")
                resource, active_job = associated
                if str(active_job.get("operation")) != expected_job_operation:
                    return incomplete("LIVE_VEHICLE_JOB_OPERATION_MISMATCH")
                residual = _finite(active_job.get("residual_work_s"), -1.0)
                remaining_energy = _finite(
                    active_job.get("remaining_nominal_dynamic_energy_j"), -1.0
                )
                status = str(active_job.get("status", ""))
                if (
                    residual <= 0
                    or remaining_energy < 0
                    or status
                    not in {
                        "WAITING",
                        "RUNNING",
                    }
                ):
                    return incomplete("LIVE_VEHICLE_JOB_RESIDUAL_INVALID")
                scheduler.sequence += 1
                job_id = f"branch-live-vehicle:{scheduler.sequence:08d}"
                scheduler.jobs[job_id] = _BranchJob(
                    job_id,
                    token,
                    "vehicle",
                    observation.vehicle_id,
                    resource,
                    residual,
                    residual,
                    remaining_energy,
                    deadline,
                    int(active_job.get("enqueue_seq", scheduler.sequence)),
                    str(completion_kind),
                )
                pool = scheduler.resources[
                    ("vehicle", observation.vehicle_id, resource)
                ]
                (pool.running if status == "RUNNING" else pool.waiting).append(job_id)

            if state in {TaskState.UL.value, TaskState.DL.value}:
                associated = transfer_rows.get(token)
                if associated is None:
                    return incomplete("LIVE_TRANSFER_MISSING")
                rsu_id, active_transfer = associated
                direction = TransferDirection(str(active_transfer.get("direction")))
                if direction.value != state or rsu_id != str(selected_rsu):
                    return incomplete("LIVE_TRANSFER_STATE_MISMATCH")
                remaining_bits = _finite(active_transfer.get("remaining_bits"), -1.0)
                if remaining_bits <= 0:
                    return incomplete("LIVE_TRANSFER_BITS_INVALID")
                scheduler.sequence += 1
                transfer_id = f"branch-live-transfer:{scheduler.sequence:08d}"
                pause_age = active_transfer.get("pause_age_s")
                scheduler.transfers[transfer_id] = _BranchTransfer(
                    transfer_id,
                    token,
                    observation.vehicle_id,
                    rsu_id,
                    direction,
                    remaining_bits,
                    paused_since_s=(
                        None if pause_age is None else -max(0.0, float(pause_age))
                    ),
                )

        expected_job_tokens = {
            token
            for token, row in zip(tokens, other_rows, strict=True)
            if str(row.get("state"))
            in {
                TaskState.PREP_WAIT.value,
                TaskState.PREP_RUN.value,
                TaskState.LOCAL_WAIT.value,
                TaskState.LOCAL_RUN.value,
                TaskState.ANON_WAIT.value,
                TaskState.ANON_RUN.value,
                TaskState.GUARD_WAIT.value,
                TaskState.GUARD_RUN.value,
                TaskState.ENCODE_WAIT.value,
                TaskState.ENCODE_RUN.value,
            }
        }
        if set(job_rows) != expected_job_tokens:
            return incomplete("LIVE_VEHICLE_JOB_SET_MISMATCH")
        expected_transfer_tokens = {
            str(row["task_token"])
            for row in other_rows
            if str(row.get("state")) in {TaskState.UL.value, TaskState.DL.value}
        }
        if set(transfer_rows) != expected_transfer_tokens:
            return incomplete("LIVE_TRANSFER_SET_MISMATCH")
        for pool in scheduler.resources.values():
            if len(pool.running) > pool.server_count:
                return incomplete("LIVE_RESOURCE_CONCURRENCY_INVALID")
        return True

    @staticmethod
    def _scheduler_push(
        scheduler: _BranchScheduler,
        time_s: float,
        priority: int,
        kind: str,
        object_id: str,
    ) -> None:
        scheduler.sequence += 1
        heapq.heappush(
            scheduler.events,
            (float(time_s), priority, scheduler.sequence, kind, object_id),
        )

    @staticmethod
    def _scheduler_residual(
        scheduler: _BranchScheduler, owner_type: str, owner_id: str, resource: str
    ) -> float:
        pool = scheduler.resources[(owner_type, owner_id, resource)]
        return sum(
            scheduler.jobs[job_id].remaining_work_s
            for job_id in (*pool.running, *pool.waiting)
            if job_id in scheduler.jobs
        )

    def _scheduler_sync_branch(
        self, scheduler: _BranchScheduler, vehicle_id: str
    ) -> None:
        branch = scheduler.branch
        branch.vehicle_queues = {
            resource: self._scheduler_residual(
                scheduler, "vehicle", vehicle_id, resource
            )
            for resource in ("accelerator", "cpu", "encoder")
        }
        branch.battery_j = scheduler.vehicle_battery_j[vehicle_id]
        for rsu_id, row in branch.rsus.items():
            for resource in ("ingress", "gpu"):
                pool = scheduler.resources[("rsu", rsu_id, resource)]
                row[f"{resource}_residual_work_s"] = self._scheduler_residual(
                    scheduler, "rsu", rsu_id, resource
                )
                row[f"{resource}_waiting"] = len(pool.waiting)
                row[f"{resource}_running"] = len(pool.running)
        branch.scheduler_physical_lyapunov = sum(
            self._theta(owner_type, resource)
            * self._scheduler_residual(scheduler, owner_type, owner_id, resource) ** 2
            for owner_type, owner_id, resource in sorted(scheduler.resources)
        )

    def _scheduler_observation(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        stage: ActionStage,
    ) -> Observation:
        self._scheduler_sync_branch(scheduler, task.vehicle_id)
        branch = scheduler.branch
        branch.slack_s = max(0.0, task.deadline_s - branch.elapsed_s)
        vehicle = thaw_json(scheduler.vehicle_observation_rows[task.vehicle_id])
        vehicle["battery_j"] = scheduler.vehicle_battery_j[task.vehicle_id]
        vehicle["memory_capacity_bytes"] = scheduler.vehicle_memory_capacity[
            task.vehicle_id
        ]
        vehicle["memory_reserved_bytes"] = scheduler.vehicle_memory_reserved[
            task.vehicle_id
        ]
        vehicle["memory_remaining_bytes"] = max(
            0,
            scheduler.vehicle_memory_capacity[task.vehicle_id]
            - scheduler.vehicle_memory_reserved[task.vehicle_id],
        )
        vehicle["descriptor_capacity"] = dict(
            scheduler.descriptor_capacity[task.vehicle_id]
        )
        vehicle["descriptors_reserved"] = dict(
            scheduler.descriptor_reserved[task.vehicle_id]
        )
        vehicle["descriptor_remaining"] = {
            resource: max(
                0,
                scheduler.descriptor_capacity[task.vehicle_id].get(resource, 0)
                - scheduler.descriptor_reserved[task.vehicle_id].get(resource, 0),
            )
            for resource in scheduler.descriptor_capacity[task.vehicle_id]
        }
        template_resources = _mapping(vehicle.get("resources"))
        vehicle["resources"] = {
            resource: {
                **thaw_json(_mapping(template_resources.get(resource))),
                "server_count": scheduler.resources[
                    ("vehicle", task.vehicle_id, resource)
                ].server_count,
                "waiting_count": len(
                    scheduler.resources[("vehicle", task.vehicle_id, resource)].waiting
                ),
                "running_count": len(
                    scheduler.resources[("vehicle", task.vehicle_id, resource)].running
                ),
                "residual_work_s": self._scheduler_residual(
                    scheduler, "vehicle", task.vehicle_id, resource
                ),
            }
            for resource in ("accelerator", "cpu", "encoder")
        }
        observation = self._future_observation(
            task.future,
            task.base_observation,
            branch,
            stage=stage,
            selected_pipeline=task.selected_pipeline,
            artifact_key=task.artifact_key,
            encoded_size_bytes=task.encoded_size_bytes,
            vehicle_state=vehicle,
        )
        return replace(
            observation,
            device_context=self._scheduler_live_device_context(
                scheduler.branch, "vehicle", task.vehicle_id
            ),
        )

    @staticmethod
    def _scheduler_live_device_context(
        branch: _PredictionState,
        owner_type: str,
        owner_id: str,
    ) -> str:
        row = ESLSMPCPolicy._branch_thermal_segment(branch, owner_type, owner_id, "all")
        state = "nominal" if row is None else row.state
        return f"{state}|nominal|normal"

    def _scheduler_execution_observation(
        self,
        scheduler: _BranchScheduler,
        action: Action,
        public_observation: Observation,
    ) -> Observation:
        """Attach live execution context without changing public mask inputs."""

        if action.kind is not ActionKind.EDGE or not action.rsu_id:
            return public_observation
        rsus = thaw_json(public_observation.rsus)
        row = dict(_mapping(rsus.get(action.rsu_id)))
        row["device_context"] = self._scheduler_live_device_context(
            scheduler.branch, "rsu", action.rsu_id
        )
        rsus[action.rsu_id] = row
        return replace(public_observation, rsus=deep_freeze(rsus))

    def _scheduler_fallback_observation(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
    ) -> Observation:
        """Expose task-owned shadow capacity as reusable by its local fallback."""

        observation = self._scheduler_observation(scheduler, task, ActionStage.READY)
        vehicle = thaw_json(observation.vehicle)
        capacity = scheduler.vehicle_memory_capacity[task.vehicle_id]
        vehicle["memory_reserved_bytes"] = max(
            0,
            scheduler.vehicle_memory_reserved[task.vehicle_id]
            - task.reserved_memory_bytes,
        )
        vehicle["memory_remaining_bytes"] = min(
            capacity,
            max(0, capacity - scheduler.vehicle_memory_reserved[task.vehicle_id])
            + task.reserved_memory_bytes,
        )
        reserved = dict(_mapping(vehicle.get("descriptors_reserved")))
        remaining = dict(_mapping(vehicle.get("descriptor_remaining")))
        for resource, count in task.reservation_tokens.items():
            reserved[resource] = max(0, int(reserved.get(resource, 0)) - count)
            remaining[resource] = min(
                scheduler.descriptor_capacity[task.vehicle_id].get(resource, 0),
                int(remaining.get(resource, 0)) + count,
            )
        vehicle["descriptors_reserved"] = reserved
        vehicle["descriptor_remaining"] = remaining
        return replace(observation, vehicle=deep_freeze(vehicle))

    def _scheduler_reserve_vehicle(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        tokens: Mapping[str, int],
        memory_bytes: int,
    ) -> bool:
        if (
            scheduler.vehicle_memory_reserved[task.vehicle_id] + memory_bytes
            > scheduler.vehicle_memory_capacity[task.vehicle_id]
        ):
            return False
        if any(
            scheduler.descriptor_reserved[task.vehicle_id].get(name, 0) + count
            > scheduler.descriptor_capacity[task.vehicle_id].get(name, 0)
            for name, count in tokens.items()
        ):
            return False
        scheduler.vehicle_memory_reserved[task.vehicle_id] += memory_bytes
        for name, count in tokens.items():
            scheduler.descriptor_reserved[task.vehicle_id][name] = (
                scheduler.descriptor_reserved[task.vehicle_id].get(name, 0) + count
            )
        task.reservation_tokens = dict(tokens)
        task.reserved_memory_bytes = memory_bytes
        return True

    @staticmethod
    def _scheduler_release_vehicle(
        scheduler: _BranchScheduler, task: _BranchTask
    ) -> None:
        scheduler.vehicle_memory_reserved[task.vehicle_id] = max(
            0,
            scheduler.vehicle_memory_reserved[task.vehicle_id]
            - task.reserved_memory_bytes,
        )
        for name, count in task.reservation_tokens.items():
            scheduler.descriptor_reserved[task.vehicle_id][name] = max(
                0,
                scheduler.descriptor_reserved[task.vehicle_id].get(name, 0) - count,
            )
        task.reservation_tokens.clear()
        task.reserved_memory_bytes = 0

    def _scheduler_enqueue_job(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        *,
        owner_type: str,
        owner_id: str,
        resource: str,
        work_s: float,
        energy_j: float,
        completion_kind: str,
    ) -> None:
        scheduler.sequence += 1
        job_id = f"branch-job:{scheduler.sequence:08d}"
        scheduler.jobs[job_id] = _BranchJob(
            job_id,
            task.task_token,
            owner_type,
            owner_id,
            resource,
            max(_EPS_DURATION_S, work_s),
            max(_EPS_DURATION_S, work_s),
            max(0.0, energy_j),
            task.deadline_s,
            scheduler.sequence,
            completion_kind,
        )
        scheduler.resources[(owner_type, owner_id, resource)].waiting.append(job_id)

    def _scheduler_enqueue_model_maintenance(
        self,
        scheduler: _BranchScheduler,
        event: Any,
    ) -> None:
        """Queue one taskless RSU GPU maintenance job in an isolated branch."""

        work_s = _finite(getattr(event, "maintenance_work_s", None), 0.0)
        energy_j = _finite(getattr(event, "maintenance_energy_j", None), 0.0)
        rsu_id = str(getattr(event, "target_id", ""))
        model_id = str(getattr(event, "model_id", "") or "")
        if (
            rsu_id not in scheduler.branch.rsus
            or not model_id
            or work_s <= 0
            or energy_j <= 0
        ):
            scheduler.branch.complete_macro_recourse = False
            scheduler.branch.incomplete_reason = (
                "SCENARIO_MODEL_MAINTENANCE_PHYSICS_MISSING"
            )
            return
        scheduler.sequence += 1
        job_id = f"branch-maintenance:{scheduler.sequence:08d}"
        duration_s = (
            scheduler.branch.elapsed_s + work_s
            if scheduler.branch.environment is None
            else scheduler.branch.environment.duration_s
        )
        scheduler.jobs[job_id] = _BranchJob(
            job_id=job_id,
            task_token="",
            owner_type="rsu",
            owner_id=rsu_id,
            resource="gpu",
            remaining_work_s=work_s,
            total_work_s=work_s,
            total_energy_j=energy_j,
            absolute_deadline_s=max(scheduler.branch.elapsed_s, duration_s),
            enqueue_seq=scheduler.sequence,
            completion_kind="MODEL_MAINTENANCE_DONE",
            maintenance_event=event,
        )
        scheduler.resources[("rsu", rsu_id, "gpu")].waiting.append(job_id)
        key = (rsu_id, model_id)
        scheduler.maintenance_job_keys[job_id] = key
        if key in scheduler.maintenance_active:
            scheduler.maintenance_waiting.setdefault(key, []).append(job_id)
        else:
            scheduler.maintenance_active[key] = job_id
        scheduler.event_trace.append(
            {
                "time_s": scheduler.branch.elapsed_s,
                "kind": "MODEL_MAINTENANCE_ENQUEUE",
                "job_id": job_id,
                "rsu_id": rsu_id,
                "model_id": model_id,
                "work_s": work_s,
                "energy_j": energy_j,
            }
        )

    @staticmethod
    def _scheduler_maintenance_dispatchable(
        scheduler: _BranchScheduler, job_id: str
    ) -> bool:
        job = scheduler.jobs[job_id]
        if job.completion_kind != "MODEL_MAINTENANCE_DONE":
            return True
        key = scheduler.maintenance_job_keys.get(job_id)
        return key is not None and scheduler.maintenance_active.get(key) == job_id

    @staticmethod
    def _scheduler_release_next_maintenance(
        scheduler: _BranchScheduler, job_id: str
    ) -> bool:
        key = scheduler.maintenance_job_keys.pop(job_id, None)
        if key is None or scheduler.maintenance_active.get(key) != job_id:
            scheduler.branch.complete_macro_recourse = False
            scheduler.branch.incomplete_reason = (
                "SCENARIO_MODEL_MAINTENANCE_CHAIN_CORRUPT"
            )
            return False
        waiting = scheduler.maintenance_waiting.get(key, [])
        if waiting:
            scheduler.maintenance_active[key] = waiting.pop(0)
            if not waiting:
                scheduler.maintenance_waiting.pop(key, None)
        else:
            scheduler.maintenance_active.pop(key, None)
        return True

    def _scheduler_remaining_job_energy_upper(
        self,
        scheduler: _BranchScheduler,
        job: _BranchJob,
    ) -> float:
        return max(
            0.0,
            job.total_energy_j * job.remaining_work_s / job.total_work_s,
        )

    def _scheduler_dispatch(
        self,
        scheduler: _BranchScheduler,
        rng: random.Random | None = None,
    ) -> None:
        fallback_rng = rng or random.Random(0)
        for key in sorted(scheduler.resources):
            pool = scheduler.resources[key]
            while len(pool.running) < pool.server_count and pool.waiting:
                eligible = [
                    job_id
                    for job_id in pool.waiting
                    if self._scheduler_maintenance_dispatchable(scheduler, job_id)
                ]
                if not eligible:
                    break
                job_id = min(
                    eligible,
                    key=lambda job_id: (
                        scheduler.jobs[job_id].absolute_deadline_s,
                        scheduler.jobs[job_id].owner_id,
                        scheduler.jobs[job_id].task_token,
                        scheduler.jobs[job_id].enqueue_seq,
                    ),
                )
                pool.waiting.remove(job_id)
                job = scheduler.jobs[job_id]
                if job.owner_type == "vehicle" and job.task_token:
                    required = self._scheduler_remaining_job_energy_upper(
                        scheduler, job
                    )
                    unavailable = self._faulted(
                        scheduler.branch,
                        "vehicle",
                        job.owner_id,
                        "all",
                    ) or (
                        required
                        > scheduler.vehicle_battery_j.get(job.owner_id, 0.0) + 1e-9
                    )
                    if unavailable:
                        scheduler.jobs.pop(job_id, None)
                        task = scheduler.tasks[job.task_token]
                        scheduler.event_trace.append(
                            {
                                "time_s": scheduler.branch.elapsed_s,
                                "kind": "DISPATCH_BATTERY_GUARD",
                                "task_token": task.task_token,
                                "resource": job.resource,
                                "remaining_energy_upper_j": required,
                                "battery_j": scheduler.vehicle_battery_j.get(
                                    job.owner_id, 0.0
                                ),
                            }
                        )
                        if job.completion_kind in {"PREP_DONE", "LOCAL_DONE"}:
                            self._scheduler_finish(
                                scheduler,
                                task,
                                success=False,
                                reason="DISPATCH_BATTERY_GUARD",
                            )
                        else:
                            self._scheduler_technical_fallback(
                                scheduler,
                                task,
                                fallback_rng,
                                reason="DISPATCH_BATTERY_GUARD",
                            )
                        continue
                pool.running.append(job_id)
                scheduler.event_trace.append(
                    {
                        "time_s": scheduler.branch.elapsed_s,
                        "kind": "JOB_START",
                        "job_id": job_id,
                        "task_token": job.task_token,
                        "owner_type": job.owner_type,
                        "owner_id": job.owner_id,
                        "resource": job.resource,
                        "deadline_s": job.absolute_deadline_s,
                    }
                )

    def _scheduler_schedule_decision(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        action: Action,
        observation: Observation,
        sample_seed: int,
    ) -> None:
        """Charge and delay every policy decision before physical commit.

        Only the proposal and a deterministic sampling substream are frozen at
        the decision epoch.  No outcome is sampled and no job, reservation,
        transfer or terminal result is created until ``DECISION_COMMIT``.  The
        commit event rebuilds the public observation, repairs the stale
        proposal, then samples the final action in the live device context.
        """

        if task.state not in {"RAW", "READY"}:
            return
        config = self.mask_engine.config
        controller_energy_j = (
            0.0 if config is None else config.controller.controller_energy_j
        )
        if (
            controller_energy_j
            > scheduler.vehicle_battery_j.get(task.vehicle_id, 0.0) + 1e-9
        ):
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason="CONTROLLER_ENERGY_GUARD",
            )
            return
        if controller_energy_j > 0 and not self._scheduler_spend_vehicle(
            scheduler, task.vehicle_id, controller_energy_j
        ):
            # The explicit precheck above makes this branch unreachable unless
            # the isolated scheduler's accounting is internally inconsistent.
            raise RuntimeError("controller energy changed during branch decision")

        stage = observation.stage
        task.pending_action = action
        task.pending_stage = stage
        task.pending_sample_seed = int(sample_seed)
        task.state = "DECISION_WAIT"
        overhead_s = 0.0 if config is None else config.controller.controller_overhead_s
        self._scheduler_push(
            scheduler,
            scheduler.branch.elapsed_s + max(0.0, overhead_s),
            40,
            "DECISION_COMMIT",
            task.task_token,
        )
        scheduler.event_trace.append(
            {
                "time_s": scheduler.branch.elapsed_s,
                "kind": "DECISION_START",
                "task_token": task.task_token,
                "stage": stage.value,
                "proposed": action.canonical_id,
                "controller_overhead_s": max(0.0, overhead_s),
                "controller_energy_j": max(0.0, controller_energy_j),
            }
        )

    def _scheduler_handle_decision_commit(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
    ) -> None:
        """Revalidate a delayed proposal against the then-current public state."""

        proposed = task.pending_action
        stage = task.pending_stage
        sample_seed = task.pending_sample_seed
        if (
            task.state != "DECISION_WAIT"
            or proposed is None
            or stage is None
            or sample_seed is None
        ):
            return

        observation = self._scheduler_observation(scheduler, task, stage)
        repair = self.repairer.repair(
            proposed,
            self._future_task_record(task.future, observation),
            observation,
        )
        action = repair.executed
        task.pending_action = None
        task.pending_stage = None
        task.pending_sample_seed = None

        scheduler.event_trace.append(
            {
                "time_s": scheduler.branch.elapsed_s,
                "kind": "DECISION_COMMIT",
                "task_token": task.task_token,
                "stage": stage.value,
                "proposed": proposed.canonical_id,
                "executed": action.canonical_id,
                "changed": repair.changed,
                "proposed_reason_codes": [
                    reason.value for reason in repair.proposed_reasons
                ],
            }
        )

        execution_observation = self._scheduler_execution_observation(
            scheduler, action, observation
        )
        if action.kind is ActionKind.EDGE and self._faulted(
            scheduler.branch, "rsu", action.rsu_id or "", "all"
        ):
            self._scheduler_technical_fallback(
                scheduler,
                task,
                random.Random(sample_seed),
                reason="RSU_DEVICE_FAULT_AT_DECISION_COMMIT",
            )
            return
        if task.latent_quality_bin is None:
            task.latent_quality_bin = self._task_latent_quality(
                scheduler.branch, task.task_token, execution_observation
            )
        if task.latent_quality_bin is None:
            scheduler.branch.complete_macro_recourse = False
            scheduler.branch.incomplete_reason = "QUALITY_SCENARIO_CELL_MISSING"
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason="DECISION_COMMIT_UNSUPPORTED",
            )
            return
        sampling_observation = self._latent_sampling_observation(
            task, execution_observation
        )
        try:
            sample_rng = random.Random(sample_seed)
            if action.kind in {ActionKind.LOCAL, ActionKind.PIPE}:
                name = "local_rows" if action.kind is ActionKind.LOCAL else "anon_rows"
                candidates = [
                    row
                    for row in self._scenario_rows_for(
                        task.future, self.scenario_source, name
                    )
                    if (
                        row.model_id == action.local_model_id
                        if action.kind is ActionKind.LOCAL
                        else row.pipeline_id == action.pipeline_id
                    )
                    and row.device_type == execution_observation.device_type
                    and row.quality_bin in execution_observation.conformal_quality_bins
                    and self._context_matches(row, execution_observation)
                ]
                if {str(row.quality_bin) for row in candidates} != set(
                    execution_observation.conformal_quality_bins
                ):
                    raise LookupError("commit lacks all quality-cell pairings")
                outcome = self._future_outcome(
                    task.future,
                    action,
                    sampling_observation,
                    sample_rng,
                )
            elif action.kind is ActionKind.EDGE:
                if task.edge_pairing_token is None:
                    task.edge_pairing_token = self._select_surrogate_pairing_token(
                        scheduler.branch, task, execution_observation
                    )
                actual_rows = self._actual_edge_rows(
                    task.future,
                    action,
                    execution_observation,
                    task.edge_pairing_token,
                    quality_bin=task.latent_quality_bin,
                )
                bounds = self._certified_edge_admission_bounds(
                    task.future, action, execution_observation
                )
                if not actual_rows or (bounds is None and not task.fixed_continuation):
                    raise LookupError(
                        "edge packet pairing or admission envelope missing"
                    )
                if bounds is not None:
                    task.admission_vram_upper_bytes = bounds[0]
                    task.admission_gpu_work_upper_s = bounds[1]
                outcome = self._future_outcome(
                    task.future,
                    action,
                    sampling_observation,
                    sample_rng,
                    pairing_token=task.edge_pairing_token,
                    quality_bin=task.latent_quality_bin,
                )
            else:
                outcome = self._row_outcome(
                    {
                        "scenario_id": f"{task.task_token}:explicit-fail",
                        "expected_duration_s": _EPS_DURATION_S,
                        "failure_probability": 1.0,
                        "completion_probability": 0.0,
                    }
                )
        except LookupError:
            scheduler.branch.complete_macro_recourse = False
            scheduler.branch.incomplete_reason = (
                "DECISION_COMMIT_ACTION_PAIRING_MISSING"
            )
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason="DECISION_COMMIT_UNSUPPORTED",
            )
            return
        self._scheduler_commit(scheduler, task, action, outcome, execution_observation)

    def _scheduler_commit(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        action: Action,
        outcome: _ScenarioOutcome,
        observation: Observation,
        *,
        failure_reason: str | None = None,
    ) -> None:
        branch = scheduler.branch
        task.last_action = action
        task.last_observation = observation
        task.last_outcome = outcome
        branch.joint_row_ids.append(outcome.row_id)
        if action.kind is ActionKind.FAIL:
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason=failure_reason or "EXPLICIT_FAIL_ACTION",
            )
            return
        if action.kind is ActionKind.LOCAL:
            realized_quality = str(outcome.values.get("quality_bin", ""))
            if realized_quality and realized_quality != task.latent_quality_bin:
                raise RuntimeError("local outcome changed the task latent quality cell")
            memory = max(0, int(outcome.values.get("vehicle_memory_upper_bytes", 0)))
            reservation_ok = True
            if task.reservation_tokens:
                additional = max(0, memory - task.reserved_memory_bytes)
                if (
                    scheduler.vehicle_memory_reserved[task.vehicle_id] + additional
                    > scheduler.vehicle_memory_capacity[task.vehicle_id]
                ):
                    reservation_ok = False
                else:
                    scheduler.vehicle_memory_reserved[task.vehicle_id] += additional
                    task.reserved_memory_bytes += additional
            else:
                reservation_ok = self._scheduler_reserve_vehicle(
                    scheduler, task, {"accelerator": 1}, memory
                )
            if not reservation_ok:
                self._scheduler_finish(
                    scheduler,
                    task,
                    success=False,
                    reason="LOCAL_RESOURCE_REJECTED",
                )
                return
            task.state = "LOCAL_WAIT"
            self._scheduler_enqueue_job(
                scheduler,
                task,
                owner_type="vehicle",
                owner_id=task.vehicle_id,
                resource="accelerator",
                work_s=max(0.0, _finite(outcome.values.get("vehicle_work_s"), 0.0)),
                energy_j=max(
                    0.0,
                    _finite(outcome.values.get("expected_vehicle_energy_j"), 0.0),
                ),
                completion_kind="LOCAL_DONE",
            )
            return
        if action.kind is ActionKind.PIPE:
            realized_quality = str(outcome.values.get("quality_bin", ""))
            if realized_quality and realized_quality != task.latent_quality_bin:
                raise RuntimeError(
                    "pipeline outcome changed the task latent quality cell"
                )
            bounds = action_estimate(
                action, observation, self.mask_engine.trace_support
            )
            memory = max(
                0,
                int(outcome.values.get("vehicle_memory_upper_bytes", 0)),
                int(max(0.0, _finite(bounds.get("vehicle_memory_upper_bytes"), 0.0))),
            )
            tokens = {"accelerator": 1, "cpu": 1, "encoder": 1}
            if not self._scheduler_reserve_vehicle(scheduler, task, tokens, memory):
                self._scheduler_finish(
                    scheduler,
                    task,
                    success=False,
                    reason="PIPELINE_RESOURCE_REJECTED",
                )
                return
            task.selected_pipeline = action.pipeline_id
            task.artifact_key = outcome.artifact_key
            task.edge_pairing_token = outcome.artifact_key
            task.encoded_size_bytes = int(outcome.values.get("encoded_size_bytes", 0))
            task.edge_outcome = outcome
            task.pending_stages = [
                (str(resource), float(work), float(energy), str(kind))
                for resource, work, energy, kind in outcome.values.get(
                    "pipeline_stage_sequence", ()
                )
            ]
            task.state = "PIPE_WAIT"
            self._scheduler_start_next_pipeline_stage(scheduler, task)
            return
        if (
            task.edge_pairing_token is None
            or outcome.artifact_key != task.edge_pairing_token
        ):
            raise RuntimeError("edge outcome is not paired with the uploaded packet")
        realized_quality = str(outcome.values.get("quality_bin", ""))
        if realized_quality and realized_quality != task.latent_quality_bin:
            raise RuntimeError("edge outcome changed the task latent quality cell")
        task.edge_outcome = outcome
        task.state = "UL"
        bits = max(
            1.0,
            float(task.encoded_size_bytes or 0) * 8
            + (
                0
                if self.mask_engine.config is None
                else self.mask_engine.config.metadata_bits
            ),
        )
        scheduler.sequence += 1
        transfer_id = f"branch-transfer:{scheduler.sequence:08d}"
        scheduler.transfers[transfer_id] = _BranchTransfer(
            transfer_id,
            task.task_token,
            task.vehicle_id,
            action.rsu_id or "",
            TransferDirection.UL,
            bits,
        )

    def _scheduler_start_next_pipeline_stage(
        self, scheduler: _BranchScheduler, task: _BranchTask
    ) -> None:
        if not task.pending_stages:
            formed = bool(task.edge_outcome and task.edge_outcome.formed_packet)
            # The frozen pipeline reservation is a shadow transaction held
            # through READY, UL, edge service and DL.  It is released only by
            # terminal cleanup, or reused by the frozen local fallback.
            task.state = "READY" if formed else "PIPE_REPAIR"
            return
        resource, work, energy, _ = task.pending_stages.pop(0)
        self._scheduler_enqueue_job(
            scheduler,
            task,
            owner_type="vehicle",
            owner_id=task.vehicle_id,
            resource=resource,
            work_s=work,
            energy_j=energy,
            completion_kind="PIPE_STAGE_DONE",
        )

    def _scheduler_start_fixed_control(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        next_step: str,
    ) -> None:
        """Start a preselected anchor decision's deterministic overhead."""

        config = self.mask_engine.config
        assert config is not None
        energy_j = max(0.0, config.controller.controller_energy_j)
        if energy_j > scheduler.vehicle_battery_j.get(task.vehicle_id, 0.0) + 1e-9:
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason="ANCHOR_CONTROLLER_ENERGY_GUARD",
            )
            return
        if energy_j and not self._scheduler_spend_vehicle(
            scheduler, task.vehicle_id, energy_j
        ):
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason="ANCHOR_CONTROLLER_BATTERY_DEPLETED",
            )
            return
        task.continuation_control_next = next_step
        task.state = "RAW_CONTROL" if next_step == "ACTION" else "READY_CONTROL"
        self._scheduler_push(
            scheduler,
            scheduler.branch.elapsed_s
            + max(0.0, config.controller.controller_overhead_s),
            40,
            "ANCHOR_CONTROL_DONE",
            task.task_token,
        )

    def _scheduler_start_next_fixed_stage(
        self, scheduler: _BranchScheduler, task: _BranchTask
    ) -> None:
        if task.continuation_stages:
            resource, work, energy, _ = task.continuation_stages.pop(0)
            task.state = "COMPUTE"
            self._scheduler_enqueue_job(
                scheduler,
                task,
                owner_type="vehicle",
                owner_id=task.vehicle_id,
                resource=resource,
                work_s=work,
                energy_j=energy,
                completion_kind="ANCHOR_COMPUTE_DONE",
            )
            return
        if task.continuation_path_kind == "local":
            failed = bool(
                task.last_outcome
                and _finite(task.last_outcome.values.get("failure_probability"), 0.0)
                >= 0.5
            )
            self._scheduler_finish(
                scheduler,
                task,
                success=not failed,
                reason=(
                    "ANCHOR_LOCAL_RESULT"
                    if not failed
                    else "ANCHOR_LOCAL_MODEL_FAILURE"
                ),
            )
            return
        self._scheduler_start_fixed_control(scheduler, task, "UPLINK")

    def _scheduler_complete_fixed_control(
        self, scheduler: _BranchScheduler, task: _BranchTask
    ) -> None:
        if task.state not in {"RAW_CONTROL", "READY_CONTROL"}:
            return
        next_step = task.continuation_control_next
        task.continuation_control_next = None
        if next_step == "ACTION":
            if not self._scheduler_reserve_vehicle(
                scheduler,
                task,
                task.continuation_action_tokens,
                task.continuation_action_memory_bytes,
            ):
                self._scheduler_finish(
                    scheduler,
                    task,
                    success=False,
                    reason="ANCHOR_ACTION_RESOURCE_REJECTED",
                )
                return
            self._scheduler_start_next_fixed_stage(scheduler, task)
            return
        if next_step == "UPLINK":
            anchor = getattr(task.future, "anchor", None)
            action = task.last_action
            if (
                anchor is None
                or action is None
                or action.kind is not ActionKind.EDGE
                or not action.rsu_id
            ):
                scheduler.branch.complete_macro_recourse = False
                scheduler.branch.incomplete_reason = (
                    "SCENARIO_ANCHOR_UPLINK_CONTINUATION_INVALID"
                )
                self._scheduler_finish(
                    scheduler,
                    task,
                    success=False,
                    reason="ANCHOR_UPLINK_UNSUPPORTED",
                )
                return
            task.state = "UL"
            scheduler.sequence += 1
            transfer_id = f"branch-anchor-transfer:{scheduler.sequence:08d}"
            scheduler.transfers[transfer_id] = _BranchTransfer(
                transfer_id,
                task.task_token,
                task.vehicle_id,
                action.rsu_id,
                TransferDirection.UL,
                max(1.0, float(getattr(anchor, "uplink_bits", 0.0))),
            )
            return
        scheduler.branch.complete_macro_recourse = False
        scheduler.branch.incomplete_reason = "SCENARIO_ANCHOR_CONTROL_TARGET_INVALID"
        self._scheduler_finish(
            scheduler,
            task,
            success=False,
            reason="ANCHOR_CONTROL_TARGET_INVALID",
        )

    def _scheduler_finish(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        *,
        success: bool,
        timeout: bool = False,
        reason: str,
    ) -> None:
        if task.state in {"DONE", "FAIL"}:
            return
        self._scheduler_cancel_activity(scheduler, task)
        task.state = "DONE" if success else "FAIL"
        if success:
            scheduler.pending_completed += 1
            fer_loss = max(
                0.0,
                _finite(
                    None
                    if task.last_outcome is None
                    else task.last_outcome.values.get("expected_fer_loss"),
                    0.0,
                ),
            )
            self._scheduler_add_normalized_cost(scheduler, "utility", fer_loss)
        else:
            scheduler.pending_failures += 1
            self._scheduler_add_normalized_cost(scheduler, "failure", 1.0)
            if timeout:
                scheduler.pending_timeouts += 1
                self._scheduler_add_normalized_cost(scheduler, "timeout", 1.0)
        scheduler.event_trace.append(
            {
                "time_s": scheduler.branch.elapsed_s,
                "kind": "TASK_TERMINAL",
                "task_token": task.task_token,
                "state": task.state,
                "reason": reason,
            }
        )

    def _scheduler_project_task_queues(self, scheduler: _BranchScheduler) -> None:
        """Apply one atomic long-term queue projection for a compound event."""

        arrivals = scheduler.pending_arrivals
        timeouts = scheduler.pending_timeouts
        failures = scheduler.pending_failures
        completed = scheduler.pending_completed
        if not any((arrivals, timeouts, failures, completed)):
            return
        config = self.mask_engine.config
        assert config is not None
        queues = scheduler.branch.virtual_queues
        queues["timeout"] = max(
            0.0,
            _finite(queues.get("timeout"), 0.0)
            + timeouts
            - config.long_term.timeout_rate_limit * arrivals,
        )
        queues["failure"] = max(
            0.0,
            _finite(queues.get("failure"), 0.0)
            + failures
            - config.long_term.failure_rate_limit * arrivals,
        )
        queues["coverage"] = max(
            0.0,
            _finite(queues.get("coverage"), 0.0)
            + config.long_term.coverage_rate_minimum * arrivals
            - completed,
        )
        scheduler.event_trace.append(
            {
                "time_s": scheduler.branch.elapsed_s,
                "kind": "VIRTUAL_QUEUE_PROJECTION",
                "arrivals": arrivals,
                "timeouts": timeouts,
                "failures": failures,
                "completed": completed,
                "timeout_queue": queues["timeout"],
                "failure_queue": queues["failure"],
                "coverage_queue": queues["coverage"],
            }
        )
        scheduler.pending_arrivals = 0
        scheduler.pending_timeouts = 0
        scheduler.pending_failures = 0
        scheduler.pending_completed = 0

    def _scheduler_cancel_activity(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        *,
        release_vehicle: bool = True,
    ) -> None:
        """Release a branch task's live jobs, packet and reservations."""

        task.pending_action = None
        task.pending_stage = None
        task.pending_sample_seed = None

        for pool in scheduler.resources.values():
            for job_id in tuple((*pool.running, *pool.waiting)):
                job = scheduler.jobs.get(job_id)
                if job is None or job.task_token != task.task_token:
                    continue
                if job_id in pool.running:
                    pool.running.remove(job_id)
                if job_id in pool.waiting:
                    pool.waiting.remove(job_id)
                scheduler.jobs.pop(job_id, None)
        for transfer_id, transfer in tuple(scheduler.transfers.items()):
            if transfer.task_token == task.task_token:
                scheduler.transfers.pop(transfer_id, None)
        if release_vehicle:
            self._scheduler_release_vehicle(scheduler, task)
        self._scheduler_release_rsu(scheduler, task)

    @staticmethod
    def _scheduler_release_rsu(scheduler: _BranchScheduler, task: _BranchTask) -> None:
        if task.admitted_rsu is not None:
            row = scheduler.branch.rsus[task.admitted_rsu]
            row["descriptors"] = max(0, int(row.get("descriptors", 0)) - 1)
            row["vram_bytes"] = max(
                0, int(row.get("vram_bytes", 0)) - task.reserved_vram_bytes
            )
            row["reserved_work_gpu_s"] = max(
                0.0,
                _finite(row.get("reserved_work_gpu_s"), 0.0) - task.reserved_gpu_work_s,
            )
            task.admitted_rsu = None
            task.reserved_vram_bytes = 0
            task.reserved_gpu_work_s = 0.0
        task.admission_vram_upper_bytes = 0
        task.admission_gpu_work_upper_s = 0.0

    def _scheduler_add_normalized_cost(
        self, scheduler: _BranchScheduler, kind: str, amount: float
    ) -> None:
        config = _cost_config(self.mask_engine)
        weights = {} if config is None else config.weights
        value = max(0.0, amount)
        if kind == "latency":
            scale = 1.0 if config is None else config.latency_scale_s
            weight = _weight(weights, "latency", "latency_weight", default=1.0)
        elif kind == "vehicle_energy":
            scale = 1.0 if config is None else config.vehicle_energy_scale_j
            weight = _weight(
                weights, "vehicle_energy", "vehicle_energy_weight", default=1.0
            )
        elif kind == "rsu_energy":
            scale = 1.0 if config is None else config.rsu_energy_scale_j
            weight = _weight(weights, "rsu_energy", "rsu_energy_weight", default=1.0)
        elif kind == "utility":
            scale = 1.0 if config is None else config.utility_scale
            weight = _weight(weights, "utility", "fer", "utility_weight", default=1.0)
        else:
            scale = 1.0
            loss = 1e6 if config is None else config.failure_loss
            value *= loss
            weight = _weight(weights, kind, f"{kind}_weight", default=1.0)
        scheduler.branch.cumulative_cost += weight * value / scale

    def _scheduler_spend_vehicle(
        self,
        scheduler: _BranchScheduler,
        vehicle_id: str,
        energy_j: float,
    ) -> bool:
        energy = max(0.0, energy_j)
        available = scheduler.vehicle_battery_j[vehicle_id]
        if energy > available + 1e-9:
            scheduler.branch.complete_macro_recourse = False
            scheduler.branch.incomplete_reason = "BRANCH_BATTERY_EXHAUSTED"
            return False
        scheduler.vehicle_battery_j[vehicle_id] = max(0.0, available - energy)
        scheduler.branch.vehicle_energy_j += energy
        scheduler.branch.vehicle_physical_energy_j[vehicle_id] = (
            scheduler.branch.vehicle_physical_energy_j.get(vehicle_id, 0.0) + energy
        )
        self._scheduler_add_normalized_cost(scheduler, "vehicle_energy", energy)
        queues = scheduler.branch.virtual_queues.setdefault("vehicle_power", {})
        if isinstance(queues, dict):
            queues[vehicle_id] = max(0.0, _finite(queues.get(vehicle_id), 0.0) + energy)
        return True

    def _scheduler_spend_vehicle_physical_only(
        self,
        scheduler: _BranchScheduler,
        vehicle_id: str,
        energy_j: float,
    ) -> bool:
        """Charge battery/physical power queue without task cost attribution."""

        energy = max(0.0, energy_j)
        available = scheduler.vehicle_battery_j[vehicle_id]
        if energy > available + 1e-9:
            scheduler.branch.complete_macro_recourse = False
            scheduler.branch.incomplete_reason = "BRANCH_BATTERY_EXHAUSTED"
            return False
        scheduler.vehicle_battery_j[vehicle_id] = max(0.0, available - energy)
        scheduler.branch.vehicle_physical_energy_j[vehicle_id] = (
            scheduler.branch.vehicle_physical_energy_j.get(vehicle_id, 0.0) + energy
        )
        queues = scheduler.branch.virtual_queues.setdefault("vehicle_power", {})
        if isinstance(queues, dict):
            queues[vehicle_id] = max(0.0, _finite(queues.get(vehicle_id), 0.0) + energy)
        return True

    def _scheduler_spend_rsu(
        self, scheduler: _BranchScheduler, rsu_id: str, energy_j: float
    ) -> None:
        energy = max(0.0, energy_j)
        self._spend_rsu(scheduler.branch, rsu_id, energy)
        self._scheduler_add_normalized_cost(scheduler, "rsu_energy", energy)

    @staticmethod
    def _scheduler_spend_rsu_physical_only(
        scheduler: _BranchScheduler, rsu_id: str, energy_j: float
    ) -> None:
        energy = max(0.0, energy_j)
        queues = scheduler.branch.virtual_queues.setdefault("rsu_power", {})
        scheduler.branch.rsu_physical_energy_j[rsu_id] = (
            scheduler.branch.rsu_physical_energy_j.get(rsu_id, 0.0) + energy
        )
        if isinstance(queues, dict):
            queues[rsu_id] = max(0.0, _finite(queues.get(rsu_id), 0.0) + energy)

    def _scheduler_job_rate(
        self, scheduler: _BranchScheduler, job: _BranchJob
    ) -> float:
        if self._faulted(scheduler.branch, job.owner_type, job.owner_id, job.resource):
            return 0.0
        return self._thermal_multiplier(
            scheduler.branch, job.owner_type, job.owner_id, job.resource
        )

    def _scheduler_transfer_snapshot(
        self, scheduler: _BranchScheduler, transfer: _BranchTransfer
    ) -> tuple[float, float, float, str, float]:
        task = scheduler.tasks[transfer.task_token]
        observation = replace(task.base_observation, vehicle_id=transfer.vehicle_id)
        return self._wireless_at(
            scheduler.branch,
            observation,
            transfer.rsu_id,
            transfer.direction,
        )

    @staticmethod
    def _scheduler_transfer_link_key(
        transfer: _BranchTransfer,
    ) -> tuple[str, str, TransferDirection]:
        return transfer.vehicle_id, transfer.rsu_id, transfer.direction

    def _scheduler_active_link_counts(
        self, scheduler: _BranchScheduler
    ) -> dict[tuple[str, str, TransferDirection], int]:
        counts: dict[tuple[str, str, TransferDirection], int] = {}
        for transfer in scheduler.transfers.values():
            rate, _, _, state, _ = self._scheduler_transfer_snapshot(
                scheduler, transfer
            )
            if state not in {"connected", "temporary_outage"}:
                continue
            if state == "connected" and rate <= 0:
                continue
            key = self._scheduler_transfer_link_key(transfer)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _scheduler_transfer_service(
        self,
        scheduler: _BranchScheduler,
        transfer: _BranchTransfer,
        active_counts: Mapping[tuple[str, str, TransferDirection], int],
    ) -> tuple[float, float, float, str, float]:
        rate, tx_power, rx_power, state, end_s = self._scheduler_transfer_snapshot(
            scheduler, transfer
        )
        count = active_counts.get(self._scheduler_transfer_link_key(transfer), 0)
        if count < 1:
            return 0.0, 0.0, 0.0, state, end_s
        share = 1.0 / count
        if state == "temporary_outage":
            return (
                0.0,
                tx_power * share,
                rx_power * share,
                state,
                end_s,
            )
        if state != "connected":
            return 0.0, 0.0, 0.0, state, end_s
        return (
            rate * share,
            tx_power * share,
            rx_power * share,
            state,
            end_s,
        )

    def _scheduler_pause_limit(self, direction: TransferDirection) -> float:
        config = self.mask_engine.config
        if config is None:
            return math.inf
        return (
            config.uplink_pause_limit_s
            if direction is TransferDirection.UL
            else config.downlink_pause_limit_s
        )

    def _scheduler_advance(self, scheduler: _BranchScheduler, target_s: float) -> bool:
        branch = scheduler.branch
        dt_s = max(0.0, target_s - branch.elapsed_s)
        if dt_s <= 0:
            return True
        active_tasks = sum(
            task.state not in {"PENDING", "DONE", "FAIL"}
            for task in scheduler.tasks.values()
        )
        self._scheduler_add_normalized_cost(scheduler, "latency", active_tasks * dt_s)
        config = self.mask_engine.config
        if config is not None:
            for vehicle_id in sorted(scheduler.vehicle_battery_j):
                params = _mapping(config.vehicle_branch_parameters.get(vehicle_id))
                failed = self._faulted(branch, "vehicle", vehicle_id, "all")
                if not failed and not self._scheduler_spend_vehicle_physical_only(
                    scheduler,
                    vehicle_id,
                    max(0.0, _finite(params.get("idle_power_w"), 0.0)) * dt_s,
                ):
                    return False
                hold_count = sum(
                    task.vehicle_id == vehicle_id
                    and task.state not in {"PENDING", "DONE", "FAIL"}
                    for task in scheduler.tasks.values()
                )
                hold_energy = (
                    0.0
                    if failed
                    else max(0.0, _finite(params.get("hold_power_w"), 0.0))
                    * hold_count
                    * dt_s
                )
                if hold_energy and not self._scheduler_spend_vehicle(
                    scheduler, vehicle_id, hold_energy
                ):
                    return False
            for rsu_id in sorted(branch.rsus):
                params = _mapping(config.rsu_branch_parameters.get(rsu_id))
                self._scheduler_spend_rsu_physical_only(
                    scheduler,
                    rsu_id,
                    max(0.0, _finite(params.get("idle_power_w"), 0.0)) * dt_s,
                )
                hold_count = sum(
                    task.admitted_rsu == rsu_id and task.state not in {"DONE", "FAIL"}
                    for task in scheduler.tasks.values()
                )
                hold_energy = (
                    max(0.0, _finite(params.get("hold_power_w"), 0.0))
                    * hold_count
                    * dt_s
                )
                if hold_energy:
                    self._scheduler_spend_rsu(scheduler, rsu_id, hold_energy)
        for pool in scheduler.resources.values():
            for job_id in tuple(pool.running):
                job = scheduler.jobs[job_id]
                served = min(
                    job.remaining_work_s,
                    self._scheduler_job_rate(scheduler, job) * dt_s,
                )
                if served <= 0:
                    continue
                job.remaining_work_s = max(0.0, job.remaining_work_s - served)
                energy = job.total_energy_j * served / job.total_work_s
                if job.owner_type == "vehicle":
                    if not self._scheduler_spend_vehicle(
                        scheduler, job.owner_id, energy
                    ):
                        return False
                elif job.completion_kind == "MODEL_MAINTENANCE_DONE":
                    self._scheduler_spend_rsu_physical_only(
                        scheduler, job.owner_id, energy
                    )
                else:
                    self._scheduler_spend_rsu(scheduler, job.owner_id, energy)
        active_link_counts = self._scheduler_active_link_counts(scheduler)
        for transfer in tuple(scheduler.transfers.values()):
            rate, tx_power, rx_power, state, _ = self._scheduler_transfer_service(
                scheduler, transfer, active_link_counts
            )
            if state not in {"connected", "temporary_outage"}:
                continue
            delivered = min(transfer.remaining_bits, rate * dt_s)
            active_s = delivered / rate if rate > 0 else dt_s
            transfer.remaining_bits = max(0.0, transfer.remaining_bits - delivered)
            if transfer.direction is TransferDirection.UL:
                if not self._scheduler_spend_vehicle(
                    scheduler, transfer.vehicle_id, tx_power * active_s
                ):
                    return False
                self._scheduler_spend_rsu(
                    scheduler, transfer.rsu_id, rx_power * active_s
                )
            else:
                if not self._scheduler_spend_vehicle(
                    scheduler, transfer.vehicle_id, rx_power * active_s
                ):
                    return False
                self._scheduler_spend_rsu(
                    scheduler, transfer.rsu_id, tx_power * active_s
                )
        if config is not None:
            vehicle_power = branch.virtual_queues.setdefault("vehicle_power", {})
            if isinstance(vehicle_power, dict):
                for owner_id, budget_w in config.vehicle_power_budgets_w.items():
                    vehicle_power[owner_id] = max(
                        0.0,
                        _finite(vehicle_power.get(owner_id), 0.0) - budget_w * dt_s,
                    )
            rsu_power = branch.virtual_queues.setdefault("rsu_power", {})
            if isinstance(rsu_power, dict):
                for owner_id, budget_w in config.rsu_power_budgets_w.items():
                    rsu_power[owner_id] = max(
                        0.0,
                        _finite(rsu_power.get(owner_id), 0.0) - budget_w * dt_s,
                    )
        for rsu_id in branch.telemetry_age_s:
            branch.telemetry_age_s[rsu_id] += dt_s
        branch.elapsed_s = target_s
        branch.slack_s = max(0.0, branch.slack_s - dt_s)
        return True

    def _scheduler_vehicle_power_w(
        self,
        scheduler: _BranchScheduler,
        vehicle_id: str,
        active_link_counts: Mapping[tuple[str, str, TransferDirection], int],
    ) -> float:
        config = self.mask_engine.config
        params = (
            {}
            if config is None
            else _mapping(config.vehicle_branch_parameters.get(vehicle_id))
        )
        failed = self._faulted(scheduler.branch, "vehicle", vehicle_id, "all")
        power = 0.0 if failed else max(0.0, _finite(params.get("idle_power_w"), 0.0))
        if not failed:
            power += max(0.0, _finite(params.get("hold_power_w"), 0.0)) * sum(
                task.vehicle_id == vehicle_id
                and task.state not in {"PENDING", "DONE", "FAIL"}
                for task in scheduler.tasks.values()
            )
        for pool in scheduler.resources.values():
            if pool.owner_type != "vehicle" or pool.owner_id != vehicle_id:
                continue
            for job_id in pool.running:
                job = scheduler.jobs[job_id]
                rate = self._scheduler_job_rate(scheduler, job)
                if rate > 0:
                    power += job.total_energy_j / job.total_work_s * rate
        for transfer in scheduler.transfers.values():
            if transfer.vehicle_id != vehicle_id:
                continue
            _, tx_power, rx_power, _, _ = self._scheduler_transfer_service(
                scheduler, transfer, active_link_counts
            )
            power += (
                tx_power if transfer.direction is TransferDirection.UL else rx_power
            )
        return max(0.0, power)

    def _scheduler_next_time(self, scheduler: _BranchScheduler) -> float:
        now = scheduler.branch.elapsed_s
        candidates = [scheduler.events[0][0]] if scheduler.events else []
        for pool in scheduler.resources.values():
            for job_id in pool.running:
                job = scheduler.jobs[job_id]
                rate = self._scheduler_job_rate(scheduler, job)
                if rate > 0:
                    candidates.append(
                        now
                        if job.remaining_work_s <= 0
                        else _strict_future_instant(now, job.remaining_work_s / rate)
                    )
        active_link_counts = self._scheduler_active_link_counts(scheduler)
        for transfer in scheduler.transfers.values():
            rate, _, _, state, end_s = self._scheduler_transfer_service(
                scheduler, transfer, active_link_counts
            )
            if state in {"permanent_loss", "handover", "missing"}:
                candidates.append(now)
            elif state == "connected" and rate > 0:
                transfer.paused_since_s = None
                completion_s = (
                    now
                    if transfer.remaining_bits <= 0
                    else _strict_future_instant(now, transfer.remaining_bits / rate)
                )
                candidates.append(min(end_s, completion_s))
            elif state == "temporary_outage":
                if transfer.paused_since_s is None:
                    transfer.paused_since_s = now
                expiry = transfer.paused_since_s + self._scheduler_pause_limit(
                    transfer.direction
                )
                candidates.append(min(end_s, expiry))
            elif end_s > now:
                candidates.append(end_s)
        for vehicle_id, battery_j in scheduler.vehicle_battery_j.items():
            power_w = self._scheduler_vehicle_power_w(
                scheduler, vehicle_id, active_link_counts
            )
            if power_w > 0:
                candidates.append(
                    now
                    if battery_j <= 0
                    else _strict_future_instant(now, battery_j / power_w)
                )
        finite = [value for value in candidates if math.isfinite(value)]
        if not finite:
            return math.inf
        earliest = min(finite)
        # Match EventQueue.pop_compound(): advance to the latest IEEE-754
        # representation of one physical instant before applying service.
        return max(
            value for value in finite if _same_representable_instant(value, earliest)
        )

    @staticmethod
    def _scheduler_pop_same_instant_events(
        scheduler: _BranchScheduler, time_s: float
    ) -> list[tuple[float, int, int, str, str]]:
        """Pop only fixed events in the current finite-ULP compound instant."""

        if (
            scheduler.events
            and scheduler.events[0][0] < time_s
            and not _same_representable_instant(scheduler.events[0][0], time_s)
        ):
            raise RuntimeError("isolated scenario event queue moved backwards")
        batch: list[tuple[float, int, int, str, str]] = []
        while scheduler.events and _same_representable_instant(
            scheduler.events[0][0], time_s
        ):
            batch.append(heapq.heappop(scheduler.events))
        return sorted(batch)

    def _scheduler_refresh_admission_bounds(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
    ) -> bool:
        """Recompute paired conservative edge bounds at UL completion."""

        action = task.last_action
        if action is None or action.kind is not ActionKind.EDGE:
            return False
        public = self._scheduler_observation(scheduler, task, ActionStage.READY)
        observation = self._scheduler_execution_observation(scheduler, action, public)
        rows = self._actual_edge_rows(
            task.future,
            action,
            observation,
            task.edge_pairing_token,
            quality_bin=task.latent_quality_bin,
        )
        if task.fixed_continuation:
            # The frozen anchor already carries an all-conformal-candidate
            # admission upper bound.  The actual artifact still needs one
            # paired row in the live RSU context for executable service, but a
            # single realized artifact is not expected to bear every candidate
            # quality label simultaneously.
            return bool(
                rows
                and task.admission_vram_upper_bytes > 0
                and task.admission_gpu_work_upper_s > 0
            )
        bounds = self._certified_edge_admission_bounds(task.future, action, observation)
        if not rows or bounds is None:
            return False
        task.admission_vram_upper_bytes = bounds[0]
        task.admission_gpu_work_upper_s = bounds[1]
        return True

    def _scheduler_admit(self, scheduler: _BranchScheduler, task: _BranchTask) -> bool:
        action = task.last_action
        outcome = task.last_outcome
        if action is None or outcome is None or action.kind is not ActionKind.EDGE:
            return False
        rsu_id = action.rsu_id or ""
        row = scheduler.branch.rsus.get(rsu_id)
        model = self.mask_engine.profile.edge_models.get(action.edge_model_id or "")
        pipeline = self.mask_engine.profile.pipelines.get(task.selected_pipeline or "")
        if row is None or model is None or pipeline is None:
            return False
        cached = _mapping(row.get("cached_models"))
        protocol = scheduler.branch.active_protocol_version
        vram = max(
            0,
            task.admission_vram_upper_bytes,
            int(outcome.values.get("vram_bytes", 0)),
        )
        gpu_work = max(
            0.0,
            task.admission_gpu_work_upper_s,
            _finite(outcome.values.get("rsu_gpu_work_s"), 0.0),
        )
        valid = bool(
            not self._faulted(scheduler.branch, "rsu", rsu_id, "all")
            and scheduler.branch.active_profile_hash
            == self.mask_engine.profile.profile_hash
            and protocol == self.mask_engine.profile.protocol_version
            and model.protocol_version == protocol
            and pipeline.protocol_version == protocol
            and rsu_id in model.supported_rsus
            and (
                not model.supported_pipelines
                or task.selected_pipeline in model.supported_pipelines
            )
            and cached.get(model.model_id) == model.model_hash
            and int(row.get("descriptors", 0)) + 1
            <= int(row.get("descriptor_capacity", 0))
            and int(row.get("vram_bytes", 0)) + vram
            <= int(row.get("vram_capacity_bytes", 0))
            and _finite(row.get("reserved_work_gpu_s"), 0.0) + gpu_work
            <= _finite(row.get("workload_capacity_gpu_s"), 0.0)
            + _RSU_WORKLOAD_CAPACITY_TOLERANCE_GPU_S
        )
        if not valid:
            return False
        row["descriptors"] = int(row.get("descriptors", 0)) + 1
        row["vram_bytes"] = int(row.get("vram_bytes", 0)) + vram
        row["reserved_work_gpu_s"] = (
            _finite(row.get("reserved_work_gpu_s"), 0.0) + gpu_work
        )
        task.admitted_rsu = rsu_id
        task.reserved_vram_bytes = vram
        task.reserved_gpu_work_s = gpu_work
        return True

    def _scheduler_technical_fallback(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        rng: random.Random,
        *,
        reason: str,
    ) -> None:
        """Apply the frozen local technical fallback after an edge-path failure.

        Every admission, edge-compute, downlink, version and RSU-fault path
        enters this single gate.  The failed activity and any partial packet
        are first cancelled without rolling back spent work or energy.  The
        exact configured local fallback is then rechecked by the shared hard
        mask and deterministic repairer using the current public observation.
        It can never be repaired to another edge route.
        """

        if task.state in {"DONE", "FAIL"}:
            return
        self._scheduler_cancel_activity(scheduler, task, release_vehicle=False)
        pipeline = self.mask_engine.profile.pipelines.get(task.selected_pipeline or "")
        if pipeline is None:
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason=reason,
            )
            return
        observation = self._scheduler_fallback_observation(scheduler, task)
        fallback = None if pipeline is None else pipeline.fallback_local_model
        proposed = (
            Action.fail(ActionStage.READY)
            if fallback is None
            else Action.local(ActionStage.READY, fallback)
        )
        repair = self.repairer.repair(
            proposed,
            self._future_task_record(task.future, observation),
            observation,
            # EDGE_FAILED is one of the repairer's frozen fallback triggers.
            # The externally visible reason below remains the concrete cause.
            failure_reason="EDGE_FAILED",
        )
        action = repair.executed
        try:
            sampling_observation = self._latent_sampling_observation(task, observation)
            if action.kind is ActionKind.LOCAL:
                rows = [
                    row
                    for row in self._scenario_rows_for(
                        task.future, self.scenario_source, "local_rows"
                    )
                    if row.model_id == action.local_model_id
                    and row.device_type == observation.device_type
                    and row.quality_bin in observation.conformal_quality_bins
                    and self._context_matches(row, observation)
                ]
                if {str(row.quality_bin) for row in rows} != set(
                    observation.conformal_quality_bins
                ):
                    raise LookupError("fallback lacks all quality-cell pairings")
            outcome = self._future_outcome(
                task.future, action, sampling_observation, rng
            )
        except LookupError:
            scheduler.branch.complete_macro_recourse = False
            scheduler.branch.incomplete_reason = f"{reason}_FALLBACK_PAIRING_MISSING"
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason=reason,
            )
            return
        scheduler.event_trace.append(
            {
                "time_s": scheduler.branch.elapsed_s,
                "kind": "DETERMINISTIC_REPAIR",
                "task_token": task.task_token,
                "proposed": proposed.canonical_id,
                "executed": action.canonical_id,
                "changed": repair.changed,
                "reason": reason,
                "proposed_reason_codes": [
                    item.value for item in repair.proposed_reasons
                ],
            }
        )
        self._scheduler_commit(
            scheduler,
            task,
            action,
            outcome,
            observation,
            failure_reason=reason,
        )

    def _scheduler_admission_fallback(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        rng: random.Random,
        *,
        reason: str = "ATOMIC_ADMISSION_REJECTED",
    ) -> None:
        self._scheduler_technical_fallback(scheduler, task, rng, reason=reason)

    def _scheduler_apply_device_fault_effects(
        self,
        scheduler: _BranchScheduler,
        offset_s: float,
        rng: random.Random,
    ) -> None:
        """Apply production-equivalent immediate effects of DEVICE_FAULT."""

        environment = scheduler.branch.environment
        if environment is None:
            return
        starts = [
            event
            for event in environment.faults
            if _same_representable_instant(event.offset_s, offset_s)
            and not event.event_type.endswith(("RECOVER", "END"))
        ]
        failed_vehicles = {
            event.target_id for event in starts if event.target_type == "vehicle"
        }
        for task in sorted(scheduler.tasks.values(), key=lambda item: item.task_token):
            if task.vehicle_id in failed_vehicles and task.state not in {
                "PENDING",
                "DONE",
                "FAIL",
            }:
                self._scheduler_finish(
                    scheduler,
                    task,
                    success=False,
                    reason="VEHICLE_DEVICE_FAULT",
                )

        for rsu_id in sorted(
            {event.target_id for event in starts if event.target_type == "rsu"}
        ):
            affected = {
                transfer.task_token
                for transfer in scheduler.transfers.values()
                if transfer.rsu_id == rsu_id
            }
            affected.update(
                job.task_token
                for job in scheduler.jobs.values()
                if job.task_token and job.owner_type == "rsu" and job.owner_id == rsu_id
            )
            affected.update(
                task.task_token
                for task in scheduler.tasks.values()
                if task.admitted_rsu == rsu_id
            )
            for task_token in sorted(affected):
                task = scheduler.tasks[task_token]
                if task.state in {"DONE", "FAIL"}:
                    continue
                self._scheduler_admission_fallback(
                    scheduler,
                    task,
                    rng,
                    reason="RSU_DEVICE_FAULT",
                )

    def _scheduler_pipeline_fallback(
        self,
        scheduler: _BranchScheduler,
        task: _BranchTask,
        rng: random.Random,
    ) -> None:
        self._scheduler_technical_fallback(
            scheduler,
            task,
            rng,
            reason="ANON_TRANSACTION_NOT_FORMED",
        )

    def _scheduler_complete_job(
        self,
        scheduler: _BranchScheduler,
        job_id: str,
        rng: random.Random,
    ) -> None:
        job = scheduler.jobs.pop(job_id)
        pool = scheduler.resources[(job.owner_type, job.owner_id, job.resource)]
        if job_id in pool.running:
            pool.running.remove(job_id)
        if job.completion_kind == "BACKGROUND_DONE":
            return
        if job.completion_kind == "MODEL_MAINTENANCE_DONE":
            if job.maintenance_event is None:
                scheduler.branch.complete_macro_recourse = False
                scheduler.branch.incomplete_reason = (
                    "SCENARIO_MODEL_MAINTENANCE_EVENT_MISSING"
                )
                return
            event = job.maintenance_event
            model_id = str(getattr(event, "model_id", "") or "")
            rsu = scheduler.branch.rsus.get(job.owner_id)
            if rsu is None or not model_id:
                scheduler.branch.complete_macro_recourse = False
                scheduler.branch.incomplete_reason = (
                    "SCENARIO_MODEL_MAINTENANCE_MODEL_MISSING"
                )
                return
            key = scheduler.maintenance_job_keys.get(job.job_id)
            if key is None or scheduler.maintenance_active.get(key) != job.job_id:
                scheduler.branch.complete_macro_recourse = False
                scheduler.branch.incomplete_reason = (
                    "SCENARIO_MODEL_MAINTENANCE_CHAIN_CORRUPT"
                )
                return
            cache_before = dict(_mapping(rsu.get("cached_models")))
            old_version = getattr(event, "old_version", None)
            if old_version is not None and cache_before.get(model_id) != old_version:
                scheduler.branch.complete_macro_recourse = False
                scheduler.branch.incomplete_reason = (
                    "SCENARIO_MODEL_MAINTENANCE_VERSION_PRECONDITION"
                )
                scheduler.event_trace.append(
                    {
                        "time_s": scheduler.branch.elapsed_s,
                        "kind": "MODEL_MAINTENANCE_PRECONDITION_FAILED",
                        "job_id": job.job_id,
                        "rsu_id": job.owner_id,
                        "model_id": model_id,
                        "expected_old_version": old_version,
                        "actual_version": cache_before.get(model_id),
                    }
                )
                return
            if not bool(getattr(event, "remove", False)) and not getattr(
                event, "new_version", None
            ):
                scheduler.branch.complete_macro_recourse = False
                scheduler.branch.incomplete_reason = (
                    "SCENARIO_MODEL_MAINTENANCE_UPDATE_MISSING"
                )
                return
            self._apply_version_event(scheduler.branch, event)
            if not self._scheduler_release_next_maintenance(scheduler, job.job_id):
                return
            scheduler.event_trace.append(
                {
                    "time_s": scheduler.branch.elapsed_s,
                    "kind": "MODEL_MAINTENANCE_COMPLETE",
                    "job_id": job.job_id,
                    "rsu_id": job.owner_id,
                    "model_id": model_id,
                    "work_s": job.total_work_s,
                    "energy_j": job.total_energy_j,
                }
            )
            return
        task = scheduler.tasks[job.task_token]
        if task.state in {"DONE", "FAIL"}:
            return
        if job.completion_kind in {"PREP_DONE", "LIVE_PREP_DONE"}:
            self._scheduler_release_vehicle(scheduler, task)
            if bool(getattr(task.future, "prep_failed", False)):
                self._scheduler_finish(
                    scheduler,
                    task,
                    success=False,
                    reason="PUBLIC_PREPROCESS_FAILURE",
                )
            else:
                task.state = "RAW"
        elif job.completion_kind == "ANCHOR_PREP_DONE":
            self._scheduler_release_vehicle(scheduler, task)
            if task.continuation_prep_failed:
                self._scheduler_finish(
                    scheduler,
                    task,
                    success=False,
                    reason="ANCHOR_PUBLIC_PREPROCESS_FAILURE",
                )
            else:
                self._scheduler_start_fixed_control(scheduler, task, "ACTION")
        elif job.completion_kind == "ANCHOR_COMPUTE_DONE":
            self._scheduler_start_next_fixed_stage(scheduler, task)
        elif job.completion_kind == "ANCHOR_LOCAL_DONE":
            failed = bool(
                task.last_outcome
                and _finite(task.last_outcome.values.get("failure_probability"), 0.0)
                >= 0.5
            )
            self._scheduler_finish(
                scheduler,
                task,
                success=not failed,
                reason=(
                    "ANCHOR_FALLBACK_RESULT"
                    if not failed
                    else "ANCHOR_FALLBACK_MODEL_FAILURE"
                ),
            )
        elif job.completion_kind in {"LOCAL_DONE", "LIVE_LOCAL_DONE"}:
            failed = bool(
                task.last_outcome
                and _finite(task.last_outcome.values.get("failure_probability"), 0.0)
                >= 0.5
            )
            self._scheduler_finish(
                scheduler,
                task,
                success=not failed,
                reason="LOCAL_RESULT" if not failed else "LOCAL_MODEL_FAILURE",
            )
        elif job.completion_kind in {"PIPE_STAGE_DONE", "LIVE_PIPE_STAGE_DONE"}:
            if job.completion_kind == "LIVE_PIPE_STAGE_DONE":
                task.artifact_key = (
                    None
                    if task.edge_outcome is None
                    else task.edge_outcome.artifact_key
                )
            self._scheduler_start_next_pipeline_stage(scheduler, task)
        elif job.completion_kind == "INGRESS_DONE":
            outcome = task.last_outcome
            assert outcome is not None and task.admitted_rsu is not None
            if bool(outcome.values.get("ingress_failure", False)):
                self._scheduler_technical_fallback(
                    scheduler,
                    task,
                    rng,
                    reason="RSU_INGRESS_FAILURE",
                )
                return
            gpu = max(0.0, _finite(outcome.values.get("rsu_gpu_work_s"), 0.0))
            task.state = "GPU_WAIT"
            self._scheduler_enqueue_job(
                scheduler,
                task,
                owner_type="rsu",
                owner_id=task.admitted_rsu,
                resource="gpu",
                work_s=gpu,
                energy_j=max(
                    0.0,
                    _finite(outcome.values.get("rsu_gpu_energy_j"), 0.0),
                ),
                completion_kind="GPU_DONE",
            )
        elif job.completion_kind == "GPU_DONE":
            outcome = task.last_outcome
            assert outcome is not None and task.admitted_rsu is not None
            result_rsu = task.admitted_rsu
            if _finite(outcome.values.get("failure_probability"), 0.0) >= 0.5:
                self._scheduler_technical_fallback(
                    scheduler,
                    task,
                    rng,
                    reason="EDGE_MODEL_FAILURE",
                )
            else:
                # Match the simulator: GPU/VRAM/model pinning is no longer
                # needed once a standalone result packet exists.
                self._scheduler_release_rsu(scheduler, task)
                task.state = "DL"
                scheduler.sequence += 1
                transfer_id = f"branch-transfer:{scheduler.sequence:08d}"
                scheduler.transfers[transfer_id] = _BranchTransfer(
                    transfer_id,
                    task.task_token,
                    task.vehicle_id,
                    result_rsu,
                    TransferDirection.DL,
                    max(1.0, _finite(outcome.values.get("result_size_bits"), 0.0)),
                )
        scheduler.event_trace.append(
            {
                "time_s": scheduler.branch.elapsed_s,
                "kind": "JOB_COMPLETION",
                "task_token": task.task_token,
                "resource": job.resource,
                "completion_kind": job.completion_kind,
            }
        )

    def _scheduler_complete_transfer(
        self,
        scheduler: _BranchScheduler,
        transfer_id: str,
        rng: random.Random,
    ) -> None:
        transfer = scheduler.transfers.pop(transfer_id)
        task = scheduler.tasks[transfer.task_token]
        if transfer.direction is TransferDirection.DL:
            version_valid = bool(
                scheduler.branch.active_profile_hash
                == self.mask_engine.profile.profile_hash
                and scheduler.branch.active_protocol_version
                == self.mask_engine.profile.protocol_version
            )
            if version_valid:
                self._scheduler_finish(
                    scheduler,
                    task,
                    success=True,
                    reason="DOWNLINK_RESULT",
                )
            else:
                self._scheduler_technical_fallback(
                    scheduler,
                    task,
                    rng,
                    reason="DOWNLINK_VERSION_MISMATCH",
                )
            return
        if not self._scheduler_refresh_admission_bounds(scheduler, task):
            self._scheduler_admission_fallback(
                scheduler,
                task,
                rng,
                reason="ADMISSION_PAIRED_MEASUREMENT_MISSING",
            )
            return
        if not self._scheduler_admit(scheduler, task):
            self._scheduler_admission_fallback(scheduler, task, rng)
            return
        outcome = task.last_outcome
        assert outcome is not None and task.admitted_rsu is not None
        ingress = max(0.0, _finite(outcome.values.get("rsu_ingress_work_s"), 0.0))
        task.state = "INGRESS_WAIT"
        self._scheduler_enqueue_job(
            scheduler,
            task,
            owner_type="rsu",
            owner_id=task.admitted_rsu,
            resource="ingress",
            work_s=ingress,
            energy_j=max(
                0.0,
                _finite(outcome.values.get("rsu_ingress_energy_j"), 0.0),
            ),
            completion_kind="INGRESS_DONE",
        )

    def _scheduler_arrival(
        self, scheduler: _BranchScheduler, task: _BranchTask
    ) -> None:
        config = self.mask_engine.config
        assert config is not None
        scheduler.pending_arrivals += 1
        if scheduler.vehicle_battery_j.get(task.vehicle_id, 0.0) <= 1e-9:
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason="BATTERY_GUARD_AT_ARRIVAL",
            )
            return
        if self._faulted(scheduler.branch, "vehicle", task.vehicle_id, "all"):
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason="VEHICLE_DEVICE_FAULT_AT_ARRIVAL",
            )
            return
        memory = max(0, int(getattr(task.future, "prep_memory_bytes", 0)))
        if not self._scheduler_reserve_vehicle(
            scheduler, task, {"accelerator": 1}, memory
        ):
            self._scheduler_finish(
                scheduler,
                task,
                success=False,
                reason="PUBLIC_PREPROCESS_RESOURCE_REJECTED",
            )
            return
        work = max(0.0, float(getattr(task.future, "prep_work_s", 0.0)))
        energy = max(0.0, float(getattr(task.future, "prep_energy_j", 0.0)))
        task.state = "PREP_WAIT"
        self._scheduler_enqueue_job(
            scheduler,
            task,
            owner_type="vehicle",
            owner_id=task.vehicle_id,
            resource="accelerator",
            work_s=work,
            energy_j=energy,
            completion_kind="PREP_DONE",
        )

    def _scheduler_decide_ready_tasks(
        self, scheduler: _BranchScheduler, rng: random.Random
    ) -> None:
        ready = [
            task
            for task in scheduler.tasks.values()
            if task.state in {"RAW", "READY"}
            and task.deadline_s > scheduler.branch.elapsed_s
            and not _same_representable_instant(
                task.deadline_s, scheduler.branch.elapsed_s
            )
        ]
        if not ready or scheduler.macro_decisions >= self.horizon_events:
            return
        scheduler.macro_decisions += 1
        scheduler.last_decision_time_s = scheduler.branch.elapsed_s
        for task in sorted(
            ready, key=lambda item: (item.deadline_s, item.vehicle_id, item.task_token)
        ):
            stage = ActionStage.RAW if task.state == "RAW" else ActionStage.READY
            observation = self._scheduler_observation(scheduler, task, stage)
            action = self._future_action(task.future, observation, scheduler.branch)
            scheduler.branch.focal_decisions += 1
            scheduler.branch.decision_trace.append(
                {
                    "decision_index": scheduler.branch.focal_decisions,
                    "macro_event_index": scheduler.macro_decisions,
                    "time_s": scheduler.branch.elapsed_s,
                    "task_token": task.task_token,
                    "vehicle_id": task.vehicle_id,
                    "stage": stage.value,
                    "action": action.canonical_id,
                }
            )
            self._scheduler_schedule_decision(
                scheduler,
                task,
                action,
                observation,
                self._conditional_decision_seed(
                    scheduler.branch,
                    task.task_token,
                    stage,
                    action,
                    scheduler.branch.elapsed_s,
                ),
            )

    def _scheduler_apply_transfer_boundaries(
        self,
        scheduler: _BranchScheduler,
        rng: random.Random,
    ) -> None:
        """Apply same-time link/mobility effects before battery and deadline."""

        for transfer_id, transfer in tuple(sorted(scheduler.transfers.items())):
            _, _, _, state, _ = self._scheduler_transfer_snapshot(scheduler, transfer)
            failed = state in {"permanent_loss", "handover", "missing"}
            failure_suffix = state.upper()
            if state == "connected":
                transfer.paused_since_s = None
            elif state == "temporary_outage":
                if transfer.paused_since_s is None:
                    transfer.paused_since_s = scheduler.branch.elapsed_s
                limit = self._scheduler_pause_limit(transfer.direction)
                expiry_s = transfer.paused_since_s + limit
                failed = (
                    scheduler.branch.elapsed_s > expiry_s
                    or _same_representable_instant(scheduler.branch.elapsed_s, expiry_s)
                )
                if failed:
                    failure_suffix = "PAUSE_EXPIRED"
            if not failed:
                continue
            task = scheduler.tasks[transfer.task_token]
            scheduler.transfers.pop(transfer_id, None)
            direction = (
                "UPLINK" if transfer.direction is TransferDirection.UL else "DOWNLINK"
            )
            self._scheduler_technical_fallback(
                scheduler,
                task,
                rng,
                reason=f"{direction}_{failure_suffix}",
            )

    def _scheduler_run(
        self,
        scheduler: _BranchScheduler,
        rng: random.Random,
        *,
        stop_before_next_macro: bool = False,
    ) -> None:
        while scheduler.branch.complete_macro_recourse:
            self._scheduler_dispatch(scheduler, rng)
            for task in sorted(
                scheduler.tasks.values(), key=lambda item: item.task_token
            ):
                if task.state == "PIPE_REPAIR":
                    self._scheduler_pipeline_fallback(scheduler, task, rng)
            ready_now = any(
                task.state in {"RAW", "READY"} for task in scheduler.tasks.values()
            )
            if stop_before_next_macro and ready_now:
                self._scheduler_project_task_queues(scheduler)
                break
            self._scheduler_decide_ready_tasks(scheduler, rng)
            self._scheduler_dispatch(scheduler, rng)
            if (
                not stop_before_next_macro
                and scheduler.macro_decisions >= self.horizon_events
                and not any(
                    task.state == "DECISION_WAIT" for task in scheduler.tasks.values()
                )
            ):
                self._scheduler_project_task_queues(scheduler)
                break
            if scheduler.tasks and all(
                task.state in {"DONE", "FAIL"} for task in scheduler.tasks.values()
            ):
                # Stale deadline/controller/environment events do not extend a
                # rollout after its last task terminates.  In particular, idle
                # power is not integrated until an unrelated battery-depletion
                # candidate after the recourse horizon has already ended.
                self._scheduler_project_task_queues(scheduler)
                break
            terminal_before = sum(
                task.state in {"DONE", "FAIL"} for task in scheduler.tasks.values()
            )
            next_time = self._scheduler_next_time(scheduler)
            if not math.isfinite(next_time):
                active = any(
                    task.state not in {"PENDING", "DONE", "FAIL"}
                    for task in scheduler.tasks.values()
                )
                if active:
                    scheduler.branch.complete_macro_recourse = False
                    scheduler.branch.incomplete_reason = (
                        "BRANCH_HAS_NO_FUTURE_SERVICE_EVENT"
                    )
                self._scheduler_project_task_queues(scheduler)
                break
            if (
                next_time < scheduler.branch.elapsed_s
                and not _same_representable_instant(
                    next_time, scheduler.branch.elapsed_s
                )
            ):
                raise RuntimeError("isolated scenario scheduler moved backwards")
            if not self._scheduler_advance(scheduler, next_time):
                self._scheduler_project_task_queues(scheduler)
                break

            # Completion events have priority over every fixed event at the
            # same timestamp, in particular over the absolute deadline.
            completed_jobs = sorted(
                job_id
                for pool in scheduler.resources.values()
                for job_id in pool.running
                if scheduler.jobs[job_id].remaining_work_s <= 1e-9
            )
            for job_id in completed_jobs:
                self._scheduler_complete_job(scheduler, job_id, rng)
            completed_transfers = sorted(
                transfer_id
                for transfer_id, transfer in scheduler.transfers.items()
                if transfer.remaining_bits <= 1e-3
            )
            for transfer_id in completed_transfers:
                self._scheduler_complete_transfer(scheduler, transfer_id, rng)

            same_time = self._scheduler_pop_same_instant_events(scheduler, next_time)

            # Phase 3: environment, link/mobility, fault, thermal and version
            # effects all precede battery guard and deadline processing.
            for _, _, _, kind, object_id in same_time:
                if kind == "ENVIRONMENT":
                    self._scheduler_sync_branch(scheduler, scheduler.focal_vehicle_id)
                    self._apply_environment_at(
                        scheduler.branch,
                        next_time,
                        scheduler.focal_vehicle_id,
                        defer_rsu_maintenance=True,
                    )
                    environment = scheduler.branch.environment
                    if environment is not None:
                        for version_event in environment.versions:
                            if (
                                version_event.target_type == "rsu"
                                and version_event.event_type
                                in {"MODEL_VERSION", "MODEL_CACHE"}
                                and _same_representable_instant(
                                    version_event.offset_s, next_time
                                )
                            ):
                                self._scheduler_enqueue_model_maintenance(
                                    scheduler, version_event
                                )
                    self._scheduler_apply_device_fault_effects(
                        scheduler, next_time, rng
                    )
                    scheduler.event_trace.append(
                        {"time_s": next_time, "kind": "ENVIRONMENT"}
                    )
            self._scheduler_apply_transfer_boundaries(scheduler, rng)

            # A valid compute/transfer completion at the exact depletion
            # instant won above.  Any remaining work now observes battery
            # guard after same-time faults, matching production priority.
            for vehicle_id, battery_j in sorted(scheduler.vehicle_battery_j.items()):
                if battery_j > 1e-9:
                    continue
                scheduler.branch.active_faults.add(("vehicle", vehicle_id, "all"))
                for task in sorted(
                    scheduler.tasks.values(), key=lambda item: item.task_token
                ):
                    if task.vehicle_id == vehicle_id and task.state not in {
                        "PENDING",
                        "DONE",
                        "FAIL",
                    }:
                        self._scheduler_finish(
                            scheduler,
                            task,
                            success=False,
                            reason="BATTERY_GUARD",
                        )

            # Phase 4: deadline; phase 5: arrivals; phase 6: delayed commits.
            for _, _, _, kind, object_id in sorted(same_time):
                if kind == "DEADLINE":
                    task = scheduler.tasks[object_id]
                    if task.state not in {"DONE", "FAIL"}:
                        self._scheduler_finish(
                            scheduler,
                            task,
                            success=False,
                            timeout=True,
                            reason="DEADLINE",
                        )
                elif kind == "ARRIVAL":
                    task = scheduler.tasks[object_id]
                    if task.state == "PENDING":
                        self._scheduler_arrival(scheduler, task)
                elif kind == "DECISION_COMMIT":
                    task = scheduler.tasks[object_id]
                    self._scheduler_handle_decision_commit(scheduler, task)
                elif kind == "ANCHOR_CONTROL_DONE":
                    task = scheduler.tasks.get(object_id)
                    if task is not None and task.state not in {"DONE", "FAIL"}:
                        self._scheduler_complete_fixed_control(scheduler, task)
            self._scheduler_project_task_queues(scheduler)
            if (
                stop_before_next_macro
                and sum(
                    task.state in {"DONE", "FAIL"} for task in scheduler.tasks.values()
                )
                > terminal_before
            ):
                break
        self._scheduler_project_task_queues(scheduler)
        self._scheduler_sync_branch(scheduler, scheduler.focal_vehicle_id)
        scheduler.branch.macro_events = scheduler.macro_decisions
        scheduler.branch.scheduler_trace = scheduler.event_trace

    def _event_heap_rollout(
        self,
        observation: Observation,
        first_action: Action,
        branch: _PredictionState,
        environment: ScenarioEnvironment | None,
        rng: random.Random,
        *,
        stop_before_next_macro: bool = False,
    ) -> _PredictionState | None:
        if environment is None:
            return None
        scheduler = self._scheduler_new(branch, observation, environment)
        if scheduler is None:
            return None
        source = self.scenario_source
        focal_future = SimpleNamespace(
            task_token=observation.task_id,
            arrival_offset_s=0.0,
            relative_deadline_s=max(_EPS_DURATION_S, observation.slack_s),
            vehicle_id=observation.vehicle_id,
            device_type=observation.device_type,
            context=None,
            quality_candidates=observation.conformal_quality_bins,
            quality_probabilities=observation.quality_probabilities,
            ood=observation.ood,
            quality_features=observation.quality_features,
            prep_work_s=0.0,
            prep_energy_j=0.0,
            prep_memory_bytes=0,
            prep_failed=False,
            local_rows=tuple(getattr(source, "local_rows", ())),
            anon_rows=tuple(getattr(source, "anon_rows", ())),
            edge_rows=tuple(getattr(source, "edge_rows", ())),
            complete_support=True,
            support_reason=None,
        )
        if observation.task_id in scheduler.tasks:
            branch.complete_macro_recourse = False
            branch.incomplete_reason = "FOCAL_AND_FUTURE_TASK_TOKEN_COLLISION"
            return branch
        focal_snapshot_rows = [
            _mapping(row)
            for row in observation.vehicle.get("active_tasks", ())
            if bool(_mapping(row).get("is_focal", False))
        ]
        if len(focal_snapshot_rows) > 1:
            branch.complete_macro_recourse = False
            branch.incomplete_reason = "FOCAL_ACTIVE_TASK_OWNERSHIP_AMBIGUOUS"
            return branch
        focal_snapshot = focal_snapshot_rows[0] if focal_snapshot_rows else {}
        owned_tokens = {
            str(name): max(0, int(count))
            for name, count in _mapping(
                focal_snapshot.get("reservation_tokens")
            ).items()
        }
        owned_memory = max(0, int(focal_snapshot.get("memory_reservation_bytes", 0)))
        if owned_memory > scheduler.vehicle_memory_reserved.get(
            observation.vehicle_id, 0
        ) or any(
            count
            > scheduler.descriptor_reserved.get(observation.vehicle_id, {}).get(name, 0)
            for name, count in owned_tokens.items()
        ):
            branch.complete_macro_recourse = False
            branch.incomplete_reason = "FOCAL_ACTIVE_TASK_OWNERSHIP_EXCEEDS_AGGREGATE"
            return branch
        focal = _BranchTask(
            observation.task_id,
            observation.vehicle_id,
            0.0,
            max(_EPS_DURATION_S, observation.slack_s),
            focal_future,
            observation,
            state=observation.stage.value,
            selected_pipeline=observation.selected_pipeline,
            artifact_key=observation.artifact_token,
            encoded_size_bytes=observation.encoded_size_bytes,
            reservation_tokens=owned_tokens,
            reserved_memory_bytes=owned_memory,
            focal=True,
        )
        focal.latent_quality_bin = self._task_latent_quality(
            branch, focal.task_token, observation
        )
        if observation.stage is ActionStage.READY:
            focal.edge_pairing_token = self._select_surrogate_pairing_token(
                branch, focal, observation
            )
        scheduler.tasks[focal.task_token] = focal
        self._scheduler_push(
            scheduler, focal.deadline_s, 20, "DEADLINE", focal.task_token
        )
        scheduler.macro_decisions = 1
        scheduler.last_decision_time_s = 0.0
        branch.focal_decisions = 1
        branch.decision_trace.append(
            {
                "decision_index": 1,
                "macro_event_index": 1,
                "time_s": 0.0,
                "task_token": focal.task_token,
                "vehicle_id": focal.vehicle_id,
                "stage": observation.stage.value,
                "action": first_action.canonical_id,
            }
        )
        self._scheduler_schedule_decision(
            scheduler,
            focal,
            first_action,
            observation,
            self._conditional_decision_seed(
                branch,
                focal.task_token,
                observation.stage,
                first_action,
                0.0,
            ),
        )
        self._scheduler_run(
            scheduler, rng, stop_before_next_macro=stop_before_next_macro
        )
        return branch

    def _rollout_future_tasks(
        self,
        observation: Observation,
        branch: _PredictionState,
        rng: random.Random,
    ) -> None:
        environment = branch.environment
        if environment is None or not branch.use_future_tasks:
            return
        future_tasks = tuple(
            sorted(
                getattr(environment, "future_tasks", ()),
                key=lambda item: (
                    float(getattr(item, "arrival_offset_s")),
                    str(getattr(item, "task_token")),
                ),
            )
        )
        for future in future_tasks:
            if branch.focal_decisions >= self.horizon_events:
                break
            if (
                not bool(getattr(future, "complete_support", False))
                or getattr(future, "vehicle_id", None) != observation.vehicle_id
            ):
                branch.complete_macro_recourse = False
                branch.incomplete_reason = (
                    "FUTURE_TASK_SUPPORT_INCOMPLETE_OR_MULTI_VEHICLE"
                )
                return
            arrival = float(getattr(future, "arrival_offset_s"))
            if arrival < branch.elapsed_s - _EPS_DURATION_S:
                branch.complete_macro_recourse = False
                branch.incomplete_reason = (
                    "OVERLAPPING_FUTURE_ARRIVAL_REQUIRES_CONCURRENT_DES"
                )
                return
            self._advance_branch(branch, arrival, observation.vehicle_id)
            deadline = arrival + float(getattr(future, "relative_deadline_s"))
            if deadline <= branch.elapsed_s + _EPS_DURATION_S:
                branch.complete_macro_recourse = False
                branch.incomplete_reason = "FUTURE_TASK_DEADLINE_AT_ARRIVAL"
                return
            branch.slack_s = deadline - branch.elapsed_s
            raw_observation = self._future_observation(
                future, observation, branch, stage=ActionStage.RAW
            )
            raw_action = self._future_action(future, raw_observation, branch)
            try:
                sampled = self._future_outcome(future, raw_action, raw_observation, rng)
            except LookupError:
                branch.complete_macro_recourse = False
                branch.incomplete_reason = "FUTURE_TASK_ACTION_PAIRING_MISSING"
                return
            raw_outcome = self._execute_prediction_action(
                raw_action, sampled, raw_observation, branch
            )
            branch.joint_row_ids.append(raw_outcome.row_id)
            branch.cumulative_cost += _task_cost_from_row(
                raw_action,
                raw_observation,
                self.mask_engine,
                raw_outcome.values,
            )
            if branch.slack_s <= _EPS_DURATION_S and not raw_outcome.formed_packet:
                branch.virtual_queues["failure"] = max(
                    0.0,
                    _finite(branch.virtual_queues.get("failure"), 0.0) + 1.0,
                )
                branch.virtual_queues["timeout"] = max(
                    0.0,
                    _finite(branch.virtual_queues.get("timeout"), 0.0) + 1.0,
                )
                continue
            if (
                raw_action.kind is ActionKind.PIPE
                and branch.focal_decisions < self.horizon_events
                and raw_outcome.formed_packet
            ):
                ready_observation = self._future_observation(
                    future,
                    observation,
                    branch,
                    stage=ActionStage.READY,
                    selected_pipeline=raw_action.pipeline_id,
                    artifact_key=raw_outcome.artifact_key,
                    encoded_size_bytes=int(
                        raw_outcome.values.get("encoded_size_bytes", 0)
                    ),
                )
                ready_action = self._future_action(future, ready_observation, branch)
                try:
                    ready_sample = self._future_outcome(
                        future, ready_action, ready_observation, rng
                    )
                except LookupError:
                    branch.complete_macro_recourse = False
                    branch.incomplete_reason = "FUTURE_READY_PAIRING_MISSING"
                    return
                ready_outcome = self._execute_prediction_action(
                    ready_action, ready_sample, ready_observation, branch
                )
                branch.joint_row_ids.append(ready_outcome.row_id)
                branch.cumulative_cost += _task_cost_from_row(
                    ready_action,
                    ready_observation,
                    self.mask_engine,
                    ready_outcome.values,
                )

    def _one_rollout(
        self,
        task: TaskRecord,
        observation: Observation,
        first_action: Action,
        scenario_index: int,
        *,
        include_diagnostics: bool = False,
        stop_before_next_macro: bool = False,
    ) -> Any:
        # The exogenous realization is common across actions for a given
        # scenario index.  Only conditional action-outcome sampling receives an
        # action-specific substream.
        common_seed = self._seed_for(observation, scenario_index)
        environment_rng = random.Random(common_seed)
        environment_sampler = getattr(self.scenario_source, "sample_environment", None)
        environment = (
            environment_sampler(environment_rng)
            if callable(environment_sampler)
            else None
        )
        branch = self._new_branch(observation, environment)
        branch.common_scenario_seed = common_seed
        initial_lyapunov = self._lyapunov(branch)
        rng = random.Random(self._action_seed(common_seed, first_action))

        event_heap_branch = self._event_heap_rollout(
            observation,
            first_action,
            branch,
            environment,
            rng,
            stop_before_next_macro=stop_before_next_macro,
        )
        if event_heap_branch is not None:
            terminal_drift = self._lyapunov(branch) - initial_lyapunov
            numerator = terminal_drift + self.lyapunov_v * branch.cumulative_cost
            result = (
                numerator,
                max(_EPS_DURATION_S, branch.elapsed_s),
                tuple(branch.joint_row_ids),
            )
            if not include_diagnostics:
                return result
            return (
                *result,
                {
                    "complete_macro_recourse": branch.complete_macro_recourse,
                    "incomplete_reason": branch.incomplete_reason,
                    "decision_count": branch.focal_decisions,
                    "macro_event_count": branch.macro_events,
                    "scheduler_kind": "isolated_continuous_time_event_heap",
                    "decision_trace": tuple(branch.decision_trace),
                    "scheduler_trace": tuple(branch.scheduler_trace),
                    "terminal_vehicle_work_s": dict(branch.vehicle_queues),
                    "terminal_virtual_queues": thaw_json(branch.virtual_queues),
                },
            )
        sampled = self._sample_outcome(first_action, observation, branch, rng)
        outcome = self._execute_prediction_action(
            first_action, sampled, observation, branch
        )
        branch.joint_row_ids.append(outcome.row_id)
        branch.cumulative_cost += _task_cost_from_row(
            first_action, observation, self.mask_engine, outcome.values
        )

        if (
            first_action.kind is ActionKind.PIPE
            and branch.focal_decisions < self.horizon_events
        ):
            recourse = self._recourse_action(
                task, observation, first_action, outcome, branch, rng
            )
            if recourse is not None:
                next_action, next_observation = recourse
                sampled_next = self._sample_recourse_outcome(
                    next_action,
                    next_observation,
                    outcome,
                    branch,
                    rng,
                )
                next_outcome = self._execute_prediction_action(
                    next_action, sampled_next, next_observation, branch
                )
                branch.joint_row_ids.append(next_outcome.row_id)
                branch.cumulative_cost += _task_cost_from_row(
                    next_action,
                    next_observation,
                    self.mask_engine,
                    next_outcome.values,
                )

        if branch.focal_decisions < self.horizon_events:
            self._rollout_future_tasks(observation, branch, rng)

        # Continue the isolated branch over future exogenous macro events even
        # after the focal task terminates.  This is the terminal-backlog part of
        # finite-H MPC and makes H>2 responsive to predicted load/fault/thermal
        # evolution without exposing evaluation-trace futures.
        if not branch.use_future_tasks:
            offsets = () if environment is None else environment.macro_event_offsets
            while (
                branch.macro_events < self.horizon_events
                and branch.next_environment_index < len(offsets)
                and branch.slack_s > 1e-12
            ):
                self._advance_branch(
                    branch,
                    offsets[branch.next_environment_index],
                    observation.vehicle_id,
                )

        terminal_drift = self._lyapunov(branch) - initial_lyapunov
        numerator = terminal_drift + self.lyapunov_v * branch.cumulative_cost
        result = (
            numerator,
            max(_EPS_DURATION_S, branch.elapsed_s),
            tuple(branch.joint_row_ids),
        )
        if not include_diagnostics:
            return result
        return (
            *result,
            {
                "complete_macro_recourse": branch.complete_macro_recourse,
                "incomplete_reason": branch.incomplete_reason,
                "decision_count": branch.focal_decisions,
                "macro_event_count": branch.macro_events,
                "scheduler_kind": "legacy_sequential_fallback",
                "decision_trace": tuple(branch.decision_trace),
                "terminal_vehicle_work_s": dict(branch.vehicle_queues),
                "terminal_virtual_queues": thaw_json(branch.virtual_queues),
            },
        )

    def _scores(
        self,
        task: TaskRecord,
        observation: Observation,
        mask: MaskResult,
        state: SimulationState | None,
    ) -> Mapping[Action, float]:
        scores: dict[Action, float] = {}
        diagnostics: dict[str, Any] = {
            "configured_horizon_events": self.horizon_events,
            "approximation_kind": "focal_task_recourse_plus_exogenous_backlog",
            "complete_macro_recourse": False,
            "scenarios": self.scenarios,
            "scenario_seed": self.scenario_seed,
            "rollout_policy": self.rollout_policy,
            "scenario_rows": {},
            "scenario_decisions": {},
            "environment_scenarios": tuple(
                (
                    getattr(
                        self.scenario_source.sample_environment(
                            random.Random(self._seed_for(observation, scenario_index))
                        ),
                        "scenario_id",
                        "relative-environment",
                    )
                    if callable(
                        getattr(self.scenario_source, "sample_environment", None)
                    )
                    else "deterministic-current-observation"
                )
                for scenario_index in range(self.scenarios)
            ),
        }
        observed_numerators: list[float] = []
        observed_durations: list[float] = []
        rollout_completeness: list[bool] = []
        incomplete_reasons: set[str] = set()
        for action in mask.allowed:
            numerators: list[float] = []
            durations: list[float] = []
            rows: list[tuple[str, ...]] = []
            decision_rows: list[tuple[Mapping[str, Any], ...]] = []
            for scenario_index in range(self.scenarios):
                numerator, duration, row_ids, rollout = self._one_rollout(
                    task,
                    observation,
                    action,
                    scenario_index,
                    include_diagnostics=True,
                )
                numerators.append(numerator)
                durations.append(duration)
                observed_numerators.append(numerator)
                observed_durations.append(duration)
                rows.append(row_ids)
                complete = bool(rollout["complete_macro_recourse"])
                rollout_completeness.append(complete)
                if not complete and rollout["incomplete_reason"]:
                    incomplete_reasons.add(str(rollout["incomplete_reason"]))
                decision_rows.append(tuple(rollout["decision_trace"]))
            # Ratio of sample means, not mean of per-scenario ratios.
            scores[action] = sum(numerators) / max(_EPS_DURATION_S, sum(durations))
            diagnostics["scenario_rows"][action.canonical_id] = tuple(rows)
            diagnostics["scenario_decisions"][action.canonical_id] = tuple(
                decision_rows
            )
        complete_macro_recourse = bool(rollout_completeness) and all(
            rollout_completeness
        )
        diagnostics["complete_macro_recourse"] = complete_macro_recourse
        diagnostics["approximation_kind"] = (
            "complete_isolated_continuous_time_macro_event_recourse"
            if complete_macro_recourse
            else "focal_task_recourse_plus_exogenous_backlog"
        )
        diagnostics["incomplete_reasons"] = tuple(sorted(incomplete_reasons))
        bounds = self.scenario_certificate_bounds
        if not bounds:
            diagnostics["scenario_error_certificate"] = {
                "valid": False,
                "reason": "PREREGISTERED_BOUNDS_MISSING",
                "action_count": len(mask.allowed),
                "scenario_count": self.scenarios,
            }
        else:
            required = {
                "numerator_abs_bound",
                "duration_lower_bound_s",
                "duration_upper_bound_s",
            }
            missing = sorted(required - set(bounds))
            if missing:
                diagnostics["scenario_error_certificate"] = {
                    "valid": False,
                    "reason": "PREREGISTERED_BOUNDS_INCOMPLETE",
                    "missing": tuple(missing),
                    "action_count": len(mask.allowed),
                    "scenario_count": self.scenarios,
                }
            else:
                confidence = bounds.get(
                    "confidence_error",
                    0.05
                    if self.mask_engine.config is None
                    else self.mask_engine.config.privacy.confidence_error,
                )
                certificate = finite_scenario_ratio_certificate(
                    action_count=len(mask.allowed),
                    scenario_count=self.scenarios,
                    confidence_error=confidence,
                    numerator_abs_bound=bounds["numerator_abs_bound"],
                    duration_lower_bound_s=bounds["duration_lower_bound_s"],
                    duration_upper_bound_s=bounds["duration_upper_bound_s"],
                )
                certificate_row = certificate.to_dict()
                if any(
                    abs(value) > bounds["numerator_abs_bound"] + 1e-12
                    for value in observed_numerators
                ) or any(
                    value < bounds["duration_lower_bound_s"] - 1e-12
                    or value > bounds["duration_upper_bound_s"] + 1e-12
                    for value in observed_durations
                ):
                    certificate_row.update(
                        {
                            "valid": False,
                            "reason": "OBSERVED_SAMPLE_EXCEEDS_PREREGISTERED_BOUND",
                            "uniform_ratio_error": None,
                            "empirical_argmin_gap_bound": None,
                        }
                    )
                diagnostics["scenario_error_certificate"] = certificate_row
        if not complete_macro_recourse:
            certificate_row = dict(diagnostics["scenario_error_certificate"])
            certificate_row.update(
                {
                    "valid": False,
                    "reason": "MACRO_RECOURSE_INCOMPLETE",
                    "uniform_ratio_error": None,
                    "empirical_argmin_gap_bound": None,
                }
            )
            diagnostics["scenario_error_certificate"] = certificate_row
        self._last_diagnostics = deep_freeze(diagnostics)
        return MappingProxyType(scores)


# Public names used by experiment configuration and older scripts.
FixedSafeNearestRSUPolicy = FixedSafeLowestLinkCostPolicy
H1SafeLyapunovPolicy = SafeLyapunovPolicy
SafeScenarioController = ESLSMPCPolicy


POLICY_REGISTRY: Mapping[str, type[SafePolicy]] = MappingProxyType(
    {
        "all_local": AllLocalPolicy,
        "fixed_safe_lowest_link_cost": FixedSafeLowestLinkCostPolicy,
        "fixed_safe_nearest": FixedSafeLowestLinkCostPolicy,
        "fixed_safe_shortest_visible_queue": FixedSafeShortestQueuePolicy,
        "safe_greedy": SafeGreedyPolicy,
        "safe_lyapunov_h1": SafeLyapunovPolicy,
        "esl_smpc": ESLSMPCPolicy,
        "safe_one_shot": SafeOneShotCommitmentPolicy,
    }
)


__all__ = [
    "AllLocalPolicy",
    "ESLSMPCPolicy",
    "FixedSafeLowestLinkCostPolicy",
    "FixedSafeNearestRSUPolicy",
    "FixedSafeShortestQueuePolicy",
    "H1SafeLyapunovPolicy",
    "POLICY_REGISTRY",
    "Policy",
    "PolicyDecision",
    "SafeGreedyPolicy",
    "SafeLyapunovPolicy",
    "SafeOneShotCommitmentPolicy",
    "SafePolicy",
    "SafeScenarioController",
    "expected_task_cost",
]
