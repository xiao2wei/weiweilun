from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
import json
from pathlib import Path
import random
from types import MethodType, SimpleNamespace

import pytest

from conftest import POLICIES
from privacy_edge_sim.enums import (
    ActionKind,
    ActionStage,
    EventKind,
    FailureReason,
    Operation,
    ReasonCode,
    ResourceKind,
    TaskState,
    TransferDirection,
)
from privacy_edge_sim.estimation import decode_context
from privacy_edge_sim.errors import ConfigError, InvariantViolation
from privacy_edge_sim.policies import (
    POLICY_REGISTRY,
    FixedSafeLowestLinkCostPolicy,
    _globally_safe_fixed_pipeline,
)
from privacy_edge_sim.paper_experiment_audits import audit_failure_cost_completeness
from privacy_edge_sim.metrics import MetricLedger
from privacy_edge_sim.packets import (
    _finalize_encoded_anon,
    _replay_anonymization_success,
    _replay_encoding_success,
    _replay_guard_success,
)
from privacy_edge_sim.profiles import deep_freeze, thaw_json
from privacy_edge_sim.resources import AdmissionRequest, ComputeJob
from privacy_edge_sim.safety import (
    Action,
    MaskResult,
    OnlineDecisionConfigView,
    RemovalRecord,
)
from privacy_edge_sim.simulator import DiscreteEventSimulator
from privacy_edge_sim.state import TaskRecord, TaskStateMachine
from privacy_edge_sim.traces import (
    ExogenousEvent,
    ScenarioEnvironment,
    ScenarioLibrary,
    ScenarioTelemetryEvent,
    ScenarioVersionEvent,
    ScenarioWirelessSegment,
    TraceBundle,
)


def _nested_event_kinds(result) -> set[str]:
    return {
        event["kind"]
        for compound in result.state.event_log
        for event in compound.get("events", ())
    }


def test_branch_scheduler_inherits_focal_live_vehicle_reservations(decision_fixture):
    fixture = decision_fixture(task_id="branch-live-vehicle-state")
    simulator = fixture.simulator
    vehicle_runtime = fixture.state.vehicles[fixture.task.vehicle_id]
    vehicle_runtime.battery_j = 321.0
    local_row = next(
        row
        for row in simulator.trace.local_rows
        if row.device_type == fixture.observation.device_type
        and row.quality_bin == "clear"
        and row.context == decode_context(fixture.observation.device_context)
    )

    def live_local_task(
        task_id: str, deadline_s: float, *, running: bool
    ) -> TaskRecord:
        task = TaskRecord(
            task_id=task_id,
            vehicle_id=fixture.task.vehicle_id,
            arrival_time_s=0.0,
            relative_deadline_s=deadline_s,
            absolute_deadline_s=deadline_s,
            raw_handle=None,
            quality_features=fixture.task.quality_features,
            quality_probabilities=(("clear", 1.0),),
            conformal_quality_bins=("clear",),
            device_context=fixture.observation.device_context,
            selected_local_model=local_row.model_id,
        )
        for target in (
            TaskState.PREP_WAIT,
            TaskState.PREP_RUN,
            TaskState.RAW,
            TaskState.LOCAL_WAIT,
        ):
            TaskStateMachine.transition(
                task, target, time_s=0.0, trigger="UNIT_LIVE_LOCAL"
            )
        if running:
            TaskStateMachine.transition(
                task, TaskState.LOCAL_RUN, time_s=0.0, trigger="UNIT_LIVE_LOCAL_RUN"
            )
        simulator.state.tasks[task_id] = task
        assert vehicle_runtime.reserve(task, {"accelerator": 1}, 16 * 1024 * 1024)
        return task

    # The later-deadline job is already running when an earlier-deadline job
    # arrives.  A non-preemptive server must retain that exact ownership.
    running_task = live_local_task("live-running", 8.0, running=True)
    waiting_task = live_local_task("live-waiting", 5.0, running=False)
    accelerator = vehicle_runtime.resources["accelerator"]
    running_job = ComputeJob(
        job_id="live-running-job",
        task_id=running_task.task_id,
        owner_type="vehicle",
        owner_id=fixture.task.vehicle_id,
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version=local_row.model_hash,
        enqueue_time_s=0.0,
        absolute_deadline_s=running_task.absolute_deadline_s,
        enqueue_seq=accelerator.next_enqueue_seq(),
        total_work_s=2.0,
        residual_work_s=1.25,
        total_dynamic_energy_j=4.0,
        consumed_dynamic_energy_j=1.5,
    )
    accelerator.enqueue(running_job)
    assert accelerator.dispatch(0.0) == [running_job]
    waiting_job = ComputeJob(
        job_id="live-waiting-job",
        task_id=waiting_task.task_id,
        owner_type="vehicle",
        owner_id=fixture.task.vehicle_id,
        operation=Operation.LOCAL_FER,
        resource_kind=ResourceKind.ACCELERATOR,
        model_or_pipeline_version=local_row.model_hash,
        enqueue_time_s=0.25,
        absolute_deadline_s=waiting_task.absolute_deadline_s,
        enqueue_seq=accelerator.next_enqueue_seq(),
        total_work_s=0.5,
        residual_work_s=0.5,
        total_dynamic_energy_j=1.5,
    )
    accelerator.enqueue(waiting_job)
    fixture.state.clock_s = 0.25
    observation = simulator._observation(fixture.task)
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    environment = replace(
        simulator.scenario_library.environment_scenarios[0],
        future_tasks=(),
    )
    branch = policy._new_branch(observation, environment)
    scheduler = policy._scheduler_new(branch, observation, environment)

    assert scheduler is not None
    assert scheduler.vehicle_battery_j[observation.vehicle_id] == 321.0
    assert scheduler.vehicle_memory_reserved[observation.vehicle_id] == 32 * 1024 * 1024
    assert scheduler.descriptor_reserved[observation.vehicle_id] == {
        "accelerator": 2,
        "cpu": 0,
        "encoder": 0,
    }
    active_rows = {
        row["state"]: row
        for row in observation.vehicle["active_tasks"]
        if not row["is_focal"]
    }
    running_token = active_rows[TaskState.LOCAL_RUN.value]["task_token"]
    waiting_token = active_rows[TaskState.LOCAL_WAIT.value]["task_token"]
    restored = scheduler.resources[("vehicle", observation.vehicle_id, "accelerator")]
    assert len(restored.running) == len(restored.waiting) == 1
    restored_running = scheduler.jobs[restored.running[0]]
    restored_waiting = scheduler.jobs[restored.waiting[0]]
    assert restored_running.task_token == running_token
    assert restored_waiting.task_token == waiting_token
    assert restored_running.remaining_work_s == pytest.approx(1.25)
    assert restored_running.total_energy_j == pytest.approx(2.5)
    assert restored_waiting.remaining_work_s == pytest.approx(0.5)
    assert restored_waiting.total_energy_j == pytest.approx(1.5)
    assert restored_running.absolute_deadline_s > restored_waiting.absolute_deadline_s
    assert scheduler.tasks[running_token].state == "LIVE_LOCAL"
    assert scheduler.tasks[waiting_token].state == "LIVE_LOCAL"

    # Calling dispatch again cannot let the earlier-deadline waiter preempt the
    # already-running later-deadline job.
    policy._scheduler_dispatch(scheduler)
    assert scheduler.resources[
        ("vehicle", observation.vehicle_id, "accelerator")
    ].running == [restored_running.job_id]
    assert scheduler.resources[
        ("vehicle", observation.vehicle_id, "accelerator")
    ].waiting == [restored_waiting.job_id]
    assert sum(
        scheduler.jobs[job_id].remaining_work_s
        for job_id in (*restored.running, *restored.waiting)
    ) == pytest.approx(1.75)
    assert sum(
        scheduler.jobs[job_id].total_energy_j
        for job_id in (*restored.running, *restored.waiting)
    ) == pytest.approx(4.0)

    probe = SimpleNamespace(
        vehicle_id=observation.vehicle_id,
        reservation_tokens={},
        reserved_memory_bytes=0,
    )
    assert policy._scheduler_reserve_vehicle(scheduler, probe, {}, 1)


def test_ready_focal_edge_failure_reuses_shadow_then_releases_it(
    decision_fixture, monkeypatch
):
    fixture = decision_fixture(task_id="ready-focal-shadow-release", deadline_s=5.0)
    simulator = fixture.simulator
    task = fixture.task
    edge = next(
        row
        for row in simulator.trace.edge_rows
        if row.rsu_id == "rsu-1" and row.quality_bin == "clear"
    )
    anon = next(
        row
        for row in simulator.trace.anon_rows
        if row.artifact_key == edge.artifact_key
    )
    pipeline = simulator.profile.pipelines[edge.pipeline_id]
    local_memory = max(
        row.memory_bytes
        for row in simulator.trace.local_rows
        if row.model_id == pipeline.fallback_local_model
        and row.device_type == "vehicle_gpu_class_a"
        and row.quality_bin == "clear"
    )
    shadow_memory = max(
        local_memory, max(attempt.peak_memory_bytes for attempt in anon.attempts)
    )
    for target in (
        TaskState.ANON_WAIT,
        TaskState.ANON_RUN,
        TaskState.GUARD_WAIT,
        TaskState.GUARD_RUN,
        TaskState.ENCODE_WAIT,
        TaskState.ENCODE_RUN,
        TaskState.READY,
    ):
        TaskStateMachine.transition(
            task, target, time_s=0.0, trigger="UNIT_READY_SNAPSHOT"
        )
    task.selected_pipeline = edge.pipeline_id
    task.artifact_key = edge.artifact_key
    anonymized = _replay_anonymization_success(
        aligned=task.aligned_handle,
        task_id=task.task_id,
        pipeline_id=pipeline.pipeline_id,
        pipeline_hash=pipeline.pipeline_hash,
        artifact_key=edge.artifact_key,
        attempt=1,
    )
    guarded = _replay_guard_success(
        anonymized,
        guard_hash=pipeline.guard_hash,
        guard_certificate_id="ready-focal-shadow-guard",
    )
    encoded = _replay_encoding_success(
        guarded,
        payload=b"a" * anon.final_encoded_size_bytes,
        encoder_hash=pipeline.encoder_hash,
        encoded_size_bytes=anon.final_encoded_size_bytes,
    )
    task.encoded_anon = _finalize_encoded_anon(
        encoded,
        profile_hash=simulator.profile.profile_hash,
        quality_bins=task.conformal_quality_bins,
    )
    task.encoded_size_bytes = anon.final_encoded_size_bytes
    task.reservation_tokens = {"accelerator": 1, "cpu": 1, "encoder": 1}
    task.memory_reservation_bytes = shadow_memory
    vehicle = simulator.state.vehicles[task.vehicle_id]
    vehicle.memory_reserved_bytes = shadow_memory
    vehicle.descriptors_reserved.update(task.reservation_tokens)
    observation = simulator._observation(task, stage=ActionStage.READY)
    assert observation.vehicle["memory_remaining_bytes"] > 0

    environment = ScenarioEnvironment(
        scenario_id="ready-focal-link-loss",
        cluster_token="ready-focal-link-loss",
        duration_s=5.0,
        wireless=(
            ScenarioWirelessSegment(
                vehicle_id=task.vehicle_id,
                rsu_id=edge.rsu_id,
                direction=TransferDirection.UL,
                start_offset_s=0.0,
                end_offset_s=0.05,
                goodput_bps=500_000.0,
                transmitter_power_w=4.0,
                receiver_power_w=1.0,
                link_state="connected",
            ),
            ScenarioWirelessSegment(
                vehicle_id=task.vehicle_id,
                rsu_id=edge.rsu_id,
                direction=TransferDirection.DL,
                start_offset_s=0.0,
                end_offset_s=5.0,
                goodput_bps=1_000_000.0,
                transmitter_power_w=4.0,
                receiver_power_w=1.0,
                link_state="connected",
            ),
        ),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
        rsu_anchors=simulator.scenario_library.environment_scenarios[0].rsu_anchors,
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    branch = policy._new_branch(observation, environment)
    original_run = policy._scheduler_run
    captured = {}

    def capture_run(scheduler, rng, *, stop_before_next_macro=False):
        focal = scheduler.tasks[task.task_id]
        captured["owned_before"] = (
            dict(focal.reservation_tokens),
            focal.reserved_memory_bytes,
        )
        original_run(scheduler, rng, stop_before_next_macro=stop_before_next_macro)
        captured["scheduler"] = scheduler

    monkeypatch.setattr(policy, "_scheduler_run", capture_run)
    policy._event_heap_rollout(
        observation,
        Action.edge(edge.rsu_id, edge.model_id),
        branch,
        environment,
        random.Random(19),
    )

    scheduler = captured["scheduler"]
    assert captured["owned_before"] == (task.reservation_tokens, shadow_memory)
    assert scheduler.tasks[task.task_id].state == "DONE"
    assert scheduler.vehicle_memory_reserved[task.vehicle_id] == 0
    assert all(
        count == 0 for count in scheduler.descriptor_reserved[task.vehicle_id].values()
    )
    assert any(
        row["kind"] == "DETERMINISTIC_REPAIR"
        and "UPLINK_" in row["reason"]
        and "LOCAL" in row["executed"]
        for row in scheduler.event_trace
    ), scheduler.event_trace


def test_subject_bearing_evaluation_artifact_is_opaque_and_task_scoped(
    decision_fixture,
):
    fixture = decision_fixture(task_id="opaque-evaluation-artifact")
    simulator = fixture.simulator
    task = fixture.task
    edge = next(
        row
        for row in simulator.trace.edge_rows
        if row.rsu_id == "rsu-1" and row.quality_bin == "clear"
    )
    anon = next(
        row
        for row in simulator.trace.anon_rows
        if row.artifact_key == edge.artifact_key
    )
    pipeline = simulator.profile.pipelines[edge.pipeline_id]
    malicious_key = (
        "subject_cluster_id=synthetic-subject-99|"
        "identity=alice@example.test|evaluation-artifact"
    )
    simulator._evaluation_edge_support = frozenset(
        (
            *simulator._evaluation_edge_support,
            (edge.rsu_id, edge.model_id, edge.pipeline_id, malicious_key),
        )
    )
    for target in (
        TaskState.ANON_WAIT,
        TaskState.ANON_RUN,
        TaskState.GUARD_WAIT,
        TaskState.GUARD_RUN,
        TaskState.ENCODE_WAIT,
        TaskState.ENCODE_RUN,
        TaskState.READY,
    ):
        TaskStateMachine.transition(
            task, target, time_s=0.0, trigger="UNIT_OPAQUE_ARTIFACT"
        )
    task.selected_pipeline = edge.pipeline_id
    task.artifact_key = malicious_key
    anonymized = _replay_anonymization_success(
        aligned=task.aligned_handle,
        task_id=task.task_id,
        pipeline_id=pipeline.pipeline_id,
        pipeline_hash=pipeline.pipeline_hash,
        artifact_key=malicious_key,
        attempt=1,
    )
    guarded = _replay_guard_success(
        anonymized,
        guard_hash=pipeline.guard_hash,
        guard_certificate_id="opaque-evaluation-artifact-guard",
    )
    encoded = _replay_encoding_success(
        guarded,
        payload=b"a" * anon.final_encoded_size_bytes,
        encoder_hash=pipeline.encoder_hash,
        encoded_size_bytes=anon.final_encoded_size_bytes,
    )
    task.encoded_anon = _finalize_encoded_anon(
        encoded,
        profile_hash=simulator.profile.profile_hash,
        quality_bins=task.conformal_quality_bins,
    )
    task.encoded_size_bytes = anon.final_encoded_size_bytes

    observation = simulator._observation(task, stage=ActionStage.READY)
    serialized = json.dumps(observation.to_dict(), sort_keys=True)
    action = Action.edge(edge.rsu_id, edge.model_id)
    mask = simulator.mask_engine.enumerate(task, observation, simulator.state)

    assert not hasattr(observation, "artifact_key")
    assert "artifact_key" not in serialized
    assert malicious_key not in serialized
    assert "synthetic-subject-99" not in serialized
    assert "alice@example.test" not in serialized
    assert observation.artifact_token
    assert observation.artifact_token.startswith("artifact-pairing-token:")
    assert observation.encoded_evidence["artifact_token"] == observation.artifact_token
    assert task.artifact_key == malicious_key
    assert action in mask.allowed
    assert malicious_key not in set(_reachable_strings(simulator.policy))
    assert "synthetic-subject-99" not in "|".join(_reachable_strings(simulator.policy))

    replay_task = TaskRecord(
        task_id="cross-task-token-replay",
        vehicle_id=task.vehicle_id,
        arrival_time_s=0.0,
        relative_deadline_s=task.relative_deadline_s,
        absolute_deadline_s=task.absolute_deadline_s,
        raw_handle=None,
        quality_features=task.quality_features,
        quality_probabilities=task.quality_probabilities,
        conformal_quality_bins=task.conformal_quality_bins,
        device_context=task.device_context,
        selected_pipeline=task.selected_pipeline,
    )
    replayed = replace(observation, task_id=replay_task.task_id)
    replay_mask = simulator.mask_engine.enumerate(replay_task, replayed)

    assert action not in replay_mask.allowed
    assert ReasonCode.PAIRED_MEASUREMENT_MISSING in replay_mask.reasons_for(action)


def test_branch_future_vehicle_observation_uses_own_frozen_template(decision_fixture):
    fixture = decision_fixture(task_id="branch-heterogeneous-vehicle")
    focal_vehicle = thaw_json(fixture.observation.vehicle)
    focal_vehicle["battery_j"] = 123.0
    observation = replace(fixture.observation, vehicle=deep_freeze(focal_vehicle))
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    environment = fixture.simulator.scenario_library.environment_scenarios[0]
    branch = policy._new_branch(observation, environment)
    scheduler = policy._scheduler_new(branch, observation, environment)

    assert scheduler is not None
    future_task = next(
        task for task in scheduler.tasks.values() if task.vehicle_id == "veh-2"
    )
    future_observation = policy._scheduler_observation(
        scheduler, future_task, ActionStage.RAW
    )
    configured = fixture.simulator.mask_engine.config.vehicle_branch_parameters["veh-2"]
    anchor = next(
        item for item in environment.vehicle_anchors if item.vehicle_id == "veh-2"
    )

    assert future_observation.vehicle_id == "veh-2"
    assert future_observation.device_type == configured["device_type"]
    assert future_observation.vehicle["battery_j"] == anchor.battery_j
    assert (
        future_observation.vehicle["memory_capacity_bytes"]
        == configured["memory_capacity_bytes"]
    )
    assert (
        future_observation.vehicle["descriptor_capacity"]
        == configured["descriptor_capacity"]
    )
    assert future_observation.vehicle["battery_j"] != 123.0
    assert (
        future_observation.vehicle["memory_capacity_bytes"]
        != observation.vehicle["memory_capacity_bytes"]
    )
    assert not {
        "true_identity",
        "true_expression_label",
        "realized_attack_outcomes",
    }.intersection(future_observation.vehicle)

    scheduler.vehicle_battery_j["veh-2"] -= 7.0
    updated = policy._scheduler_observation(scheduler, future_task, ActionStage.RAW)
    assert updated.vehicle["battery_j"] == anchor.battery_j - 7.0


def test_branch_future_vehicle_inherits_nonzero_causal_anchor_workload(
    decision_fixture,
):
    fixture = decision_fixture(task_id="branch-causal-anchor-workload")
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    environment = next(
        item
        for item in fixture.simulator.scenario_library.environment_scenarios
        if any(
            anchor.vehicle_id == "veh-2"
            and anchor.active_task_count > 0
            and sum(
                float(resource["residual_work_s"])
                for resource in anchor.resources.values()
            )
            > 0
            for anchor in item.vehicle_anchors
        )
        and any(task.vehicle_id == "veh-2" for task in item.future_tasks)
    )
    anchor = next(
        item for item in environment.vehicle_anchors if item.vehicle_id == "veh-2"
    )
    configured = fixture.simulator.mask_engine.config.vehicle_branch_parameters["veh-2"]
    branch = policy._new_branch(fixture.observation, environment)
    scheduler = policy._scheduler_new(branch, fixture.observation, environment)

    assert anchor.complete_support
    assert 0 < anchor.battery_j < configured["initial_battery_j"]
    assert anchor.active_task_count == len(anchor.tasks) == 1
    assert anchor.memory_reserved_bytes > 0
    assert sum(anchor.descriptors_reserved.values()) > 0
    assert scheduler is not None
    assert scheduler.vehicle_battery_j["veh-2"] == pytest.approx(anchor.battery_j)
    accelerator = scheduler.resources[("vehicle", "veh-2", "accelerator")]
    assert len(accelerator.running) == 1
    restored = scheduler.jobs[accelerator.running[0]]
    assert restored.remaining_work_s == pytest.approx(
        anchor.resources["accelerator"]["residual_work_s"]
    )
    assert restored.total_energy_j > 0
    assert restored.task_token == anchor.tasks[0].task_token


def test_branch_missing_nonfocal_vehicle_anchor_is_conservatively_incomplete(
    decision_fixture,
):
    fixture = decision_fixture(task_id="branch-missing-causal-anchor")
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    source = fixture.simulator.scenario_library.environment_scenarios[0]
    assert any(task.vehicle_id == "veh-2" for task in source.future_tasks)
    environment = replace(
        source,
        vehicle_anchors=tuple(
            anchor for anchor in source.vehicle_anchors if anchor.vehicle_id != "veh-2"
        ),
    )
    branch = policy._new_branch(fixture.observation, environment)

    assert policy._scheduler_new(branch, fixture.observation, environment) is None
    assert branch.incomplete_reason == "SCENARIO_VEHICLE_ANCHOR_INCOMPLETE"


@pytest.mark.parametrize("policy_name", POLICIES)
def test_all_six_policies_use_same_hard_mask_and_repair_gate(
    policy_name, decision_fixture
):
    fixture = decision_fixture()
    policy_type = POLICY_REGISTRY[policy_name]
    kwargs = (
        {"scenario_source": fixture.simulator.scenario_library}
        if policy_name == "esl_smpc"
        else {}
    )
    policy = policy_type(
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        **kwargs,
    )

    decision = policy.decide(fixture.task, fixture.observation, fixture.state)

    assert policy.mask_engine is fixture.simulator.mask_engine
    assert policy.repairer is fixture.simulator.repairer
    assert decision.executed in decision.mask.allowed
    assert decision.executed not in decision.mask.removed
    assert decision.repair.executed == decision.executed
    assert set(decision.scores) <= set(decision.mask.allowed)


def test_online_policy_view_exposes_only_frozen_budget_parameters(decision_fixture):
    fixture = decision_fixture(task_id="budget-view")
    view = fixture.simulator.mask_engine.config

    assert isinstance(view, OnlineDecisionConfigView)
    assert view.vehicle_power_budgets_w == {
        row.vehicle_id: row.average_power_budget_w
        for row in fixture.simulator.config.vehicles
    }
    assert view.rsu_power_budgets_w == {
        row.rsu_id: row.average_power_budget_w for row in fixture.simulator.config.rsus
    }
    assert (
        view.long_term.failure_rate_limit
        == fixture.simulator.config.long_term.failure_rate_limit
    )
    with pytest.raises(TypeError):
        view.vehicle_power_budgets_w[fixture.task.vehicle_id] = 0.0
    with pytest.raises(TypeError):
        view.rsu_power_budgets_w["rsu-1"] = 0.0


def test_h1_virtual_drift_matches_projected_queue_bank_increments(decision_fixture):
    fixture = decision_fixture(task_id="h1-queue-increments")
    policy = POLICY_REGISTRY["safe_lyapunov_h1"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
    )
    observation = replace(
        fixture.observation,
        virtual_queues={
            "vehicle_power": {fixture.task.vehicle_id: 10.0},
            "rsu_power": {"rsu-1": 7.0},
            "timeout": 3.0,
            "failure": 4.0,
            "coverage": 5.0,
        },
    )
    action = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))
    row = {
        "expected_duration_s": 2.0,
        "expected_vehicle_energy_j": 50.0,
        "timeout_probability": 0.25,
        "failure_probability": 0.5,
        "completion_probability": 0.5,
        "expected_arrivals": 2.0,
    }
    config = fixture.simulator.config
    budget = next(
        item.average_power_budget_w
        for item in config.vehicles
        if item.vehicle_id == fixture.task.vehicle_id
    )

    def projected_increment(queue, delta):
        after = max(0.0, queue + delta)
        return 0.5 * (after * after - queue * queue)

    expected = sum(
        (
            projected_increment(10.0, 50.0 - budget * 2.0),
            projected_increment(
                3.0,
                0.25 - config.long_term.timeout_rate_limit * 2.0,
            ),
            projected_increment(
                4.0,
                0.5 - config.long_term.failure_rate_limit * 2.0,
            ),
            projected_increment(
                5.0,
                config.long_term.coverage_rate_minimum * 2.0 - 0.5,
            ),
        )
    )
    assert policy._virtual_drift(action, observation, row) == pytest.approx(expected)


def test_h1_uses_configured_cost_not_provider_expected_cost(decision_fixture):
    fixture = decision_fixture(task_id="h1-configured-cost")
    policy = POLICY_REGISTRY["safe_lyapunov_h1"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
    )
    action = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))
    common = {
        "expected_duration_s": 0.4,
        "expected_vehicle_energy_j": 2.0,
        "expected_rsu_energy_j": 0.0,
        "expected_fer_loss": 0.2,
        "failure_probability": 0.1,
        "completion_probability": 0.9,
        "vehicle_work_s": 0.4,
    }
    low = policy.score_action(
        action,
        fixture.observation,
        outcome_override={**common, "expected_cost": -1e12},
    )
    high = policy.score_action(
        action,
        fixture.observation,
        outcome_override={**common, "expected_cost": 1e12},
    )
    assert low == pytest.approx(high)


def test_h1_supports_per_resource_theta(decision_fixture):
    fixture = decision_fixture(task_id="h1-resource-theta")
    action = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))
    row = {"vehicle_work_s": {"accelerator": 0.5}}
    unit = POLICY_REGISTRY["safe_lyapunov_h1"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        physical_queue_weight=0.0,
        vehicle_resource_theta={"accelerator": 1.0},
    )
    double = POLICY_REGISTRY["safe_lyapunov_h1"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        physical_queue_weight=0.0,
        vehicle_resource_theta={"accelerator": 2.0},
    )
    first = unit._physical_drift(action, fixture.observation, row)
    assert first > 0
    assert double._physical_drift(action, fixture.observation, row) == pytest.approx(
        2.0 * first
    )


@pytest.mark.parametrize(
    "policy_name",
    ["fixed_safe_lowest_link_cost", "fixed_safe_shortest_visible_queue"],
)
def test_fixed_safe_baselines_freeze_pipeline_and_model_once(
    policy_name, decision_fixture
):
    first = decision_fixture(task_id=f"{policy_name}-one")
    second = decision_fixture(task_id=f"{policy_name}-two")
    policy = POLICY_REGISTRY[policy_name](
        first.simulator.mask_engine,
        first.simulator.repairer,
    )
    frozen_pipeline = _globally_safe_fixed_pipeline(first.simulator.mask_engine)
    frozen_model = min(first.simulator.profile.edge_models)
    assert policy.pipeline_id == frozen_pipeline
    assert policy.edge_model_id == frozen_model

    for fixture in (first, second):
        decision = policy.decide(fixture.task, fixture.observation, fixture.state)
        if decision.proposed.pipeline_id is not None:
            assert decision.proposed.pipeline_id == frozen_pipeline
        if decision.proposed.edge_model_id is not None:
            assert decision.proposed.edge_model_id == frozen_model


def test_esl_diagnostics_withdraw_uncertified_performance_bound(decision_fixture):
    fixture = decision_fixture(task_id="esl-certificate")
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    decision = policy.decide(fixture.task, fixture.observation, fixture.state)

    assert decision.diagnostics["complete_macro_recourse"] is True, (
        decision.diagnostics["incomplete_reasons"]
    )
    assert (
        decision.diagnostics["approximation_kind"]
        == "complete_isolated_continuous_time_macro_event_recourse"
    )
    certificate = decision.diagnostics["scenario_error_certificate"]
    assert certificate["valid"] is False
    assert certificate["reason"] == "PREREGISTERED_BOUNDS_MISSING"


def test_every_policy_runs_same_frozen_workload_to_absorbing_states(
    policy_results, trace
):
    expected_tasks = {arrival.task_id for arrival in trace.arrivals}
    for policy_name in POLICIES:
        result = policy_results[policy_name]
        assert set(result.state.tasks) == expected_tasks
        assert result.profile is not None
        assert result.trace is trace
        assert not result.invariant_failures
        assert result.state.invariant_checks > 0
        assert all(task.terminal for task in result.state.tasks.values())
        assert result.state.event_log
        assert any(task.action_audit for task in result.state.tasks.values())
        assert result.ledger.resource_rows
        assert result.ledger.virtual_queue_rows


def test_policy_order_does_not_change_frozen_environment(policy_results):
    reference = policy_results[POLICIES[0]]
    reference_external = {
        kind
        for kind in _nested_event_kinds(reference)
        if kind
        in {"ARRIVAL", "LINK_CHANGE", "THERMAL_CHANGE", "DEVICE_FAULT", "MODEL_VERSION"}
    }
    for policy_name in reversed(POLICIES):
        result = policy_results[policy_name]
        observed = {
            kind
            for kind in _nested_event_kinds(result)
            if kind
            in {
                "ARRIVAL",
                "LINK_CHANGE",
                "THERMAL_CHANGE",
                "DEVICE_FAULT",
                "MODEL_VERSION",
            }
        }
        assert observed == reference_external


def test_all_local_baseline_exercises_local_success_and_prep_failure(policy_results):
    result = policy_results["all_local"]
    local = result.state.tasks["task-001"]
    failed = result.state.tasks["task-007"]

    assert local.state is TaskState.DONE
    assert local.actual_path == ["PREP", "LOCAL_FER"]
    assert local.result_valid
    assert local.vehicle_energy_j > 0
    assert local.rsu_energy_j == 0
    assert failed.state is TaskState.FAIL
    assert failed.failure_reason is FailureReason.PREP_FAILED


def test_fixed_safe_policy_exercises_multi_attempt_edge_round_trip(
    policy_results, profile
):
    result = policy_results["fixed_safe_lowest_link_cost"]
    task = result.state.tasks["task-001"]

    assert task.state is TaskState.DONE
    assert task.attempt_started_count == 2
    assert [row["attempt"] for row in task.anon_attempt_audit] == [1, 2]
    assert task.anon_attempt_audit[0]["guard_passed"] is False
    assert task.anon_attempt_audit[1]["guard_passed"] is True
    assert task.anon_attempt_audit[1]["encode_success"] is True
    assert task.anon_attempt_audit[1]["encoded_size_bytes"] > 0
    assert all(row["latency_s"] > 0 for row in task.anon_attempt_audit)
    assert all(row["vehicle_energy_j"] > 0 for row in task.anon_attempt_audit)
    assert task.anon_attempt_audit[0]["failure_reason"] == "GUARD_REJECTED"
    assert task.anon_attempt_audit[1]["failure_reason"] is None
    assert any(phase.startswith("UL:") for phase in task.actual_path)
    assert any(phase.startswith("EDGE:") for phase in task.actual_path)
    assert any(phase.startswith("DL:") for phase in task.actual_path)
    assert [(row["direction"], row["status"]) for row in task.network_audit] == [
        ("UL", "START"),
        ("UL", "DONE"),
        ("DL", "START"),
        ("DL", "DONE"),
    ]
    assert (
        task.network_audit[1]["delivered_bits"] == task.network_audit[0]["total_bits"]
    )
    assert (
        task.network_audit[3]["delivered_bits"] == task.network_audit[2]["total_bits"]
    )
    assert task.vehicle_energy_j > 0 and task.rsu_energy_j > 0
    admission = next(row for row in task.rsu_audit if "admission" in row)
    assert admission["admission"] == "ACCEPT"
    assert (
        admission["pinned_model_hash"]
        == profile.edge_models["edge_fer_full_v1"].model_hash
    )


def test_lowest_link_cost_reads_production_observation_ul_goodput(decision_fixture):
    fixture = decision_fixture(task_id="production-link-cost-field")
    action_1 = Action.edge("rsu-1", min(fixture.simulator.profile.edge_models))
    action_2 = Action.edge("rsu-2", min(fixture.simulator.profile.edge_models))
    links = {key: dict(value) for key, value in fixture.observation.links.items()}
    assert "ul_goodput_bps" in links["rsu-1"]
    assert "goodput_bps" not in links["rsu-1"]
    links["rsu-1"]["ul_goodput_bps"] = 8_000_000.0
    links["rsu-2"]["ul_goodput_bps"] = 2_000_000.0
    observation = replace(fixture.observation, links=links)

    assert FixedSafeLowestLinkCostPolicy._link_cost(
        action_1, observation
    ) < FixedSafeLowestLinkCostPolicy._link_cost(action_2, observation)


def test_one_shot_and_safe_greedy_share_the_same_raw_first_action(decision_fixture):
    fixture = decision_fixture(task_id="one-shot-raw-parity", deadline_s=10.0)
    greedy = POLICY_REGISTRY["safe_greedy"](
        fixture.simulator.mask_engine, fixture.simulator.repairer
    )
    one_shot = POLICY_REGISTRY["safe_one_shot"](
        fixture.simulator.mask_engine, fixture.simulator.repairer
    )

    greedy_decision = greedy.decide(fixture.task, fixture.observation, fixture.state)
    one_shot_decision = one_shot.decide(
        fixture.task, fixture.observation, fixture.state
    )

    assert one_shot_decision.executed == greedy_decision.executed
    diagnostics = one_shot_decision.diagnostics
    if one_shot_decision.executed.kind.value == "PIPE":
        assert diagnostics["commitment"]["pipeline_id"] == (
            one_shot_decision.executed.pipeline_id
        )
        assert diagnostics["commitment"]["snapshot_is_reservation"] is False
    else:
        assert diagnostics["commitment"] is None


def test_one_shot_end_to_end_commitment_and_changed_state_safe_repair(
    config,
    profile,
    trace,
):
    control_simulator = DiscreteEventSimulator(config, profile, trace, "safe_one_shot")
    control = control_simulator.run()
    committed = None
    for task in sorted(control.state.tasks.values(), key=lambda item: item.task_id):
        raw = next(
            (
                row["diagnostics"]
                for row in task.action_audit
                if isinstance(row.get("diagnostics"), Mapping)
                and row["diagnostics"].get("phase") == "RAW_COMMIT"
                and row["diagnostics"].get("commitment")
            ),
            None,
        )
        ready = next(
            (
                row["diagnostics"]
                for row in task.action_audit
                if isinstance(row.get("diagnostics"), Mapping)
                and row["diagnostics"].get("phase") == "READY_ATTEMPT"
            ),
            None,
        )
        if raw is not None and ready is not None:
            committed = (task, raw["commitment"], ready)
            break
    assert committed is not None, "fixture must exercise a complete one-shot path"
    control_task, commitment, ready_diagnostics = committed
    assert ready_diagnostics["outcome"] == "COMMITTED_EDGE"
    assert commitment["snapshot_is_reservation"] is False
    assert not control_simulator.policy._commitments

    ready_time = next(
        row.time_s
        for row in control_task.phase_history
        if row.current is TaskState.READY
    )
    fault_time = (float(commitment["committed_at_s"]) + float(ready_time)) / 2.0
    fault = ExogenousEvent(
        event_id="one-shot-state-change",
        time_s=fault_time,
        event_type="DEVICE_FAULT_PERMANENT",
        target_type="rsu",
        target_id=str(commitment["rsu_id"]),
        resource="all",
        old_version=None,
        new_version=None,
        permanent=True,
        details={},
    )
    changed_trace = replace(
        trace,
        exogenous_events=tuple(
            sorted(
                (*trace.exogenous_events, fault),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    changed_simulator = DiscreteEventSimulator(
        config, profile, changed_trace, "safe_one_shot"
    )
    changed = changed_simulator.run()
    changed_task = changed.state.tasks[control_task.task_id]
    repaired = next(
        row["diagnostics"]
        for row in changed_task.action_audit
        if isinstance(row.get("diagnostics"), Mapping)
        and row["diagnostics"].get("phase") == "READY_ATTEMPT"
    )

    # Policies only see the last delivered public snapshot.  This permanent
    # fault occurs after RAW commitment but before its telemetry is delivered,
    # so the READY hard mask can still regard the commitment as visible-safe.
    # The simulator-side execution gate must nevertheless recheck live state
    # before constructing an uplink and apply the frozen local repair.
    assert repaired["hard_mask_allowed"] is True
    assert repaired["outcome"] == "COMMITTED_EDGE"
    assert not any(
        step == f"UL:{commitment['rsu_id']}" for step in changed_task.actual_path
    )
    fallback = next(
        row
        for row in changed_task.action_audit
        if row.get("repair") == "FROZEN_LOCAL_FALLBACK"
    )
    assert fallback["trigger"] == "EDGE_FAILED"
    assert changed_task.actual_path[-1] == "LOCAL_FER"
    assert not changed_simulator.policy._commitments


def test_failure_integrity_audit_uses_measured_retry_downlink_and_rsu_costs(
    policy_results,
):
    result = policy_results["fixed_safe_lowest_link_cost"]
    rows = MetricLedger.task_rows(result.state, result.config)
    normalized = []
    for row in rows:
        item = dict(row)
        item["anon_attempts"] = json.loads(item["anon_attempts_json"])
        item["network_audit"] = json.loads(item["network_audit_json"])
        normalized.append(item)
    report = audit_failure_cost_completeness(
        normalized,
        result.ledger.action_rows,
        result.ledger.event_rows,
    )

    assert report["omissions"]["retry"]["tasks_affected"] >= 1
    assert report["omissions"]["retry"]["vehicle_energy_j"] > 0
    assert report["omissions"]["downlink"]["tasks_affected"] >= 1
    assert report["omissions"]["downlink"]["latency_s"] > 0
    assert report["omissions"]["rsu_energy"]["rsu_energy_j"] > 0
    assert report["omissions"]["anonymization_failure"]["failed_attempt_count"] >= 1
    assert report["omissions"]["anonymization_failure"]["executed_work_s"] > 0
    assert report["omissions"]["anonymization_failure"]["vehicle_energy_j"] > 0
    assert (
        report["omissions"]["anonymization_failure"]["reason_counts"]["GUARD_REJECTED"]
        >= 1
    )
    assert (
        report["omissions"]["local_fallback"]["accounting"] == "exact_structural_count"
    )
    assert (
        report["omissions"]["explicit_fail_action"]["accounting"]
        == "exact_POLICY_DECISION_executed_count"
    )


def test_synthetic_event_loop_audits_interruptions_thermal_faults_and_version(
    policy_results, profile
):
    result = policy_results["fixed_safe_lowest_link_cost"]
    kinds = _nested_event_kinds(result)

    assert {
        "LINK_CHANGE",
        "THERMAL_CHANGE",
        "DEVICE_FAULT",
        "MODEL_VERSION",
        EventKind.RSU_SNAPSHOT.value,
    } <= kinds
    rsu_1_hash = result.state.rsus["rsu-1"].admission.cached_models["edge_fer_full_v1"]
    assert rsu_1_hash != profile.edge_models["edge_fer_full_v1"].model_hash
    assert result.state.rsus["rsu-2"].failed is False
    link_states = {segment.link_state for segment in result.trace.wireless}
    assert {"temporary_outage", "handover", "permanent_loss"} <= link_states
    assert any(segment.service_rate_multiplier < 1 for segment in result.trace.thermal)


def test_policy_callback_receives_only_minimal_views(decision_fixture):
    fixture = decision_fixture(task_id="policy-isolation")

    class InspectingPolicy:
        def __init__(self):
            self.task_view = None
            self.state_view = None

        def choose_action(self, task_view, observation, state_view):
            self.task_view = task_view
            self.state_view = state_view
            return Action.fail(observation.stage)

    policy = InspectingPolicy()
    fixture.simulator.policy = policy
    chosen = fixture.simulator._policy_choose(
        fixture.task,
        fixture.observation,
        fixture.simulator.mask_engine.enumerate(
            fixture.task,
            fixture.observation,
            fixture.state,
        ),
    )

    assert chosen == Action.fail(fixture.observation.stage)
    assert not hasattr(policy.task_view, "raw_handle")
    assert not hasattr(policy.task_view, "true_identity")
    assert not hasattr(policy.task_view, "realized_fer_loss")
    assert not hasattr(policy.state_view, "tasks")
    assert not hasattr(policy.state_view, "transfers")
    assert [field.name for field in fields(policy.state_view)] == ["clock_s"]
    assert policy.state_view.clock_s == fixture.state.clock_s


@pytest.mark.parametrize("policy_name", ("safe_lyapunov_h1", "esl_smpc"))
def test_delayed_execution_repair_preserves_policy_alternative_scores(
    decision_fixture,
    policy_name,
    monkeypatch,
):
    """A stale first choice is repaired with H=1/ESL scores, not generic costs."""

    # Pixelate is feasible at t=0 but its optimistic duration exceeds the
    # remaining slack after the frozen 0.8 ms controller overhead.  Blur and
    # local remain feasible at commit time.
    fixture = decision_fixture(
        task_id=f"delayed-policy-score-{policy_name}",
        deadline_s=0.0697,
    )
    simulator = fixture.simulator
    simulator.policy = POLICY_REGISTRY[policy_name](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    pixelate = Action.pipeline("pixelate_strong_v1")
    blur = Action.pipeline("blur_balanced_v1")
    local = Action.local(ActionStage.RAW, "local_fer_compact_v1")

    # Model a capacity condition that removes blur only at the decision epoch.
    # It is safe again at commit time, so it is intentionally absent from the
    # frozen decision-epoch score map and can only win after a current rescore.
    real_enumerate = simulator.mask_engine.enumerate

    def changing_mask(task, observation, state=None):
        result = real_enumerate(task, observation, state)
        if observation.time_s > 0 or blur not in result.allowed:
            return result
        removed = dict(result.removed)
        removed[blur] = (ReasonCode.VEHICLE_CAPACITY,)
        records = dict(result.records)
        records[blur] = RemovalRecord(
            blur,
            (ReasonCode.VEHICLE_CAPACITY,),
        )
        return MaskResult(
            stage=result.stage,
            candidates=result.candidates,
            allowed=tuple(action for action in result.allowed if action != blur),
            removed=removed,
            records=records,
        )

    monkeypatch.setattr(simulator.mask_engine, "enumerate", changing_mask)

    score_calls = {"count": 0}

    def policy_scores(_self, _task, _observation, mask, _state):
        score_calls["count"] += 1
        diagnostic_field = (
            "_last_h1_diagnostics"
            if policy_name == "safe_lyapunov_h1"
            else "_last_diagnostics"
        )
        setattr(
            _self,
            diagnostic_field,
            deep_freeze({"score_call": score_calls["count"]}),
        )
        rank = {
            pixelate: 0.0,
            blur: 1.0,
            local: 2.0,
            Action.fail(ActionStage.RAW): 100.0,
        }
        return {action: rank[action] for action in mask.allowed}

    simulator.policy._scores = MethodType(policy_scores, simulator.policy)
    initial_mask = simulator.mask_engine.enumerate(
        fixture.task, simulator._observation(fixture.task), fixture.state
    )
    assert {pixelate, local} <= set(initial_mask.allowed)
    assert blur not in initial_mask.allowed

    simulator._make_decisions()

    assert simulator._pending_decisions[fixture.task.task_id] == pixelate
    assert blur not in simulator._pending_decision_scores[fixture.task.task_id]
    decision_epoch_diagnostics = simulator.policy._diagnostics()
    assert decision_epoch_diagnostics["score_call"] == 1
    dispatch = next(
        event
        for event in simulator.state.events.cancel_task(fixture.task.task_id)
        if event.kind is EventKind.DISPATCH_DECISION
    )
    simulator.state.clock_s = dispatch.time_s

    current_observation = simulator._observation(fixture.task)
    current_mask = simulator.mask_engine.enumerate(
        fixture.task, current_observation, fixture.state
    )
    assert pixelate not in current_mask.allowed
    assert {blur, local} <= set(current_mask.allowed)
    # This confirms the fixture distinguishes the desired policy ranking from
    # the repairer's generic LOCAL-before-PIPE fallback estimate.
    assert (
        simulator.repairer.repair(
            pixelate,
            fixture.task,
            current_observation,
            fixture.state,
        ).executed
        == local
    )

    simulator._handle_decision_commit(dispatch)

    assert fixture.task.state is TaskState.ANON_WAIT
    assert fixture.task.selected_pipeline == "blur_balanced_v1"
    assert fixture.task.selected_local_model is None
    assert fixture.task.task_id not in simulator._pending_decision_scores
    assert score_calls["count"] == 2
    assert simulator.policy._diagnostics() == decision_epoch_diagnostics
    commit_audit = next(
        row
        for row in reversed(fixture.task.action_audit)
        if row.get("executed_stage") == ActionStage.RAW.value
    )
    assert commit_audit["changed"] is True
    assert commit_audit["repair_score_source"] == "current_policy_rescore"
    assert commit_audit["executed"]["kind"] == ActionKind.PIPE.value
    assert commit_audit["executed"]["pipeline_id"] == "blur_balanced_v1"
    assert commit_audit["scores"][blur.canonical_id] == 1.0
    assert commit_audit["scores"][local.canonical_id] == 2.0


def test_controller_scenario_library_is_separate_and_has_no_future_environment(
    decision_fixture,
):
    fixture = decision_fixture(task_id="scenario-isolation")
    simulator = fixture.simulator

    assert simulator.scenario_trace.source_path != simulator.trace.source_path
    assert simulator.scenario_library.split_role == "training_validation"
    assert not hasattr(simulator.scenario_library, "arrivals")
    assert not hasattr(simulator.scenario_library, "wireless")
    assert not hasattr(simulator.scenario_library, "exogenous_events")
    assert not hasattr(simulator.scenario_library, "metadata")
    assert not hasattr(simulator.scenario_library, "source_path")
    assert not hasattr(simulator.estimator, "evaluation_trace")
    assert not hasattr(simulator.estimator, "trace")
    assert not hasattr(simulator.estimator, "config")
    assert simulator.estimator.requires_evaluation_pair is True
    assert not hasattr(simulator.estimator, "evaluation_edge_support")


def _reachable_policy_objects(root):
    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            yield value
            continue
        if isinstance(value, (bytes, int, float, bool, type(None), Enum, Path)):
            continue
        if callable(value):
            continue
        marker = id(value)
        if marker in seen:
            continue
        seen.add(marker)
        yield value
        if isinstance(value, Mapping):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif is_dataclass(value):
            pending.extend(getattr(value, item.name) for item in fields(value))
        else:
            if hasattr(value, "__dict__"):
                pending.extend(vars(value).values())
            slots = getattr(type(value), "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            pending.extend(
                getattr(value, slot)
                for slot in slots
                if isinstance(slot, str) and hasattr(value, slot)
            )


def test_policy_reachable_scenario_objects_have_no_trace_or_subject_identity(
    decision_fixture,
):
    fixture = decision_fixture(task_id="scenario-object-graph")
    simulator = fixture.simulator
    policy_type = POLICY_REGISTRY["esl_smpc"]
    policy = policy_type(
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    policy.decide(fixture.task, fixture.observation, fixture.state)

    reachable = tuple(_reachable_policy_objects(policy))
    assert not any(isinstance(value, TraceBundle) for value in reachable)
    assert isinstance(policy.mask_engine.config, OnlineDecisionConfigView)
    assert set(policy.mask_engine.config.seeds) == {"scenario"}
    reachable_strings = {value for value in reachable if isinstance(value, str)}
    evaluation_artifacts = {
        row.artifact_key for row in simulator.trace.edge_rows if row.artifact_key
    }
    evaluation_subjects = {row.subject_cluster_id for row in simulator.trace.anon_rows}
    assert evaluation_artifacts.isdisjoint(reachable_strings)
    assert evaluation_subjects.isdisjoint(reachable_strings)
    for value in reachable:
        names = {item.name for item in fields(value)} if is_dataclass(value) else set()
        names.update(vars(value) if hasattr(value, "__dict__") else ())
        assert "subject_cluster_id" not in names
        assert "fixture_key" not in names
        assert "row_id" not in names
        assert not {
            "trace_path",
            "scenario_trace_path",
            "vehicles",
            "rsus",
        }.intersection(names)

    with pytest.raises(TypeError, match="identity-free ScenarioLibrary"):
        policy_type(
            simulator.mask_engine,
            simulator.repairer,
            scenario_source=simulator.trace,
        )


def test_simulator_rejects_evaluation_trace_as_scenario(config, profile, trace):
    with pytest.raises(
        ConfigError, match="separate training/validation scenario trace"
    ):
        DiscreteEventSimulator(
            config,
            profile,
            trace,
            "esl_smpc",
            scenario_trace=trace,
        )


def test_scenario_library_preserves_joint_pairing_with_opaque_tokens(decision_fixture):
    simulator = decision_fixture(task_id="scenario-pairing").simulator
    library = simulator.scenario_library
    assert isinstance(library, ScenarioLibrary)

    source_artifacts = {
        row.artifact_key
        for row in simulator.scenario_trace.anon_rows
        if row.artifact_key is not None
    }
    scenario_tokens = {
        row.artifact_token
        for row in library.anon_rows
        if row.artifact_token is not None
    }
    assert scenario_tokens
    assert scenario_tokens.isdisjoint(source_artifacts)
    assert all(not hasattr(row, "subject_cluster_id") for row in library.anon_rows)
    assert all(not hasattr(row, "subject_cluster_id") for row in library.local_rows)
    assert all(
        not hasattr(row, "row_id")
        for row in (*library.anon_rows, *library.local_rows, *library.edge_rows)
    )

    anon_by_token = {
        row.artifact_token: row
        for row in library.anon_rows
        if row.formed_packet and row.artifact_token is not None
    }
    assert anon_by_token
    for edge in library.edge_rows:
        anon = anon_by_token[edge.artifact_token]
        measurement = anon.fer_measurements[edge.model_id]
        assert edge.pipeline_id == anon.pipeline_id
        assert edge.quality_bin == anon.quality_bin
        assert edge.model_hash == measurement.model_hash
        assert edge.fer_loss == measurement.fer_loss


def test_scenario_library_exposes_only_relative_joint_environment(decision_fixture):
    library = decision_fixture(
        task_id="relative-environment"
    ).simulator.scenario_library

    assert library.environment_scenarios
    assert any(item.wireless for item in library.environment_scenarios)
    assert any(item.thermal for item in library.environment_scenarios)
    assert any(item.faults for item in library.environment_scenarios)
    assert any(item.background_loads for item in library.environment_scenarios)
    assert any(item.telemetry for item in library.environment_scenarios)
    for item in library.environment_scenarios:
        assert not hasattr(item, "anchor_time_s")
        assert not hasattr(item, "source_task_id")
        assert 0 < item.duration_s
        assert all(
            0 <= row.start_offset_s < row.end_offset_s <= item.duration_s + 1e-12
            for row in (*item.wireless, *item.thermal)
        )


def test_scenario_library_freezes_telemetry_schedule_and_version_events(
    decision_fixture,
):
    fixture = decision_fixture(task_id="scenario-telemetry-version-library")
    source = fixture.simulator.scenario_trace
    profile = fixture.simulator.profile
    model = profile.edge_models[min(profile.edge_models)]
    rsu_id = min(model.supported_rsus)
    added = (
        ExogenousEvent(
            "scenario-profile-version",
            0.31,
            "PROFILE_VERSION",
            "deployment",
            "global",
            "profile",
            profile.profile_hash,
            "future-profile",
            False,
            {},
        ),
        ExogenousEvent(
            "scenario-protocol-version",
            0.32,
            "PROTOCOL_VERSION",
            "deployment",
            "global",
            "protocol",
            profile.protocol_version,
            "future-protocol",
            False,
            {},
        ),
        ExogenousEvent(
            "scenario-model-version",
            0.33,
            "MODEL_VERSION",
            "rsu",
            rsu_id,
            "gpu",
            model.model_hash,
            "future-model",
            False,
            {"model_id": model.model_id},
        ),
    )
    trace = replace(
        source,
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, *added),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    library = ScenarioLibrary.from_trace(
        trace,
        rsu_snapshot_period_s=0.25,
        rsu_telemetry_delay_s=0.10,
        rsu_telemetry_quantum_work_s=0.01,
        rsu_telemetry_drop_every=2,
    )
    environment = library.environment_scenarios[0]

    first_rsu = min(item.rsu_id for item in environment.telemetry)
    first = next(
        item
        for item in environment.telemetry
        if item.rsu_id == first_rsu and item.sample_sequence == 1
    )
    dropped = next(
        item
        for item in environment.telemetry
        if item.rsu_id == first_rsu and item.sample_sequence == 2
    )
    assert first.offset_s == pytest.approx(0.25)
    assert first.delivery_offset_s == pytest.approx(0.35)
    assert first.work_quantum_s == pytest.approx(0.01)
    assert first.dropped is False
    assert dropped.dropped is True
    assert dropped.delivery_offset_s is None
    assert {item.event_type for item in environment.versions}.issuperset(
        {"MODEL_VERSION", "PROFILE_VERSION", "PROTOCOL_VERSION"}
    )
    assert {0.31, 0.32, 0.33}.issubset(set(environment.macro_event_offsets))


def _reachable_strings(root):
    pending = [root]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            yield value
            continue
        if isinstance(value, (bytes, int, float, bool, type(None), Enum, Path)):
            continue
        marker = id(value)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(value, Mapping):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
        elif is_dataclass(value):
            pending.extend(getattr(value, item.name) for item in fields(value))
        else:
            if hasattr(value, "__dict__"):
                pending.extend(vars(value).values())
            slots = getattr(type(value), "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            pending.extend(
                getattr(value, slot)
                for slot in slots
                if isinstance(slot, str) and hasattr(value, slot)
            )


def test_scenario_future_tasks_are_relative_and_identity_unlinkable(decision_fixture):
    simulator = decision_fixture(task_id="future-task-isolation").simulator
    source = simulator.scenario_trace
    library = simulator.scenario_library
    future_tasks = tuple(
        task
        for environment in library.environment_scenarios
        for task in environment.future_tasks
    )

    assert future_tasks
    assert all(
        0 <= task.arrival_offset_s <= environment.duration_s + 1e-12
        for environment in library.environment_scenarios
        for task in environment.future_tasks
    )
    for environment in library.environment_scenarios:
        tokens = [task.task_token for task in environment.future_tasks]
        assert len(tokens) == len(set(tokens))

    forbidden_field_names = {
        "absolute_deadline_s",
        "arrival_time_s",
        "fixture_key",
        "row_id",
        "subject_cluster_id",
        "task_id",
        "true_identity",
    }
    for task in future_tasks:
        assert forbidden_field_names.isdisjoint(item.name for item in fields(task))
        assert tuple(name for name, _ in task.quality_probabilities) == (
            task.quality_candidates
        )
        assert sum(probability for _, probability in task.quality_probabilities) == (
            pytest.approx(1.0)
        )
        with pytest.raises(TypeError):
            task.quality_features["forbidden-mutation"] = 1.0

    source_identities = (
        {row.task_id for row in source.arrivals}
        | {row.fixture_key for row in source.arrivals}
        | {row.row_id for row in source.prep_rows}
        | {row.row_id for row in source.anon_rows}
        | {row.row_id for row in source.local_rows}
        | {row.row_id for row in source.edge_rows}
        | {
            row.subject_cluster_id
            for row in (*source.anon_rows, *source.local_rows)
            if row.subject_cluster_id is not None
        }
        | {
            row.artifact_key
            for row in (*source.anon_rows, *source.edge_rows)
            if row.artifact_key is not None
        }
    )
    assert source_identities.isdisjoint(_reachable_strings(future_tasks))


def test_scenario_future_tasks_reference_complete_sanitized_rows(decision_fixture):
    library = decision_fixture(task_id="future-task-support").simulator.scenario_library
    future_tasks = tuple(
        task
        for environment in library.environment_scenarios
        for task in environment.future_tasks
    )

    assert future_tasks
    assert all(task.complete_support for task in future_tasks)
    assert all(task.support_reason is None for task in future_tasks)
    assert any(task.ood and task.prep_failed for task in future_tasks)
    for task in future_tasks:
        assert all(row in library.local_rows for row in task.local_rows)
        assert all(row in library.anon_rows for row in task.anon_rows)
        assert all(row in library.edge_rows for row in task.edge_rows)
        for quality_bin in task.quality_candidates:
            assert any(row.quality_bin == quality_bin for row in task.local_rows)
            matching_anon = tuple(
                row for row in task.anon_rows if row.quality_bin == quality_bin
            )
            assert matching_anon
            artifact_tokens = {
                row.artifact_token
                for row in matching_anon
                if row.formed_packet and row.artifact_token is not None
            }
            assert any(row.artifact_token in artifact_tokens for row in task.edge_rows)


def test_scenario_future_task_marks_incomplete_per_bin_support(decision_fixture):
    source = decision_fixture(
        task_id="future-task-missing-support"
    ).simulator.scenario_trace
    missing_challenging_local = replace(
        source,
        local_rows=tuple(
            row for row in source.local_rows if row.quality_bin != "challenging"
        ),
    )
    library = ScenarioLibrary.from_trace(missing_challenging_local)
    affected = tuple(
        task
        for environment in library.environment_scenarios
        for task in environment.future_tasks
        if "challenging" in task.quality_candidates
    )

    assert affected
    assert all(not task.complete_support for task in affected)
    assert all("local:challenging" in (task.support_reason or "") for task in affected)


def test_scenario_cluster_sampler_selects_cluster_before_complete_row():
    rows = [
        SimpleNamespace(cluster_token="cluster-a", scenario_id=f"a-{index}")
        for index in range(5)
    ] + [SimpleNamespace(cluster_token="cluster-b", scenario_id="b-0")]

    class ScriptedRng:
        def __init__(self):
            self.calls = []

        def randrange(self, upper):
            self.calls.append(upper)
            return 1 if len(self.calls) == 1 else 0

    rng = ScriptedRng()
    selected = ScenarioLibrary._cluster_sample(rows, rng)

    assert selected.scenario_id == "b-0"
    assert rng.calls == [2, 1]


def test_esl_horizon_uses_shared_environment_and_isolates_real_state(decision_fixture):
    fixture = decision_fixture(task_id="horizon-environment", deadline_s=10.0)
    before = {
        "battery": fixture.state.vehicles["veh-1"].battery_j,
        "vehicle_work": {
            name: pool.residual_work_s
            for name, pool in fixture.state.vehicles["veh-1"].resources.items()
        },
        "rsu": {
            rsu_id: runtime.admission.snapshot()
            for rsu_id, runtime in fixture.state.rsus.items()
        },
        "virtual": (
            dict(fixture.state.virtual_queues.vehicle_power),
            dict(fixture.state.virtual_queues.rsu_power),
            fixture.state.virtual_queues.failure,
            fixture.state.virtual_queues.coverage,
        ),
    }
    policy_h2 = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=2,
        scenarios=4,
        scenario_source=fixture.simulator.scenario_library,
    )
    policy_h8 = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=8,
        scenarios=4,
        scenario_source=fixture.simulator.scenario_library,
    )

    h2 = policy_h2.decide(fixture.task, fixture.observation, fixture.state)
    h8 = policy_h8.decide(fixture.task, fixture.observation, fixture.state)

    assert (
        h2.diagnostics["environment_scenarios"]
        == h8.diagnostics["environment_scenarios"]
    )
    assert any(
        abs(h2.scores[action] - h8.scores[action]) > 1e-12 for action in h2.scores
    )
    assert fixture.state.vehicles["veh-1"].battery_j == before["battery"]
    assert {
        name: pool.residual_work_s
        for name, pool in fixture.state.vehicles["veh-1"].resources.items()
    } == before["vehicle_work"]
    assert {
        rsu_id: runtime.admission.snapshot()
        for rsu_id, runtime in fixture.state.rsus.items()
    } == before["rsu"]
    assert (
        dict(fixture.state.virtual_queues.vehicle_power),
        dict(fixture.state.virtual_queues.rsu_power),
        fixture.state.virtual_queues.failure,
        fixture.state.virtual_queues.coverage,
    ) == before["virtual"]


class _FixedScenarioSource:
    def __init__(self, library, environment):
        self.anon_rows = library.anon_rows
        self.local_rows = library.local_rows
        self.edge_rows = library.edge_rows
        self._library = library
        self._environment = environment

    def sample_environment(self, rng):
        return self._environment

    def sample_rows(self, rows, rng):
        return self._library.sample_rows(rows, rng)


def _connected_focal_environment(fixture, environment):
    rsu_ids = tuple(sorted(fixture.observation.rsus))
    return replace(
        environment,
        duration_s=max(10.0, environment.duration_s),
        wireless=tuple(
            ScenarioWirelessSegment(
                vehicle_id=fixture.observation.vehicle_id,
                rsu_id=rsu_id,
                direction=direction,
                start_offset_s=0.0,
                end_offset_s=max(10.0, environment.duration_s),
                goodput_bps=100_000_000.0,
                transmitter_power_w=4.0,
                receiver_power_w=1.0,
                link_state="connected",
            )
            for rsu_id in rsu_ids
            for direction in (TransferDirection.UL, TransferDirection.DL)
        ),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=tuple(ScenarioTelemetryEvent(0.0, rsu_id) for rsu_id in rsu_ids),
        versions=(),
        future_tasks=(),
    )


def _expanded_focal_memory(observation):
    vehicle = thaw_json(observation.vehicle)
    vehicle["memory_capacity_bytes"] = 2_000_000_000
    vehicle["memory_remaining_bytes"] = (
        vehicle["memory_capacity_bytes"] - vehicle["memory_reserved_bytes"]
    )
    return replace(observation, vehicle=deep_freeze(vehicle))


def _two_quality_focal(fixture):
    return replace(
        _expanded_focal_memory(fixture.observation),
        quality_probabilities=(("clear", 1.0), ("challenging", 0.0)),
    )


def test_edge_realized_artifact_uses_one_quality_but_admission_bounds_all_cells(
    decision_fixture, monkeypatch
):
    fixture = decision_fixture(
        task_id="one-realized-quality-edge-envelope",
        deadline_s=10.0,
        bins=("clear", "challenging"),
    )
    observation = _two_quality_focal(fixture)
    library = fixture.simulator.scenario_library
    environment = _connected_focal_environment(
        fixture, library.environment_scenarios[0]
    )
    source = _FixedScenarioSource(library, environment)
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=3,
        scenarios=1,
        scenario_source=source,
        rollout_policy="fixed_safe_lowest_link_cost",
    )
    first = Action.pipeline(_globally_safe_fixed_pipeline(policy.mask_engine))
    captured = []
    original_admit = policy._scheduler_admit

    def capture_admission(scheduler, task):
        action = task.last_action
        public = policy._scheduler_observation(scheduler, task, ActionStage.READY)
        execution = policy._scheduler_execution_observation(scheduler, action, public)
        actual = policy._actual_edge_rows(
            task.future,
            action,
            execution,
            task.edge_pairing_token,
            quality_bin=task.latent_quality_bin,
        )
        certified = policy._certified_edge_admission_bounds(
            task.future, action, execution
        )
        accepted = original_admit(scheduler, task)
        captured.append(
            {
                "accepted": accepted,
                "actual_quality": {row.quality_bin for row in actual},
                "actual_artifacts": {row.artifact_token for row in actual},
                "latent_quality": task.latent_quality_bin,
                "outcome_quality": task.last_outcome.values["quality_bin"],
                "rsu_id": action.rsu_id,
                "model_id": action.edge_model_id,
                "pipeline_id": execution.selected_pipeline,
                "certified": certified,
                "task_bound": (
                    task.admission_vram_upper_bytes,
                    task.admission_gpu_work_upper_s,
                ),
                "reserved": (
                    task.reserved_vram_bytes,
                    task.reserved_gpu_work_s,
                ),
            }
        )
        return accepted

    monkeypatch.setattr(policy, "_scheduler_admit", capture_admission)
    rollout = policy._one_rollout(
        fixture.task,
        observation,
        first,
        scenario_index=0,
        include_diagnostics=True,
    )

    assert rollout[3]["complete_macro_recourse"] is True
    assert [row["stage"] for row in rollout[3]["decision_trace"]] == ["RAW", "READY"]
    assert "|EDGE|" in rollout[3]["decision_trace"][1]["action"]
    assert len(captured) == 1
    admission = captured[0]
    assert admission["accepted"] is True
    assert admission["actual_quality"] == {"clear"}
    assert len(admission["actual_artifacts"]) == 1
    assert admission["latent_quality"] == admission["outcome_quality"] == "clear"
    assert admission["certified"] is not None
    assert admission["task_bound"] == admission["certified"]
    assert admission["reserved"] == admission["certified"]
    assert admission["certified"][0] == max(
        row.vram_bytes
        for row in source.edge_rows
        if row.rsu_id == admission["rsu_id"]
        and row.model_id == admission["model_id"]
        and row.pipeline_id == admission["pipeline_id"]
        and row.quality_bin in observation.conformal_quality_bins
    )
    assert admission["certified"][1] == max(
        row.gpu_work_s
        for row in source.edge_rows
        if row.rsu_id == admission["rsu_id"]
        and row.model_id == admission["model_id"]
        and row.pipeline_id == admission["pipeline_id"]
        and row.quality_bin in observation.conformal_quality_bins
    )


@pytest.mark.parametrize("missing", ("formed_packet", "paired_edge"))
def test_rollout_is_incomplete_when_any_quality_candidate_loses_edge_pairing(
    decision_fixture, missing
):
    fixture = decision_fixture(
        task_id=f"missing-quality-cell-{missing}",
        deadline_s=10.0,
        bins=("clear", "challenging"),
    )
    observation = _two_quality_focal(fixture)
    library = fixture.simulator.scenario_library
    environment = _connected_focal_environment(
        fixture, library.environment_scenarios[0]
    )
    source = _FixedScenarioSource(library, environment)
    pipeline_id = _globally_safe_fixed_pipeline(fixture.simulator.mask_engine)
    if missing == "formed_packet":
        source.anon_rows = tuple(
            replace(
                row,
                formed_packet=False,
                final_encoded_size_bytes=0,
                artifact_token=None,
                fer_measurements={},
            )
            if row.pipeline_id == pipeline_id and row.quality_bin == "challenging"
            else row
            for row in source.anon_rows
        )
    else:
        source.edge_rows = tuple(
            row for row in source.edge_rows if row.quality_bin != "challenging"
        )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=2,
        scenarios=1,
        scenario_source=source,
        rollout_policy="fixed_safe_lowest_link_cost",
    )
    rollout = policy._one_rollout(
        fixture.task,
        observation,
        Action.pipeline(pipeline_id),
        scenario_index=0,
        include_diagnostics=True,
    )

    assert [row["stage"] for row in rollout[3]["decision_trace"]] == ["RAW", "READY"]
    assert "|EDGE|" in rollout[3]["decision_trace"][1]["action"]
    assert rollout[3]["complete_macro_recourse"] is False
    assert rollout[3]["incomplete_reason"] == "DECISION_COMMIT_ACTION_PAIRING_MISSING"


def test_pipe_edge_technical_local_fallback_keeps_one_latent_quality(
    decision_fixture, monkeypatch
):
    fixture = decision_fixture(
        task_id="technical-fallback-quality-cell",
        deadline_s=10.0,
        bins=("clear", "challenging"),
    )
    observation = _two_quality_focal(fixture)
    library = fixture.simulator.scenario_library
    environment = _connected_focal_environment(
        fixture, library.environment_scenarios[0]
    )
    source = _FixedScenarioSource(library, environment)
    source.edge_rows = tuple(replace(row, failed=True) for row in source.edge_rows)
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=3,
        scenarios=1,
        scenario_source=source,
        rollout_policy="fixed_safe_lowest_link_cost",
    )
    first = Action.pipeline(_globally_safe_fixed_pipeline(policy.mask_engine))
    commits = []
    original_commit = policy._scheduler_commit

    def capture_commit(
        scheduler, task, action, outcome, execution_observation, *, failure_reason=None
    ):
        commits.append(
            {
                "kind": action.kind.value,
                "quality": outcome.values.get("quality_bin"),
                "latent": task.latent_quality_bin,
                "failure_reason": failure_reason,
            }
        )
        return original_commit(
            scheduler,
            task,
            action,
            outcome,
            execution_observation,
            failure_reason=failure_reason,
        )

    monkeypatch.setattr(policy, "_scheduler_commit", capture_commit)
    rollout = policy._one_rollout(
        fixture.task,
        observation,
        first,
        scenario_index=0,
        include_diagnostics=True,
    )

    assert [row["kind"] for row in commits] == ["PIPE", "EDGE", "LOCAL"]
    assert {row["quality"] for row in commits} == {"clear"}
    assert {row["latent"] for row in commits} == {"clear"}
    assert commits[-1]["failure_reason"] == "EDGE_MODEL_FAILURE"
    assert any(
        row["kind"] == "DETERMINISTIC_REPAIR" and row["reason"] == "EDGE_MODEL_FAILURE"
        for row in rollout[3]["scheduler_trace"]
    )
    assert any(
        row["kind"] == "TASK_TERMINAL" and row["state"] == "DONE"
        for row in rollout[3]["scheduler_trace"]
    )


class _RecordingScenarioSource:
    def __init__(self, library, fixture):
        self.anon_rows = library.anon_rows
        self.local_rows = library.local_rows
        self.edge_rows = library.edge_rows
        self._library = library
        self._fixture = fixture
        self.selected_environments = []

    def sample_environment(self, rng):
        selected = self._library.sample_environment(rng)
        self.selected_environments.append(selected.scenario_id)
        return _connected_focal_environment(self._fixture, selected)

    def sample_rows(self, rows, rng):
        return self._library.sample_rows(rows, rng)


def test_common_latent_quality_and_environment_are_action_order_independent(
    decision_fixture,
):
    fixture = decision_fixture(
        task_id="common-random-substreams",
        deadline_s=10.0,
        bins=("clear", "challenging"),
    )
    observation = _expanded_focal_memory(fixture.observation)
    library = fixture.simulator.scenario_library
    local = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))
    pipeline = Action.pipeline(
        _globally_safe_fixed_pipeline(fixture.simulator.mask_engine)
    )

    def run_order(actions):
        source = _RecordingScenarioSource(library, fixture)
        policy = POLICY_REGISTRY["esl_smpc"](
            fixture.simulator.mask_engine,
            fixture.simulator.repairer,
            horizon_events=2,
            scenarios=1,
            scenario_source=source,
            rollout_policy="fixed_safe_lowest_link_cost",
        )
        results = {
            action.canonical_id: policy._one_rollout(
                fixture.task,
                observation,
                action,
                scenario_index=3,
                include_diagnostics=True,
            )
            for action in actions
        }
        return results, tuple(source.selected_environments)

    forward, forward_environments = run_order((local, pipeline))
    reverse, reverse_environments = run_order((pipeline, local))
    quality_by_row = {
        row.scenario_id: row.quality_bin
        for row in (*library.local_rows, *library.anon_rows, *library.edge_rows)
    }
    forward_qualities = {
        quality_by_row[result[2][0]] for result in forward.values() if result[2]
    }
    reverse_qualities = {
        quality_by_row[result[2][0]] for result in reverse.values() if result[2]
    }

    assert forward == reverse
    assert len(set(forward_environments)) == 1
    assert forward_environments == reverse_environments
    assert len(forward_qualities) == 1
    assert forward_qualities == reverse_qualities


def test_h3_executes_future_raw_and_ready_as_real_macro_decisions(decision_fixture):
    fixture = decision_fixture(task_id="h3-future-macros", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    environment = library.environment_scenarios[0]
    source = _FixedScenarioSource(
        library, replace(environment, future_tasks=(environment.future_tasks[0],))
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=3,
        scenarios=1,
        scenario_source=source,
        rollout_policy="fixed_safe_lowest_link_cost",
    )
    first = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))
    battery_before = fixture.state.vehicles[fixture.task.vehicle_id].battery_j
    admission_before = {
        rsu_id: runtime.admission.snapshot()
        for rsu_id, runtime in fixture.state.rsus.items()
    }

    rollout = policy._one_rollout(
        fixture.task,
        fixture.observation,
        first,
        scenario_index=0,
        include_diagnostics=True,
    )
    repeated = policy._one_rollout(
        fixture.task,
        fixture.observation,
        first,
        scenario_index=0,
        include_diagnostics=True,
    )
    details = rollout[3]
    decisions = details["decision_trace"]

    assert rollout == repeated
    assert details["complete_macro_recourse"] is True
    assert details["decision_count"] == 3
    assert [row["stage"] for row in decisions] == ["RAW", "RAW", "READY"]
    assert "|PIPE|" in decisions[1]["action"]
    assert "|EDGE|" in decisions[2]["action"]
    assert decisions[1]["task_token"] == decisions[2]["task_token"]
    assert decisions[1]["task_token"].startswith("scenario:")
    assert fixture.state.vehicles[fixture.task.vehicle_id].battery_j == battery_before
    assert {
        rsu_id: runtime.admission.snapshot()
        for rsu_id, runtime in fixture.state.rsus.items()
    } == admission_before


def test_h4_event_heap_handles_overlapping_multivehicle_jobs_without_mutation(
    decision_fixture,
):
    fixture = decision_fixture(task_id="h4-concurrent-vehicles", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    environment = library.environment_scenarios[0]
    source = _FixedScenarioSource(library, environment)
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=4,
        scenarios=1,
        scenario_source=source,
        rollout_policy="fixed_safe_lowest_link_cost",
    )
    first = Action.pipeline(
        _globally_safe_fixed_pipeline(fixture.simulator.mask_engine)
    )
    battery_before = {key: row.battery_j for key, row in fixture.state.vehicles.items()}

    rollout = policy._one_rollout(
        fixture.task,
        fixture.observation,
        first,
        scenario_index=0,
        include_diagnostics=True,
    )
    repeated = policy._one_rollout(
        fixture.task,
        fixture.observation,
        first,
        scenario_index=0,
        include_diagnostics=True,
    )
    details = rollout[3]

    assert rollout == repeated
    assert details["scheduler_kind"] == "isolated_continuous_time_event_heap"
    assert details["complete_macro_recourse"] is True
    assert details["macro_event_count"] == 4
    assert {row["vehicle_id"] for row in details["decision_trace"]} == {
        "veh-1",
        "veh-2",
    }
    prep_starts = [
        row
        for row in details["scheduler_trace"]
        if row["kind"] == "JOB_START"
        and row["resource"] == "accelerator"
        and row["time_s"] == pytest.approx(0.1)
    ]
    assert {row["owner_id"] for row in prep_starts} == {"veh-1", "veh-2"}
    assert {
        key: row.battery_j for key, row in fixture.state.vehicles.items()
    } == battery_before


def test_event_heap_vehicle_resource_is_nonpreemptive_edf(decision_fixture):
    fixture = decision_fixture(task_id="branch-edf", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    environment = library.environment_scenarios[0]
    later, urgent = environment.future_tasks[:2]
    later = replace(later, vehicle_id="veh-1", relative_deadline_s=1.2)
    urgent = replace(urgent, vehicle_id="veh-1", relative_deadline_s=0.5)
    source = _FixedScenarioSource(
        library, replace(environment, future_tasks=(later, urgent))
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=2,
        scenarios=1,
        scenario_source=source,
        rollout_policy="all_local",
    )
    first = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))

    details = policy._one_rollout(
        fixture.task,
        fixture.observation,
        first,
        scenario_index=0,
        include_diagnostics=True,
    )[3]
    prep_starts = [
        row
        for row in details["scheduler_trace"]
        if row["kind"] == "JOB_START"
        and row["resource"] == "accelerator"
        and row["task_token"] in {later.task_token, urgent.task_token}
    ]

    assert [row["task_token"] for row in prep_starts] == [
        urgent.task_token,
        later.task_token,
    ]
    assert prep_starts[1]["time_s"] >= prep_starts[0]["time_s"] + urgent.prep_work_s


def test_h1_macro_interval_uses_partial_cost_before_earlier_external_raw(
    decision_fixture,
):
    fixture = decision_fixture(task_id="h1-partial-macro", deadline_s=10.0)
    estimator = fixture.simulator.estimator
    estimator._next_external_raw_offsets_s = (0.05,)

    row = estimator._with_macro_interval(
        {
            "expected_duration_s": 0.2,
            "expected_vehicle_energy_j": 2.0,
            "expected_rsu_energy_j": 1.0,
            "expected_loss": 0.4,
            "expected_failure": 0.1,
        },
        fixture.observation,
        vehicle_resources=("accelerator",),
    )

    assert row["expected_next_macro_interval_s"] == pytest.approx(0.05)
    assert row["interval_expected_vehicle_energy_j"] == pytest.approx(0.5)
    assert row["interval_expected_rsu_energy_j"] == pytest.approx(0.25)
    assert row["action_completed_before_macro_probability"] == 0.0
    assert row["interval_completion_probability"] == 0.0
    assert row["interval_expected_failure"] == 0.0
    assert row["interval_expected_arrivals"] == 1.0
    assert row["vehicle_service_s"]["accelerator"] == pytest.approx(0.05)


def test_h1_macro_interval_conservatively_accounts_for_unidentified_ready_job(
    decision_fixture,
):
    fixture = decision_fixture(task_id="h1-unidentified-ready", deadline_s=10.0)
    estimator = fixture.simulator.estimator
    estimator._next_external_raw_offsets_s = (float("inf"),)
    vehicle = dict(fixture.observation.vehicle)
    resources = {key: dict(value) for key, value in vehicle["resources"].items()}
    resources["accelerator"].update(
        {"residual_work_s": 0.03, "running_count": 1, "waiting_count": 0}
    )
    vehicle["resources"] = resources
    observation = replace(fixture.observation, vehicle=vehicle)

    row = estimator._with_macro_interval(
        {"expected_duration_s": 0.2, "expected_vehicle_energy_j": 2.0},
        observation,
        vehicle_resources=("accelerator",),
    )

    assert row["expected_next_macro_interval_s"] == pytest.approx(0.03)
    assert row["interval_expected_vehicle_energy_j"] == pytest.approx(0.3)
    assert row["action_completed_before_macro_probability"] == 0.0


def test_h1_policy_uses_event_heap_until_earlier_other_vehicle_raw(decision_fixture):
    fixture = decision_fixture(task_id="h1-event-heap-partial", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    environment = library.environment_scenarios[0]
    future = environment.future_tasks[1]
    future = replace(
        future,
        arrival_offset_s=max(1e-6, 0.05 - future.prep_work_s),
    )
    source = _FixedScenarioSource(library, replace(environment, future_tasks=(future,)))
    source.local_rows = tuple(
        replace(row, service_work_s=0.2)
        if row.model_id == min(fixture.simulator.profile.local_models)
        else row
        for row in source.local_rows
    )
    policy = POLICY_REGISTRY["safe_lyapunov_h1"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenarios=1,
        scenario_source=source,
    )
    action = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))
    mask = fixture.simulator.mask_engine.enumerate(
        fixture.task, fixture.observation, fixture.state
    )
    scores = policy._scores(fixture.task, fixture.observation, mask, fixture.state)
    engine = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=2,
        scenarios=1,
        scenario_source=source,
        lyapunov_v=policy.lyapunov_v,
        physical_queue_weight=policy.physical_queue_weight,
        vehicle_resource_theta=policy.vehicle_resource_theta,
        rsu_resource_theta=policy.rsu_resource_theta,
    )
    rollout = engine._one_rollout(
        fixture.task,
        fixture.observation,
        action,
        0,
        include_diagnostics=True,
        stop_before_next_macro=True,
    )

    assert rollout[1] == pytest.approx(0.05)
    assert scores[action] == pytest.approx(rollout[0] / rollout[1])
    assert rollout[3]["scheduler_kind"] == "isolated_continuous_time_event_heap"
    assert not any(
        row.get("completion_kind") == "LOCAL_DONE"
        and row.get("task_token") == fixture.task.task_id
        for row in rollout[3]["scheduler_trace"]
    )


def test_h1_pipeline_failure_runs_deterministic_fallback_before_terminal_macro(
    decision_fixture,
):
    fixture = decision_fixture(task_id="h1-pipe-fallback", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    environment = replace(library.environment_scenarios[0], future_tasks=())
    source = _FixedScenarioSource(library, environment)
    pipeline_id = _globally_safe_fixed_pipeline(fixture.simulator.mask_engine)
    source.anon_rows = tuple(
        replace(
            row,
            formed_packet=False,
            final_encoded_size_bytes=0,
            artifact_token=None,
            fer_measurements={},
        )
        if row.pipeline_id == pipeline_id
        else row
        for row in source.anon_rows
    )
    engine = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=2,
        scenarios=1,
        scenario_source=source,
    )
    action = Action.pipeline(pipeline_id)

    rollout = engine._one_rollout(
        fixture.task,
        fixture.observation,
        action,
        0,
        include_diagnostics=True,
        stop_before_next_macro=True,
    )
    trace = rollout[3]["scheduler_trace"]

    assert any(
        row["kind"] == "DETERMINISTIC_REPAIR"
        and row["reason"] == "ANON_TRANSACTION_NOT_FORMED"
        for row in trace
    )
    assert any(
        row["kind"] == "TASK_TERMINAL" and row["state"] == "DONE" for row in trace
    )
    assert rollout[3]["decision_count"] == 1
    assert len(rollout[2]) == 2


def test_h3_policy_diagnostics_certify_supported_synthetic_recourse(
    decision_fixture,
):
    fixture = decision_fixture(task_id="h3-supported-policy", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    # Use a complete clear-quality future task and shift only its relative
    # arrival inside this isolated test window.  The gap is deliberately above
    # every certified focal action duration, so this exercises the controller's
    # declared non-overlap complete-recourse domain rather than the conservative
    # concurrent-arrival fallback.
    environment = library.environment_scenarios[0]
    first_future = environment.future_tasks[0]
    formed_rows = tuple(row for row in first_future.anon_rows if row.formed_packet)
    formed_artifacts = {row.artifact_token for row in formed_rows}
    first_future = replace(
        first_future,
        arrival_offset_s=0.70,
        anon_rows=formed_rows,
        edge_rows=tuple(
            row
            for row in first_future.edge_rows
            if row.artifact_token in formed_artifacts
        ),
    )
    source = _FixedScenarioSource(
        library,
        replace(
            environment,
            future_tasks=(first_future,),
        ),
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=3,
        scenarios=1,
        scenario_source=source,
        rollout_policy="fixed_safe_lowest_link_cost",
    )

    decision = policy.decide(fixture.task, fixture.observation, fixture.state)

    assert decision.diagnostics["incomplete_reasons"] == (), decision.diagnostics[
        "incomplete_reasons"
    ]
    assert decision.diagnostics["complete_macro_recourse"] is True
    assert (
        decision.diagnostics["approximation_kind"]
        == "complete_isolated_continuous_time_macro_event_recourse"
    )
    assert decision.diagnostics["incomplete_reasons"] == ()
    assert all(
        len(scenario[0]) == 3
        for scenario in decision.diagnostics["scenario_decisions"].values()
    )
    certificate = decision.diagnostics["scenario_error_certificate"]
    assert certificate["valid"] is False
    assert certificate["reason"] == "PREREGISTERED_BOUNDS_MISSING"


def test_future_pi0_action_changes_isolated_virtual_queue(decision_fixture):
    fixture = decision_fixture(task_id="future-pi0-queue", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    environment = library.environment_scenarios[0]
    source = _FixedScenarioSource(
        library, replace(environment, future_tasks=(environment.future_tasks[0],))
    )
    first = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))

    def rollout(rollout_policy):
        policy = POLICY_REGISTRY["esl_smpc"](
            fixture.simulator.mask_engine,
            fixture.simulator.repairer,
            horizon_events=3,
            scenarios=1,
            scenario_source=source,
            rollout_policy=rollout_policy,
        )
        return policy._one_rollout(
            fixture.task,
            fixture.observation,
            first,
            scenario_index=0,
            include_diagnostics=True,
        )

    fixed = rollout("fixed_safe_lowest_link_cost")
    local = rollout("all_local")
    assert fixed[3]["complete_macro_recourse"] is True
    assert local[3]["complete_macro_recourse"] is True
    assert "|PIPE|" in fixed[3]["decision_trace"][1]["action"]
    assert "|LOCAL|" in local[3]["decision_trace"][1]["action"]
    assert fixed[3]["terminal_virtual_queues"] != local[3]["terminal_virtual_queues"]
    assert fixture.state.virtual_queues.vehicle_power[fixture.task.vehicle_id] == 0.0


def test_future_ready_recourse_respects_version_and_admission_snapshot(
    decision_fixture,
):
    fixture = decision_fixture(task_id="future-version-admission", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    environment = library.environment_scenarios[0]
    source = _FixedScenarioSource(
        library, replace(environment, future_tasks=(environment.future_tasks[0],))
    )
    first = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))

    def actions(observation):
        policy = POLICY_REGISTRY["esl_smpc"](
            fixture.simulator.mask_engine,
            fixture.simulator.repairer,
            horizon_events=3,
            scenarios=1,
            scenario_source=source,
            rollout_policy="fixed_safe_lowest_link_cost",
        )
        result = policy._one_rollout(
            fixture.task,
            observation,
            first,
            scenario_index=0,
            include_diagnostics=True,
        )
        return result[3]["decision_trace"]

    stale_versions = dict(fixture.observation.versions)
    edge_versions = {
        key: dict(value)
        for key, value in fixture.observation.versions["edge_models"].items()
    }
    edge_versions[min(edge_versions)]["model_hash"] = "0" * 64
    stale_versions["edge_models"] = edge_versions
    stale_actions = actions(replace(fixture.observation, versions=stale_versions))
    assert "|LOCAL|" in stale_actions[2]["action"]

    full_rsus = {key: dict(value) for key, value in fixture.observation.rsus.items()}
    for row in full_rsus.values():
        row["descriptors"] = row["descriptor_capacity"]
    capacity_actions = actions(replace(fixture.observation, rsus=full_rsus))
    assert "|LOCAL|" in capacity_actions[2]["action"]


def test_future_pipeline_deadline_updates_timeout_queue(decision_fixture):
    fixture = decision_fixture(task_id="future-deadline", deadline_s=10.0)
    library = fixture.simulator.scenario_library
    environment = library.environment_scenarios[0]
    # The action is optimistic-feasible under the frozen support bound, while
    # this complete joint scenario row has a long tail and misses deadline.
    source_future = environment.future_tasks[0]
    slow_anon = tuple(
        replace(
            row,
            attempts=(
                replace(row.attempts[0], anon_work_s=row.attempts[0].anon_work_s + 1.0),
                *row.attempts[1:],
            ),
        )
        if row.pipeline_id == "pixelate_strong_v1"
        else row
        for row in source_future.anon_rows
    )
    future = replace(
        source_future,
        relative_deadline_s=0.2,
        anon_rows=slow_anon,
    )
    source = _FixedScenarioSource(
        library,
        replace(
            environment,
            future_tasks=(future,),
        ),
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        horizon_events=3,
        scenarios=1,
        scenario_source=source,
        rollout_policy="fixed_safe_lowest_link_cost",
    )
    first = Action.local(ActionStage.RAW, min(fixture.simulator.profile.local_models))
    result = policy._one_rollout(
        fixture.task,
        fixture.observation,
        first,
        scenario_index=0,
        include_diagnostics=True,
    )

    assert result[3]["complete_macro_recourse"] is True
    assert "|PIPE|" in result[3]["decision_trace"][1]["action"]
    assert result[3]["terminal_virtual_queues"]["timeout"] == pytest.approx(1.0)
    assert result[3]["terminal_virtual_queues"]["failure"] == pytest.approx(1.0)


def test_pipeline_rollout_does_not_charge_fer_before_ready_recourse(decision_fixture):
    fixture = decision_fixture(task_id="pipe-no-early-fer")
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    action = next(
        action
        for action in fixture.simulator.mask_engine.enumerate(
            fixture.task, fixture.observation, fixture.state
        ).allowed
        if action.pipeline_id
    )
    row = policy._trace_candidates(action, fixture.observation)[0]
    outcome = policy._row_outcome(row)

    assert outcome.values["expected_fer_loss"] == 0.0
    assert outcome.values["completion_probability"] == 0.0


def test_edge_prediction_requires_ul_admission_gpu_and_dl(decision_fixture):
    fixture = decision_fixture(task_id="edge-full-path", deadline_s=10.0)
    simulator = fixture.simulator
    edge = next(
        row
        for row in simulator.scenario_library.edge_rows
        if row.rsu_id == "rsu-1" and row.quality_bin == "clear"
    )
    anon = next(
        row
        for row in simulator.scenario_library.anon_rows
        if row.artifact_token == edge.artifact_token
    )
    observation = replace(
        fixture.observation,
        stage=ActionStage.READY,
        task_state=TaskState.READY,
        selected_pipeline=edge.pipeline_id,
        artifact_token=edge.artifact_token,
        encoded_size_bytes=anon.final_encoded_size_bytes,
        encoded_evidence=deep_freeze(
            {
                "message_source_type": "EncodedAnon",
                "artifact_token": edge.artifact_token,
                "pipeline_id": edge.pipeline_id,
                "profile_hash": simulator.profile.profile_hash,
            }
        ),
    )
    environment = ScenarioEnvironment(
        scenario_id="unit-relative-environment",
        cluster_token="unit-cluster",
        duration_s=10.0,
        wireless=tuple(
            ScenarioWirelessSegment(
                vehicle_id=observation.vehicle_id,
                rsu_id=edge.rsu_id,
                direction=direction,
                start_offset_s=0.0,
                end_offset_s=10.0,
                goodput_bps=10_000_000.0,
                transmitter_power_w=4.0 if direction is TransferDirection.UL else 10.0,
                receiver_power_w=1.0,
                link_state="connected",
            )
            for direction in (TransferDirection.UL, TransferDirection.DL)
        ),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(ScenarioTelemetryEvent(0.0, edge.rsu_id),),
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    branch = policy._new_branch(observation, environment)
    sampled = policy._row_outcome(edge)
    executed = policy._execute_prediction_action(
        Action.edge(edge.rsu_id, edge.model_id), sampled, observation, branch
    )

    compute_only = edge.ingress_work_s + edge.gpu_work_s
    assert executed.values["completion_probability"] == 1.0
    assert executed.duration_s > compute_only
    assert executed.values["expected_vehicle_energy_j"] > 0
    assert (
        executed.values["expected_rsu_energy_j"]
        > edge.ingress_energy_j + edge.gpu_energy_j
    )
    assert (
        branch.rsus[edge.rsu_id]["descriptors"]
        == observation.rsus[edge.rsu_id]["descriptors"]
    )

    stale_versions = dict(observation.versions)
    edge_versions = {
        key: dict(value) for key, value in observation.versions["edge_models"].items()
    }
    edge_versions[edge.model_id]["model_hash"] = "0" * 64
    stale_versions["edge_models"] = edge_versions
    stale_observation = replace(observation, versions=stale_versions)
    rejected_branch = policy._new_branch(stale_observation, environment)
    rejected = policy._execute_prediction_action(
        Action.edge(edge.rsu_id, edge.model_id),
        sampled,
        stale_observation,
        rejected_branch,
    )
    assert rejected.values["completion_probability"] == 0.0
    assert (
        rejected_branch.rsus[edge.rsu_id]["descriptors"]
        == stale_observation.rsus[edge.rsu_id]["descriptors"]
    )


def test_scenario_rsu_snapshot_is_sampled_frozen_and_delivered_causally(
    decision_fixture,
):
    fixture = decision_fixture(task_id="branch-causal-telemetry", deadline_s=10.0)
    simulator = fixture.simulator
    rsu_id = min(fixture.observation.rsus)
    environment = ScenarioEnvironment(
        scenario_id="causal-telemetry",
        cluster_token="causal-telemetry-cluster",
        duration_s=1.0,
        wireless=(),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(
            ScenarioTelemetryEvent(
                offset_s=0.10,
                rsu_id=rsu_id,
                delivery_offset_s=0.30,
                sample_sequence=1,
                work_quantum_s=0.10,
            ),
        ),
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    branch = policy._new_branch(fixture.observation, environment)
    original_descriptors = int(fixture.observation.rsus[rsu_id]["descriptors"])
    branch.rsus[rsu_id]["descriptors"] = original_descriptors + 1
    branch.rsus[rsu_id]["reserved_work_gpu_s"] = 0.16

    policy._advance_branch(branch, 0.10, fixture.observation.vehicle_id)
    branch.rsus[rsu_id]["descriptors"] = original_descriptors + 2
    branch.rsus[rsu_id]["reserved_work_gpu_s"] = 0.37
    policy._advance_branch(branch, 0.20, fixture.observation.vehicle_id)
    assert branch.public_rsus[rsu_id]["descriptors"] == original_descriptors

    future = simulator.scenario_library.environment_scenarios[0].future_tasks[0]
    future = replace(
        future,
        vehicle_id=fixture.observation.vehicle_id,
        arrival_offset_s=0.0,
        relative_deadline_s=1.0,
    )
    before_delivery = policy._future_observation(
        future,
        fixture.observation,
        branch,
        stage=ActionStage.RAW,
    )
    assert before_delivery.rsus[rsu_id]["descriptors"] == original_descriptors

    policy._advance_branch(branch, 0.30, fixture.observation.vehicle_id)
    after_delivery = policy._future_observation(
        future,
        fixture.observation,
        branch,
        stage=ActionStage.RAW,
    )
    assert after_delivery.rsus[rsu_id]["descriptors"] == original_descriptors + 1
    assert after_delivery.rsus[rsu_id]["reserved_work_gpu_s"] == pytest.approx(0.2)
    assert branch.rsus[rsu_id]["descriptors"] == original_descriptors + 2
    assert after_delivery.rsus[rsu_id]["snapshot_age_s"] == pytest.approx(0.20)


@pytest.mark.parametrize(
    "event_type", ["MODEL_VERSION", "PROFILE_VERSION", "PROTOCOL_VERSION"]
)
def test_scenario_version_change_during_uplink_blocks_atomic_admission(
    event_type,
    decision_fixture,
):
    fixture = decision_fixture(task_id=f"branch-{event_type.lower()}", deadline_s=10.0)
    simulator = fixture.simulator
    edge = next(
        row
        for row in simulator.scenario_library.edge_rows
        if row.rsu_id == "rsu-1" and row.quality_bin == "clear"
    )
    anon = next(
        row
        for row in simulator.scenario_library.anon_rows
        if row.artifact_token == edge.artifact_token
    )
    observation = replace(
        fixture.observation,
        stage=ActionStage.READY,
        task_state=TaskState.READY,
        selected_pipeline=edge.pipeline_id,
        artifact_token=edge.artifact_token,
        encoded_size_bytes=anon.final_encoded_size_bytes,
    )
    version = ScenarioVersionEvent(
        offset_s=0.05,
        event_type=event_type,
        target_type="rsu" if event_type == "MODEL_VERSION" else "deployment",
        target_id=edge.rsu_id if event_type == "MODEL_VERSION" else "global",
        resource="gpu" if event_type == "MODEL_VERSION" else "version",
        old_version=(
            edge.model_hash
            if event_type == "MODEL_VERSION"
            else simulator.profile.profile_hash
            if event_type == "PROFILE_VERSION"
            else simulator.profile.protocol_version
        ),
        new_version=f"invalid-{event_type.lower()}",
        model_id=edge.model_id if event_type == "MODEL_VERSION" else None,
        remove=False,
        maintenance_work_s=0.04 if event_type == "MODEL_VERSION" else None,
        maintenance_energy_j=2.0 if event_type == "MODEL_VERSION" else None,
    )
    environment = ScenarioEnvironment(
        scenario_id=f"during-ul-{event_type.lower()}",
        cluster_token="during-ul-version-cluster",
        duration_s=2.0,
        wireless=tuple(
            ScenarioWirelessSegment(
                vehicle_id=observation.vehicle_id,
                rsu_id=edge.rsu_id,
                direction=direction,
                start_offset_s=0.0,
                end_offset_s=2.0,
                goodput_bps=1_000_000.0,
                transmitter_power_w=4.0,
                receiver_power_w=1.0,
                link_state="connected",
            )
            for direction in (TransferDirection.UL, TransferDirection.DL)
        ),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
        versions=(version,),
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    branch = policy._new_branch(observation, environment)
    before = dict(branch.rsus[edge.rsu_id])
    executed = policy._execute_prediction_action(
        Action.edge(edge.rsu_id, edge.model_id),
        policy._row_outcome(edge),
        observation,
        branch,
    )

    assert executed.values["completion_probability"] == 0.0
    assert branch.rsus[edge.rsu_id]["descriptors"] == before["descriptors"]
    assert branch.rsus[edge.rsu_id]["vram_bytes"] == before["vram_bytes"]


def test_scenario_profile_change_removes_future_actions_before_repair(
    decision_fixture,
):
    fixture = decision_fixture(task_id="branch-version-repair", deadline_s=10.0)
    simulator = fixture.simulator
    source_environment = simulator.scenario_library.environment_scenarios[0]
    future = replace(
        source_environment.future_tasks[0],
        vehicle_id=fixture.observation.vehicle_id,
        arrival_offset_s=0.10,
        relative_deadline_s=1.0,
    )
    environment = replace(
        source_environment,
        wireless=(),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
        versions=(
            ScenarioVersionEvent(
                0.10,
                "PROFILE_VERSION",
                "deployment",
                "global",
                "profile",
                simulator.profile.profile_hash,
                "invalid-profile",
                None,
                False,
            ),
        ),
        future_tasks=(future,),
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
        rollout_policy="fixed_safe_lowest_link_cost",
    )
    branch = policy._new_branch(fixture.observation, environment)
    policy._advance_branch(branch, 0.10, fixture.observation.vehicle_id)
    future_observation = policy._future_observation(
        future,
        fixture.observation,
        branch,
        stage=ActionStage.RAW,
    )
    task = policy._future_task_record(future, future_observation)
    mask = simulator.mask_engine.enumerate(task, future_observation)
    repaired = policy._future_action(future, future_observation, branch)

    assert {action.kind.value for action in mask.allowed} == {"FAIL"}
    assert repaired.kind.value == "FAIL"


def test_scenario_scheduler_commits_model_cache_only_after_maintenance_completion(
    decision_fixture,
):
    fixture = decision_fixture(task_id="scenario-maintenance-commit", deadline_s=10.0)
    simulator = fixture.simulator
    environment = simulator.scenario_library.environment_scenarios[0]
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    branch = policy._new_branch(fixture.observation, environment)
    scheduler = policy._scheduler_new(branch, fixture.observation, environment)
    assert scheduler is not None
    rsu_id = "rsu-1"
    model = simulator.profile.edge_models["edge_fer_full_v1"]
    old_hash = branch.rsus[rsu_id]["cached_models"][model.model_id]
    event = ScenarioVersionEvent(
        offset_s=0.0,
        event_type="MODEL_VERSION",
        target_type="rsu",
        target_id=rsu_id,
        resource="model_cache",
        old_version=old_hash,
        new_version="scenario-post-maintenance-model",
        model_id=model.model_id,
        remove=False,
        maintenance_work_s=0.20,
        maintenance_energy_j=5.0,
    )

    policy._scheduler_enqueue_model_maintenance(scheduler, event)
    maintenance_id = next(
        job_id
        for job_id, job in scheduler.jobs.items()
        if job.completion_kind == "MODEL_MAINTENANCE_DONE"
    )
    policy._scheduler_dispatch(scheduler, random.Random(1))
    assert maintenance_id in scheduler.resources[("rsu", rsu_id, "gpu")].running

    policy._scheduler_advance(scheduler, 0.10)
    assert branch.rsus[rsu_id]["cached_models"][model.model_id] == old_hash
    assert branch.rsu_energy_j[rsu_id] == 0.0
    assert branch.rsu_physical_energy_j[rsu_id] > 0.0

    policy._scheduler_advance(scheduler, 0.20)
    policy._scheduler_complete_job(scheduler, maintenance_id, random.Random(2))
    assert (
        branch.rsus[rsu_id]["cached_models"][model.model_id]
        == "scenario-post-maintenance-model"
    )
    assert any(
        row["kind"] == "MODEL_MAINTENANCE_COMPLETE" for row in scheduler.event_trace
    )


def test_scenario_scheduler_serializes_chained_maintenance_per_rsu_model(
    decision_fixture,
):
    fixture = decision_fixture(task_id="scenario-maintenance-chain", deadline_s=10.0)
    simulator = fixture.simulator
    environment = simulator.scenario_library.environment_scenarios[0]
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    branch = policy._new_branch(fixture.observation, environment)
    scheduler = policy._scheduler_new(branch, fixture.observation, environment)
    assert scheduler is not None
    rsu_id = "rsu-1"
    model = simulator.profile.edge_models["edge_fer_full_v1"]
    old_hash = branch.rsus[rsu_id]["cached_models"][model.model_id]
    first = ScenarioVersionEvent(
        0.0,
        "MODEL_VERSION",
        "rsu",
        rsu_id,
        "model_cache",
        old_hash,
        "scenario-chain-v2",
        model.model_id,
        False,
        0.20,
        5.0,
    )
    second = ScenarioVersionEvent(
        0.01,
        "MODEL_CACHE",
        "rsu",
        rsu_id,
        "model_cache",
        "scenario-chain-v2",
        "scenario-chain-v3",
        model.model_id,
        False,
        0.01,
        1.0,
    )

    policy._scheduler_enqueue_model_maintenance(scheduler, first)
    policy._scheduler_enqueue_model_maintenance(scheduler, second)
    first_id = next(
        job_id
        for job_id, job in scheduler.jobs.items()
        if job.maintenance_event is first
    )
    second_id = next(
        job_id
        for job_id, job in scheduler.jobs.items()
        if job.maintenance_event is second
    )
    gpu = scheduler.resources[("rsu", rsu_id, "gpu")]

    policy._scheduler_dispatch(scheduler, random.Random(11))
    assert first_id in gpu.running
    assert second_id in gpu.waiting
    assert second_id not in gpu.running

    assert policy._scheduler_advance(scheduler, 0.20)
    policy._scheduler_complete_job(scheduler, first_id, random.Random(12))
    assert branch.rsus[rsu_id]["cached_models"][model.model_id] == "scenario-chain-v2"

    policy._scheduler_dispatch(scheduler, random.Random(13))
    assert second_id in gpu.running
    assert policy._scheduler_advance(scheduler, 0.21)
    policy._scheduler_complete_job(scheduler, second_id, random.Random(14))
    assert branch.rsus[rsu_id]["cached_models"][model.model_id] == "scenario-chain-v3"
    assert branch.complete_macro_recourse
    assert [
        row["job_id"]
        for row in scheduler.event_trace
        if row["kind"] == "MODEL_MAINTENANCE_COMPLETE"
    ][-2:] == [first_id, second_id]


def test_scenario_scheduler_rejects_maintenance_old_version_mismatch(
    decision_fixture,
):
    fixture = decision_fixture(
        task_id="scenario-maintenance-precondition", deadline_s=10.0
    )
    simulator = fixture.simulator
    environment = simulator.scenario_library.environment_scenarios[0]
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    branch = policy._new_branch(fixture.observation, environment)
    scheduler = policy._scheduler_new(branch, fixture.observation, environment)
    assert scheduler is not None
    rsu_id = "rsu-1"
    model = simulator.profile.edge_models["edge_fer_full_v1"]
    before = branch.rsus[rsu_id]["cached_models"][model.model_id]
    event = ScenarioVersionEvent(
        0.0,
        "MODEL_VERSION",
        "rsu",
        rsu_id,
        "model_cache",
        "not-the-current-version",
        "must-not-commit",
        model.model_id,
        False,
        0.01,
        1.0,
    )

    policy._scheduler_enqueue_model_maintenance(scheduler, event)
    job_id = next(
        job_id
        for job_id, job in scheduler.jobs.items()
        if job.maintenance_event is event
    )
    policy._scheduler_dispatch(scheduler, random.Random(21))
    assert policy._scheduler_advance(scheduler, 0.01)
    policy._scheduler_complete_job(scheduler, job_id, random.Random(22))

    assert branch.rsus[rsu_id]["cached_models"][model.model_id] == before
    assert not branch.complete_macro_recourse
    assert branch.incomplete_reason == "SCENARIO_MODEL_MAINTENANCE_VERSION_PRECONDITION"
    assert scheduler.event_trace[-1]["kind"] == (
        "MODEL_MAINTENANCE_PRECONDITION_FAILED"
    )


def test_edge_estimate_includes_current_two_sided_wireless_energy(decision_fixture):
    fixture = decision_fixture(task_id="edge-estimate-energy", deadline_s=10.0)
    simulator = fixture.simulator
    edge = next(
        row
        for row in simulator.trace.edge_rows
        if row.rsu_id == "rsu-1" and row.quality_bin == "clear"
    )
    anon = next(
        row
        for row in simulator.trace.anon_rows
        if row.artifact_key == edge.artifact_key
    )
    observation = replace(
        fixture.observation,
        stage=ActionStage.READY,
        task_state=TaskState.READY,
        selected_pipeline=edge.pipeline_id,
        artifact_token=edge.artifact_key,
        encoded_size_bytes=anon.final_encoded_size_bytes,
    )
    bounds = simulator.estimator.action_bounds(
        action=Action.edge(edge.rsu_id, edge.model_id), observation=observation
    )
    information_ablation = simulator.estimator.information_ablation_bounds(
        action=Action.edge(edge.rsu_id, edge.model_id), observation=observation
    )
    matching = simulator.estimator._edge_rows(
        Action.edge(edge.rsu_id, edge.model_id), observation
    )
    compute_energy = sum(
        row.ingress_energy_j + row.gpu_energy_j for row in matching
    ) / len(matching)

    assert observation.links[edge.rsu_id]["ul_transmitter_power_w"] > 0
    assert observation.links[edge.rsu_id]["dl_receiver_power_w"] > 0
    assert bounds["expected_vehicle_energy_j"] > 0
    assert bounds["vehicle_energy_upper_j"] >= bounds["expected_vehicle_energy_j"]
    assert bounds["expected_rsu_energy_j"] > compute_energy
    assert (
        information_ablation["conservative_output_size_cost"]
        >= (information_ablation["observed_output_size_cost"])
    )
    assert (
        information_ablation["conservative_stale_queue_cost"]
        >= (information_ablation["observed_fresh_queue_cost"])
    )


def test_unavailable_edge_link_bounds_are_json_finite_and_conservatively_missing(
    decision_fixture,
):
    fixture = decision_fixture(
        task_id="edge-estimate-unavailable-link", deadline_s=10.0
    )
    simulator = fixture.simulator
    edge = next(
        row
        for row in simulator.trace.edge_rows
        if row.rsu_id == "rsu-1" and row.quality_bin == "clear"
    )
    anon = next(
        row
        for row in simulator.trace.anon_rows
        if row.artifact_key == edge.artifact_key
    )
    links = thaw_json(fixture.observation.links)
    links[edge.rsu_id]["ul_goodput_bps"] = 0.0
    links[edge.rsu_id]["dl_goodput_bps"] = 0.0
    observation = replace(
        fixture.observation,
        stage=ActionStage.READY,
        task_state=TaskState.READY,
        selected_pipeline=edge.pipeline_id,
        artifact_token=edge.artifact_key,
        encoded_size_bytes=anon.final_encoded_size_bytes,
        links=deep_freeze(links),
    )
    bounds = simulator.estimator.action_bounds(
        action=Action.edge(edge.rsu_id, edge.model_id), observation=observation
    )

    assert bounds["optimistic_duration_s"] is None
    assert bounds["expected_duration_s"] is None
    assert bounds["vehicle_energy_upper_j"] is None
    assert bounds["expected_vehicle_energy_j"] is None
    assert bounds["link_cost"] is None
    json.dumps(bounds, allow_nan=False)


def test_ready_rollout_uses_training_pair_not_evaluation_artifact(decision_fixture):
    fixture = decision_fixture(task_id="ready-scenario-separation")
    simulator = fixture.simulator
    evaluation_anon, evaluation_edge = min(
        (
            (anon, edge)
            for anon in simulator.trace.anon_rows
            for edge in simulator.trace.edge_rows
            if anon.formed_packet
            and anon.artifact_key
            and anon.artifact_key == edge.artifact_key
            and anon.pipeline_id == edge.pipeline_id
            and anon.quality_bin == edge.quality_bin == "clear"
            and anon.device_type == fixture.observation.device_type
            and anon.context == decode_context(fixture.observation.device_context)
            and edge.rsu_id == "rsu-2"
            and edge.context
            == decode_context(fixture.observation.rsus[edge.rsu_id]["device_context"])
        ),
        key=lambda pair: (pair[0].row_id, pair[1].row_id),
    )
    capability_token = simulator.mask_engine._pairing_tokens.issue(
        fixture.task.task_id,
        [
            (
                evaluation_edge.rsu_id,
                evaluation_edge.model_id,
                evaluation_edge.pipeline_id,
            )
        ],
    )
    ready_observation = replace(
        fixture.observation,
        stage=ActionStage.READY,
        task_state=TaskState.READY,
        selected_pipeline=evaluation_edge.pipeline_id,
        artifact_token=capability_token,
        encoded_size_bytes=evaluation_anon.final_encoded_size_bytes,
    )
    action = Action.edge(evaluation_edge.rsu_id, evaluation_edge.model_id)
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )

    assert simulator.estimator.has_edge_support(
        rsu_id=evaluation_edge.rsu_id,
        model_id=evaluation_edge.model_id,
        pipeline_id=evaluation_edge.pipeline_id,
        artifact_token=ready_observation.artifact_token,
        evaluation_pair_supported=True,
        quality_bins=ready_observation.conformal_quality_bins,
        profile_hash=simulator.profile.profile_hash,
        device_type=ready_observation.device_type,
        device_context=ready_observation.device_context,
        rsu_context=ready_observation.rsus[evaluation_edge.rsu_id]["device_context"],
    )
    assert not simulator.estimator.has_edge_support(
        rsu_id=evaluation_edge.rsu_id,
        model_id=evaluation_edge.model_id,
        pipeline_id=evaluation_edge.pipeline_id,
        artifact_token=ready_observation.artifact_token,
        evaluation_pair_supported=False,
        quality_bins=ready_observation.conformal_quality_bins,
        profile_hash=simulator.profile.profile_hash,
        device_type=ready_observation.device_type,
        device_context=ready_observation.device_context,
        rsu_context=ready_observation.rsus[evaluation_edge.rsu_id]["device_context"],
    )

    candidates = policy._trace_candidates(action, ready_observation)
    assert candidates
    assert all(row.artifact_token != evaluation_edge.artifact_key for row in candidates)
    _, _, sampled_rows = policy._one_rollout(
        fixture.task,
        ready_observation,
        action,
        scenario_index=0,
    )
    assert sampled_rows[0] in {row.scenario_id for row in candidates}


def test_synthetic_split_metadata_does_not_claim_subject_independence(decision_fixture):
    simulator = decision_fixture(task_id="synthetic-split-metadata").simulator
    for frozen_trace in (simulator.trace, simulator.scenario_trace):
        split = frozen_trace.metadata["data_split"]
        assert "independent_from_other_synthetic_split" not in split
        assert split["independence_scope"] == "independent_random_realization_only"
        assert split["artifact_namespace_disjoint"] is True
        assert split["fixture_namespace_disjoint"] is False
        assert split["subject_population_disjoint"] is False


def test_rsu_scheduler_snapshot_is_detached_from_live_admission(
    decision_fixture, profile, config
):
    fixture = decision_fixture(task_id="snapshot-isolation")
    runtime = fixture.state.rsus["rsu-1"]
    model = profile.edge_models["edge_fer_full_v1"]
    before = fixture.simulator._observation(fixture.task).rsus["rsu-1"]

    accepted, reasons = runtime.admission.admit(
        AdmissionRequest(
            task_id="unpublished-reservation",
            descriptor_count=1,
            vram_bytes=1,
            conservative_work_gpu_s=0.001,
            model_id=model.model_id,
            model_hash=model.model_hash,
            protocol_version=config.protocol_version,
            message_valid=True,
        )
    )
    after = fixture.simulator._observation(fixture.task).rsus["rsu-1"]

    assert accepted and not reasons
    assert runtime.admission.snapshot().descriptors == 1
    assert before["descriptors"] == after["descriptors"] == 0
    assert before["reserved_work_gpu_s"] == after["reserved_work_gpu_s"] == 0.0
    fixture.state.clock_s = 0.25
    fixture.simulator._handle_rsu_snapshot(
        SimpleNamespace(object_id="rsu-1", payload=None)
    )
    if runtime.current_snapshot_time_s != 0.25:
        fixture.simulator._handle_rsu_snapshot(
            SimpleNamespace(
                object_id="rsu-1",
                payload={
                    "phase": "delivery",
                    "sample_time_s": 0.25,
                    "snapshot": fixture.simulator._rsu_public_snapshot(
                        runtime, "rsu-1", 0.25
                    ),
                },
            )
        )
    refreshed = fixture.simulator._observation(fixture.task).rsus["rsu-1"]
    assert refreshed["snapshot_time_s"] == 0.25
    assert refreshed["descriptors"] == 1
    runtime.admission.release("unpublished-reservation")


def test_atomic_admission_uses_candidate_profile_envelope_not_hidden_true_quality(
    decision_fixture,
):
    """A hidden g* cannot shrink the all-candidate RSU reservation request."""

    candidate_bins = ("clear", "challenging")
    live_work_capacity = 0.019

    def exercise(true_quality: str):
        fixture = decision_fixture(
            task_id=f"admission-envelope-{true_quality}",
            deadline_s=10.0,
            bins=candidate_bins,
        )
        simulator = fixture.simulator
        task = fixture.task
        task.true_quality_region = true_quality
        rsu_id = "rsu-1"
        model_id = "edge_fer_full_v1"
        pipeline_id = "pixelate_strong_v1"
        vehicle_context = decode_context(fixture.observation.device_context)
        rsu_context = simulator._device_context("rsu", rsu_id, simulator.state.clock_s)
        pairs = sorted(
            (
                (anon, edge)
                for anon in simulator.trace.anon_rows
                for edge in simulator.trace.edge_rows
                if anon.formed_packet
                and anon.artifact_key
                and anon.pipeline_id == pipeline_id
                and anon.quality_bin == true_quality
                and anon.device_type == fixture.observation.device_type
                and anon.context == vehicle_context
                and edge.artifact_key == anon.artifact_key
                and edge.pipeline_id == pipeline_id
                and edge.quality_bin == true_quality
                and edge.rsu_id == rsu_id
                and edge.model_id == model_id
                and edge.context == rsu_context
            ),
            key=lambda pair: (pair[0].row_id, pair[1].row_id),
        )
        assert pairs
        anon, actual_edge = pairs[0]
        pipeline = simulator.profile.pipelines[pipeline_id]
        for target in (
            TaskState.ANON_WAIT,
            TaskState.ANON_RUN,
            TaskState.GUARD_WAIT,
            TaskState.GUARD_RUN,
            TaskState.ENCODE_WAIT,
            TaskState.ENCODE_RUN,
            TaskState.READY,
        ):
            TaskStateMachine.transition(
                task, target, time_s=0.0, trigger="UNIT_ADMISSION_ENVELOPE"
            )
        task.selected_pipeline = pipeline_id
        task.artifact_key = anon.artifact_key
        anonymized = _replay_anonymization_success(
            aligned=task.aligned_handle,
            task_id=task.task_id,
            pipeline_id=pipeline_id,
            pipeline_hash=pipeline.pipeline_hash,
            artifact_key=anon.artifact_key,
            attempt=1,
        )
        guarded = _replay_guard_success(
            anonymized,
            guard_hash=pipeline.guard_hash,
            guard_certificate_id=f"guard-{task.task_id}",
        )
        encoded = _replay_encoding_success(
            guarded,
            payload=b"a" * anon.final_encoded_size_bytes,
            encoder_hash=pipeline.encoder_hash,
            encoded_size_bytes=anon.final_encoded_size_bytes,
        )
        task.encoded_anon = _finalize_encoded_anon(
            encoded,
            profile_hash=simulator.profile.profile_hash,
            quality_bins=candidate_bins,
        )
        task.encoded_size_bytes = anon.final_encoded_size_bytes

        admission = simulator.state.rsus[rsu_id].admission
        admission.workload_capacity_gpu_s = live_work_capacity
        before = admission.snapshot()
        simulator._start_edge_action(task, rsu_id, model_id)
        transfer = simulator.state.transfers[task.current_transfer_id or ""]
        simulator._admit_at_rsu(task, transfer)
        after = admission.snapshot()
        audit = next(row for row in task.rsu_audit if "admission" in row)
        return (
            audit,
            actual_edge,
            before,
            after,
            simulator.profile.edge_models[model_id].deployment_resource_bounds,
        )

    clear, clear_edge, clear_before, clear_after, profile_bounds = exercise("clear")
    challenging, challenging_edge, challenging_before, challenging_after, _ = exercise(
        "challenging"
    )

    # This is the counterexample: an exact-g* request would fit only for clear.
    assert clear_edge.gpu_work_s < live_work_capacity < challenging_edge.gpu_work_s
    assert float(profile_bounds["max_gpu_work_s"]) > live_work_capacity

    # Both requests instead reserve the same preregistered all-candidate bound,
    # so changing hidden g* cannot alter the request or atomic result.
    assert clear["request"] == challenging["request"]
    assert clear["request"]["candidate_quality_bins"] == list(candidate_bins)
    assert clear["request"]["conservative_work_gpu_s"] == pytest.approx(
        profile_bounds["max_gpu_work_s"]
    )
    assert clear["admission"] == challenging["admission"] == "REJECT"
    assert (
        clear["reason_codes"]
        == challenging["reason_codes"]
        == ["RSU_WORKLOAD_CAPACITY"]
    )
    assert clear_before == clear_after
    assert challenging_before == challenging_after


def test_terminal_tasks_hold_no_live_jobs_transfers_or_reservations(policy_results):
    for result in policy_results.values():
        assert all(task.current_job_id is None for task in result.state.tasks.values())
        assert all(
            task.current_transfer_id is None for task in result.state.tasks.values()
        )
        assert all(
            task.memory_reservation_bytes == 0 for task in result.state.tasks.values()
        )
        assert all(not task.reservation_tokens for task in result.state.tasks.values())
        assert not result.state.transfers
        for vehicle in result.state.vehicles.values():
            assert vehicle.memory_reserved_bytes == 0
            assert all(value == 0 for value in vehicle.descriptors_reserved.values())
            for resource in vehicle.resources.values():
                assert resource.running_count == 0
                assert resource.waiting_count == 0
        for rsu in result.state.rsus.values():
            assert rsu.ingress.running_count == rsu.ingress.waiting_count == 0
            assert rsu.gpu.running_count == rsu.gpu.waiting_count == 0
            assert rsu.admission.snapshot().reservations == ()


def test_deadline_aborts_task_and_never_counts_late_result(
    config, profile, deadline_trace
):
    result = DiscreteEventSimulator(
        config,
        profile,
        deadline_trace,
        "all_local",
        policy_name="deadline-fixture",
    ).run()
    task = result.state.tasks["task-001"]

    assert task.state is TaskState.FAIL
    assert task.failure_reason is FailureReason.TIMEOUT
    assert task.terminal_time_s == pytest.approx(task.absolute_deadline_s)
    assert not task.result_valid
    assert not result.invariant_failures


def test_battery_depletion_is_a_physical_event_and_never_goes_negative(
    config, profile, trace
):
    vehicles = list(config.vehicles)
    vehicles[0] = replace(
        vehicles[0],
        battery_capacity_j=0.1,
        initial_battery_j=0.1,
    )
    low_battery_config = replace(config, vehicles=tuple(vehicles))
    result = DiscreteEventSimulator(
        low_battery_config,
        profile,
        trace,
        "all_local",
        policy_name="battery-fixture",
    ).run()
    vehicle = result.state.vehicles["veh-1"]
    its_tasks = [
        task for task in result.state.tasks.values() if task.vehicle_id == "veh-1"
    ]

    assert vehicle.battery_j == 0
    assert vehicle.battery_depleted and vehicle.failed
    assert its_tasks
    assert all(task.state is TaskState.FAIL for task in its_tasks)
    assert all(task.failure_reason is FailureReason.BATTERY_GUARD for task in its_tasks)
    assert not result.invariant_failures


def test_dispatch_battery_recheck_can_legally_fallback_from_anon_wait(decision_fixture):
    fixture = decision_fixture(task_id="battery-recheck", deadline_s=10.0)
    simulator = fixture.simulator
    task = fixture.task
    vehicle = simulator.state.vehicles[task.vehicle_id]

    simulator._start_pipeline_action(task, "pixelate_strong_v1")
    anon_job = next(
        job
        for pool in vehicle.resources.values()
        for job in pool.active_jobs_for_task(task.task_id)
    )
    local_rows = [
        row
        for row in simulator.trace.local_rows
        if row.model_id == "local_fer_compact_v1"
        and row.device_type == vehicle.device_type
        and row.quality_bin == task.true_quality_region
        and row.context.thermal_state == "nominal"
    ]
    local_energy_upper = max(row.dynamic_energy_j for row in local_rows)
    local_required = local_energy_upper
    anon_required = anon_job.total_dynamic_energy_j
    assert anon_required > local_required
    vehicle.battery_j = (local_required + anon_required) / 2.0

    simulator._dispatch_all()

    assert task.state in {TaskState.LOCAL_WAIT, TaskState.LOCAL_RUN}
    assert task.selected_local_model == "local_fer_compact_v1"
    assert any(
        row.get("repair") == "FROZEN_LOCAL_FALLBACK" for row in task.action_audit
    )
    output_rows = simulator.ledger._action_output_rows(simulator.state)
    fallback_rows = [
        row for row in output_rows if row.get("repair") == "FROZEN_LOCAL_FALLBACK"
    ]
    assert fallback_rows
    assert all(row["record_kind"] == "EXECUTION_REPAIR" for row in fallback_rows)


@pytest.mark.parametrize(
    ("fault_time_s", "failed_direction"),
    [(0.30, "UL"), (0.34250, "DL")],
)
def test_rsu_permanent_fault_cancels_inflight_radio_and_prevents_edge_result(
    fault_time_s,
    failed_direction,
    config,
    profile,
    trace,
):
    fault = ExogenousEvent(
        event_id=f"test-rsu-fault-{failed_direction.lower()}",
        time_s=fault_time_s,
        event_type="DEVICE_FAULT_PERMANENT",
        target_type="rsu",
        target_id="rsu-1",
        resource="all",
        old_version=None,
        new_version=None,
        permanent=True,
        details={},
    )
    fault_trace = replace(
        trace,
        exogenous_events=tuple(
            sorted(
                (*trace.exogenous_events, fault),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )

    result = DiscreteEventSimulator(
        config,
        profile,
        fault_trace,
        "fixed_safe_lowest_link_cost",
    ).run()
    task = result.state.tasks["task-001"]
    direction_rows = [
        row for row in task.network_audit if row["direction"] == failed_direction
    ]

    assert any(row["status"] == "FAIL" for row in direction_rows)
    assert not any(
        row["status"] == "DONE" and row["time_s"] > fault_time_s
        for row in direction_rows
    )
    assert task.actual_path[-1] == "LOCAL_FER"
    if failed_direction == "UL":
        assert not any(row.get("admission") == "ACCEPT" for row in task.rsu_audit)


def test_profile_change_during_uplink_blocks_admission_and_old_profile_fallback(
    config,
    profile,
    trace,
):
    version_event = ExogenousEvent(
        event_id="test-profile-version-during-ul",
        time_s=0.30,
        event_type="PROFILE_VERSION",
        target_type="deployment",
        target_id="global-profile",
        resource="profile",
        old_version=profile.profile_hash,
        new_version="unprofiled-profile-version",
        permanent=False,
        details={},
    )
    version_trace = replace(
        trace,
        exogenous_events=tuple(
            sorted(
                (*trace.exogenous_events, version_event),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )

    result = DiscreteEventSimulator(
        config,
        profile,
        version_trace,
        "fixed_safe_lowest_link_cost",
    ).run()
    task = result.state.tasks["task-001"]

    assert any(
        row["direction"] == "UL" and row["status"] == "DONE"
        for row in task.network_audit
    )
    assert not any(row.get("admission") == "ACCEPT" for row in task.rsu_audit)
    assert task.state is TaskState.FAIL
    assert task.actual_path[-1].startswith("UL:")


@pytest.mark.parametrize(
    ("event_type", "old_version", "new_version"),
    [
        ("PROFILE_VERSION", "profile", "unprofiled-profile-version"),
        ("PROTOCOL_VERSION", "protocol", "anon-fer/2.0"),
    ],
)
def test_profile_or_protocol_change_during_downlink_invalidates_result(
    event_type,
    old_version,
    new_version,
    config,
    profile,
    trace,
):
    resolved_old = (
        profile.profile_hash if old_version == "profile" else config.protocol_version
    )
    event = ExogenousEvent(
        event_id=f"test-{event_type.lower()}-during-dl",
        time_s=0.33635,
        event_type=event_type,
        target_type="deployment",
        target_id="global",
        resource="version",
        old_version=resolved_old,
        new_version=new_version,
        permanent=False,
        details={},
    )
    changed = replace(
        trace,
        exogenous_events=tuple(
            sorted(
                (*trace.exogenous_events, event),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )

    result = DiscreteEventSimulator(
        config,
        profile,
        changed,
        "fixed_safe_lowest_link_cost",
    ).run()
    task = result.state.tasks["task-001"]

    assert any(
        row["direction"] == "DL" and row["status"] == "DONE"
        for row in task.network_audit
    )
    assert task.state is TaskState.FAIL
    assert task.failure_reason is FailureReason.VERSION_MISMATCH
    assert not task.result_valid


def test_vehicle_local_model_version_change_blocks_new_old_model_work(
    config,
    profile,
    trace,
):
    model = profile.local_models["local_fer_compact_v1"]
    event = ExogenousEvent(
        event_id="test-local-model-version",
        time_s=0.10,
        event_type="MODEL_VERSION",
        target_type="vehicle",
        target_id="veh-1",
        resource="local_model",
        old_version=model.model_hash,
        new_version="unprofiled-local-model-version",
        permanent=False,
        details={"model_id": model.model_id},
    )
    changed = replace(
        trace,
        exogenous_events=tuple(
            sorted(
                (*trace.exogenous_events, event),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    result = DiscreteEventSimulator(config, profile, changed, "all_local").run()
    task = result.state.tasks["task-001"]

    assert task.state is TaskState.FAIL
    assert "LOCAL_FER" not in task.actual_path


def test_rsu_model_version_commits_only_after_taskless_gpu_maintenance(
    config,
    profile,
    trace,
):
    simulator = DiscreteEventSimulator(config, profile, trace, "all_local")
    rsu = simulator.state.rsus["rsu-1"]
    model = profile.edge_models["edge_fer_full_v1"]
    old_hash = model.model_hash
    new_hash = "post-maintenance-unprofiled-model"
    pinned_request = AdmissionRequest(
        task_id="already-pinned",
        descriptor_count=1,
        vram_bytes=1,
        conservative_work_gpu_s=0.001,
        model_id=model.model_id,
        model_hash=old_hash,
        protocol_version=config.protocol_version,
        message_valid=True,
    )
    accepted, reasons = rsu.admission.admit(pinned_request)
    assert accepted and not reasons

    maintenance = ExogenousEvent(
        event_id="unit-rsu-model-maintenance",
        time_s=0.0,
        event_type="MODEL_VERSION",
        target_type="rsu",
        target_id="rsu-1",
        resource="model_cache",
        old_version=old_hash,
        new_version=new_hash,
        permanent=False,
        details={"model_id": model.model_id},
        maintenance_work_s=0.20,
        maintenance_energy_j=10.0,
    )
    simulator._handle_version_event(SimpleNamespace(payload=maintenance))
    jobs = [
        job
        for job in rsu.gpu.jobs.values()
        if job.operation is Operation.RSU_MODEL_MAINTENANCE
    ]
    assert len(jobs) == 1
    job = jobs[0]
    assert job.task_id is None
    assert rsu.admission.cached_models[model.model_id] == old_hash

    simulator._dispatch_all()
    assert job.start_time_s == 0.0
    simulator._advance_to(0.10)
    assert rsu.admission.cached_models[model.model_id] == old_hash
    assert rsu.system_maintenance_energy_j == pytest.approx(5.0)
    assert sum(task.rsu_energy_j for task in simulator.state.tasks.values()) == 0.0

    simulator._advance_to(0.20)
    simulator._handle_compute_completion(
        SimpleNamespace(
            object_id=job.job_id,
            version_token=job.completion_version,
        )
    )
    assert rsu.admission.cached_models[model.model_id] == new_hash
    assert rsu.admission.pinned_model("already-pinned") == (model.model_id, old_hash)
    assert rsu.system_maintenance_energy_j == pytest.approx(10.0)
    assert rsu.physical_energy_j > rsu.system_maintenance_energy_j
    assert rsu.gpu.busy_server_seconds == pytest.approx(0.20)

    rejected, rejection_reasons = rsu.admission.admit(
        replace(pinned_request, task_id="new-old-model-request")
    )
    assert not rejected
    assert rejection_reasons == ("MODEL_CACHE_MISSING",)
    assert {
        row["event_kind"]
        for row in simulator.ledger.event_rows
        if row.get("event_id") == maintenance.event_id
    } == {
        "RSU_MODEL_MAINTENANCE_ENQUEUE",
        "RSU_MODEL_MAINTENANCE_START",
        "RSU_MODEL_MAINTENANCE_COMPLETE",
    }
    rsu.admission.release("already-pinned")


def test_rsu_serializes_chained_model_maintenance_and_checks_old_version(
    config,
    profile,
    trace,
):
    simulator = DiscreteEventSimulator(config, profile, trace, "all_local")
    rsu = simulator.state.rsus["rsu-1"]
    assert rsu.gpu.server_count >= 2
    model = profile.edge_models["edge_fer_full_v1"]
    first = ExogenousEvent(
        "unit-chain-v2",
        0.0,
        "MODEL_VERSION",
        "rsu",
        "rsu-1",
        "model_cache",
        model.model_hash,
        "unit-chain-v2-hash",
        False,
        {"model_id": model.model_id},
        0.20,
        5.0,
    )
    second = ExogenousEvent(
        "unit-chain-v3",
        0.01,
        "MODEL_CACHE",
        "rsu",
        "rsu-1",
        "model_cache",
        "unit-chain-v2-hash",
        "unit-chain-v3-hash",
        False,
        {"model_id": model.model_id},
        0.01,
        1.0,
    )

    simulator._handle_version_event(SimpleNamespace(payload=first))
    simulator._handle_version_event(SimpleNamespace(payload=second))
    first_id = next(
        job_id
        for job_id, event in simulator._rsu_maintenance_events.items()
        if event is first
    )
    second_id = next(
        job_id
        for job_id, event in simulator._rsu_maintenance_events.items()
        if event is second
    )
    first_job = rsu.gpu.jobs[first_id]
    second_job = rsu.gpu.jobs[second_id]

    simulator._dispatch_all()
    assert rsu.gpu.running_count == 1
    assert first_job.status.value == "RUNNING"
    assert second_job.status.value == "WAITING"

    simulator._advance_to(0.20)
    simulator._handle_compute_completion(
        SimpleNamespace(
            object_id=first_id,
            version_token=first_job.completion_version,
        )
    )
    assert rsu.admission.cached_models[model.model_id] == "unit-chain-v2-hash"

    simulator._dispatch_all()
    assert second_job.status.value == "RUNNING"
    assert second_job.start_time_s == pytest.approx(0.20)
    simulator._advance_to(0.21)
    simulator._handle_compute_completion(
        SimpleNamespace(
            object_id=second_id,
            version_token=second_job.completion_version,
        )
    )
    assert rsu.admission.cached_models[model.model_id] == "unit-chain-v3-hash"
    assert [
        row["event_id"]
        for row in simulator.ledger.event_rows
        if row.get("event_kind") == "RSU_MODEL_MAINTENANCE_COMPLETE"
        and row.get("event_id") in {first.event_id, second.event_id}
    ] == [first.event_id, second.event_id]

    invalid = ExogenousEvent(
        "unit-chain-invalid-old",
        0.21,
        "MODEL_VERSION",
        "rsu",
        "rsu-1",
        "model_cache",
        "not-the-current-version",
        "must-not-commit",
        False,
        {"model_id": model.model_id},
        0.01,
        1.0,
    )
    simulator._handle_version_event(SimpleNamespace(payload=invalid))
    invalid_id = next(
        job_id
        for job_id, event in simulator._rsu_maintenance_events.items()
        if event is invalid
    )
    invalid_job = rsu.gpu.jobs[invalid_id]
    simulator._dispatch_all()
    simulator._advance_to(0.22)
    with pytest.raises(
        InvariantViolation, match="RSU_MODEL_MAINTENANCE_VERSION_PRECONDITION"
    ):
        simulator._handle_compute_completion(
            SimpleNamespace(
                object_id=invalid_id,
                version_token=invalid_job.completion_version,
            )
        )
    assert rsu.admission.cached_models[model.model_id] == "unit-chain-v3-hash"


def test_rsu_model_cache_removal_waits_for_gpu_maintenance_completion(
    config,
    profile,
    trace,
):
    simulator = DiscreteEventSimulator(config, profile, trace, "all_local")
    rsu = simulator.state.rsus["rsu-1"]
    model = profile.edge_models["edge_fer_full_v1"]
    event = ExogenousEvent(
        event_id="unit-rsu-model-cache-remove",
        time_s=0.0,
        event_type="MODEL_CACHE",
        target_type="rsu",
        target_id="rsu-1",
        resource="model_cache",
        old_version=None,
        new_version=None,
        permanent=False,
        details={"model_id": model.model_id, "remove": True},
        maintenance_work_s=0.05,
        maintenance_energy_j=2.0,
    )

    simulator._handle_version_event(SimpleNamespace(payload=event))
    job = next(
        job
        for job in rsu.gpu.jobs.values()
        if job.operation is Operation.RSU_MODEL_MAINTENANCE
    )
    assert rsu.admission.cached_models[model.model_id] == model.model_hash
    simulator._dispatch_all()
    simulator._advance_to(0.05)
    assert rsu.admission.cached_models[model.model_id] == model.model_hash
    simulator._handle_compute_completion(
        SimpleNamespace(
            object_id=job.job_id,
            version_token=job.completion_version,
        )
    )
    assert model.model_id not in rsu.admission.cached_models


def test_all_policy_actions_are_auditable_and_never_select_removed_action(
    policy_results,
):
    for result in policy_results.values():
        for task in result.state.tasks.values():
            for action_row in task.action_audit:
                if "executed" not in action_row:
                    continue
                executed = action_row["executed"]
                assert executed
                mask_rows = {
                    row["action_id"]: row
                    for audit in task.mask_audit
                    for row in audit.get("rows", audit.get("actions", ()))
                }
                # Some terminal/fallback actions have a dedicated audit record;
                # whenever the originating mask row is retained it must be allowed.
                action_id = "|".join(
                    (
                        executed.get("stage", ""),
                        executed.get("kind", ""),
                        executed.get("local_model_id", ""),
                        executed.get("pipeline_id", ""),
                        executed.get("rsu_id", ""),
                        executed.get("edge_model_id", ""),
                    )
                )
                if action_id in mask_rows:
                    assert mask_rows[action_id]["allowed"] is True


def test_branch_decision_overhead_can_lose_to_deadline_without_sampling_outcome(
    decision_fixture,
):
    fixture = decision_fixture(task_id="branch-decision-deadline", deadline_s=10.0)
    simulator = fixture.simulator
    overhead = simulator.config.controller.controller_overhead_s
    assert overhead > 0
    slack = overhead / 2
    observation = replace(
        fixture.observation,
        absolute_deadline_s=fixture.observation.time_s + slack,
        slack_s=slack,
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        horizon_events=2,
        scenarios=1,
        scenario_source=simulator.scenario_library,
    )
    action = Action.local(ActionStage.RAW, min(simulator.profile.local_models))

    _, duration_s, sampled_rows, details = policy._one_rollout(
        fixture.task,
        observation,
        action,
        scenario_index=0,
        include_diagnostics=True,
        stop_before_next_macro=True,
    )

    assert duration_s == pytest.approx(slack)
    assert sampled_rows == ()
    assert any(row["kind"] == "DECISION_START" for row in details["scheduler_trace"])
    assert not any(
        row["kind"] == "DECISION_COMMIT" for row in details["scheduler_trace"]
    )
    assert any(
        row["kind"] == "TASK_TERMINAL" and row["reason"] == "DEADLINE"
        for row in details["scheduler_trace"]
    )
