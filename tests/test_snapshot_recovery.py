from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import random
from typing import Any

import pytest

from privacy_edge_sim.enums import ActionStage, TaskState
from privacy_edge_sim.packets import AlignedTensorHandle, RawImageHandle
from privacy_edge_sim.policies import POLICY_REGISTRY
from privacy_edge_sim.profiles import deep_freeze, thaw_json
from privacy_edge_sim.safety import Action
from privacy_edge_sim.state import TaskRecord, TaskStateMachine


def _move_to_raw(task: TaskRecord) -> None:
    for state in (TaskState.PREP_WAIT, TaskState.PREP_RUN, TaskState.RAW):
        TaskStateMachine.transition(task, state, time_s=0.0, trigger="TEST_SETUP")


def _install_second_raw_task(fixture) -> TaskRecord:
    source = fixture.task
    deadline_s = source.absolute_deadline_s + 1.0
    task = TaskRecord(
        task_id="private-second-real-task-id",
        vehicle_id=source.vehicle_id,
        arrival_time_s=0.0,
        relative_deadline_s=deadline_s,
        absolute_deadline_s=deadline_s,
        raw_handle=RawImageHandle("private-second-raw-handle"),
        aligned_handle=AlignedTensorHandle("private-second-aligned-handle"),
        quality_features=source.quality_features,
        quality_probabilities=source.quality_probabilities,
        conformal_quality_bins=source.conformal_quality_bins,
        ood=source.ood,
        true_identity="simulator-only-second-person",
        true_expression_label="simulator-only-second-label",
        true_quality_region=source.true_quality_region,
        realized_attack_outcomes={"identity": False},
    )
    fixture.simulator.state.tasks[task.task_id] = task
    _move_to_raw(task)
    return task


def _capture_same_time_second_observation(decision_fixture, monkeypatch):
    fixture = decision_fixture(task_id="private-first-real-task-id", deadline_s=4.0)
    simulator = fixture.simulator
    second = _install_second_raw_task(fixture)
    local = Action.local(ActionStage.RAW, min(simulator.profile.local_models))
    observations = {}

    def choose(task, observation, mask):
        assert local in mask.candidates
        observations[task.task_id] = observation
        return local

    monkeypatch.setattr(simulator, "_policy_choose", choose)
    simulator._make_decisions()
    return fixture, second, local, observations[second.task_id]


def _clean_environment(simulator):
    environment = next(
        row
        for row in simulator.scenario_library.environment_scenarios
        if all(anchor.active_task_count == 0 for anchor in row.rsu_anchors)
    )
    return replace(
        environment,
        wireless=(),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
        versions=(),
        future_tasks=(),
        vehicle_anchors=(),
    )


def _policy_and_scheduler(simulator, observation, environment):
    policy = POLICY_REGISTRY["esl_smpc"](
        simulator.mask_engine,
        simulator.repairer,
        scenario_source=simulator.scenario_library,
    )
    branch = policy._new_branch(observation, environment)
    scheduler = policy._scheduler_new(branch, observation, environment)
    return policy, branch, scheduler


def _all_mapping_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            nested for item in value.values() for nested in _all_mapping_keys(item)
        }
    if isinstance(value, (tuple, list)):
        return {nested for item in value for nested in _all_mapping_keys(item)}
    return set()


def test_second_same_time_observation_exposes_first_pending_decision_anonymously(
    decision_fixture, monkeypatch
):
    fixture, second, local, observation = _capture_same_time_second_observation(
        decision_fixture, monkeypatch
    )
    first = fixture.task

    assert observation.task_id == second.task_id
    assert observation.time_s == pytest.approx(0.0)
    first_row = next(
        row for row in observation.vehicle["active_tasks"] if not row["is_focal"]
    )
    pending = first_row["pending_decision"]
    assert first_row["task_token"] != first.task_id
    assert pending["proposed"] == local.to_dict()
    assert pending["remaining_overhead_s"] == pytest.approx(
        fixture.simulator.config.controller.controller_overhead_s
    )
    assert pending["controller_energy_already_charged"] is True
    assert not {"task_id", "job_id"}.intersection(_all_mapping_keys(first_row))
    assert first.task_id not in repr(thaw_json(first_row))


def test_esl_restores_pending_decision_without_redecision_or_second_energy_charge(
    decision_fixture, monkeypatch
):
    fixture, _, local, observation = _capture_same_time_second_observation(
        decision_fixture, monkeypatch
    )
    simulator = fixture.simulator
    environment = _clean_environment(simulator)
    policy, branch, scheduler = _policy_and_scheduler(
        simulator, observation, environment
    )
    assert scheduler is not None
    pending_row = next(
        row for row in observation.vehicle["active_tasks"] if not row["is_focal"]
    )
    token = pending_row["task_token"]
    restored = scheduler.tasks[token]
    remaining_overhead_s = pending_row["pending_decision"]["remaining_overhead_s"]
    assert restored.state == "DECISION_WAIT"
    assert restored.pending_action == local
    assert any(
        time_s == pytest.approx(remaining_overhead_s)
        and kind == "DECISION_COMMIT"
        and object_id == token
        for time_s, _, _, kind, object_id in scheduler.events
    )

    def unexpected_redecision(*args, **kwargs):
        pytest.fail("restored pending action was proposed a second time")

    monkeypatch.setattr(policy, "_scheduler_schedule_decision", unexpected_redecision)
    original_commit = policy._scheduler_commit
    committed = {}

    def stop_after_commit(scheduler_arg, task, action, outcome, observation, **kwargs):
        committed.update(time_s=scheduler_arg.branch.elapsed_s, action=action)
        original_commit(scheduler_arg, task, action, outcome, observation, **kwargs)
        scheduler_arg.branch.complete_macro_recourse = False

    monkeypatch.setattr(policy, "_scheduler_commit", stop_after_commit)
    vehicle_id = observation.vehicle_id
    initial_battery_j = scheduler.vehicle_battery_j[vehicle_id]
    active_links = policy._scheduler_active_link_counts(scheduler)
    physical_power_w = policy._scheduler_vehicle_power_w(
        scheduler, vehicle_id, active_links
    )

    policy._scheduler_run(scheduler, random.Random(37))

    assert committed["time_s"] == pytest.approx(remaining_overhead_s)
    assert committed["action"] == local
    expected_physical_energy_j = physical_power_w * remaining_overhead_s
    assert initial_battery_j - scheduler.vehicle_battery_j[vehicle_id] == pytest.approx(
        expected_physical_energy_j
    )
    assert branch.vehicle_physical_energy_j[vehicle_id] == pytest.approx(
        expected_physical_energy_j
    )
    assert not any(
        row["kind"] == "DECISION_START" and row.get("task_token") == token
        for row in scheduler.event_trace
    )
    assert any(
        row["kind"] == "DECISION_COMMIT"
        and row["task_token"] == token
        and row["time_s"] == pytest.approx(remaining_overhead_s)
        for row in scheduler.event_trace
    )


def test_pending_decision_exact_deadline_is_cancelled_before_commit(
    decision_fixture, monkeypatch
):
    fixture, _, _, observation = _capture_same_time_second_observation(
        decision_fixture, monkeypatch
    )
    vehicle = thaw_json(observation.vehicle)
    pending = next(row for row in vehicle["active_tasks"] if not row["is_focal"])
    remaining = pending["pending_decision"]["remaining_overhead_s"]
    pending["deadline_offset_s"] = remaining
    changed = replace(observation, vehicle=deep_freeze(vehicle))
    policy, branch, scheduler = _policy_and_scheduler(
        fixture.simulator, changed, _clean_environment(fixture.simulator)
    )
    assert scheduler is not None
    token = pending["task_token"]

    def forbidden_commit(*args, **kwargs):
        pytest.fail("deadline must absorb the task before pending decision commit")

    monkeypatch.setattr(policy, "_scheduler_commit", forbidden_commit)
    policy._scheduler_run(scheduler, random.Random(41))

    assert scheduler.tasks[token].state == "FAIL"
    assert any(
        row["kind"] == "TASK_TERMINAL"
        and row["task_token"] == token
        and row["reason"] == "DEADLINE"
        for row in scheduler.event_trace
    )
    assert branch.complete_macro_recourse is True


@pytest.mark.parametrize(
    "tamper",
    ("unowned_aggregate", "owned_but_aggregate_missing"),
)
def test_live_reservation_ownership_mismatch_is_conservatively_incomplete(
    decision_fixture, tamper
):
    fixture = decision_fixture(task_id=f"live-ownership-{tamper}")
    observation = fixture.observation
    vehicle = thaw_json(observation.vehicle)
    if tamper == "unowned_aggregate":
        vehicle["memory_reserved_bytes"] = 1
        vehicle["memory_remaining_bytes"] = vehicle["memory_capacity_bytes"] - 1
    else:
        vehicle.pop("memory_reserved_bytes")
        focal = next(row for row in vehicle["active_tasks"] if row["is_focal"])
        focal["memory_reservation_bytes"] = 1
    changed = replace(observation, vehicle=deep_freeze(vehicle))
    _, branch, scheduler = _policy_and_scheduler(
        fixture.simulator, changed, _clean_environment(fixture.simulator)
    )

    assert scheduler is None
    assert branch.complete_macro_recourse is False
    assert branch.incomplete_reason == "LIVE_RESERVATION_OWNERSHIP_MISMATCH"


def _tamper_active_rsu_anchor(environment, tamper: str):
    rsu = next(anchor for anchor in environment.rsu_anchors if anchor.active_task_count)
    vehicle = next(
        anchor
        for anchor in environment.vehicle_anchors
        if any(task.rsu_id == rsu.rsu_id for task in anchor.tasks)
    )
    task = next(task for task in vehicle.tasks if task.rsu_id == rsu.rsu_id)

    if tamper == "aggregate":
        changed_rsu = replace(rsu, descriptors_reserved=rsu.descriptors_reserved + 1)
        return replace(
            environment,
            rsu_anchors=tuple(
                changed_rsu if anchor.rsu_id == rsu.rsu_id else anchor
                for anchor in environment.rsu_anchors
            ),
        )
    if tamper == "task":
        changed_vehicle = replace(
            vehicle,
            tasks=tuple(
                item for item in vehicle.tasks if item.task_token != task.task_token
            ),
        )
        return replace(
            environment,
            vehicle_anchors=tuple(
                changed_vehicle if anchor.vehicle_id == vehicle.vehicle_id else anchor
                for anchor in environment.vehicle_anchors
            ),
        )
    if tamper == "job":
        resources = thaw_json(rsu.resources)
        resource_name = next(
            name
            for name, row in resources.items()
            if row["running_jobs"] or row["waiting_jobs"]
        )
        row = resources[resource_name]
        bucket = "running_jobs" if row["running_jobs"] else "waiting_jobs"
        row[bucket] = tuple(
            item for item in row[bucket] if item["task_token"] != task.task_token
        )
        changed_rsu = replace(rsu, resources=deep_freeze(resources))
        return replace(
            environment,
            rsu_anchors=tuple(
                changed_rsu if anchor.rsu_id == rsu.rsu_id else anchor
                for anchor in environment.rsu_anchors
            ),
        )
    changed_task = replace(
        task, admission_vram_upper_bytes=task.admission_vram_upper_bytes + 1
    )
    changed_vehicle = replace(
        vehicle,
        tasks=tuple(
            changed_task if item.task_token == task.task_token else item
            for item in vehicle.tasks
        ),
    )
    return replace(
        environment,
        vehicle_anchors=tuple(
            changed_vehicle if anchor.vehicle_id == vehicle.vehicle_id else anchor
            for anchor in environment.vehicle_anchors
        ),
    )


@pytest.mark.parametrize("tamper", ("aggregate", "task", "job", "reservation"))
def test_rsu_anchor_tampering_is_conservatively_incomplete(decision_fixture, tamper):
    fixture = decision_fixture(task_id=f"rsu-anchor-{tamper}")
    environment = next(
        row
        for row in fixture.simulator.scenario_library.environment_scenarios
        if any(anchor.active_task_count for anchor in row.rsu_anchors)
    )
    _, baseline_branch, baseline = _policy_and_scheduler(
        fixture.simulator, fixture.observation, environment
    )
    assert baseline is not None, baseline_branch.incomplete_reason

    changed = _tamper_active_rsu_anchor(environment, tamper)
    _, branch, scheduler = _policy_and_scheduler(
        fixture.simulator, fixture.observation, changed
    )

    assert scheduler is None
    assert branch.complete_macro_recourse is False
    assert branch.incomplete_reason == "SCENARIO_RSU_ANCHOR_AGGREGATE_MISMATCH"
