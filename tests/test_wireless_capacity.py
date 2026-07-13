from __future__ import annotations

import math
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from privacy_edge_sim.enums import TransferDirection, TransferStatus
from privacy_edge_sim.policies import (
    POLICY_REGISTRY,
    _BranchJob,
    _BranchTask,
    _BranchTransfer,
)
from privacy_edge_sim.traces import (
    ScenarioEnvironment,
    ScenarioThermalSegment,
    ScenarioWirelessSegment,
)


def _runtime_transfer(
    transfer_id: str,
    *,
    vehicle_id: str,
    rsu_id: str,
    direction: TransferDirection,
    status: TransferStatus = TransferStatus.ACTIVE,
) -> SimpleNamespace:
    return SimpleNamespace(
        transfer_id=transfer_id,
        vehicle_id=vehicle_id,
        rsu_id=rsu_id,
        direction=direction,
        status=status,
    )


def test_production_link_service_is_shared_without_capacity_or_power_duplication(
    decision_fixture,
):
    fixture = decision_fixture(task_id="shared-production-radio")
    simulator = fixture.simulator
    now_s = simulator.state.clock_s
    rsu_id = next(
        rsu
        for rsu, link in fixture.observation.links.items()
        if link["ul_link_state"] == "connected" and link["ul_goodput_bps"] > 0
    )
    first = _runtime_transfer(
        "share-1",
        vehicle_id=fixture.task.vehicle_id,
        rsu_id=rsu_id,
        direction=TransferDirection.UL,
    )
    second = _runtime_transfer(
        "share-2",
        vehicle_id=fixture.task.vehicle_id,
        rsu_id=rsu_id,
        direction=TransferDirection.UL,
    )
    simulator.state.transfers = {first.transfer_id: first, second.transfer_id: second}

    counts = simulator._active_link_counts(now_s)
    services = [
        simulator._transfer_service(transfer, now_s, counts)
        for transfer in (first, second)
    ]
    segment = simulator._wireless_segment(
        fixture.task.vehicle_id, rsu_id, TransferDirection.UL, now_s
    )

    assert segment is not None
    assert services[0][0] == pytest.approx(segment.goodput_bps / 2)
    assert services[1][0] == pytest.approx(segment.goodput_bps / 2)
    assert sum(row[0] for row in services) == pytest.approx(segment.goodput_bps)
    assert sum(row[1] for row in services) == pytest.approx(segment.transmitter_power_w)
    assert sum(row[2] for row in services) == pytest.approx(segment.receiver_power_w)


def test_production_paused_packets_share_link_level_outage_power(
    decision_fixture, monkeypatch
):
    fixture = decision_fixture(task_id="shared-production-outage-power")
    simulator = fixture.simulator
    segment = SimpleNamespace(
        link_state="temporary_outage",
        goodput_bps=0.0,
        transmitter_power_w=3.2,
        receiver_power_w=1.4,
    )
    monkeypatch.setattr(simulator, "_wireless_segment", lambda *args: segment)
    transfers = [
        _runtime_transfer(
            f"paused-{index}",
            vehicle_id=fixture.task.vehicle_id,
            rsu_id="rsu-1",
            direction=TransferDirection.UL,
            status=TransferStatus.PAUSED,
        )
        for index in range(2)
    ]
    simulator.state.transfers = {
        transfer.transfer_id: transfer for transfer in transfers
    }

    counts = simulator._active_link_counts(simulator.state.clock_s)
    services = [
        simulator._transfer_service(transfer, simulator.state.clock_s, counts)
        for transfer in transfers
    ]

    assert [row[0] for row in services] == [0.0, 0.0]
    assert sum(row[1] for row in services) == pytest.approx(3.2)
    assert sum(row[2] for row in services) == pytest.approx(1.4)


def _branch_scheduler(decision_fixture, environment: ScenarioEnvironment):
    fixture = decision_fixture(task_id="branch-radio")
    frozen_environment = fixture.simulator.scenario_library.environment_scenarios[0]
    environment = replace(
        environment,
        # These focused fixtures override only the exogenous process under
        # test.  Preserve the complete, t=0 RSU state frozen by the joint
        # scenario library instead of silently constructing an anchorless
        # private branch.
        rsu_anchors=frozen_environment.rsu_anchors,
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    branch = policy._new_branch(fixture.observation, environment)
    scheduler = policy._scheduler_new(branch, fixture.observation, environment)
    assert scheduler is not None
    return fixture, policy, scheduler


def _minimal_future(vehicle_id: str, *, prep_failed: bool = False):
    return SimpleNamespace(
        arrival_offset_s=0.0,
        relative_deadline_s=1.0,
        vehicle_id=vehicle_id,
        device_type="vehicle_gpu_class_a",
        context=None,
        quality_candidates=("clear",),
        quality_probabilities=(("clear", 1.0),),
        quality_features=(0.8, 0.2),
        ood=False,
        prep_work_s=0.05,
        prep_energy_j=0.5,
        prep_memory_bytes=1024,
        prep_failed=prep_failed,
        local_rows=(),
        anon_rows=(),
        edge_rows=(),
        complete_support=True,
    )


def _empty_branch_scheduler(scheduler) -> None:
    scheduler.events.clear()
    scheduler.jobs.clear()
    scheduler.transfers.clear()
    scheduler.tasks.clear()
    for pool in scheduler.resources.values():
        pool.running.clear()
        pool.waiting.clear()


def test_branch_completion_time_is_strictly_future_below_one_ulp(decision_fixture):
    environment = ScenarioEnvironment(
        scenario_id="strict-future-branch",
        cluster_token="strict-future-branch",
        duration_s=10.0,
        wireless=(),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    fixture, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    _empty_branch_scheduler(scheduler)
    now_s = 2.8596357755652373
    scheduler.branch.elapsed_s = now_s
    scheduler.vehicle_battery_j[fixture.task.vehicle_id] = 1e12
    job = _BranchJob(
        "sub-ulp-job",
        "sub-ulp-task",
        "vehicle",
        fixture.task.vehicle_id,
        "accelerator",
        1e-16,
        1e-16,
        0.0,
        10.0,
        1,
        "BACKGROUND_DONE",
    )
    scheduler.jobs[job.job_id] = job
    scheduler.resources[
        ("vehicle", fixture.task.vehicle_id, "accelerator")
    ].running.append(job.job_id)

    assert policy._scheduler_next_time(scheduler) == math.nextafter(now_s, math.inf)


def test_branch_compound_uses_finite_ulp_not_fixed_absolute_tolerance(
    decision_fixture,
):
    environment = ScenarioEnvironment(
        scenario_id="finite-ulp-branch",
        cluster_token="finite-ulp-branch",
        duration_s=1.0,
        wireless=(),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    _, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    _empty_branch_scheduler(scheduler)
    rounded_sum = 0.1 + 0.2
    policy._scheduler_push(scheduler, 0.3, 20, "DEADLINE", "deadline")
    policy._scheduler_push(scheduler, rounded_sum, 10, "COMPLETION", "completion")
    policy._scheduler_push(
        scheduler, rounded_sum + 1e-12, 30, "ARRIVAL", "physically-later"
    )

    compound = policy._scheduler_pop_same_instant_events(scheduler, rounded_sum)

    assert {row[3] for row in compound} == {"DEADLINE", "COMPLETION"}
    assert [row[4] for row in scheduler.events] == ["physically-later"]


def test_branch_thermal_selector_prefers_exact_resource_then_latest_start(
    decision_fixture,
):
    fixture = decision_fixture(task_id="branch-thermal-precedence")
    rows = (
        ScenarioThermalSegment(
            "vehicle", "veh-1", "all", 0.0, 2.0, "all-hot", 0.2, 3.0
        ),
        ScenarioThermalSegment(
            "vehicle", "veh-1", "accelerator", 0.0, 2.0, "exact-old", 0.6, 1.4
        ),
        ScenarioThermalSegment(
            "vehicle", "veh-1", "accelerator", 1.0, 2.0, "exact-new", 0.4, 1.8
        ),
    )
    environment = ScenarioEnvironment(
        scenario_id="thermal-precedence",
        cluster_token="thermal-precedence",
        duration_s=2.0,
        wireless=(),
        thermal=rows,
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    branch = policy._new_branch(fixture.observation, environment)

    branch.elapsed_s = 0.5
    assert policy._thermal_multiplier(
        branch, "vehicle", "veh-1", "accelerator"
    ) == pytest.approx(0.6)
    branch.elapsed_s = 1.5
    assert policy._thermal_multiplier(
        branch, "vehicle", "veh-1", "accelerator"
    ) == pytest.approx(0.4)
    segment = policy._branch_thermal_segment(branch, "vehicle", "veh-1", "accelerator")
    assert segment is not None
    assert segment.dynamic_power_multiplier == pytest.approx(1.8)


def test_branch_failed_preprocess_consumes_measured_work_and_energy(
    decision_fixture,
):
    environment = ScenarioEnvironment(
        scenario_id="failed-prep-cost",
        cluster_token="failed-prep-cost",
        duration_s=1.0,
        wireless=(),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    fixture, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    future = _minimal_future(fixture.task.vehicle_id, prep_failed=True)
    task = _BranchTask(
        "failed-prep",
        fixture.task.vehicle_id,
        0.0,
        1.0,
        future,
        fixture.observation,
    )
    scheduler.tasks[task.task_token] = task
    energy_before = scheduler.branch.vehicle_energy_j

    policy._scheduler_arrival(scheduler, task)
    assert task.state == "PREP_WAIT"
    policy._scheduler_dispatch(scheduler, random.Random(11))
    job_id = next(
        job_id
        for pool in scheduler.resources.values()
        for job_id in pool.running
        if scheduler.jobs[job_id].task_token == task.task_token
    )
    job = scheduler.jobs[job_id]
    assert policy._scheduler_advance(
        scheduler, scheduler.branch.elapsed_s + job.remaining_work_s
    )
    policy._scheduler_complete_job(scheduler, job_id, random.Random(12))
    policy._scheduler_project_task_queues(scheduler)

    assert task.state == "FAIL"
    assert (
        scheduler.branch.vehicle_energy_j - energy_before >= future.prep_energy_j - 1e-9
    )
    assert any(
        row["kind"] == "TASK_TERMINAL" and row["reason"] == "PUBLIC_PREPROCESS_FAILURE"
        for row in scheduler.event_trace
    )


def test_branch_compound_projects_arrival_and_completion_once(decision_fixture):
    environment = ScenarioEnvironment(
        scenario_id="compound-vq",
        cluster_token="compound-vq",
        duration_s=1.0,
        wireless=(),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    fixture, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    done = _BranchTask(
        "same-time-done",
        fixture.task.vehicle_id,
        0.0,
        1.0,
        _minimal_future(fixture.task.vehicle_id),
        fixture.observation,
        state="LOCAL_WAIT",
    )
    arrival = _BranchTask(
        "same-time-arrival",
        fixture.task.vehicle_id,
        0.0,
        1.0,
        _minimal_future(fixture.task.vehicle_id),
        fixture.observation,
    )
    scheduler.tasks[done.task_token] = done
    scheduler.tasks[arrival.task_token] = arrival
    coverage_before = float(scheduler.branch.virtual_queues["coverage"])

    policy._scheduler_finish(scheduler, done, success=True, reason="LOCAL_RESULT")
    policy._scheduler_arrival(scheduler, arrival)
    policy._scheduler_project_task_queues(scheduler)

    beta = fixture.simulator.config.long_term.coverage_rate_minimum
    assert scheduler.branch.virtual_queues["coverage"] == pytest.approx(
        max(0.0, coverage_before + beta - 1.0)
    )
    projections = [
        row
        for row in scheduler.event_trace
        if row["kind"] == "VIRTUAL_QUEUE_PROJECTION"
    ]
    assert len(projections) == 1
    assert projections[0]["arrivals"] == 1
    assert projections[0]["completed"] == 1


def test_branch_dispatch_uses_paired_remaining_energy_bound(decision_fixture):
    thermal = ScenarioThermalSegment(
        "vehicle", "veh-1", "accelerator", 0.2, 0.8, "hot", 0.25, 2.0
    )
    environment = ScenarioEnvironment(
        scenario_id="dispatch-energy-upper",
        cluster_token="dispatch-energy-upper",
        duration_s=1.0,
        wireless=(),
        thermal=(thermal,),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    fixture, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    task = _BranchTask(
        "dispatch-energy-task",
        fixture.task.vehicle_id,
        0.0,
        1.0,
        _minimal_future(fixture.task.vehicle_id),
        fixture.observation,
        state="PREP_WAIT",
    )
    scheduler.tasks[task.task_token] = task
    scheduler.sequence += 1
    job = _BranchJob(
        "dispatch-energy-job",
        task.task_token,
        "vehicle",
        task.vehicle_id,
        "accelerator",
        0.5,
        1.0,
        2.0,
        1.0,
        scheduler.sequence,
        "PREP_DONE",
    )
    scheduler.jobs[job.job_id] = job
    scheduler.resources[("vehicle", task.vehicle_id, "accelerator")].waiting.append(
        job.job_id
    )
    required = policy._scheduler_remaining_job_energy_upper(scheduler, job)
    assert required == pytest.approx(1.0)
    scheduler.vehicle_battery_j[task.vehicle_id] = required - 0.01

    policy._scheduler_dispatch(scheduler, random.Random(13))

    assert task.state == "FAIL"
    assert job.job_id not in scheduler.jobs
    assert any(row["kind"] == "DISPATCH_BATTERY_GUARD" for row in scheduler.event_trace)


def test_branch_link_service_is_shared_without_capacity_or_power_duplication(
    decision_fixture,
):
    fixture = decision_fixture(task_id="branch-radio-template")
    vehicle_id = fixture.task.vehicle_id
    rsu_id = "rsu-1"
    segment = ScenarioWirelessSegment(
        vehicle_id=vehicle_id,
        rsu_id=rsu_id,
        direction=TransferDirection.UL,
        start_offset_s=0.0,
        end_offset_s=5.0,
        goodput_bps=100.0,
        transmitter_power_w=4.0,
        receiver_power_w=2.0,
        link_state="connected",
    )
    environment = ScenarioEnvironment(
        scenario_id="shared-branch-radio",
        cluster_token="shared-branch-cluster",
        duration_s=5.0,
        wireless=(segment,),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    fixture, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    task = _BranchTask(
        "branch-task",
        fixture.task.vehicle_id,
        0.0,
        5.0,
        SimpleNamespace(),
        fixture.observation,
        state="UL",
    )
    scheduler.tasks[task.task_token] = task
    scheduler.transfers = {
        token: _BranchTransfer(
            token,
            task.task_token,
            task.vehicle_id,
            rsu_id,
            TransferDirection.UL,
            100.0,
        )
        for token in ("branch-share-1", "branch-share-2")
    }

    counts = policy._scheduler_active_link_counts(scheduler)
    services = [
        policy._scheduler_transfer_service(scheduler, transfer, counts)
        for transfer in scheduler.transfers.values()
    ]

    assert [row[0] for row in services] == pytest.approx([50.0, 50.0])
    assert sum(row[0] for row in services) == pytest.approx(100.0)
    assert sum(row[1] for row in services) == pytest.approx(4.0)
    assert sum(row[2] for row in services) == pytest.approx(2.0)


def test_branch_temporary_outage_fails_exactly_at_configured_pause_limit(
    decision_fixture,
):
    template = decision_fixture(task_id="branch-pause-template")
    vehicle_id = template.task.vehicle_id
    rsu_id = "rsu-1"
    outage = ScenarioWirelessSegment(
        vehicle_id=vehicle_id,
        rsu_id=rsu_id,
        direction=TransferDirection.DL,
        start_offset_s=0.0,
        end_offset_s=3.0,
        goodput_bps=0.0,
        transmitter_power_w=2.0,
        receiver_power_w=1.0,
        link_state="temporary_outage",
    )
    environment = ScenarioEnvironment(
        scenario_id="branch-pause-limit",
        cluster_token="branch-pause-cluster",
        duration_s=3.0,
        wireless=(outage,),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    fixture, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    task = _BranchTask(
        "paused-dl-task",
        fixture.task.vehicle_id,
        0.0,
        10.0,
        SimpleNamespace(),
        fixture.observation,
        state="DL",
    )
    scheduler.tasks[task.task_token] = task
    transfer = _BranchTransfer(
        "paused-dl",
        task.task_token,
        task.vehicle_id,
        rsu_id,
        TransferDirection.DL,
        100.0,
    )
    scheduler.transfers[transfer.transfer_id] = transfer

    policy._scheduler_run(scheduler, random.Random(7), stop_before_next_macro=True)

    assert scheduler.branch.elapsed_s == pytest.approx(
        fixture.simulator.config.downlink_pause_limit_s
    )
    assert task.state == "FAIL"
    assert not scheduler.transfers
    assert transfer.remaining_bits == pytest.approx(100.0)
    assert scheduler.branch.vehicle_energy_j > 0
    assert scheduler.branch.rsu_energy_j[rsu_id] > 0


def test_branch_paused_packets_share_link_level_outage_power(decision_fixture):
    template = decision_fixture(task_id="branch-pause-power-template")
    vehicle_id = template.task.vehicle_id
    rsu_id = "rsu-1"
    outage = ScenarioWirelessSegment(
        vehicle_id=vehicle_id,
        rsu_id=rsu_id,
        direction=TransferDirection.UL,
        start_offset_s=0.0,
        end_offset_s=3.0,
        goodput_bps=0.0,
        transmitter_power_w=3.2,
        receiver_power_w=1.4,
        link_state="temporary_outage",
    )
    environment = ScenarioEnvironment(
        scenario_id="branch-pause-power",
        cluster_token="branch-pause-power-cluster",
        duration_s=3.0,
        wireless=(outage,),
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    fixture, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    task = _BranchTask(
        "paused-ul-task",
        fixture.task.vehicle_id,
        0.0,
        10.0,
        SimpleNamespace(),
        fixture.observation,
        state="UL",
    )
    scheduler.tasks[task.task_token] = task
    scheduler.transfers = {
        token: _BranchTransfer(
            token,
            task.task_token,
            task.vehicle_id,
            rsu_id,
            TransferDirection.UL,
            100.0,
        )
        for token in ("paused-branch-1", "paused-branch-2")
    }

    counts = policy._scheduler_active_link_counts(scheduler)
    services = [
        policy._scheduler_transfer_service(scheduler, transfer, counts)
        for transfer in scheduler.transfers.values()
    ]

    assert [row[0] for row in services] == [0.0, 0.0]
    assert sum(row[1] for row in services) == pytest.approx(3.2)
    assert sum(row[2] for row in services) == pytest.approx(1.4)


def test_branch_uncovered_wireless_gap_discards_partial_packet(decision_fixture):
    template = decision_fixture(task_id="branch-wireless-gap-template")
    vehicle_id = template.task.vehicle_id
    rsu_id = "rsu-1"
    segments = tuple(
        ScenarioWirelessSegment(
            vehicle_id=vehicle_id,
            rsu_id=rsu_id,
            direction=TransferDirection.DL,
            start_offset_s=start,
            end_offset_s=end,
            goodput_bps=100.0,
            transmitter_power_w=2.0,
            receiver_power_w=1.0,
            link_state="connected",
        )
        for start, end in ((0.0, 0.1), (0.5, 1.0))
    )
    environment = ScenarioEnvironment(
        scenario_id="branch-wireless-gap",
        cluster_token="branch-wireless-gap-cluster",
        duration_s=1.0,
        wireless=segments,
        thermal=(),
        faults=(),
        background_loads=(),
        telemetry=(),
    )
    fixture, policy, scheduler = _branch_scheduler(decision_fixture, environment)
    task = _BranchTask(
        "gap-dl-task",
        fixture.task.vehicle_id,
        0.0,
        1.0,
        SimpleNamespace(),
        fixture.observation,
        state="DL",
    )
    scheduler.tasks[task.task_token] = task
    transfer = _BranchTransfer(
        "gap-dl",
        task.task_token,
        task.vehicle_id,
        rsu_id,
        TransferDirection.DL,
        20.0,
    )
    scheduler.transfers[transfer.transfer_id] = transfer

    policy._scheduler_run(scheduler, random.Random(13), stop_before_next_macro=True)

    assert scheduler.branch.elapsed_s == pytest.approx(0.1)
    assert task.state == "FAIL"
    assert transfer.remaining_bits == pytest.approx(10.0)
    assert not scheduler.transfers
