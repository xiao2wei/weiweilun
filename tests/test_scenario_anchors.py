from __future__ import annotations

from dataclasses import replace

import pytest

from privacy_edge_sim.traces import (
    DeviceContext,
    ExogenousEvent,
    ScenarioLibrary,
    load_trace,
)
from privacy_edge_sim.enums import TransferDirection


def _anchor_parameters(config, profile, *, descriptor_capacity: int | None = None):
    vehicles = {
        row.vehicle_id: {
            "device_type": row.device_type,
            "initial_battery_j": row.initial_battery_j,
            "battery_capacity_j": row.battery_capacity_j,
            "memory_capacity_bytes": row.memory_capacity_bytes,
            "idle_power_w": row.idle_power_w,
            "hold_power_w": row.hold_power_w,
            "descriptor_capacity": {
                "accelerator": row.accelerator_descriptors,
                "cpu": row.cpu_descriptors,
                "encoder": row.encoder_descriptors,
            },
            "server_count": {"accelerator": 1, "cpu": 1, "encoder": 1},
        }
        for row in config.vehicles
    }
    rsus = {
        row.rsu_id: {
            "descriptor_capacity": (
                descriptor_capacity
                if descriptor_capacity is not None and row.rsu_id == "rsu-1"
                else row.descriptor_capacity
            ),
            "vram_capacity_bytes": row.vram_capacity_bytes,
            "workload_capacity_gpu_s": row.workload_capacity_gpu_s,
            "ingress_servers": 1,
            "gpu_servers": row.gpu_servers,
            "idle_power_w": row.idle_power_w,
            "hold_power_w": row.hold_power_w,
            "cached_models": {
                model_id: profile.edge_models[model_id].model_hash
                for model_id in row.cached_models
            },
        }
        for row in config.rsus
    }
    return vehicles, rsus


def _contended_trace(config, profile):
    source = load_trace(config.scenario_trace_path, profile)
    arrivals = tuple(
        replace(
            row,
            vehicle_id="veh-1",
            arrival_time_s=0.10,
            relative_deadline_s=5.0,
        )
        for row in (source.arrivals[0], source.arrivals[2])
    )
    edge_rows = tuple(
        replace(
            row,
            rsu_id="rsu-1",
            ingress_work_s=0.60,
            ingress_energy_j=5.0,
            gpu_work_s=0.60,
            gpu_energy_j=9.0,
            vram_bytes=134_217_728,
            failed=False,
            fer_loss=0.2,
        )
        for row in source.edge_rows
    )
    wireless = tuple(
        replace(row, goodput_bps=100_000_000.0, link_state="connected")
        for row in source.wireless
    )
    probes = tuple(
        ExogenousEvent(
            event_id=f"anchor-probe-{index}",
            time_s=time_s,
            event_type="LINK_CHANGE",
            target_type="rsu",
            target_id="rsu-1",
            resource="radio",
            old_version=None,
            new_version=None,
            permanent=False,
            details={},
        )
        for index, time_s in enumerate((0.35, 0.55, 1.20, 3.00), start=1)
    )
    return replace(
        source,
        arrivals=arrivals,
        edge_rows=edge_rows,
        wireless=wireless,
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, *probes),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )


def _library(config, profile, *, descriptor_capacity: int | None = None):
    vehicles, rsus = _anchor_parameters(
        config, profile, descriptor_capacity=descriptor_capacity
    )
    return ScenarioLibrary.from_trace(
        _contended_trace(config, profile),
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )


def _environment_at(library, elapsed_s: float):
    return next(
        environment
        for environment in library.environment_scenarios
        if environment.duration_s == pytest.approx(8.0 - elapsed_s)
    )


def test_joint_rsu_anchor_preserves_ingress_edf_contention_and_pinned_rows(
    config, profile
):
    environment = _environment_at(_library(config, profile), 0.55)
    rsu = next(anchor for anchor in environment.rsu_anchors if anchor.rsu_id == "rsu-1")
    active_tasks = tuple(
        task
        for vehicle in environment.vehicle_anchors
        for task in vehicle.tasks
        if task.rsu_id == "rsu-1"
    )

    assert rsu.complete_support
    assert rsu.active_task_count == rsu.descriptors_reserved == 2
    assert rsu.vram_reserved_bytes == 2 * 134_217_728
    assert rsu.workload_reserved_gpu_s == pytest.approx(1.20)
    assert rsu.resources["ingress"]["server_count"] == 1
    assert rsu.resources["ingress"]["running_count"] == 1
    assert rsu.resources["ingress"]["waiting_count"] == 1
    assert len(active_tasks) == 2
    assert {task.state for task in active_tasks} == {"RSU_INGRESS"}
    assert all(
        task.model_hash == next(iter(rsu.cached_models.values()))
        for task in active_tasks
    )
    assert all(task.fallback_local_rows for task in active_tasks)


def test_joint_rsu_atomic_capacity_rejection_has_no_partial_side_effect(
    config, profile
):
    environment = _environment_at(
        _library(config, profile, descriptor_capacity=1), 0.35
    )
    rsu = next(anchor for anchor in environment.rsu_anchors if anchor.rsu_id == "rsu-1")
    active_tasks = tuple(
        task
        for vehicle in environment.vehicle_anchors
        for task in vehicle.tasks
        if task.rsu_id == "rsu-1"
    )

    assert rsu.descriptor_capacity == 1
    assert rsu.descriptors_reserved == rsu.active_task_count == 1
    assert rsu.vram_reserved_bytes == 134_217_728
    assert rsu.workload_reserved_gpu_s == pytest.approx(0.60)
    assert len(active_tasks) == 1
    assert rsu.resources["ingress"]["running_count"] == 1
    assert rsu.resources["ingress"]["waiting_count"] == 0


def test_joint_rsu_anchor_midstage_energy_and_terminal_release(config, profile):
    library = _library(config, profile)
    mid = _environment_at(library, 1.20)
    late = _environment_at(library, 3.00)
    mid_rsu = next(anchor for anchor in mid.rsu_anchors if anchor.rsu_id == "rsu-1")
    late_rsu = next(anchor for anchor in late.rsu_anchors if anchor.rsu_id == "rsu-1")
    mid_states = {
        task.state
        for vehicle in mid.vehicle_anchors
        for task in vehicle.tasks
        if task.rsu_id == "rsu-1"
    }

    assert mid_states.intersection({"RSU_INGRESS", "RSU_GPU"})
    assert mid_rsu.physical_energy_j > 0
    assert any(
        float(row["remaining_dynamic_energy_j"]) > 0
        for row in mid_rsu.resources.values()
    )
    assert late_rsu.physical_energy_j > mid_rsu.physical_energy_j
    assert late_rsu.descriptors_reserved == 0
    assert late_rsu.vram_reserved_bytes == 0
    assert late_rsu.workload_reserved_gpu_s == pytest.approx(0.0)
    assert late_rsu.active_task_count == 0
    assert all(
        row["running_count"] == row["waiting_count"] == 0
        for row in late_rsu.resources.values()
    )


def test_joint_anchor_paired_energy_is_not_rescaled_by_dynamic_power_multiplier(
    config, profile
):
    source = _contended_trace(config, profile)
    vehicles, rsus = _anchor_parameters(config, profile)

    def library_with_multiplier(multiplier: float):
        trace = replace(
            source,
            thermal=tuple(
                replace(row, dynamic_power_multiplier=multiplier)
                for row in source.thermal
            ),
        )
        return ScenarioLibrary.from_trace(
            trace,
            metadata_bits=config.metadata_bits,
            vehicle_anchor_parameters=vehicles,
            rsu_anchor_parameters=rsus,
        )

    low = _environment_at(library_with_multiplier(0.2), 1.20)
    high = _environment_at(library_with_multiplier(2.0), 1.20)
    low_vehicles = {row.vehicle_id: row for row in low.vehicle_anchors}
    high_vehicles = {row.vehicle_id: row for row in high.vehicle_anchors}
    low_rsus = {row.rsu_id: row for row in low.rsu_anchors}
    high_rsus = {row.rsu_id: row for row in high.rsu_anchors}

    assert set(low_vehicles) == set(high_vehicles)
    assert set(low_rsus) == set(high_rsus)
    for vehicle_id in low_vehicles:
        assert low_vehicles[vehicle_id].battery_j == pytest.approx(
            high_vehicles[vehicle_id].battery_j
        )
        assert low_vehicles[vehicle_id].physical_energy_j == pytest.approx(
            high_vehicles[vehicle_id].physical_energy_j
        )
    for rsu_id in low_rsus:
        assert low_rsus[rsu_id].physical_energy_j == pytest.approx(
            high_rsus[rsu_id].physical_energy_j
        )


def test_scenario_library_contains_identity_free_prep_rows(config, profile):
    library = _library(config, profile)

    assert library.prep_rows
    assert all(row.service_work_s > 0 for row in library.prep_rows)
    assert all(row.dynamic_energy_j >= 0 for row in library.prep_rows)
    assert not any(
        hasattr(row, field_name)
        for row in library.prep_rows
        for field_name in ("row_id", "fixture_key", "true_identity", "true_label")
    )


def test_joint_anchor_pause_expiry_preserves_partial_cost_then_releases_resources(
    config, profile
):
    source = _contended_trace(config, profile)
    first_ul = next(
        row
        for row in source.wireless
        if row.vehicle_id == "veh-1"
        and row.rsu_id == "rsu-1"
        and row.direction.value == "UL"
        and row.start_time_s == 0.0
    )
    split = (
        replace(
            first_ul,
            segment_id=f"{first_ul.segment_id}-partial",
            end_time_s=0.22,
            goodput_bps=1_000_000.0,
            link_state="connected",
        ),
        replace(
            first_ul,
            segment_id=f"{first_ul.segment_id}-outage",
            start_time_s=0.22,
            goodput_bps=0.0,
            link_state="temporary_outage",
        ),
    )
    probes = tuple(
        ExogenousEvent(
            event_id=f"pause-probe-{index}",
            time_s=time_s,
            event_type="LINK_CHANGE",
            target_type="rsu",
            target_id="rsu-1",
            resource="radio",
            old_version=None,
            new_version=None,
            permanent=False,
            details={},
        )
        for index, time_s in enumerate((0.25, 0.40), start=1)
    )
    trace = replace(
        source,
        arrivals=(source.arrivals[0],),
        local_rows=tuple(
            replace(row, service_work_s=0.50, dynamic_energy_j=4.0)
            for row in source.local_rows
        ),
        wireless=tuple(
            sorted(
                (
                    *(
                        row
                        for row in source.wireless
                        if row.segment_id != first_ul.segment_id
                    ),
                    *split,
                ),
                key=lambda row: (row.start_time_s, row.segment_id),
            )
        ),
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, *probes),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    library = ScenarioLibrary.from_trace(
        trace,
        uplink_pause_limit_s=0.10,
        downlink_pause_limit_s=0.10,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    partial = _environment_at(library, 0.25)
    expired = _environment_at(library, 0.40)
    partial_transfer = next(
        transfer
        for vehicle in partial.vehicle_anchors
        for transfer in vehicle.transfers
    )
    expired_rsu = next(
        anchor for anchor in expired.rsu_anchors if anchor.rsu_id == "rsu-1"
    )

    assert 0 < partial_transfer.remaining_bits < partial_transfer.total_bits
    assert partial_transfer.status == "temporary_outage"
    assert partial_transfer.pause_age_s == pytest.approx(0.03)
    assert not any(vehicle.transfers for vehicle in expired.vehicle_anchors)
    assert expired_rsu.descriptors_reserved == 0
    assert expired_rsu.vram_reserved_bytes == 0
    assert expired_rsu.workload_reserved_gpu_s == pytest.approx(0.0)


def test_same_timestamp_scenario_arrivals_are_retained_as_offset_zero_tasks(
    config, profile
):
    environment = _environment_at(_library(config, profile), 0.10)
    arrivals = tuple(
        task for task in environment.future_tasks if task.arrival_offset_s == 0.0
    )

    assert len(arrivals) == 2
    assert len({task.task_token for task in arrivals}) == 2


def test_anchor_left_limit_does_not_merge_arrival_one_picosecond_later(config, profile):
    source = _contended_trace(config, profile)
    first_time_s = 0.1
    second_time_s = first_time_s + 1e-12
    probe_time_s = first_time_s + 0.5e-12
    first, second = source.arrivals
    probe = ExogenousEvent(
        event_id="finite-ulp-left-limit-probe",
        time_s=probe_time_s,
        event_type="LINK_CHANGE",
        target_type="vehicle",
        target_id=first.vehicle_id,
        resource="radio",
        old_version=None,
        new_version=None,
        permanent=False,
        details={},
    )
    trace = replace(
        source,
        arrivals=(
            replace(first, arrival_time_s=first_time_s),
            replace(second, arrival_time_s=second_time_s),
        ),
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, probe),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    library = ScenarioLibrary.from_trace(
        trace,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    target_duration_s = trace.horizon_end_s - probe_time_s
    environment = min(
        library.environment_scenarios,
        key=lambda row: abs(row.duration_s - target_duration_s),
    )
    vehicle = next(
        row for row in environment.vehicle_anchors if row.vehicle_id == first.vehicle_id
    )

    assert environment.duration_s == target_duration_s
    assert len(vehicle.tasks) == 1


def test_joint_anchor_carries_permanent_vehicle_fault_and_releases_rsu(config, profile):
    source = _contended_trace(config, profile)
    events = (
        ExogenousEvent(
            event_id="vehicle-permanent-anchor-fault",
            time_s=0.30,
            event_type="DEVICE_FAULT_PERMANENT",
            target_type="vehicle",
            target_id="veh-1",
            resource="all",
            old_version=None,
            new_version=None,
            permanent=True,
            details={},
        ),
        ExogenousEvent(
            event_id="vehicle-permanent-anchor-probe",
            time_s=0.31,
            event_type="LINK_CHANGE",
            target_type="vehicle",
            target_id="veh-1",
            resource="radio",
            old_version=None,
            new_version=None,
            permanent=False,
            details={},
        ),
    )
    trace = replace(
        source,
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, *events),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    library = ScenarioLibrary.from_trace(
        trace,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    environment = _environment_at(library, 0.31)
    vehicle = next(
        anchor for anchor in environment.vehicle_anchors if anchor.vehicle_id == "veh-1"
    )
    rsu = next(anchor for anchor in environment.rsu_anchors if anchor.rsu_id == "rsu-1")

    assert vehicle.failed and vehicle.permanent_failure
    assert not vehicle.tasks
    assert vehicle.memory_reserved_bytes == 0
    assert sum(vehicle.descriptors_reserved.values()) == 0
    assert rsu.descriptors_reserved == 0
    assert rsu.vram_reserved_bytes == 0
    assert rsu.workload_reserved_gpu_s == pytest.approx(0.0)


def test_joint_anchor_replays_prep_controller_and_pipeline_resources(config, profile):
    source = load_trace(config.scenario_trace_path, profile)
    probes = tuple(
        ExogenousEvent(
            event_id=f"stage-probe-{index}",
            time_s=time_s,
            event_type="LINK_CHANGE",
            target_type="vehicle",
            target_id="veh-1",
            resource="radio",
            old_version=None,
            new_version=None,
            permanent=False,
            details={},
        )
        for index, time_s in enumerate(
            (0.105, *(round(0.14 + 0.04 * index, 3) for index in range(35))),
            start=1,
        )
    )
    trace = replace(
        source,
        arrivals=(replace(source.arrivals[0], relative_deadline_s=5.0),),
        prep_rows=tuple(
            replace(row, service_work_s=0.04, dynamic_energy_j=0.8)
            for row in source.prep_rows
        ),
        anon_rows=tuple(
            replace(
                row,
                attempts=tuple(
                    replace(
                        attempt,
                        anon_work_s=0.04,
                        guard_work_s=(None if attempt.guard_work_s is None else 0.04),
                        encode_work_s=(None if attempt.encode_work_s is None else 0.04),
                    )
                    for attempt in row.attempts
                ),
            )
            for row in source.anon_rows
        ),
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, *probes),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    vehicles["veh-1"]["controller_overhead_s"] = 0.03
    vehicles["veh-1"]["controller_energy_j"] = 0.02
    library = ScenarioLibrary.from_trace(
        trace,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    anchored_tasks = tuple(
        task
        for environment in library.environment_scenarios
        for vehicle in environment.vehicle_anchors
        if vehicle.vehicle_id == "veh-1"
        for task in vehicle.tasks
    )
    states = {task.state for task in anchored_tasks}
    compute_resources = {
        task.resource for task in anchored_tasks if task.state == "COMPUTE"
    }
    first = _environment_at(library, 0.105)
    first_vehicle = next(
        anchor for anchor in first.vehicle_anchors if anchor.vehicle_id == "veh-1"
    )

    assert "PREP" in states
    assert states.intersection({"RAW_CONTROL", "READY_CONTROL"})
    assert {"accelerator", "cpu", "encoder"}.issubset(compute_resources)
    compute_tasks = tuple(task for task in anchored_tasks if task.state == "COMPUTE")
    assert any(task.remaining_vehicle_stages for task in compute_tasks)
    assert all(
        stage.resource in {"accelerator", "cpu", "encoder"}
        and stage.work_s > 0
        and stage.energy_j >= 0
        for task in compute_tasks
        for stage in task.remaining_vehicle_stages
    )
    assert any(task.controller_remaining_s > 0 for task in anchored_tasks)
    assert first_vehicle.physical_energy_j > 0
    assert first_vehicle.battery_j < vehicles["veh-1"]["initial_battery_j"]


def test_background_uplink_bits_include_protocol_metadata(config, profile):
    trace = load_trace(config.scenario_trace_path, profile)
    vehicles, rsus = _anchor_parameters(config, profile)
    without_metadata = ScenarioLibrary.from_trace(
        trace,
        metadata_bits=0,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    with_metadata = ScenarioLibrary.from_trace(
        trace,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    zero_loads = without_metadata.environment_scenarios[0].background_loads
    metadata_loads = with_metadata.environment_scenarios[0].background_loads
    paired = tuple(
        (zero, metadata)
        for zero, metadata in zip(zero_loads, metadata_loads, strict=True)
        if zero.path_kind == "edge" and zero.complete_support
    )

    assert paired
    assert all(
        metadata.uplink_bits - zero.uplink_bits == config.metadata_bits
        for zero, metadata in paired
    )
    for _, load in paired:
        anon = next(
            row
            for row in with_metadata.anon_rows
            if row.artifact_token == load.artifact_token
        )
        assert load.uplink_bits == (
            anon.final_encoded_size_bytes * 8 + config.metadata_bits
        )


def test_infinite_pause_limit_does_not_create_nonfinite_expiry(config, profile):
    """The default unlimited pause policy has no synthetic expiry event."""

    trace = load_trace(config.scenario_trace_path, profile)
    vehicles, rsus = _anchor_parameters(config, profile)

    library = ScenarioLibrary.from_trace(
        trace,
        uplink_pause_limit_s=None,
        downlink_pause_limit_s=None,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )

    assert library.environment_scenarios


def test_controller_completion_at_anchor_boundary_starts_action(config, profile):
    source = load_trace(config.scenario_trace_path, profile)
    arrival = replace(
        source.arrivals[0],
        arrival_time_s=0.10,
        relative_deadline_s=5.0,
    )
    probe = ExogenousEvent(
        event_id="controller-completion-boundary",
        time_s=0.17,
        event_type="LINK_CHANGE",
        target_type="vehicle",
        target_id=arrival.vehicle_id,
        resource="radio",
        old_version=None,
        new_version=None,
        permanent=False,
        details={},
    )
    trace = replace(
        source,
        arrivals=(arrival,),
        prep_rows=tuple(replace(row, service_work_s=0.04) for row in source.prep_rows),
        anon_rows=tuple(
            replace(
                row,
                attempts=tuple(
                    replace(
                        attempt,
                        anon_work_s=0.20,
                        guard_work_s=(None if attempt.guard_work_s is None else 0.20),
                        encode_work_s=(None if attempt.encode_work_s is None else 0.20),
                    )
                    for attempt in row.attempts
                ),
            )
            for row in source.anon_rows
        ),
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, probe),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    vehicles[arrival.vehicle_id]["controller_overhead_s"] = 0.03
    vehicles[arrival.vehicle_id]["controller_energy_j"] = 0.02
    library = ScenarioLibrary.from_trace(
        trace,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    environment = _environment_at(library, 0.17)
    vehicle = next(
        row
        for row in environment.vehicle_anchors
        if row.vehicle_id == arrival.vehicle_id
    )
    task = next(iter(vehicle.tasks))

    assert task.state == "COMPUTE"
    assert task.controller_remaining_s == pytest.approx(0.0)
    assert task.controller_next is None
    assert task.memory_reserved_bytes == task.action_memory_bytes
    assert task.action_descriptor_tokens
    assert vehicle.resources[task.resource]["waiting_count"] == 1

    # A controller-overhead completion is a DISPATCH_DECISION event.  At the
    # exact absolute deadline the task is terminated before that proposal may
    # create a reservation or compute job.
    deadline_trace = replace(
        trace,
        arrivals=(replace(arrival, relative_deadline_s=0.07),),
    )
    deadline_library = ScenarioLibrary.from_trace(
        deadline_trace,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    deadline_environment = _environment_at(deadline_library, 0.17)
    deadline_vehicle = next(
        row
        for row in deadline_environment.vehicle_anchors
        if row.vehicle_id == arrival.vehicle_id
    )
    assert not deadline_vehicle.tasks
    assert deadline_vehicle.memory_reserved_bytes == 0
    assert sum(deadline_vehicle.descriptors_reserved.values()) == 0


def test_edge_shadow_memory_covers_frozen_local_fallback(config, profile):
    source = load_trace(config.scenario_trace_path, profile)
    fallback_memory = 310_000_000
    arrival = replace(
        source.arrivals[0],
        arrival_time_s=0.10,
        relative_deadline_s=5.0,
    )
    probe = ExogenousEvent(
        event_id="shadow-memory-anchor",
        time_s=0.30,
        event_type="LINK_CHANGE",
        target_type="vehicle",
        target_id=arrival.vehicle_id,
        resource="radio",
        old_version=None,
        new_version=None,
        permanent=False,
        details={},
    )
    trace = replace(
        source,
        arrivals=(arrival,),
        prep_rows=tuple(replace(row, service_work_s=0.01) for row in source.prep_rows),
        anon_rows=tuple(
            replace(
                row,
                attempts=tuple(
                    replace(
                        attempt,
                        anon_work_s=0.005,
                        guard_work_s=(None if attempt.guard_work_s is None else 0.005),
                        encode_work_s=(
                            None if attempt.encode_work_s is None else 0.005
                        ),
                        peak_memory_bytes=64_000_000,
                    )
                    for attempt in row.attempts
                ),
            )
            for row in source.anon_rows
        ),
        local_rows=tuple(
            replace(row, memory_bytes=fallback_memory, service_work_s=1.0)
            for row in source.local_rows
        ),
        wireless=tuple(
            replace(row, goodput_bps=100_000_000.0, link_state="connected")
            for row in source.wireless
        ),
        edge_rows=tuple(
            replace(row, ingress_work_s=1.0, gpu_work_s=1.0) for row in source.edge_rows
        ),
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, probe),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    library = ScenarioLibrary.from_trace(
        trace,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    load = library.environment_scenarios[0].background_loads[0]
    environment = _environment_at(library, 0.30)
    task = next(
        task
        for vehicle in environment.vehicle_anchors
        for task in vehicle.tasks
        if task.path_kind == "edge"
    )

    assert load.complete_support
    assert max(row.memory_bytes for row in load.fallback_local_rows) == fallback_memory
    assert load.vehicle_memory_bytes == fallback_memory
    assert task.action_memory_bytes == fallback_memory
    assert task.memory_reserved_bytes == fallback_memory


def _multi_quality_admission_trace(config, profile):
    source = load_trace(config.scenario_trace_path, profile)
    arrival = replace(
        source.arrivals[1],
        arrival_time_s=0.10,
        relative_deadline_s=5.0,
    )
    probe = ExogenousEvent(
        event_id="multi-quality-admission-anchor",
        time_s=0.25,
        event_type="LINK_CHANGE",
        target_type="rsu",
        target_id="rsu-2",
        resource="radio",
        old_version=None,
        new_version=None,
        permanent=False,
        details={},
    )
    return replace(
        source,
        arrivals=(arrival,),
        prep_rows=tuple(replace(row, service_work_s=0.01) for row in source.prep_rows),
        anon_rows=tuple(
            replace(
                row,
                attempts=tuple(
                    replace(
                        attempt,
                        anon_work_s=0.005,
                        guard_work_s=(None if attempt.guard_work_s is None else 0.005),
                        encode_work_s=(
                            None if attempt.encode_work_s is None else 0.005
                        ),
                    )
                    for attempt in row.attempts
                ),
            )
            for row in source.anon_rows
        ),
        edge_rows=tuple(
            replace(
                row,
                ingress_work_s=0.50,
                ingress_energy_j=5.0,
                gpu_work_s=0.20 if row.quality_bin == "clear" else 0.80,
                gpu_energy_j=2.0 if row.quality_bin == "clear" else 8.0,
                vram_bytes=(100_000_000 if row.quality_bin == "clear" else 300_000_000),
                failed=False,
            )
            for row in source.edge_rows
        ),
        wireless=tuple(
            replace(row, goodput_bps=100_000_000.0, link_state="connected")
            for row in source.wireless
        ),
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, probe),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )


def test_multi_quality_admission_uses_upper_bound_but_executes_realized_row(
    config, profile
):
    trace = _multi_quality_admission_trace(config, profile)
    vehicles, rsus = _anchor_parameters(config, profile)
    library = ScenarioLibrary.from_trace(
        trace,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    load = library.environment_scenarios[0].background_loads[0]
    environment = _environment_at(library, 0.25)
    task = next(
        task
        for vehicle in environment.vehicle_anchors
        for task in vehicle.tasks
        if task.path_kind == "edge"
    )
    rsu = next(row for row in environment.rsu_anchors if row.rsu_id == task.rsu_id)

    assert load.complete_support
    assert {row.quality_bin for row in load.edge_rows} == {"clear", "challenging"}
    assert load.realized_quality_bin == task.realized_quality_bin == "clear"
    assert task.vram_bytes == 100_000_000
    assert task.gpu_total_work_s == pytest.approx(0.20)
    assert task.gpu_total_energy_j == pytest.approx(2.0)
    assert task.admission_vram_upper_bytes == 300_000_000
    assert task.admission_gpu_work_upper_s == pytest.approx(0.80)
    assert rsu.vram_reserved_bytes == 300_000_000
    assert rsu.workload_reserved_gpu_s == pytest.approx(0.80)


def test_missing_quality_candidate_edge_pair_marks_background_incomplete(
    config, profile
):
    trace = _multi_quality_admission_trace(config, profile)
    trace = replace(
        trace,
        edge_rows=tuple(
            row for row in trace.edge_rows if row.quality_bin != "challenging"
        ),
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    library = ScenarioLibrary.from_trace(
        trace,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    load = library.environment_scenarios[0].background_loads[0]
    future = library.environment_scenarios[0].future_tasks[0]

    assert not load.complete_support
    assert load.support_reason == "quality_candidate_edge_pair_missing"
    assert not future.complete_support
    assert "edge:challenging" in (future.support_reason or "")


@pytest.mark.parametrize("mismatch", ("device_type", "quality_bin", "context"))
def test_background_joint_pairing_rejects_mismatched_dimensions(
    config, profile, mismatch
):
    source = load_trace(config.scenario_trace_path, profile)
    if mismatch == "device_type":
        anon_rows = tuple(
            replace(row, device_type="unpaired-device") for row in source.anon_rows
        )
    elif mismatch == "quality_bin":
        anon_rows = tuple(
            replace(row, quality_bin="unpaired-quality") for row in source.anon_rows
        )
    else:
        unpaired = DeviceContext(
            thermal_state="unpaired",
            power_mode="unpaired",
            memory_pressure="unpaired",
        )
        anon_rows = tuple(replace(row, context=unpaired) for row in source.anon_rows)
    trace = replace(
        source,
        arrivals=(replace(source.arrivals[0], relative_deadline_s=5.0),),
        anon_rows=anon_rows,
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    library = ScenarioLibrary.from_trace(
        trace,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    load = library.environment_scenarios[0].background_loads[0]

    assert not load.complete_support
    assert load.support_reason == "joint_edge_pair_missing"
    assert not load.edge_rows


def test_same_time_transfer_completion_precedes_link_boundary_and_deadline(
    config, profile
):
    source = load_trace(config.scenario_trace_path, profile)
    arrival = replace(
        source.arrivals[0],
        arrival_time_s=0.10,
        relative_deadline_s=5.0,
    )
    calibration_probe = ExogenousEvent(
        event_id="same-time-order-calibration",
        time_s=0.50,
        event_type="LINK_CHANGE",
        target_type="vehicle",
        target_id=arrival.vehicle_id,
        resource="radio",
        old_version=None,
        new_version=None,
        permanent=False,
        details={},
    )
    base = replace(
        source,
        arrivals=(arrival,),
        prep_rows=tuple(replace(row, service_work_s=0.005) for row in source.prep_rows),
        anon_rows=tuple(
            replace(
                row,
                attempts=tuple(
                    replace(
                        attempt,
                        anon_work_s=0.002,
                        guard_work_s=(None if attempt.guard_work_s is None else 0.002),
                        encode_work_s=(
                            None if attempt.encode_work_s is None else 0.002
                        ),
                    )
                    for attempt in row.attempts
                ),
            )
            for row in source.anon_rows
        ),
        local_rows=tuple(replace(row, service_work_s=1.0) for row in source.local_rows),
        edge_rows=tuple(
            replace(
                row,
                ingress_work_s=0.01,
                gpu_work_s=0.01,
                result_size_bits=10_000,
                failed=False,
            )
            for row in source.edge_rows
        ),
        wireless=tuple(
            replace(
                row,
                goodput_bps=(
                    1_000.0 if row.direction is TransferDirection.DL else 100_000_000.0
                ),
                link_state="connected",
            )
            for row in source.wireless
        ),
        exogenous_events=tuple(
            sorted(
                (*source.exogenous_events, calibration_probe),
                key=lambda row: (row.time_s, row.event_id),
            )
        ),
    )
    vehicles, rsus = _anchor_parameters(config, profile)
    calibration = ScenarioLibrary.from_trace(
        base,
        metadata_bits=config.metadata_bits,
        vehicle_anchor_parameters=vehicles,
        rsu_anchor_parameters=rsus,
    )
    calibration_environment = _environment_at(calibration, 0.50)
    calibration_transfer = next(
        transfer
        for vehicle in calibration_environment.vehicle_anchors
        for transfer in vehicle.transfers
        if transfer.direction is TransferDirection.DL
    )
    dl_start_s = (
        0.50
        - (calibration_transfer.total_bits - calibration_transfer.remaining_bits)
        / 1_000.0
    )
    service_duration_s = 0.20
    boundary_s = dl_start_s + service_duration_s
    template = next(
        row
        for row in base.wireless
        if row.vehicle_id == arrival.vehicle_id
        and row.rsu_id == calibration_transfer.rsu_id
        and row.direction is TransferDirection.DL
    )
    other_wireless = tuple(
        row
        for row in base.wireless
        if not (
            row.vehicle_id == arrival.vehicle_id
            and row.rsu_id == calibration_transfer.rsu_id
            and row.direction is TransferDirection.DL
        )
    )

    def environment_with(rate_bps: float, deadline_s: float):
        split = (
            replace(
                template,
                segment_id="same-time-order-connected",
                start_time_s=0.0,
                end_time_s=boundary_s,
                goodput_bps=rate_bps,
                link_state="connected",
            ),
            replace(
                template,
                segment_id="same-time-order-permanent-loss",
                start_time_s=boundary_s,
                end_time_s=base.horizon_end_s,
                goodput_bps=0.0,
                link_state="permanent_loss",
            ),
        )
        trace = replace(
            base,
            arrivals=(replace(arrival, relative_deadline_s=deadline_s),),
            wireless=tuple(
                sorted(
                    (*other_wireless, *split),
                    key=lambda row: (row.start_time_s, row.segment_id),
                )
            ),
        )
        library = ScenarioLibrary.from_trace(
            trace,
            metadata_bits=config.metadata_bits,
            vehicle_anchor_parameters=vehicles,
            rsu_anchor_parameters=rsus,
        )
        return _environment_at(library, boundary_s)

    exact_rate = calibration_transfer.total_bits / service_duration_s
    exact = environment_with(exact_rate, 5.0)
    slightly_slow = environment_with(
        calibration_transfer.total_bits / (service_duration_s + 0.01), 5.0
    )
    exact_deadline = environment_with(
        exact_rate,
        boundary_s - arrival.arrival_time_s,
    )
    exact_vehicle = next(
        row for row in exact.vehicle_anchors if row.vehicle_id == arrival.vehicle_id
    )
    slow_vehicle = next(
        row
        for row in slightly_slow.vehicle_anchors
        if row.vehicle_id == arrival.vehicle_id
    )
    deadline_vehicle = next(
        row
        for row in exact_deadline.vehicle_anchors
        if row.vehicle_id == arrival.vehicle_id
    )

    # Exact delivery is completed before the new permanent-loss segment is
    # applied.  The slower control differs by one variable and therefore enters
    # frozen local fallback at the same boundary.
    assert not exact_vehicle.tasks
    assert not exact_vehicle.transfers
    assert [task.state for task in slow_vehicle.tasks] == ["LOCAL_FALLBACK"]
    assert not slow_vehicle.transfers
    # Completion at the absolute deadline is also terminal and releases every
    # reservation.  Scenario anchors intentionally omit terminal labels, so
    # this assertion covers boundary cleanup while production metrics cover the
    # DONE-vs-timeout classification itself.
    assert not deadline_vehicle.tasks
    assert not deadline_vehicle.transfers
    assert deadline_vehicle.memory_reserved_bytes == 0
    assert sum(deadline_vehicle.descriptors_reserved.values()) == 0
