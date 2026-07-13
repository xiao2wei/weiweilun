from __future__ import annotations

from privacy_edge_sim.enums import TransferDirection
from privacy_edge_sim.policies import POLICY_REGISTRY
from privacy_edge_sim.traces import (
    ScenarioEnvironment,
    ScenarioFaultEvent,
    ScenarioThermalSegment,
    ScenarioWirelessSegment,
)


def test_branch_telemetry_samples_current_rsu_thermal_context(decision_fixture):
    fixture = decision_fixture(task_id="branch-hot-rsu-snapshot")
    rsu_id = "rsu-1"
    environment = ScenarioEnvironment(
        scenario_id="hot-rsu-context",
        cluster_token="hot-rsu-context-cluster",
        duration_s=1.0,
        wireless=tuple(
            ScenarioWirelessSegment(
                vehicle_id=fixture.task.vehicle_id,
                rsu_id=rsu_id,
                direction=direction,
                start_offset_s=0.0,
                end_offset_s=1.0,
                goodput_bps=1_000_000.0,
                transmitter_power_w=2.0,
                receiver_power_w=1.0,
                link_state="connected",
            )
            for direction in (TransferDirection.UL, TransferDirection.DL)
        ),
        thermal=(
            ScenarioThermalSegment(
                owner_type="rsu",
                owner_id=rsu_id,
                resource="all",
                start_offset_s=0.05,
                end_offset_s=1.0,
                state="hot",
                service_rate_multiplier=0.6,
                dynamic_power_multiplier=0.8,
            ),
        ),
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
    assert branch.public_rsus[rsu_id]["device_context"].startswith("nominal|")

    branch.elapsed_s = 0.10
    sampled = policy._sample_public_rsu(branch, rsu_id, work_quantum_s=0.0)

    assert sampled is not None
    assert sampled["device_context"] == "hot|nominal|normal"


def test_branch_device_fault_is_device_wide_even_with_resource_metadata(
    decision_fixture,
):
    fixture = decision_fixture(task_id="branch-device-fault-normalization")
    rsu_id = "rsu-1"
    environment = ScenarioEnvironment(
        scenario_id="rsu-device-fault",
        cluster_token="rsu-device-fault-cluster",
        duration_s=1.0,
        wireless=(),
        thermal=(),
        faults=(
            ScenarioFaultEvent(
                offset_s=0.1,
                event_type="DEVICE_FAULT_START",
                target_type="rsu",
                target_id=rsu_id,
                resource="gpu",
                permanent=False,
            ),
        ),
        background_loads=(),
        telemetry=(),
    )
    policy = POLICY_REGISTRY["esl_smpc"](
        fixture.simulator.mask_engine,
        fixture.simulator.repairer,
        scenario_source=fixture.simulator.scenario_library,
    )
    branch = policy._new_branch(fixture.observation, environment)
    branch.elapsed_s = 0.1

    policy._apply_environment_at(branch, 0.1, fixture.task.vehicle_id)
    sampled = policy._sample_public_rsu(branch, rsu_id, work_quantum_s=0.0)

    assert policy._faulted(branch, "rsu", rsu_id, "all")
    assert policy._faulted(branch, "rsu", rsu_id, "gpu")
    assert sampled is not None and sampled["failed"] is True
