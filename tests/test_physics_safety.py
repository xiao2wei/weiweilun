from __future__ import annotations

import math
from dataclasses import replace

import pytest

from privacy_edge_sim.enums import ActionKind, FailureReason, ReasonCode, TaskState
from privacy_edge_sim.errors import InvariantViolation, TransitionError
from privacy_edge_sim.invariants import assert_all_invariants
from privacy_edge_sim.physics import (
    ServiceSegment,
    completion_time,
    integrate_until_complete,
)
from privacy_edge_sim.safety import (
    Action,
    ObservationBuilder,
    _ArtifactPairingTokenRegistry,
)
from privacy_edge_sim.simulator import _strict_future_completion_time
from privacy_edge_sim.state import TaskRecord, TaskStateMachine


def _segments(rate: float) -> tuple[ServiceSegment, ...]:
    return (
        ServiceSegment(0.0, 1.0, rate, 2.0, 3.0),
        ServiceSegment(1.0, 2.0, rate, 2.0, 3.0),
    )


def test_pairing_token_namespace_collision_keeps_capability_exact_and_task_scoped():
    registry = _ArtifactPairingTokenRegistry()
    task_id = "same-task-retry"
    first_capability = ("rsu-a", "edge-a", "pipeline-a")
    first_token = registry.issue(task_id, (first_capability,))
    assert first_token is not None

    # Sanitized identifiers are allowed to resemble an issued token.  They
    # must remain ordinary tuple members rather than being recursively
    # resolved through the token namespace.
    retry_capability = (first_token, "edge-b", "pipeline-b")
    retry_token = registry.issue(task_id, (retry_capability,))

    assert retry_token is not None and retry_token != first_token
    assert registry.allows(
        task_id,
        first_token,
        rsu_id=first_capability[0],
        model_id=first_capability[1],
        pipeline_id=first_capability[2],
    )
    assert registry.allows(
        task_id,
        retry_token,
        rsu_id=retry_capability[0],
        model_id=retry_capability[1],
        pipeline_id=retry_capability[2],
    )
    assert not registry.allows(
        task_id,
        retry_token,
        rsu_id=first_capability[0],
        model_id=first_capability[1],
        pipeline_id=first_capability[2],
    )
    assert not registry.allows(
        "other-task",
        retry_token,
        rsu_id=retry_capability[0],
        model_id=retry_capability[1],
        pipeline_id=retry_capability[2],
    )


def test_more_bits_cannot_complete_earlier_under_same_service_trace():
    small = completion_time(100.0, start_s=0.0, segments=_segments(100.0))
    large = completion_time(150.0, start_s=0.0, segments=_segments(100.0))
    assert small == pytest.approx(1.0)
    assert large == pytest.approx(1.5)
    assert large >= small


def test_pointwise_lower_goodput_cannot_complete_earlier():
    fast = completion_time(120.0, start_s=0.0, segments=_segments(100.0))
    slow = completion_time(120.0, start_s=0.0, segments=_segments(80.0))
    assert slow is not None and fast is not None
    assert slow >= fast


def test_service_integral_stops_exactly_inside_segment_and_pairs_both_energies():
    result = integrate_until_complete(50.0, start_s=0.0, segments=_segments(100.0))
    assert result.completed
    assert result.delivered == pytest.approx(50.0)
    assert result.end_s == pytest.approx(0.5)
    assert result.side_a_energy_j == pytest.approx(1.0)
    assert result.side_b_energy_j == pytest.approx(1.5)


def test_zero_service_interruption_preserves_time_and_bilateral_energy():
    segments = (
        ServiceSegment(0.0, 1.0, 100.0, 2.0, 3.0),
        ServiceSegment(1.0, 2.0, 0.0, 1.0, 1.5, available=False),
        ServiceSegment(2.0, 3.0, 100.0, 2.0, 3.0),
    )
    result = integrate_until_complete(150.0, start_s=0.0, segments=segments)
    assert result.completed and result.end_s == pytest.approx(2.5)
    assert result.delivered == pytest.approx(150.0)
    assert result.side_a_energy_j == pytest.approx(4.0)
    assert result.side_b_energy_j == pytest.approx(6.0)


def test_service_integral_stop_inside_uncovered_gap_never_returns_future_time():
    result = integrate_until_complete(
        1.0,
        start_s=0.0,
        stop_s=5.0,
        segments=(ServiceSegment(10.0, 20.0, 1.0),),
    )

    assert not result.completed
    assert result.delivered == 0.0
    assert result.end_s == pytest.approx(5.0)


def test_sub_ulp_positive_service_gets_a_strict_future_completion_time():
    now_s = 2.8596357755652373
    remaining_bits = 2.4399469111813232e-9
    finish = _strict_future_completion_time(now_s, remaining_bits, 6_000_000.0)

    assert finish > now_s
    assert finish == pytest.approx(
        math.nextafter(now_s, math.inf),
        rel=0.0,
        abs=0.0,
    )


def test_sub_epsilon_positive_interval_is_physically_integrated(decision_fixture):
    simulator = decision_fixture(task_id="sub-epsilon-advance").simulator
    simulator.state.clock_s = 2.8596357755652373
    start_s = simulator.state.clock_s
    finish_s = math.nextafter(start_s, math.inf)
    vehicle = next(iter(simulator.state.vehicles.values()))
    before = vehicle.physical_energy_j

    simulator._advance_to(finish_s)

    assert simulator.state.clock_s == finish_s
    assert vehicle.physical_energy_j > before


def test_appending_executed_failed_attempt_cannot_reduce_time_or_energy(trace):
    row = next(row for row in trace.anon_rows if len(row.attempts) >= 2)
    prefix = row.attempts[:1]
    extended = row.attempts[:2]
    prefix_work = sum(attempt.executed_work_s for attempt in prefix)
    prefix_energy = sum(attempt.executed_energy_j for attempt in prefix)
    extended_work = sum(attempt.executed_work_s for attempt in extended)
    extended_energy = sum(attempt.executed_energy_j for attempt in extended)

    assert prefix[0].guard_passed is False or prefix[0].anon_oom
    assert extended_work >= prefix_work
    assert extended_energy >= prefix_energy
    assert extended_work > prefix_work or extended_energy > prefix_energy


def test_thermal_throttling_cannot_shorten_same_work_completion():
    nominal = completion_time(1.0, start_s=0.0, segments=_segments(1.0))
    throttled = completion_time(1.0, start_s=0.0, segments=_segments(0.5))
    assert nominal == pytest.approx(1.0)
    assert throttled == pytest.approx(2.0)
    assert throttled >= nominal


def test_ood_and_unsafe_privacy_actions_are_hard_removed(decision_fixture):
    ood = decision_fixture(ood=True)
    ood_mask = ood.simulator.mask_engine.enumerate(ood.task, ood.observation, ood.state)
    pipeline_actions = [
        action for action in ood_mask.candidates if action.kind is ActionKind.PIPE
    ]
    assert pipeline_actions
    assert all(
        ReasonCode.OOD in ood_mask.reasons_for(action) for action in pipeline_actions
    )

    risky = decision_fixture(bins=("clear", "challenging"))
    risky_mask = risky.simulator.mask_engine.enumerate(
        risky.task, risky.observation, risky.state
    )
    blur = Action.pipeline("blur_balanced_v1")
    pixelate = Action.pipeline("pixelate_strong_v1")
    assert ReasonCode.PRIVACY_RISK in risky_mask.reasons_for(blur)
    assert ReasonCode.PRIVACY_RISK not in risky_mask.reasons_for(pixelate)
    assert risky_mask.records[pixelate].details["privacy"]["safe"] is True


def test_unsupported_and_version_mismatch_actions_are_hard_removed(decision_fixture):
    fixture = decision_fixture()
    unsupported_observation = replace(
        fixture.observation, device_context="unseen|unseen|unseen"
    )
    unsupported = fixture.simulator.mask_engine.enumerate(
        fixture.task, unsupported_observation, fixture.state
    )
    assert any(
        ReasonCode.JOINT_TRACE_MISSING in unsupported.reasons_for(action)
        for action in unsupported.candidates
        if action.kind is ActionKind.PIPE
    )

    versions = dict(fixture.observation.versions)
    versions["protocol_version"] = "incompatible-protocol"
    incompatible_observation = replace(fixture.observation, versions=versions)
    incompatible = fixture.simulator.mask_engine.enumerate(
        fixture.task, incompatible_observation, fixture.state
    )
    for action in incompatible.candidates:
        if action.kind is not ActionKind.FAIL:
            assert ReasonCode.PROTOCOL_MISMATCH in incompatible.reasons_for(action)


def test_support_provider_internal_type_error_is_never_swallowed(decision_fixture):
    fixture = decision_fixture(task_id="provider-type-error")

    class BrokenProvider:
        def has_anon_support(self, **kwargs):
            raise TypeError("provider implementation bug")

    fixture.simulator.mask_engine.trace_support = BrokenProvider()
    action = Action.pipeline(min(fixture.simulator.profile.pipelines))

    with pytest.raises(TypeError, match="provider implementation bug"):
        fixture.simulator.mask_engine._call_support("anon", action, fixture.observation)


def test_deterministic_repair_cannot_bypass_privacy_mask(decision_fixture):
    fixture = decision_fixture(bins=("clear", "challenging"))
    proposed = Action.pipeline("blur_balanced_v1")
    decision = fixture.simulator.repairer.repair(
        proposed, fixture.task, fixture.observation, fixture.state
    )

    assert decision.changed
    assert ReasonCode.PRIVACY_RISK in decision.proposed_reasons
    assert decision.executed in decision.mask.allowed
    assert decision.executed != proposed


def test_shorter_deadline_cannot_expand_nonfailure_action_set(decision_fixture):
    fixture = decision_fixture(deadline_s=10.0)
    long_mask = fixture.simulator.mask_engine.enumerate(
        fixture.task, fixture.observation, fixture.state
    )
    short_observation = replace(
        fixture.observation,
        absolute_deadline_s=0.001,
        slack_s=0.001,
    )
    short_mask = fixture.simulator.mask_engine.enumerate(
        fixture.task, short_observation, fixture.state
    )

    long_nonfail = {
        action for action in long_mask.allowed if action.kind is not ActionKind.FAIL
    }
    short_nonfail = {
        action for action in short_mask.allowed if action.kind is not ActionKind.FAIL
    }
    assert short_nonfail <= long_nonfail


def test_observation_builder_never_exposes_simulator_only_truth(decision_fixture):
    fixture = decision_fixture()
    serialized = fixture.observation.to_dict()
    flattened_keys = " ".join(str(key).lower() for key in _walk_keys(serialized))

    for forbidden in (
        "raw_handle",
        "aligned_handle",
        "true_identity",
        "true_expression_label",
        "true_quality_region",
        "realized_attack_outcomes",
        "realized_fer_loss",
        "future_trace",
    ):
        assert forbidden not in flattened_keys

    assert fixture.task.true_identity == "simulator-only-person"
    assert fixture.task.true_quality_region is not None
    assert fixture.task.realized_attack_outcomes


@pytest.mark.parametrize(
    "context",
    [
        {"metadata": {"true_identity": "leak"}},
        {"metadata": {"true_quality_region": "leak"}},
        {"metadata": {"raw_handle": "leak"}},
        {"metadata": {"future_trace": [1, 2, 3]}},
    ],
)
def test_observation_builder_rejects_leaking_context(decision_fixture, context):
    fixture = decision_fixture()
    with pytest.raises(ValueError):
        ObservationBuilder.build(
            fixture.task,
            fixture.state,
            profile=fixture.simulator.profile,
            context=context,
        )


def test_observation_builder_rejects_nested_task_object_under_innocent_key(
    decision_fixture,
):
    fixture = decision_fixture()
    with pytest.raises(ValueError):
        ObservationBuilder.build(
            fixture.task,
            fixture.state,
            profile=fixture.simulator.profile,
            metadata={"innocent": fixture.task},
        )


def test_terminal_invariant_rejects_ghost_job_pointer(decision_fixture):
    fixture = decision_fixture(task_id="ghost-terminal")
    fixture.simulator._terminate_fail(
        fixture.task,
        FailureReason.POLICY_EXPLICIT_FAIL,
        "UNIT_TERMINATE",
    )
    fixture.task.current_job_id = "ghost-job"

    with pytest.raises(InvariantViolation) as caught:
        assert_all_invariants(fixture.state, fixture.simulator.profile)
    assert caught.value.detail.code == "TERMINAL_RESOURCE_LEAK"


def test_retry_counter_has_no_off_by_one_escape(decision_fixture):
    task = decision_fixture().task
    assert task.mark_anon_enqueued(2) == 1
    assert task.mark_anon_enqueued(2) == 2
    with pytest.raises(InvariantViolation) as caught:
        task.mark_anon_enqueued(2)
    assert caught.value.detail.code == "RETRY_LIMIT_EXCEEDED"
    assert task.attempt_started_count == 2


def test_downlink_without_valid_result_cannot_complete():
    task = TaskRecord(
        task_id="downlink-fail",
        vehicle_id="veh-1",
        arrival_time_s=0.0,
        relative_deadline_s=5.0,
        absolute_deadline_s=5.0,
        raw_handle=None,
    )
    path = (
        TaskState.PREP_WAIT,
        TaskState.PREP_RUN,
        TaskState.RAW,
        TaskState.ANON_WAIT,
        TaskState.ANON_RUN,
        TaskState.GUARD_WAIT,
        TaskState.GUARD_RUN,
        TaskState.ENCODE_WAIT,
        TaskState.ENCODE_RUN,
        TaskState.READY,
        TaskState.UL,
        TaskState.EDGE_WAIT,
        TaskState.EDGE_RUN,
        TaskState.DL,
    )
    for target in path:
        TaskStateMachine.transition(task, target, time_s=0.0, trigger=target.value)

    task.result_valid = False
    with pytest.raises(TransitionError) as caught:
        TaskStateMachine.transition(
            task, TaskState.DONE, time_s=1.0, trigger="INVALID_DL"
        )
    assert caught.value.detail.code == "DONE_RESULT_INVALID"


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)
