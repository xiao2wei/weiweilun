from __future__ import annotations

import json

from privacy_edge_sim.enums import Operation
from privacy_edge_sim.profiles import canonical_document_sha256
from privacy_edge_sim.simulator import DiscreteEventSimulator
from privacy_edge_sim.traces import load_trace


def test_joint_ingress_failure_charges_ingress_and_never_starts_gpu(
    config, profile, tmp_path
):
    document = json.loads(config.trace_path.read_text(encoding="utf-8"))
    for row in document["edge_fer"]:
        row["ingress_failed"] = True
    document["trace_hash"] = canonical_document_sha256(document, "trace_hash")
    path = tmp_path / "ingress-failure-trace.json"
    path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    trace = load_trace(path, profile)

    result = DiscreteEventSimulator(
        config,
        profile,
        trace,
        "fixed_safe_lowest_link_cost",
        policy_name="fixed_safe_lowest_link_cost",
    ).run()

    affected = [
        task
        for task in result.state.tasks.values()
        if any(
            row.get("phase") == "ingress_done" and row.get("valid") is False
            for row in task.rsu_audit
        )
    ]
    assert affected, "the fixed edge baseline must exercise the ingress path"
    for task in affected:
        assert Operation.RSU_INGRESS.value in task.end_times
        assert Operation.EDGE_FER.value not in task.enqueue_times
        assert task.rsu_energy_j > 0.0
        assert (
            any(
                row.get("repair") == "FROZEN_LOCAL_FALLBACK"
                and row.get("trigger") == "EDGE_FAILED"
                for row in task.action_audit
            )
            or task.failure_reason.value == "EDGE_FAILED"
        )
    assert all(rsu.admission.descriptors == 0 for rsu in result.state.rsus.values())
    assert not result.invariant_failures
