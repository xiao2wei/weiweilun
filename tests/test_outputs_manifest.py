from __future__ import annotations

from dataclasses import replace

import pytest

from privacy_edge_sim.manifest import build_manifest
from privacy_edge_sim.metrics import _config_core


def _manifest_kwargs(result, core_digest: str) -> dict:
    return {
        "config": result.config,
        "profile": result.profile,
        "state": result.state,
        "metrics_artifacts": {
            "core_digest": core_digest,
            "files": {},
            "parquet_status": {},
            "controller_diagnostics": {},
        },
        "trace_bundle": result.trace,
        "scenario_trace_bundle": result.scenario_trace,
    }


@pytest.mark.parametrize("digest", ["0" * 63, "A" * 64, "g" * 64])
def test_manifest_rejects_noncanonical_core_digest(policy_results, digest):
    result = policy_results["all_local"]
    with pytest.raises(ValueError, match="lowercase SHA-256 core_digest"):
        build_manifest(**_manifest_kwargs(result, digest))


def test_manifest_rejects_invariant_status_contradictions(policy_results):
    result = policy_results["all_local"]
    kwargs = _manifest_kwargs(result, "0" * 64)
    assert result.state.invariant_checks > 0

    with pytest.raises(ValueError, match="contradicts invariant failure/check counts"):
        build_manifest(
            **kwargs,
            invariant_failures=({"code": "TEST_FAILURE"},),
            invariants_passed=True,
        )

    with pytest.raises(ValueError, match="contradicts invariant failure/check counts"):
        build_manifest(**kwargs, invariant_failures=(), invariants_passed=False)


def test_core_configuration_includes_snapshot_period(config):
    original = _config_core(config)
    changed = _config_core(
        replace(config, rsu_snapshot_period_s=config.rsu_snapshot_period_s / 2.0)
    )

    assert original["rsu_snapshot_period_s"] == config.rsu_snapshot_period_s
    assert changed["rsu_snapshot_period_s"] != original["rsu_snapshot_period_s"]
    assert changed != original


def test_event_output_rows_have_explicit_record_kind_and_attempt_time(policy_results):
    result = policy_results["safe_lyapunov_h1"]
    rows = result.ledger._event_output_rows(result.state)
    event_batches = [row for row in rows if "events" in row]
    attempts = [row for row in rows if row.get("audit_type") == "ANON_ATTEMPT"]

    assert event_batches
    assert attempts
    assert all(row["record_kind"] == "EVENT_BATCH" for row in event_batches)
    assert all(
        row["record_kind"] == "ANON_ATTEMPT" and "time_s" in row for row in attempts
    )
