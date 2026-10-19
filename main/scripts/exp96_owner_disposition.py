"""Explicit owner risk disposition; never promotes incomplete engineering to PASS."""

import math
from pathlib import Path

from scripts import exp96_multigpu as m

SCHEMA = "exp96-owner-disposition-v1"
BASE = "d92e6d839ec142892de8a7cf239329acaded5e6d"
WAIVERS = [
    "unfinished_D4W1536_packing_and_checkpoint",
    "unfinalized_aggregate_resource_profiles",
]


def validate(path, commit):
    q = m.read(path)
    owner = m.read(q["owner_decision"])
    if (
        q.get("schema") != SCHEMA
        or q.get("verdict") != "OWNER_AUTHORIZED_WITH_UNVERIFIED_LARGE_PACKING"
        or q.get("expected_commit") != commit
        or q.get("limits") != m.LIMITS
        or q.get("compositions") != [list(c) for c in m.COMPOSITIONS]
        or q.get("waived_requirements") != WAIVERS
        or m.digest(q["owner_decision"]) != q["owner_decision_sha256"]
        or owner.get("authorize_submission") is not True
        or owner.get("accept_unverified_large_packing") is not True
        or owner.get("limit_su") != 600
        or not owner.get("verbatim_instructions")
    ):
        raise ValueError("explicit immutable owner disposition required")
    if m.training_fingerprint(commit) != m.training_fingerprint(m.BASE_SMOKE_SHA):
        raise ValueError("scientific source changed")
    # Only the admission mechanism and documentation may differ from the actual test.
    changed = m.checked(
        ["git", "diff", "--name-only", BASE, commit], cwd=m.MAIN.parent
    ).splitlines()
    allowed = {
        "main/scripts/exp96_multigpu.py",
        "main/scripts/exp96_owner_disposition.py",
        "main/tests/test_exp96_owner_disposition.py",
        "main/docs/exp96_owner_disposition.md",
        "main/docs/AGENT_HANDOFF.md",
        "main/docs/EXPERIMENT_STATUS.md",
    }
    if not changed or not set(changed) <= allowed:
        raise ValueError("unreviewed source compatibility difference")
    required = {q["partial_qualification"], q["owner_decision"]}
    if not required <= set(q["evidence_sha256"]):
        raise ValueError("decision and original outcome must be hash-bound")
    for f, expected in q["evidence_sha256"].items():
        if m.digest(f) != expected:
            raise ValueError("owner disposition evidence changed: " + f)
    raw = m.read(q["partial_qualification"])
    if (
        raw.get("expected_commit") != BASE
        or raw.get("verdict") != "INCOMPLETE"
        or raw.get("error") != "TimeoutError: targeted1680s workload cap reached"
        or not all(
            raw.get(k)
            for k in (
                "claims_race_passed",
                "production_orchestration_passed",
                "gpu_startup_passed",
            )
        )
    ):
        raise ValueError("unexpected partial qualification outcome")
    from scripts.exp96_multigpu_qualification import reusable_checks

    reusable_checks(raw["reused_checks"]["path"], commit, raw["reused_checks"])
    if m.digest(raw["original_smoke"]) != m.BASE_SMOKE_DIGEST:
        raise ValueError("original smoke changed")
    from scripts.exp96_multigpu_qualification import validate_probe_result
    import numpy as np
    from scripts.exp12_fullwidth_gate import gpu_placement_ok

    root = Path(q["partial_qualification"]).parent
    for lane, count in enumerate((7, 5)):
        phase = root / "qualification" / f"phase_{lane}"
        if m.read(phase / "phase_complete.json")["children"] != count:
            raise ValueError("completed phase missing")
        cpus = []
        for ordinal in range(count):
            receipt_path = phase / f"{lane}_{ordinal}" / "child_receipt.json"
            x = m.read(receipt_path)
            if (
                x["status"] != "PASS"
                or x["source"] != BASE
                or x["architecture"] != m.ARCHS[lane]
                or x["runtime"]["platform"] != "gpu"
                or x["runtime"]["matmul_precision"] != "tensorfloat32"
                or not gpu_placement_ok(x["restored_placement"])
                or m.digest(x["curve_path"]) != x["curve_sha256"]
                or str(receipt_path) not in q["evidence_sha256"]
                or x["curve_path"] not in q["evidence_sha256"]
            ):
                raise ValueError("completed fitting evidence invalid")
            cpus.extend(x["cpu_ids"])
            with np.load(x["curve_path"]) as data:
                result = {
                    n: {
                        k[len(n) + 1 :]: data[k]
                        for k in data.files
                        if k.startswith(n + "_")
                    }
                    for n in ("current", "fresh")
                }
            validate_probe_result(result)
            if ordinal == 0 and x["roundtrip_differences"] != []:
                raise ValueError("completed checkpoint round trip failed")
        if len(cpus) != len(set(cpus)):
            raise ValueError("CPU isolation failed")
    if set(q["profiles"]) != set(m.ARCHS) or q["gpu_headroom_mb"] != 2048:
        raise ValueError("resource reservations incomplete")
    for arch, p in q["profiles"].items():
        values = (
            p["gpu_memory_mb"],
            p["host_ram_gb"],
            p["checkpoint_minutes"],
            p["estimated_minutes"],
        )
        if (
            p["cpus"] != m.CPUS[arch]
            or not all(math.isfinite(v) and v > 0 for v in values)
            or p["measurement_status"] != "planning_reservation_not_peak_qualification"
        ):
            raise ValueError("invalid disclosed planning reservation")
        if p["gpu_memory_mb"] * m.LIMITS[arch] + 2048 > 40960:
            raise ValueError("packing reservation exceeds physical VRAM")
    return q
