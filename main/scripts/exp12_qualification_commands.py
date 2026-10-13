#!/usr/bin/env python3
"""Write (never run) the reviewed-shape commands for Exp1/2 CUDA qualification on Delta.

The output is a bash file for a person to inspect and run after GPU authorization. It holds:
  - the six cold CUDA identity cases (scripts/sbatch_exp12_identity_case.sh), each with a
    walltime sized from measured A100 costs instead of the earlier uniform four-hour cap;
  - the CUDA numerical-bounds measurement (scripts/sbatch_exp12_numeric_bounds.sh);
  - optionally, the CPU-only two-node shared-filesystem claim check
    (scripts/sbatch_exp12_shared_fs_claim_check.sh) on an explicitly named CPU partition.

Nothing here changes an approval gate: the parent gate still needs all six identity
certificates. The two D4W1536 cases come first because they bound the widest single-critic
and twin-critic programs; every case is submitted at once so no case waits on another.

Submissions are idempotent: each is reserved atomically under QUAL_ROOT/submissions/<name>
and its Slurm job id recorded durably, so rerunning the script never submits twice. A
reservation without a job id (sbatch failed or was interrupted) stops every later run until
it is resolved with `reconcile --job-id N` or `reconcile --release`; `status` lists them.
"""

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

# (case, walltime). Estimates (docs/exp12_min_qualification.md): fork at check 2 is 50,000
# interactions for DMC/MyoSuite and 100,000 for HumanoidBench, plus 2 x 1,000 continuation
# steps and three Method C probe checks. D4W1536 dog-run is anchored on measured A100 costs
# (42.4 interactions/s, ~3.5 s per 1,000-step fit); the others use earlier, unverified rates.
IDENTITY_CASES = (
    ("D4W1536_dog-run", "01:30:00"),        # est. 25-30 min
    ("D4W1536_h1-run-v0", "03:00:00"),      # est. 60-75 min; twin critic, weakest rate estimate
    ("D4W1024_dog-run", "01:30:00"),        # est. 15-20 min
    ("D4W1024_myo-key-turn", "01:30:00"),   # est. 15-25 min; MyoSuite A100 rate unmeasured
    ("D4W1536_myo-key-turn", "01:30:00"),   # est. 25-35 min; MyoSuite A100 rate unmeasured
    ("D4W1024_h1-run-v0", "03:00:00"),      # est. 35-45 min
)
PRIORITY = ("D4W1536_dog-run", "D4W1536_h1-run-v0")
IDENTITY_RESOURCES = ("--cpus-per-task=16", "--mem=64G")
NUMERICS_TIME = "00:30:00"
CLAIM_CHECK_TIME = "00:15:00"
SHA = re.compile(r"[0-9a-f]{40}")


def _walltime_minutes(value):
    hours, minutes, seconds = (int(x) for x in value.split(":"))
    return hours * 60 + minutes + (seconds > 0)


SUBMIT_ONCE = r"""
# Each submission is reserved atomically (mkdir) before sbatch and its job id is recorded
# durably (temporary file, fsync, rename). A rerun skips recorded submissions and stops at
# a reservation without a job id: that needs `exp12_qualification_commands.py reconcile`.
SUBMISSIONS="$QUAL_ROOT/submissions"
mkdir -p "$SUBMISSIONS"
submit_once() {
  local name=$1 absent=$2
  shift 2
  local dir="$SUBMISSIONS/$name" out id
  # A reservation only counts for the commit it was made for; anything else fails closed.
  if [ -e "$dir" ] && ! grep -qF "\"commit\":\"$FINAL_SHA\"" "$dir/reserved.json" 2>/dev/null; then
    echo "FOREIGN: $name is reserved for another commit (or has no readable reservation);" \
      "use a qualification root for $FINAL_SHA" >&2
    exit 3
  fi
  if [ -f "$dir/job_id" ]; then
    echo "already submitted: $name (job $(cat "$dir/job_id"))"
    return 0
  fi
  if [ -e "$dir" ]; then
    echo "PARTIAL: $name is reserved without a recorded job id; check sacct for job name" \
      "exp12q_${FINAL_SHA:0:12}_$name, then reconcile (exp12_qualification_commands.py reconcile)" >&2
    exit 3
  fi
  if [ -e "$absent" ]; then
    echo "refusing to submit $name: $absent already exists; reconcile" >&2
    exit 3
  fi
  if ! mkdir "$dir" 2>/dev/null; then
    echo "PARTIAL: $name was reserved concurrently; reconcile before rerunning" >&2
    exit 3
  fi
  printf '{"name":"%s","commit":"%s","reserved_at":"%s","host":"%s"}\n' \
    "$name" "$FINAL_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" > "$dir/reserved.json"
  if ! out=$("$@"); then
    echo "sbatch failed for $name; the reservation is kept: reconcile before rerunning" >&2
    exit 3
  fi
  id=${out%%;*}
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "unparseable sbatch output for $name: $out; reconcile" >&2
    exit 3
  fi
  printf '%s\n' "$id" > "$dir/.job_id.tmp"
  sync "$dir/.job_id.tmp" 2>/dev/null || sync
  mv "$dir/.job_id.tmp" "$dir/job_id"
  echo "submitted $name as job $id"
}
"""


def submission_names(cpu_claim_check=False):
    names = [f"identity_{case}" for case, _ in IDENTITY_CASES] + ["numerics_01"]
    return names + (["claim_check_01"] if cpu_claim_check else [])


def render(final_sha, checkout, qual_root, hb_env, cpu_partition=None, cpu_account=None):
    if not SHA.fullmatch(final_sha):
        raise ValueError("final SHA must be a full lowercase 40-character hash")
    for name, path in (("checkout", checkout), ("qualification root", qual_root), ("HB_ENV", hb_env)):
        if not str(path).startswith("/"):
            raise ValueError(f"{name} must be absolute")
    if (cpu_partition is None) != (cpu_account is None):
        raise ValueError("the CPU claim check needs both --cpu-partition and --cpu-account")
    q = shlex.quote
    tag = f"exp12q_{final_sha[:12]}"
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by scripts/exp12_qualification_commands.py. Review before running;",
        "# running it submits GPU jobs, which needs explicit authorization. Safe to rerun:",
        "# recorded submissions are skipped and a partial one stops the script.",
        "set -euo pipefail",
        f"FINAL_SHA={final_sha}",
        f"FINAL_CHECKOUT={q(str(checkout))}",
        f"QUAL_ROOT={q(str(qual_root))}",
        f"HB_ENV={q(str(hb_env))}",
        'test "$(git -C "$FINAL_CHECKOUT" rev-parse HEAD)" = "$FINAL_SHA"',
        'test -z "$(git -C "$FINAL_CHECKOUT" status --porcelain --untracked-files=all)"',
        'test -x "$HB_ENV/bin/python"',
        'cd "$FINAL_CHECKOUT/main"',
        'mkdir -p "$QUAL_ROOT/slurm" "$QUAL_ROOT/identity"',
        SUBMIT_ONCE.strip("\n"),
        "",
        "# Six cold identity cases, one A100 each, all submitted together (priority cases first).",
    ]
    for case, walltime in IDENTITY_CASES:
        kind = "priority" if case in PRIORITY else "remaining"
        root = f'"$QUAL_ROOT/identity/{case}_cold_01"'
        lines += [
            f"# {case}: {kind}, cap {walltime}",
            f"submit_once identity_{case} {root} \\",
            f'  env EXPECTED_COMMIT="$FINAL_SHA" IDENTITY_CASE={case} IDENTITY_CACHE_MODE=cold \\',
            f'  IDENTITY_ROOT={root} HB_ENV="$HB_ENV" \\',
            f"  sbatch --parsable --export=ALL {' '.join(IDENTITY_RESOURCES)} --time={walltime} \\",
            f"    --job-name={tag}_identity_{case} \\",
            '    --output="$QUAL_ROOT/slurm/identity_%j.out" --error="$QUAL_ROOT/slurm/identity_%j.err" \\',
            "    scripts/sbatch_exp12_identity_case.sh",
        ]
    lines += [
        "",
        "# CUDA numerical bounds: the three diagnostics-isolation tests in both modes; proposes only.",
        'submit_once numerics_01 "$QUAL_ROOT/numerics_01" \\',
        '  env EXPECTED_COMMIT="$FINAL_SHA" NUMERICS_ROOT="$QUAL_ROOT/numerics_01" \\',
        f"  sbatch --parsable --export=ALL {' '.join(IDENTITY_RESOURCES)} --time={NUMERICS_TIME} \\",
        f"    --job-name={tag}_numerics_01 \\",
        '    --output="$QUAL_ROOT/slurm/numerics_%j.out" --error="$QUAL_ROOT/slurm/numerics_%j.err" \\',
        "    scripts/sbatch_exp12_numeric_bounds.sh",
    ]
    if cpu_partition is not None:
        lines += [
            "",
            "# CPU-only two-node shared-filesystem claim check (no GPU).",
            'submit_once claim_check_01 "$QUAL_ROOT/claim_check_01" \\',
            '  env EXPECTED_COMMIT="$FINAL_SHA" EXP12_CLAIM_CHECK_ROOT="$QUAL_ROOT/claim_check_01" \\',
            f"  sbatch --parsable --export=ALL --account={q(cpu_account)} --partition={q(cpu_partition)} \\",
            f"    --nodes=2 --ntasks-per-node=1 --cpus-per-task=2 --mem=4G --time={CLAIM_CHECK_TIME} \\",
            f"    --job-name={tag}_claim_check_01 \\",
            '    --output="$QUAL_ROOT/slurm/claim_check_%j.out" --error="$QUAL_ROOT/slurm/claim_check_%j.err" \\',
            "    scripts/sbatch_exp12_shared_fs_claim_check.sh",
        ]
    return "\n".join(lines) + "\n"


JOB_ID = re.compile(r"[0-9]+")


def _submissions(qual_root):
    return Path(qual_root) / "submissions"


def _reserved_commit(entry):
    try:
        return json.loads((entry / "reserved.json").read_text()).get("commit")
    except (OSError, ValueError, AttributeError):
        return None


def reconcile_status(qual_root, names=None, commit=None):
    """SUBMITTED (job id recorded), PARTIAL (reserved, no job id) or NOT_SUBMITTED, for the
    expected submissions and any other reservation found. With commit, a reservation made
    for any other commit (or without a readable reservation) is FOREIGN, never SUBMITTED."""
    directory = _submissions(qual_root)
    found = sorted(p.name for p in directory.iterdir() if not p.name.startswith(".")) if directory.is_dir() else []
    status = {}
    for name in list(names or submission_names()) + [n for n in found if n not in (names or submission_names())]:
        entry = directory / name
        job = entry / "job_id"
        if entry.exists() and commit is not None and _reserved_commit(entry) != commit:
            status[name] = {"state": "FOREIGN", "reserved_commit": _reserved_commit(entry)}
        elif job.is_file() and JOB_ID.fullmatch(job.read_text().strip()):
            status[name] = {"state": "SUBMITTED", "job_id": job.read_text().strip()}
        elif entry.exists():
            status[name] = {"state": "PARTIAL"}
        else:
            status[name] = {"state": "NOT_SUBMITTED"}
    return status


def reconcile(qual_root, name, job_id=None, release=False, reason=None, commit=None):
    """Resolve one PARTIAL reservation explicitly, after checking Slurm (sacct --name):
    record the job id that was found, or release the reservation when no job exists. The
    reservation's files are kept (released ones move under submissions/.released/)."""
    import os
    import time
    import uuid

    if (job_id is None) == (not release):
        raise ValueError("give exactly one of a job id or release")
    if not reason or not str(reason).strip():
        raise ValueError("a reconciliation needs a reason (what Slurm showed)")
    entry = _submissions(qual_root) / name
    state = reconcile_status(qual_root, [name], commit)[name]["state"]
    if state != "PARTIAL":
        raise ValueError(f"{name} is {state}; only a PARTIAL reservation can be reconciled")
    record = {"name": name, "reason": str(reason), "reconciled_at": time.time(),
              "action": "record" if job_id is not None else "release"}
    if job_id is not None:
        if not JOB_ID.fullmatch(str(job_id)):
            raise ValueError(f"invalid Slurm job id {job_id!r}")
        record["job_id"] = str(job_id)
    with (entry / "reconciled.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    if job_id is not None:
        temporary = entry / ".job_id.tmp"
        with temporary.open("w") as stream:
            stream.write(f"{job_id}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, entry / "job_id")
    else:
        released = _submissions(qual_root) / ".released"
        released.mkdir(exist_ok=True)
        entry.rename(released / f"{name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action")
    status = sub.add_parser("status", help="show each submission's reservation state")
    status.add_argument("--qual-root", required=True)
    status.add_argument("--final-sha", required=True, help="reservations for any other commit are FOREIGN")
    fix = sub.add_parser("reconcile", help="resolve one PARTIAL reservation")
    fix.add_argument("--qual-root", required=True)
    fix.add_argument("--name", required=True)
    group = fix.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-id")
    group.add_argument("--release", action="store_true")
    fix.add_argument("--reason", required=True)
    fix.add_argument("--final-sha", required=True, help="only this commit's reservations can be reconciled")
    parser.add_argument("--final-sha")
    parser.add_argument("--checkout", help="clean detached checkout root of FINAL_SHA")
    parser.add_argument("--qual-root")
    parser.add_argument("--hb-env", help="reviewed HumanoidBench environment prefix")
    parser.add_argument("--cpu-partition", help="CPU partition for the claim check (optional)")
    parser.add_argument("--cpu-account", help="Slurm account for that CPU partition")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.action == "status":
        print(json.dumps(reconcile_status(args.qual_root, commit=args.final_sha), indent=2))
        return 0
    if args.action == "reconcile":
        print(json.dumps(reconcile(args.qual_root, args.name, args.job_id, args.release, args.reason,
                                   args.final_sha), indent=2))
        return 0
    missing = [n for n in ("final_sha", "checkout", "qual_root", "hb_env", "out") if getattr(args, n) is None]
    if missing:
        parser.error("writing the script needs " + ", ".join("--" + m.replace("_", "-") for m in missing))
    text = render(args.final_sha, args.checkout, args.qual_root, args.hb_env,
                  args.cpu_partition, args.cpu_account)
    with args.out.open("x") as stream:
        stream.write(text)
    total = sum(_walltime_minutes(t) for _, t in IDENTITY_CASES) + _walltime_minutes(NUMERICS_TIME)
    print(f"wrote {args.out}; A100-hour caps: {total / 60:.1f} (identity + numerics); nothing submitted",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
