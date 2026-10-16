# Approval-pending Exp96 allocation

This is an operational queue reservation. It does **not** authorize any of the
96 official parents. The job enters the normal Slurm queue without a hold. If
approval is incomplete when it starts, the startup gate writes
`queue_v1/slurm/startup_gate_<jobid>.json` with `BLOCKED` and exits 3 before
`srun`, the durable ledger, or any training process. It does not wait on an
allocated GPU. A `PASS` receipt binds the live approval hash, source SHA,
immutable queue, actual Slurm resources, pinned Python stack, and an A100 smoke
report; the existing worker checks that hash again and rechecks it during
execution. A Kaggle T4 smoke cannot satisfy the A100 gate. The separate
scientific calibration and owner decisions remain mandatory.

Use a **fresh clean checkout** of the final published operational source.
The repository has a nested `main/` directory. Choose absolute, empty campaign
and queue paths on a filesystem shared by all eventual allocations. On Delta:

```bash
BASE=/work/hdd/biqc/skaveti1/exp96
mkdir -p "$BASE"
SHA='REPLACE_WITH_PUBLISHED_40_HEX_GATE_COMMIT'
git clone https://github.com/kaves29/scaling-drl-research.git "$BASE/checkout_$SHA"
cd "$BASE/checkout_$SHA"
git fetch origin codex/exp96-pending-gate
git checkout --detach "$SHA"
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain --untracked-files=all)"
cd main
CAMPAIGN=$BASE/campaign_${SHA}
QUEUE=$BASE/queue_${SHA}
STUDY=$BASE/study_approval_${SHA}.json
ALLOCATION=$BASE/allocation_approval_${SHA}.json
python scripts/exp96_campaign.py prepare --staged \
  --root "$CAMPAIGN" --queue "$QUEUE" --commit "$SHA" --approval "$STUDY"
python - "$QUEUE/queue.json" "$SHA" <<'PY'
import json, sys
queue = json.load(open(sys.argv[1]))
assert queue['commit'] == sys.argv[2]
assert len(queue['parents']) == len({p['id'] for p in queue['parents']}) == 96
print('frozen 96-parent queue:', sys.argv[1])
PY
sha256sum "$QUEUE/queue.json" "$CAMPAIGN/exp96_plan.json"
python scripts/exp96_campaign.py pending-command \
  --queue "$QUEUE" --approval "$ALLOCATION" \
  --cpus 16 --memory-gb 64 --wall-minutes 2880 > "$BASE/pending_command_${SHA}.sh"
cat "$BASE/pending_command_${SHA}.sh"
# Review the printed command and 48-A100-GPU-hour maximum before executing:
bash "$BASE/pending_command_${SHA}.sh"
```

The last command submits **one** normal-priority one-A100 allocation: 16 CPUs,
64 GiB, 48 hours maximum. It has no `--hold`, `--begin`, or dependency. It can
leave the queue at any time. These resource values must match the later explicit
allocation approval and actual Slurm assignment; otherwise startup exits. If
these values are not the resources the owner authorizes, change the
`pending-command` arguments **before** submitting. The startup receipt, stdout,
and stderr live under `$QUEUE/slurm/`.

While the job is pending, complete the external study and allocation approval
JSONs from `configs/exp96_study_approval.template.json` and
`configs/exp96_allocation_approval.template.json`. The study approval must bind
the final SHA, 96 parents, the approved ten-target method and calibration,
cross-seed plan, allocation/storage decision, and a SHA-256-verified **A100**
`smoke.json` with `PASS` at the same source SHA. The allocation approval must
bind `sha256sum "$QUEUE/queue.json"`, `sha256sum "$STUDY"`, and the exact 1-GPU/16-CPU/64-GiB/2880-minute
request, along with approved measured admission profiles. Do not create
`approved: true` or decision references without the corresponding owner
decisions. Run `python scripts/exp96_campaign.py allocation-command --queue
"$QUEUE" --approval "$ALLOCATION"` as a read-only final validation once both
files exist; it prints a command and submits nothing. If they are still missing
when Slurm starts the queued job, the allocation exits 3. A later attempt
requires a **new** `sbatch` submission, using the same immutable queue only if
the source and plan still match.

To inspect or cancel the submitted job before it starts:

```bash
JOB=<job-id-from-sbatch>
squeue -j "$JOB" -o '%i %j %T %R %S'
scontrol show job "$JOB" | grep -E 'JobState=|Reason=|NumNodes=|NumCPUs=|TRES=|TimeLimit='
scancel "$JOB"  # only after deciding to cancel; do not run for inspection
sacct -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
find "$QUEUE/attempts" -name start.json -print 2>/dev/null
find "$QUEUE/ledger" \( -name claim -o -name completed.json \) -print 2>/dev/null
find "$CAMPAIGN" -name DONE -print
```

For a never-started or startup-blocked allocation, the final three searches
must be empty. A `BLOCKED` startup receipt and an empty attempt/ledger search
confirm that no official parent was admitted. If a parent **did** start, keep
its evidence and use the ordinary ledger reconciliation; do not erase or reuse
an output directory. If Kaggle finds a source defect, cancel while pending,
publish a corrected source revision, and create a new campaign and queue at
that revision. Kaggle T4 success can check code portability but cannot approve
A100 memory/concurrency, scientific calibration, or official parent launch.
