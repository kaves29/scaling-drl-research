# Exp1/Exp2 remaining qualification (2026-10-11)

This is a decision aid, not launch approval. Pilot job 22800749 and its
`7d0883cfa99d27fed062d77a68cbbfa27e560061` checkout are untouched. Use
the exact eventual campaign SHA in every new GPU validation.

## Scientific findings from existing evidence

The uploaded seed-990 and seed-991 D4W1024 JSONL files have SHA-256
`89846b78306e84b337a6919ae5a1b20809913f7b0bc10e1d32109c6ca8137b5d`
and `d6dc126939d82f145f1ea1b4be60d87a36751970c365bdaf5ae5d8f5333ca84e`.
Each has 100 unique, valid pairs. Independent float32 subtraction, middle-three
IQM, seeded 10,000-draw bootstrap and strict positive lower-bound checks
reproduce all rows exactly: 13 and 6 firings. Exact enumeration of the 5^5
possible round resamples changes neither count. The seed-991 file records a
clean `9d6a82d4aef81299901177865e383640439cde2a` A100 run; the original
JSONL omits embedded source provenance, supplied separately in the Block B
archive. Both fail the approved maximum of five firings per 100 pairs.

Within a pair the two critic initializations and warm-up replay are fixed.
Each of five rounds samples a new pool from that one replay and uses new target
and minibatch keys. The bootstrap resamples five paired round differences;
it estimates round variation conditional on the initializations and replay,
not variation across new pairs or replay buffers. Different initializations
can yield different fitting residuals even before training. This is a
plausible source of conditional pair effects, not a demonstrated code bug.
The predeclared within-original W p=.0693 and cross-seed X p=.02285 do not
cross their .01 diagnostic boundary; one pair fired in both samples. The
exploratory original-round shuffle p≈.003 suggests pair/replay clustering,
without identifying its cause. Both files reuse numbered initialization keys
across seeds. The reverse-tail counts (6 and 11) and no persistent re-firing
provide no evidence for a fixed directional arithmetic bug. The independent
fresh pairs do not measure repeated consecutive checks on one evolving critic.

The Block B report provides the D4W1536 hopper-hop five final scores,
baselines and losses, but no original stepwise fitting curves, teacher pools,
warm-up replay or checkpoint. At the approved 25,600 pool,
IQM(P)=.3008451164, IQM(b)=.4994128446 and P/b=.6023976348. The reported
1,600 and 6,400 ratios are .98675 and .84510. One thousand 256-element
minibatches mean nominal draws per pool item fall from 160 to 40 to 10 as
the pool grows; the fit budget remains fixed. Baseline b stays near .5 while
final loss rises to about .2 at 25,600. All fifteen saved round scores equal
float32(b − final loss), and all three reported IQM ratios recompute exactly.
This establishes configured-pool
fresh-critic underfitting under the approved finite fit, not trained-critic
plasticity degradation. Architecture scaling, optimization, replay/target
complexity and GPU numerics remain competing explanations. Final summaries
cannot reconstruct intermediate curves or the exact original GPU targets.
The existing `capture_exp12_range_evidence.py` can retain these arrays in a
separately authorized unchanged-setting replay if the original raw artifacts
cannot be recovered. No optimizer, target, pool or threshold change is implied.

## Minimal GPU qualification; commands are for future authorization

Use a fresh detached checkout; never reuse the pilot checkout. All outputs
below are new absolute directories outside Git. Set `FINAL_SHA` to the
reviewed campaign commit, then verify `git rev-parse HEAD` and clean status.
The Slurm account/partition are in the reviewed scripts. Put `QUAL_ROOT` on
the same shared filesystem planned for the durable campaign queue; a pass on
a different mount does not establish its atomic claim behavior.

```bash
cd "$FINAL_CHECKOUT/main"
test "$(git rev-parse HEAD)" = "$FINAL_SHA"
test -z "$(git status --porcelain --untracked-files=all)"
QUAL_ROOT="/work/hdd/biqc/skaveti1/exp12_qualification/$FINAL_SHA"
mkdir -p "$QUAL_ROOT/slurm"

# Gate G1: two-node shared-mount atomic claim, recovery and binding.
# 8 A100, 32 CPUs, 128 GB across nodes; expected 2–5 min, 10 min cap,
# at most 1.33 A100-h.
EXPECTED_COMMIT="$FINAL_SHA" EXP12_PREFLIGHT_ROOT="$QUAL_ROOT/fsbind_01" \
 sbatch --export=ALL --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 \
 --cpus-per-task=16 --mem=64G --time=00:10:00 \
 --output="$QUAL_ROOT/slurm/fsbind_%j.out" \
 --error="$QUAL_ROOT/slurm/fsbind_%j.err" \
 scripts/sbatch_exp12_dynamic_delta_preflight.sh

# Gate G2: three serial versus three concurrent architecture profiles.
# Required only before approving more than one process per GPU. The scheduler
# admission cap defaults to one, so a conservative launch need not await G2.
# 1 A100, 16 CPUs, 64 GB; 20–45 min planning estimate, 60 min cap,
# at most 1 A100-h. Actual concurrent throughput is unmeasured.
EXPECTED_COMMIT="$FINAL_SHA" EXP12_PACK_ROOT="$QUAL_ROOT/pack_01" \
 sbatch --export=ALL --cpus-per-task=16 --mem=64G --time=01:00:00 \
 --output="$QUAL_ROOT/slurm/pack_%j.out" \
 --error="$QUAL_ROOT/slurm/pack_%j.err" \
 scripts/sbatch_exp12_dynamic_pack_profile.sh

# Gate G3: one full-setting cold identity certificate per scaled size/suite.
# Start with ONE case; continue only if source, backend, Check 1 and bitwise
# certificate pass. Each: 1 A100, 16 CPUs, 64 GB; 1–4 h uncertain,
# 4 h cap (≤4 A100-h). Earlier cases exceeded a one-hour cap.
case_id=D4W1024_dog-run
cache_mode=cold
identity_root="$QUAL_ROOT/identity/${case_id}_${cache_mode}_01"
mkdir -p "$QUAL_ROOT/identity"
EXPECTED_COMMIT="$FINAL_SHA" IDENTITY_CASE="$case_id" \
 IDENTITY_CACHE_MODE="$cache_mode" IDENTITY_ROOT="$identity_root" \
 HB_ENV="$APPROVED_HB_ENV" \
 sbatch --export=ALL --cpus-per-task=16 --mem=64G --time=04:00:00 \
 --output="$QUAL_ROOT/slurm/identity_%j.out" \
 --error="$QUAL_ROOT/slurm/identity_%j.err" \
 scripts/sbatch_exp12_identity_case.sh
```

Remaining cold cases, each submitted separately after reviewing the first:
`D4W1536_dog-run`, `D4W1024_myo-key-turn`, `D4W1536_myo-key-turn`,
`D4W1024_h1-run-v0`, `D4W1536_h1-run-v0`. Set `case_id` and a fresh
`identity_root` before each submission. Six cold caps total ≤24 A100-h;
prior Block B cases exceeded a one-hour limit, so four hours is a request,
not a measured completion guarantee. Warm-cache repetitions (up to six, another
≤24 A100-h) are conditional on the approved cache qualification requirement;
the earlier full-width cold/warm PASS supports only its source and dog-run
engineering case, not six final-source certificates. Stop on timeout or a
failed certificate; inspect evidence before further cases.

G1 PASS requires two distinct nodes, four A100 CUDA devices and CPU backend
per worker, at least 16 bound CPUs, one atomic link and claim winner, preserved
retired claim, `verdict.json` PASS and exit 0. G2 passes as a *measurement*
when all six profiles, logs and `comparison.json` exist, rates are finite,
all processes exit 0, and device/host memory is within the reviewed admission
profile. Its end-to-end serial/mixed wall comparison includes identical
diagnostic warmup and probe work, but is only one workload observation. Only
enable mixed packing if observed total throughput improves and memory
headroom holds; this does not qualify ceilings 7/4/2 automatically.
An OOM, nonfinite rate, missing process or timeout means packing remains
unqualified; single-process placement can still be reviewed separately.
G3 requires exact source/backend, complete Check 1 and bitwise identity
certificate with no differences. Existing reduced-budget Block B attempts and
the earlier full-width dog gate cannot substitute for other suites or source
revisions. All jobs above are diagnostics; none submits a parent or injected arm.

## Read-only pilot examination after job 22800749 ends

Resolve `PILOT_PLAN` and its one `entries[0].checkpoint_dir` from the actual
approved submission record. Do not infer paths from historical examples.

```bash
sacct -P -j 22800749 \
 --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,TotalCPU,MaxRSS
PILOT_RUN=$(python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert len(p["entries"])==1; print(p["entries"][0]["checkpoint_dir"])' "$PILOT_PLAN")
PILOT_SHA=7d0883cfa99d27fed062d77a68cbbfa27e560061
python scripts/exp12_launch_plan.py validate-run --run "$PILOT_RUN" \
 --commit "$PILOT_SHA" --out "$NEW_EXTERNAL_AUDIT/pilot_validation.json"
cat "$PILOT_RUN/DONE" "$PILOT_RUN/run_metadata.json"
cat "$PILOT_RUN/state/LATEST" "$PILOT_RUN/probes/probe_checks.csv"
if test -e "$PILOT_RUN/fork/FORK_READY"; then
  cat "$PILOT_RUN/fork/fork.json"
else
  printf '%s\n' 'NO NATURAL FORK RECORDED'
fi
```

Compare DONE, terminal checkpoint counters, probe-check index and Slurm
elapsed time; inspect saved launch metadata for CPU/CUDA, complete state,
resume launches and W&B run IDs. Inspect GPU-utilization samples only if
the pilot actually recorded them; `sacct` CPU/RSS fields do not measure GPU
utilization. A `FORK_READY` marker needs the natural-trigger and Check 1
validation, not just file existence. Inspect a positive-control JSON only if
one was actually generated. A pilot-based positive-control result has source
`7d0883c...`; the dynamic Exp2 queue currently requires evidence from its
exact campaign source, so the pilot cannot silently freeze m for a later SHA.

## Capacity and decisions

The 195 parents comprise 35 million interactions per architecture, 105
million total. Each architecture contributes 12.5 million DMC-hard, 2.5
million DMC-medium, 10 million MyoSuite and 10 million HumanoidBench
interactions. The historical Block B profiles measured dog-run single-job
training rates 112.5/65.8/46.3 interactions/s for D2/D4W1024/D4W1536;
MyoSuite myo-key-turn rates were 117.5/66.1/45.1. Homogeneous packing raised
D2 aggregate throughput (four jobs: 261.3/s), but two D4W1536 jobs reached
only 49.8/s versus 46.3/s alone, before probes/evaluations. Mixed packing,
HumanoidBench and final-source full-run rates remain unmeasured. The prior
conditional non-HB training estimate is 314.12 A100-h if all DMC and Myo
cells match those representatives. Its optimistic HB-equals-dog and pooled
probe proxy gives ~466 A100-h baseline or ~635 A100-h if all scaled parents
fork late; both omit startup, evaluation and I/O. A separate uniform use of
the newer 42.44/s D4W1536 dog-run window gives ~687 A100-h parent training.
These conflicting scenarios show that A+B's 504 A100-h may suffice for an
optimistic Exp1-only workload, but cannot be defended as enough for Exp1+Exp2.
Under the 687 h uniform scenario ~183 more physical A100-h are needed before
overhead; under the 635 h all-trigger proxy ~131 more are needed before omitted
overhead. Neither is a lower bound or measured forecast. D4W1536 compute and
long HumanoidBench budgets are likely critical; CPU-heavy suites may bind
first. The shared ledger retains validated completions across later
allocations, so an additional approved allocation draws only pending work.

| Blocker | Resolution | GPU? | Owner science decision? | Scope |
|---|---|---|---|---|
| D4W1024 null 13/100 and 6/100 | Disposition failed ≤5/100 gate; any new criterion is an explicit amendment | No for disposition | Yes | D4W1024; fixed 195 grid if unamended |
| Hopper D4W1536 P/b=.6024 | Recover original curves/targets or approve unchanged-setting evidence capture; disposition failed range gate | Only if original arrays absent | Yes to exclusion/criterion change | Hopper D4W1536; full gate if unamended |
| Shared mount and binding | G1 | Yes, ≤1.33 A100-h | No | Dynamic A/B allocations |
| GPU packing | G2 before approving admission cap above one; single-process fallback is explicit | Yes if packing desired | No | Capacity, not scientific estimand |
| Six identity cells | G3 cold, warm as required by approved cache policy | Yes, ≤24 A100-h cold cap | No unless failures demand reinterpretation | Scaled parents/Exp2 |
| Positive control and m | Review natural final-source dev fork and unchanged positive-control run; owner freezes m | Likely yes; pilot source differs | Yes | Grid prerequisite; injected Exp2 |
| Numeric bounds and source freeze | Review existing evidence, record approved decisions | Conditional | Yes | Confirmatory grid |
| A+B capacity | Pilot/per-suite/packing rates, then approve resources or further allocation | G2/pilot evidence | Resource approval | Grid and Exp2 |

No approval is inferred by this document or by successful diagnostic execution.
