# Exploratory Method C construct controls

This plan is separate from the approved Method C probe. It does not alter a
training run, its checkpoints, its trigger, or the 96-run protocol. Source of
the seven pilot runs: `f8619dd5f494399d9febc79fc9092010234c714d`.
The archived pilot evidence is not a replacement for the full checkpoint and
replay states required by these controls.

## Lyle et al. (2023) source comparison

The repository records verbatim excerpts of *Understanding Plasticity in
Neural Networks*, arXiv:2303.01486, sections 3.1 and appendix A.2 in
`docs/exp12_decisions.md` lines 423–458. The current cloud proxy rejected
direct arXiv PDF/HTML and GitHub search requests (HTTP 403). Consequently,
the excerpts below are a secondary local record of the paper, and the
official implementation has **not** yet been independently inspected. No
claim of line-by-line code fidelity is made. A copy of the paper appendix and
the official repository/ref is needed to complete that part of the review.

| Component | Lyle paper / archived appendix evidence | Current Method C source | Classification |
|---|---|---|---|
| Random fitting task | `g(x)=a+sin(10^5 f(x;ω0))`, with a fresh `ω0` from the initial-parameter distribution | `probe.py:_base_targets`, own randomly initialized scalar critic; scale is **inside** sine | Faithful task family; continuous state-action input is a SAC adaptation |
| Offset | `a` is the probed network's mean prediction | `probe.py:probe_round` and `_fit` use each critic's mean over the matched pool | Faithful to the documented paper wording, but current and fresh have different complete numerical targets when offsets differ. The shared sine component and target variance are equal. This limits a strict same-target interpretation. |
| Plasticity score | `P(θ)=b−E_ℓ[ℓ(θ*)]`, with `b` target variance for squared loss; appendix says optimize for 2,000 steps | `probe.py:probe_round`, `summarize`; full-pool final MSE, variance baseline, 2,000 continuous AdamW steps | Faithful score and current budget; finite pool and optimizer are SAC adaptations |
| Target draws | Appendix A.2 records 10 independent random target functions | `probe.py:run_probe` uses five independent `probe_key(seed,check,round)` teachers and replay pools | Approved experimental adaptation, less Monte Carlo information than the cited design; Lyle's result does not justify a five-round bootstrap's operating characteristics |
| Optimizer | Appendix says use the network's optimizer | `probe.py:critic_optimizer`, fresh AdamW state per fit; parameters/state carry from 1,000 to 2,000 | Faithful optimizer *type* and continuous fitting; resetting trained optimizer moments is a declared comparability choice, not demonstrated to reproduce the paper's state handling |
| Inputs and fitting loss | Paper's sampled prediction objectives over its RL replay setting | 25,600 sampled replay `(s,a)` pairs, current observation normalization, 256-sample minibatches, final full-pool MSE | Deliberate SAC continuous-control adaptation; pool size, replay distribution, normalization and stochastic mini-batches require construct validation here |
| Loss direction and summary | Paper writes `P(θt)−P(θ0)` (negative under loss) | `summarize` reports `P(fresh)−P(trained)` so positive indicates loss; IQM of five paired differences | Deliberate sign/aggregation adaptation; sign changes interpretation, not fitted scores |
| Decision rule | The archived paper excerpts define a measurement, not this intervention eligibility test | `trigger.py` uses 10,000-resample 95% paired IQM percentile CI, zero lower-bound threshold and two consecutive scheduled checks | Project-specific operational rule. Its false-firing rate, temporal dependence and sensitivity cannot be inherited from Lyle |
| Network/setting | The paper's value-learning networks and objectives | SAC continuous-control critic, residual SimBa sizes, separate fresh reference saved at warmup | Scientific transfer, not a faithful replication; an observed fitting gap alone does not isolate intrinsic plasticity |

The official code may clarify optimizer state, target sampling and numerical
details not settled by the available paper excerpts. Until reviewed, those
points are **unverified**, not presumed matches. This plan proposes no
scientific change.

## Existing evidence and required artifacts

The uploaded seven-run pilot archive contains check summaries and compressed
probe curves. Myo Key Turn fresh 2,000-step final MSE/b rose from early to
late checks (roughly 0.006→0.140 at D2W512, 0.008→0.197 at D4W1024, and
0.008→0.273 at D4W1536). These are **different replay pools and teachers**;
they establish neither a state effect nor a replay effect. Trained late Myo
critics had still larger ratios, but a finite-budget fitting gap is not by
itself proof of intrinsic plasticity loss.

The archive excludes `agent_ckpt/`, `obs_rms.pkl`, `buffer.npz`,
`buffer_meta.pkl`, `meta.pkl`, `LATEST`, `fresh_critic/`, and per-check
`probes/check_02.npz` / `check_20.npz`. The original pilot root is:

```text
/work/hdd/biqc/skaveti1/exp12_prod_pilot7_20261014T213210Z
```

The required run roots are its `checkpoints/run5_D4W1024_myo-key-turn_seed1`
and `checkpoints/run7_D4W1536_myo-key-turn_seed1`, with `fork/state/LATEST`
at 50,000 and `state/LATEST` at 500,000. The D2W512 run is
`checkpoints/run4_D2W512_myo-key-turn_seed1`; a 50,000 state may have been
removed because this run had no retained fork. The harness inventory mode
checks this explicitly. It will not reconstruct an early trained checkpoint
from curves or an unrelated checkpoint. No environment state is required for
replaying a *saved* replay pool and critic; it would be required to regenerate
the original training trajectory or missing replay, which this plan avoids.

## Minimal matrix and interpretation

| Order | Control | Conditions | What it separates | Stop / interpretation rule |
|---|---|---|---|---|
| 1 | Fixed-pool crossover | D4W1024 and D4W1536, 50k/500k critic × 50k/500k replay pool, matched five teachers and minibatches; fresh on both pools | Critic state, pool and interaction | Diagonal fresh/current output must bitwise reproduce saved original check-02/check-20 arrays before any crossover finding is trusted. Pool-specific normalization keeps identical network inputs within a pool; off-diagonal use of a foreign normalizer is a disclosed factor. Offsets are network-specific, so full targets are constant-shifted rather than identical. |
| 2 | Fresh/fresh negative control | Four independent parameter pairs per architecture/pool, five new teacher rounds per pair, both pairing directions recorded; D2 late and early only if a valid early state exists | Initialization asymmetry and late-pool difficulty | Descriptive exact-binomial intervals use four **pairs**, not eight reversed directions. Four pairs cannot establish a 5% population false-positive rate. Check-02 and check-20 are not consecutive; this does not estimate the production two-check false-trigger probability. |
| 3 | Optimization conditioning | One matched round per D4 architecture/pool, fresh/early/late parameters | Start error vs speed; gradient, parameter, target-scale diagnostics | Separate instrumented scan; do not substitute its values for the official score or trigger. Interpret observation-normalizer mismatch and offset effects. |
| Conditional | Target scale and 4k/8k fitting | Late pool for selected D4 cases only after reviewing core results | Probe-task and finite-budget sensitivity | Separate exploratory run and owner-approved GPU budget. The first 2k draws match Method C, AdamW state continues, and no official 2k endpoint changes. |

The core requests about 272 full 2,000-step fits (up to 312 if D2 early is
retained). A prior A100 A/B/C job measured **only the incremental second
1,000 steps** at 1.848 s per D4W1024 fit and 3.469 s per D4W1536 fresh fit.
Those measurements suggest approximately 21 minutes of raw D4 fitting for
the core, before compilation, checkpoint I/O, profiling and instrumented
extra work. D2 and trained-fit times are not measured here. Expected elapsed
time is provisionally 1–3 hours; the hard request/cap is one A100 for six
hours (≤6 allocated GPU-hours). The script exits when finished; it does not
consume six hours merely because they were requested. Sixteen CPUs and 64 GiB
host RAM allow replay and checkpoint restoration; 40 GiB A100 memory is
expected to suffice based on pilot fitting, but this exact combined harness
has no GPU memory measurement yet. A failed CUDA/memory preflight stops the
job. The optional extension/scale work is **not** in this allocation.

## Exact Delta preflight and submission, awaiting authorization

The code checkout containing this report must be a clean detached checkout of
the **published harness commit**, distinct from the immutable pilot checkout.
The account/partition are the previously used `biqc-delta-gpu` / `gpuA100x4`;
verify live `sacctmgr`/`sinfo` eligibility before submission. The pinned
Python is `/u/skaveti1/.conda/envs/scaling-drl-py31213/bin/python`.

```bash
set -euo pipefail
HARNESS_SHA=REPLACE_WITH_PUBLISHED_HARNESS_SHA
CHECKOUT=/work/hdd/biqc/skaveti1/exp12_method_c_controls/$HARNESS_SHA
PILOT_ROOT=/work/hdd/biqc/skaveti1/exp12_prod_pilot7_20261014T213210Z
LOG_ROOT=/work/hdd/biqc/skaveti1/exp12_method_c_controls/logs/$HARNESS_SHA
EVIDENCE_ROOT=/work/hdd/biqc/skaveti1/exp12_method_c_controls/evidence/${HARNESS_SHA}_core
PYTHON_BIN=/u/skaveti1/.conda/envs/scaling-drl-py31213/bin/python
test ! -e "$CHECKOUT" && test ! -e "$EVIDENCE_ROOT"
mkdir -p "$(dirname "$CHECKOUT")" "$LOG_ROOT"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$HARNESS_SHA"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$HARNESS_SHA"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
for run in run4_D2W512 run5_D4W1024 run7_D4W1536; do
  case "$run" in run4*) arch=D2W512;; run5*) arch=D4W1024;; run7*) arch=D4W1536;; esac
  test -f "$PILOT_ROOT/checkpoints/${run}_${arch}_myo-key-turn_seed1/state/LATEST"
done
test -f "$PILOT_ROOT/checkpoints/run5_D4W1024_myo-key-turn_seed1/fork/state/LATEST"
test -f "$PILOT_ROOT/checkpoints/run7_D4W1536_myo-key-turn_seed1/fork/state/LATEST"
sacctmgr -n show assoc user="$USER" account=biqc-delta-gpu format=account,partition
sinfo -p gpuA100x4 -o '%P %G %c %m %l'
cd "$CHECKOUT/main"
"$PYTHON_BIN" scripts/method_c_validity_controls.py --mode inventory \
  --run-dir "$PILOT_ROOT/checkpoints/run5_D4W1024_myo-key-turn_seed1"
sbatch --parsable \
  --export="ALL,HARNESS_SHA=$HARNESS_SHA,PILOT_ROOT=$PILOT_ROOT,EVIDENCE_ROOT=$EVIDENCE_ROOT,PYTHON_BIN=$PYTHON_BIN" \
  --output="$LOG_ROOT/slurm-%j.out" --error="$LOG_ROOT/slurm-%j.err" \
  scripts/sbatch_method_c_validity_controls.sh
```

The evidence root is new and source-specific. It contains an input-hash
manifest, per-condition curves, paired bootstrap decisions, and persistent
partial results. Slurm stdout/stderr are retained separately under
`$LOG_ROOT`. On a failed diagonal reproduction, the job stops before
interpreting the crossover. On timeout the exit is 124; completed condition
files remain but the missing conditions are **not** silently counted as
negative. On a CUDA or provenance failure the job stops. No auto-requeue or
production launch is present.

After core evidence is inspected, separately authorize and budget any
`--mode scales` or `--mode extension --extension-steps 4000|8000` runs; those
controls are not prerequisites to collecting the core evidence. Their
results cannot rewrite approved Method C outcomes.

## Readiness pending controls

The existing pilot verifies that Method C records curves and operational
checks on trained critics. It does not yet establish calibrated fresh/fresh
behavior on Myo late replay, nor isolate critic-state from replay-task
difficulty, nor prove the 2,000-step gap persists at longer budgets. The
planned 96-run interpretation is therefore construct-validity limited until
the core controls are examined. The controls can motivate a *separate* owner
methodology decision; they do not themselves change the approved protocol or
prove that plasticity causes actor performance differences.
