# Option D: separate ten-target follow-up, pending authorization

The **authorized original A/B/C plus D2 job remains pinned to** `18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af`, one A100/16 CPUs/64 GiB, 4 h 30 min. This branch is for a *different*, unsubmitted Option D job. It does not modify or substitute the authorized source. The original D1 stage has a 200-minute cap; adding five more paired 2,000-step targets to each of its 100 pairs would approximately double its dominant fitting work, with no measured spare capacity. Therefore D cannot reliably fit in the 4.5-hour original allocation.

| Procedure | Targets and steps | Target and fit | Timing during RL | Statistic and decision |
|---|---|---|---|---|
| A, approved project | 5 independently keyed target functions × 1,000 steps | SAC critic/replay, each critic's mean offset, `sin(1e5 * fresh critic prediction)`, critic AdamW, new fit optimizer state | 20 even checks across each parent budget | Paired fresh-minus-current P, five-round IQM, 10k-resample 95% percentile-bootstrap lower bound > 0; two consecutive checks for an eligible trigger. |
| B, exploratory project | Same five functions and fits as A | Same | Same offline replay | Same lower bound > fixed pooled-null p95 `0.0009080017606417335`. This is not active production methodology. |
| C, exploratory project | Same five functions, optimizer continued to 2,000 steps | Same | Same offline replay if full trained trajectory existed | Five-round IQM and project's existing bootstrap lower bound > 0. |
| D, **Lyle-inspired** | First five C functions plus five new independently keyed functions, each fit to 2,000 steps | Same adapted SAC setup; not an exact reproduction of the paper's architecture/data | Fresh-null only in prepared follow-up | Ten-target descriptive IQM and, separately, the **project's** 10k-resample bootstrap decision. No claim that Lyle used this trigger. |
| Lyle et al., 2023, *Understanding Plasticity in Neural Networks* | 10 randomly sampled target functions, optimizer run 2,000 steps each, per Appendix A.2 quote preserved in `docs/exp12_decisions.md` | Section 3.1 quote: `g(x)=a+sin(10^5 f(x;ω0))`, with freshly sampled `ω0` and `a` the probed network's mean prediction; baseline `b` is target variance; use the network's optimizer from current parameters | **Unverified from available excerpts**; do not equate the project's 20-check schedule with the paper's | `P(θ)=b−E[final loss]`, and paper's plasticity-loss sign is `P(θ_t)−P(θ_0)`; project reports fresh-minus-current, the opposite sign. The cited passages do **not** specify our bootstrap CI or binary trigger. |

The project repository transcribes paper passages and cites arXiv `2303.01486` in `docs/exp12_decisions.md:423–452`. Direct access to the original PDF and an official implementation was unavailable in this environment (external HTTP access returned 403); evaluation frequency, exact optimizer hyperparameters in the paper, and any official code path remain **unverified**. The table only treats the quoted paper text as checked repository evidence. Do not describe D as a reproduction.

## Independence and matching

`probe_key(seed, check_index, round_index)` folds the round index into a dedicated PRNG stream. `probe_round` then splits the key into a **target initialization key** and a **minibatch-index key**; `probe_rng` also includes the round index for the replay pool. Thus rounds 0–4 already generate five distinct pseudorandom teacher functions, not five minibatch repeats of one teacher. The additional rounds use indices 5–9 and unique teacher keys. A focused CPU test verifies ten different key pairs and different generated target arrays. Within each round, fresh/current critics share the same teacher, replay pool and minibatch indices, while their fitted parameters and AdamW states remain distinct. The first 1,000 optimizer steps retain the existing index stream; the next 1,000 continue the live AdamW state.

To avoid refitting the first five targets for every pair, the follow-up runs **only rounds 5–9** for the same 100 seed-991 null pairs. First it reruns original pair 0's first five rounds and requires bitwise equality of both critics' curves/endpoints, identical bootstrap bounds, and matching JAX/CUDA/runtime metadata. A mismatch aborts before the 100-pair extra stage. It then combines the additional 2,000-step FP32 paired losses with the completed first-five `null_991/pair_*.json` from the original job. The combiner requires exact clean source identities, all 100 unique pair files, matching seed/check/pair IDs, distinct expected teacher keys, finite losses, and original result checksums. It reports both ten-target **mean** and project IQM descriptively, then the project's bootstrap verdict in a separate field. Source commits of both halves remain explicit; they are never represented as one production run.

This follow-up is **not submitted or approved**. It requests one A100/16 CPUs/64 GiB, with a **4-hour Slurm maximum**, a 10-minute reference-pair cap and a 200-minute extra-fitting cap. Estimated elapsed time is **1.5–3 hours**, extrapolated from the original D1 stage's unmeasured per-size share; maximum allocation cost is 4 A100-hours. It must await a complete, integrity-verified original 100-pair result and separate owner authorization. If that original job is incomplete, the follow-up preflight refuses to run. It does not assess trained-critic sensitivity or actual Exp2 trigger eligibility.

After a complete original result and **separate authorization**, use a new clean detached checkout of the published Option D branch. Set `FIRST_FIVE_JOB_ROOT` to the exact original output root from the preceding command; the wrapper verifies its receipt, all recorded file checksums, 100 pair files and original source commit. The Option D output root must be new:

```bash
set -euo pipefail
D_REV=REPLACE_WITH_PUBLISHED_OPTION_D_COMMIT
D_CHECKOUT=/work/hdd/biqc/skaveti1/exp12_exploration_checkouts/$D_REV
FIRST_FIVE_JOB_ROOT=/work/hdd/biqc/skaveti1/exp12_exploration_results/18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af_consolidated_d1_d2
OPTION_D_ROOT=/work/hdd/biqc/skaveti1/exp12_exploration_results/${D_REV}_option_d_extra_targets
test ! -e "$D_CHECKOUT" && test ! -e "$OPTION_D_ROOT"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$D_CHECKOUT"
git -C "$D_CHECKOUT" checkout --detach "$D_REV"
test "$(git -C "$D_CHECKOUT" rev-parse HEAD)" = "$D_REV"
test -z "$(git -C "$D_CHECKOUT" status --porcelain --untracked-files=all)"
cd "$D_CHECKOUT/main"
export EXPECTED_COMMIT="$D_REV" FIRST_FIVE_JOB_ROOT OPTION_D_ROOT
sbatch --export=ALL --output="${OPTION_D_ROOT}.slurm-%j.out" \
  --error="${OPTION_D_ROOT}.slurm-%j.err" scripts/sbatch_exp12_option_d_followup.sh
```

## Original authorized job: commands to execute on Delta

Delta could not be accessed from this Cloud workspace: there is no Delta mount, and SSH to the login host failed name resolution. These commands run **only the original 4.5-hour job**, after the user verifies Delta connectivity and stages copies of both historical inputs. They do not submit Option D.

```bash
set -euo pipefail
REV=18c76da70c7e7c53bb01cae0afec8b8ae4dfb3af
CHECKOUT=/work/hdd/biqc/skaveti1/exp12_exploration_checkouts/$REV
HOPPER_CAPTURE_ARCHIVE=/work/hdd/biqc/skaveti1/exp12_exploration_inputs/hopper_capture_22817406.tar.gz
NULL_991_JSONL=/work/hdd/biqc/skaveti1/exp12_exploration_inputs/null_pairs_D4W1024_seed991.jsonl
PROBE_DIAGNOSTIC_ROOT=/work/hdd/biqc/skaveti1/exp12_exploration_results/${REV}_consolidated_d1_d2
test -f "$HOPPER_CAPTURE_ARCHIVE" && test -f "$NULL_991_JSONL"
echo 'e96cb06b953ed1acef29850861b5371c91766c58e1b4e0eb9a0b86acec0d4868  '"$HOPPER_CAPTURE_ARCHIVE" | sha256sum -c -
echo 'd6dc126939d82f145f1ea1b4be60d87a36751970c365bdaf5ae5d8f5333ca84e  '"$NULL_991_JSONL" | sha256sum -c -
test ! -e "$CHECKOUT" && test ! -e "$PROBE_DIAGNOSTIC_ROOT"
git clone --no-checkout https://github.com/kaves29/scaling-drl-research.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach "$REV"
test "$(git -C "$CHECKOUT" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$CHECKOUT" status --porcelain --untracked-files=all)"
cd "$CHECKOUT/main"
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate scaling-drl-py31213
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1
python -B -c 'import sys; from scripts.exp12_launch_plan import require_pinned_stack; assert sys.version_info[:3] == (3,12,13); require_pinned_stack()'
python -B -m unittest -q tests.test_compare_probe_decisions tests.test_paired_probe_budget
unset JAX_PLATFORM_NAME
scontrol show partition gpuA100x4 | grep -E 'PartitionName=|MaxTime=|State='
sacctmgr -nP show associations where user="$USER" account=biqc-delta-gpu
df -h "$(dirname "$PROBE_DIAGNOSTIC_ROOT")" "$HOPPER_CAPTURE_ARCHIVE"
export EXPECTED_COMMIT="$REV" HOPPER_CAPTURE_ARCHIVE NULL_991_JSONL PROBE_DIAGNOSTIC_ROOT
mkdir -p "$(dirname "$PROBE_DIAGNOSTIC_ROOT")"
sbatch --export=ALL --output="${PROBE_DIAGNOSTIC_ROOT}.slurm-%j.out" \
  --error="${PROBE_DIAGNOSTIC_ROOT}.slurm-%j.err" \
  scripts/sbatch_exp12_paired_probe_budget.sh
```

If either input is absent, **stop before `sbatch`**. Search only the owner's original Delta evidence area for the named artifact, verify the candidate SHA-256, and copy it to the required input path; never edit the source. A missing or unequal hash is a failed preflight, not a reason to bypass the check. The output root, Slurm stdout/stderr, `exit_status.txt`, stage exit codes, `null_991/summary.json`, D2 round/summary files, telemetry and `SHA256SUMS` are the evidence to preserve.

Interpret A/B/C fresh-null counts with their exact intervals as **per-check** observations under one seed, not true-positive accuracy or production two-check trigger rates. Verify the historical 1,000-step reproduction before attributing D2 changes to fitting budget. A/C/B adoption, a lower range gate, an exclusion, and Option D submission remain separate owner decisions.
