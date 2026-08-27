---
exea_version: 1
owner_email: you@example.com
github_username: kaves29
# The runner injects your token as $EXEA_GITHUB_PAT at run time.
# You submit it once at reserve.exealabs.org/dashboard.
# NEVER write the real token in this file — this repo is public.
github_pat: ${EXEA_GITHUB_PAT}
# Where checkpoints live, relative to the repo root. This directory is what
# gets snapshotted at 13:00 and restored the next day.
checkpoint_dir: ./checkpoints
---

# exea.md

Angle 1 training: baseline SimBa critic (D2W512), quadruped-run, seed 108,
1,000,000 env steps, with critic-degradation / pathology-propagation onset
tracking enabled — feeds `results/ledgers/angle_1/architectures/D2W512/`,
which every downstream angle (2A, 2B) in this repo depends on.

Fill in `owner_email` above before pushing this file — it was left as a
placeholder deliberately, since this repo (and this file) is public.

## SETUP

Runs once per session, before START or RESUME. Installs dependencies and
prints which JAX device(s) are visible so a silent CPU fallback on a booked
GPU node is caught immediately rather than discovered three hours later.

```bash
cd SparseNetwork4DRL
pip install -r requirements.txt
python -c "
import jax
devices = jax.devices()
print('[exea setup] JAX devices:', devices)
if not any(d.platform in ('gpu', 'cuda', 'tpu') for d in devices):
    print(
        '[exea setup] WARNING: no GPU/TPU device detected - training will '
        'run on CPU and may be far too slow to make real progress in a '
        '3-hour window. requirements.txt does not pin jax/jaxlib directly '
        '(only pulled in transitively via flax==0.8.4), so a fresh '
        'environment can silently resolve to CPU-only jax. If this is '
        'unexpected here, install a CUDA-matched build explicitly, e.g. '
        'pip install -U \"jax[cuda12]==0.4.34\", then re-run SETUP.'
    )
"
```

## START

Runs on the FIRST session only, when no checkpoint exists yet. Launches
Angle 1 training via this repo's existing experiment registry (run.py).

Checkpointing note: `run.py`'s own `--checkpoint_start_frac` default is 0.4
(no checkpoint is written until 40% of training has elapsed) and
`--checkpoint_interval` defaults to 100,000 interaction steps — both far too
coarse for a 3-hour session budget, so both are overridden explicitly below.
`--checkpoint_interval 2000` is a conservative starting point, NOT a
calibrated value — this machine's actual interaction-steps/sec on your
booked GPU is unknown; if a quick benchmark shows you're running much
faster or slower, adjust this so a checkpoint lands every few minutes, not
every few seconds or every hour.

`angle_1.py`'s own training loop already detects an existing checkpoint
(`$EXEA_CHECKPOINT_DIR/meta.pkl`) and resumes from it automatically when
present — so this command is intentionally IDENTICAL to the one in RESUME
below; the underlying script, not this file, decides which path to take.

```bash
cd SparseNetwork4DRL
python run.py --experiment angle_1 --config_path ./configs --config_name base_sac \
  --overrides env_name=quadruped-run \
  --overrides seed=108 \
  --overrides critic_num_blocks=2 \
  --overrides critic_hidden_dim=512 \
  --overrides critic_degradation=true \
  --overrides pathology_prop=true \
  --checkpoint_dir "$EXEA_CHECKPOINT_DIR" \
  --checkpoint_interval 2000 \
  --checkpoint_start_frac 0
```

## RESUME

Runs on EVERY session after the first. Identical to START — see the note
above: `angle_1.py` loads `$EXEA_CHECKPOINT_DIR/meta.pkl` if present and
continues from `interaction_step + 1`, restoring agent params, optimizer
state, observation-normalization statistics, replay buffer, and RNG state.
Verified locally: killed mid-run at interaction_step 5000, re-invoked this
exact command, and it resumed at 5001 with update_step/update_counter
exactly consistent with uninterrupted training (not from zero).

```bash
cd SparseNetwork4DRL
python run.py --experiment angle_1 --config_path ./configs --config_name base_sac \
  --overrides env_name=quadruped-run \
  --overrides seed=108 \
  --overrides critic_num_blocks=2 \
  --overrides critic_hidden_dim=512 \
  --overrides critic_degradation=true \
  --overrides pathology_prop=true \
  --checkpoint_dir "$EXEA_CHECKPOINT_DIR" \
  --checkpoint_interval 2000 \
  --checkpoint_start_frac 0
```

## STOP

Runs at 13:00 PST. Sends SIGTERM and waits for a clean exit. `angle_1.py`
has no special-case signal handling — its safety net is purely the
`--checkpoint_interval` cadence above (at most ~2000 interaction steps of
progress since the last checkpoint is lost on a mid-run kill), which is why
that interval must actually be tuned to this machine's real throughput.

```bash
kill -TERM "$EXEA_JOB_PID" 2>/dev/null || true
wait "$EXEA_JOB_PID" 2>/dev/null || true
```

## SAVE

Runs after STOP. Commits only the canonical, small evidence artifacts this
repo already treats as immutable research output — never the checkpoint
directory itself (that rides in the disk snapshot, per the rules above):

- `results/ledgers/` — the onset-event CSV ledger (utils/onset_ledger.py),
  the canonical source every downstream angle reads from.
- `results/metrics/` — persisted per-run interaction-step metrics used for
  post-hoc onset detection (analysis/metrics_store.py).

```bash
cd SparseNetwork4DRL
git add results/ledgers results/metrics
git commit -m "exea: session $EXEA_SESSION_COUNT angle_1 D2W512 quadruped-run seed108" || echo "nothing to commit"
git push origin HEAD
```
