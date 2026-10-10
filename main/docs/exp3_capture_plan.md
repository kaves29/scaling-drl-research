# Prospective Exp3 artifact capture during Exp1/Exp2 source runs (2026-10-10)

Branch `claude/exp3-pilot-review` (base: Codex `142a7fc`, integration `844e3c0`). This is an engineering plan for
data the three pilots need. It sets no scientific parameter, and every value marked **owner** is undecided.
Nothing historical is reconstructed: an artifact that a run did not save is unavailable for that run.

## 1. What the pilots consume (as implemented)
| Pilot | Inputs at the fork | Post-fork inputs |
|---|---|---|
| 1, guidance | Complete fork state (actor + Adam, critic, target, alpha, normalization, replay for the panel); approved panel | Matched U/I agent states at equal steps (actor + Adam, critics, alpha, normalization). No replay needed (`load_replay=False`). |
| 2, passive | Complete pre-injection fork state, including replay, global and agent RNG, n-step queue and normalization; the 256-pair `panel.npz` | Ordered raw arrival streams: U for Stage A, plus I for Stage B. Each arrival carries the transition, the source normalization in force, the update count, and (new) the source agent key. |
| 3, frozen targets | Complete fork state (batches come from the fork replay) | Matched U/I agent states: frozen actors, target-network evaluators, online critic + Adam. No replay needed. |

## 2. Where each item comes from: Exp1 parent vs Exp2 arm
The Exp1 parent **is** the U arm: after `ForkNow`, the same process continues as the Exp2 control. The injected
arm is a separate `exp2_arm` job.

| Item | Produced by | Stage | Already saved and retained today? | Capture needed |
|---|---|---|---|---|
| Complete fork state (`fork/state`), `panel.npz`, `check1_pre/control.npz`, `fork.json`, fresh critic | Exp1 parent at f*_run | fork | **Yes**: written by `fork.write_fork`, never deleted | Archive only |
| Pre-fork history (probe records, ledgers) | Exp1 parent | pre-fork | Yes (ledger, probes) | None. No pilot reads pre-fork states or transitions **(spec: confirm none is wanted)** |
| U at the fork | = fork state | fork | Yes | None |
| I at the fork (post-injection, after Check 1) | `exp2_arm` first save | fork | **Written, then deleted** by the next routine save | Retain |
| U matched states at fork + j·Δ | Exp1 parent (control) routine saves | post-fork | **Written, then deleted** | Retain |
| I matched states at fork + j·Δ | `exp2_arm` routine saves | post-fork | **Written, then deleted** | Retain |
| U arrival stream | Exp1 parent (control) | post-fork | **Never written** | Record |
| I arrival stream | `exp2_arm` | post-fork | **Never written** | Record |
| Source agent key per arrival | both | post-fork | Never written | Record (new capture; needed only if S1 is chosen, and it costs 8 B per arrival) |

**Matching comes for free.** Both entry points save every `checkpoint_interval = N/20` (one save per check; same
manifest value for parent and arm, `checkpoint_start_frac 0`), and a fork happens at a check step, a multiple of
N/20. Both arms therefore already write states at the identical steps fork + j·N/20, for j = 1…5 up to `arm_end`.
Retention keeps those writes, so no extra save, update or RNG draw is needed. Δ = N/20 is a property of the
existing cadence. Retaining fewer offsets is an **owner** choice.

## 3. Mechanism (implemented, opt-in, methodology-neutral)
`record_exp2_scope(stream_root, chunk_size, …, retain={"root", "until_step", "replay"})`, through
`scripts/exp3_record_source.py --retain-root … --retain-until-step … --retain-replay retain|omit`:
- **Stream:** raw `buffer.add` input before n-step mutation; source `obs_rms` at add time, which is exactly the
  statistics used by that step's updates; the update count; the agent key. Chunks are published only with a
  routine save (C3 fix), so a resumed job's stream always matches its restored state. Provenance records the arm,
  the actual injection `(m, seed)` (C1), the fork state hash and the source code.
- **Retention:** every routine save under `<run_dir>/state` at a step in `[fork_step, until_step]` is hard-linked
  (copied if links are unavailable) into `root/step_<step>/` with `snapshot.json` (file SHA-256s, replay
  retained/omitted, dimensions). The `exp2_arm` post-injection fork-step save is included. A re-save at the same
  step after a resume must be byte-identical; otherwise a conflict marker is written, the source keeps running,
  and adapters refuse that snapshot.
- **Lean snapshots** (`replay: omit`) load in `Artifact(..., load_replay=False)` after checksum verification; any
  replay use is refused. Pilots 1 and 3 need no post-fork replay as implemented.

Verified on CPU through the **real** `exp1.run` (forced dev trigger) and `exp2_arm.run` (`tests/test_exp3_capture.py`):
- C1 provenance;
- retained steps exactly {fork + 20…200} for U and {fork, fork + 20…200} for I;
- the routine roots still hold only LATEST;
- lean snapshots pair under `require_matched`;
- Pilot 1 runs at fork time and post-fork on these artifacts;
- **the recorded and retained parent's final state is bit-identical to an unrecorded parent's**;
- passive U given the captured keys reproduces the retained active U snapshot bit for bit.

## 4. Sizes, overhead and memory
N = 500,000 interaction steps (1M env steps at action repeat 2) for the candidate DMC and MyoSuite tasks; checkpoint
interval 25,000; arm horizon 125,000 arrivals.

| Quantity | dog-run | humanoid-walk | MyoSuite (4 candidates) | Basis |
|---|---|---|---|---|
| obs / action dims | 223 / 38 | 67 / 21 | 83–115 / 39 | measured (env creation) |
| Agent checkpoint, D4W1536 | 1.22 GB | 1.22 GB | 1.22 GB | analytic upper bound: critic 75.7–75.9M params × (params, target, 2 Adam moments) × 4 B, plus actor. Measured untrained: 0.53–0.54 GiB (zero moments compress) |
| Agent checkpoint, D4W1024 | 0.54 GB | 0.54 GB | 0.54 GB | as above (33.7–33.9M critic params); untrained 0.24 GiB |
| Replay per transition (float32) | 1,948 B | 632 B | 832–1,088 B | (2·obs + act + 3) × 4 |
| Replay at the latest possible fork (475k) | 0.93 GB | 0.30 GB | 0.40–0.52 GB | |
| Stream per arrival | ≈ 7.4 KB | ≈ 2.35 KB | ≈ 1.6–2.1 KB | measured on hopper (613 B) and myo-reach (2,113 B); DMC observations are float64 |
| Stream per arm to `arm_end` (125k) | 0.93 GB | 0.29 GB | 0.20–0.26 GB | |

Per designated cell, retaining fork + 5 offsets (11 agent states plus 2 streams):
- **D4W1536, lean:** about 15 GB (dog-run), about 14 GB (humanoid-walk) and about 14 GB (MyoSuite).
- **D4W1024, lean:** about 8 GB, about 6.5 GB and about 6.4 GB.
- **Keeping replay** in snapshots adds about 11 × the replay size (up to +10 GB per dog-run cell).
- **Twenty cells:** about 285 GB lean if all are D4W1536, about 135 GB if all are D4W1024, about 420 GB for both
  scaled sizes.
- **U stream beyond `arm_end`:** the parent continues to `control_end` (up to N), and the U stream continues with
  it, adding up to about (N − fork − 0.25N) × bytes. For an early dog-run fork that is about 2.4 GB. Truncating U
  recording at `arm_end` is an **owner** choice; Stage B cannot use arrivals beyond `arm_end` in any case.

Overhead and memory:
- **Runtime:** per arrival, one copy of a few KB plus one 8-byte key read (the step already synchronizes on
  `np.array(actions)`). Per routine save, hard links plus one SHA-256 pass over the retained files: about 1.2 GB,
  roughly 2–5 s, six times per run. Recording performs no extra environment step, update, evaluation, save or RNG
  draw (bit-identical final state, CPU). A100 timing is **unmeasured** (GPU validation step 3).
- **Recorder memory:** pending arrivals up to one save interval: about 185 MB (dog-run), about 60 MB (humanoid-walk)
  and about 50 MB (MyoSuite).
- **Pilot memory:**
  - Pilot 2 holds 2 learners (4 for Stage B, sequentially by source), each with a full replay in host memory
    (≤ about 1.2 GB) and a training state on the device (≤ about 1.2 GB for D4W1536).
  - Pilot 1 per-state Jacobians need chunk × 171k actor parameters × 4 B per retained family (4 families), about
    0.7 GB for a chunk of 256.
  - **Pilot 1 storage** is about 2.7 MB per state per actor, or about 8 GB per cell for 1,000 panel states and 3
    actors. Whether to store full per-state parameter gradients or only summaries is an **owner** choice.

## 5. Can Exp1 production proceed before capture is enabled?
- **Yes, for every D2W512 parent** (they never fork) **and for the scaled parents of the 7 tasks outside the Exp3
  candidates.** Exp3 needs nothing from them.
- **Yes, for pre-fork Exp3 data in all cells.** The fork state and its panels are already written and retained,
  and no pilot reads pre-fork data.
- **Not without loss, for the scaled parents (D4W1024 and D4W1536) of dog-run, humanoid-walk and the MyoSuite
  candidates:**
  - those parents are the U source;
  - if they run unrecorded, the historical U stream and U matched states are lost permanently;
  - recovering them later needs a separately authorized U continuation from the retained fork state (about one
    arm-length run per cell), and that continuation equals the historical Exp2 control only if GPU training is
    deterministic, which is still unverified;
  - until the MyoSuite pair is chosen, all 4 MyoSuite tasks are candidates. Choosing the pair first frees 2 of them.
- **No new code path is forced on Exp1 production.** Capture is opt-in per designated run, through the recording
  launcher. Resumed segments must relaunch with `--resume-stream` and the same retention options; the C3 fix makes
  that safe.

Required before using capture in any production run:
1. GPU recording parity: the recorded and an unrecorded parent must be bit-identical on the A100 (harness exactness
   suite; `test_recording_and_retention_do_not_change_the_source_trajectory`).
2. Codex's engineering review of the recorder, retention and lean-snapshot code.
3. A storage allocation for the retained totals above.
