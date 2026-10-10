# Independent review of Codex's Exp3 capture corrections (2026-10-10)

Reviewed: `codex/exp3-capture-fixes` @ `853b6a523799132cf96a2e889150c10e3ea1c2ac` (engineering commit `b747975`),
based on my `5764bd0`. Review branch: `claude/exp3-capture-fixes-review`. No merge, no GPU job, no scientific
setting chosen. Reconciled methodology: [exp3_methodology_reconciled.md](exp3_methodology_reconciled.md).

**Overall: FAIL at `853b6a5`** (one remaining defect, R1); **PASS after the isolated fix on this branch**, pending
Codex's review of that fix. Codex's diagnosis of my earlier C3 fix is **correct**: publishing arrivals before the
checkpoint save let a failed save leave the stream ahead of LATEST.

## 1. Item-by-item
| Item | Verdict | Basis |
|---|---|---|
| Checkpoint-before-stream publication | **PASS** | Stage chunks, write the journal, run the unchanged save, publish one manifest at the saved step. Traced: failure before LATEST (old cursor; staged chunks are unreferenced evidence); failure after LATEST before publication (restart publishes); death after publication before journal removal (idempotent); empty interval (no-op). Codex tests cover a failed save and both sides. |
| Journal recovery | **PASS** | Only the old or new cursor is accepted. Staged bytes are re-verified (checksums, order, count); the base manifest hash is guarded; lineage must be equal. |
| Partial publication | **PASS** | One manifest per interval, with unique chunk names (no overwrite of referenced bytes). |
| Crash/restart consistency | **PASS for streams; see R1 for retention** | My real-entry test (arm dies after its final save, before DONE; resumed) ends with a sealed stream of exactly `arm_end − fork` arrivals. Exp1's restart from the fork state keeps cursor = fork step when no post-fork save occurred. |
| Retention conflict detection | **FAIL → fixed (R1)** | The retained/omitted mode bug is fixed correctly, but the byte-hash comparison flags logically identical re-saves as conflicts |
| Immutable provenance | **PASS** | `record_metadata` only appends to `launches` on restart; the fingerprint excludes exactly that key. `code_version()` = {commit, dirty (tracked files)}, so resumed segments must use the identical clean commit (strict, intended). |
| Captured-key validation | **PASS** | uint32 with shape (2,) is required at append and read time; no silent cast |
| dtype/signed-zero exactness | **PASS** | dtype, shape and bytes are compared on the original arrays before float64 descriptive errors; repeat detection is likewise byte-exact |
| Completed-capture finalization | **PASS** (minor note) | Verifies source metadata, identity, DONE and LATEST step, arm/injection, journal, cursor and checksums, then seals; no trainer is run. It does not re-verify `fork_state_sha256` against the fork directory (optional hardening). |
| Writer lock, retention-root guards, `--retain-until-arm-end` | **PASS** (CPU) | Lustre/NFS lock semantics remain unqualified (Codex states this). |

## 2. Remaining defect (fixed here)
**R1. Retention declares a conflict when a resumed source re-saves the same step, even though the state is
bitwise identical.**
- *Root cause:* two saves of one logical state are not byte-identical: Orbax OCDBT data and manifest files, and the
  `meta.pkl` pickle bytes, all differ. Yet `state_differences` is `[]` and every `meta` field is equal
  (`test_resave_bytes_differ_but_logical_state_is_identical`). `retain_state` compared file SHA-256s, so it wrote a
  `CONFLICT_*` marker, and `read_snapshot`/`Artifact` then refuse that snapshot permanently.
- *Real trigger:* a source that dies after its final in-window routine save and before DONE re-saves the same step
  on restart (`save(status="complete")` is unconditional). This hits every injected arm's `arm_end` snapshot, and
  U's when `control_end == arm_end`.
- *Reproduction on the real entry points* (`FinalSaveDeathTest`: real Exp1 with a forced dev trigger, then a
  recorded and retained `exp2_arm` killed between its final save and DONE, then resumed). At `853b6a5` this yields
  `snap_i/CONFLICT_step_…json`, and the arm-end matched snapshot becomes unusable.
- *Fix:*
  - `retain_state` still accepts byte-identical re-saves immediately.
  - Otherwise it compares the logical state with the existing exact comparator: agent leaves (shape, dtype,
    bytes), `obs_rms`, and `meta` (ignoring `wandb_run_id`, as `state_differences` does); replay too when retained.
  - Identical state writes an `EQUIVALENT_*` audit record and keeps the original snapshot.
  - Any difference still writes `CONFLICT_*`, listing the differing components, and adapters still refuse it.
- *Tests* (`tests/test_exp3_capture_review.py`, 4 tests):
  - before the fix: 3 failures and 2 errors;
  - after: all pass, both replay modes;
  - a genuinely different re-save still conflicts and is refused.

## 3. Other limits (not defects; unchanged)
- **Durability:** under power loss, or with Lustre/NFS rename/flock semantics, it is unproven. GPU and Delta
  qualification are needed.
- **Old streams:** streams without the new immutable bindings cannot be resumed (correct: nothing is invented).
- **Recorder exit without a fork:** if the source never forks, the recorder exits with "source did not execute a
  training continuation". Orchestration must treat this as "no Exp3 source", not as a failed Exp1 run.
- **Orphan chunks:** they remain as evidence and consume space.

## 4. Can the corrected capture code be integrated?
- **For development GPU validation: yes, conditionally.**
  - *Content:* `853b6a5` plus this branch's R1 fix, as one integration candidate, after Codex reviews the R1 change.
  - *Use:* (a) the tiny-fixture exactness harness on one A100; (b) recording parity, journal/crash boundaries and
    resource measurement on one real fork (the dev pilot's natural fork, or an owner-authorized forced dev trigger).
  - It is opt-in and changes no Exp1/Exp2 code path when unused.
- **For production capture on designated Exp1/Exp2 sources: no, not yet.** Still required:
  1. GPU recording parity (bitwise) and GPU journal/crash recovery;
  2. Delta filesystem lock and rename qualification;
  3. owner scope, retention and storage decisions;
  4. m frozen (for I arms);
  5. orchestration that distinguishes a no-fork exit from a failure.

## 5. CPU verification on this branch
| Run | Result |
|---|---|
| New review tests, before the fix (code at `853b6a5`) | 1 OK, 3 FAIL, 2 ERROR (R1 reproduced, including on real entry points) |
| New review tests, after the fix | 4/4 OK |
| Full Exp3 suite after the fix (Codex's 80 + 4 new) | **84/84 OK**, 911 s (log SHA-256 `bde1c151…f8aab8`) |
