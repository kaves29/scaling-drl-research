# Exp3 retention integrity remediation (E1–E3), 2026-10-10

Branch `claude/exp3-retention-integrity-fixes`, derived from Codex's failing-test commit
`0b287e479834de0fa095b2161749fc8eca353751` (which contains my `91e370a` plus Codex's report and adversarial
tests). No test was weakened, no scientific setting changed, nothing merged, no GPU job submitted.

## Reproduction first
At `0b287e4`, `tests.test_exp3_retention_adversarial` gave **18 tests, 5 failures**: exactly Codex's five IDs (two
E1, one E2, two E3).

## Fixes (`experiments/exp3/retention.py`, `recording.py`, `streams.py`)
**E1: an existing snapshot is verified before anything is accepted.**
- At an existing step, `retain_state` now calls `read_snapshot(target)` first. That checks the current bytes
  against the record's checksums, the file set and conflict markers. Only then does it consider the byte-equality
  fast path or the logical fallback.
- A corrupted snapshot raises "existing retained snapshot failed integrity" and is never returned or overwritten.

**E2: logical equality is never accepted alone.**
- New `validate_checkpoint(state_dir)` checks integrity and **device-backed restorability**, independently of
  value equality. In order:
  1. `_sharding` parses (base64 keys, JSON entries);
  2. every entry is a well-formed `SingleDeviceSharding` (anything else is refused as incompatible);
  3. the entries cover exactly the leaves of the checkpoint's msgpack tree;
  4. every saved device exists on this host (otherwise "restore unverifiable", never assumed);
  5. Orbax tree metadata is readable and matches that structure;
  6. an **actual Orbax restore with the saved shardings** succeeds;
  7. the device-backed values are bitwise equal to the NumPy read.
- It runs before every **new publication**; the result is recorded in `snapshot.json` as `"validation"`. It runs on
  **both** checkpoints before the logical fallback may write `EQUIVALENT`.
- Legitimate re-serialization (different Orbax/pickle/NPZ bytes, identical values, valid metadata) is still
  `EQUIVALENT`. Real differences are still `CONFLICT`. The device-independent NumPy comparison is never presented
  as proof of restoration.

**E3: there is no combined COMPLETE when retention is unhealthy, and reasons are explicit.**
- *Live source runs:* a retention failure is recorded as `FAILED_step_<n>_*.json`, with the reason, instead of
  aborting. Retention runs after the source's checkpoint is committed, so an abort would crash every restart in
  the same place. A misconfigured retention root still fails fast, before training.
- *Required snapshots:* the stream manifest records every published routine checkpoint step
  (`checkpoint_steps`). Required snapshots are those steps inside the window, plus a non-control arm's
  post-injection save at the fork.
- *Verdicts:* `capture_report` gives separate verdicts:
  - `stream_verdict`;
  - `retention.verdict` (COMPLETE / FAILED / NOT_REQUESTED), with explicit reasons: missing required step, read
    failure (checksum or conflict), CONFLICT or FAILED records, or a stream that predates the step record;
  - `capture_verdict`, which is COMPLETE only if both are.
- *Where it is enforced:* both normal recorder exit and `finalize_capture` write `capture_report.json`, and they
  raise `CaptureIncomplete` (a `ValueError`) carrying the report whenever the capture is not complete.
  Finalization still seals a verified-complete stream; the stream verdict is reported separately and is never
  mistaken for usable retention.
- *Re-runs:* reruns are idempotent when healthy, and failures persist across reruns.
- `scientific_qualification` remains `false`.

**Preserved:**
- my logical re-serialization fix (real-entry `FinalSaveDeathTest`, now also verifying a COMPLETE capture report
  and exactly one `EQUIVALENT` record);
- strict comparisons and the conflict refusal in adapters;
- the journal and publication protocol.

## New regression coverage (`tests/test_exp3_retention_integrity.py`, 12 tests)
- **Corrupted metadata**, each rejected with a specific reason both at initial publication (nothing published) and
  at re-save (no `EQUIVALENT`, original still readable), in both replay modes:
  - invalid `_sharding` JSON;
  - a missing leaf;
  - an extra leaf;
  - an incompatible sharding type;
  - an unavailable device;
  - corrupt `_METADATA`.
- **Corrupt existing snapshot**, refused by both the fast path and the logical fallback.
- **Live recorded source** whose retention fails at one step: the source runs to completion, a FAILED record is
  written, the stream is sealed COMPLETE, and the capture is `INCOMPLETE` with explicit reasons.
- **Real Exp1 fork + Exp2 arm finalization:**
  - a missing intermediate snapshot fails;
  - a missing fork-step injected snapshot fails;
  - interrupted finalization resumes to the same COMPLETE verdict, idempotently, with no markers;
  - a corrupt endpoint fails repeatedly and is not cleared by a rerun.
- **Healthy resumed arm:** the capture report is COMPLETE, with exactly one EQUIVALENT record and no FAILED or
  CONFLICT.

## Verification (CPU, Python 3.11 venv, JAX/JAXLIB 0.4.34; not GPU)
| Run | Result |
|---|---|
| Codex adversarial (18) + new integrity (12) | **30/30 OK** (the 5 previously failing pass) |
| Full Exp3 suite (all 9 Exp3 modules) | **114/114 OK, 0 skips**, 814 s (log SHA-256 `7e6c7425…caacf`) |
| Exp1/Exp2 regressions (`test_exp12_bitwise_comparison`, `_fork`, `_injection`, `_manifest`) | **50 run, 49 OK + 1 skipped** (HumanoidBench reach test needs the separate `venv_hb` environment, absent here), 704 s (`52ef537a…e5e5`) |
| Baseline isolation / configurations vs `844e3c0` | **337/337 baseline files unchanged; 195/195 parent configurations match** (`c8177cea…c7`, identical to Codex's receipt) |
| Exp12 break checks | **74/74 mutations fail when injected and pass when restored**; exit 0 (`501d2b7a…8356`) |

Two follow-ups surfaced by the full run, both resolved:
- `finalize_capture` read the fork plan unconditionally (a KeyError for a capture without retention, a regression
  in this change). It now requires the plan only when retention was requested, and its absence is then an explicit
  error.
- Codex's `test_identical_retention_is_idempotent` retained a synthetic directory with no agent checkpoint, which
  publication now correctly refuses (E2). The test now saves a real state, and its assertion (identical
  re-retention is idempotent, no conflict, readable snapshot) is unchanged. No check was relaxed.

## Limits (unchanged)
- **Validation cost:** it adds one device-backed restore per retained publication (about 6–12 per designated run)
  and per re-save fallback. GPU time and memory for it are **unmeasured** (job 22765261 measured restore at
  2.5 s, for an ordinary D4W1536 state).
- **Host dependence:** validation on a host lacking the saved devices refuses with "restore unverifiable". For
  example, CPU-only finalization of a GPU checkpoint that needs a *new* publication or a re-save fallback. That is
  explicit, not a pass.
- **Not qualified:** Delta filesystem semantics, power-loss durability and cross-node behaviour.
