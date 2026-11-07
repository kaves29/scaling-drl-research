# Independent PR #1 review and first bounded A100 qualification

Reviewed PR: https://github.com/kaves29/scaling-drl-research/pull/1

Original head: `4afc772d5f2beab625e3b513e6f53cc72ea8e1c6`.
Integration starting point: `8bb0c0fb80bdd5eb128c4d6c848c56c125420a3f`.
Original PR history is retained through review merge
`09a0af582e4e49e3ec8dff3384292bf5385b89a5`. Resolve the final published
integration commit from Git and the delivery report; never submit a moving tip.

## Findings and corrections

| Original issue | Engineering-only review correction |
|---|---|
| Startup backend/device record did not establish placement of restored training state | Observe actual first-updated and saved actor/critic/target/temperature parameters; wrap the existing trainer restore to record its actual source, completion, saved/restored counters and parameter devices. No device transfers or initialization changes. |
| Deterministic fresh restart could match references without demonstrating checkpoint restoration | Require one observed restore from the selected trained step60 checkpoint in cases60/95. Preserve its on-disk copy before later saves replace it. Case5 explicitly exercises fresh restart before any save, not restoration. |
| Missing reference state could be called nondeterminism; missing/corrupt comparison inputs could escape without useful status | Missing/corrupt successful-run states are INCOMPLETE. Persist error evidence and partial summaries. Failed actual restore attempts remain observable and cannot be hidden solely by their missing final state. |
| Empty/partial cases and the second resumed/reference comparison were insufficiently guarded | Require all explicitly declared scenarios and both comparisons. `--crash-step 95` selects an explicitly scoped shard; omission is not a full-suite PASS. |
| DONE plus agreeing prematurely stopped states could pass | Check the existing fixture endpoint:300 interactions/582 updates. The counter follows warmup10, UTD2:2×(300−10+1), not a new training budget or numerical tolerance. |
| Child output was buffered in the parent until exit | Stream combined stdout/stderr to per-child files; save command/environment/paths before launching, update receipts and backend observations on disk, and preserve summaries incrementally. |
| Two agreeing references were presented as proof of backend determinism; differing references as proof of a GPU cause | Retain exit labels for compatibility but describe sample-level evidence and unproven causes. Repetition variability does not authorize altered exact-equality expectations. Concurrent relaunch errors stay visible. |
| Proposed one-hour/16-CPU/64GB job exceeded approved diagnostic resources | Do not adopt it. Tracked wrapper uses oneA100/fourCPUs/32GB/seven minutes and one300-second total workload cap with15-second kill grace. It never submits itself. |

The probe uses `exp12_subprocess_runner.exp1_run` directly, bypassing the old
CPU-forcing test helper. That runner invokes the real `experiments.exp1.run`,
whose `build(latest)` calls `Exp12Trainer.restore`, including the original
SAC Orbax loader, replay, environment/global RNG, normalizer and loop-state
restoration. Complete final checkpoints are compared with the unchanged
`state_differences`: parameters/targets, optimizer/temperature/JAX RNG/reference
batch, replay arrays and metadata/n-step queue, normalizer, global/environment/
action RNG, counters, probes, logging/diagnostic windows and evaluation metadata.
Only its existing wandb_run_id exemption is retained. Dtype/signed-zero/NaN
array rules are unchanged. Old numerical zero-difference norms alone did not
prove the newer representation-exact Check1 rule.

Backend observations query device metadata only. They introduce no RNG draw,
fit, update, array placement, precision context or scientific configuration.
No production file is changed by this review. The atomic-cache correction and
pinned scientific/pilot reports were already integrated at8bb0c0f; PR #1 does
not reintroduce older code. All195 parent fingerprints still match the untouched
2d007663 baseline exactly. Baseline files outside the diagnostic/documentation
allowlist were independently checked by Git blob hash.

## CPU receipts and limits

Raw receipts/artifacts remain outside Git at
`/workspace/scratch/exp12-pr1-review/`.

- Original PR:9/9 synthetic unit tests; original eight-child real CPU probe PASS.
- Focused regression:47 tests,46 passed/one optional filelock cache-eviction
  dependency skip; runtime control flow, checkout guard, SAC checkpoint,
  metric/checkpoint contract, cache and probe tests. Final diagnostic-only
  follow-ups separately pass23/23 unit tests, including failed restore,
  premature DONE, backend evidence and retained preflight failure output.
- Instrumented eight-child probe: PASS at all5/60/95 crash cases. Original and
  instrumented complete final states match at ref/ref2/crash5/crash60/crash95,
  each at300 interactions/582 updates. Final checkpoint receipts are rerun after
  endpoint/completion guards; result recorded below.
- Explicit four-child crash95 CPU shard: PASS, actual step60/update102 restore;
  observed child times37.6/34.1/28.9/28.7s (sum129.3s, excludes parent overhead).
  This run precedes the final endpoint-receipt addition; endpoints were already
  independently read as300/582. It is not a measured GPU time.
- 195/195 exact parent configurations match the prior baseline; no settings adopted.
- Earlier integration receipts remain valid for unchanged production paths:
  25/25 fork/kill/resume/identity methods,74/74 mutations,65/65 simulator
  restoration and the broader focused/methodology suites. They are not new
  GPU evidence or a new whole-repository run for this PR.
- `bash -n`, embedded Python parsing, diff checks and resource/timeout census pass.
  Ordinary bad-commit preflight is executed on an isolated local Git fixture:
  exit2 before backend initialization, retained status/checksums. No Slurm call.

Initial test attempts included a shared-subprocess mock before imports and
incorrect module/working-directory names. These test-invocation defects were
corrected; final unit/focused runs above pass. All attempts are retained.
JAX/Orbax deprecation and saved-sharding warnings are visible in raw logs; they
are not swallowed or treated as GPU passes. The GPU, Delta filesystem and full
pilot-size save/restore remain unexecuted here.

Final complete CPU endpoint/completion receipt: **PASS**, all eight children at
all three scenarios. Five final states also match the original PR CPU probe
bit-for-bit; each completed endpoint is300/582 and both actual restores record
step60/update102 and completed=true. Final child time sum304.8s (excluding
parent startup/comparison), supporting explicit first-shard selection rather
than assuming the whole probe fits300s. Receipt:
`final-review-receipt.json`, SHA-256 `f9149a4fca1ef5cf35df8a05a8a089a877c83b06ce68b5dee8214dbd6e32a82b`.

## Next action and remaining blockers

First execute the prepared `sbatch_exp12_gpu_resume_gate.sh` only after owner
submission authorization, at the delivered final integration SHA. Exact detached
checkout, setup, submission, outputs, packaging and qualification conditions are
in [exp1_a100_minimal_gate.md](exp1_a100_minimal_gate.md). Planning estimate is
roughly three to five minutes, not a demonstrated A100 timing. Timeout is
INCOMPLETE; do not enlarge resources or shorten fixture settings automatically.

PASS qualifies only tiny single-critic GPU cross-process restore at crash95.
Current-source in-process bit-exact Check1/fork and D4W1536 dog-run seed102
checkpoint smoke remain relevant next GPU shards. This is not full-width
cold/warm identity, twin/Humanoid, Exp2 restart or scientific qualification.
The old full-width identity_warm cost subtotal exceeds300s; keep it held.

No independently demonstrated production engineering defect remains after
these corrections. A full-length dog-run development parent still needs the
owner's trigger disposition, approval of its development/positive-control role,
and an appropriate full-run allocation/execution authorization; the historical
proposed eight-hour job is not approved by this integration. Fresh-null/hopper,
natural positive-control/m, broader CUDA identity and pending scientific bounds
continue to block the195-run confirmatory campaign, not independent preparation
of this development attempt. All numerical tolerances and threshold0 stay fixed.

Native Git integration/push is available. GitHub CLI credentials are invalid and
public API access is blocked by the environment's CONNECT proxy403. Report
Git ancestry/publication verification separately from unverified PR UI status;
do not fabricate a GitHub API merge receipt.
