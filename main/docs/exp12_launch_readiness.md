# Exp1/Exp2 engineering launch readiness

**Historical pre-integration snapshot.** The separate candidate integration was subsequently authorized and completed. Its current591-test/73-mutation CPU evidence, actual HB coverage and scientific blockers are in [exp12_integrated_readiness.md](exp12_integrated_readiness.md). The prior uncreated/integration-permission/HB-skipped status below describes this earlier revision, not the new candidate. Entropy authority/notation is clarified in [entropy_temperature_reconciliation.md](entropy_temperature_reconciliation.md); no target sign was changed.

2026-10-08 (America/New_York). **Production: BLOCKED.** Reviewed engineering work is ready for local review and separately approved GPU validation. CPU results do not establish scientific qualification, A100 performance, or HumanoidBench execution. No Delta access, job submission, push, methodology merge, historical deletion, or scientific-setting change occurred in this sprint.

## Authority and revision identity

Authority is `.claude/methodology-exp1-exp2.md`, including its chronological amendments (a)–(z), corroborated by the recorded lead answers in `docs/exp12_decisions.md`. Later amendments supersede earlier alternatives. `docs/methodology_reconstruction.md` and `docs/methodology_implementation_audit.md` explain implementation; `docs/remaining_scientific_decisions.md` records unresolved choices rather than granting permission. Old CLAUDE/Angle descriptions and superseded range/architecture examples are not current experiment specifications.

| Revision / branch | Verified state and purpose |
|---|---|
| `781dc6c93022a82b68a3210617db93225284b268` | Original frozen checkout preserved unchanged |
| `312bb01a7a6aa2e3a34a74dd557f05543081d0f5`, `claude/eloquent-fermat-inxqlt` | Live GitHub branch head; launcher/backend fixes; actual job 22740708 source |
| `f6b7f73b018eac365819ce64c90d22942b211f42`, `codex/a100-runtime-debug` | **Published**; local, tracking, and live remote heads agree; diagnostic instrumentation based on 312bb01 |
| `a3f44aa56455e3aae4b36459a78c97237bad62a8`, `codex/methodology-reconstruction` | Local, unpublished, separately audited/tested approved corrections and engineering safeguards; not merged |
| `6208ba980af5b89b01891d8c6e8373283ed86fed`, `codex/exp12-launch-readiness` | This sprint's checkout/packaging/provenance implementation commit, based directly on f6b7f73 |
| `eaa3baa9fdd4f9d88214e2db99f2371f8a1c6a1d`, same branch | Read-only parent marker/source inventory; subsequent documentation commit completes this local-only branch |

Live remote inspection also found `main` at `6ffccbaf11bd8f60ce663b59ac546e9ff7cffd9c`, `delta-fixes` at `d52b0c121723b1ba9ba33850296b8e741ad25c25`, and `claude/jolly-keller-lwr6xi` at `8709696d2a9272857a9569590b6a00fa51555e40`. These are not reviewed integration candidates. Do not substitute them by branch name. Local investigation/audit checkouts retain their existing untracked reports/scripts.

The explicitly approved twin correction in a3f44aa computes each Q's FP32 paired loss before averaging. Its demonstrated 1.49e-8 order difference can affect a near-zero HumanoidBench trigger; its approval is recorded in the decision report. It does not authorize merging that branch now or resolving any other scientific question.

## Methodology and production-path review

References below are relative to `main/` at a3f44aa unless a different revision is named. Each source claim was checked independently of test verdicts. MATCH describes the implemented contract, not empirical scientific acceptance.

| Contract and authority | Executable path / finding | Status and scientific consequence |
|---|---|---|
| Fixed D1W128 actor; D2W512/D4W1024/D4W1536 per-Q critics; amendment (y) | `generate_manifest.py:211`, `configs/base_exp12.yaml:19`, SAC network construction | MATCH; 195 unique parents, 39 environment/architecture cells |
| All 13 environments and raw-step budgets; amendment (u) | Generator environment/budget lists; independently composed census below | MATCH; actual environment execution remains unverified for HB |
| Five seeds 1–5, UTD=2, repeat=2, one training environment | Generator, base config, `experiments/exp12/trainer.py` | MATCH; all 195 configs checked, 130 potential scaled parents, not 130 guaranteed eligible forks |
| AdamW actor/critic/temperature lr=1e-4; wd=.01/.01/0; alpha=.01; tau=.005; discounts .99 DMC/HB, .95 Myo | `configs/agent/sac_simba.yaml`, environment YAML, `scale_rl/agents/sac/sac_agent.py`, `sac_update.py` | MATCH to recorded G1 settings except the entropy authority conflict below; no optimizer or initialization edits |
| Single-Q DMC/Myo, twin clipped-Q HB; amendment (z) | `critic_use_cdq`, SAC critic/actor updates, `experiments/exp12/twin.py` | MATCH; HB Q sizes/counts and synthetic twins validated on CPU, simulator/GPU qualification absent |
| Five paired probe rounds, 1000 fresh-state AdamW fitting steps, 25600 sampled positions, batch 256, chunk 2560, own-mean offset, sin(1e5*f) teacher | `probe.py:60–153`, `run_probes.py`; independent scalar/statistical references | MATCH; targets/pool/minibatches paired, separate RNG streams, fits copies without mutating training state; CUDA target/fitting quality still unqualified |
| Same-run critic captured before first update; check 0 at 5000 transitions, 20 budget-fraction checks | `trainer.py:167`, `run_probes.py`, `probe.py:44` | MATCH; fresh-versus-fresh calibration is a different estimand from production longitudinal comparison |
| Paired L, IQM, 10000-resample 95% percentile interval, strict lower bound >0; first two consecutive checks, completion k=2…19 | `probe.py:161`, `trigger.py:42–74`, `run_probes.py:102` | MATCH at a3; frozen/runtime branches retain the older twin arithmetic order and must not be used as the corrected production revision |
| Exp1 final-checkpoint architecture endpoint, paired-seed bootstrap/IQM reporting, dev exclusion | `analysis/exp1_analysis.py`, `ledger.py`, strict `analysis/exp12_validation.py` | MATCH at corrected reference; trigger-conditioned output is not substituted for the primary endpoint |
| Scaled-only forks, full state restore, original process becomes restored control, injected arm starts from same fork, horizon .25B, max total 1.2B | `exp1.py:87–215`, `exp2_arm.py:87–180`, `fork.py:96`, `trainer.py:207–285`, `state.py` | MATCH; control end=max(B,f+.25B), injected end=f+.25B; checkpoint/config/device guards, full RNG/environment/window state persisted |
| Nikishin identity-preserving old+new−copy, new/copied heads equal, frozen copy, trunk state retained, new optimizer state fresh | `injection.py`, trainer injection and injection tests | MATCH by source/CPU; real GPU Check 1 and complete-state identity gates pending |
| Natural-trigger, full-setting PC; smallest last/half/all within .10 of best, pooled noise/L_trigger ≤.10, positive L_trigger | `m_selection.py`, `scripts/positive_control.py`; amendments (d)–(f) | MATCH to implemented approved ranking/noise rule; no qualifying PC/m freeze available; required magnitude/sign of rescue remains unspecified |
| Check 1: shared 256 panel FP32, control/identity bit-exact, injected arm existing 64-eps scaled rule; Check 2 paired probe comparison without exclusion on failure | `fork.py:110–155`, `exp2_arm.py:53`, `analysis/exp12_validation.py:687–950` | MATCH; no tolerance selected here. Base YAML's “PROPOSED” comment is stale relative to amendment (m)/recorded approval |
| Post-fork evaluations at 0…25 every .01B, 10 episodes, raw returns and paired-seed IQM bands | `fork.py:200–240`, `exp2_analysis.py` | MATCH; exactly one eligible seed requires an approved reporting convention and currently fails explicitly |
| TF32 training/action/probe/eval, FP32 Check 1 and actor KL/churn forward diagnostics, no FP64 training | `precision.py`, diagnostic contexts, SAC explicit CPU initialization at `sac_agent.py:112,150` | MATCH; explicit CPU parameter initialization preserved; CPU cannot validate CUDA numerical bounds |
| Complete-population certification, missing/incomplete evidence refused | a3 `analysis/exp12_validation.py:963`, analysis publish paths | MATCH for strict confirmatory analysis; completion markers alone are not certificates or full checkpoint CRC/restore verification |

Independent census budgets (raw steps; interaction budget is half):

| Environments | Raw budget each |
|---|---:|
| dog-run, dog-trot, humanoid-run, humanoid-walk, humanoid-stand | 1,000,000 |
| swimmer-swimmer15, hopper-hop | 500,000 |
| myo-key-turn, myo-pen-twirl, myo-pose-hard, myo-reach | 1,000,000 |
| h1-reach-v0, h1-run-v0 | 2,000,000 |

The census independently enumerates all architecture/environment/seed keys and composes every actual generated command. It checks actor/critic dimensions, budgets/cadence, UTD/repeat, single/twin selection, optimizer settings, probe/trigger/fork constants, unset injection m, and absent forced triggers. It verifies the current negative entropy coefficient while exposing the conflicting authority; it does **not** certify that unresolved choice.

## Engineering findings and completed fixes

1. **Confirmed checkout-isolation bug:** Block A/B Slurm wrappers hard-coded `/work/hdd/biqc/skaveti1/exp12/main`, performed environment/GPU setup before their optional commit check, and allowed dirty source. They could execute an older checkout despite submission from a reviewed detached revision. New wrappers use `SLURM_SUBMIT_DIR`, require full exact EXPECTED_COMMIT, check tracked/staged source and untracked files outside their existing generated `main/logs/`, require tracked driver/config files, and reject BLOCKB_TEST_HOOKS before module/CUDA setup. Slurm spool and detached-HEAD regressions verify this without cluster access. Tracked changes under logs remain rejected. Resource declarations, validation commands, caches, environment settings and timeouts are unchanged.
2. **New artifact packager:** `scripts/package_exp12_diagnostic.py` uses the standard library and local Git objects only. It preserves complete raw profile/trace/backend/exit/commit and Slurm bytes, optional stacks/identity/profile verdicts, immutable revision-specific YAML/launcher/trace/profile source, member SHA-256 sums, manifest and archive sidecar. It refuses mismatched recorded commits/job IDs, symlinks and overwrite; marks missing/partial evidence; never calls Slurm/JAX or infers qualification from exit zero. Its core required-file list is for profile diagnostics; identity traces are included when present, and the manifest explicitly reports an absent profile trace. Caches/temp/checkpoints/unlisted large files are excluded. Job ID is caller supplied and cross-checked where the historical artifacts allow it.
3. **Future runtime provenance:** wrapper_start now records actual entry-point argv and Slurm job ID. This adds one JSON record's fields without changing the wrapped computation or synchronization. The original frozen runtime branches are untouched.
4. **No empirical runtime fix claimed:** job 22740708 confirms successful CPU/CUDA initialization at 312bb01, then an internal 300s timeout. Complete trace has 1833 valid records, 886 matched begin/end pairs, last progress 5900/update1802 at t≈102s, followed by approximately174s silence. Recorded compilation (288 calls, ≈25s inclusive compile time) ended before the silent interval. No profile.json/validation.json or stack evidence exists. First populated pending-metrics flush around interaction6000 is the leading source-supported hypothesis, not an observed blocking call. The full f6 boundary/flush/watchdog instrumentation is published for the next same-budget diagnostic. No demonstrated causal link to the older six-hour failure exists.
5. **Read-only queue omission guard:** `scripts/check_exp12_campaign_inventory.py` independently enumerates195 parent identities, flags malformed/stale DONE, inconsistent/unready fork headers, missing terminal state structure, incompatible initial/resume source provenance and aliased/escaping directories. It creates only a new requested report, never modifies markers/payloads/claims or calls Slurm, and labels every payload/qualification as not assessed. A regression demonstrates that the existing generator calls a malformed marker “done” while this guard blocks it. This is an explicit prequeue mitigation; generator/worker marker semantics are unchanged. Full replay CRC/Orbax restore and scientific certification remain separate requirements.

Production orchestration/recovery review:

- Explicitly select `--experiment exp1/exp2_arm --config_name base_exp12` and generator `--grid exp12`. Generic run/preflight/claim defaults still select older Angle paths. `claim_launcher.py --phase-files` must name the Exp12 manifests; do not use the older packing launchers' 32-slot/8-GPU assumptions.
- Parent grid has 195 unique checkpoint/log identities. Only scaled parents with FORK_READY and frozen m produce injected-arm commands, grouped by originating GPU model. D2 parents never fork. Failed Check 2 remains in the primary eligible comparison. No eligible-parent count is invented before real triggers exist.
- DONE presence alone suppresses queueing and runtime startup. Malformed/stale markers can silently omit work from an active queue, but a3 strict final analysis rejects invalid/missing parents/arms. Reconcile all195 independently against source states and ledgers before recovery or acceptance. Do not remove markers, orphan claims or payloads automatically.
- Claims use atomic transition guards and preserve orphan guards for manual recovery. Scheduler ownership tests mock Slurm; no remote scheduler was queried. Full state saves atomically repoint LATEST after agent/replay/meta/normalization/window state, keeping fresh/fork/latest state; failed saves leave the previous complete pointer. a3 confines pointers and validates replay headers before replay mutation. Final analysis checks saved evidence but is not exhaustive large-payload CRC/Orbax restore validation.
- Runtime config changes are refused on resume; source/simulator changes currently warn and enter launch history rather than immediately fail. a3 confirmatory analysis refuses incompatible commits, dirty launches, GPU models, precision, runtime and simulator versions across **every launch**. Freeze source/runtime per campaign, never resume a production run across revisions, and use fresh qualification output roots.
- a3 refuses existing Exp12 manifest files rather than overwriting. f6 alone does not have all a3 safeguards. Generate initial master commands and each later recovery/arm queue in fresh directories; a filtered active queue need not contain195 commands and is not the study's expected population.
- Existing resources are declarations, not measured qualification: Block A1A100/4CPU/32G/90min; Block B4A100/64CPU/128G/8h with unchanged7.5h driver limit; HB1A100/16CPU/64G/4h; runtime1A100/4CPU/32G/7min with300s inner timeout. Campaign wall time, packing, VRAM/host RAM, fork I/O/storage and quota remain unverified. Do not extrapolate approval from historical D6 estimates or choose new limits.
- Historical Angle modules are imported by the registry and provide shared pending metric replay, symbols, analysis/storage infrastructure and regression coverage. Importing them does not dispatch their training function for explicit exp1/exp2_arm. Removing them is neither authorized nor supported as a fix for the observed stall. Preserve historical sources. See the separate `legacy_methodology_cleanup_audit.md` for its complete inventory; no cleanup was performed.

## Scientific blockers: evidence, hypotheses and smallest next evidence

| Gate | Established evidence | Unresolved cause / minimum evidence |
|---|---|---|
| D4W1024 fresh null | Block B22706349, source `b4a90cb4e6ef692f5bf857fdc574c4a146c2d309`:13/100 checks, above approved5%; D2=3/100, D4W1536=5/100 | First obtain `gpu3/null_dog_run/null_pairs_D4W1024.jsonl`, corresponding summary and other architectures' controls plus launch/config/resume provenance. Recompute all five-round intervals and duplicate/index/provenance identities before any repeat. Those individual rows are absent from the attachment |
| D4W1536 hopper range | Attachment includes all9 range JSON rows. Independently reproduced configured25600 ratio **0.6023976348282202**, P_IQM=.3008451164, b_IQM=.4994128446, MSE_IQM=.1999773383; all five round ratios .559–.649 | Raw pools/targets/weights/offsets and launch/runtime provenance absent; fitting utility does not save learning curves. A separately approved diagnostic must record existing-step convergence/target/baseline evidence while preserving1000 steps/25600 pool/optimizer/precision. No smaller-pool substitution |
| Entropy authority conflict | Original M says +A/2; G1/released settings and current coefficient say −A/2; implemented loss alpha(H−h*) gives opposite gradient signs | Lead must explicitly reconcile the objective/target convention. No engineering-only sign replacement exists |
| Positive control / m | Approved natural full-setting PC ranking/noise rules implemented; no qualifying observed result or frozen m | Obtain natural-trigger D4W1536 dog-run PC at final approved source/runtime. Lead decides whether ranking alone suffices or an additional successful-rescue criterion is needed, then reviews/freezes m |
| CUDA bounds / identity / HB | Real job proves backend startup, not completed runtime/profile or identity qualification | Full-setting diagnostics on/off comparisons and same-model identity gates for both scaled sizes across DMC/Myo/HB, actual HB package/environment qualification, then lead-approved numerical bounds; all four current proposed bounds remain None |

Ranked null hypotheses: fixed-initialization fitting heterogeneity across rounds and five-round percentile-IQM coverage limitations; then unverified stale/mixed artifacts; then numerical/sampling/other task effects. Existing data cannot rank these by observed round-level statistics. Independent CPU Gaussian toys show ≈90.4% two-sided coverage for the empirical five-round interval and increasing zero-exclusion with artificial shared pair effects; they do not reproduce critic fitting or identify the13% CUDA cause. Illustrative IID binomial Pr≥13 at p=.05=.001464348 is not a new gate. Per-check13% must not be squared into a production false-fork rate: two consecutive longitudinal checks have unmeasured dependence.

Ranked hopper hypotheses: finite-budget fitting/conditioning and task-specific sampled input/target complexity; then TF32/sine sensitivity and chunk-dependent compilation/numerics. Smaller-pool ratios .9867(1600), .8451(6400), .6024(25600) confound pool distribution/coverage and evaluation chunks320/1280/2560. Five poor fresh rounds establish measurement-range failure, not training-induced degradation or critic-to-actor causality. Approved twin subtraction cannot explain either single-Q DMC gate. Standalone historical range-verdict and null-resume defects were corrected in a3, but neither proves an error in these actual observations. A descriptive older one-check probability print remains stale; production's two-consecutive rule is correct.

Minimum GPU follow-ups require lead approval of diagnostic design, not new production methodology: replay an exactly recorded null pair with identical immutable pool/keys/parameters; capture losses/gradients/final MSE at existing hopper fitting steps and teacher/offset/baseline summaries with unchanged fitting budget. Such instrumentation would change the diagnostic source/overhead and requires separately labelled comparison. Precision contrasts, extra initialization streams, repeats or pool/step alternatives must be approved explicitly. Historical complete null-grid cost≈6042s and hopper3architecture×3pool cost≈198s on A100; per-architecture timing/VRAM is absent, so a precise single-follow-up budget is not estimable. No experiment was submitted.

These failures prevent qualification of the measurement procedure and block confirmatory launch. They do not establish failure or success of the project's primary scientific hypotheses. Changing thresholds to obtain passes would invalidate the current qualification contract.

## Completed CPU validation and limitations

Final current-run results and log fingerprints are recorded in the companion validation commands document. Completed corrected-reference full discovery: **223 methods,221 passes+2 HB dependency skips,1659.063s**. Completed engineering-branch full other-module discovery: **302/302 passes,816.377s**. Final engineering/manifest group: **83/83**,90.948s; final13 inventory methods rerun after strict dirty-boolean/GPU-model header guards, all pass. Corrected-reference break harness: **73/73**, including all72 original mutations and approved twin-order mutation. Independent references verified195 configurations,39 cells,130 possible scaled parents,68 IQM/CI fixtures,4 exact-bootstrap crosschecks, trigger/pairing arithmetic and all9 original hopper range rows. Tests run with pinned JAX/JAXLIB0.4.34, Python3.12.14, NumPy1.26.4, Flax0.8.4, Optax0.2.3, Orbax0.5.3, JAX_PLATFORMS=cpu, no compilation cache. `uv --no-cache pip check` reports all117 installed packages compatible. This is not Delta's Python3.12.13/CUDA environment.

The original187+2 expectation predates later regression additions. At a3 discovery contains223 methods (221 passes+2 explicit absent-HB skips); the final harness contains the72 original mutations plus one approved twin-order check=73. New engineering branch adds34 methods; no assertion/tolerance was weakened and no dependency skip is counted as a pass. Separate revisions are tested separately; a combined integration tree is **UNVERIFIED** until explicitly authorized and tested. The two skipped methods are actual HB reach-evaluation seeding and the HB per-suite pipeline; synthetic twin coverage does not replace either.

## Reviewed integration plan — not executed

1. Keep781/312/f6 runtime comparison revisions immutable. Review engineering commit6208ba9 and a3's exact diff/approval provenance; retain both histories.
2. Obtain explicit authorization to merge a3 on a new integration branch based on this engineering branch. The781→f6 change set has8 paths;781→a3 has24; **their intersection is empty**. This lowers textual-conflict risk but proves neither behavioral compatibility nor scientific qualification. The observer wraps probe/fitting/state/Check2 APIs whose existing signatures remain unchanged; a3 adds paired_loss/validation internals. Re-run integration-level tests regardless.
3. After approval, make a normal non-squashed merge preserving a3 and runtime commits; inspect full merge diff, canonical document/config/SAC hashes and approved twin arithmetic. Do not merge into main or the diagnostic-comparison branch by default.
4. Run complete integrated CPU discovery,73 mutation checks, independent195 census, new checkout/artifact/provenance tests and shared older-regression coverage. Keep result provenance revision-specific. No current test result is assigned to an uncreated merge commit.
5. Resolve scientific choices, obtain final-source GPU qualification/PC/m/runtime evidence, approve deployment resources, freeze study manifest/source/runtime/config identities, publish only upon authorization, and launch only upon explicit approval. A successful short runtime diagnostic alone cannot unlock production.

## Production launch checklist

| Required item | Implemented | CPU-validated | GPU-validated | Scientific qualification / blocker |
|---|---|---|---|---|
| Approved195 parent grid / fixed actor / budgets / seeds | YES | YES, independent census | NO full grid | Entropy choice unresolved |
| Probe/paired trigger/final endpoint | YES at a3 | YES references/tests/mutations | Not qualified | Fresh-null and hopper gates FAILED |
| Literal approved twin arithmetic | YES at a3 | YES | NO HB qualification | Approved correction; integration pending |
| Fork/control/injection/complete restore/Check1/Check2 | YES | YES fixture tests | NO full gate | Both sizes×all suites still required |
| Natural PC / m freeze | YES guards at a3 | YES fixtures | NO qualifying PC | Success semantics/m decision required |
| Full-population confirmatory report / missing-data refusal | YES at a3 | YES fixtures | No production data | One-seed uncertainty convention unresolved |
| Checkout isolation / full diagnostic artifacts | YES this branch | YES regressions and original1833-record archive | NO new job | Ready for engineering review only |
| A100 backend initialization | YES312 | Mock+CPU coverage | YES22740708 startup | Runtime stage stalls/timeout unresolved |
| Runtime/probe/save/restore/eval/packing performance | Instrumentation ready | CPU reproducer only | Incomplete | Complete same-budget next trace needed |
| HB runtime / twin simulator coverage | Code/qualification scripts exist | Synthetic coverage;2 dependency skips | NO | Compatible reviewed HB environment needed |
| Exact-source recovery / marker census / full payload integrity | Guards/history/certification | Pointer/header/ownership tests | NO campaign restore | Enforce frozen revision; inspect payloads/markers; no automatic repair |
| Reviewed combined integration commit | Plan prepared | NO combined tree | NO | Explicit merge approval first |
| Production GPU/cache/resources/storage/time limits | Tools/scripts exist | Syntax/orchestration only | NO qualified plan | Evidence and lead deployment choices required |
| Frozen study manifest / immutable master195 / arm provenance | Tools exist | YES fixture/composition tests | Not yet created for campaign | Requires approved final revision, m, GPU/runtime |
| Launch authorization | NO | N/A | N/A | No jobs authorized/submitted by this sprint |

Approval priorities:

1. Reconcile the entropy objective/target authority and decide the disposition/additional evidence for both failed measurement gates. No production criterion is relaxed here.
2. Authorize a separately named integration of the already-approved a3 correction with reviewed engineering commits, followed by complete combined-source CPU validation. This permission is separate from approval of the twin arithmetic itself.
3. Approve the final-source GPU diagnostic/qualification design and obtain genuine natural PC evidence; decide successful-rescue semantics and freeze m. Extra settings, precision contrasts, fitting budgets or thresholds need explicit scientific approval.
4. Approve CUDA diagnostic numerical bounds and any cache/determinism qualification convention from actual measurements; obtain real HB identity/correctness coverage.
5. Decide the one-eligible-seed reporting convention and review the measured GPU/resource/packing/storage/runtime deployment plan. Only then freeze/publish the final study and authorize production launch.

**READY:** engineering fixes, CPU verification artifacts, packaging, source/integration review and next-stage instructions. **BLOCKED:** scientific gates/choices, integration permission, PC/m, GPU/HB/runtime/deployment qualification and launch permission. **UNVERIFIED:** uncreated combined revision, exact stall operation, four CUDA bounds, real195 outcomes/eligible population, production storage/packing/recovery performance. No deadline changes these statuses.

The five-minute diagnostic remains manual: no generator, packager, inventory or new preflight automatically submits it. Only the explicitly labelled documentation contains submission commands. Review here used a separate source-preservation/adversarial pass and independent CPU references; it does not imply another reviewer or an executed combined-source/GPU qualification.
