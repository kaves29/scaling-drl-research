# Integrated Exp1/Exp2 candidate readiness

This report supersedes the earlier **integration-not-executed** status in `exp12_launch_readiness.md`. The latest task explicitly authorized a separate candidate integration. Production/main, the launcher/backend branch, the frozen diagnostics and the original methodology/investigation branches remain untouched. No push, Delta access, job submission, scientific choice, timeout increase or historical deletion occurred.

## Candidate and provenance

The local branch is `codex/exp12-integrated-candidate` in `/workspace/scaling-drl-exp12-candidate`, created as an isolated shared clone. A non-squashed merge `a181391ed030257cf8c72989f27e0929becd7895` joins the readiness/runtime history at `8ebda5fee04c2e2f5c88460ba7706bea667a9bd0` with the approved methodology correction commit `a3f44aa56455e3aae4b36459a78c97237bad62a8`. There were no textual conflicts; compatibility is assessed through the integrated tests, not inferred from that fact.

New engineering commits are `aad2a155d57de2336f24712ec44ec81e3d1108c9` (qualification-evidence guards and independent references) and `ae9eef356c8ac7f8ee56b8ed2b34fea473a72403` (same-setting evidence capture, all-task CPU simulation checks and resumable validation). A final report-only commit records completed verification; use the branch's actual final HEAD rather than assuming a documentation hash in this file.

Live read-only GitHub inspection confirmed `codex/a100-runtime-debug` remains published at `f6b7f73b018eac365819ce64c90d22942b211f42`, and `claude/eloquent-fermat-inxqlt` remains at `312bb01a7a6aa2e3a34a74dd557f05543081d0f5`. Neither the methodology-reconstruction branch nor this candidate is published. Original frozen `781dc6c93022a82b68a3210617db93225284b268` is preserved.

The approved twin correction is explicitly distinguished from a newly selected scientific change: each network's FP32 loss is subtracted before the twin mean, as amendment(z) requires and the lead expressly approved. Its near-zero difference can affect a HumanoidBench trigger. No additional scientific arithmetic/settings change is inferred from that approval.

## Methodology preservation

Authority remains `.claude/methodology-exp1-exp2.md` and its chronological amendments, corroborated by the approved answers in `docs/exp12_decisions.md`. Reconstruction/audit/decision documents explain provenance; they do not authorize new scientific choices. The source-level itemized matches in `methodology_implementation_audit.md` and `exp12_launch_readiness.md` remain applicable to the integrated paths below. New entropy analysis is in [entropy_temperature_reconciliation.md](entropy_temperature_reconciliation.md); the investigation and exact missing evidence are in [plasticity_validation_investigation.md](plasticity_validation_investigation.md).

| Approved contract | Integrated executable path / verification |
|---|---|
| D1W128 actor; critics D2W512/D4W1024/D4W1536;13 tasks×five seeds | `generate_manifest.py`, `configs/base_exp12.yaml`, SAC construction;195 unique parents/39 cells/130 potential scaled parents independently checked |
| Raw budgets: DMC medium500k, DMC hard/Myo1M, HB2M; repeat2/UTD2/one env | Composed canonical configs, `Exp12Trainer`; all195 budget/checkpoint intervals match the frozen revision |
| AdamW lr1e-4, weight decay .01 actor/critic and0 temperature; alpha.01; tau.005; gamma.99 DMC/HB/.95 Myo | SAC/config files byte-unchanged; analytic temperature/Polyak references, initialization/warm-up/recovery tests |
| Current negative entropy target with explicit G1 approval | `sac_agent.py:447`, `sac_update.py:156`; independently derived gradient/density; unsigned higher-level wording remains an interpretation question, not an integration blocker or an implemented correction |
| Single DMC/Myo critic; independent clipped twins for HB | SAC/twin modules; synthetic twin contracts, actual CPU HB pipeline, per-network approved probe-loss order |
| Five paired probe rounds,1,000 fresh AdamW steps,batch256,pool25,600,sin1e5 target, own fixed pre-fit offset | `experiments/exp12/probe.py`; source review, independent analytic fitter, exact pairing/RNG/state preservation and unchanged configuration hashes |
|20 scheduled checks,stored untrained critic before first update; percentile-IQM10k bootstrap95%,strict zero threshold,two contiguous eligible checks through19 | `run_probes.py`, `trigger.py`; independent sorting/quantiles/enumeration/eligibility; source and config preserved |
| Nominal Exp1 endpoint; full scaled-only Exp2 fork/control/injection/25%-budget continuation≤1.2B | `exp1.py`, `exp2_arm.py`, `fork.py`; full-state/kill-resume/last-eligible-boundary tests; strict full-population reporting |
| Nikishin frozen original head, matched fresh/copy heads, preserved trainable trunk moments and fresh new-head state | `injection.py`; identity/gradient/optimizer/target/twin tests and mutation detection; m remains unfrozen |
| Check1 FP32,control/identity exact,injected64 eps; Check2 paired report without exclusion | `fork.py`, `checks.py`; actual CPU fork pathways, malformed-evidence guard, per-Q paired references and complete-output tests |
| Training/action/probe/eval precision policy and dedicated eval RNG/restoration/schedule | Precision,SAC,trainer/fork modules; byte-preservation of precision/configs, CPU references and simulator restoration; no CUDA qualification inferred |
| Full195-parent population,eligibility/null accounting,matched arms/provenance and missing-data refusal | Completeness/ledger/analysis modules; adversarial certificates retain failed Check2, refuse missing/short/duplicate/mixed/invalid evidence; one-seed CI convention remains blocked pending decision |

`scripts/check_exp12_config_fingerprints.py` runs composition in two separate interpreters and compares every resolved parent config plus checkpoint interval/experiment, excluding only repository storage-root keys. **195/195 match `781dc6c` exactly**, aggregate SHA-256 `0b73fde152c36c3525f4c4335ed362e71cbb40b823e7ff376244cd04992d8a15`. Canonical methodology documents, configs and the entire SAC/shared-network implementation are unchanged against781. Runtime diagnostic/profiling/trace/timeout files receive no new edit in this candidate beyond the inherited reviewed commits. No threshold, statistical rule, training/probe budget, initialization order or architecture is selected/modified here.

## Demonstrated engineering changes

1. Check1 rejects broadcastable malformed shapes, wrong ranks/dtypes, empty or nonfinite panels. The prior code accepted `q.shape=(4,1)` against `(4,)` when values happened to agree. Valid panel arithmetic and64/0-eps comparisons are untouched.
2. Null resumption independently verifies saved paired losses, IQM, CI, bootstrap settings and firing flags. A contradictory saved `fired=True` with all-zero losses/CI was accepted previously; now it is rejected. Invalid-pair counts are exposed without choosing an invalid-pair acceptance criterion.
3. Canonical PC qualification verifies complete first-natural-trigger records, per-round arithmetic, CI/flags, schedule, eligibility, state step/f_star and horizons before agent creation/loading. Forced/noncanonical/test-only evidence cannot qualify m. Existing PC artifacts are preserved; runtime/source/reproduction provenance is explicit, with no unapproved reproduction cutoff.
4. BlockB range reporting checks all three required sizes rather than the observed subset. Null reporting enforces the pre-specified≥100 pairs per size and recomputes flag consistency. Descriptive rate flags and a complete population rule are distinguished; invalid-pair interpretation remains with the lead.
5. Readiness history supplies strict checkout guards, manual-only diagnostics, complete diagnostic packaging/checksums, invocation provenance and a read-only195-parent inventory. The generator's thin DONE/FORK_READY scheduling markers remain **planning aids**: use the mandatory inventory preflight before generating manifests, and the strict population certificate for analysis. Marker/header inspection does not validate every large checkpoint payload or qualify scientific results.
6. New calibration evidence retention delegates to the original same-setting probe, preserves returned arrays, captures actual teacher pools/replay/normalizer/curves/provenance, and refuses CPU full-setting A100 capture. CPU tests include a live tiny-probe unchanged-array check; no GPU capture is claimed.
7. CPU tooling checks all65 task/seed simulator cases, provides independent phase-sensitivity references, and isolates test modules with resumable, source-aware, checksum-verified exit/results receipts. Raw outputs and dependency overlays stay outside Git.
8. The original [legacy methodology/timeout audit](legacy_methodology_cleanup_audit.md) is retained with an explicit historical banner; its312bb01 body is byte-preserved. Actual Angle1 metric-flush infrastructure remains a live dependency. No historical source is deleted, and no evidence establishes that legacy scientific algorithms caused22740708.

## Completed validation

**591/591 distinct tests passed across all63 test modules, zero failures/errors/skips, exit0.** This consists of289 Exp12 methods and302 other repository methods. Independent discovery found exactly the same591 unique IDs, with no loader error or duplicate. Every completed module's process exit, successful unittest receipt, owned log SHA-256 and unchanged source fingerprint was independently verified. This is a completed module-isolated run; the interrupted monolithic run is not counted. Module isolation preserves tests/settings, avoids retaining the entire repository's compiled test state in one process, and survives environment interruption through explicit receipts. See [exp12_cpu_validation_setup.md](exp12_cpu_validation_setup.md) for reproduction/environment limits.

The tested executable revision is `ae9eef356c8ac7f8ee56b8ed2b34fea473a72403`, with executable/configuration fingerprint `014f9214f557a668a6520cb17896a2d4ee65747ea7671c9c43b8046ffcfff46b`. The final commit only adds/annotates reports; its identical executable/configuration bytes are verified separately. Do not claim that GPU evidence from781/312/f6 qualifies this integrated source.

The earlier187+2 expectation described an older Exp12 subset. The current run includes the expanded baseline, approved correction regressions, readiness/evidence/tooling regressions and every other test module; it does not lower assertions or count supplemental reruns as new methods. Both formerly skipped HB methods actually ran, as did the additional conditional HB foundations subtests. The upstream dependency-lock limitations remain explicit.

Also completed: **73/73 mutation checks with HB available**, no PROBLEM, exit0 (all72 original mutations retained plus the approved twin arithmetic mutation);195 exact config fingerprints and independent census;68 bootstrap/IQM fixtures plus four exact enumerations and all nine historical range rows; independent fitting/entropy/Polyak/RNG tests; actual HB's two previously skipped tests (2/2,115.915s) and the HB-enabled foundations module (12/12,429.182s, including additional conditional subtests); all13 tasks×five seeds=65 exact simulator/action-RNG/next-transition cases. Black/pyflakes on all ten new analysis/tool/regression Python files, relevant launcher/shell bash-n and Git whitespace checks pass. These are CPU engineering checks, not full-budget training or CUDA/scientific qualification. Supplemental repeated tests are not added as new distinct methods.

Generated receipts/logs remain outside Git under `/workspace/scratch/exp12-integrated/`. Reproducible source, tests and reports alone are committed. Evidence hashes:

| Evidence file | SHA-256 |
|---|---|
| `isolated-cpu/summary.json` | `d347b25f67419c2225170b04a2fa66fbb50f9cbdf9faed6d46e9ddde47fb9bcd` |
| `isolated-cpu/modules.json` | `50c031794d449a1c15ad3d6b19d83961eff0f29e1a6e1d49f5cebdcd7961d62f` |
| `isolated-cpu/provenance.json` | `6ed3d4eb97fbc40ef7b17ffd1a709d10b6eefbc8382388b4ec5f86d51571f406` |
| `discovery-final.json` | `6f0f0b097ba0c9c66e5db6a2971bf554374ddfff74db7079b49a4e501382f574` |
| `break-checks-hb.log` | `e0567d98efaf1881e40d69872c81ec7ff2c4b2cb96b02504d341b7f90f230ed0` |
| `config-fingerprints.json` | `e611e09b59d7e6db7e94d6c8ad5c9c49ea81053b3b3af53eb7401824a5bc3ad0` |
| `plasticity-reference.json` | `fbeea7ea82d8cbafc35846fdca5363f94eadc00060c5ace4654edf0e0c27d3c7` |
| `cpu-phase-sensitivity.json` | `45a35a15a881fa5219ec460d91d9febe3d149b11d11fdace233d52e8a63acc9d` |
| `cpu-environments-65.log` | `1ddea4ed8d34d506037d508c9e5ab3110c6fbf4c972f0fd4392a0ff897b65156` |

## Readiness and remaining blockers

| Item | Implemented | CPU verification | GPU/scientific status |
|---|---|---|---|
| Reviewed integrated candidate/history/provenance | YES |591/591 tests, complete receipts/checksums verified | READY for engineering review; not production/main |
|195 parent configs / fixed actor / budgets / seeds | YES |195 exact matches | Entropy wording clarification and final-source qualification remain |
| Probe/trigger/twin/endpoint implementation | YES | Independent references and adversarial coverage | Fresh-null13% and hopper.602 gates failed historically; final-source CUDA qualification absent |
| Fork/complete restore/control/identity/injection/Check1/Check2 | YES | Tests plus all-task simulator coverage | Both scaled sizes in each suite on approved GPU still required |
| Real HB simulator paths | YES | Actual H1 reach/run CPU checks; upstream-lock differences explicit | CPU adapter coverage does not qualify production-size CUDA twins/environment |
| Natural PC / m / successful-rescue interpretation | YES | Arithmetic and input/state/identity/optimizer/forced-refusal paths | No qualifying natural full-setting PC supplied; lead must freeze m and decide any additional rescue/reproduction criterion |
| Result completeness / exact-source scheduling / recovery | YES | Adversarial certificates and marker inventory | Mandatory preflight; real large-payload/shared-filesystem recovery still unverified |
| Runtime stage diagnosis and evidence packaging | YES | Observer/provenance/packaging tests |22740708 timed out; exact blocking operation unresolved; six-hour failure relation unproven |
| Statistical reporting | YES | Independent aggregation/pairing/completeness checks | One-eligible-seed uncertainty convention needs approval; no silent missing-result acceptance |
| CUDA numerical qualification / resources/cache/packing/storage | Diagnostic tools exist | Structural/finite/dtype/integer guards verified | Four numerical bounds remain unset; actual A100/HB performance/memory/packing/source-specific identity evidence required |
| Production launch | Candidate prepared | CPU evidence does not authorize launch | BLOCKED; no campaign/job authorization and no artificially waived gate |

## Review and deployment plan

1. Review this separate candidate's retained merge history and engineering diff. Keep781/312/f6 unchanged for comparisons. No merge into main/launcher or push is performed.
2. Transport the local candidate Git bundle into the normal repository, verify its checksum/prerequisites/full commit and create a separate review branch. Do not force-push or substitute an unreviewed branch by name. Final commands are in [exp12_integrated_validation_commands.md](exp12_integrated_validation_commands.md).
3. The next already-reviewed runtime comparison remains **published f6**, profile mode/D4W1536,300-second internal timeout/7-minute Slurm limit. The integrated production candidate is a different source identity. Obtain the full boundary/flush/watchdog/error trace; do not increase budgets or infer GPU outcomes.
4. After approval, run unchanged final-source qualification and the minimum same-setting range capture. Recover raw null records first; obtain a natural full-setting PC (or train its unchanged development run if no valid fork exists). A PC recommendation/exit0 is not lead-frozen m or proof of positive rescue.
5. Obtain/approve the genuine final-source numerical, identity, environment, cache/resource/packing/storage/runtime evidence and scientific decisions. Freeze source/config/runtime/master195 manifest and m. Preflight existing roots before generation; preserve the immutable master population and explicit eligible/null-arm census. Use strict complete-result analysis, including Check2 failures.
6. Publish/deploy/launch only after explicit authorization. No deadline substitutes for the failed measurement or GPU gates.

Approval priorities are: (a) failed measurement gates and any altered discriminating experiment; (b) interpretation of unsigned entropy wording while preserving G1 until clarified; (c) genuine natural PC, any extra rescue/reproduction criterion and m freeze; (d) four CUDA diagnostic bounds, one-seed reporting and evidence-supported deployment policy. Integration and local CPU progress are **not** blocked by these pending choices.

## Production-launch checklist

| Gate | Current status / required evidence |
|---|---|
| Authoritative amendment/decision mapping; approved correction provenance | DONE; no new scientific decision selected |
| Separate integrated revision, retained histories and final engineering diff review | DONE locally; normal-repository review/publication not performed |
| Complete current CPU discovery, dependency skips accounted for, independent references/mutations | DONE;591/591,zero skips,73/73 mutations; actual HB CPU coverage |
|195 unique canonical parent commands,39 cells,five seeds,budgets and scientific config fingerprints | DONE; freeze immutable campaign master after final approval |
| Same-budget f6 runtime stage boundaries/stacks/synchronization evidence | PENDING approved A100 run; exact blocker unresolved |
| Final-source probe/trigger/range/null and numerical CUDA qualification | BLOCKED by failed historical null/range gates, absent final-source GPU evidence and unset approved comparison bounds |
| Canonical natural development fork; full PC/noise/sensitivity evidence; lead-frozen m | PENDING; no forced/reduced substitute or silently added rescue threshold |
| Complete-state/checkpoint/identity/injection for both scaled sizes in each suite on the production GPU | PENDING GPU evidence, actual HB production-size twins and cache on/off equivalence |
| Environment, runtime, precision/cache policy and exact GPU model fixed per suite/source | PENDING final-source GPU dependency/runtime qualification and approval |
| VRAM/host memory,packing/concurrency,wall time,shared-filesystem save/restore/storage/quota,recovery | UNVERIFIED in production; declarations and tiny CPU restore are not measurements |
| Inventory preflight before queue generation; state/metadata/payload and claim reconciliation on recovery | Tools/CPU guards DONE; actual production payloads/claims not supplied; preserve and reject incompatible states |
| Frozen source/config/runtime/study master,eligible/null-arm census and paired-arm provenance | Tools DONE; actual frozen campaign artifact cannot be fabricated before m/runtime/source approval |
| Strict complete-population reporting,Check2 failures retained,one-eligible-seed uncertainty convention | CPU guards DONE; lead decision on one-seed convention remains |
| Explicit publication/deployment and job/campaign authorization | NOT GIVEN; no push or submission performed |

No pending gate is converted into READY by the engineering integration or by the deadline.
