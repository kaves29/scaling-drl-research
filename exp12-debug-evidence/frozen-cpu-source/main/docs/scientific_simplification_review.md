# Scientific simplification review

Scope: the isolated successor to approved `781dc6c`, not the checkout used by the queued diagnostic. Simplification is justified by approved equations and state/stream invariants, not by easier test passage. [Reconstruction](methodology_reconstruction.md) defines the contracts; [audit](methodology_implementation_audit.md) identifies source computations.

## Implemented reductions and their preservation arguments

| Change | Mathematical / numerical contract | RNG, optimizer and checkpoints | Eligibility, estimands and reporting | Reproducibility |
|---|---|---|---|---|
| One `probe.paired_loss` for longitudinal L and validation | Single subtraction unchanged. Twin computes **literal mean of per-network losses**, as (z). This is an approved scientific correction, **not** a bit-preserving refactor. Independent FP32 scalar fixture detects old order | Reads existing host score arrays only; no fitting, target, sampling, RNG or optimizer call changes | HB L/near-zero trigger may change, explicitly approved. Check2 primary remains mean-P difference; ancillary L uses same corrected contract | Approval and source revision documented; old results stay identified by old revision |
| One approved range rule via `range_criterion_rows` | Finite, unique configured ratio≥.9; no P,b,fit/pool computation changes. Old10–90% retained solely as information | Verdict runs after identical probing; no extra random draw/state change | Valid reported BlockB rule unchanged. Standalone verdict/exit now agree; missing/ambiguous inputs cannot falsely pass. BlockB still labels rule failure separately from crash | Exact JSON evidence/CLI semantics; existing historical evidence not rewritten |
| PC input/qualification guards | Finite four five-round series; valid pooled SD, IQM recovery and ranking expression/order unchanged. **No minimum recovery gate added** | Guard invalid inputs/configs; no valid probe initialization/order/state change | Canonical dev run/natural trigger preserved. `--allow_any_setting` remains useful for toy diagnostic reports but never freezes/qualifies m | Report explicitly records test-only/forced status; old measured recovery values retained |
| Phase labels in Exp2 loader/plot | No numerical transformation, sample addition or replacement | Read-only derived DataFrame; stored arm CSV bytes unchanged | Distinguishes inherited pre-injection longitudinal fork point from post-injection values. Check2 remains the actual at-fork post-injection comparison. Raw primary return statistics unaffected | Hollow labeled point preserves visibility of original sample; source evidence unchanged |
| CUDA helper validates before casting | Original finite-float max-absolute/scale formula unchanged. Counts/shapes/dtypes must agree; integer/bool exact; nonfinite model state rejected; only named diagnostic NaN masks allowed | Test comparator only; no production RNG/model/update/restore change | Prevents false oracle success/skips; bounds still None; no selected numerical threshold | Measurements remain comparable for valid float arrays; structural/integer/nonfinite failures are explicit |
| Pointer/replay/null/manifest guards | Valid restore arrays/order and valid null measurements unchanged | NPY header validation precedes mutation without allocating a second replay; original restoration assigns same state, consumes no RNG. Invalid pointer rejected | No new experiment-eligibility/statistical rule. Stale/duplicate/unknown artifacts stop explicitly; no valid no-trigger parent is excluded | Preserve old manifests/evidence; generate commands in fresh directories; exact null config/commit/runtime provenance required for resumption |

These changes introduce no general-purpose framework. The shared functions are small rules already used by multiple scientific outputs. Input guards distinguish invalid evidence from a failed approved scientific criterion; they do not turn failures into successes.

The validator now calls the shared longitudinal-loss function, avoiding divergent production/validation equations. This intentionally increases common-code dependence, so the focused operation-order test uses scalar NumPy float32 operations independently. The CPU bootstrap reference sorts resampled rows and explicitly averages the middle three; it does not call the production IQM function. Existing SAC tests calculate minimum targets and losses directly; the fetched released SAC source independently corroborates equations. Validation agreement alone is not proof of correctness.

## Necessary complexity retained

- **Complete state restoration**: simulator/reset/action-space RNG and wrapper counters, replay n-step queue/index, observation RMS, JAX and global RNGs, optimizer moments/counts and diagnostics windows are required for a causal fork. Replacing them with a seed or model-weights-only save changes trajectories.
- **Original → saved control → identity comparisons**: comparing only two restores could miss a defect common to both. Original/control panel and source comparisons remain.
- **Separate trunk/new AdamW state**: one common count or reset trunk moments changes bias correction. No generalized optimizer rewrite is warranted.
- **Grouped old+(new−copy)**: algebraically equivalent reassociation can change finite-precision identity. It is preserved.
- **Separate per-round/per-Q evidence**: aggregate-only storage cannot diagnose pairing, cancellation/order, twin-network errors or nonfinite inputs. Preserving evidence is more useful than reducing columns.
- **Independent population/provenance certification**: scanning only available directories silently shrinks the study. The explicit195/130 expectations, endpoint/schedule checks, staged output and source-change detection remain despite their length.
- **Dedicated RNG streams and isolated evaluation**: replacing streams with one global generator, merging repeated evaluations or caching stochastic target draws can alter the experiment.
- **Different bootstrap designs**: independent architecture/environment-stratified Exp1 resampling, paired-round trigger/Check2 resampling and paired-seed Exp2 trajectory resampling represent different units. One generic bootstrap interface risks accidental pairing or changed draws.
- **Different precision scopes**: training TF32, Check1 highest and diagnostic-only policy forwards highest are approved distinctions. Consolidating to a blanket FP32 context would change training/probes.

## Candidates reviewed but not implemented

| Candidate | Why preservation is not established / why deferred |
|---|---|
| Merge JAX fitting/forward graphs or change chunk sizes to reduce compilation | Mathematically equivalent GPU graphs can change TF32 kernels, sine targets, operation order and state identity. Needs numerical evidence and approval where scientific outcomes could differ |
| Replace explicit injection optimizer with a generic partition framework | Current custom state count/moment transfer is auditable and covered; no demonstrated simplification benefit outweighs checkpoint compatibility risk |
| Replace seeded paired bootstrap with rliable universally | Approved details explicitly specify NumPy Generator resampling for the trigger (D B1); changing implementation/stream consumption can change intervals. Exp1 already uses rliable's approved stratified design |
| Remove checkpoint/probe/evaluation columns or source snapshots | Loses independent evidence needed to certify causal comparisons and completeness |
| Recalculate old twin L/trigger tables using the corrected order | Would silently replace historical results and potentially change eligibility. Must preserve original revision and seek approval for any reanalysis plan |
| Infer post-injection k=0 longitudinal L from inherited pre-fork row | Incorrect measurement phase; use the actual Check2 evidence instead of inventing a sample |
| Remove old range-rule evidence from historical reports | Loses provenance. Only active acceptance/exit logic was corrected |
| Automatically delete stale DONE/fork/manifest/cache files | Can destroy user evidence or obscure an interrupted run. Current guards stop and preserve files; scheduling classification still needs source-aware review |
| Fully decode all multi-GB checkpoint tensors at each analysis invocation | Stronger payload certification is useful, but repeated large restores are not a simple neutral improvement. Keep runtime restore checks and state comparisons; propose an explicit integrity workflow after reviewing costs |
| Set positive-control recovery>0, calibrate null threshold to p95, increase probe fitting budget or reduce pool | These are scientific decisions, not simplifications or pass-rate repairs. No change made |

The implementation is simpler where rules previously conflicted, while retaining the scientific distinctions needed for reproducibility. Remaining scientific decisions are listed separately; unresolved gates cannot be removed through refactoring.
