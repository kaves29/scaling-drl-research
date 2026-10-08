# Approved methodology reconstruction

**Revision context:** this reconstruction describes the separately approved a3 corrections, now retained in the isolated integrated candidate. See [exp12_integrated_readiness.md](exp12_integrated_readiness.md) for current engineering evidence and [entropy_temperature_reconciliation.md](entropy_temperature_reconciliation.md) for the more precise interpretation of the unsigned entropy wording. No authoritative amendment or implementation setting is changed by these contextual links.

Reconstructed on 2026-10-07 from the approved sources, independently of test outcomes. Baseline: `781dc6c93022a82b68a3210617db93225284b268`. Work is isolated on `codex/methodology-reconstruction`; the pending diagnostic's checkout/revision is unchanged. This document specifies the science; [the audit](methodology_implementation_audit.md) describes implementation and evidence.

## Authority and unresolved conventions

1. `.claude/methodology-exp1-exp2.md` (M), including amendments (a)–(z), is authoritative. Superseded text is historical, not an alternative setting.
2. `docs/exp12_decisions.md` (D), especially the approved answers at lines 362–422, 590–615, 683–715, 1008–1065, and the pre-result decision table at 1734–1762, supplies implementation details and gates.
3. The lead's explicit 2026-10-07 approval authorizes the literal twin loss correction: per-network float32 subtraction, then averaging. It does not change any threshold, fitting, RNG or Check 2's comparison of mean scores.
4. Released SimBa `SonyResearch/simba@7d0358b` and the research definitions quoted in D clarify semantics, without overriding approval. The released SAC update and config were independently retrieved over HTTPS for this audit. SHA-256: update `220efe7122b69d72043a2fc41b416c382cedfd59cfa92aba590f320fd3075dbc`; config `c74f18920f66a8b3691d91fe29ec6d7e06f28aec782c9a3f557a7e2e6038cbe8`.

**Entropy conflict remains open.** M:59 specifies `|A|/2`; D:303–311 (G1) specifies `−|A|/2`. Released SimBa's config also uses `−0.5`, and its objective is `alpha * (entropy − target_entropy)`. The negative setting therefore targets negative differential entropy; this cannot be dismissed as a notation-only sign conversion. No implementation change is authorized by resolving this conflict ourselves.

## SAC equations and execution order

Let `alpha = exp(log_alpha)`, `H = −mean(log pi(a|s))`, `d` be actual termination, and `Qmin` be Q for a single critic or `min(Q1,Q2)` for a twin. The inherited G1/released-SimBa equations are:

- Actor: `J_pi = mean(alpha * log pi(a|s) − Qmin(s,a))`, with reparameterized tanh-Gaussian actions.
- Critic target: `y = r + gamma * (1−d) * (Qtarget_min(s',a') − alpha * log pi(a'|s'))`, n-step = 1. Time-limit truncation bootstraps; termination does not (M amendment (s)).
- Single critic: `J_Q = mean((Q−y)^2)`. Twin: `J_Q = mean((Q1−y)^2 + (Q2−y)^2)`, the inherited released-SimBa sum, not a mean over the two networks. Each uses the shared minimum-based target (M (z)).
- Temperature: `J_alpha = alpha * mean(H−h*)`; its derivative with respect to log alpha is `alpha*(H−h*)`. Current approved G1 setting is `h*=−|A|/2`, subject to the conflict above.
- Polyak: `Qtarget_params <- .005 * Q_params + .995 * Qtarget_params` after each critic update.

Each interaction performs two sequential SAC updates once replay reaches minimum length. Within each update: actor uses old Q/alpha; temperature uses that actor objective's sampled entropy; critic uses the updated actor and updated alpha with the existing target critic; then Polyak updates the targets. Training RNG splits and scan sequencing are preserved. This order is inherited behavior, not a newly chosen simultaneous-update scheme.

Actor/critic AdamW: learning rate `1e-4`, weight decay `.01`, betas `(.9,.999)`, epsilon `1e-8`, epsilon-root 0, default Optax AdamW behavior on all trainable parameters. Temperature AdamW: `1e-4`, weight decay 0, initial alpha `.01`. Batch256; replay capacity1M/minimum5000; observation running mean/variance normalization retained. Target tau `.005`; action repeat2; UTD2; five confirmatory seeds1–5. During warm-up, action-space random actions override policy actions while observation statistics continue updating (M (h); D G1 and approved follow-up).

The fixed SimBa residual actor is D1W128. Critic network sizes are D2W512, D4W1024, D4W1536 (M:34–44, (y)). DMC/MyoSuite use one Q; HumanoidBench uses two independently initialized networks of that size and two targets (M (z)). Tanh-Gaussian log-standard-deviation mapping remains the inherited `[-10,2]` mapping; no mixed precision or optimizer replacement is introduced.

## Environments and budgets

Budgets count raw environment steps; one interaction uses two raw steps. Every row has all three critic architectures and seeds1–5: `13 × 3 × 5 = 195` parents. Sources: M:65–90, amendments (a), (i), (s), (y), (z); D approved G2 follow-up and task mapping.

| Repository task | Registered/semantic task | Raw budget | Interaction budget | Outer raw horizon / gamma |
|---|---|---:|---:|---|
| dog-run | DMC dog run | 1,000,000 | 500,000 | 1000 / .99 |
| dog-trot | DMC dog trot | 1,000,000 | 500,000 | 1000 / .99 |
| humanoid-run | DMC humanoid run | 1,000,000 | 500,000 | 1000 / .99 |
| humanoid-walk | DMC humanoid walk | 1,000,000 | 500,000 | 1000 / .99 |
| humanoid-stand | DMC humanoid stand | 1,000,000 | 500,000 | 1000 / .99 |
| swimmer-swimmer15 | DMC swimmer swimmer15 | 500,000 | 250,000 | 1000 / .99 |
| hopper-hop | DMC hopper hop | 500,000 | 250,000 | 1000 / .99 |
| myo-key-turn | myoHandKeyTurnFixed-v0 | 1,000,000 | 500,000 | 100 / .95 |
| myo-pen-twirl | myoHandPenTwirlFixed-v0 | 1,000,000 | 500,000 | 100 / .95; inner registered limit50 retained |
| myo-pose-hard | myoHandPoseRandom-v0 | 1,000,000 | 500,000 | 100 / .95 |
| myo-reach | myoHandReachFixed-v0 | 1,000,000 | 500,000 | 100 / .95 |
| h1-reach-v0 | no-hands H1 reach | 2,000,000 | 1,000,000 | 1000 / .99 |
| h1-run-v0 | no-hands H1 run | 2,000,000 | 1,000,000 | 1000 / .99 |

Discount is `clip(1−5/L, .95, .995)`, `L=max_episode_steps/action_repeat`. The MyoSuite outer100 and PenTwirl inner50 are explicitly approved SimBa behavior, not accidental replacements. Shelf-place and the old D6W1536 size are superseded; their historical evidence remains identified by the size/task actually measured. HumanoidBench registration/stepping on the production GPU installation remains a validation prerequisite.

## Exp1 probe, trigger and scientific endpoint

Sources: M:91–124, (c), (l), (y), (z); D A1–A9, B1–B5, C1–C3, quoted Lyle definitions at 423–458.

1. Immediately before the first SAC update, save the run's untrained critic and perform check0 after random replay warm-up. This same fresh parameter reference is used throughout the run.
2. Schedule checks1–20 at `k*N/20` interactions. Each check contains five rounds; each round samples25600 replay `(observation, action)` pairs **with replacement**, normalizes observations with the run's current statistics, and uses dedicated probe RNG streams.
3. Initialize a same-architecture target network once per round and compute `base(x)=sin(100000*f(x;w0))`. For each probed critic independently, freeze its pre-fit pool mean `a`; regression targets are `a+base`. Current and fresh share inputs, base targets and all1000 minibatch index arrays, batch256; their constant offsets may differ.
4. Fit **copies** for1000 optimizer steps with fresh critic optimizer state. For an injected critic, keep the injection trainability mask and optimizer partitioning. Final MSE is measured over the entire pool in chunks2560, rather than the final minibatch. Baseline `b=Var(base)`; `P=b−final_pool_MSE`; single-network `L_r=P_fresh,r−P_current,r`. Positive L means worse plasticity than the same run's fresh reference. No division by b in longitudinal L.
5. Twins probe each Q against its own fresh Q on the same targets/order; each Q has its own offset. Literal float32 order: `L_r=mean_q(P_fresh,q,r−P_current,q,r)`. Aggregate P is separately the mean of Q scores. All per-Q observations are retained. Check 2's different score estimand remains unchanged.
6. Check statistic: IQM, `trim_mean(.25)`. With five values it is the average of the sorted middle three. Resample paired per-round L10000 times; compute IQM of each bootstrap sample; percentile95% interval. A check fires iff finite lower bound **strictly >0**. Dedicated bootstrap seed stream is independent of training/probing.
7. `f*_run` is the check completing the first two consecutive firing checks. Its completion must be within checks2–19, i.e. at or before95% of N. A missing/invalid check cannot bridge the sequence. Nonfinite checks are marked invalid and do not fire (B5). Never-triggered-by95% is an explicit valid null eligibility result, not an omitted run.
8. Primary Exp1 endpoint is **check20 at nominal N**, even if restored control training continues beyond N. For each scaled architecture, estimate `IQM(final L across65 runs) − IQM(default final L across65 runs)`. rliable stratifies by13 environments, resamples architectures independently, uses50000 percentile-bootstrap replications and95% intervals. Do not pair architectures merely because their seed integers match. Trajectories, each seed, f* and probe curves are descriptive; no new slope endpoint or trigger-to-population inference.

## Exp2 causal intervention and reporting

Sources: M:137–222, amendments (b), (d), (e), (g), (m)–(r), (t), (u), (z); D E/F/I and approved answers.

Only scaled parents with eligible f* fork: at most130 candidates. At the exact fork step, save **complete state**: online/target Q, actor, temperature, optimizer moments/counts, JAX/training/NumPy/Python RNGs, replay arrays/index/n-step queue, observation statistics, full simulator/wrapper/reset/action-space state, current observations/timestep, update counters, diagnostics/reference windows, metrics and probe records. Restart the original process through the same saved-state restore path as the independent arm. Original continuation is control, to `max(N, f*+.25N)`. Injection continues exactly `.25N` after f*, never beyond `1.2N`.

Injection: shared trunk plus `old(z)+(new(z)−copy(z))`. Head = last m residual blocks, post-LayerNorm and output layer. Preserve old head; initialize new; copy new bit-for-bit. Freeze old/copy **parameters**, while gradients through their input still reach the trunk. Preserve trunk optimizer moments and count; start new-head optimizer fresh; actor/temperature unchanged. Target uses its own preserved trunk/old head and copies the online new/copy initialization. Polyak update continues over the complete target tree. Twins inject both Q networks and both targets, same m with independent initialization keys (M (z)).

The unshrunk D4W1536 dog-run development run uses a seed outside1–5 and the natural approved trigger. Healthy reference is its fresh critic, `L_healthy=0`. Compare degraded plus injected last/half/all on shared probe streams. `recovery=(L_trigger−L_injected)/L_trigger`. Noise is pooled **within-series** SD of the four five-round loss series divided by L_trigger. Stop if no eligible trigger, `L_trigger<=0` or noise≥.10. Select the smallest head whose recovery is within.10 of best; report all candidates. Forced/test-only, incomplete or nonfinite evidence cannot qualify m. Shared-offset sensitivity repeats are informational and excluded from this noise statistic. The lead freezes m before confirmatory Exp2; no m is selected by this audit. A separate minimum successful recovery criterion has not been approved.

Check1: one fixed256-row normalized replay panel, local highest/FP32 Q and dQ/da. Original/control/identity must agree bit-for-bit. Injected comparison allows64 float32 eps times the pre-injection maximum magnitude, separately for Q and gradient; zero reference scale implies zero allowed absolute difference. Twins evaluate both Qs and gradient of their minimum. Failure stops before continuation training. Check2: paired per-round `mean_q P_injected−mean_q P_control`, IQM and95% interval over five rounds; pass iff finite lower bound>0. It is **reporting only**, never a primary population filter. Injection's inherited longitudinal fork row is pre-injection; actual post-injection-at-fork probe is Check2, not an invented longitudinal sample.

Post-fork evaluation at `j*N/100` since fork, `j=0..25`, ten deterministic-policy episodes at every point, independent evaluation RNGs/envs with training RNG states restored afterward. Record raw episodic return, lengths and Myo success. Primary report is per-environment raw-return injected-minus-control trajectories: episode means within each seed, IQM across eligible paired seeds,10000 percentile-bootstrap replications over seeds, identical resampled seed indices at every timepoint. Report every seed and the complete eligibility census. No cross-environment normalized pooling or new scalar endpoint. Check2-success-only plots are secondary and labeled; failed Check2 remains in primary. One eligible seed lacks an approved inferential uncertainty convention; current confirmatory plotting stops explicitly.

Supporting actor diagnostics: `KL(pi_new || pi_old)` of diagonal pre-tanh Gaussians on256 replay reference observations refreshed each2000-interaction logging window with dedicated RNG; existing deterministic-action L2 churn; per-update gradient norm/window population SD; sampled mean|action| and fraction|action|>.99. Actor-gradient cosine remains supporting/future-mechanism evidence. They are not independent causal proof.

## Precision, integrity and gates

M (v)/(x): GPU float32 matmuls use TF32 for training, sampling, probing and evaluation; no bf16/fp16, x64 off. Check1 uses local highest/FP32. Only diagnostic churn/KL policy forwards additionally use local highest/FP32; their non-forward arithmetic and sampling semantics are preserved. CPU cannot qualify CUDA arithmetic. Record GPU model/JAX/jaxlib/CUDA runtime; the **entire grid and both arms use one GPU model** (M (u)), not merely equal models within a pair.

M (t): routine checkpoints everyN/20, never reuse state directories, publish LATEST atomically only after save, retain latest routine + fork + fresh. Resume restores all state rather than rebuilding optimizer/replay. Confirmatory analyses require a separately frozen full-grid config/revision/GPU/runtime manifest and complete source evidence; missing populations/endpoints/forks/evaluations, duplicates and unapproved settings are rejected. Exploratory output is explicitly uncertified; B5 invalid intermediate checks are reported and cannot fire, while initial/final required endpoints must be finite.

| Gate / purpose | Approved acceptance / disposition | Source |
|---|---|---|
| CPU/GPU invariants and mutation sensitivity | All applicable assertions pass; every mutant fails and restored invariant passes; dependency skips are not passes | D validation plan / lead break-check requirement |
| CUDA identity per scaled size/suite | Complete restored-control/identity state agrees bit-for-bit on production GPU model | M (b), (p), (u) |
| Check1 injection identity | Control/identity exact; injection≤64 eps at FP32; otherwise stop | M (m), (v), (z) |
| Fresh-critic configured-pool fit | IQM(P)/IQM(b)≥.9 at every approved size; no automatic fallback | M (w), (y); D:1745 |
| Fresh-pair operational-null noise | Per-check fire rate≤5%; any excess goes to the lead, no automatic p95 threshold | D:683–715, 1746 |
| PC / freeze intervention | Natural qualifying dev fork; finite five-round inputs; L>0; pooled noise<.10; approved smallest-m rule; lead freezes m | M (d), (e), (q), (y) |
| Check2 rescue evidence | Lower95% bound>0; report failure and retain primary fork | M (n), (z) |
| Actor KL GPU numerical check | Existing well-conditioned tolerance rtol1e-4; near-deterministic case informational | D:1743 and approved Block B answers |
| Diagnostic on/off numerical comparisons | Four bounds remain None; measurement is not a pass, and the proposed10× conversion is unapproved | D:1742 |
| Production preflight/runtime/capacity | Preflight for each size and CUDA identity; no approved packing, wall-time or storage numerical threshold | D:1757–1762 |
| HB execution coverage | Requires registered/stepping production dependencies and twin validation; UNAVAILABLE in Block B does not certify HB | M (a), (k), (z); D:1761 |

The five-minute A100 runtime diagnostic has its own frozen revision and an engineering purpose. Its queued result is unknown and does not replace the scientific gates above.
