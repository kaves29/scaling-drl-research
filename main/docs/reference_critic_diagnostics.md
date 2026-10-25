# Reference-backed critic diagnostics v1

24 October 2026. Feature branch starts at audit PR13
`5179c43b4a45f6d9032a385adbf72713e2456373`. Original campaign source:
`f597ee9898e4d401912092c336c3e5330cf2fda1`.

## Findings and attribution

**EchoCritic shares the scaling implementation's dormancy and rank definitions.**
The scaling and EchoCritic `get_dormant_ratio` functions are AST-identical.
The `get_srank` mathematical bodies are AST-identical after removing EchoCritic's
CPU-device-placement wrapper. Both call sites capture Dense-module outputs and
use final encoder features for rank. Gradient norm is likewise the global
pre-AdamW training-loss gradient norm. These are shared/inherited measurement
semantics, **not independently discovered EchoCritic bugs**. Code identity and
chronology do not prove unique file-copy genealogy versus a common upstream.

The original metrics remain meaningful with their actual definitions. Calling
the singular-value count “conventional stable rank,” or interpreting the Dense
preactivation ratio as all post-ReLU inactivity, would be naming/interpretation
errors. The added measurements are alternatives and missing observations; they
are not repairs chosen to make rank and probe results agree.

All three authors' repositories were fetched and pinned; actual definitions
and call sites were read. The paper URLs returned HTTP403 through this Cloud
network. **Paper-to-code agreement is unverified**; no equation/page attribution
is invented. Repository/readme association verifies which authors' implementation
is being compared, not whether its paper uses the exact same edge-case policy.

| Reference | Pinned revision / definitions / call sites |
| --- | --- |
| [Network Sparsity paper](https://arxiv.org/abs/2506.17204) / authors' code | `lilucse/SparseNetwork4DRL` **613a2b9e5e51fafa56d66699e5e2e87d999a8905**. [metrics.py:88](https://github.com/lilucse/SparseNetwork4DRL/blob/613a2b9e5e51fafa56d66699e5e2e87d999a8905/scale_rl/networks/metrics.py#L88), [rank:237](https://github.com/lilucse/SparseNetwork4DRL/blob/613a2b9e5e51fafa56d66699e5e2e87d999a8905/scale_rl/networks/metrics.py#L237), [capture/calls:195](https://github.com/lilucse/SparseNetwork4DRL/blob/613a2b9e5e51fafa56d66699e5e2e87d999a8905/scale_rl/agents/sac/sac_update.py#L195), [residual block:31](https://github.com/lilucse/SparseNetwork4DRL/blob/613a2b9e5e51fafa56d66699e5e2e87d999a8905/scale_rl/networks/layers.py#L31), [gradient timing:82](https://github.com/lilucse/SparseNetwork4DRL/blob/613a2b9e5e51fafa56d66699e5e2e87d999a8905/scale_rl/networks/trainer.py#L82), [replay call:143](https://github.com/lilucse/SparseNetwork4DRL/blob/613a2b9e5e51fafa56d66699e5e2e87d999a8905/run.py#L143). |
| [Dormant Neuron paper](https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf) / ReDo | `google/dopamine` **5873f5494ee0c2d7c016d0ab2ad530354fec59d0**. [score:303](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/weight_recyclers.py#L303), [aggregation:274](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/weight_recyclers.py#L274), [post-ReLU DQN sites:64](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/networks.py#L64), [residual sites:120](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/networks.py#L120), [replay/capture:238](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/recycled_dqn_agents.py#L238), [default statistics batch:98](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/recycled_dqn_agents.py#L98). |
| [Loss of Plasticity paper](https://www.nature.com/articles/s41586-024-07711-7) / authors' code | `shibhansh/loss-of-plasticity` **a6b79580d85f3025bdb601566d3627c5f489f13b**. [rank helpers:143–214](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/utils/miscellaneous.py#L143), [MNIST calls:141](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/permuted_mnist/online_expr.py#L141), [post-activation features:61](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/nets/deep_ffnn.py#L61), [RL call:219–254](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/rl/run_ppo.py#L219), [PPO gradient/clipping:141](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/algos/rl/ppo.py#L141). |

Original EchoCritic references are in the [source-linked audit map](forensic_probe_dataflow.md).
Current pre-feature source matches that historical metric implementation.
Pinned file hashes and reference excerpts/fingerprints are under
`main/tests/reference_critic_fixtures/`; licensed ReDo/LoP excerpts and license
texts are retained for offline CPU formula tests. Tests execute only these small
formula excerpts, not reference training/recycling programs. LoP arithmetic runs
through a documented minimal NumPy backend adapter; this is not Torch runtime
or GPU qualification.

Additional inspected call sites: ReDo's [TF-Agents SAC capture/construction:78–124](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/tfagents/sac_train_eval.py#L78),
[linear output head:189](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/tfagents/sac_train_eval.py#L189),
[capture timing:387](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/tfagents/sac_train_eval.py#L387),
[score/aggregate:518–604](https://github.com/google/dopamine/blob/5873f5494ee0c2d7c016d0ab2ad530354fec59d0/dopamine/labs/redo/tfagents/sac_train_eval.py#L518),
and LoP's [MNIST plot normalization:22–27](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/permuted_mnist/plots/bp_metrics.py#L22).

## Formula and context comparison

Let `a[i,j]` denote activity of unit j over examples/spatial positions i;
`u[j]=mean_i(abs(a[i,j]))`, `s[j]=u[j]/(mean_j(u[j])+1e-9)`.
Let `σ` be the descending nonnegative singular values of examples×features X.

| Measurement / implementation | Formula and units | Normalization, thresholds, zero behavior | Locations / included layers / inputs | Gradient timing / paper agreement |
| --- | --- | --- | --- | --- |
| EchoCritic original/current and scaling dormancy | `100*count(s<=τ)/units`, per layer and concatenated unit masks | Layer-relative activity; τ=.1/.2 in critic call. Denominator1e-9; all-zero layer100%. Historical τ0 path uses `isclose`, not strict elementwise-zero. | All captured non-predictor Dense outputs, including projection, expanded pre-ReLU and contracted residual affine. Current normalized replay batch256. Scaling structural call every5,000 interactions; EchoCritic every2,000. | Not a gradient; shared formula/hooks verified. Exact paper correspondence unverified. |
| ReDo default score/ratio | Same relative absolute-activity formula, percentage unit-weighted total | Inclusive `<=dead_neurons_threshold`; constructor default0; configurable threshold. Optional `sub_mean_score` and CBP branches exist; v1 ports default `False`, not the centered/CBP branch. All-zero scores0. Source comment “sum to one” does not match formula: approximately **mean** one. | DQN Conv/Dense outputs **after actual ReLU**, identity capture sites; final output excluded. Residual network records explicit activation sites, not every affine. Default replay statistics batch256 after preprocessing; per-channel reduction includes spatial axes. | DQN agent measures online network after update. Exact paper match unverified; no claim all ReDo network variants share identical locations. |
| ReDo TF-Agents SAC variant | Same mean-absolute/relative activity formula and unit-weighted percentage | Default threshold0, epsilon1e-9; first min(log_batch_size,N) examples, default256. Activation-gradient/weight alternatives exist and are not adopted. | Keras Dense.call returns its configured activation: hidden Dense layers have ReLU; **the scalar linear value head is also visited** by the recursive Dense collector and included by its aggregate. | Activations cached during critic-loss forward before critic optimizer application; later logging does not make them post-update activations. Paper agreement unverified. |
| EchoCritic/scaling `srank` | Smallest k with cumulative `sum(σ[:k])>=.99*sum(σ)`; count | Uncentered, no squaring/no width division. Zero matrix returns1 by loop convention. | Final encoder output after LayerNorm for residual model; batch256→ceiling256 at campaign widths512/1024/1536. | Not conventional stable rank. Shared formula verified; EchoCritic places SVD on CPU. Paper equation correspondence unverified. |
| LoP entropy effective rank | `exp(-sum(p*log(p)))`, `p=σ/sum(abs(σ))`, 0log0 terms skipped | Uncentered helper; source float32 entropy accumulation. All-zero normalized spectrum has NaNs, skipped by comparison, and returns1; v1 explicitly retains return1 without dividing zero. | MNIST uses first2,000 permuted/shuffled examples, post-hidden-activation matrices, each hidden layer; not an RL replay critic batch. | Code definition/call verified; not independently verified against inaccessible paper. |
| LoP approximate rank | Minimal k with cumulative `σ²/sum(σ²)>=.99`; count | Squared-energy threshold, distinct from singular-sum count; zero helper returns1 after NaN comparison. | Same MNIST matrices/helper. | Code verified; no silent substitution into EchoCritic's `srank`. |
| LoP absolute approximate rank | Minimal k with cumulative `σ/sum(σ)>=.99`; count | Unsquared singular-value threshold; zero return1. | Also fourth helper return used by RL caller on last policy hidden features, 1,000-slot short-term buffer. RL variable named `stable_rank` holds this **count**, not conventional stable rank. | Code call verified; do not attribute conventional stable-rank formula to this variable. |
| EchoCritic/scaling critic gradient | `sqrt(sum_over_all_leaves(sum(g²)))`; TD-loss derivative units | Dense campaign: all parameter leaves, including head/normalization. Sparse branch masks gradients for reported norm; not relevant to original dense grid. | SAC TD-loss minibatches256; window mean of **per-update norms**. Not norm of mean gradient, probe gradient, or optimizer displacement. | Computed before AdamW; no decay term in raw gradient. Scaling correspondence verified. LoP PPO code instead includes combined policy/value gradient clipping before optimizer; do not equate those quantities. Paper match unverified. |
| New conventional stable rank | `sum(σ²)/max(σ)²`; dimensionless | Explicit zero convention0; uncentered default; no width normalization | Same final encoder output used for added rank family | User-requested conventional definition, not a verified LoP/scaling paper attribution. |

The selected new hidden-activation population follows ReDo's **DQN-style**
post-activation/excluded-output convention mapped explicitly to our residual
critic. It does not claim to reproduce every ReDo variant: the TF-Agents SAC
collector includes the linear head and has different cached-forward timing.
These are verified reference implementation differences, not grounds to declare
one variant wrong. Per-layer values and separately retained affine measurements
make the choice inspectable.

LoP's MNIST plotting code sums entropy ranks across three hidden layers and
divides by`3*2000/100`, a percentage of the configured hidden-unit total. The
raw helpers do not do that normalization. Our metrics retain raw values; applying
that width divisor to a256-example/1536-feature batch would impose an artificial
maximum of256/1536 per layer. No such normalization is introduced.

The LoP RL short-term feature buffer is initialized with zeros and the step0
call can precede filling it. Its call site is not a template for changing our
sampling schedule. This implementation borrows verified formulas only; it does
not adopt recycling, clipping, PPO updates or the reference training schedules.

## Additive implementation and metric names

Implementation: `scale_rl/diagnostics/critic_metrics.py`. Three observation-only
`sow` sites were added after actual ReLU operations in residual/MLP blocks,
under collection `critic_diagnostics_v1`. They add no modules, parameters,
initialization draws, BatchNorm state or optimizer state. Normal training calls
do not request that collection. Unit tests compare archived/current parameter
trees, predictions, historical metric values and synthetic gradients exactly.

Supported v1 model: plain **single `SACCritic`**, residual or MLP. Twin/injected
critics fail explicitly rather than folding critic/head axes into batch/unit axes.
Do not enable this observer on Exp2 injected/twin arms; they require separate
adapters and validation. No frozen historical source file or result was edited.

| Population / namespace | What is included | Meaning |
| --- | --- | --- |
| `critic_diag/v1/{stream}/post_relu/...` | Actual block ReLU outputs; residual expansion ReLU, both MLP ReLUs | ReDo **formula** applied to explicitly mapped activation neurons. This is a new measurement in EchoCritic, not retroactive ReDo values. |
| `.../{stream}/dense_affine/...` | Encoder projection and block Dense outputs, predictor excluded | Separate observation of historical affine population; float32 CPU reductions can differ at rounding boundaries from JAX reductions. Historical calculations stay untouched. |
| `.../{stream}/.../{layer}/relative_tau_0.1/percent` (also0,.2) | Per-unit relative masks, inclusive threshold, no global cross-layer normalization | Percentage, not fraction. Default thresholds explicitly recorded in each receipt. |
| `.../{stream}/.../{layer}/exact_zero/percent` | Units for which **every** supplied activation is exactly zero | No `isclose`; distinct from relative threshold and underflowed mean scores. |
| `.../{stream}/{population}/aggregate/.../unit_weighted_percent` | All selected unit masks concatenated within that population | ReDo/scaling-style neuron-weighted aggregation on the named population. |
| `.../aggregate/.../layer_mean_percent` | Equal mean of per-layer percentages | **New aggregate**; does not replace neuron weighting or merge affine/post-ReLU populations. |
| `.../{stream}/encoder_output_uncentered/analysis64/stable_rank` | Final encoder, no dataset centering | Conventional stable rank, zero0. |
| `.../analysis64/entropy_effective_rank_lop` | Same X; exponential entropy of normalized singular values | LoP code formula, explicit zero1 fallback retained. |
| `.../analysis64/squared_energy99_count_lop` | Same X; squared singular-value threshold | Distinct count, zero1 fallback retained. |
| `.../analysis64/singular_sum99_count_analysis64` | Same X; unsquared threshold count | Additional CPU-float64 computation, zero1. It can differ at numerical boundaries from historical SVD. |
| `.../encoder_output_uncentered/singular_sum99_count_historical` | Direct alias of already-emitted `train/critic_srank` for current replay | Exact historical numeric value, no recomputation. Existing `train/critic_srank` remains unchanged. |
| `.../current_replay/td_gradient_pre_adamw_global_l2_window_mean_historical` | Direct alias of existing logging-window `train/critic_gnorm` | No new differentiation, no update/probe-gradient confusion. |
| `.../{layer}/{mask}/transition_v1_new/...` | Consecutive saved masks in one session/stream | New persistence/overlap/reactivation diagnostics; not a paper-defined plasticity estimator. |

Residual mapping is deliberately restricted: the initial projection, contraction
Dense and residual sum have **no following ReLU at that location** in this
architecture. We do not label them ReLU neurons. LayerNorm is not counted as
another activation population; final LayerNorm output is the representation
matrix for rank. Predictor is excluded from dormancy/rank, but remains included
in the original training gradient. Thus no aggregate quietly mixes these sites.

### Precision, degenerate cases and gradients

Forward passes use the model's existing parameters/dtype/matmul setting; no global
precision config is changed. Activity reductions are CPU float32; epsilon1e-9
and thresholds cast tofloat32 are explicit. Nonfinite inputs/reductions fail
rather than yielding fabricated healthy zero metrics.

Added rank SVD is CPU NumPy float64, stably rescaled first (rank ratios/counts
unchanged algebraically). No network-width division. `examples`, `features`,
`batch_ceiling=min(N,D)` and `zero_spectrum` are emitted. `rank_metrics(center=True)`
optionally column-centers examples, with ceiling `min(N-1,D)`; the runner uses
uncentered features only. Centered calculations must be named separately by
callers. Zero-matrix1 fallbacks are code conventions, not a claim mathematical
rank is1, and can exceed a degenerate centered ceiling0. Historical zero1 stays
unchanged; stable-rank zero0 is an explicitly declared convention for undefined0/0.

Per-layer raw gradients cannot be reconstructed from logged global norms.
`available_gradient_arrays(tree, kind=...)` therefore accepts **already available**
arrays only and returns CPU-float64 per-layer/global L2. Mandatory kind is one
of `raw_td_gradient_pre_adamw`, `optimizer_update_array`, `probe_gradient`; keys
stay distinct. This new adapter never differentiates or updates a model. It is
not wired into SAC's compiled update payload: doing so would add transfers and
need a separate implementation/qualification. The sidecar records only the
historically available global window norm. No per-layer/probe gradients are
claimed measured in the original campaign or in the new training integration.

### Inputs, persistence and output

The Exp12 trainer samples its historical diagnostic batch **once**, passes it to
its existing `agent.get_metrics` (which preprocesses it in place), then supplies
those same preprocessed observations/actions to the optional observer. No extra
buffer sample, training key, NumPy/Python random draw or normalizer update occurs.
Historical logging reset, training schedule, evaluation, fitting and checkpoints
retain their calculations/order. Sidecars do not add W&B log calls or change
historical metric keys/configuration fingerprints.

`current_replay` reflects current visitation and normalization. Optional
`frozen_preprocessed` copies fixed already-normalized observations/actions and
never renormalizes them using later statistics. This fixes **both inputs and
preprocessing**, enabling future parameter-change comparisons. It does not fix
the changing input distribution of current-replay diagnostics. Inputs are stored
with immutable byte-backed arrays, saved as `frozen_inputs.npz`, and hashed in
every receipt. Raw replay/normalizer snapshots are not reconstructed from them.

Each observation writes exclusive `interaction_*.json` and `.npz` files outside
source/training checkpoints. JSON records counters, architecture, precision,
input hashes, mask-key/layer mapping, scalar names and mask-file SHA256. NPZ
contains boolean per-unit masks for each stream/population/threshold and exact
zeros, with `allow_pickle=False` reading supported. Unit index means that layer's
fixed parameter coordinate, not a neuron identity invariant under permutations
or architectural surgery.

For previous mask A and current B: persistent=`|A∩B|`, reactivated=`|A\B|`,
newly dormant=`|B\A|`; conditional persistence/reactivation divide by`|A|`, Jaccard
by`|A∪B|`. Empty denominators are null, not invented0. ReDo has a related
conditional-intersection logger but uses0 when previous dormant count is0.
Our complete `transition_v1_new` bundle, its null policy and extra statistics
are explicitly new diagnostics. Current-input transitions remain confounded by
input changes and do not establish irreversible/chronic neuron death.

A new process/observer session starts a new persistence series. Outputs are
partitioned by run-path hash and OS-generated UUID, not training RNG. Saved
masks allow explicit later comparisons across sessions; no continuity across
resume is silently inferred. Changed layer populations/shapes are rejected.

## Opt-in use and cost

Default root unset: **no added observer forward/SVD/sampling/files**. Existing
historical diagnostics keep their original cost. These settings are instructions
for an authorized future run, not authorization to start one now:

```bash
export ECHOCRITIC_CRITIC_DIAGNOSTICS_ROOT=/absolute/new/diagnostic-sidecars
export ECHOCRITIC_CRITIC_DIAGNOSTICS_RANK=1        # default0: no additional SVD
export ECHOCRITIC_CRITIC_DIAGNOSTICS_EVERY=5       # default1: every logging event
export ECHOCRITIC_CRITIC_DIAGNOSTICS_FREEZE_FIRST=1
# Alternatively, omit FREEZE_FIRST and supply an existing preprocessed NPZ:
# export ECHOCRITIC_CRITIC_DIAGNOSTICS_REFERENCE=/absolute/frozen_inputs.npz
```

Do not set both frozen options. Flags accept0/1. Output must be outside the
repository/checkpoint tree. `EVERY` counts existing diagnostic logging events;
it changes only optional observation cadence, not training/logging cadence.
Each measured stream adds one forward and CPU transfer of selected Dense/ReLU
activations; frozen mode adds a second. Captured memory scales with batch×sum
layer widths. Each enabled rank SVD costs roughly`O(min(N,D)^2*max(N,D))`, per
stream; masks scale with selected units×thresholds, not batch×units. For wide
critics these costs are material and can affect wall-clock throughput. No
representative GPU or full-model runtime benchmark is claimed.

Tiny CPU example (no training or fit):

```bash
JAX_PLATFORMS=cpu python -m scripts.critic_diagnostics_cpu_example \
  --out /absolute/fresh/tiny-example
```

Its two observations use the same untrained width4 residual critic and8-example
fixture; counters are synthetic labels, optimizer updates0. Actual output:
post-ReLU exact-zero25%, relativeτ.1 dormancy31.25%; affine exact-zero0% and
relativeτ.1 neuron-weighted4.16667%. Stable rank1.0631756583, entropy rank1.8538457566,
singular-sum99 count3, squared-energy99 count2. These are **synthetic example
values**, not evidence about any campaign phenotype. Full example output,
mask NPZ and logs remain in the task artifacts.

## Verification and historical consequences

`tests/test_reference_critic_diagnostics.py` covers zero/constant/rank-one/
orthogonal matrices, centering/ceilings/scaling, exact versus relative zeros,
inclusive thresholds/all-zero layers, unit versus layer weighting, actual
post-ReLU capture for residual/MLP, pinned formula equivalence, zero fallback,
known mask transitions, frozen-input immutability, no SVD when disabled, real
logging-block batch reuse, archived/current outputs/params/legacy metrics,
unchanged synthetic gradients and unchanged optimizer/normalizer/training RNG.
**22 CPU tests passed** (final run7.76s). The existing no-training methodology
reference also passed:195 generated commands,39 configurations,130 eligibility
candidates; those commands were not executed as training.
Only tensors/tiny networks and isolated logging blocks run: **no RL training,
new scientific probe fits or GPUs**. CPU results do not establish GPU bitwise
qualification. See [the validation receipt](reference_critic_diagnostics_validation.json) for
exact results/source hashes. Full mutation/restoration and broad training/fit
suites were not rerun under this request's no-training/no-probe-fit constraint.

Confirmed measurement bugs explaining the campaign disagreement: **none established**.
Historical stale W&B fresh summary and late metric effective-n labels remain
separate already documented reporting defects; this feature does not silently
apply their candidate patches. Definition inheritance, naming ambiguity,
alternative metrics and missing measurements are separate from defects.

The recorded DMC score recovery, fresh drift, large-humanoid rank-count decrease,
all48 Myo rank-count decreases, and pen architecture ordering remain valid
**under their original definitions**. Their values are not rewritten. Neither
“all post-ReLU neurons stayed active” nor “conventional stable rank collapsed”
follows from the historical keys. Persistence/reactivation, conventional stable
rank, entropy rank and fixed-input parameter effects require additional saved
activations or parameters/inputs and cannot be asserted retroactively.

Retrospective needs are precise: layer post-activation arrays (or original
parameter checkpoint plus matching replay observations/actions and normalization
statistics) for dormancy; final encoder feature matrix or full singular spectrum
for alternative ranks; per-unit masks with layer/coordinate identities and
counter/input provenance for persistence; original per-layer gradients/updates
for their decomposition. Existing scalar ratios/counts/global norms and one
Myo probe-loss NPZ do not contain that information. Original measured masks,
full activations and matching production states are absent in supplied ZIPs.
No new metric has been measured in the original campaign.
