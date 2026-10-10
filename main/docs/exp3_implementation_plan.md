# Exp3 development-pilot implementation contract

Base: `844e3c0cc24998b25c3d1e667bcbfe056b1b336c`. No Exp1/Exp2 entry point,
configuration, SAC update, optimizer, checkpoint format or scientific rule changes.

1. Strict trusted-local artifact adapter reuses create_agent, injected-network
   reconstruction, Orbax restoration and replay restoration. Reject missing state
   and inconsistent provenance. No optimizer or transition history reconstruction.
2. Ordered chunked streams record raw buffer arrivals and source normalization
   with explicit update counts. An opt-in trainer adapter records existing source
   runs without introducing action selection, RNG consumption or save calls.
3. Three isolated pilots reuse Trainer/AdamW, SAC distributions, clipped-double-Q
   semantics, ordinary SAC updates and evaluation. Direction/magnitude are explicit
   action-gradient diagnostic interventions, never ordinary training modifications.
4. CPU end-to-end and negative tests cover state, pairing, stream corruption,
   passivity, targets, optimizer matching, recording parity and restart fidelity.
5. Versioned config/manifest and CLI reject unresolved scientific settings. GPU
   validation and cost measurement are prepared only, not executed or submitted.

Exact seeds are 1–5 from generate_manifest.EXP12_SEEDS. DMC tasks dog-run and
humanoid-walk use dmc_hard. The repository registers FOUR MyoSuite Exp1/Exp2
tasks: myo-key-turn, myo-pen-twirl, myo-pose-hard, myo-reach. Selection of the
requested TWO is unresolved; materialization refuses to silently choose.

Historical artifact dependency matrix (each selected environment × seed):

| Data | Pilot1 | Pilot2 A/B | Pilot3 | Historical recovery |
|---|---|---|---|---|
| Immutable complete fork + run metadata | required | required | required for pairing | supported if retained |
| Matched U/I post-fork checkpoints | required | active references only | frozen actors/evaluators | only actual retained checkpoints |
| Initial replay + normalization + optimizers/RNG | required | required | required | complete state, not weights-only exports |
| Fixed fork panel | reusable optional panel | evaluation panel optional | reusable optional panel | panel.npz, with original provenance |
| Ordered raw U stream | optional source panel | required A/B | batches can come from replay | NOT reconstructible from replay snapshot |
| Ordered raw I stream | optional source panel | required B only | optional sensitivity input | NOT reconstructible from replay snapshot |
| Evaluation returns | existing reference outcomes | active references only | descriptive reference only | raw saved episode records |

No real per-environment/seed Exp2 inputs are present in this Cloud workspace.
Existing scratch runs are CPU engineering fixtures, not scientific pilot evidence.
The matrix is materialized by the manifest/audit command for every requested cell.
Missing streams require new, explicitly authorized source continuations from an
exact fork, not fabricated historical trajectories. Recording changes I/O cost;
its no-trajectory-change claim must pass exact CPU parity and later GPU checks.

Unset scientific choices: MyoSuite pair, architecture/checkpoint selection, panel
size/provenance, normalization/alpha matching rule, intervention space and zero
signal handling, source continuation horizon, passive update timing/budget,
evaluation/checkpoint cadence, target diagnostic budget/evaluator/alpha.
No run is authorized by this infrastructure or by tiny CPU fixture parameters.
