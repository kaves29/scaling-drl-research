# Experiment 2 Method C source integration candidate — 28 October 2026

Review candidate only. Production, GPU jobs and scientific qualification remain
NO-GO. PR #19 and approved integration are not modified. This report will be
completed with exact-source CPU receipts before delivery.

## Verified authority and selection

Remote refs verified before editing: integration/exp12
`844e3c0cc24998b25c3d1e667bcbfe056b1b336c`, PR #19 head
`5482b2a516de5d51dab2c1bd7a4e86a49832b768`, published main
`087feab341b9be1f377697668f2d1c750ec0467f`. Integration844 is an ancestor
of main087. A branch name alone is not approval: AGENTS.md gives chronological
`main/.claude/methodology-exp1-exp2.md` amendments and `exp12_decisions.md`
authority, and amendments (aa)/(ab) explicitly record the owner's decisions.
The older decisions text's “we use 1000” is historical and superseded by (aa).
The decisions file itself is identical between844 and087 and is not rewritten.

| Source (full SHA) | Imported paths / purpose |
|---|---|
| `c7fc4ea37a1b8ad5427f79bbc804f3a1238b006b` | Approved Method C producer `experiments/exp12/probe.py`, sole config change `probe.steps=2000`, amendment (aa), producer tests and CPU methodology census. These exact producer/config blobs are unchanged through087. |
| `d65f9bb798f5c5c74b1857a6329669e068ddf517` | Reviewed `exp1.py` late-control schedule restore and preserved aborted fresh state; `exp12/state.py` checkpoint durability; `exp12/fork.py` atomic panel archive; fork/resume tests. Required to preserve the reviewed deferred-arm lifecycle, not a new training intervention. |
| `b225577c2ba5a18b2ad6d14c47b34a00b5443e05` | Historical range-capture tests explicitly retain their1,000-step protocol and reject relabeling Method C as historical evidence. |
| `957be5c63a48b0b90dd28ee4b361e3c6d373ab05` | Independent consolidated review documentation, retained as a dated historical source. |
| `6eb397d0c485a5db202806df8a0308e48e46b05d` | Exact amendment (ab), deferred-arm lifecycle regression and compatibility report. Complete methodology file imported byte-for-byte from087 includes (aa)/(ab). |
| PR19 `5482b2a516de5d51dab2c1bd7a4e86a49832b768` | Exact source-admission guard/test, frozen-config curve certifier/test, accurate fixture and dated reports. |
| This candidate | New producer→runner NPZ→ledger→curve certification CPU test; exact durability test class extracted without unrelated scheduler dependencies; integration map and current report/handoff. |

Machine-readable per-file source bindings and SHA256s are in
`exp2_method_c_integration_sources.json`. Runtime producer/config/restore files
are exact087 blobs; admission/certifier files are exactPR19 blobs. Imported
historical reports retain their original dates/status and are not current
qualification claims. Current status is this report.

No main087 merge is performed. Unrelated dynamic scheduler, allocation, approval
templates, launch tooling, numeric-bound tooling, pilots and newer alternative
probes are excluded. In particular, this is not promotion of main087's launch
system or its production approvals. Amendment (ab)'s permission to qualify Exp1
separately is retained; this candidate supplies and tests the deferred-arm
runtime, not a new launch system. The legacy integration launcher may still
impose stricter launch gates. Production launch-tool integration/review remains
separate; no candidate command bypasses those gates.

## Preserved semantics and planned CPU evidence

Five targets,2,000 continuous AdamW updates, first1,000 original draws and next
1,000 from `fold_in(idx_key,1)`; own-mean offsets, scores, bootstrap/trigger,
eligibility, injection, precision, evaluation and budgets are not retuned.
`injection.m` and force-trigger hooks remain null in canonical configuration.
Existing tiny lifecycle tests use pre-existing dev-only hooks as test fixtures;
they are not evidence of natural eligibility or positive-control sensitivity.

The added array-IO integration test uses the full canonical probe configuration
(including25,600 sample pool,256 minibatch,2,560 eval chunk,5 rounds and2,000
updates), production optimizer construction, runner saving, ledger copying and
actual arm Check2 generation. A tiny linear network and synthetic replay replace
full-width architecture/environment execution. Its unchanged loss arrays must
pass the shared curve certifier. Full-population/source certification is tested
separately with explicitly synthetic complete-study fixtures. Neither is a CUDA
qualification or a natural full-scale scientific positive control.
