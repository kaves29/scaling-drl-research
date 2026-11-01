# m-control CPU validation receipt

Base: `38e5abb7325a949e23440d6197b5d7d9ecd57c43` (PR #21).
Implementation checkpoint: `3b37d7c93eada9a692fa94267f70527810f8da66`.
Later receipt edits are documentation only; resolve final head from the PR.

From `main/`, environment:

```bash
export JAX_PLATFORMS=cpu EXP12_JAX_CACHE_DIR=off WANDB_MODE=disabled
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/workspace/venvs/scaling-drl/bin/python -m unittest \
 tests.test_m_control_pilot tests.test_exp12_positive_control \
 tests.test_exp12_injection tests.test_exp12_method_c \
 tests.test_exp2_method_c_integration tests.test_exp2_method_c_certifier -v
/workspace/venvs/scaling-drl-hb/bin/python -u -m tests.exp12_break_checks
/workspace/venvs/scaling-drl-hb/bin/python scripts/check_exp12_cpu_environments.py
```

Results:

- Final focused suite: **44 passed**, 74.910s, no skips, exit0, clean implementation
  checkpoint above. Includes a real tiny-network ten-target/2,000-step end-to-end
  calibration, all five curves,40 partial-round files and sensitivity arrays.
  Forced synthetic evidence returns STOP and never qualifies m. Saved checkpoint
  files remain byte-for-byte unchanged. A separate equality check verifies that
  enabling round retention leaves probe results unchanged.
- Mutation rerun: **74/74 passed**, exit0, same final executable/configuration
  bytes. Diagnostic harness preserves ordinary checks and additionally emits
  error tracebacks. It began before the implementation commit and ended after
  it; do not describe its entire process as starting at the committed SHA.
  Initial evolving-worktree attempt had72 successful checks and two restored
  test failures (twin metadata and control-restore check); both pass on the
  complete rerun. Initial hidden tracebacks were unavailable, so the exact
  initiating cause of those transient failures is unestablished. No test,
  tolerance, mutation or production behavior was weakened to obtain the rerun.
- Simulator restoration: **65/65 cells**, actual CPU wrappers including
  HumanoidBench, exit0. This is the existing13-task/5-seed simulator check,
  not a new scientific campaign or proof of seed9 performance.
- New files pass Black; changed Python parses; `git diff --check` passes.
- Warnings: existing Orbax aggregate/sharding notices, Gymnasium float32 bounds,
  matplotlib/Pyparsing deprecations and unwritable font cache. The isolated
  no-renderer pilot unit invocation also warned about missing X11. The actual
  simulator checks used working EGL software rendering.

Raw logs, diagnostic harness and checksum inventory are outside Git at
`/workspace/artifacts/m-control-implementation-20261101/`. Copy them separately
if needed. Final source SHA and log hashes are recorded in that directory's
receipt. Tests do not qualify CUDA, full-size VRAM/storage, runtime or rescue.
No GPU model execution, training campaign, Slurm submission, merge or automatic
m freeze was performed. See the pilot handoff for the remaining Delta gates.


## Additional pre-queue CPU checks

Owner requested a final CPU readiness check. On clean, fixed commit
`6c3e33cfaa5b34e28a3cd64e9f711e885049df7d`, the focused command above passed **49/49 tests**
in75.685s, and the unchanged full mutation command passed **74/74**, exit0.
The source and working tree remained fixed throughout both completed runs.
This closes the earlier mutation-run commit-boundary limitation for this source.

Five additional tests use real harmless CPU subprocesses to verify successful
completion, nonzero exit, retained logs/reservations, retry refusal, timeout
process-group cleanup, parent-SIGTERM cleanup, and rejection of wrong/dirty
source or absent/multiple GPU bindings before any child or reservation. Git
identity and CUDA binding are fixtures; no test claims real CUDA qualification.
No production code or scientific configuration changed in this follow-up.
The previous65 simulator cells apply to identical executable/configuration bytes.

Evidence: `/workspace/artifacts/m-control-prequeue-20261101/`, including exact
source SHA, completed logs and checksums. The earlier uncommitted-source test
attempt was deliberately stopped and is not counted. No additional CPU check is
currently recommended before Delta preflight. Delta must still verify exact
source/environment, allocation/binding, CUDA execution, storage and budget before
queueing; full-size runtime/memory and natural rescue remain unmeasured.
