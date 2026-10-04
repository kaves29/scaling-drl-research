# Exp 1/2: commands that NEED CUDA VERIFICATION (run in this order)

Everything here has passed on CPU (Linux, jaxlib 0.4.34). These commands check
the same properties on the real Linux + CUDA stack. Run them from `main/` with
the project env activated, on one otherwise-idle GPU, and send me the logs and
JSON files.

Each step's pass/fail rule was pre-specified in `docs/exp12_decisions.md` on
2026-10-04, before any CUDA result existed. None of these commands launches or
schedules a grid run.

## 0. Setup
```bash
python -c "import jax; print(jax.devices()); import jaxlib; print(jaxlib.__version__)"
pip install -r requirements.txt                                     # adds rliable + pinned deps; no existing pin changes
bash scripts/install_humanoid_bench.sh /abs/path/to/humanoid-bench  # pinned commit, --no-deps, editable
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl                          # HumanoidBench builds an offscreen renderer
OUT=/abs/path/to/exp12_cuda_checks && mkdir -p $OUT
```

## 1. Tests and break checks
Run the tests twice: once with default XLA flags and once with deterministic GPU
ops. If only the deterministic run passes, GPU reductions are non-deterministic
from run to run. That decides whether the identity fork can be bit-exact on
CUDA, and needs your decision on a tolerance; nothing is loosened silently.
```bash
python -m unittest discover -s tests -p "test_exp12_*.py" -t . -v 2>&1 | tee $OUT/1_tests_default.log
XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python -m unittest discover -s tests -p "test_exp12_*.py" -t . -v 2>&1 | tee $OUT/1_tests_deterministic.log
python tests/exp12_break_checks.py 2>&1 | tee $OUT/1_break_checks.log
```

## 2. Fresh-critic dynamic range (fresh critics only)
Pre-specified rule: PASS if, at all three sizes, 0.1·b ≤ IQM(P) ≤ 0.9·b at the
configured pool (25,600). The round-to-round spread is always reported. The
verdict is written to `range_verdict.json`. If it fails, the fallback ladder is
a smaller pool first, then more probe steps; each step is proposed to you, not
applied.
```bash
python scripts/probe_fresh_checks.py --mode range --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out_dir $OUT/2_range_dog_run
python scripts/probe_fresh_checks.py --mode range --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out_dir $OUT/2_range_hopper_hop
```

## 3. Fresh-pair null (at least 100 pairs per size)
Every pair's per-round P, b and L, its IQM and its bootstrap interval are
appended to `null_pairs_<arch>.jsonl` as each pair finishes. Re-running the
same command resumes. The summary reports the per-check fire rate and the 95th
percentile of L. Pre-specified: if the fire rate exceeds 5% at any size, you
decide whether to adopt a null-calibrated threshold; nothing is adopted
automatically.
```bash
python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --null_pairs 100 --out_dir $OUT/3_null_dog_run
```

## 4. Probe overhead, training throughput and peak memory per critic size
dog-run (1M env steps; also times angle_1 against exp1 with probes off), and
hopper-hop (500k env steps, the worst case for probe overhead as a share of the
run):
```bash
python scripts/profile_exp12.py --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --train_steps 3000 --probe_repeats 2 \
  --angle1 --compare_steps 8000 --out $OUT/4_profile_dog_run.json
python scripts/profile_exp12.py --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --train_steps 3000 --probe_repeats 2 --out $OUT/4_profile_hopper_hop.json
```
The key fields are `train_it_per_s_probes_off`, `probe_check_s`,
`probe_overhead_pct_of_wallclock`, `peak_device_bytes` (per process; the
recommended concurrent jobs per GPU come from it) and `ratio_exp1_over_angle1`.
