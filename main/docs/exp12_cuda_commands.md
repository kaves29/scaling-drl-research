# Exp 1/2: commands that NEED CUDA VERIFICATION

Everything here has passed on CPU (Linux, jaxlib 0.4.34). These commands check
the same properties on the real Linux + CUDA stack. Run them from `main/` with
the project env activated, on one GPU, and send me the output.

## 0. Environment sanity and HumanoidBench install
```bash
python -c "import jax; print(jax.devices()); import jaxlib; print(jaxlib.__version__)"
pip install -r requirements.txt              # adds rliable and its pinned deps; no existing pin changes
bash scripts/install_humanoid_bench.sh /abs/path/to/humanoid-bench   # pinned commit, --no-deps, editable
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   # needed for HumanoidBench (offscreen renderer at construction)
```

## 1. Phase 1–2 invariants on GPU (bit-exact resume, probes do not change training)
Run twice: once with default XLA flags and once with deterministic GPU ops. If
the default run fails but the deterministic run passes, GPU reductions are
non-deterministic run to run. That decides whether the identity fork can be
bit-exact on CUDA, and needs your decision on a tolerance (R-CHECKS); I will
not loosen anything silently.
```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python -m unittest tests.test_exp12_foundations tests.test_exp12_probe tests.test_exp12_phase3 -v 2>&1 | tee cuda_p123_default.log
XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python -m unittest tests.test_exp12_foundations tests.test_exp12_probe tests.test_exp12_phase3 -v 2>&1 | tee cuda_p123_deterministic.log
python tests/exp12_break_checks.py 2>&1 | tee cuda_break_checks.log
```

## 2. Probe overhead, training throughput and peak memory per critic size
dog-run (1M env steps; also times angle_1 vs exp1 with probes off):
```bash
python scripts/profile_exp12.py --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --train_steps 3000 --probe_repeats 2 \
  --angle1 --compare_steps 8000 --out profile_dog_run.json
```
hopper-hop (500k env steps; the worst case for probe overhead as a share of the run):
```bash
python scripts/profile_exp12.py --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --train_steps 3000 --probe_repeats 2 --out profile_hopper_hop.json
```
Send me both JSON files. The key fields are `train_it_per_s_probes_off`,
`probe_check_s`, `probe_overhead_pct_of_wallclock`, `peak_device_bytes` and
`ratio_exp1_over_angle1`.
Notes:
- Run with nothing else on the GPU, so the throughput numbers are solo
  numbers.
- `peak_device_bytes` is per process. The recommended concurrent jobs per GPU
  come from it.

## 3. Probe dynamic range on fresh critics (decision 2026-10-04; fresh critics only)
The buffer is filled to 5,000 transitions by the random warm-up and the critic
is never trained. Each pool size gives the fresh critic's per-round scores, IQM,
std and range across the 5 rounds. The 25,600 row is the run's actual check-0
probe.
```bash
python scripts/probe_fresh_checks.py --mode range --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out fresh_range_dog_run.json
python scripts/probe_fresh_checks.py --mode range --env hopper-hop --env_group dmc_medium \
  --archs D2W512 D4W1024 D6W1536 --pools 1600 6400 25600 --out fresh_range_hopper_hop.json
```

## 4. False-trigger rate of the Exp 1 trigger under a null (B4; fresh critics only)
Each pair is two independently initialised fresh critics compared exactly like
current vs fresh. Neither has lost plasticity, so every trigger is a false
trigger.
```bash
python scripts/probe_fresh_checks.py --mode null --env dog-run --env_group dmc_hard \
  --archs D2W512 D4W1024 D6W1536 --null_pairs 20 --out fresh_null_dog_run.json
```
