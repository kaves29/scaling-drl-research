# Exp 1/2: commands that NEED CUDA VERIFICATION

Everything here has passed on CPU (Linux, jaxlib 0.4.34). These commands check
the same properties on the real Linux + CUDA stack. Run them from `main/` with
the project env activated, on one GPU, and send me the output.

## 0. Environment sanity
```bash
python -c "import jax; print(jax.devices()); import jaxlib; print(jaxlib.__version__)"
```

## 1. Phase 1–2 invariants on GPU (bit-exact resume, probes do not change training)
Run twice: once with default XLA flags and once with deterministic GPU ops. If
the default run fails but the deterministic run passes, GPU reductions are
non-deterministic run to run. That decides whether the identity fork can be
bit-exact on CUDA, and needs your decision on a tolerance (R-CHECKS); I will
not loosen anything silently.
```bash
export MUJOCO_GL=egl
python -m unittest tests.test_exp12_foundations tests.test_exp12_probe -v 2>&1 | tee cuda_p12_default.log
XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  python -m unittest tests.test_exp12_foundations tests.test_exp12_probe -v 2>&1 | tee cuda_p12_deterministic.log
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
