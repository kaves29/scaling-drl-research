"""Exercise one real replay checkpoint, including a branch interruption and restart."""
import argparse
import hashlib
import json
import pickle
import subprocess
from pathlib import Path


QUALIFICATION_UPDATES = 5000


def configure_qualification(runner):
    """Bind a shortened engineering schedule without changing the pilot schedule."""
    if runner.update_budget < QUALIFICATION_UPDATES:
        raise ValueError("parent budget is below the qualification update count")
    runner.parent_update_budget = runner.update_budget
    runner.update_budget = QUALIFICATION_UPDATES
    runner.checks = runner.probes = [QUALIFICATION_UPDATES]
    runner.config_hash = hashlib.sha256(
        (runner.config_hash + ":qualification-5000-v1").encode()
    ).hexdigest()


def configure_a100_smoke(runner):
    """Use a non-scientific, source-bound A100 engineering smoke schedule."""
    if runner.update_budget < 100:
        raise ValueError("parent budget is below the smoke update count")
    runner.parent_update_budget = runner.update_budget
    runner.update_budget = 100
    runner.checks = runner.probes = [100]
    runner.cfg.online_exp2.branch_raw_steps = 20
    runner.cfg.online_exp2.measurement_raw_steps = [0, 10, 20]
    runner.cfg.online_exp2.diagnostic_save_every = 5
    runner.cfg.probe.rounds = 1
    runner.cfg.probe.steps = 5
    runner.cfg.probe.pool_size = 256
    runner.cfg.probe.batch_size = 32
    runner.cfg.probe.eval_chunk = 64
    runner.config_hash = hashlib.sha256(
        (runner.config_hash + ":a100-smoke-100-20-v1").encode()
    ).hexdigest()


def validate_evidence(rows, checkpoint, horizon):
    diagnostics = [row for row in rows if row.get("type") == "pilot_diagnostic"
                   and row.get("checkpoint") == checkpoint]
    conditions = {f"{arch}_{mode}" for arch in ("small", "large")
                  for mode in ("frozen", "learning")}
    if (len(diagnostics) != 1 or set(diagnostics[0]["branch_counters"]) != conditions
            or any(c["raw_env_steps"] != horizon
                   for c in diagnostics[0]["branch_counters"].values())):
        raise ValueError("incomplete branch evidence")
    probes = {row["architecture"] for row in rows if row.get("type") == "probe"
              and row.get("check") == checkpoint}
    if probes != {"small", "large"}:
        raise ValueError("incomplete plasticity evidence")


def main():
    import jax
    from experiments.exp1 import compose_config
    from experiments.exp12.precision import set_matmul_precision
    from experiments.exp12.replay_exp2_pilot import ReplayExp2Pilot

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", type=json.loads, required=True)
    parser.add_argument("--phase", choices=("interrupt", "resume"), required=True)
    parser.add_argument("--a100-smoke", action="store_true")
    args = parser.parse_args()
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines()
    if len(devices) != 1:
        raise RuntimeError("qualification requires exactly one visible GPU")
    model, memory = (part.strip() for part in devices[0].rsplit(",", 1))
    if (jax.default_backend() != "gpu" or
            (args.a100_smoke and ("A100" not in model or int(memory) < 75000)) or
            (not args.a100_smoke and model != "NVIDIA L4")):
        raise RuntimeError("qualification requires the requested GPU and JAX GPU backend")
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "qualification-interruption.json"
    if (args.phase == "interrupt") == marker.exists():
        raise ValueError("qualification phase does not match interruption state")
    identity = args.identity
    set_matmul_precision()
    cfg = compose_config(str(Path(__file__).parents[2] / "configs"),
                         "online_exp2_pilot_" + identity["reference_regime"],
                         ["env=" + ("myosuite_hard" if identity["environment"].startswith("myo-") else "dmc_hard"),
                          "env_name=" + identity["environment"], "seed=" + str(identity["seed"])],
                         allow_pilot=True)

    class StopQualification(Exception):
        pass

    class Qualifier(ReplayExp2Pilot):
        def __init__(self, *runner_args):
            super().__init__(*runner_args)
            (configure_a100_smoke if args.a100_smoke else configure_qualification)(self)

        def _check_resource_cap(self):
            super()._check_resource_cap()
            if args.phase == "interrupt":
                for path in self.root.glob("diagnostics/update_*/small_learning.pkl"):
                    stamp = (path, path.stat().st_mtime_ns)
                    if getattr(self, "_qualification_stamp", None) == stamp:
                        continue
                    self._qualification_stamp = stamp
                    state = pickle.loads(path.read_bytes())
                    if not state["complete"] and state["raw_env_steps"] >= int(self.cfg.online_exp2.branch_raw_steps) // 2:
                        marker.write_text(json.dumps({"checkpoint": state["identity"]["checkpoint"],
                                                      "raw_env_steps": state["raw_env_steps"]}))
                        raise StopQualification()

        def _checkpoint(self, trainer):
            super()._checkpoint(trainer)
            if args.phase == "resume":
                interrupted = json.loads(marker.read_text())
                checkpoint = trainer.update_step
                expected_checkpoint = 100 if args.a100_smoke else QUALIFICATION_UPDATES
                if checkpoint != expected_checkpoint or checkpoint != interrupted["checkpoint"]:
                    raise ValueError("restart did not resume interrupted checkpoint")
                validate_evidence(self.rows, checkpoint, int(self.cfg.online_exp2.branch_raw_steps))
                diagnostic = next(row for row in self.rows if row.get("type") == "pilot_diagnostic"
                                  and row.get("checkpoint") == checkpoint)
                receipt = {"source_commit": self.source_commit,
                           "manifest_sha256": hashlib.sha256(args.parent_manifest.read_bytes()).hexdigest(),
                           "gpu_model": model, "parent_restore": True, "full_branch_horizon": True,
                           "plasticity_probe": True, "restart": True, "qualified_workers": 1,
                           "identity": identity, "checkpoint": checkpoint,
                           "qualification_updates": QUALIFICATION_UPDATES,
                           "parent_update_budget": self.parent_update_budget,
                           "qualification_protocol": "replay-qualification-5000-v1",
                           "branch_horizon": int(self.cfg.online_exp2.branch_raw_steps),
                           "branch_counters": diagnostic["branch_counters"],
                           "probe_architectures": ["small", "large"],
                           "interruption": interrupted}
                if args.a100_smoke:
                    receipt.update(smoke_only=True, qualified_workers=0, qualification_updates=100,
                                   qualification_protocol="a100-smoke-100-20-v1",
                                   gpu_memory_mib=int(memory),
                                   smoke_probe={"rounds": 1, "steps": 5, "pool_size": 256,
                                                "batch_size": 32, "eval_chunk": 64},
                                   full_branch_horizon=False, plasticity_probe=False)
                result_name = "a100-smoke-result.json" if args.a100_smoke else "qualification-result.json"
                (self.root / result_name).write_text(json.dumps(receipt, indent=2))
                raise StopQualification()

    runner = Qualifier(cfg, str(args.output.resolve()), args.parent_manifest, identity)
    try:
        runner.run()
    except StopQualification:
        pass
    else:
        raise RuntimeError("qualification did not reach expected stopping point")
    if args.phase == "interrupt" and not marker.exists():
        raise RuntimeError("qualification did not interrupt a saved branch")
    result_name = "a100-smoke-result.json" if args.a100_smoke else "qualification-result.json"
    if args.phase == "resume" and not (args.output / result_name).exists():
        raise RuntimeError("qualification did not produce evidence")


if __name__ == "__main__":
    main()
