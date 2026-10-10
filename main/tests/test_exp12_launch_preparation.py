"""Adversarial CPU checks for prepared gates; never run a GPU workload."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.exp12_fullwidth_gate import (
    cache_observer,
    classify,
    gpu_placement_ok,
    validate_spec,
)
from scripts.exp12_launch_plan import (
    GATES,
    allocation_matches,
    array_matches,
    approve,
    cells,
    postvalidate,
    sha,
    validate_replay_normalization,
)


class FullwidthContractTest(unittest.TestCase):
    def test_auxiliary_checkpoint_payloads_are_required_and_consistent(self):
        import pickle
        import numpy as np
        from experiments.exp12.state import BUFFER_ARRAYS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                validate_replay_normalization(root)
            (root / "obs_rms.pkl").write_bytes(
                pickle.dumps(dict(mean=np.zeros(2), var=np.ones(2), count=2))
            )
            (root / "buffer_meta.pkl").write_bytes(
                pickle.dumps(
                    dict(num_in_buffer=2, current_idx=2, n_step_transitions=[])
                )
            )
            arrays = {
                k.lstrip("_"): np.zeros((2, 1), dtype=np.float32) for k in BUFFER_ARRAYS
            }
            np.savez(root / "buffer.npz", **arrays)
            validate_replay_normalization(root)
            arrays["actions"] = np.zeros((1, 1), dtype=np.float32)
            np.savez(root / "buffer.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "replay payload"):
                validate_replay_normalization(root)

    def spec(self):
        return dict(
            expected_commit="a" * 40,
            environment="dog-run",
            seed=102,
            checkpoint_step=6000,
            crash_step=6001,
            end_step=6061,
        )

    def records(self):
        placement = {
            f"{name}.{field}": dict(
                floating_leaves=1, platforms=["gpu"], dtypes=["float32"]
            )
            for name in ("_actor", "_critic", "_target_critic", "_temperature")
            for field in ("params", "opt_state")
        }
        return {
            stage: dict(
                complete=True,
                backend="gpu",
                placements={"saved": copy.deepcopy(placement)},
                cache_hits=1 if stage == "reference_warm" else 0,
                cache_misses=1,
                gpu_cache_hits=1 if stage == "reference_warm" else 0,
                gpu_cache_misses=1,
                device_memory_stats=dict(peak_bytes_in_use=1),
                cache_initially_empty=stage == "reference_cold",
                scientific_invariant_failures=[],
            )
            for stage in ("reference_cold", "reference_warm", "crash", "resume")
        }

    def differences(self):
        return dict(references=[], resume_vs_cold=[], resume_vs_warm=[], roundtrip=[])

    def test_no_tiny_warmup_or_confirmatory_seed(self):
        validate_spec(self.spec())
        for key, value in (
            ("checkpoint_step", 95),
            ("seed", 1),
            ("environment", "made-up"),
            ("end_step", 6000),
        ):
            spec = self.spec()
            spec[key] = value
            with self.assertRaises(ValueError):
                validate_spec(spec)

    def test_gpu_optimizer_fallback_rejected(self):
        records = self.records()
        self.assertEqual(classify(records, self.differences()), "PASS")
        records["resume"]["placements"]["saved"]["_critic.opt_state"]["platforms"] = [
            "cpu"
        ]
        self.assertEqual(
            classify(records, self.differences()), "INFRASTRUCTURE_FAILURE"
        )

    def test_missing_cache_evidence_cannot_pass(self):
        records = self.records()
        records["reference_warm"]["gpu_cache_hits"] = 0
        self.assertEqual(classify(records, self.differences()), "INCOMPLETE")

    def test_cache_instrumentation_preserves_executable_and_tracks_backend(self):
        from types import SimpleNamespace

        counters, records = {}, []
        executable = object()

        def cached(backend, value, *, marker):
            self.assertEqual((value, marker), (3, 4))
            counters["/jax/compilation_cache/cache_hits"] = 1
            return executable

        result = cache_observer(cached, counters, records)(
            SimpleNamespace(platform="gpu"), 3, marker=4
        )
        self.assertIs(result, executable)
        self.assertEqual(records, [dict(platform="gpu", hits=1, misses=0)])

    def test_partial_placement_snapshot_cannot_pass(self):
        snapshot = self.records()["resume"]["placements"]["saved"]
        del snapshot["_temperature.params"]
        self.assertFalse(gpu_placement_ok(snapshot))

    def test_wrong_precision_or_missing_memory_cannot_pass(self):
        records = self.records()
        records["resume"]["placements"]["saved"]["_actor.params"]["dtypes"] = [
            "float64"
        ]
        self.assertEqual(
            classify(records, self.differences()), "INFRASTRUCTURE_FAILURE"
        )
        records = self.records()
        records["reference_warm"]["device_memory_stats"] = None
        self.assertEqual(classify(records, self.differences()), "INCOMPLETE")

    def test_missing_optimizer_moments_cannot_pass(self):
        snapshot = self.records()["resume"]["placements"]["saved"]
        snapshot["_actor.opt_state"] = dict(floating_leaves=0, platforms=[])
        self.assertFalse(gpu_placement_ok(snapshot))
        snapshot = self.records()["resume"]["placements"]["saved"]
        snapshot["_target_critic.opt_state"] = dict(floating_leaves=0, platforms=[])
        self.assertTrue(gpu_placement_ok(snapshot))

    def test_nondeterminism_is_distinct_from_resume_defect(self):
        diffs = self.differences()
        diffs["references"] = ["agent:rng"]
        self.assertEqual(classify(self.records(), diffs), "NONDETERMINISTIC_BACKEND")
        diffs["references"] = []
        diffs["resume_vs_warm"] = ["buffer:action"]
        self.assertEqual(classify(self.records(), diffs), "RESUME_DEFECT")

    def test_check1_failure_is_not_completion(self):
        records = self.records()
        records["reference_cold"]["scientific_invariant_failures"] = ["target Check1"]
        self.assertEqual(
            classify(records, self.differences()), "SCIENTIFIC_INVARIANT_FAILURE"
        )


class LaunchPreparationTest(unittest.TestCase):
    def test_parent_array_cannot_silently_omit_runs(self):
        environment = dict(
            SLURM_ARRAY_TASK_ID="0",
            SLURM_ARRAY_TASK_COUNT="195",
            SLURM_ARRAY_TASK_MIN="0",
            SLURM_ARRAY_TASK_MAX="194",
            SLURM_ARRAY_TASK_STEP="1",
        )
        array_matches("parents", 0, False, environment)
        with self.assertRaisesRegex(ValueError, "array index"):
            array_matches("parents", 0, False, {})
        environment["SLURM_ARRAY_TASK_COUNT"] = "194"
        with self.assertRaisesRegex(ValueError, "exactly"):
            array_matches("parents", 0, False, environment)
        array_matches("parents", 0, True, environment)

    def test_pilot_array_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must not"):
            array_matches("pilot", 0, False, {"SLURM_ARRAY_TASK_ID": "0"})

    def test_allocation_requires_actual_approved_limits(self):
        resources = dict(cpus=4, memory_gb=32, wall_minutes=7)
        environment = dict(SLURM_CPUS_PER_TASK="4", SLURM_MEM_PER_NODE="32768")
        allocation_matches(resources, environment, "JobId=1 TimeLimit=00:07:00")
        with self.assertRaisesRegex(ValueError, "wall time"):
            allocation_matches(resources, environment, "JobId=1 TimeLimit=00:08:00")
        with self.assertRaisesRegex(ValueError, "CPU/memory"):
            allocation_matches(resources, {}, "TimeLimit=00:07:00")

    def test_actual_source_preflight_failure_preserves_evidence(self):
        import os
        import subprocess
        from scripts.exp12_launch_plan import MAIN

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            spec.write_text("{}")
            env = dict(
                os.environ,
                EXPECTED_COMMIT="a" * 40,
                GATE_SPEC=str(spec),
                GATE_ROOT=str(root / "evidence"),
                GATE_STAGE="reference_cold",
                SLURM_JOB_ID="fixture",
            )
            result = subprocess.run(
                ["bash", str(MAIN / "scripts/sbatch_exp12_fullwidth_gate.sh")],
                cwd=MAIN,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("HEAD mismatch", result.stderr)
            attempt = root / "evidence/attempts/reference_cold_fixture"
            self.assertEqual(
                json.loads((attempt / "process.json").read_text())["exit"], 2
            )
            self.assertTrue((attempt / "SHA256SUMS").exists())

    def approval(self, stage):
        return dict(
            approved=True,
            stage=stage,
            expected_commit="a" * 40,
            gates={
                n: dict(status="APPROVED", decision_reference="owner-written approval")
                for n in GATES[stage]
            },
            resources=dict(gpus=1, cpus=4, memory_gb=32, wall_minutes=7, concurrency=1),
        )

    def test_scientific_gates_fail_closed(self):
        approval = self.approval("parents")
        with self.assertRaisesRegex(ValueError, "verified evidence"):
            approve(approval, "parents", "a" * 40)
        approval["approved"] = False
        with self.assertRaisesRegex(ValueError, "explicit"):
            approve(approval, "parents", "a" * 40)

    def test_stale_gpu_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(dict(verdict="PASS", expected_commit="b" * 40)))
            approval = self.approval("pilot")
            approval["gates"]["fullwidth_resume_identity"]["evidence"] = [
                dict(path=str(path), sha256=sha(path))
            ]
            with self.assertRaisesRegex(ValueError, "stale/failed"):
                approve(approval, "pilot", "a" * 40)
            path.write_text(
                json.dumps(
                    dict(
                        verdict="PASS",
                        expected_commit="a" * 40,
                        qualification="fullwidth_resume_identity",
                        coverage=[
                            dict(
                                architecture="D4W1536",
                                suite="dmc",
                                environment="dog-run",
                            )
                        ],
                    )
                )
            )
            approval["gates"]["fullwidth_resume_identity"]["evidence"][0]["sha256"] = (
                sha(path)
            )
            self.assertEqual(approve(approval, "pilot", "a" * 40)["gpus"], 1)

    def test_one_case_cannot_certify_all_scaled_suites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(
                json.dumps(
                    dict(
                        verdict="PASS",
                        expected_commit="a" * 40,
                        qualification="cuda_identity_fork",
                        coverage=[
                            dict(
                                architecture="D4W1536",
                                suite="dmc",
                                environment="dog-run",
                            )
                        ],
                    )
                )
            )
            approval = self.approval("parents")
            approval["gates"]["cuda_identity_scaled_suites"]["evidence"] = [
                dict(path=str(path), sha256=sha(path))
            ]
            with self.assertRaisesRegex(ValueError, "coverage"):
                approve(approval, "parents", "a" * 40)

    def test_cpu_identity_certificate_cannot_be_gpu_qualified(self):
        from scripts.exp12_launch_plan import certify_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("run", "arm"):
                p = root / name
                p.mkdir()
                (p / "run_metadata.json").write_text(
                    json.dumps(
                        dict(
                            code=dict(commit="a" * 40, dirty=False),
                            launches=[
                                dict(
                                    commit="a" * 40,
                                    dirty=False,
                                    device=dict(platform="cpu", device_kind="cpu"),
                                )
                            ],
                        )
                    )
                )
            with self.assertRaisesRegex(ValueError, "GPU/source"):
                certify_identity(
                    root / "run", root / "arm", "a" * 40, root / "certificate.json"
                )
            self.assertFalse((root / "certificate.json").exists())

    def test_saved_gpu_sharding_refuses_cpu_or_missing_moments(self):
        import base64
        from scripts.exp12_launch_plan import verify_saved_gpu_sharding

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agent_ckpt").mkdir()
            fields = [
                n + ".params.kernel"
                for n in ("actor", "critic", "target_critic", "temperature")
            ]
            fields += [
                n + ".opt_state.0." + moment + ".weight"
                for n in ("actor", "critic", "temperature")
                for moment in ("mu", "nu")
            ]
            data = {
                base64.b64encode(n.encode()).decode(): json.dumps(
                    dict(sharding_type="SingleDeviceSharding", device_str="cuda:0")
                )
                for n in fields
            }
            path = root / "agent_ckpt/_sharding"
            path.write_text(json.dumps(data))
            verify_saved_gpu_sharding(
                root
            )  # encoded metadata fixture, no GPU execution
            key = base64.b64encode(b"actor.opt_state.0.mu.weight").decode()
            data[key] = json.dumps(
                dict(sharding_type="SingleDeviceSharding", device_str="TFRT_CPU_0")
            )
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "optimizer moments"):
                verify_saved_gpu_sharding(root)
            del data[key]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "optimizer moments"):
                verify_saved_gpu_sharding(root)

    def test_pinned_stack_matches_reviewed_requirements(self):
        from scripts.exp12_launch_plan import require_pinned_stack
        from unittest.mock import patch

        require_pinned_stack()
        with patch("importlib.metadata.version", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "requirement mismatch"):
                require_pinned_stack()

    def test_complete_population_and_unchanged_pilot(self):
        from generate_manifest import EXP12_ARCHS, EXP12_ENVS, exp12_overrides
        from scripts.exp12_launch_plan import config_hash

        rows = cells(Path("/external/new/parents"), "parents")
        self.assertEqual(len(rows), 195)
        self.assertEqual(len({r["run_key"] for r in rows}), 195)
        for row in rows:
            architecture = next(a for a in EXP12_ARCHS if a[0] == row["architecture"])
            expected = exp12_overrides(
                architecture,
                row["environment"],
                dict(EXP12_ENVS)[row["environment"]],
                row["seed"],
                "/other/results",
            )
            self.assertEqual(row["config_hash"], config_hash(expected))
        pilot = cells(Path("/external/new/dev"), "pilot")[0]
        self.assertEqual(
            (pilot["environment"], pilot["architecture"], pilot["seed"]),
            ("dog-run", "D4W1536", 102),
        )
        self.assertNotIn("testing.force_trigger_check", " ".join(pilot["overrides"]))
        self.assertIn("run_role=dev", pilot["overrides"])
        self.assertEqual(
            pilot["command"][-3:], ["25000", "--checkpoint_start_frac", "0.0"]
        )

    def test_exit_zero_or_done_alone_never_qualifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "DONE").write_text("{}")
            report = postvalidate(root, "a" * 40, root / "report.json")
            self.assertEqual(report["engineering_verdict"], "INCOMPLETE")
            self.assertEqual(report["scientific_verdict"], "NOT_QUALIFIED")

    def test_submission_prints_only_approved_resources_and_rejects_changed_approval(
        self,
    ):
        from scripts.exp12_launch_plan import prepare, submission_command

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "gpu.json"
            receipt.write_text(
                json.dumps(
                    dict(
                        verdict="PASS",
                        expected_commit="a" * 40,
                        qualification="fullwidth_resume_identity",
                        coverage=[
                            dict(
                                architecture="D4W1536",
                                suite="dmc",
                                environment="dog-run",
                            )
                        ],
                    )
                )
            )
            approval = self.approval("pilot")
            approval["gates"]["fullwidth_resume_identity"]["evidence"] = [
                dict(path=str(receipt), sha256=sha(receipt))
            ]
            approved = root / "approval.json"
            approved.write_text(json.dumps(approval))
            plan = root / "plan.json"
            prepare("pilot", root / "new-pilot", "a" * 40, approved, plan)
            command = submission_command(plan)
            self.assertIn("--cpus-per-task=4", command)
            self.assertIn("--mem=32G", command)
            self.assertIn("--time=7", command)
            self.assertNotIn("--array", command)
            self.assertFalse((root / "new-pilot").exists())
            approval["approved"] = False
            approved.write_text(json.dumps(approval))
            with self.assertRaisesRegex(ValueError, "changed"):
                submission_command(plan)


if __name__ == "__main__":
    unittest.main()
