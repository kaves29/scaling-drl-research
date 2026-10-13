"""Adversarial CPU checks for the shared, restartable Exp1/Exp2 scheduler."""

import json
import multiprocessing
import tempfile
import threading
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest import mock

from scripts import exp12_dynamic_scheduler as scheduler


def item(root, arch="D4W1536", kind="exp1", seed=1):
    return {
        "id": f"{kind}_{arch}_{seed}",
        "kind": kind,
        "row": {
            "architecture": arch,
            "environment": "dog-run",
            "group": "dmc_hard",
            "seed": seed,
            "config_hash": "config" + str(seed),
            "checkpoint_dir": str(root / f"run_{seed}"),
        },
    }


def owner(job, token):
    return {"job_id": str(job), "token": token, "host": "test", "pid": 1}


def pinned_launch(commit):
    return {
        "commit": commit,
        "dirty": False,
        "device": {
            "platform": "gpu",
            "device_kind": scheduler.GPU_KIND,
            "device_count": 1,
        },
        "runtime": {
            "jax": "0.4.34",
            "jaxlib": "0.4.34",
            "matmul_precision": "tensorfloat32",
        },
    }


def claim_in_process(root, number):
    path = Path(root)
    ledger = scheduler.Ledger(path, query=lambda job: True)
    return ledger.claim(item(path), owner(number, str(number)), lambda _: None)


def evidence(run):
    (run / "state").mkdir(parents=True)
    (run / "DONE").write_text('{"interaction_step": 500000}')
    (run / "run_metadata.json").write_text('{"code": "pinned"}')
    (run / "state" / "LATEST").write_text("step_000500000_abcd1234")


class DurableLedgerTest(unittest.TestCase):
    def test_one_immutable_queue_per_campaign_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            rows = [
                {
                    "environment": "dog-run",
                    "architecture": "D2W512",
                    "seed": n,
                    "config_hash": f"config-{n}",
                }
                for n in range(195)
            ]
            plan = {
                "stage": "parents",
                "commit": "a" * 40,
                "root": str(root / "campaign"),
                "entries": rows,
            }
            with mock.patch.object(scheduler, "_check_plan", return_value=plan):
                queue = scheduler.prepare(plan_path, root / "queue-one")
                self.assertEqual(len(queue["parents"]), 195)
                scheduler.load_queue(root / "queue-one")
                with self.assertRaises(FileExistsError):
                    scheduler.prepare(plan_path, root / "queue-two")
            self.assertFalse((root / "queue-two").exists())

    def test_concurrent_claims_have_one_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            ledger = scheduler.Ledger(root, query=lambda job: True)
            results = []
            gate = threading.Barrier(12)

            def race(number):
                gate.wait()
                results.append(
                    ledger.claim(row, owner(number + 1, str(number)), lambda _: None)
                )

            threads = [threading.Thread(target=race, args=(n,)) for n in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(sum(results), 1)
            claim = root / "ledger" / "exp1" / row["id"] / "claim"
            self.assertTrue((claim / "ready").exists())

    def test_separate_processes_cannot_claim_same_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with ProcessPoolExecutor(
                max_workers=8, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                results = list(
                    pool.map(
                        claim_in_process,
                        [directory] * 8,
                        list(range(1, 9)),
                    )
                )
            self.assertEqual(sum(results), 1)

    def test_completed_row_never_reclaimed_and_tampering_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            ledger = scheduler.Ledger(root, query=lambda job: False)
            first = owner(101, "first")
            self.assertTrue(ledger.claim(row, first, lambda _: None))
            evidence(Path(row["row"]["checkpoint_dir"]))
            receipt = root / "receipt.json"
            receipt.write_text('{"engineering_verdict":"COMPLETE"}')
            ledger.finish(row, first, **scheduler.bound_receipt(row, receipt))
            self.assertEqual(ledger.state(row), "complete")
            self.assertFalse(ledger.claim(row, owner(102, "second"), lambda _: receipt))
            (Path(row["row"]["checkpoint_dir"]) / "DONE").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "changed"):
                ledger.state(row)

    def test_stale_claim_recovery_preserves_old_owner_and_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            live = {"101"}
            ledger = scheduler.Ledger(root, query=lambda job: job in live)
            first, second = owner(101, "first"), owner(102, "second")
            self.assertTrue(ledger.claim(row, first, lambda _: None))
            attempt = root / "attempts" / row["id"] / "first"
            attempt.mkdir(parents=True)
            (attempt / "training.log").write_text("partial training")
            self.assertFalse(ledger.claim(row, second, lambda _: None))
            live.clear()
            self.assertTrue(ledger.claim(row, second, lambda _: None))
            retired = root / "ledger" / "exp1" / row["id"] / "retired_claims"
            self.assertEqual(len(list(retired.iterdir())), 1)
            self.assertEqual((attempt / "training.log").read_text(), "partial training")

    def test_valid_done_reconciles_and_invalid_done_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = item(root, seed=1)
            evidence(Path(good["row"]["checkpoint_dir"]))
            receipt = root / "validated.json"
            receipt.write_text('{"engineering_verdict":"COMPLETE"}')
            ledger = scheduler.Ledger(root, query=lambda job: False)
            self.assertFalse(ledger.claim(good, owner(1, "one"), lambda _: receipt))
            self.assertEqual(ledger.state(good), "complete")
            bad = item(root, seed=2)
            evidence(Path(bad["row"]["checkpoint_dir"]))
            self.assertFalse(
                ledger.claim(
                    bad,
                    owner(2, "two"),
                    lambda _: (_ for _ in ()).throw(
                        ValueError("invalid terminal checkpoint")
                    ),
                )
            )
            self.assertEqual(ledger.state(bad), "blocked")
            self.assertFalse(ledger.claim(bad, owner(3, "three"), lambda _: receipt))

    def test_reviewed_retry_preserves_failure_and_requires_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            ledger = scheduler.Ledger(root, query=lambda job: False)
            original = owner(1, "first")
            self.assertTrue(ledger.claim(row, original, lambda _: None))
            ledger.finish(row, original, error="interrupted after checkpoint")
            blocked = ledger._row_dir(row) / "blocked.json"
            decision = root / "retry.json"
            payload = {
                "approved": True,
                "expected_commit": "a" * 40,
                "item_id": row["id"],
                "blocked_sha256": scheduler.sha(blocked),
                "decision_reference": "owner-reviewed-checkpoint",
            }
            decision.write_text(json.dumps(payload))
            arm_row = dict(row, kind="exp2_injected")
            arm_owner = owner(2, "arm")
            self.assertTrue(ledger.claim(arm_row, arm_owner, lambda _: None))
            ledger.finish(arm_row, arm_owner, error="interrupted before checkpoint")
            arm_decision = root / "arm_retry.json"
            arm_decision.write_text(json.dumps(dict(
                payload, blocked_sha256=scheduler.sha(ledger._row_dir(arm_row) / "blocked.json"))))
            with mock.patch.object(scheduler, "validate_resume", return_value=False):
                with self.assertRaisesRegex(ValueError, "valid checkpoint"):
                    ledger.review_retry(arm_row, "a" * 40, arm_decision)
            self.assertEqual(ledger.state(arm_row), "blocked")
            self.assertEqual(ledger.state(row), "blocked")
            payload["blocked_sha256"] = "wrong"
            decision.write_text(json.dumps(payload))
            with mock.patch.object(scheduler, "validate_resume", return_value=True):
                with self.assertRaisesRegex(ValueError, "matching recovery"):
                    ledger.review_retry(row, "a" * 40, decision)
            payload["blocked_sha256"] = scheduler.sha(blocked)
            decision.write_text(json.dumps(payload))
            with mock.patch.object(scheduler, "validate_resume", return_value=True):
                ledger.review_retry(row, "a" * 40, decision)
            self.assertEqual(ledger.state(row), "pending")
            self.assertEqual(
                len(list((ledger._row_dir(row) / "retired_blocks").iterdir())), 1
            )
            self.assertEqual(
                len(list((ledger._row_dir(row) / "retry_decisions").iterdir())), 1
            )

    def test_partial_claim_or_guard_never_stolen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            ledger = scheduler.Ledger(root, query=lambda job: False)
            path = ledger._row_dir(row)
            (path / "claim").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "partial claim"):
                ledger.claim(row, owner(1, "one"), lambda _: None)
            (path / "transition_guard").mkdir()
            self.assertFalse(ledger.claim(row, owner(2, "two"), lambda _: None))

    def test_resume_refuses_missing_checkpoint_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            run = Path(row["row"]["checkpoint_dir"])
            run.mkdir()
            (run / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "code": {"commit": "old", "dirty": False},
                        "config_hash": row["row"]["config_hash"],
                        "identity": {
                            "architecture": row["row"]["architecture"],
                            "environment": row["row"]["environment"],
                            "seed": row["row"]["seed"],
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "source/config"):
                scheduler.validate_resume(row, "new")
            metadata = json.loads((run / "run_metadata.json").read_text())
            metadata["code"]["commit"] = "new"
            metadata["launches"] = [pinned_launch("new")]
            (run / "run_metadata.json").write_text(json.dumps(metadata))
            # No committed state and no fork: an Exp1 parent restarts from scratch (False),
            # an injected arm still refuses.
            self.assertFalse(scheduler.validate_resume(row, "new"))
            arm = dict(row, kind="exp2_injected")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                scheduler.validate_resume(arm, "new")
            state = run / "state" / "step_000025000_abcd1234"
            state.mkdir(parents=True)
            (run / "state" / "LATEST").write_text(state.name)
            with (
                mock.patch(
                    "experiments.exp12.state.latest_state_dir", return_value=state
                ),
                mock.patch(
                    "experiments.exp12.state.load_meta",
                    return_value={"interaction_step": 25000},
                ),
                mock.patch(
                    "experiments.exp12.state.load_agent_tree",
                    return_value={
                        "actor": object(),
                        "critic": object(),
                        "target_critic": object(),
                        "rng": object(),
                    },
                ),
                mock.patch("scripts.exp12_launch_plan.validate_replay_normalization"),
            ):
                self.assertTrue(scheduler.validate_resume(row, "new"))

    def test_parent_recovers_from_committed_natural_fork_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = item(root)
            run = Path(row["row"]["checkpoint_dir"])
            (run / "fork").mkdir(parents=True)
            (run / "fork" / "FORK_READY").write_text("natural")
            (run / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "code": {"commit": "a" * 40, "dirty": False},
                        "launches": [pinned_launch("a" * 40)],
                        "config_hash": row["row"]["config_hash"],
                        "identity": {
                            "architecture": row["row"]["architecture"],
                            "environment": row["row"]["environment"],
                            "seed": row["row"]["seed"],
                        },
                    }
                )
            )
            fork_state = run / "fork" / "state" / "step_000050000_abcd1234"
            with (
                mock.patch.object(
                    scheduler, "_natural_fork", return_value={"fork_step": 50000}
                ),
                mock.patch(
                    "experiments.exp12.state.latest_state_dir",
                    side_effect=[None, fork_state],
                ),
                mock.patch(
                    "experiments.exp12.state.load_meta",
                    return_value={"interaction_step": 50000},
                ),
                mock.patch(
                    "experiments.exp12.state.load_agent_tree",
                    return_value={
                        "actor": object(),
                        "critic": object(),
                        "target_critic": object(),
                        "rng": object(),
                    },
                ),
                mock.patch("scripts.exp12_launch_plan.validate_replay_normalization"),
            ):
                self.assertTrue(scheduler.validate_resume(row, "a" * 40))

    def test_arm_identity_has_separate_source_m_and_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_dir = (
                root / "checkpoints" / "exp1" / "D4W1536" / "dog-run" / "seed_1"
            )
            (parent_dir / "fork").mkdir(parents=True)
            (parent_dir / "fork" / "fork.json").write_text('{"fork_step": 50000}')
            parent = {
                "id": "parent-source",
                "kind": "exp1",
                "row": {
                    "architecture": "D4W1536",
                    "environment": "dog-run",
                    "seed": 1,
                    "group": "dmc_hard",
                    "budget": 1_000_000,
                    "run_key": "exp1_D4W1536_dog-run_seed1",
                    "checkpoint_dir": str(parent_dir),
                    "overrides": [
                        "critic_num_blocks=4",
                        "critic_hidden_dim=1536",
                        "env_name=dog-run",
                        "env=dmc_hard",
                        "seed=1",
                        f"results_root={root / 'results'}",
                    ],
                    "command": [
                        "python",
                        "-u",
                        "run.py",
                        "--experiment",
                        "exp1",
                        "--checkpoint_dir",
                        str(parent_dir),
                    ],
                },
            }
            with mock.patch.object(scheduler, "config_hash", return_value="arm-config"):
                last = scheduler._arm_item(parent, "a" * 40, "last")
                half = scheduler._arm_item(parent, "a" * 40, "half")
            self.assertNotEqual(last["id"], half["id"])
            self.assertNotEqual(last["id"], parent["id"])
            self.assertEqual(last["row"]["source_id"], parent["id"])
            self.assertEqual(
                last["row"]["checkpoint_dir"],
                str(
                    root
                    / "checkpoints"
                    / "exp2_arm"
                    / "D4W1536"
                    / "dog-run"
                    / "seed_1"
                    / "injected"
                ),
            )
            self.assertEqual(last["row"]["command"][3:5], ["--experiment", "exp2_arm"])
            self.assertIn("fork.arm=injected", last["row"]["overrides"])
            (parent_dir / "fork" / "fork.json").write_text('{"fork_step": 100000}')
            with mock.patch.object(scheduler, "config_hash", return_value="arm-config"):
                changed = scheduler._arm_item(parent, "a" * 40, "last")
            self.assertNotEqual(last["id"], changed["id"])

    def test_deferred_arm_discovery_after_parent_completion_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_dir = root / "checkpoints/exp1/D4W1536/dog-run/seed_1"
            (parent_dir / "fork").mkdir(parents=True)
            (parent_dir / "fork/fork.json").write_text('{"fork_step": 50000}')
            (parent_dir / "fork/FORK_READY").write_text("source")
            (parent_dir / "DONE").write_text('{"interaction_step": 500000}')
            parent = {"id": "source", "kind": "exp1", "row": {
                "architecture": "D4W1536", "environment": "dog-run", "seed": 1,
                "group": "dmc_hard", "budget": 1_000_000, "run_key": "source",
                "checkpoint_dir": str(parent_dir), "overrides": ["seed=1"],
                "command": ["python", "-u", "run.py", "--experiment", "exp1",
                            "--checkpoint_dir", str(parent_dir)],
            }}
            queue = {"parents": [parent], "commit": "a" * 40}
            (root / "queue").mkdir()
            with (mock.patch.object(scheduler, "_natural_fork", return_value={"fork_step": 50000}),
                  mock.patch.object(scheduler, "config_hash", return_value="arm-config")):
                first = scheduler.discover_arms(root / "queue", queue, "half")
                second = scheduler.discover_arms(root / "queue", queue, "half")
            self.assertEqual(first, second)
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["row"]["source_dir"], str(parent_dir))
            self.assertEqual(len(list((root / "queue/arms").glob("*.json"))), 1)

    def test_injected_arm_completion_requires_full_eval_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent"
            arm = root / "arm"
            results_root = root / "results"
            (parent / "fork").mkdir(parents=True)
            arm.mkdir()
            fork = {
                "fork_step": 10,
                "arm_end_step": 20,
                "horizon_steps": 10,
                "eval_every_steps": 10,
                "eval_episodes": 2,
            }
            fork_path = parent / "fork" / "fork.json"
            fork_path.write_text(json.dumps(fork))
            row = {
                "id": "arm-id",
                "kind": "exp2_injected",
                "row": {
                    "architecture": "D4W1536",
                    "environment": "dog-run",
                    "seed": 1,
                    "config_hash": "arm-config",
                    "checkpoint_dir": str(arm),
                    "source_dir": str(parent),
                    "source_id": "parent-id",
                    "fork_sha256": scheduler.sha(fork_path),
                    "injection_m": "half",
                    "run_key": "exp1_D4W1536_dog-run_seed1",
                },
            }
            (arm / "DONE").write_text(
                json.dumps(
                    {
                        "arm": "injected",
                        "interaction_step": 20,
                    }
                )
            )
            (arm / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "code": {"commit": "a" * 40, "dirty": False},
                        "launches": [pinned_launch("a" * 40)],
                        "config_hash": "arm-config",
                        "identity": {
                            "experiment": "exp2_arm",
                            "architecture": "D4W1536",
                            "environment": "dog-run",
                            "seed": 1,
                        },
                        "resolved_config": {
                            "fork": {"source": str(parent), "arm": "injected"},
                            "injection": {"m": "half"},
                            "results_root": str(results_root),
                        },
                    }
                )
            )
            result = results_root / "exp12" / "exp2" / row["row"]["run_key"]
            injected = result / "arm_injected"
            injected.mkdir(parents=True)
            (result / "check1_injected.json").write_text('{"pass":true}')
            (result / "check2.json").write_text('{"pass":false}')
            (injected / "checks.csv").write_text("arm,check_index\ninjected,2\n")
            (injected / "metrics.csv").write_text("arm,env_step\ninjected,20\n")
            evals = injected / "eval_episodes.csv"
            evals.write_text(
                "arm,eval_index,interaction_step,episode\n"
                "injected,0,10,0\ninjected,0,10,1\n"
            )
            with (
                mock.patch(
                    "experiments.exp12.state.latest_state_dir",
                    return_value=arm / "state",
                ),
                mock.patch(
                    "experiments.exp12.state.load_meta",
                    return_value={
                        "interaction_step": 20,
                        **{
                            name: 0
                            for name in (
                                "update_step",
                                "update_counter",
                                "numpy_rng_state",
                                "python_rng_state",
                                "train_env",
                                "eval_env",
                                "observations",
                                "timestep",
                                "meters",
                                "metrics_rows",
                                "eval_rows",
                                "extra_state",
                                "agent_window_buffers",
                            )
                        },
                    },
                ),
                mock.patch(
                    "experiments.exp12.state.load_agent_tree",
                    return_value={
                        "rng": 1,
                        "actor": 1,
                        "critic": 1,
                        "target_critic": 1,
                        "temperature": 1,
                    },
                ),
                mock.patch("scripts.exp12_launch_plan.validate_replay_normalization"),
            ):
                with self.assertRaisesRegex(ValueError, "evaluation schedule"):
                    scheduler.validate_arm(row, "a" * 40, root)
                evals.write_text(
                    evals.read_text() + "injected,1,20,0\ninjected,1,20,1\n"
                )
                receipt = scheduler.validate_arm(row, "a" * 40, root)
            self.assertEqual(json.loads(receipt.read_text())["id"], "arm-id")


class ResourceAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.approval = {
            "resources": {"cpus_per_node": 16, "memory_gb_per_node": 64},
            "admission": {
                "max_processes_per_gpu": 7,
                "gpu_headroom_gb": 3,
                "host_headroom_gb": 4,
                "stop_margin_minutes": 10,
                "architectures": {
                    arch: {
                        "cpus": 2,
                        "host_ram_gb": 5,
                        "gpu_memory_gb": 5 if arch == "D2W512" else 10,
                        "estimated_minutes_by_group": {"dmc_hard": 60},
                        "min_interactions_per_second": 10.0,
                    }
                    for arch in scheduler.WEIGHTS
                },
            },
        }

    def test_mixed_packing_waits_for_real_throughput_and_enforces_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            d2, d4 = item(root, "D2W512", seed=1), item(root, "D4W1536", seed=2)
            active = [
                {
                    "item": d4,
                    "gpu": "0",
                    "profile": self.approval["admission"]["architectures"]["D4W1536"],
                    "rate": None,
                }
            ]
            admission = self.approval["admission"]
            resources = self.approval["resources"]
            deadline = time.time() + 4 * 3600
            check = lambda: scheduler.can_admit(
                d2,
                "0",
                active,
                admission,
                resources,
                deadline,
                (40960, 11000),
                40,
            )
            self.assertFalse(check())  # no measured progress yet
            active[0]["rate"] = 9.0
            self.assertFalse(check())  # below approved throughput floor
            active[0]["rate"] = 11.0
            self.assertTrue(check())  # compatible mixed architecture
            admission["max_processes_per_gpu"] = 1
            self.assertFalse(check())  # unqualified packing can be disabled explicitly
            del admission["max_processes_per_gpu"]
            self.assertFalse(check())  # missing policy also fails closed to one process
            admission["max_processes_per_gpu"] = 7
            self.assertFalse(
                scheduler.can_admit(
                    d2,
                    "0",
                    active,
                    admission,
                    resources,
                    deadline,
                    (40960, 39000),
                    40,
                )
            )
            self.assertFalse(
                scheduler.can_admit(
                    d2,
                    "0",
                    active,
                    admission,
                    resources,
                    deadline,
                    (40960, 11000),
                    6,
                )
            )
            self.assertFalse(
                scheduler.can_admit(
                    d2,
                    "0",
                    active,
                    admission,
                    resources,
                    time.time() + 30 * 60,
                    (40960, 11000),
                    40,
                )
            )

    def test_seven_four_two_are_ceilings_not_launch_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = self.approval["admission"]["architectures"]["D2W512"]
            profile.update(cpus=1, host_ram_gb=1, gpu_memory_gb=1)
            six = [
                {
                    "item": item(root, "D2W512", seed=n),
                    "gpu": "0",
                    "profile": profile,
                    "rate": 11.0,
                }
                for n in range(1, 7)
            ]
            candidate = item(root, "D2W512", seed=7)
            deadline = time.time() + 4 * 3600
            self.assertTrue(
                scheduler.can_admit(
                    candidate,
                    "0",
                    six,
                    self.approval["admission"],
                    self.approval["resources"],
                    deadline,
                    (40960, 0),
                    60,
                )
            )
            seven = six + [
                {
                    "item": candidate,
                    "gpu": "0",
                    "profile": profile,
                    "rate": 11.0,
                }
            ]
            self.assertFalse(
                scheduler.can_admit(
                    item(root, "D2W512", seed=8),
                    "0",
                    seven,
                    self.approval["admission"],
                    self.approval["resources"],
                    deadline,
                    (40960, 0),
                    60,
                )
            )

    def test_allocation_shape_and_positive_control_fail_closed(self):
        with mock.patch.object(
            scheduler,
            "load_queue",
            return_value={
                "commit": "a" * 40,
            },
        ), mock.patch.object(scheduler, "sha", return_value="digest"):
            root = Path("/tmp/dynamic-queue-for-test")
            approval = {
                "approved": True,
                "allocation": "B",
                "expected_commit": "a" * 40,
                "queue_sha256": "digest",
                "decision_reference": "owner",
                "resources": {
                    "nodes": 2,
                    "gpus_per_node": 4,
                    "wall_minutes": 2880,
                    "cpus_per_node": 16,
                    "memory_gb_per_node": 64,
                    "humanoid_bench_python": "/approved/hb/bin/python",
                },
                "admission": self.approval["admission"],
                "exp2": {"approved": False},
            }
            for profile in approval["admission"]["architectures"].values():
                profile["estimated_minutes_by_group"] = {
                    "dmc_hard": 60,
                    "dmc_medium": 40,
                    "myosuite_simba": 90,
                    "humanoid_bench": 120,
                }
                profile["estimated_arm_minutes_by_group"] = {
                    "dmc_hard": 30,
                    "dmc_medium": 20,
                    "myosuite_simba": 45,
                    "humanoid_bench": 60,
                }
            scheduler.validate_approval(approval, root, "B")
            approval["admission"]["max_processes_per_gpu"] = 8
            with self.assertRaisesRegex(ValueError, "GPU process cap"):
                scheduler.validate_approval(approval, root, "B")
            approval["admission"]["max_processes_per_gpu"] = 7
            with tempfile.TemporaryDirectory() as evidence_dir:
                report = Path(evidence_dir) / "positive_control.json"
                report.write_text(
                    json.dumps(
                        {
                            "status": "stop",
                            "chosen_m": None,
                            "source_code": {"commit": "a" * 40, "dirty": False},
                            "allow_any_setting": False,
                            "trigger_check": {"forced": False},
                        }
                    )
                )
                approval["exp2"] = {
                    "approved": True,
                    "injection_m": "last",
                    "decision_reference": "owner-exp2",
                    "positive_control_evidence": str(report),
                    "positive_control_sha256": "digest",
                }
                with self.assertRaisesRegex(ValueError, "positive control"):
                    scheduler.validate_approval(approval, root, "B")
            approval["exp2"] = {"approved": False}
            approval["resources"]["nodes"] = 1
            with self.assertRaisesRegex(ValueError, "exactly"):
                scheduler.validate_approval(approval, root, "B")

    def test_actual_two_node_allocation_must_match_approved_resources(self):
        approval = {
            "resources": {
                "nodes": 2,
                "gpus_per_node": 4,
                "cpus_per_node": 16,
                "memory_gb_per_node": 64,
                "wall_minutes": 2880,
            },
        }
        environment = {
            "SLURM_CPUS_PER_TASK": "16",
            "SLURM_MEM_PER_NODE": str(64 * 1024),
            "SLURM_GPUS_ON_NODE": "4",
            "SLURM_JOB_NUM_NODES": "2",
        }
        description = "JobId=11 TimeLimit=2-00:00:00 EndTime=2026-10-15T12:00:00"
        with mock.patch.dict("os.environ", environment), mock.patch(
            "os.sched_getaffinity", return_value=set(range(16))
        ):
            self.assertGreater(
                scheduler._actual_allocation(
                    approval, description, ["0", "1", "2", "3"]
                ),
                0,
            )
            with self.assertRaisesRegex(ValueError, "node/GPU allocation"):
                scheduler._actual_allocation(approval, description, ["0", "1", "2"])


if __name__ == "__main__":
    unittest.main()
