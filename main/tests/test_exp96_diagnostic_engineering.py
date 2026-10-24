"""Fail-closed diagnostic configuration and numerical engineering contracts."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.exp96_fixed_task_diagnostics import ARCHS, FROZEN, Investigation
from utils.run_metadata import _hash, _strip_locations


class MetadataFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for env in ("dog-run", "humanoid-run", "myo-key-turn", "myo-pen-twirl"):
            for arch, (depth, width) in ARCHS.items():
                for seed in range(1, 9):
                    cp = self.root / "checkpoints" / env / arch / str(seed)
                    cp.mkdir(parents=True)
                    cfg = dict(
                        seed=seed,
                        env_name=env,
                        agent=dict(
                            critic_learning_rate=1e-4,
                            critic_weight_decay=0.01,
                            critic_block_type="residual",
                            critic_num_blocks=depth,
                            critic_hidden_dim=width,
                            critic_use_cdq=False,
                            mixed_precision=False,
                            critic_sparsity=0.0,
                        ),
                    )
                    row = dict(
                        environment=env,
                        architecture=arch,
                        seed=seed,
                        run_key=f"exp1_{arch}_{env}_seed{seed}",
                        checkpoint_dir=str(cp),
                        config_hash=_hash(_strip_locations(cfg)),
                    )
                    meta = dict(
                        code=dict(commit=FROZEN, dirty=False),
                        identity={
                            k: row[k]
                            for k in ("environment", "architecture", "seed", "run_key")
                        },
                        config_hash=row["config_hash"],
                        resolved_config=cfg,
                        launches=[dict(commit=FROZEN, dirty=False)],
                    )
                    (cp / "run_metadata.json").write_text(json.dumps(meta))
                    self.rows.append(row)
        (self.root / "queue").mkdir()
        self.queue = dict(commit=FROZEN, parents=[dict(row=r) for r in self.rows])
        self.save_queue()
        self.args = SimpleNamespace(
            root=self.root / "output",
            campaign=self.root,
            commit="a" * 40,
            cap_seconds=600,
        )

    def save_queue(self):
        (self.root / "queue/queue.json").write_text(json.dumps(self.queue))

    def edit_metadata(self, edit):
        row = self.rows[0]
        p = Path(row["checkpoint_dir"]) / "run_metadata.json"
        meta = json.loads(p.read_text())
        edit(meta)
        p.write_text(json.dumps(meta))


class OptimizerBindingTest(MetadataFixture):
    def test_constructor_uses_archived_optimizer_instead_of_generic_sac(self):
        from scripts import exp96_fixed_task_diagnostics as runner

        original = runner.critic_optimizer
        with mock.patch.object(runner, "critic_optimizer", wraps=original) as factory:
            investigation = Investigation(self.args)
        cfg = factory.call_args.args[0]
        self.assertEqual(cfg.critic_learning_rate, 1e-4)
        self.assertEqual(cfg.critic_weight_decay, 0.01)
        self.assertEqual(len(investigation.optimizer_provenance), 96)

    def test_missing_metadata_is_rejected_without_fallback(self):
        (Path(self.rows[0]["checkpoint_dir"]) / "run_metadata.json").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            Investigation(self.args)

    def test_source_identity_and_config_hash_mismatches_are_rejected(self):
        for mutate in (
            lambda m: m["code"].update(commit="b" * 40),
            lambda m: m["identity"].update(seed=99),
            lambda m: m["resolved_config"]["agent"].update(critic_learning_rate=0.002),
            lambda m: m["launches"][0].update(dirty=True),
        ):
            p = Path(self.rows[0]["checkpoint_dir"]) / "run_metadata.json"
            before = p.read_text()
            self.edit_metadata(mutate)
            with self.assertRaises(ValueError):
                Investigation(self.args)
            p.write_text(before)

    def test_duplicate_queue_identity_is_rejected(self):
        self.queue["parents"].append(copy.deepcopy(self.queue["parents"][0]))
        self.save_queue()
        with self.assertRaises(ValueError):
            Investigation(self.args)

    def test_valid_but_conflicting_parent_optimizers_are_rejected(self):
        def change(m):
            m["resolved_config"]["agent"]["critic_learning_rate"] = 0.0003
            m["config_hash"] = _hash(_strip_locations(m["resolved_config"]))
            self.rows[0]["config_hash"] = m["config_hash"]

        self.edit_metadata(change)
        self.save_queue()
        with self.assertRaises(ValueError):
            Investigation(self.args)


class NumericalInstrumentationTest(unittest.TestCase):
    def setUp(self):
        from tests.test_fixed_task import FixedTaskTests

        self.f = FixedTaskTests()
        self.f.setUp()

    def test_first_update_matches_independent_adamw_reference_and_preserves_input(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from experiments.exp12 import fixed_task as ft

        f = self.f
        state = ft.FitState(f.p, f.tx.init(f.p))
        before_hash = ft.tree_sha256(state)

        def loss(params):
            ix = f.idx[0]
            q = f.net.apply({"params": params}, f.o[ix], f.a[ix]).reshape(-1)
            return jnp.mean((q - f.z[ix]) ** 2)

        grads = jax.grad(loss)(f.p)
        updates, _ = f.tx.update(grads, state.optimizer, f.p)
        fitted = optax.apply_updates(f.p, updates)
        q0 = f.net.apply({"params": f.p}, f.o, f.a).reshape(-1)
        q1 = f.net.apply({"params": fitted}, f.o, f.a).reshape(-1)
        report = ft.first_update(f.net, f.tx, state, f.o, f.a, f.z, f.idx[0], 16)
        np.testing.assert_allclose(
            report["prediction_change_rms"],
            jnp.sqrt(jnp.mean((q1 - q0) ** 2)),
            rtol=1e-4,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            report["full_pool_mse_after"], jnp.mean((q1 - f.z) ** 2), rtol=1e-6
        )
        np.testing.assert_allclose(
            report["gradient_l2"], optax.global_norm(grads), rtol=1e-6
        )
        self.assertEqual(ft.tree_sha256(state), before_hash)

    def test_continuous_2k_4k_8k_matches_monolithic_real_residual_critic(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from experiments.exp12 import fixed_task as ft
        from scale_rl.agents.sac.sac_network import SACCritic

        f = self.f
        net = SACCritic("residual", 2, 16, jnp.float32)
        params = net.init(jax.random.PRNGKey(7), f.o[:1], f.a[:1])["params"]
        initial = ft.FitState(params, f.tx.init(params))
        st = initial
        curves = []
        for k in range(8):
            st, curve = ft.fit_segment(
                net, f.tx, st, f.o, f.a, f.z, f.idx[k * 1000 : (k + 1) * 1000]
            )
            curves.append(curve)
            if k + 1 in (2, 4, 8):
                full, losses = ft.fit_segment(
                    net, f.tx, initial, f.o, f.a, f.z, f.idx[: (k + 1) * 1000]
                )
                np.testing.assert_allclose(
                    jnp.concatenate(curves), losses, rtol=2e-5, atol=2e-6
                )
                for a, b in zip(
                    jax.tree_util.tree_leaves(st), jax.tree_util.tree_leaves(full)
                ):
                    np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
                self.assertEqual(int(st.optimizer[0].count), (k + 1) * 1000)

    def test_trained_and_fork_checkpoint_parameters_are_exact_and_read_only(self):
        import orbax.checkpoint as ocp
        from experiments.exp12.diagnostic_artifacts import load_critic
        from experiments.exp12 import fixed_task as ft

        f = self.f
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind, relative in [("latest", "state"), ("early", "fork/state")]:
                folder = root / relative / "snapshot"
                folder.mkdir(parents=True)
                (folder.parent / "LATEST").write_text("snapshot")
                ocp.PyTreeCheckpointer().save(
                    str(folder / "agent_ckpt"),
                    {"critic": {"params": f.p}, "actor": {"params": f.p}},
                )
                before = {
                    str(p.relative_to(root)): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                }
                params, _ = load_critic(root, kind)
                self.assertEqual(ft.tree_sha256(params), ft.tree_sha256(f.p))
                after = {
                    str(p.relative_to(root)): p.read_bytes()
                    for p in root.rglob("*")
                    if p.is_file()
                }
                self.assertEqual(before, after)

    def test_failed_gpu_guard_is_retained_and_nonfinite_rejected(self):
        import numpy as np
        from experiments.exp12 import fixed_task as ft

        report = ft.equivalence_report(
            np.ones(2000), 0.1502770483493805, np.ones(2000), 0.1543287783493805
        )
        self.assertFalse(report["passed_existing_final_guard"])
        self.assertAlmostEqual(report["final_abs_guard"], 0.0007513852417469024)
        with self.assertRaises(ValueError):
            ft.equivalence_report([np.nan], 1, [1], 1)
        self.assertIsNone(ft.curve_summary([0.0, 0.0])["early_peak_over_initial"])


class PlanAndRunnerTest(MetadataFixture):
    # Reuses validated canonical metadata fixtures; inherited provenance regressions
    # remain useful while running the actual fit path below.
    def test_explicit_fit_persists_continuation_and_first_update(self):
        from tests.test_fixed_task import FixedTaskTests
        from experiments.exp12.probe_tasks import FrozenTask
        import numpy as np

        f = FixedTaskTests()
        f.setUp()
        runner = Investigation(self.args)
        runner.eval_chunk = 16
        with mock.patch.object(
            runner, "model", return_value=(f.net, f.p)
        ), mock.patch.object(runner, "allowed", return_value=True):
            self.assertTrue(
                runner.fit(
                    "controlled",
                    "myo-pen-twirl",
                    "D2W512",
                    5,
                    "fresh",
                    f.o,
                    f.a,
                    f.z,
                    f.idx,
                    "shared",
                    8000,
                    shared_offset=2.0,
                )
            )
        row = runner.rows[-1]
        self.assertEqual(
            [m["step"] for m in row["measurements"]], list(range(0, 8001, 1000))
        )
        self.assertEqual(row["optimizer_count"], 8000)
        self.assertEqual(row["status"], "COMPLETE")
        self.assertTrue(row["starting_parameters_unchanged"])
        self.assertGreater(row["first_update"]["prediction_change_rms"], 0)
        task = FrozenTask.load(runner.root / "controlled/numeric_task")
        np.testing.assert_array_equal(task.targets, np.asarray(f.z + 2))
        with np.load(runner.root / "controlled/curve_through_8000.npz") as z:
            self.assertEqual(z["losses"].shape, (8000,))

    def test_plan_executor_reuses_identical_tasks_across_architectures(self):
        from tests.test_fixed_task import FixedTaskTests
        from scripts import exp96_fixed_task_diagnostics as module

        f = FixedTaskTests()
        f.setUp()
        runner = Investigation(self.args)
        runner.eval_chunk = 16
        main = Path(__file__).resolve().parents[1]
        runner.plan = json.loads(
            (main / "configs/diagnostics/architectures.json").read_text()
        )
        runner.plan["conditions"][0]["architectures"] = ["D2W512", "D4W1024"]
        with mock.patch.object(
            runner, "pool", return_value=("pool", f.o, f.a)
        ), mock.patch.object(
            runner, "model", return_value=(f.net, f.p)
        ), mock.patch.object(
            runner, "allowed", return_value=True
        ), mock.patch.object(
            module, "_base_targets", return_value=f.z
        ):
            self.assertTrue(runner.run())
        self.assertEqual(len(runner.rows), 2)
        self.assertEqual(runner.rows[0]["task_id"], runner.rows[1]["task_id"])
        self.assertEqual(
            runner.rows[0]["final_state_sha256"], runner.rows[1]["final_state_sha256"]
        )
        self.assertEqual(runner.condition_status[0]["status"], "COMPLETE")

    def test_budget_skip_is_explicit_and_unknown_targets_fail(self):
        runner = Investigation(self.args)
        with mock.patch.object(runner, "allowed", return_value=False):
            self.assertFalse(
                runner.fit(
                    "skip",
                    "dog-run",
                    "D2W512",
                    1,
                    "fresh",
                    None,
                    None,
                    None,
                    list(range(2000)),
                    "zero",
                    2000,
                )
            )
        self.assertEqual(runner.rows[-1]["status"], "SKIPPED_BUDGET")
        with self.assertRaises(ValueError):
            runner.fit(
                "bad",
                "dog-run",
                "D2W512",
                1,
                "fresh",
                None,
                None,
                None,
                [],
                "typo",
                2000,
            )

    def test_all_templates_load_and_reject_silent_parameter_changes(self):
        from experiments.exp12.diagnostic_plan import load_plan

        main = Path(__file__).resolve().parents[1]
        paths = list((main / "configs/diagnostics").glob("*.json"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            self.assertEqual(len(load_plan(path)[1]), 64)
        plan = json.loads((main / "configs/diagnostics/pen_targets.json").read_text())
        plan["conditions"][0]["learning_rate"] = 1e-3
        path = self.root / "bad-plan.json"
        path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            load_plan(path)

    def test_latest_cannot_escape_original_checkpoint(self):
        from experiments.exp12.diagnostic_artifacts import state_dir

        state = self.root / "state"
        state.mkdir()
        (state / "LATEST").write_text("../../outside")
        with self.assertRaises(ValueError):
            state_dir(self.root, "latest")


if __name__ == "__main__":
    unittest.main()
