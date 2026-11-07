"""CPU-only guard checks for the manually submitted runtime diagnostic."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/sbatch_exp12_runtime_diagnostic.sh"
)
MOCK_PYTHON = r"""
import json, os, sys
from pathlib import Path
root = Path(os.environ['OUT'])
mode = os.environ.get('FAKE_MODE', '')
backend_env = {name: os.environ.get(name) for name in
    ('JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'JAX_ENABLE_X64')}
with (root / 'python-envs.jsonl').open('a') as calls:
    calls.write(json.dumps(dict(args=sys.argv[1:], env=backend_env)) + '\n')
if sys.argv[1:] == ['-']:
    source = sys.stdin.read()
    if 'import jax' in source:
        (root / 'backend-env.json').write_text(json.dumps({name: os.environ.get(name) for name in
            ('JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'JAX_ENABLE_X64')}))
        if mode in ('backend-startup', 'missing-cpu', 'cpu-only'):
            import jax
            from jax._src import xla_bridge
            from types import SimpleNamespace
            device = SimpleNamespace(platform='gpu', device_kind=os.environ['EXPECTED_GPU_MODEL'])
            class FakeCudaClient:
                platform = 'gpu'
                def device_count(self): return 1
                def process_index(self): return 0
                def devices(self): return [device]
                def local_devices(self): return [device]
            if mode != 'cpu-only':
                xla_bridge.register_backend_factory('cuda', FakeCudaClient, priority=200, fail_quietly=False)
            xla_bridge.hardware_utils.has_visible_nvidia_gpu = lambda: mode != 'cpu-only'
            if mode == 'missing-cpu':
                xla_bridge.register_backend_factory('cpu', lambda: None, priority=0, fail_quietly=False)
            exec(compile(source, '<real backend preflight; mock CUDA client>', 'exec'))
            cpu = jax.local_devices(backend='cpu')[0]
            with jax.default_device(cpu):
                assert jax.config.jax_default_device is cpu
        else:
            (root / 'backend.json').write_text('{"mock": true}')
    else:
        exec(compile(source, '<real output validator>', 'exec'))
elif sys.argv[1] == 'scripts/compare_identity_fork.py':
    (root / 'identity.json').write_text(json.dumps(dict(pass_=True, differences=[], fork_step=12000,
        interaction_step=13000, identity_interaction_step=13000)).replace('"pass_"', '"pass"'))
elif 'run.py' in sys.argv:
    with (root / 'identity-commands.jsonl').open('a') as commands:
        commands.write(json.dumps(dict(args=sys.argv[1:], cache=os.environ['EXP12_JAX_CACHE_DIR'])) + '\n')
    target = Path(sys.argv[sys.argv.index('--out') + 1])
    target.write_text(json.dumps(dict(event='summary', error=None)) + '\n')
else:
    with (root / 'profile-calls.txt').open('a') as calls:
        calls.write('call\n')
    (root / 'profile-args.json').write_text(json.dumps(sys.argv[1:]))
    if mode == 'profile-error':
        sys.exit(9)
    row = dict(arch=os.environ.get('DIAGNOSTIC_ARCH', 'D4W1536'), env='dog-run', num_interaction_steps=500000,
               train_it_per_s_probes_off=1.0, probe_check_s=1.0, fork_save_s=1.0,
               fork_restore_s=1.0, fork_state_bytes=100, post_fork_eval_s=1.0,
               fork_buffer_transitions=475000, post_fork_eval_episodes=10,
               probe_check_s_all=[1.0], peak_device_bytes=None)
    if mode == 'nonfinite':
        row['probe_check_s'] = float('nan')
    if mode != 'missing-output':
        (root / 'profile.json').write_text(json.dumps([row]))
    totals = {name: {'calls': count} for name, count in
              [('update_many', 1062), ('probe_round', int(os.environ.get('FAKE_ROUNDS', '10'))),
               ('_fit', int(os.environ.get('FAKE_FITS', '20')))]}
    if mode == 'wrong-count':
        totals['update_many']['calls'] += 1
    events = [dict(event='progress', interaction_step=6061, update_step=2124),
              dict(event='summary', error=None, totals=totals)]
    if mode == 'cache-error':
        events.insert(0, dict(event='end', stage='cache_read', error='ValueError'))
    (root / 'trace.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
"""


class RuntimeDiagnosticLauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.main = self.repo / "main"
        scripts = self.main / "scripts"
        scripts.mkdir(parents=True)
        for name in ("profile_exp12.py", "trace_exp12_runtime.py"):
            (scripts / name).touch()
        (self.main / "run.py").touch()
        (scripts / "compare_identity_fork.py").touch()
        self.script = scripts / SCRIPT.name
        shutil.copyfile(SCRIPT, self.script)
        self.git("init", "-b", "claude/eloquent-fermat-inxqlt")
        self.git("add", "main")
        self.git(
            "-c",
            "user.name=Guard Test",
            "-c",
            "user.email=guard@example.invalid",
            "commit",
            "-m",
            "fixture",
        )
        self.commit = self.git("rev-parse", "HEAD").strip()
        mock_bin = self.root / "bin"
        mock_bin.mkdir()
        python = mock_bin / "python"
        python.write_text(f"#!{sys.executable}\n" + MOCK_PYTHON)
        python.chmod(0o755)
        timeout = mock_bin / "timeout"
        timeout.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$@" > "$OUT/timeout-args.txt"\n'
            'if [ "${FAKE_MODE:-}" = timeout ]; then exit 124; fi\n'
            'shift 3\nexec "$@"\n'
        )
        timeout.chmod(0o755)
        self.out = self.root / "output"
        self.env = dict(
            os.environ,
            PATH=str(mock_bin) + os.pathsep + os.environ["PATH"],
            SLURM_JOB_ID="123",
            SLURM_SUBMIT_DIR=str(self.main),
            EXPECTED_COMMIT=self.commit,
            EXPECTED_GPU_MODEL="Mock A100",
            OUT=str(self.out),
        )

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, text=True, stderr=subprocess.DEVNULL
        )

    def run_script(self, **updates):
        env = dict(self.env, **updates)
        return subprocess.run(
            ["bash", str(self.script)],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )

    def test_no_allocation_refuses_before_output(self):
        del self.env["SLURM_JOB_ID"]
        self.assert_preflight_failure(self.run_script(), "SLURM_JOB_ID")
        self.assertFalse(self.out.exists())

    def assert_preflight_failure(self, result, reason):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preflight failed:", result.stderr)
        self.assertIn(reason, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_wrong_revision_refuses_before_output(self):
        self.assert_preflight_failure(
            self.run_script(EXPECTED_COMMIT="0" * 40), "HEAD mismatch"
        )
        self.assertFalse(self.out.exists())

    def test_exact_revision_on_other_branch_is_accepted(self):
        self.git("switch", "-c", "other")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.out / "commit.txt").read_text().strip(), self.commit)

    def test_detached_exact_revision_is_accepted_in_all_modes(self):
        self.git("switch", "--detach", self.commit)
        self.assertEqual(self.git("branch", "--show-current").strip(), "")
        for mode in ("profile", "identity_cold", "identity_warm"):
            with self.subTest(mode=mode):
                out = self.root / mode
                result = self.run_script(DIAGNOSTIC_MODE=mode, OUT=str(out))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((out / "commit.txt").read_text().strip(), self.commit)
                self.assertTrue((out / "validation.json").is_file())

    def test_detached_wrong_revision_is_rejected(self):
        self.git("switch", "--detach", self.commit)
        self.assert_preflight_failure(
            self.run_script(EXPECTED_COMMIT="0" * 40), "HEAD mismatch"
        )
        self.assertFalse(self.out.exists())

    def test_expected_commit_must_be_full_literal_hash(self):
        for expected in ("HEAD", self.commit[:12], "z" * 40):
            with self.subTest(expected=expected):
                self.assert_preflight_failure(
                    self.run_script(EXPECTED_COMMIT=expected), "40-character"
                )
                self.assertFalse(self.out.exists())

    def test_missing_required_environment_is_explained(self):
        for name in (
            "SLURM_SUBMIT_DIR",
            "OUT",
            "EXPECTED_COMMIT",
            "EXPECTED_GPU_MODEL",
        ):
            with self.subTest(name=name):
                value = self.env.pop(name)
                try:
                    self.assert_preflight_failure(self.run_script(), name)
                    self.assertFalse(self.out.exists())
                finally:
                    self.env[name] = value

    def test_missing_required_files_are_explained(self):
        for mode, name in (
            ("profile", "scripts/profile_exp12.py"),
            ("profile", "scripts/trace_exp12_runtime.py"),
            ("identity_cold", "run.py"),
            ("identity_warm", "scripts/compare_identity_fork.py"),
        ):
            with self.subTest(mode=mode, name=name):
                path = self.main / name
                contents = path.read_bytes()
                path.unlink()
                try:
                    self.assert_preflight_failure(
                        self.run_script(DIAGNOSTIC_MODE=mode), "required file: " + name
                    )
                    self.assertFalse(self.out.exists())
                finally:
                    path.write_bytes(contents)

    def test_invalid_submit_directory_is_explained(self):
        self.assert_preflight_failure(
            self.run_script(SLURM_SUBMIT_DIR=str(self.root / "missing")),
            "cannot enter SLURM_SUBMIT_DIR",
        )
        self.assertFalse(self.out.exists())

    def test_missing_git_metadata_is_explained(self):
        metadata = self.repo / ".git"
        saved = self.root / "saved-git"
        metadata.rename(saved)
        try:
            self.assert_preflight_failure(self.run_script(), "cannot resolve Git HEAD")
            self.assertFalse(self.out.exists())
        finally:
            saved.rename(metadata)

    def test_tracked_changes_are_rejected_when_detached(self):
        self.git("switch", "--detach", self.commit)
        (self.main / "scripts/profile_exp12.py").write_text("# changed\n")
        self.assert_preflight_failure(self.run_script(), "working tree is not clean")
        self.assertFalse(self.out.exists())

    def test_staged_changes_are_rejected_when_detached(self):
        self.git("switch", "--detach", self.commit)
        (self.main / "scripts/profile_exp12.py").write_text("# changed\n")
        self.git("add", "main/scripts/profile_exp12.py")
        self.assert_preflight_failure(self.run_script(), "working tree is not clean")
        self.assertFalse(self.out.exists())

    def test_dirty_checkout_refuses_before_output(self):
        (self.main / "untracked.txt").touch()
        self.assert_preflight_failure(self.run_script(), "working tree is not clean")
        self.assertFalse(self.out.exists())

    def test_reused_output_is_preserved(self):
        self.out.mkdir()
        sentinel = self.out / "existing.txt"
        sentinel.write_text("preserve")
        self.assert_preflight_failure(
            self.run_script(), "cannot create new OUT directory"
        )
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_output_inside_checkout_refuses(self):
        target = self.main / "output"
        self.assert_preflight_failure(
            self.run_script(OUT=str(target)), "outside the checkout"
        )
        self.assertFalse(target.exists())

    def test_invalid_output_paths_are_explained(self):
        for out, reason in (
            ("relative-output", "OUT must be absolute"),
            (str(self.root / "missing" / "output"), "OUT parent does not exist"),
        ):
            with self.subTest(out=out):
                self.assert_preflight_failure(self.run_script(OUT=out), reason)
                self.assertFalse(self.out.exists())

    def test_invalid_diagnostic_selections_are_explained(self):
        for updates, reason in (
            ({"DIAGNOSTIC_MODE": "invalid"}, "unsupported DIAGNOSTIC_MODE"),
            ({"DIAGNOSTIC_ARCH": "D6W1536"}, "unsupported DIAGNOSTIC_ARCH"),
            (
                {"DIAGNOSTIC_MODE": "identity_cold", "DIAGNOSTIC_ARCH": "D2W512"},
                "identity mode requires a D4 architecture",
            ),
        ):
            with self.subTest(updates=updates):
                self.assert_preflight_failure(self.run_script(**updates), reason)
                self.assertFalse(self.out.exists())

    def test_non_a100_refuses_before_output(self):
        self.assert_preflight_failure(
            self.run_script(EXPECTED_GPU_MODEL="Other GPU"),
            "EXPECTED_GPU_MODEL must name an A100",
        )
        self.assertFalse(self.out.exists())

    def test_success_requires_bounded_command_and_expected_outputs(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.out / "exit_status.txt").read_text(), "0\n")
        self.assertEqual(
            (self.out / "timeout-args.txt").read_text().splitlines(),
            ["-k", "15", "300", "bash"],
        )
        args = json.loads((self.out / "profile-args.json").read_text())
        self.assertIn("--synchronize", args)
        self.assertIn("--boundary-detail", args)
        self.assertEqual(args[args.index("--stack-after") + 1], "60")
        self.assertEqual(args[args.index("--archs") + 1], "D4W1536")
        self.assertEqual(args[args.index("--train_steps") + 1], "60")
        self.assertEqual(args[args.index("--warmup_steps") + 1], "1001")
        self.assertEqual(args[args.index("--probe_repeats") + 1], "1")
        self.assertNotIn("--override", args)
        validation = json.loads((self.out / "validation.json").read_text())
        self.assertFalse(validation["memory_measurement_available"])
        self.assertEqual(validation["sac_updates"], 2124)

    def test_all_modes_export_cuda_and_cpu_without_legacy_default_override(self):
        for mode in ("profile", "identity_cold", "identity_warm"):
            with self.subTest(mode=mode):
                out = self.root / mode
                result = self.run_script(
                    OUT=str(out),
                    DIAGNOSTIC_MODE=mode,
                    JAX_PLATFORMS="cpu",
                    JAX_PLATFORM_NAME="cpu",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    json.loads((out / "backend-env.json").read_text()),
                    {
                        "JAX_PLATFORMS": "cuda,cpu",
                        "JAX_PLATFORM_NAME": None,
                        "JAX_ENABLE_X64": "false",
                    },
                )
                calls = [
                    json.loads(line)
                    for line in (out / "python-envs.jsonl").read_text().splitlines()
                ]
                self.assertGreater(len(calls), 1)
                for call in calls:
                    self.assertEqual(
                        call["env"],
                        {
                            "JAX_PLATFORMS": "cuda,cpu",
                            "JAX_PLATFORM_NAME": None,
                            "JAX_ENABLE_X64": "false",
                        },
                    )

    def test_real_jax_registry_allows_explicit_cpu_device_with_mock_cuda(self):
        result = self.run_script(FAKE_MODE="backend-startup", JAX_PLATFORM_NAME="cpu")
        log = (self.out / "profile.log").read_text()
        self.assertEqual(result.returncode, 0, result.stderr + log)
        metadata = json.loads((self.out / "backend.json").read_text())
        self.assertEqual(metadata["jax_platforms"], "cuda,cpu")
        self.assertTrue(metadata["cpu_devices"])
        self.assertTrue((self.out / "validation.json").exists())

    def test_missing_cpu_backend_stops_before_profile_execution(self):
        result = self.run_script(FAKE_MODE="missing-cpu")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Unable to initialize backend 'cpu'", (self.out / "profile.log").read_text()
        )
        self.assertFalse((self.out / "profile-calls.txt").exists())
        self.assertFalse((self.out / "validation.json").exists())

    def test_cpu_only_backend_cannot_pass_gpu_preflight(self):
        result = self.run_script(FAKE_MODE="cpu-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AssertionError", (self.out / "profile.log").read_text())
        self.assertFalse((self.out / "profile-calls.txt").exists())
        self.assertFalse((self.out / "backend.json").exists())

    def test_real_probe_schedule_matches_validator(self):
        import jax.numpy as jnp
        from experiments.exp12 import probe
        from experiments.exp12.run_probes import run_probe_networks

        cfg = probe.ProbeConfig(
            rounds=5,
            steps=2,
            checks=20,
            pool_size=8,
            batch_size=2,
            target_scale=1.0,
            eval_chunk=8,
        )
        critics = {name: (None, {}) for name in ("current", "fresh")}
        with mock.patch.object(
            probe, "sample_pool", return_value=(jnp.zeros((8, 2)), jnp.zeros((8, 1)))
        ), mock.patch.object(
            probe, "_base_targets", return_value=jnp.arange(8, dtype=jnp.float32)
        ), mock.patch.object(
            probe, "_fit", return_value=(jnp.zeros(2), jnp.float32(0), jnp.float32(0))
        ) as fits, mock.patch.object(
            probe, "probe_round", wraps=probe.probe_round
        ) as rounds:
            for k in (1, 2):
                run_probe_networks(None, None, None, critics, None, None, 990, k, cfg)
        result = self.run_script(
            FAKE_ROUNDS=str(rounds.call_count), FAKE_FITS=str(fits.call_count)
        )
        self.assertEqual(
            result.returncode, 0, result.stderr + (self.out / "profile.log").read_text()
        )

    def test_identity_modes_keep_the_existing_gate_and_cache_scenarios(self):
        for mode in ("identity_cold", "identity_warm"):
            with self.subTest(mode=mode):
                out = self.root / mode
                result = self.run_script(OUT=str(out), DIAGNOSTIC_MODE=mode)
                self.assertEqual(
                    result.returncode,
                    0,
                    result.stderr + (out / "profile.log").read_text(),
                )
                calls = [
                    json.loads(line)
                    for line in (out / "identity-commands.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(calls), 2)
                for call in calls:
                    args = call["args"]
                    for value in (
                        "num_env_steps=240000",
                        "fork.identity_snapshot_steps=1000",
                        "testing.stop_after_identity_snapshot=true",
                        "testing.force_trigger_check=2",
                        "seed=101",
                        "critic_num_blocks=4",
                        "critic_hidden_dim=1536",
                    ):
                        self.assertIn(value, args)
                    self.assertFalse(any(value.startswith("probe.") for value in args))
                self.assertIn("fork.arm=identity", calls[1]["args"])
                self.assertEqual(
                    calls[0]["cache"] == calls[1]["cache"], mode == "identity_warm"
                )
                self.assertTrue((out / "validation.json").exists())

    def test_only_approved_profile_architectures_are_accepted(self):
        for arch in ("D2W512", "D4W1024"):
            with self.subTest(arch=arch):
                out = self.root / arch
                self.assertEqual(
                    self.run_script(OUT=str(out), DIAGNOSTIC_ARCH=arch).returncode, 0
                )
        self.assertNotEqual(self.run_script(DIAGNOSTIC_ARCH="D6W1536").returncode, 0)
        self.assertNotEqual(
            self.run_script(
                DIAGNOSTIC_ARCH="D2W512", DIAGNOSTIC_MODE="identity_cold"
            ).returncode,
            0,
        )
        self.assertFalse(self.out.exists())

    def test_failure_keeps_status_and_never_retries(self):
        result = self.run_script(FAKE_MODE="profile-error")
        self.assertEqual(result.returncode, 9)
        self.assertEqual((self.out / "exit_status.txt").read_text(), "9\n")
        self.assertFalse((self.out / "validation.json").exists())
        self.assertEqual((self.out / "profile-calls.txt").read_text(), "call\n")

    def test_timeout_keeps_status_without_profile(self):
        self.assertEqual(self.run_script(FAKE_MODE="timeout").returncode, 124)
        self.assertEqual((self.out / "exit_status.txt").read_text(), "124\n")
        self.assertFalse((self.out / "profile-args.json").exists())

    def test_invalid_outputs_fail_validation(self):
        for mode in ("missing-output", "nonfinite", "cache-error", "wrong-count"):
            with self.subTest(mode=mode):
                out = self.root / mode
                result = self.run_script(OUT=str(out), FAKE_MODE=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotEqual((out / "exit_status.txt").read_text(), "0\n")
                self.assertFalse((out / "validation.json").exists())


if __name__ == "__main__":
    unittest.main()
