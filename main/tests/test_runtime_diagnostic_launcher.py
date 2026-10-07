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
if sys.argv[1:] == ['-']:
    source = sys.stdin.read()
    if 'import jax' in source:
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
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse(self.out.exists())

    def test_wrong_revision_refuses_before_output(self):
        self.assertNotEqual(self.run_script(EXPECTED_COMMIT="0" * 40).returncode, 0)
        self.assertFalse(self.out.exists())

    def test_wrong_branch_refuses_before_output(self):
        self.git("switch", "-c", "other")
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse(self.out.exists())

    def test_dirty_checkout_refuses_before_output(self):
        (self.main / "untracked.txt").touch()
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse(self.out.exists())

    def test_reused_output_is_preserved(self):
        self.out.mkdir()
        sentinel = self.out / "existing.txt"
        sentinel.write_text("preserve")
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_output_inside_checkout_refuses(self):
        target = self.main / "output"
        self.assertNotEqual(self.run_script(OUT=str(target)).returncode, 0)
        self.assertFalse(target.exists())

    def test_non_a100_refuses_before_output(self):
        self.assertNotEqual(
            self.run_script(EXPECTED_GPU_MODEL="Other GPU").returncode, 0
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
        self.assertEqual(args[args.index("--archs") + 1], "D4W1536")
        self.assertEqual(args[args.index("--train_steps") + 1], "60")
        self.assertEqual(args[args.index("--warmup_steps") + 1], "1001")
        self.assertEqual(args[args.index("--probe_repeats") + 1], "1")
        self.assertNotIn("--override", args)
        validation = json.loads((self.out / "validation.json").read_text())
        self.assertFalse(validation["memory_measurement_available"])
        self.assertEqual(validation["sac_updates"], 2124)

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
