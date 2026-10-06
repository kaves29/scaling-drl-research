"""Break-and-restore evidence: each mutation must make its invariant test fail.

Not collected by unittest discovery (no test_ prefix). Run:
    python tests/exp12_break_checks.py
Each line reports whether the named test FAILED under the mutation (expected)
and PASSED again without it.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from experiments.exp12 import ledger, probe, trigger  # noqa: E402

real_probe_round = probe.probe_round
real_fit = probe._fit
real_summarize = probe.summarize


def unpaired_probe_round(target_def, critics, tx, obs, act, key, cfg, shared_offset=None):
    out = {}
    for i, (name, critic) in enumerate(critics.items()):
        out.update(real_probe_round(target_def, {name: critic}, tx, obs, act, jax.random.fold_in(key, i), cfg,
                                    shared_offset))
    return out


def zero_offset_fit(network_def, tx, params, obs, act, base, idx, chunk, offset=None):
    losses, final, offset = real_fit(network_def, tx, params, obs, act, base - 0.0, idx, chunk)
    # Re-run with a = 0 by shifting the targets back by the critic's own offset.
    losses0, final0, _ = real_fit(network_def, tx, params, obs, act, base - offset, idx, chunk)
    return losses0, final0, jnp.zeros_like(offset)


def flipped_summarize(result, current="current", fresh="fresh"):
    return real_summarize(result, current=fresh, fresh=current)


def no_baseline_probe_round(*args, **kwargs):
    out = real_probe_round(*args, **kwargs)
    for v in out.values():
        v["score"] = -v["final_loss"]
    return out


def always_valid_summarize(result, current="current", fresh="fresh"):
    s = real_summarize(result, current, fresh)
    s["valid"] = True
    return s


real_load = ledger.load


def flipped_triggered(ci_low, null_threshold):
    return bool(np.isfinite(ci_low) and ci_low < -null_threshold)


def non_strict_triggered(ci_low, null_threshold):
    return bool(np.isfinite(ci_low) and ci_low >= null_threshold)


def mean_bootstrap_interval(loss_rounds, seed, check_index, reps, confidence):
    x = np.asarray(loss_rounds, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return float("nan"), float("nan")
    rng = np.random.default_rng([seed, trigger.BOOT_STREAM, check_index])
    stats = x[rng.integers(0, x.size, size=(reps, x.size))].mean(1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def keep_dev_load(results_root=None, include_dev=False, require_complete=True):
    return real_load(results_root, include_dev=True, require_complete=require_complete)


def source_mutation(module, attr, old, new=None):
    """attr rebuilt from module's source with textual changes (for logic written inline).
    `old` is one anchor (with `new`) or a list of (anchor, replacement) pairs; each anchor must occur once."""
    import inspect

    source = inspect.getsource(module)
    for a, b in ([(old, new)] if new is not None else old):
        if source.count(a) != 1:
            raise ValueError(f"mutation anchor must occur once in {module.__name__}: {a!r}")
        source = source.replace(a, b)
    namespace = {"__name__": module.__name__ + "_mutated"}
    exec(compile(source, module.__file__, "exec"), namespace)
    return namespace[attr]


from experiments.exp12 import trainer as trainer_module  # noqa: E402

PHASE3 = "tests.test_exp12_phase3"
MUTATIONS = [
    ("pairing: current/fresh get different targets+minibatches", probe, "probe_round", unpaired_probe_round,
     "tests.test_exp12_probe.ProbePairingTest.test_identical_inputs_targets_and_minibatches"),
    ("offset: a forced to 0 instead of each critic's own mean", probe, "_fit", zero_offset_fit,
     "tests.test_exp12_probe.ProbePairingTest.test_per_critic_offset_is_each_critics_own_mean"),
    ("offset: a forced to 0 (known-answer test)", probe, "_fit", zero_offset_fit,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_frozen_constant_critic_has_known_score"),
    ("baseline b dropped from P", probe, "probe_round", no_baseline_probe_round,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_frozen_constant_critic_has_known_score"),
    ("sign: L = P(current) - P(fresh)", probe, "summarize", flipped_summarize,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_sign_convention_plasticity_loss_positive_when_current_is_worse"),
    ("non-finite check treated as valid", probe, "summarize", always_valid_summarize,
     "tests.test_exp12_probe.ProbeArithmeticTest.test_non_finite_round_marks_check_invalid"),
    ("trigger fires on the gain side (upper tail)", trigger, "triggered", flipped_triggered,
     f"{PHASE3}.TriggerTest.test_current_worse_triggers_and_reverse_never_does"),
    ("trigger fires when the lower bound equals 0", trigger, "triggered", non_strict_triggered,
     f"{PHASE3}.TriggerTest.test_edge_cases"),
    ("f*_run allowed at 20/20 (past 95% of budget)", trigger, "last_eligible_check", lambda checks, eligible_fraction: checks,
     f"{PHASE3}.TriggerTest.test_f_star_first_eligible_check_only"),
    ("bootstrap statistic is the mean, not the IQM", trigger, "bootstrap_interval", mean_bootstrap_interval,
     f"{PHASE3}.TriggerTest.test_statistic_is_iqm_not_mean"),
    ("dev runs not excluded by default", ledger, "load", keep_dev_load,
     f"{PHASE3}.LedgerTest.test_round_trip_and_dev_excluded_by_default"),
    ("A9: random action only on step 1 (angle_1 rule) instead of until min_length", trainer_module,
     "Exp12Trainer",
     source_mutation(trainer_module, "Exp12Trainer", "if not self.buffer.can_sample():", "if self.timestep is None:"),
     "tests.test_exp12_foundations.SimbaRandomWarmupTest.test_random_until_min_length_then_policy"),
]


from experiments.exp12 import fork, injection, m_selection  # noqa: E402

real_inject = injection.inject


def fresh_trunk_state(old_state, trunk_params, num_blocks, m, learning_rate, weight_decay):
    import optax

    return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay).init(trunk_params)


def shared_count_inject(*args, **kwargs):
    critic, target = real_inject(*args, **kwargs)
    new = critic.opt_state["new"]
    new = (new[0]._replace(count=critic.opt_state["trunk"][0].count), *new[1:])
    return critic.replace(opt_state={**critic.opt_state, "new": new}), target


def argmax_select_m(recoveries, similar_within):
    return max(m_selection.M_LABELS, key=lambda m: recoveries[m])


INJ = "tests.test_exp12_injection.InjectionInvariantsTest"
PC = "tests.test_exp12_positive_control"
FORK = "tests.test_exp12_fork.ForkUnitTest"
MUTATIONS += [
    ("injection: Q = old + new (subtraction of the frozen copy removed)", injection, "InjectedSACCritic",
     source_mutation(injection, "InjectedSACCritic", "return self.old(z) + (self.new(z) - self.copy(z))",
                     "return self.old(z) + self.new(z)"),
     f"{INJ}.test_predictions_unchanged_bit_for_bit_and_action_gradients_within_dtype_tolerance"),
    ("injection: frozen heads' parameter gradients not stopped", injection, "InjectedSACCritic",
     source_mutation(injection, "InjectedSACCritic", [
         ('params["old"] = jax.lax.stop_gradient(params["old"])', "pass"),
         ('params["copy"] = jax.lax.stop_gradient(params["copy"])', "pass")]),
     f"{INJ}.test_gradients_reach_earlier_blocks_through_the_frozen_head"),
    ("injection: plain AdamW on every parameter (frozen heads decay)", injection, "inject",
     source_mutation(injection, "inject", [
         ("tx = injected_optimizer(learning_rate, weight_decay)",
          "tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)"),
         ("    opt_state = {\n", "    opt_state = tx.init(params)\n    _unused = {\n")]),
     f"{INJ}.test_training_keeps_frozen_heads_bit_identical_and_moves_trainable_parts"),
    ("injection: trunk optimizer state recreated (moments and count reset)", injection, "_carry_trunk_state",
     fresh_trunk_state, f"{INJ}.test_optimizer_state_rules"),
    ("injection: new head shares the trunk's step count", injection, "inject", shared_count_inject,
     f"{INJ}.test_optimizer_state_rules"),
    ("injection: copy head initialised independently of the new head", injection, "inject",
     source_mutation(injection, "inject",
                     "copy_head = jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), new_head)",
                     "copy_head = _Head(m, hidden, original.dtype).init(jax.random.fold_in(key, 1), "
                     "jnp.zeros((1, hidden), original.dtype))[\"params\"]"),
     f"{INJ}.test_new_copies_identical_and_freshly_initialised"),
    ("injection: target critic's new head differs from the online one", injection, "inject",
     source_mutation(injection, "inject", '"new": jax.tree_util.tree_map(jnp.array, new_head),',
                     '"new": jax.tree_util.tree_map(lambda x: x + 1e-3, new_head),'),
     f"{INJ}.test_new_copies_identical_and_freshly_initialised"),
    ("m rule: best recovery instead of the smallest m within 0.10", m_selection, "select_m", argmax_select_m,
     f"{PC}.MSelectionArithmeticTest.test_select_smallest_m_within_tolerance"),
    ("probe noise not expressed in recovery units", m_selection, "evaluate",
     source_mutation(m_selection, "evaluate", 'out["noise"] = noise_sd / l_trigger',
                     'out["noise"] = noise_sd'),
     f"{PC}.MSelectionArithmeticTest.test_evaluate_stops_on_noise"),
    ("recovery not relative to the fresh critic's level (L = 0)", m_selection, "recovery",
     lambda l_trigger, l_injected: (l_trigger - l_injected) / (l_trigger - 0.1),
     f"{PC}.MSelectionArithmeticTest.test_recovery_toward_the_fresh_critic"),
    ("probe noise: range instead of pooled SD", m_selection, "pooled_sd",
     lambda series: max(float(np.ptp(x)) for x in series.values()),
     f"{PC}.MSelectionArithmeticTest.test_pooled_sd"),
    ("probe noise stops only above 0.10, not at it", m_selection, "evaluate",
     source_mutation(m_selection, "evaluate", 'if out["noise"] >= noise_threshold:',
                     'if out["noise"] > noise_threshold + 1e-12:'),
     f"{PC}.MSelectionArithmeticTest.test_noise_exactly_at_the_threshold_stops"),
    ("shared offset ignored (each critic keeps its own)", probe, "probe_round",
     source_mutation(probe, "probe_round", "    if shared_offset is not None:\n", "    if False:\n"),
     f"{PC}.SharedOffsetTest.test_shared_offset_modes"),
    ("Check 1 ignores dQ/da", fork, "check1",
     source_mutation(fork, "check1", '"pass": dq <= tol * eps * scale_q and dg <= tol * eps * scale_g,',
                     '"pass": dq <= tol * eps * scale_q,'),
     f"{FORK}.test_check1_compares_values_and_action_gradients"),
    ("Check 1: the control and identity arm get the 64 eps tolerance too", fork, "check1",
     source_mutation(fork, "check1", 'tol = tolerance_eps if injected and "after" in (a, b) else 0.0',
                     "tol = tolerance_eps"),
     f"{FORK}.test_check1_control_and_identity_must_be_bit_exact"),
    ("Check 1 tolerance effectively infinite (a broken injection passes)", fork, "check1",
     source_mutation(fork, "check1", 'tol = tolerance_eps if injected and "after" in (a, b) else 0.0',
                     'tol = 1e12 if injected and "after" in (a, b) else 0.0'),
     f"{FORK}.test_check1_fails_when_the_injection_construction_is_broken"),
    ("arms may run on a different device model", fork, "check_same_device", lambda run_dir: None,
     f"{FORK}.test_both_arms_must_run_on_the_fork_device_model"),
    ("f*_run on one firing check although 2 consecutive are required", trigger, "f_star",
     source_mutation(trigger, "f_star", "if run >= consecutive_checks:", "if run >= 1:"),
     f"{PHASE3}.TriggerTest.test_f_star_first_eligible_check_only"),
    ("fork plan: injected arm runs to N instead of fork + 25% N", fork, "fork_plan",
     source_mutation(fork, "fork_plan", "arm_end = fork_step + horizon", "arm_end = max(n, fork_step)"),
     f"{FORK}.test_fork_plan"),
    ("post-fork eval does not restore the agent's key", fork, "post_fork_eval",
     source_mutation(fork, "post_fork_eval", "        core._rng = saved_key\n", ""),
     f"{FORK}.test_post_fork_eval_leaves_training_state_untouched"),
    ("post-fork eval does not restore the global numpy RNG", fork, "post_fork_eval",
     source_mutation(fork, "post_fork_eval", "        np.random.set_state(np_state)\n", ""),
     f"{FORK}.test_post_fork_eval_leaves_training_state_untouched"),
    ("post-fork eval leaves the global numpy RNG unseeded (Reach goals follow training)", fork,
     "post_fork_eval", source_mutation(fork, "post_fork_eval", "        np.random.seed(env_seed)\n", ""),
     f"{FORK}.test_post_fork_eval_leaves_training_state_untouched"),
    ("post-fork eval does not restore Python's random", fork, "post_fork_eval",
     source_mutation(fork, "post_fork_eval", "        random.setstate(py_state)\n", ""),
     f"{FORK}.test_post_fork_eval_leaves_training_state_untouched"),
]


from analysis import exp2_analysis  # noqa: E402
from experiments.exp12 import diagnostics  # noqa: E402
from scale_rl.agents.sac import sac_agent, sac_update  # noqa: E402

DIAG = "tests.test_exp12_diagnostics"
EA = "tests.test_exp12_exp2_analysis.SyntheticResultsTest"
KL_LINE = "jnp.mean(new_dist.kl_divergence(old_dist))"
MUTATIONS += [
    # The scanned update is jitted: each mutation rebuilds it, so no compiled version is reused.
    ("I1: KL(pi_{t-1} || pi_t) instead of KL(pi_t || pi_{t-1})", sac_agent, "_update_sac_networks_scan",
     source_mutation(sac_agent, "_update_sac_networks_scan", KL_LINE, "jnp.mean(old_dist.kl_divergence(new_dist))"),
     f"{DIAG}.KnownAnswerTest.test_policy_kl_is_the_closed_form_gaussian_kl_new_vs_old"),
    ("I1: wrong KL formula (mean term only, no variance terms)", sac_agent, "_update_sac_networks_scan",
     source_mutation(sac_agent, "_update_sac_networks_scan", KL_LINE,
                     "jnp.mean(0.5 * jnp.sum(((new_dist.mean() - old_dist.mean()) / old_dist.stddev()) ** 2, axis=-1))"),
     f"{DIAG}.KnownAnswerTest.test_policy_kl_is_the_closed_form_gaussian_kl_new_vs_old"),
    ("(x): diagnostic forward passes left at the job's precision (TF32 on the GPU)", sac_agent,
     "_update_sac_networks_scan",
     source_mutation(sac_agent, "_update_sac_networks_scan",
                     "        return jax.default_matmul_precision(DIAGNOSTICS_PRECISION)",
                     "        return contextlib.nullcontext()"),
     f"{DIAG}.DiagnosticsPrecisionTest.test_diagnostic_forward_passes_are_highest_and_training_matmuls_keep_the_run_setting"),
    ("diagnostics perturb the actor update", sac_agent, "_update_sac_networks_scan",
     source_mutation(sac_agent, "_update_sac_networks_scan", "    if kl_ref_observations is not None:\n",
                     "    if kl_ref_observations is not None:\n        new_actor = new_actor.replace(params="
                     "jax.tree_util.tree_map(lambda p: p * 1.0001, new_actor.params))\n"),
     f"{DIAG}.KnownAnswerTest.test_diagnostics_never_change_the_update"),
    ("I3: saturation of the deterministic mean action, not the sampled one", sac_update, "update_actor",
     source_mutation(sac_update, "update_actor", "jnp.mean(jnp.abs(actions) > saturation_threshold)",
                     "jnp.mean(jnp.abs(jnp.tanh(dist.distribution.mean())) > saturation_threshold)"),
     f"{DIAG}.KnownAnswerTest.test_saturation_is_the_fraction_of_sampled_components_beyond_the_threshold"),
    ("I2: sample SD (ddof=1) instead of the population SD", diagnostics.ActorDiagnostics, "window_metrics",
     source_mutation(diagnostics, "ActorDiagnostics", "float(np.std(values))",
                     "float(np.std(values, ddof=1))").window_metrics,
     f"{DIAG}.KnownAnswerTest.test_gnorm_std_is_the_population_sd_of_the_window"),
    ("I2: the gnorm window is never reset", diagnostics.ActorDiagnostics, "window_metrics",
     source_mutation(diagnostics, "ActorDiagnostics", "values, self.gnorm = self.gnorm, []",
                     "values = self.gnorm").window_metrics,
     f"{DIAG}.KnownAnswerTest.test_gnorm_std_is_the_population_sd_of_the_window"),
    ("I1: KL reference drawn from the global numpy RNG", diagnostics.ActorDiagnostics, "update_kwargs",
     source_mutation(diagnostics, "ActorDiagnostics",
                     "idx = rng.integers(0, trainer.buffer._num_in_buffer, size=self.reference_size)",
                     "idx = np.random.randint(0, trainer.buffer._num_in_buffer, size=self.reference_size)"
                     ).update_kwargs,
     f"{DIAG}.ReferenceBatchTest.test_reference_batch_per_window_from_a_dedicated_stream"),
    ("I1: KL reference never redrawn after the first window", diagnostics.ActorDiagnostics, "update_kwargs",
     source_mutation(diagnostics, "ActorDiagnostics", "if window != self.window:",
                     "if self.window is None:").update_kwargs,
     f"{DIAG}.ReferenceBatchTest.test_reference_batch_per_window_from_a_dedicated_stream"),
    ("Exp 2: paired difference control - injected", exp2_analysis, "paired_returns",
     source_mutation(exp2_analysis, "paired_returns", 'per_eval["injected"] - per_eval["control"]',
                     'per_eval["control"] - per_eval["injected"]'),
     f"{EA}.test_paired_difference_is_injected_minus_control_mean_return"),
    ("Exp 2: forks with an unfinished arm enter the paired graphs", exp2_analysis, "paired_returns",
     source_mutation(exp2_analysis, "paired_returns", "complete = forks[forks.complete].run_key",
                     "complete = forks.run_key"),
     f"{EA}.test_band_per_environment_over_complete_confirmatory_forks"),
    ("Exp 2: development runs not excluded by default", exp2_analysis, "load",
     source_mutation(exp2_analysis, "load", "runs, _ = ledger.load(results_root, include_dev=include_dev,",
                     "runs, _ = ledger.load(results_root, include_dev=True,"),
     f"{EA}.test_tables_and_files"),
    ("Exp 2: the secondary analysis uses every fork", exp2_analysis, "run_analysis",
     source_mutation(exp2_analysis, "run_analysis", "success = forks[forks.check2_pass == True].run_key",
                     "success = forks.run_key"),
     f"{EA}.test_secondary_is_the_check2_success_subset"),
]


import generate_manifest  # noqa: E402

MAN = "tests.test_exp12_manifest.Exp12GridTest"
MUTATIONS += [
    ("manifest: one fixed save interval instead of one save per probe check", generate_manifest, "add_exp12_grid",
     source_mutation(generate_manifest, "add_exp12_grid",
                     "interval = EXP12_BUDGETS[env_group] // EXP12_ACTION_REPEAT // EXP12_CHECKS",
                     "interval = 12501"),
     f"{MAN}.test_every_job_composes_with_the_methodology_budget"),
    ("manifest: arm jobs for the default critic too", generate_manifest, "add_exp12_grid",
     source_mutation(generate_manifest, "add_exp12_grid", "if arch[0] not in EXP12_FORKING or not",
                     "if not"),
     f"{MAN}.test_arm_jobs_only_for_completed_forks_grouped_by_device"),
    ("manifest: arm jobs not grouped by the fork's device model", generate_manifest, "add_exp12_grid",
     source_mutation(generate_manifest, "add_exp12_grid", 'f"exp2_arms_{_device_slug(device)}.txt"',
                     '"exp2_arms.txt"'),
     f"{MAN}.test_arm_jobs_only_for_completed_forks_grouped_by_device"),
    ("manifest: DONE runs queued again", generate_manifest, "add_exp12_grid",
     source_mutation(generate_manifest, "add_exp12_grid", '                if state != "done":\n',
                     "                if True:\n"),
     f"{MAN}.test_done_and_resume_classification"),
    ("manifest: relative --ckpt-root accepted", generate_manifest, "main",
     source_mutation(generate_manifest, "main", 'ckpt_root = require_absolute(args.ckpt_root or "", "--ckpt-root")',
                     "ckpt_root = os.path.abspath(args.ckpt_root)"),
     f"{MAN}.test_cli_writes_the_manifests"),
    ("manifest: arm config differs from its parent's (results_root dropped)", generate_manifest, "add_exp12_grid",
     source_mutation(generate_manifest, "add_exp12_grid", "arm_overrides = overrides + [",
                     "arm_overrides = overrides[:-1] + ["),
     f"{MAN}.test_arm_jobs_only_for_completed_forks_grouped_by_device"),
]


from experiments.exp12 import envs as exp12_envs  # noqa: E402

HOR = "tests.test_exp12_foundations.DiscountAndHorizonTest"
MUTATIONS += [
    ("truncation treated as terminal in the critic target", sac_update, "update_critic",
     source_mutation(sac_update, "update_critic",
                     'target_q = batch["reward"] + (gamma**n_step) * (1 - batch["terminated"]) * next_q',
                     'target_q = batch["reward"] + (gamma**n_step) * (1 - jnp.maximum(batch["terminated"], '
                     'batch["truncated"])) * next_q'),
     f"{HOR}.test_truncation_bootstraps_and_termination_does_not"),
    ("the reset observation stored as next_observation on truncation", trainer_module, "Exp12Trainer",
     source_mutation(trainer_module, "Exp12Trainer", "if terminateds[env_idx] or truncateds[env_idx]:",
                     "if terminateds[env_idx]:"),
     f"{HOR}.test_truncation_bootstraps_and_termination_does_not"),
    ("PenTwirl's registered 50-step limit replaced", exp12_envs, "make_myosuite_env",
     source_mutation(exp12_envs, "make_myosuite_env", "myo_gym.make(MYOSUITE_TASKS_DICT[env_name], seed=seed)",
                     "myo_gym.make(MYOSUITE_TASKS_DICT[env_name], seed=seed, max_episode_steps=1000)"),
     f"{HOR}.test_pen_twirl_keeps_its_registered_50_step_limit_as_in_simba"),
]


from scale_rl.agents.sac import sac_network  # noqa: E402

TWIN = "tests.test_exp12_twin_critic"
TD_LINE = '"train/td_error_var": (td_vars[0] + td_vars[1]) / 2 if critic_use_cdq else jnp.var(td_error),'
MUTATIONS += [
    # The twin-critic tests call sac_network / sac_update directly (not through a jitted scan).
    ("twin critic: vmap with in_axes=None (the pre-fix form)", sac_network, "SACClippedDoubleCritic",
     source_mutation(sac_network, "SACClippedDoubleCritic", [("in_axes=0,", "in_axes=None,"),
                                                            ("(tile(observations), tile(actions))", "(observations, actions)")]),
     f"{TWIN}.TwinCriticDefectTest.test_twin_critic_initialises_with_two_independent_networks"),
    ("twin critic: td_error never set in the clipped-double-Q branch (the pre-fix form)", sac_update, "update_critic",
     source_mutation(sac_update, "update_critic", [(TD_LINE, '"train/td_error_var": jnp.var(td_error),'),
                                                   ("        if critic_use_cdq:\n            critic_info[", "        if False:\n            critic_info[")]),
     f"{TWIN}.TwinCriticDefectTest.test_td_error_var_is_the_mean_of_the_two_networks"),
    ("twin critic: td_error_var pooled over both networks instead of their mean", sac_update, "update_critic",
     source_mutation(sac_update, "update_critic", "(td_vars[0] + td_vars[1]) / 2",
                     "jnp.var(jnp.concatenate([pred_q1 - target_q, pred_q2 - target_q]))"),
     f"{TWIN}.TwinCriticDefectTest.test_td_error_var_is_the_mean_of_the_two_networks"),
    ("twin critic: grad cosine ignores critic_use_cdq (the pre-fix form)", sac_update, "compute_actor_gradient_cosine",
     source_mutation(sac_update, "compute_actor_gradient_cosine",
                     "        if critic_use_cdq:  # the actor loss's min(Q1, Q2), as in update_actor\n"
                     "            q_val = jnp.minimum(q_val[0], q_val[1])\n", ""),
     f"{TWIN}.TwinCriticDefectTest.test_actor_grad_cosine_uses_min_of_the_two_networks"),
    ("twin critic: grad cosine on Q1 only", sac_update, "compute_actor_gradient_cosine",
     source_mutation(sac_update, "compute_actor_gradient_cosine", "q_val = jnp.minimum(q_val[0], q_val[1])",
                     "q_val = q_val[0]"),
     f"{TWIN}.TwinCriticDefectTest.test_actor_grad_cosine_uses_min_of_the_two_networks"),
    ("twin critic: Q2's structural metrics read from network 0", sac_update, "get_critic_with_metrics",
     source_mutation(sac_update, "get_critic_with_metrics", "jnp.array(activi)[:,1,:,:]", "jnp.array(activi)[:,0,:,:]"),
     f"{TWIN}.TwinCriticPathTest.test_structural_metrics_are_per_network"),
    ("twin critic: target computed from Q1's target only", sac_update, "update_critic",
     source_mutation(sac_update, "update_critic", "next_q = jnp.minimum(next_q1, next_q2).reshape(-1)",
                     "next_q = next_q1.reshape(-1)"),
     f"{TWIN}.TwinCriticPathTest.test_critic_loss_uses_the_shared_min_target_and_each_network_its_own_term"),
]

from experiments import exp1 as exp1_module  # noqa: E402
from experiments.exp12 import twin as twin_module  # noqa: E402


def metadata_without_critic_count(resolved_cfg, identity, launch):
    return exp1_module.__dict__["_real_build_run_metadata"](resolved_cfg, identity,
                                                           {k: v for k, v in launch.items() if k != "critic_count"})


exp1_module._real_build_run_metadata = exp1_module.build_run_metadata

MUTATIONS += [
    ("twin probe: per-round L from network 1 only", twin_module, "combine",
     source_mutation(twin_module, "combine", 'np.mean([p["score"] for p in parts], axis=0)', 'parts[0]["score"]'),
     f"{TWIN}.TwinProbeTest.test_combined_loss_is_the_mean_of_the_two_networks"),
    ("twin probe: both views take network 1's parameters", twin_module, "expand",
     source_mutation(twin_module, "expand", "view = (single, network_params(critic[1], k))",
                     "view = (single, network_params(critic[1], 0))"),
     f"{TWIN}.TwinProbeTest.test_expand_gives_each_network_its_own_slice"),
    ("twin injection: target trunk and frozen head taken from the online critic", injection, "inject_twin",
     source_mutation(injection, "inject_twin", "split_params(target_critic.params[TWIN], num_blocks, m)",
                     "split_params(critic.params[TWIN], num_blocks, m)"),
     f"{TWIN}.TwinInjectionTest.test_same_head_in_both_networks_and_targets"),
    ("twin injection: one new head shared by both networks", injection, "inject_twin",
     source_mutation(injection, "inject_twin", "for k in jax.random.split(key, n)]", "for k in [key] * n]"),
     f"{TWIN}.TwinInjectionTest.test_same_head_in_both_networks_and_targets"),
    ("twin injection: trunk Adam state reset instead of carried", injection, "inject_twin",
     source_mutation(injection, "inject_twin",
                     '"trunk": _carry_trunk_state(per_network, trunk, num_blocks, m, learning_rate, weight_decay),',
                     '"trunk": optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay).init(trunk),'),
     f"{TWIN}.TwinInjectionTest.test_training_keeps_frozen_heads_and_optimizer_rules"),
    ("twin Check 1: dQ/da of Q1 instead of min(Q1, Q2)", fork, "panel_q_and_grad",
     source_mutation(fork, "panel_q_and_grad", "jnp.minimum(*q_fn(b)).sum()", "q_fn(b)[0].sum()"),
     f"{TWIN}.TwinCheck1Test.test_panel_values"),
    ("twin metadata: critic count not recorded", exp1_module, "build_run_metadata", metadata_without_critic_count,
     f"{TWIN}.TwinForkEndToEndTest.test_runs_are_twin_and_record_the_critic_count"),
    ("Check 1: original-to-restored control comparison skipped", fork, "validate_control_restore",
     lambda *args, **kwargs: None,
     "tests.test_exp12_fork.ForkEndToEndTest.test_control_stops_before_ready_if_restore_changes_panel"),
]

def _run(test_id):
    suite = unittest.defaultTestLoader.loadTestsFromName(test_id)
    result = unittest.TextTestRunner(stream=open("/dev/null", "w"), verbosity=0).run(suite)
    return result.wasSuccessful(), "error" if result.errors else "assertion"


def main():
    ok = True
    import importlib

    for label, module, attr, replacement, test_id in MUTATIONS:
        test_module = importlib.import_module(test_id.rsplit(".", 2)[0])
        patches = [mock.patch.object(module, attr, replacement)]
        if getattr(test_module, attr, None) is getattr(module, attr):  # names a test module imported directly
            patches.append(mock.patch.object(test_module, attr, replacement))
        for p in patches:
            p.start()
        try:
            broken_passes, how = _run(test_id)
        finally:
            for p in patches:
                p.stop()
        restored_passes, _ = _run(test_id)
        status = "OK" if (not broken_passes and restored_passes) else "PROBLEM"
        ok &= status == "OK"
        print(f"[{status}] {label}: mutated -> {'PASS' if broken_passes else f'FAIL ({how})'}, "
              f"restored -> {'PASS' if restored_passes else 'FAIL'}  ({test_id.split('.')[-1]})")
    return 0 if ok else 1


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
