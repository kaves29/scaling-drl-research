"""Unapplied candidate patches; functions operate on copied source strings only."""

from copy import deepcopy
import difflib
from pathlib import Path
import subprocess


def replicate_configuration(config):
    """Candidate for the verified EchoCritic coupled-seed schema, not arbitrary studies."""
    result = deepcopy(config)
    seed = result.pop("seed", None)
    for section in ("agent", "env"):
        values = result.get(section)
        if isinstance(values, dict) and "seed" in values:
            if values["seed"] != seed:
                raise ValueError("unsupported non-coupled seed configuration")
            values.pop("seed")
    return result


def centered_residual_source(source):
    changes = (
        ("    targets = offset + base_targets\n", ""),
        (
            "(q - targets[batch_idx]) ** 2",
            "((q - offset) - base_targets[batch_idx]) ** 2",
        ),
        (
            "(predict(fitted) - targets) ** 2",
            "((predict(fitted) - offset) - base_targets) ** 2",
        ),
    )
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("proposal source anchor does not match")
        source = source.replace(old, new)
    return source


def seed_grouping_source(source):
    old = 'cfg = {k: v for k, v in r["config"].items() if k != "seed"}'
    if source.count(old) != 1:
        raise ValueError("proposal source anchor does not match")
    # Embed the reviewed candidate, never monkeypatch the production module.
    import inspect

    helper = inspect.getsource(replicate_configuration)
    source = source.replace(
        "import argparse\n", "import argparse\nfrom copy import deepcopy\n"
    )
    source = source.replace(
        "def json_safe(value):", helper + "\n\ndef json_safe(value):"
    )
    return source.replace(old, 'cfg = replicate_configuration(r["config"])')


def write_patches(repository, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    definitions = (
        (
            "f597ee9898e4d401912092c336c3e5330cf2fda1",
            "main/experiments/exp12/probe.py",
            centered_residual_source,
            "extreme_offset_centered_residual.patch",
        ),
        (
            "f8ad82f064547c76e12539804c98be75b32656d8",
            "main/analysis/campaign/pipeline.py",
            seed_grouping_source,
            "coupled_seed_grouping.patch",
        ),
    )
    for revision, path, transform, name in definitions:
        original = subprocess.check_output(
            ["git", "-C", str(repository), "show", revision + ":" + path], text=True
        )
        candidate = transform(original)
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(True),
                candidate.splitlines(True),
                fromfile="a/" + path,
                tofile="b/" + path,
            )
        )
        (output / name).write_text(diff)


def contemporary_log_source(source):
    old = '"probe/check_index": int(k), "probe/loss_iqm": float(row["loss_iqm"]),'
    if source.count(old) != 1:
        raise ValueError("proposal source anchor does not match")
    return source.replace(
        old,
        old
        + '\n                       "probe/score_current_iqm": float(row["score_current_iqm"]),\n                       "probe/score_fresh_iqm": float(row["score_fresh_iqm"]),',
    )


def effective_n_source(source):
    old = "   m,lo,hi=ci(d[metric+'_'+label]);row[label]=m;row[label+'_ci_low']=lo;row[label+'_ci_high']=hi"
    if source.count(old) != 1:
        raise ValueError("proposal source anchor does not match")
    return source.replace(
        old,
        old
        + "\n   row[label+'_n_seeds']=int(np.isfinite(np.asarray(d[metric+'_'+label],float)).sum())",
    )
