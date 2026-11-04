import json

import pytest

from experiments.exp12.official_exp1 import population
from scripts.generate_remaining_exp1_manifest import COMPLETED, select_remaining


def fixture(tmp_path):
    entries = [
        {
            **row,
            "command": [
                "python",
                "run.py",
                "--checkpoint_dir",
                str(
                    tmp_path
                    / f"{row['environment']}_{row['architecture']}_{row['seed']}"
                ),
            ],
        }
        for row in population()
    ]
    return {"entries": entries}, json.loads(COMPLETED.read_text())


def test_exact_counts_and_all_eight_partials_remain_missing(tmp_path):
    plan, completed = fixture(tmp_path)
    result = select_remaining(plan, completed, list(range(1, 11)))
    assert len(result["entries"]) == 242 and len(result["reused"]) == 88
    assert {a: r["remaining"] for a, r in result["counts"].items()} == {
        "D2W512": 78,
        "D4W1024": 78,
        "D4W1536": 86,
    }
    expected = {
        (e, "D4W1536", s)
        for e in ["dog-run", "humanoid-run", "myo-key-turn", "myo-pen-twirl"]
        for s in [7, 8]
    }
    pending = {
        (r["environment"], r["architecture"], r["seed"]) for r in result["entries"]
    }
    assert expected <= pending
    assert (
        not result["approved"] and not result["executable"] and not result["submitted"]
    )
    assert not pending & {
        (r["environment"], r["architecture"], r["seed"]) for r in result["reused"]
    }


def test_duplicate_or_foreign_completions_rejected(tmp_path):
    plan, completed = fixture(tmp_path)
    completed["completed"].append(completed["completed"][0])
    with pytest.raises(ValueError, match="duplicate"):
        select_remaining(plan, completed, list(range(1, 11)))
    completed["completed"].pop()
    completed["completed"][0]["source"] = "b" * 40
    with pytest.raises(ValueError, match="foreign"):
        select_remaining(plan, completed, list(range(1, 11)))


def test_partial_output_never_resumed_or_overwritten(tmp_path):
    plan, completed = fixture(tmp_path)
    directory = tmp_path / "dog-run_D4W1536_7"
    directory.mkdir()
    (directory / "partial.log").write_text("preserve")
    with pytest.raises(ValueError, match="fresh root"):
        select_remaining(plan, completed, list(range(1, 11)))
    assert (directory / "partial.log").read_text() == "preserve"


def test_seed_mismatch_and_duplicate_population_fail_closed(tmp_path):
    plan, completed = fixture(tmp_path)
    with pytest.raises(ValueError, match="freeze"):
        select_remaining(plan, completed, [*range(1, 9), 10, 11])
    plan["entries"].append(plan["entries"][0])
    with pytest.raises(ValueError, match="population"):
        select_remaining(plan, completed, list(range(1, 11)))


def test_missing_completion_or_partial_in_completion_registry_is_rejected(tmp_path):
    plan, completed = fixture(tmp_path)
    completed["completed"].pop()
    with pytest.raises(ValueError, match="88 completed"):
        select_remaining(plan, completed, list(range(1, 11)))
    plan, completed = fixture(tmp_path)
    large = next(
        row for row in completed["completed"] if row["architecture"] == "D4W1536"
    )
    large["seed"] = 7
    with pytest.raises(ValueError, match="88 completed"):
        select_remaining(plan, completed, list(range(1, 11)))


def test_architecture_cli_lists_partition_without_changing_commands(
    tmp_path, monkeypatch
):
    import shlex
    import sys
    from scripts import generate_remaining_exp1_manifest as generator

    plan, _ = fixture(tmp_path)
    monkeypatch.setattr(generator, "build", lambda root: plan)
    output = tmp_path / "plan.json"
    directory = tmp_path / "architectures"
    argv = [
        "generator",
        "--root",
        str(tmp_path / "fresh"),
        "--seeds",
        *map(str, range(1, 11)),
        "--out",
        str(output),
        "--architecture-dir",
        str(directory),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    generator.main()
    result = json.loads(output.read_text())
    combined = []
    for arch, count in [("D2W512", 78), ("D4W1024", 78), ("D4W1536", 86)]:
        commands = [
            shlex.split(line)
            for line in (directory / f"{arch}.txt").read_text().splitlines()
        ]
        assert len(commands) == count
        assert commands == [
            r["command"] for r in result["entries"] if r["architecture"] == arch
        ]
        combined.extend(map(tuple, commands))
    assert len(combined) == len(set(combined)) == 242
    original = (directory / "D2W512.txt").read_bytes()
    output.unlink()
    with pytest.raises(FileExistsError):
        generator.main()
    assert not output.exists()
    assert (directory / "D2W512.txt").read_bytes() == original
