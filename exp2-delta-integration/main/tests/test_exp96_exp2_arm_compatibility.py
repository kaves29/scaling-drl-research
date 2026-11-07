"""Source and configuration identity for deferred ten-target Exp2 arms."""

import json
from pathlib import Path
from unittest import mock

from scripts import exp12_dynamic_scheduler as scheduler
from scripts import exp96_campaign
from scripts.exp12_launch_plan import config_hash as five_target_hash


def test_ten_target_arm_uses_parent_base_config_and_exact_fork(tmp_path):
    commit = "a" * 40
    campaign_root = tmp_path / "campaign"
    parent_row = next(
        row
        for row in exp96_campaign.rows(campaign_root)
        if row["architecture"] == "D4W1024"
    )
    fork = (
        tmp_path
        / "campaign"
        / "checkpoints"
        / "exp1"
        / "D4W1024"
        / parent_row["environment"]
        / f"seed_{parent_row['seed']}"
        / "fork"
    )
    fork.mkdir(parents=True)
    (fork / "fork.json").write_text(json.dumps({"fork_step": 10_000}))
    parent = {
        "id": scheduler.identity("exp1", parent_row, commit),
        "kind": "exp1",
        "row": parent_row,
    }

    arm = scheduler._arm_item(parent, commit, "last")
    row = arm["row"]
    assert row["config_hash"] == exp96_campaign._config_hash(row["overrides"])
    assert row["config_hash"] != five_target_hash(row["overrides"])
    assert "base_exp12_ten" in row["command"]
    assert row["checkpoint_dir"] != parent_row["checkpoint_dir"]
    assert row["fork_sha256"] == scheduler.sha(fork / "fork.json")
    assert row["source_id"] == parent["id"]
    assert row["source_config_hash"] == parent_row["config_hash"]
    assert row["study"] == exp96_campaign.VERSION
    assert scheduler._arm_item(parent, commit, "last") == arm
    assert scheduler._arm_item(parent, "b" * 40, "last")["id"] != arm["id"]
    (fork / "fork.json").write_text(json.dumps({"fork_step": 20_000}))
    assert scheduler._arm_item(parent, commit, "last")["id"] != arm["id"]


def test_historical_arm_keeps_five_target_hash(tmp_path):
    root = tmp_path / "campaign" / "checkpoints"
    row = next(
        row
        for row in exp96_campaign.rows(tmp_path / "campaign")
        if row["architecture"] == "D4W1536"
    )
    row = {
        **row,
        "checkpoint_dir": str(
            root / "exp1" / "D4W1536" / row["environment"] / f"seed_{row['seed']}"
        ),
    }
    row.pop("study")
    fork = Path(row["checkpoint_dir"]) / "fork"
    fork.mkdir(parents=True)
    (fork / "fork.json").write_text("{}")
    arm = scheduler._arm_item(
        {"id": "parent", "kind": "exp1", "row": row}, "a" * 40, "last"
    )
    assert arm["row"]["config_hash"] == five_target_hash(arm["row"]["overrides"])


def test_ten_target_injected_arm_never_uses_parent_completion_validator():
    item = {"kind": "exp2_injected", "row": {"study": exp96_campaign.VERSION}}
    with mock.patch.object(scheduler, "validate_arm", return_value="arm") as arm:
        with mock.patch.object(
            exp96_campaign, "validate_parent", side_effect=AssertionError
        ):
            assert scheduler.validate_done(item, "a" * 40, "/tmp/queue") == "arm"
    arm.assert_called_once()


def test_interrupted_arm_before_first_checkpoint_revalidates_fork(tmp_path):
    row = {
        "checkpoint_dir": str(tmp_path / "arm"),
        "source_dir": str(tmp_path / "parent"),
        "source_id": "parent-id",
        "source_config_hash": "hash",
        "study": exp96_campaign.VERSION,
        "architecture": "D4W1024",
        "environment": "dog-run",
        "seed": 1,
        "group": "dmc_hard",
        "budget": 1_000_000,
        "run_key": "exp1_D4W1024_dog-run_seed1",
        "overrides": [
            "env_name=dog-run",
            "fork.source=/tmp/parent",
            "fork.arm=injected",
            "injection.m=last",
        ],
    }
    (tmp_path / "arm").mkdir()
    item = {"kind": "exp2_injected", "id": "arm-id", "row": row}
    with mock.patch.object(
        scheduler, "_metadata_matches", return_value={}
    ), mock.patch.object(
        scheduler, "_natural_fork", return_value={"fork_step": 10_000}
    ) as check:
        assert scheduler.validate_resume(item, "a" * 40) is False
    parent, commit = check.call_args.args
    assert parent["id"] == "parent-id" and parent["row"]["config_hash"] == "hash"
    assert parent["row"]["study"] == exp96_campaign.VERSION
    assert commit == "a" * 40
    import shutil

    shutil.rmtree(tmp_path / "arm")
    with mock.patch.object(
        scheduler, "_natural_fork", return_value={"fork_step": 10_000}
    ) as check:
        assert scheduler.validate_resume(item, "a" * 40) is False
    assert check.call_args.args[0]["id"] == "parent-id"
    (tmp_path / "arm").mkdir()
    with mock.patch.object(
        scheduler, "_metadata_matches", return_value={}
    ), mock.patch.object(
        scheduler, "_natural_fork", side_effect=ValueError("invalid fork")
    ):
        import pytest

        with pytest.raises(ValueError, match="invalid fork"):
            scheduler.validate_resume(item, "a" * 40)


def test_reviewed_retry_of_arm_without_own_checkpoint_requires_valid_fork(tmp_path):
    item = {
        "kind": "exp2_injected",
        "id": "arm-id",
        "row": {
            "checkpoint_dir": str(tmp_path / "arm"),
            "source_dir": str(tmp_path / "parent"),
            "source_id": "parent-id",
            "source_config_hash": "hash",
            "study": exp96_campaign.VERSION,
            "architecture": "D4W1024",
            "environment": "dog-run",
            "seed": 1,
            "group": "dmc_hard",
            "budget": 1_000_000,
            "run_key": "exp1_D4W1024_dog-run_seed1",
            "overrides": [
                "env_name=dog-run",
                "fork.source=/tmp/parent",
                "fork.arm=injected",
                "injection.m=last",
            ],
        },
    }
    ledger = scheduler.Ledger(tmp_path / "queue", query=lambda _: False)
    owner = {"job_id": "123", "token": "abc", "host": "node"}
    assert ledger.claim(item, owner, lambda _: None)
    ledger.finish(item, owner, error="killed before first arm checkpoint")
    blocked = ledger._row_dir(item) / "blocked.json"
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "approved": True,
                "expected_commit": "a" * 40,
                "item_id": item["id"],
                "blocked_sha256": scheduler.sha(blocked),
                "decision_reference": "reviewed",
            }
        )
    )
    import pytest

    with mock.patch.object(
        scheduler, "_natural_fork", side_effect=ValueError("invalid fork")
    ):
        with pytest.raises(ValueError, match="invalid fork"):
            ledger.review_retry(item, "a" * 40, decision)
    assert ledger.state(item) == "blocked"
    with mock.patch.object(
        scheduler, "_natural_fork", return_value={"fork_step": 10_000}
    ):
        ledger.review_retry(item, "a" * 40, decision)
    assert ledger.state(item) == "pending"
    assert list((ledger._row_dir(item) / "retired_blocks").glob("*.json"))
