from collections import Counter

import pytest

from infra.delta.replay_pilot import identities_for_shard, identities_from_manifest


def paired_manifest():
    return {"runs": [
        {"environment": env, "seed": seed, "architecture": architecture}
        for env, seeds in (("dog-run", (1, 2, 3)),
                           ("humanoid-run", (1, 2, 3)),
                           ("myo-key-turn", (1, 2)),
                           ("myo-pen-twirl", (1, 2)))
        for seed in seeds
        for architecture in ("D2W512", "D4W1536")
    ]}


def test_manifest_is_ten_matched_pairs_across_four_pilot_environments():
    identities = identities_from_manifest(paired_manifest())
    pairs = Counter((row["environment"], row["seed"]) for row in identities)
    assert len(identities) == 20
    assert len(pairs) == 10
    assert set(pairs.values()) == {2}
    assert Counter(env for env, _ in pairs) == {
        "dog-run": 3, "humanoid-run": 3,
        "myo-key-turn": 2, "myo-pen-twirl": 2,
    }
    assert {row["reference_regime"] for row in identities} == {
        "reference_small", "reference_large"
    }


def test_three_shards_keep_each_small_large_pair_together_and_balance_load():
    identities = identities_from_manifest(paired_manifest())
    shards = [identities_for_shard(identities, index, 3) for index in range(3)]
    assert [len(shard) for shard in shards] == [8, 6, 6]
    assert sorted(row["environment"] + str(row["seed"])
                  for shard in shards for row in shard) == sorted(
        row["environment"] + str(row["seed"]) for row in identities
    )
    for shard in shards:
        by_pair = {}
        for row in shard:
            by_pair.setdefault((row["environment"], row["seed"]), set()).add(
                row["reference_regime"]
            )
        assert all(regimes == {"reference_small", "reference_large"}
                   for regimes in by_pair.values())


def test_shard_arguments_are_checked():
    identities = identities_from_manifest(paired_manifest())
    with pytest.raises(ValueError):
        identities_for_shard(identities, 3, 3)
    with pytest.raises(ValueError):
        identities_for_shard(identities, 0, 0)


def test_nrp_jobs_request_one_l40_each_and_use_separate_shards(tmp_path):
    from infra.nrp.heldout_jobs import build_jobs

    jobs = build_jobs(
        namespace="research", image="registry.example/exp2@sha256:abc",
        pvc="pilot-data", l40_selector=("gpu.example/model", "L40"),
        manifest="/data/exp2/parents/parents.json", parent_root="/data/exp2/parents",
        results_root="/data/exp2/results/heldout", repo_root="/opt/research",
        python="/opt/venv/bin/python", wandb_secret="wandb-api-key",
        cpu="8", memory="64Gi", mode="fresh", batch_id="batch-001",
    )
    assert len(jobs) == 3
    assert len({job["metadata"]["name"] for job in jobs}) == 3
    for index, job in enumerate(jobs):
        assert job["metadata"]["namespace"] == "research"
        spec = job["spec"]
        assert "activeDeadlineSeconds" not in spec
        pod = spec["template"]["spec"]
        assert pod["nodeSelector"] == {"gpu.example/model": "L40"}
        container = pod["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
        command = container["command"][2]
        assert "--workers-per-gpu 7" in command
        assert f"--shard-index {index}" in command
        assert "--shard-count 3" in command
        assert "--gpu-model L40" in command
        assert "--max-wall-seconds 7050" in command
        assert f"/shard-{index}" in command
        assert "--fresh" in command


def test_nrp_resume_specs_reuse_result_dirs_without_fresh_marker():
    from infra.nrp.heldout_jobs import build_jobs

    jobs = build_jobs(
        namespace="research", image="registry.example/exp2@sha256:abc",
        pvc="pilot-data", l40_selector=("gpu.example/model", "L40"),
        manifest="/data/exp2/parents/parents.json", parent_root="/data/exp2/parents",
        results_root="/data/exp2/results/heldout", repo_root="/opt/research",
        python="python", wandb_secret="wandb-api-key", cpu="8", memory="64Gi",
        mode="resume", batch_id="batch-002",
    )
    for index, job in enumerate(jobs):
        command = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
        assert "--fresh" not in command
        assert f"/shard-{index}" in command


def test_nrp_job_generator_rejects_mutable_image_tag():
    from infra.nrp.heldout_jobs import build_jobs

    with pytest.raises(ValueError, match="pinned by digest"):
        build_jobs(
            namespace="research", image="registry.example/exp2:latest",
            pvc="pilot-data", l40_selector=("gpu.example/model", "L40"),
            manifest="/data/exp2/parents/parents.json", parent_root="/data/exp2/parents",
            results_root="/data/exp2/results/heldout", repo_root="/opt/research",
            python="python", wandb_secret="wandb-api-key", cpu="8", memory="64Gi",
            mode="fresh", batch_id="batch-003",
        )
@pytest.mark.parametrize("mutate", ["missing_regime", "missing_env", "duplicate"])
def test_manifest_rejects_incomplete_or_duplicate_pairs(mutate):
    manifest = paired_manifest()
    if mutate == "missing_regime":
        manifest["runs"].pop()
    elif mutate == "missing_env":
        manifest["runs"] = [r for r in manifest["runs"]
                            if r["environment"] != "myo-pen-twirl"]
    else:
        manifest["runs"].append(dict(manifest["runs"][0]))
    with pytest.raises(ValueError):
        identities_from_manifest(manifest)
