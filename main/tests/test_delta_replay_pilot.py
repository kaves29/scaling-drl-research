from collections import Counter

import pytest

from infra.delta.replay_pilot import identities_from_manifest


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
