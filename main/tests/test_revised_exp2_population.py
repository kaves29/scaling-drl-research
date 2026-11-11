"""Population and schedule for the owner's revised 20-run pilot."""
from collections import Counter

from experiments.exp12.revised_exp2_population import population, checkpoint_interactions


def test_matched_reference_runs_and_checkpoint_schedule():
    rows = population()
    assert len(rows) == 20
    groups = Counter((r["environment"], r["seed"]) for r in rows)
    assert len(groups) == 10 and set(groups.values()) == {2}
    assert all({r["reference_regime"] for r in rows
                if (r["environment"], r["seed"]) == key}
               == {"reference_small", "reference_large"} for key in groups)
    assert Counter(r["environment"] for r in rows) == {
        "dog-run": 6, "humanoid-run": 6, "myo-key-turn": 4, "myo-pen-twirl": 4}
    checkpoints = checkpoint_interactions()
    assert len(checkpoints) == len(set(checkpoints)) == 15
    assert checkpoints == sorted(checkpoints) and checkpoints[-1] == 500_000
    assert all(abs(checkpoints[i] * 2 - (i + 1) * 1_000_000 / 15) <= 2
               for i in range(15))
