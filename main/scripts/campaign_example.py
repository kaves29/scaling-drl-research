"""Generate explicit synthetic export/example outputs; never reads real evidence."""

import argparse
from pathlib import Path
from tests.test_campaign_offline import fixture
from analysis.campaign.pipeline import run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", required=True)
a = p.parse_args()
root = Path(a.out)
root.mkdir(parents=True, exist_ok=False)
fixture(root / "synthetic_103.zip")
run(root / "synthetic_103.zip", root / "analysis", reps=100)
(root / "SYNTHETIC.txt").write_text(
    "All records are deterministic synthetic fixtures. No campaign scientific results.\n"
)
