# Remaining Experiment 1 runs

Owner direction: credit only the 88 completed historical Exp96 cells. Do not
count or resume the eight partial runs. Their cells remain required, so they
appear as fresh reruns in the missing-run plan; their existing evidence is not
deleted. Retain all eleven environments, three sizes and ten seeds per cell.

|Critic|Required|Completed historical cells|Remaining|
|---|---:|---:|---:|
|D2W512|110|32|78|
|D4W1024|110|32|78|
|D4W1536|110|24|86|
|Total|330|88|242|

The seven newly added environments require 70 runs per architecture. In each of
the four historical environments, small/medium require two additional seeds;
large requires four: seeds7/8 must be redone plus the two added official seeds.
Thus each historical environment contributes eight new runs, and seven new
environments contribute thirty each:32+210=242.

## Program

`scripts/generate_remaining_exp1_manifest.py` reuses the official source/config
manifest generator, validates the completed-cell registry, and emits commands
only for the missing cells. It never submits or starts a training process and
does not assign GPUs, waves or partitions. Existing reviewed launch/preflight
machinery must consume this selection; no separate unqualified runner is added.

Example planning invocation from `main/` (IDs1–10 are provisional, not a new
seed approval):

```bash
python scripts/generate_remaining_exp1_manifest.py \
  --root /absolute/fresh_campaign_root \
  --seeds 1 2 3 4 5 6 7 8 9 10 \
  --out /absolute/remaining.json \
  --commands-out /absolute/remaining_commands.txt
```

The JSON retains the official configuration hashes, source identity, argv lists,
counts, original reused run IDs and a checksum of the completion registry.
Optional command lines are shell-quoted and must be run from `main/` in the
reviewed environment. Planning output remains approved/executable/submitted=false.
Output files are created exclusively; existing files are never overwritten.

If a missing cell's checkpoint directory already exists, generation stops rather
than silently resuming a partial or overwriting results. Use a new campaign root.
Do not execute the command list directly to bypass allocation/source/budget gates.

## Completion evidence and source boundaries

`configs/exp1_completed_historical.json` records only the 88 original completed
IDs at f597ee9898e4d401912092c336c3e5330cf2fda1 and the checksum of their archived
inventory. Existing resolved-config checks found matching core settings for
all96 original cells; the 88 completed histories reach the nominal endpoint with
finite final deficits. This registry is an owner-authorized scheduling exclusion,
not a substitute for full checkpoint/artifact/source admission into final analysis.
Original labels/source hashes must remain intact. The resulting new-run ledger
alone contains242 cells; final330-cell certification still needs the reviewed
historical-cohort admission/materialization route. No such gate is silently waived.

## Seed IDs

Ten official IDs must be supplied explicitly and match the official source's
population. The existing profile still uses provisional1–10; the seed9
development-exclusion decision is pending. If IDs1–8,10,11 are chosen, first
update/freeze that official profile, then regenerate. Both candidate sets include
all original seeds1–8, so the78/78/86 counts are unchanged. Other seed sets require
recalculation; an incompatible profile fails closed rather than quietly running
the wrong seed.

## Validation

Focused CPU tests cover exact counts, no overlap with reused identities, all eight
partial cells remaining missing, duplicates/foreign source, missing/partial
completion rows, incompatible seed lists and refusal to reuse existing outputs.
The real CLI is also exercised to produce the242-entry preview; no model fit,
training, GPU job, source promotion or merge is performed.

## Architecture-specific lists

Use `--architecture-dir /absolute/new_directory` to emit three disjoint command
lists (78 small,78 medium,86 large) alongside the manifest. See the
[four-GPU resource assessment](exp1_allocation_planning_20261104.md) before choosing
concurrency or walltime; requested10/6/3 packing and10/18/48 hours are not qualified.
