# Modal replay pilot

This package is infrastructure for the approved fixed-Exp1-replay pilot. It
requires the replay runner from the same reviewed source commit. It does not
make a full-budget runtime claim.

## Credentials and local client

Use `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` environment secrets. Do not put
values in Git, shell history, or reports. This Cloud environment's HTTP proxy
requires `pip install 'modal[api-proxy-support]'`. If Modal's grpclib TLS context
uses certifi without the environment's trusted proxy CA, configure the client
to use the system trust bundle; never disable TLS verification.

The initial 16 November Cloud authentication attempt returned `AuthError: Token ID
is malformed`. After credentials were updated, Cloud requests timed out before
authentication. Delta authentication was reported successful separately. Check
the intended Modal environment and create the two volumes and W&B secret there
using a working authenticated client; never expose credential values.

## Direct Delta transfer

Run only on a node permitted by NCSA for the transfer. In an isolated Python
environment install Modal and authenticate with `modal setup` or securely
provided environment credentials. Create `exp1-pilot-states` in the intended
Modal environment, then use:

```bash
python main/infra/modal/transfer_parents.py \
  --base /projects/biqc/skaveti1/exp96/campaign_owner_f597ee9_20261019T230748Z/campaign/checkpoints/exp1 \
  --manifest-dir /path/to/small-transfer-receipts --upload
```

This uploads selected files directly from Delta; no replay staging on a
MacBook is required. Existing destination files are not overwritten by default.
The destination layout is `/exp1-pilot-parents/{architecture}/{environment}/seed_N`.
Review metadata schema against actual parent files before uploading; missing
or mismatched fields stop explicitly.

## Verification and launch gates

From a clean, committed checkout containing the runner and this tooling:

```bash
modal run main/infra/modal/exp2_modal.py --mode verify
```

Verification reads all source-checksum files on a CPU-only Modal container and
writes `verification.json` only on success. The GPU launcher additionally needs
a real L4 qualification receipt at
`/results/replay-pilot/l4-qualification-<source-commit>.json`.
It binds exact source/manifest/GPU and requires parent restore, full branch,
probe and restart evidence plus measured worker capacity. The source inventory and its recorded Modal verification are stored under
`/exp1-pilot-transfer-receipts/` on the parent volume. Verification checks the
receipt against the source inventory, rehashes each uploaded file, checks the
parent population and metadata, and derives the runner's `manifest.json` and
`verification.json`. It requires the exact recorded inventory and receipt.

After verification, qualification uses the first manifest identity and exactly
5,000 offline updates, isolated in a source-bound qualification directory. It
interrupts a saved learning branch and resumes in a separate process, then checks
all four full branch horizons and both probes. The pilot retains its full parent
update budget and schedule. Qualification covers only one worker and does not
establish compatibility for every parent or environment. Actual GPU qualification
remains outstanding; do not hand-write passing evidence.

```bash
modal run --detach main/infra/modal/exp2_modal.py --mode qualify
```

`--budget-usd` sets the cumulative spending ceiling (maximum $40), including
earlier reservations. `--qualification-usd` limits each qualification reservation
(default $5) so funds remain for training. Failed sessions retain their full
reservations in the ledger.

To run qualification and, only if its persisted evidence is valid, start the
pilot automatically, use the detached controller:

```bash
modal run --detach main/infra/modal/exp2_modal.py --mode qualify_then_train --budget-usd 40 --qualification-usd 5
```

The controller resumes from a valid receipt for the same source and manifest.
It requests up to seven pilot workers but respects the measured worker count in
the receipt; the current qualification produces only a one-worker receipt.
It stops if qualification fails or the remaining reservation is insufficient. Prior reservations remain charged against the cap.
The ledger is an estimate, not an account-wide spending guarantee; run only one
controller or manual session at a time to avoid concurrent ledger updates.

After qualification, a manual pilot can also be started:

```bash
modal run --detach main/infra/modal/exp2_modal.py --mode train
```

Create `exp2-wandb` as a Modal Secret containing `WANDB_API_KEY`, without logging
the value. Source provenance is shipped as a Git bundle and restored as a real
checkout. One L4 container receives all workers, rather than allocating a GPU
for each process. Finished slots refill without waiting for a whole batch.
Unqualified concurrency is not admitted. OOM evidence reduces subsequent
concurrency; other errors stop and preserve logs.

GPU functions are limited to 23 hours; work sessions reserve at most 22 hours
and leave shutdown time. Resume uses the same durable output root and source.
Standalone qualification requests 8 CPU cores and 48 GiB RAM. Training requests
8 CPU cores and 75 GiB RAM; the single-container qualification-to-training
controller also reserves and bills 75 GiB throughout. These are requested
resources, not measured requirements. Before starting each worker, training
requires 11 GiB of free GPU memory and 9 GiB of cgroup host-memory headroom,
and staggers launches by 30 seconds. These thresholds round up the sampled
one-worker eager-run footprint (8.28 GiB GPU, 6.99 GiB RSS) and add 2 GiB; that
sample did not reach full branches and is not a peak or compiled-run estimate.
These checks do not guarantee that later allocations fit or qualify seven
concurrent workers. Their charges can dominate GPU-only cost estimates.
The local ledger reserves published GPU/CPU/RAM rates plus a $2 allowance.
It is an estimate, not an account-wide hard spending limit, and includes no
unknown prior usage. Verify the account budget and current pricing before
launch. Automatic provider retries are disabled to avoid untracked charges.


## A100 engineering smoke

The separate `a100_smoke` mode requests one A100 80 GB, 8 CPUs and 20 GiB
host RAM. It runs 100 replay updates, 20 raw steps in each branch and a reduced
one-round, five-step probe. It interrupts a saved learning branch and resumes
in another process. The shortened run is engineering smoke evidence only: it
cannot satisfy the full qualification gate or start pilot training, and says
nothing about nine-worker capacity.

```bash
modal run --detach main/infra/modal/exp2_modal.py --mode a100_smoke --budget-usd 40
```

The function has a hard 600-second timeout, with 540 seconds for subprocess
work and 60 seconds reserved for shutdown. The existing cumulative ledger is
preserved; it reserves the full 600 seconds at the published A100-80GB + 8 CPU
+ 20 GiB rate of $0.0008432 per second ($0.50592). That is an estimate based on
pricing checked 17 November 2026, not an account-wide cap; image build, storage,
external costs and changes to rates are not included. Automatic retries are
turned off. Running out of time or failing any check does not produce a passing
smoke result.

## A100 replay pilot

The A100 pilot requests one A100 80 GB, eight CPU cores and 75 GiB host RAM,
with hard CPU and RAM limits. It starts up to eight workers with dynamic refill,
30-second staggering and live GPU/cgroup memory checks. The successful A100
smoke is only an engineering gate: it does not qualify eight-way concurrency,
full branches or scientific results. Memory checks are heuristics; later GPU or
host allocation can still cause OOM. Non-OOM worker failures stop the session.
A completed worker must have a source- and identity-matched analysis artifact.

```bash
modal run --detach main/infra/modal/exp2_modal.py --mode a100_pilot --budget-usd 40
```

Outputs are isolated under `/results/replay-pilot-a100/<source-commit>/`. The
launcher binds the exact smoke receipt and checks that experiment, runtime and
configuration files match the smoke source; launcher and test changes are
allowed. W&B runs receive tags `a100-pilot`, `short-smoke-only` and
`concurrency-unqualified`.

The existing shared reservation ledger remains in force. The estimated A100
plus 8 CPU and 75 GiB rate is $0.0009653 per second ($3.47508/hour); the
launcher reserves the remaining amount up to the $40 cumulative ceiling and
stops before that session reservation expires. This ledger cannot enforce an
absolute account-wide cap: other sessions, storage, builds, usage beyond its
history and price changes are outside it. Avoid concurrent sessions and use an
account-level spending control for an absolute limit.
