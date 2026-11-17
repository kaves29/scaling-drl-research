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
a real L4 qualification receipt at `/results/replay-pilot/l4-qualification.json`.
It binds exact source/manifest/GPU and requires parent restore, full branch,
probe and restart evidence plus measured worker capacity. The source inventory and its recorded Modal verification are stored under
`/exp1-pilot-transfer-receipts/` on the parent volume. Verification checks the
receipt against the source inventory, rehashes each uploaded file, checks the
parent population and metadata, and derives the runner's `manifest.json` and
`verification.json`. It requires the exact recorded inventory and receipt.

After verification, qualification uses the first manifest identity, interrupts
a saved learning branch and resumes in a separate process, then checks all four
full branch horizons and both probes. It can qualify only one worker; it does
not establish compatibility for every parent or environment. It reserves up to
$10 against the same local budget ledger as training. Actual GPU qualification
remains outstanding; do not hand-write passing evidence.

```bash
modal run --detach main/infra/modal/exp2_modal.py --mode qualify
```

After qualification:

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
Requested 8 CPU cores and 48 GiB RAM are conservative initial host resources,
not measured requirements. Their charges can dominate GPU-only cost estimates.
The local ledger reserves published GPU/CPU/RAM rates plus a $2 allowance.
It is an estimate, not an account-wide hard spending limit, and includes no
unknown prior usage. Verify the account budget and current pricing before
launch. Automatic provider retries are disabled to avoid untracked charges.
