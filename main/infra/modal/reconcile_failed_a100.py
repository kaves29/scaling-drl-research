"""Prepare an auditable, bounded adjustment for one stopped A100 attempt.

Inputs are local snapshots of the current ledger, `modal app list --json`, and
`modal billing report --start 2026-11-17T03:00:00 --end 2026-11-17T04:00:00
--resolution h --show-resources --json`, collected after that interval closed.
The output is a new local file; this tool never uploads or overwrites a ledger.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

APP_ID = "ap-ofE9iKlGWLLKdtZElwFuNr"
BEFORE = Decimal("31.01166296")
CURRENT = Decimal("39.99957126")
RETAINED = Decimal("1.00000000")
END = datetime(2026, 11, 17, 4, 0, tzinfo=timezone.utc)
RATE = Decimal("0.0009653")


def reconcile(ledger, apps, billing, now, hashes):
    """Return a new ledger; reject later reservations or incomplete evidence."""
    if now.tzinfo is None or now < END:
        raise ValueError("billing interval has not closed")
    if Decimal(str(ledger.get("reserved_usd"))) != CURRENT:
        raise ValueError("ledger changed since failed reservation")
    adjustments = list(ledger.get("adjustments", []))
    if any(item.get("app_id") == APP_ID for item in adjustments):
        raise ValueError("app already reconciled")
    matching = [item for item in apps if item.get("app_id") == APP_ID]
    if len(matching) != 1 or matching[0].get("state") != "stopped" or str(matching[0].get("tasks")) != "0":
        raise ValueError("failed app is not confirmed stopped")
    app = matching[0]
    start = datetime.fromisoformat(app["created_at"])
    stop = datetime.fromisoformat(app["stopped_at"])
    earliest = datetime(2026, 11, 17, 3, 45, tzinfo=timezone.utc)
    latest = datetime(2026, 11, 17, 3, 50, tzinfo=timezone.utc)
    if (start.tzinfo is None or stop.tzinfo is None or start >= stop
            or start < earliest or stop > latest or (stop - start).total_seconds() > 180):
        raise ValueError("unexpected app runtime")
    rows = [row for row in billing if row.get("object_id") == APP_ID]
    if (not rows or any(row.get("interval_start") != "2026-11-17T03:00:00"
                        or row.get("environment") != "main" for row in rows)
            or not {"A100-80GB", "CPU", "Memory"}.issubset({r.get("resource") for r in rows})):
        raise ValueError("incomplete app billing report")
    try:
        amounts = [Decimal(str(row["cost"])) for row in rows]
        if any(not x.is_finite() or x < 0 for x in amounts):
            raise ValueError("invalid billed amount")
        actual = sum(amounts, Decimal(0))
    except (InvalidOperation, KeyError) as exc:
        raise ValueError("invalid billed amount") from exc
    runtime_ceiling = Decimal(str((stop - start).total_seconds())) * RATE
    if actual > RETAINED or runtime_ceiling > RETAINED or CURRENT - BEFORE < RETAINED:
        raise ValueError("one-dollar retained reservation is insufficient")
    new_total = BEFORE + RETAINED
    adjustments.append({"app_id": APP_ID, "prior_reserved_usd": str(CURRENT),
                        "failed_reservation_usd": str(CURRENT - BEFORE),
                        "retained_usd": str(RETAINED), "new_reserved_usd": str(new_total),
                        "reported_app_cost_usd": str(actual), "runtime_cost_bound_usd": str(runtime_ceiling),
                        "app_created_at": app["created_at"], "app_stopped_at": app["stopped_at"],
                        "evidence_sha256": hashes, "recorded_at": now.isoformat()})
    return {**ledger, "reserved_usd": float(new_total), "adjustments": adjustments}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("ledger", "apps", "billing", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.billing.stat().st_mtime < END.timestamp():
        raise ValueError("billing snapshot predates the end of the interval")
    data = {name: getattr(args, name).read_bytes() for name in ("ledger", "apps", "billing")}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
    result = reconcile(*(json.loads(data[name]) for name in ("ledger", "apps", "billing")),
                       datetime.now(timezone.utc), hashes)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Prepared {args.output}; the remote ledger was not modified.")


if __name__ == "__main__":
    main()
