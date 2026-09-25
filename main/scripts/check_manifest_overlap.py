#!/usr/bin/env python3
"""Refuse to launch two or more job-list manifests concurrently if they
contain overlapping --checkpoint_dir values.

Background: on 2026-09-20/21, job_list_part2.txt (created for a capacity-
adding relaunch) turned out to be byte-identical to the last 50 lines of
job_list.txt. Both were then run concurrently, launching 50 configs twice
against the same --checkpoint_dir and the same log file. See
.claude/research-methodology.md's "Infrastructure Safety Rules" section.

Usage:
    python scripts/check_manifest_overlap.py job_list.txt job_list_part2.txt [...]

Exit 0 and print OK if no --checkpoint_dir value appears in more than one
file. Exit 1 and list every overlapping path (and which files it came
from) otherwise.
"""
import argparse
import re
import sys
from collections import defaultdict

CKPT_DIR_RE = re.compile(r"--checkpoint_dir\s+(\S+)")


def extract_checkpoint_dirs(path):
    dirs = []
    with open(path) as f:
        for line in f:
            m = CKPT_DIR_RE.search(line)
            if m:
                dirs.append(m.group(1))
    return dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+", help="Two or more job-list files intended to run concurrently.")
    args = parser.parse_args()

    if len(args.manifests) < 2:
        print("Need at least 2 manifest files to check for overlap; nothing to compare.")
        return 0

    owner = defaultdict(list)
    for path in args.manifests:
        for d in extract_checkpoint_dirs(path):
            owner[d].append(path)

    overlaps = {d: files for d, files in owner.items() if len(set(files)) > 1}

    if not overlaps:
        print(f"OK: no overlapping --checkpoint_dir values across {len(args.manifests)} manifest(s).")
        return 0

    print(
        f"REFUSING TO PROCEED: {len(overlaps)} overlapping --checkpoint_dir "
        f"value(s) found across manifests intended to run concurrently."
    )
    print("Running these together would launch two processes against the same checkpoint/log path.\n")
    for d, files in sorted(overlaps.items()):
        print(f"  {d}")
        for f in files:
            print(f"    - {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
