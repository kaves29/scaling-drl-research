#!/usr/bin/env python3
"""Multi-node cooperative launcher for the Angle 1 grid (2026-09-27).

Coordinates two or more independent SLURM allocations (different nodes,
sharing /work/hdd) working through the same phase1_jobs.txt ->
phase2_jobs.txt -> phase3_jobs.txt queue without ever running the same job
twice, and without any phase2/phase3 job starting until every phase1 job
is DONE - checked fresh each time, not just at manifest-generation time,
since another node may still be working through phase1 while this one's
own slots have run out of phase1 work.

Claiming: atomic os.mkdir() of a "claim" directory inside each job's own
--checkpoint_dir. mkdir is atomic on a POSIX filesystem (including NFS,
which /work/hdd is) - exactly why it's used here instead of a lock file
(a lock file's "check then create" is not itself atomic without this same
primitive underneath it anyway). If the mkdir fails, the claim already
exists; this launcher then decides whether it's a live claim (skip - the
owning SLURM job is still running, per squeue) or a stale one (the owning
job has ended, e.g. an allocation ran out mid-job - safe to steal and
resume from meta.pkl, which run.py's own resumability already handles).
All creation, stale recovery and release transitions take the same atomic
claim_guard directory. Never remove an orphan guard while a launcher could
still hold it; stop all relevant owners before manual recovery. All cooperating
launchers must use this protocol; do not mix old and new revisions.

Usage:
    python scripts/claim_launcher.py --concurrency 32 --num-gpus 8
    python scripts/claim_launcher.py --concurrency 16 --num-gpus 4 --dry-run

Run from the repo root (the same directory phase1_jobs.txt etc. live in) -
matches how generate_manifest.py itself is always invoked.
"""
import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_manifest as gm  # reuses gm.DONE_MARKER - the same signal generate_manifest.py's own classify() treats as authoritative "done" for any job launched through this pipeline (the metrics-CSV fallback in classify() only exists for pre-DONE-marker legacy runs, not relevant to anything this launcher runs).

PHASE_FILES = ["phase1_jobs.txt", "phase2_jobs.txt", "phase3_jobs.txt"]
CLAIM_DIR_NAME = "claim"
OWNER_FILE_NAME = "owner.json"
CLAIM_GUARD_DIR_NAME = "claim_guard"
POLL_INTERVAL_SECONDS = 180  # "every few minutes"

_log_lock = threading.Lock()


def _extract_checkpoint_dir(cmd: str) -> str:
    marker = "--checkpoint_dir "
    idx = cmd.index(marker) + len(marker)
    return cmd[idx:].split(" ", 1)[0]


def _is_done(ckpt_dir: str) -> bool:
    return (Path(ckpt_dir) / gm.DONE_MARKER).exists()


def _claim_dir(ckpt_dir: str) -> Path:
    return Path(ckpt_dir) / CLAIM_DIR_NAME


def _read_owner(claim_dir: Path):
    try:
        return json.loads((claim_dir / OWNER_FILE_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive_on_this_host(hostname: str, pid) -> bool:
    if hostname != socket.gethostname() or pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _claim_owner_alive(owner: dict) -> bool:
    """A claim is alive if EITHER its owning SLURM job is still running
    (the real, cross-node case) OR - only meaningful when there's no real
    SLURM job at all, i.e. SLURM_JOB_ID is unset - its owning process is
    still alive on the SAME host (the login-node dry-run testing case:
    two launcher processes on the same login node have no SLURM job to
    check via squeue at all, so a bare "job_id not found in squeue" would
    wrongly call every claim stale the instant it's made). Real production
    runs always have a real SLURM_JOB_ID, so the squeue check is what
    actually matters there.
    """
    job_id = owner.get("slurm_job_id")
    if job_id and job_id != "none":
        try:
            out = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return True  # can't confirm either way - be conservative, don't steal
        if out.returncode == 0 and out.stdout.strip():
            return True
        if out.returncode == 0:
            return False  # squeue ran fine and found nothing - genuinely gone
        return True  # squeue itself failed - don't trust that as "stale"
    return _pid_alive_on_this_host(owner.get("hostname"), owner.get("pid"))


@contextlib.contextmanager
def _claim_guard(ckpt_dir):
    """Serialize claim transitions; an orphan guard requires manual recovery."""
    guard = Path(ckpt_dir) / CLAIM_GUARD_DIR_NAME
    try:
        guard.mkdir()
    except FileExistsError:
        yield False
        return
    try:
        yield True
    finally:
        guard.rmdir()


def _try_claim(ckpt_dir: str, my_job_id: str, my_host: str) -> bool:
    """Claim fresh/stale work under the same atomic transition guard."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    with _claim_guard(ckpt_dir) as acquired:
        if not acquired:
            return False
        cdir = _claim_dir(ckpt_dir)
        try:
            cdir.mkdir()
        except FileExistsError:
            owner = _read_owner(cdir)
            if owner is None or _claim_owner_alive(owner):
                return False
            (cdir / OWNER_FILE_NAME).unlink()
            cdir.rmdir()
            cdir.mkdir()
        (cdir / OWNER_FILE_NAME).write_text(
            json.dumps(
                {
                    "slurm_job_id": my_job_id,
                    "hostname": my_host,
                    "pid": os.getpid(),
                    "claimed_at": time.time(),
                }
            )
        )
        return True


def _release_claim(ckpt_dir: str) -> bool:
    """Release only this process's claim under the transition guard."""
    try:
        with _claim_guard(ckpt_dir) as acquired:
            if not acquired:
                return False
            cdir = _claim_dir(ckpt_dir)
            owner = _read_owner(cdir)
            if owner is None or (
                owner.get("pid") != os.getpid()
                or owner.get("hostname") != socket.gethostname()
                or owner.get("slurm_job_id") != os.environ.get("SLURM_JOB_ID", "none")
            ):
                return False
            (cdir / OWNER_FILE_NAME).unlink()
            cdir.rmdir()
            return True
    except OSError:
        return False


def _phase_fully_done(phase_file: str, repo_root: Path) -> bool:
    lines = (repo_root / phase_file).read_text().splitlines()
    return all(
        _is_done(_extract_checkpoint_dir(line)) for line in lines if line.strip()
    )


def _log(log_path: Path, slot: int, msg: str) -> None:
    with _log_lock:
        with open(log_path, "a") as f:
            f.write(f"{time.time():.6f} [slot{slot}] {msg}\n")


def _worker(slot, num_gpus, dry_run, my_job_id, my_host, repo_root, log_path, deadline=None, phase_files=None):
    gpu = slot % num_gpus
    phase_files = PHASE_FILES if phase_files is None else phase_files
    for phase_idx, phase_file in enumerate(phase_files):
        if deadline is not None and time.time() > deadline:
            _log(log_path, slot, "max-seconds reached (test mode) - stopping")
            return
        for prior in phase_files[:phase_idx]:
            while not _phase_fully_done(prior, repo_root):
                _log(log_path, slot, f"waiting on {prior} to finish (elsewhere) before starting {phase_file}")
                time.sleep(POLL_INTERVAL_SECONDS)

        while True:
            if deadline is not None and time.time() > deadline:
                _log(log_path, slot, "max-seconds reached (test mode) - stopping")
                return
            lines = [l for l in (repo_root / phase_file).read_text().splitlines() if l.strip()]
            claimed_something = False
            for cmd in lines:
                ckpt_dir = _extract_checkpoint_dir(cmd)
                if _is_done(ckpt_dir):
                    continue
                if not _try_claim(ckpt_dir, my_job_id, my_host):
                    continue
                claimed_something = True
                _log(log_path, slot, f"claimed {ckpt_dir}")
                if dry_run:
                    print(f"[dry-run][slot{slot}] would run: {cmd}")
                    time.sleep(
                        2
                    )  # hold briefly so a concurrent tester can genuinely race this claim
                    released = _release_claim(ckpt_dir)
                    _log(
                        log_path,
                        slot,
                        f"{'released' if released else 'release blocked'} {ckpt_dir}",
                    )
                else:
                    env = dict(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    rc = subprocess.run(
                        ["bash", "-c", cmd], cwd=str(repo_root), env=env
                    ).returncode
                    if rc != 0:
                        _log(
                            log_path,
                            slot,
                            f"FAILED rc={rc}: {ckpt_dir} - releasing claim for retry",
                        )
                        if not _release_claim(ckpt_dir):
                            _log(
                                log_path,
                                slot,
                                f"release blocked: {ckpt_dir}; inspect claim/guard before recovery",
                            )
                    else:
                        _log(log_path, slot, f"completed: {ckpt_dir}")
                        # Claim intentionally left in place: the DONE marker
                        # (written by run.py itself on success) is the real
                        # completion signal, and _is_done() already
                        # short-circuits past any job with one, claim
                        # directory or not.
                break  # re-scan from the top of the phase file after every attempt
            if not claimed_something:
                if _phase_fully_done(phase_file, repo_root):
                    _log(log_path, slot, f"{phase_file} fully done")
                    break
                _log(log_path, slot, f"nothing claimable in {phase_file} right now - waiting")
                time.sleep(POLL_INTERVAL_SECONDS)
    _log(log_path, slot, "finished all phases")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent job slots (N).")
    parser.add_argument("--num-gpus", type=int, required=True, help="GPUs on this node - slots are pinned round-robin via slot %% num_gpus.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Claim + print what would run, briefly hold the claim, then release - never actually launches training.",
    )
    parser.add_argument("--log-file", default=None, help="Defaults to claim_launcher_<job-or-pid>.log in the repo root.")
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="Test-only: stop after this many seconds regardless of phase completion. "
        "Real launches never set this - dry-run mode otherwise has no natural end, "
        "since nothing it does ever writes a DONE marker.",
    )
    parser.add_argument(
        "--phase-files", nargs="+", default=PHASE_FILES,
        help="Manifests run in order, each only after every job of the previous ones is DONE "
        "(default: the Angle 1 phase files). Exp 1/2: exp12_exp1_jobs.txt, then, on the GPU model "
        "that produced the forks, exp2_arms_<device>.txt.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    my_job_id = os.environ.get("SLURM_JOB_ID", "none")
    my_host = socket.gethostname()
    log_path = Path(args.log_file) if args.log_file else repo_root / f"claim_launcher_{my_job_id}_{os.getpid()}.log"
    deadline = (time.time() + args.max_seconds) if args.max_seconds is not None else None

    print(f"[claim_launcher] concurrency={args.concurrency} num_gpus={args.num_gpus} dry_run={args.dry_run}")
    print(f"[claim_launcher] slurm_job_id={my_job_id} hostname={my_host} log={log_path}")

    threads = [
        threading.Thread(target=_worker, args=(slot, args.num_gpus, args.dry_run, my_job_id, my_host, repo_root, log_path, deadline,
                                               args.phase_files))
        for slot in range(args.concurrency)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("[claim_launcher] all slots finished all phases")


if __name__ == "__main__":
    main()
