"""Generate three independent Kubernetes Jobs for the held-out L40 pilot."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import shlex


SHARDS = 3
WORKERS_PER_GPU = 7
CHECKPOINTS = 15
WALL_SECONDS = 2 * 60 * 60
WORK_SECONDS = WALL_SECONDS - 150  # leave time for checkpoint save and container shutdown


def parse_selector(value):
    key, separator, label = value.partition("=")
    if not separator or not key or not label:
        raise argparse.ArgumentTypeError("GPU selector must be a key=value pair")
    return key, label


def build_jobs(*, namespace, image, pvc, l40_selector, manifest, parent_root,
               results_root, repo_root, python, wandb_secret, cpu, memory,
               mode, batch_id):
    """Build shard-paired Job resources; cluster-specific values are explicit."""
    if mode not in ("fresh", "resume"):
        raise ValueError("mode must be fresh or resume")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?", batch_id):
        raise ValueError("batch_id must be a DNS-safe lowercase label of at most 40 chars")
    if not image or not namespace or not pvc or not wandb_secret:
        raise ValueError("namespace, image, PVC, and W&B secret name are required")
    if "@sha256:" not in image:
        raise ValueError("image must be pinned by digest so all three jobs use identical code")

    jobs = []
    for shard_index in range(SHARDS):
        results = f"{results_root.rstrip('/')}/shard-{shard_index}"
        args = [
            "--manifest", manifest,
            "--results", results,
            "--heldout",
            "--workers-per-gpu", str(WORKERS_PER_GPU),
            "--checkpoint-count", str(CHECKPOINTS),
            "--gpu-model", "L40",
            "--gpu-count", "1",
            "--max-wall-seconds", str(WORK_SECONDS),
            "--shard-count", str(SHARDS),
            "--shard-index", str(shard_index),
        ]
        if mode == "fresh":
            args.append("--fresh")
        shell = "cd " + shlex.quote(repo_root.rstrip("/") + "/main")
        shell += " && exec " + shlex.quote(python) + " -m infra.nrp.replay_pilot "
        shell += shlex.join(args)
        name = f"exp2-heldout-l40-{batch_id}-s{shard_index}"
        container = {
            "name": "pilot",
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["/bin/bash", "-lc", shell],
            "env": [
                {"name": "WANDB_API_KEY", "valueFrom": {"secretKeyRef": {
                    "name": wandb_secret, "key": "WANDB_API_KEY"}}},
                {"name": "WANDB_PROJECT", "value": "Experiment-2-Pilot-Testing"},
                {"name": "PYTHONUNBUFFERED", "value": "1"},
            ],
            "resources": {
                "requests": {"cpu": str(cpu), "memory": str(memory), "nvidia.com/gpu": 1},
                "limits": {"cpu": str(cpu), "memory": str(memory), "nvidia.com/gpu": 1},
            },
            "volumeMounts": [{"name": "pilot-data", "mountPath": "/data/exp2"}],
        }
        jobs.append({
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {"app.kubernetes.io/name": "exp2-heldout-pilot",
                           "exp2-pilot/shard": str(shard_index),
                           "exp2-pilot/batch": batch_id},
            },
            "spec": {
                "backoffLimit": 0,
                "template": {
                    "metadata": {"labels": {"app.kubernetes.io/name": "exp2-heldout-pilot",
                                               "exp2-pilot/shard": str(shard_index)}},
                    "spec": {
                        "restartPolicy": "Never",
                        "terminationGracePeriodSeconds": 180,
                        "nodeSelector": {l40_selector[0]: l40_selector[1]},
                        "containers": [container],
                        "volumes": [{"name": "pilot-data", "persistentVolumeClaim": {
                            "claimName": pvc}}],
                    },
                },
            },
        })
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True,
                        help="immutable image containing the pinned source and CUDA JAX dependencies")
    parser.add_argument("--pvc", required=True,
                        help="shared claim mounted at /data/exp2 with parents and results")
    parser.add_argument("--l40-selector", type=parse_selector, required=True,
                        help="verified Kubernetes node label as key=value")
    parser.add_argument("--manifest", default=(
        "/data/exp2/parents/replay_pilot_parent_manifest_ea3c112.json"))
    parser.add_argument("--parent-root", default="/data/exp2/parents",
                        help="documentation/validation path for the manifest parent tree")
    parser.add_argument("--results-root", default="/data/exp2/results/heldout_l40_3gpu_2h")
    parser.add_argument("--repo-root", default="/opt/scaling-drl-research")
    parser.add_argument("--python", default="python")
    parser.add_argument("--wandb-secret", required=True,
                        help="existing Kubernetes Secret name with key WANDB_API_KEY")
    parser.add_argument("--cpu", default="8", help="per-pod CPU request/limit, verify quota first")
    parser.add_argument("--memory", default="64Gi", help="per-pod memory request/limit, verify quota first")
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--batch-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.manifest.startswith(args.parent_root.rstrip("/") + "/"):
        parser.error("manifest must be located under --parent-root")
    jobs = build_jobs(
        namespace=args.namespace, image=args.image, pvc=args.pvc,
        l40_selector=args.l40_selector, manifest=args.manifest,
        parent_root=args.parent_root, results_root=args.results_root,
        repo_root=args.repo_root, python=args.python,
        wandb_secret=args.wandb_secret, cpu=args.cpu, memory=args.memory,
        mode=args.mode, batch_id=args.batch_id,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        path = args.out_dir / f"{job['metadata']['name']}.json"
        path.write_text(json.dumps(job, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
