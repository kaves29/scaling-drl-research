"""Shape-only FP32 storage estimates for every approved m; never selects source cells."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import jax
    import jax.numpy as jnp
    from scale_rl.agents.sac.sac_network import SACCritic, SACActor
    from experiments.exp12.injection import split_params, head_blocks

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-inventory", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    count = lambda tree: sum(
        math.prod(x.shape) for x in jax.tree_util.tree_leaves(tree)
    )
    for task in json.loads(Path(args.task_inventory).read_text())["tasks"]:
        obs, act = task["observation_dim"], task["action_dim"]
        o = jax.ShapeDtypeStruct((1, obs), jnp.float32)
        a = jax.ShapeDtypeStruct((1, act), jnp.float32)
        actor = jax.eval_shape(
            SACActor("residual", 1, 128, act, jnp.float32).init,
            jax.random.PRNGKey(0),
            observations=o,
        )["params"]
        for width in (1024, 1536):
            critic = jax.eval_shape(
                SACCritic("residual", 4, width, jnp.float32).init,
                jax.random.PRNGKey(0),
                observations=o,
                actions=a,
            )["params"]
            n, actor_n = count(critic), count(actor)
            ordinary = (4 * n + 3 * actor_n + 3) * 4
            for m in ("last", "half", "all"):
                _, head = split_params(critic, 4, head_blocks(m, 4))
                extra = 16 * count(head)
                replay = (2 * obs + act + 3) * 4 * 475000
                streams = 2 * task["npz_bytes_per_arrival"] * 125000
                rows.append(
                    dict(
                        environment=task["environment"],
                        architecture=f"D4W{width}",
                        injection_m=m,
                        actor_parameters=actor_n,
                        critic_parameters=n,
                        head_parameters=count(head),
                        ordinary_agent_dense_bytes=ordinary,
                        injected_agent_extra_dense_bytes=extra,
                        twelve_agent_states_plus_fork_replay_two_streams_GB=(
                            12 * ordinary + 6 * extra + replay + streams
                        )
                        / 1e9,
                    )
                )
    report = dict(
        scope="shape-only payload estimates; no weights/m/source cells selected; excludes extra control capture, filesystem metadata, encoding effects and pilot outputs",
        rows=rows,
    )
    with open(args.out, "x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
