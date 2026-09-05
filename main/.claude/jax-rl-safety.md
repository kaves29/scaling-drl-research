RL / JAX Safety Rules

For JAX/Flax code:

Be explicit about array shapes at boundaries between environment, network, loss,
gradient, and logging code.

Treat shape changes as high-risk in jitted/vectorized code.

Avoid Python-side control flow that depends on traced JAX values.

Avoid boolean indexing or other dynamic-shape operations inside jit unless their
behavior is known to be static and safe.

Be careful with jit, vmap, scan, grad, pmap, and nested transformations:
preserve the existing transformation structure unless a change is necessary.

Preserve gradient flow intentionally. Any new stop_gradient, detach, masking,
averaging, or reduction must have a clear reason.

For gradients, verify scalar-output requirements and resulting shapes before
assuming an implementation is valid.

When changing a loss or gradient calculation, verify numerical finiteness and basic
magnitude/sanity checks, not just execution.

For RL code generally:

Keep environment interaction, replay/storage, target construction, actor updates,
critic updates, evaluation, and logging conceptually separable.

Do not change update ordering or frequency accidentally.

Do not change train/eval mode behavior accidentally.

Do not hide NaNs, infinities, exploding norms, or invalid rewards by clipping or
replacing them unless that behavior is an explicit part of the experiment.

## Known JAX Pitfalls

jax.grad requires scalar output.

Boolean array indexing inside jit can create
dynamic output shapes.

Empty metric buffers need explicit guards.

Do not introduce stop_gradient/detach without
scientific justification.