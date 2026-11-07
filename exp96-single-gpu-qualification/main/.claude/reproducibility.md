# Reproducibility

Randomness is experimental state.

Every run must have an explicit seed.

Preserve deterministic propagation through:
- Python random
- NumPy
- JAX
- environment seeding
- replay-buffer sampling
- agent-specific RNG streams

Independent agents must not accidentally consume
a shared global RNG stream.

Same:
- seed
- configuration
- environment
- code
should reproduce the same experiment as closely as
the backend permits.

Record enough metadata to reconstruct:
- seed
- architecture
- environment
- experiment
- configuration
- run identity
- onset step
- checkpoint/snapshot identity

Checkpoint validity requires:
- parameters
- optimizer state
- counters
- RNG state
- replay state
- configuration/provenance
as applicable.

Never silently resume from another experimental condition.