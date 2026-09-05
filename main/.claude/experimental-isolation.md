# Experimental Isolation

Before changing experimental code, identify:

1. independent variable
2. controlled variables
3. measured outcome
4. potential confounders

Never accidentally alter:

- architecture
- seed
- optimizer
- learning rate
- batch size
- replay behavior
- UTD
- reward scaling
- action scaling
- normalization
- discounting
- evaluation protocol
- environment configuration
- initialization
- checkpoint state
- RNG behavior

A comparison involving two agents must explicitly document:
- what is shared
- what is independent
- why anything shared does not introduce a confound

Never share:
- replay buffers
- optimizer state
- hidden training state
- parameters
- RNG state

unless the methodology explicitly requires it.

If a proposed optimization changes experimental isolation,
stop and ask before implementing it.