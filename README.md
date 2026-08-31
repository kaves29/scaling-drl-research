# Critic-to-Actor Leakage: Understanding Pathology Propagation in Scaled Actor-Critic DRL
 
## Abstract
 
Architectural advancements in deep reinforcement learning (DRL) have dramatically extended scaling capabilities by injecting simplicity bias into networks to arrive at simpler solutions and delay optimization pathologies. However, recent work within the field suggests such advancements fail to mitigate completely, and pathologies re-emerge at larger scales to reduce the learning capabilities of the agent. Although existing work extensively studies optimization pathologies in DRL, it fails to address the causal interaction between critic-side degradation and actor-side optimization under scaling. To address this, we empirically study the mechanistic relationship between a degrading critic and actor optimization under critic-side scaling, including whether the critic degradation goes undetected by actor-side diagnostics. We fix the actor-side architecture in SimBa-based Soft Actor-Critic (SAC), and systematically scale both the width and depth of the critic-side architecture to investigate the downstream influence of a degraded, scaled critic on actor optimization. We measure critic-side pathology metrics alongside actor-side response metrics across various DMC and MyoSuite tasks to assess whether actor-side degradation results from naively scaling critics. This unique characterization between the actor and critic side pathologies is directly relevant to deployment settings requiring a small-scale actor network, where the actor is optimized by a scaled training-time critic, whose optimization pathologies may silently corrupt the actor network. Developing a deep understanding of this mechanism is crucial, as it opens opportunities for future advancements that expand the scaling capabilities of DRL agents. Our code will be open-sourced on GitHub. 
 
## Built On
 
This codebase is built on top of [SparseNetwork4DRL](https://github.com/lilucse/SparseNetwork4DRL), which provided the base JAX/Flax SAC training loop, environment wrappers, and SimBa architecture implementation. All research design, experimental methodology, and modifications described here are original to this project.
 
## Software / Stack
 
- **JAX / Flax** — core training implementation (SimBa-based SAC)
- **dm_control** — DeepMind Control Suite environments
- **MyoSuite** — musculoskeletal simulation environments
- **Weights & Biases** — experiment tracking and logging
## Status
 
Research in progress, developed toward ISEF and eventual publication.
