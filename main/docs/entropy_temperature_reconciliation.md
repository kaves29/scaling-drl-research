# Entropy-temperature authority and equations

This is a clarification audit, not a new scientific decision. The integrated candidate retains the existing `temp_target_entropy_coef=-0.5`. No actor, critic, temperature, initialization, optimizer, or configuration code is changed by this audit.

## Authority traced independently

| Source | What it actually establishes |
|---|---|
| `.claude/methodology-exp1-exp2.md:50` | The original configuration list says **“Entropy: abs(A)/2”**. It gives neither a signed entropy variable nor a temperature equation. Later amendments do not explicitly replace this line. |
| `docs/exp12_decisions.md:303` (G1) | Explicitly proposes **target entropy `−|A|/2`**, initial alpha `.01`, temperature weight decay zero, and inherited optimizer behavior. Its explanation about a sign convention is an interpretation, not an equation demonstrating equivalence. |
| `docs/exp12_decisions.md:412` | The recorded lead answer approves keeping G1's existing values, with UTD=2. This is explicit provenance for retaining the current negative target. |
| Released `SonyResearch/simba@7d0358b` | Config coefficient `−0.5`, entropy statistic `−mean(log pi)`, and temperature objective `alpha*(entropy-target_entropy)`. Config/update SHA-256 values are recorded in `methodology_reconstruction.md`. This corroborates implementation, without overriding the lead's methodology. |
| `docs/methodology_reconstruction.md`, `docs/methodology_implementation_audit.md`, `docs/remaining_scientific_decisions.md` | Prior audits flag the unsigned-versus-negative wording. They do not grant authority to choose a replacement sign. |

The source documents therefore do **not** establish two equally explicit signed equations: G1 supplies one, while the original line is underspecified. It is possible that the original line denotes a magnitude or a mean-log-density target. It is also possible that it intends a positive differential-entropy target. The wording alone cannot establish which was intended. This ambiguity does not prevent integrating and testing the unchanged G1 implementation.

## Implemented equations, including the action transform

Let `d=|A|`, `alpha=exp(z)`, `ell=E[log pi(a|s)]`, and `H=-ell`. Here `a=tanh(u)`, with `u` sampled from a diagonal Gaussian. `NormalTanhPolicy` returns a transformed distribution, and its log density includes the tanh Jacobian:

`log pi(a|s) = sum_i [log Normal(atanh(a_i); mu_i, sigma_i) - log(1-a_i^2)]`.

Consequently `train/entropy=-mean(log_probs)` is a Monte Carlo differential-entropy estimate in action space, not its negation and not the untransformed Gaussian's entropy. Differential entropy may be negative. Both `±d/2` are mathematically feasible targets on `[-1,1]^d`, whose maximum entropy is `d*log(2)`; feasibility alone does not choose one.

The actual paths are `scale_rl/networks/policies.py:44`, `scale_rl/agents/sac/sac_update.py:39`, `sac_update.py:156`, and `sac_agent.py:447`:

- Actor objective: `J_pi = E[alpha*log pi - Q_min]`.
- Temperature objective: `J_alpha = alpha*(H-h*)`.
- Derivative with respect to `z`: `dJ_alpha/dz = alpha*(H-h*)`.
- Configured target: `h* = -0.5*d`, computed from the actual action dimension.
- Existing temperature AdamW has no weight decay; when entropy is below its target, the descent direction increases alpha, and conversely.

In mean-log-density notation the same current objective is `J_alpha=-alpha*(ell+h*)`, with `ell*= -h* = +d/2`. This is a valid notation conversion because **both the variable and the target change sign**. Changing only the configured target from `−d/2` to `+d/2`, while leaving the measured `H` and objective unchanged, changes scientific behavior.

## Independent CPU verification

`tests/test_exp12_independent_numerics.py` verifies the transformed density with an independent NumPy change-of-variables calculation, and checks the actual temperature update against the analytic gradient. For `d=2`, `H=0`, and alpha `.01`:

| Interpretation | Signed H target | Gradient in log alpha | Descent direction |
|---|---:|---:|---|
| Existing G1 / original magnitude or mean-log-density reading | `−1` | `+.01` | alpha decreases |
| Original line interpreted literally as positive H | `+1` | `−.01` | alpha increases |

The test uses SGD to expose the derivative without Adam's normalization; production remains AdamW. A separate scalar reference (`scripts/methodology_cpu_reference.py`) confirms the same opposite gradient signs. Fixture precision comparisons do not introduce a CUDA tolerance or a scientific gate.

## Exact remaining decision

There is **no confirmed implementation sign bug** relative to the approved G1 answer or released SimBa. There is an unresolved interpretation of the higher-level unsigned wording. Calling that wording “notation only” without defining its variable would overstate the evidence; calling it an explicit conflicting positive-H equation would also overstate it.

The lead can clarify that the original line denotes the current negative-H target's magnitude/equivalent positive mean-log-density target, or state that positive H was intended and explicitly authorize a scientific correction. The latter would change training and require new source/config provenance and qualification. This audit selects neither alternative, proposes no code change, and does not block local integration or CPU engineering verification.
