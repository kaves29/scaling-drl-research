# Testing and Verification

Tests must target failure modes that could produce
scientifically misleading results.

Prioritize:
- metric correctness
- seed propagation
- configuration propagation
- experimental isolation
- checkpoint correctness
- numerical sanity
- real execution paths

A clean exit code is not sufficient evidence.

After:
- training changes → perform a real smoke test
- metric changes → inspect actual metric values
- checkpoint changes → test save + resume
- WandB changes → verify the real run exists
- multi-seed launches → verify distinct seeds and output paths

Do not claim real-run validation if only mocks were used.

