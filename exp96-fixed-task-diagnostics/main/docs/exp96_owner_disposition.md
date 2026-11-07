# Explicit owner disposition for the 96-parent launch

On 19 October 2026 the owner explicitly accepted the unfinished large-worker
qualification and unfinalized aggregate profiles and instructed submission after
reviewing actual small/medium fitting throughput. This is a risk acceptance,
not an engineering PASS. Original job23021908 remains FAILED/INCOMPLETE.

The separate `exp96-owner-disposition-v1` path accepts only an immutable explicit
owner decision, original failed qualification and actual completed7small/5medium
receipts, scientific-byte compatibility, retained smoke and prior assignment/claim
provenance. Source, queue, scientific approvals, physical memory accounting,
checkpoint recovery, result completion validation and the cumulative600SU ceiling
remain enforced. Historical gate behavior is unchanged without this new schema.
The source difference allowlist permits only the new admission mechanism, narrow
tests and handoff documentation. No science/configuration bytes change.

Resource reservations are planning estimates, explicitly not measured worst-case
peaks. Small/medium fitting completed;4large fitting and checkpoint recovery are
unverified. Runtime admission retains2GiB GPU headroom, CPU isolation and host RAM
reserves. Homogeneous7/5/4 configurations only, exactly the test compositions;
untested mixtures remain disabled. GPU errors retain failed attempts for review.

Use a fresh source-bound96-parent queue and the existing atomic shared budget.
Cancel/reconcile the obsolete held allocation before reserving a replacement.
The replacement is one8A100/128CPU/2007GiB allocation up to48hours, normal queue,
no artificial hold/dependency. Further allocations require available budget;
600SU does not guarantee completion of all96 parents.
