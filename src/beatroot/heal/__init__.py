"""The healing loop: incidents become proposals. Spec §9, Task 16.

`beatroot.store.incidents.IncidentLog` is the input side — every escalation,
drift finding, infeasibility and unrecognised ingredient the running system
hits lands there. This package is the output side: `heal.cluster.cluster`
groups those incidents into repeated patterns, and `heal.proposals.propose`
turns each pattern into a `Proposal`.

See `heal.proposals` for the design position this package exists to hold:
rule-changing proposals are written to disk for a human to review and are
NEVER auto-applied; only additive eval-case generation is.
"""
