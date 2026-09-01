---
id: assess_trust
name: Assess Trust
tier: T3
llm_permitted: true
triggers_on: ["after SCORE"]
priority: 50
---

## When to use

After nutrition has been computed and constraints checked for the leading
candidate, before ranking or explanation happen. This is the gate: a
candidate that doesn't clear it is refused and escalated — never ranked,
never explained, never shown to the user.

## The pattern

1. Compute two deterministic signals: `catalog_coverage` (from
   `compute_nutrition`, mass-weighted) and `constraint_completeness`
   (fraction of the ConstraintSet CONCLUSIVELY evaluated — satisfied or
   violated, not merely "not marked uncheckable"; a constraint that never
   made it into either bucket counts against completeness exactly like one
   explicitly marked uncheckable).
2. Fold in the model's self-assessment as the third signal, weighted 0.25
   against the two deterministic signals' combined 0.75 (0.45 coverage,
   0.30 completeness). Clamp it to `[0, 1]` and treat a missing or NaN
   value as NEUTRAL (`neutral_model_default`, 0.5) — never as confident,
   because defaulting a silent model to "confident" would let it inflate
   its own trust score by doing nothing at all.
3. Compute the weighted composite, then SEPARATELY check whether either
   deterministic signal falls below `weak_signal_floor`. Record whichever
   one is weaker — never the model signal — as `failing_signal`.
4. Gate CONJUNCTIVELY: `composite >= refusal_threshold` AND
   `failing_signal is None`. A weighted average always lets one strong
   signal mask one that's critically weak: `constraint_completeness`
   (0.30) plus `model_self_assessment` (0.25) alone sum to exactly the
   default 0.55 threshold, so a fully-checked constraint set paired with a
   maximally confident model can carry the composite to the bar no matter
   how thin `catalog_coverage` is. The floor check is what stops that.
5. On refusal, name `failing_signal` in the escalation, never return a
   generic apology — the name is what lets a user understand what to fix,
   and what the healing loop (Task 16) clusters on to notice a pattern
   across many refusals.

## Pitfalls

- **Gating on the weighted composite alone.** This is the exact mistake
  the conjunctive check exists to prevent — see step 4's arithmetic. A
  weighted average cannot see that one input sat at zero while the other
  two carried it over the line; only a separate per-signal floor can.
- **Letting `failing_signal` ever name the model axis.** Even when the
  model's self-assessment is numerically the weakest of the three inputs,
  `failing_signal` must only ever point at `catalog_coverage` or
  `constraint_completeness`. A confident model must never be made to look
  like the cause of a refusal it didn't cause — that trains the user, and
  the healing loop, to distrust the wrong signal.
- **Defaulting a missing self-assessment to confident.** Treat `None` (or
  NaN — comparisons against NaN are always false, so a naive `min`/`max`
  clamp silently fails to catch it) as neutral, not as 1.0 and not as 0.0.
  A model that says nothing is not the same as a model that says "I'm
  sure."
- **Returning a generic refusal message.** "Something went wrong, please
  try again" teaches the user nothing about which constraint or which
  coverage gap caused it, and gives the healing loop nothing to cluster
  on. Every refusal names its failing signal and the numeric threshold it
  missed.
