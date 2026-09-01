---
id: check_feasibility
name: Check Constraint Feasibility
tier: T0
llm_permitted: false
triggers_on: ["after COMPILE"]
priority: 20
---

## When to use

Immediately after a profile has been compiled into a typed ConstraintSet, and
before any retrieval or model call. This skill answers one question: does the
catalog contain any meal at all that satisfies every constraint? Running it
first means an impossible profile costs zero tokens.

## The pattern

1. Evaluate every recipe against the full ConstraintSet deterministically.
2. If any survive, hand them to retrieval and stop.
3. If none survive, walk the constraint lattice: drop each soft constraint in
   turn and recount; if no single drop helps, try pairs.
4. Rank the relaxations that produce survivors by how many meals they unlock.
5. Never propose relaxing a MEDICAL or RELIGIOUS constraint. List those as
   locked so the user can see they were considered and deliberately protected.

## Pitfalls

- **Proposing a medical relaxation.** "Allow peanuts" is never an option, no
  matter how many meals it unlocks. Filter by severity before ranking.
- **Reporting infeasibility without a path forward.** A bare "no meals found"
  is a dead end; the ranked ladder is the product.
- **Exploring triples.** The combinatorics stop being useful and start being
  slow. If no pair helps, say so and recommend a profile review.
- **Letting the model near this.** Feasibility is arithmetic over a finite set.
  A model adds latency, cost, and the possibility of a wrong answer.
