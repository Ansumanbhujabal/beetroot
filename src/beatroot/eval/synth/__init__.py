"""Synthetic data generation for the eval suite. Spec §12.

`profiles.generate_profiles` produces constraint profiles with a FREE, exact
oracle (brute-force enumeration over the finite catalog). `adversarial.
generate_adversarial` produces attack cases across ten families — injection,
synonym-evasion-in-the-constraint, case/whitespace variance, homoglyphs,
transitive/composite allergens, contradictory over-constraint, constraint
flooding, numeric boundary values, empty/degenerate input, and plausible
but catalog-absent vocabulary — every one built against the catalog's own
real vocabulary. `eval.runners.simulation` runs this generator at scale.
"""
