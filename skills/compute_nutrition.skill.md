---
id: compute_nutrition
name: Compute Nutrition
tier: T0
llm_permitted: false
triggers_on: ["after RETRIEVE"]
priority: 40
---

## When to use

For every candidate that survives retrieval, before trust is scored and
before any explanation is generated. This skill produces the one and only
`NutritionFacts` a downstream skill is ever allowed to cite — its
`provenance` field is typed `Literal["computed"]`, so there is no
constructor path anywhere in this system that can build a `NutritionFacts`
from a model's guess.

## The pattern

1. Sum per-ingredient contributions from the trusted catalog's `per_100g`
   values, scaled by the recipe's own gram amounts, across six fields:
   kcal, protein_g, carbs_g, fat_g, sodium_mg, fibre_g. Pure arithmetic, no
   interpretation.
2. Track two running totals per recipe: `total_mass` (grams across every
   ingredient) and `covered_mass` (grams of ingredient whose catalog entry
   actually carried usable data for that field set).
3. `coverage` is `covered_mass / total_mass` — mass-weighted, not a count of
   ingredients with data over a count of ingredients. An ingredient with
   partial data (say, kcal and protein known, sodium missing) contributes
   its grams at `present_fields / total_fields`, never zero and never full
   credit.
4. Reject, don't silently zero, malformed inputs: a negative or NaN gram
   amount raises `MalformedRecipeError` rather than being treated as a
   zero contribution — a recipe with corrupt ingredient data should fail
   loudly, not quietly under-report itself as merely low-coverage.
5. The resulting `coverage` field is what `assess_trust` reads for the
   `catalog_coverage` signal (0.45 of the composite) — this skill IS that
   signal's source, not a decoration that happens to sit next to it.

## Pitfalls

- **Weighting coverage by ingredient count instead of mass.** A recipe
  missing data for a 1g pinch of chili powder and one missing data for its
  200g chicken breast are not equally uncertain, and a per-ingredient
  count scores them identically. Mass-weighting exists specifically because
  a per-ingredient scheme produces confident-looking coverage numbers for a
  recipe whose main protein is actually unverified.
- **Letting the model estimate a missing value.** This is the exact failure
  this skill was built to close off: model-generated nutrition figures ran
  roughly 1.5x high in production. `NutritionFacts.provenance` being pinned
  to `"computed"` in the type system isn't decoration — there is no
  constructor path that accepts a model-supplied number, on purpose.
- **Treating `field in per_100g` as "the field is usable."**
  `per_100g.protein_g: null` passes a naive presence check, earns full
  coverage credit, and then crashes the moment it's cast to float. A field
  only counts as usable if it's a real, finite, non-negative number — the
  null case has to be caught explicitly, never assumed away by a `in` check.
- **Assuming a recipe's `ingredients[].grams` is already single-serving.**
  It isn't guaranteed to be by construction — a real defect corrected
  across 86 recipes had `grams` reflecting batch yield rather than a single
  portion, which silently inflates every downstream total (kcal, protein,
  cost) by whatever the batch multiple was. This skill trusts `grams` at
  face value; that assumption is the catalog's to guarantee, but any change
  upstream to how portions are authored deserves an audit of every value
  this skill has ever produced from that data.
