---
id: explain_choice
name: Explain Choice
tier: T2
llm_permitted: true
triggers_on: ["after TRUST"]
priority: 60
---

## When to use

Last step before VERIFY, after a single candidate has been ranked, its
nutrition computed, and its trust score has cleared the gate. This is the
only skill in the pipeline whose entire job is prose — by the time it runs,
the decision itself is already made and already grounded in verified data.
`explain_choice` cannot change what was chosen; it can only fail to explain
it well.

## The pattern

1. Assemble the "verified facts" block from `compute_nutrition`'s output
   only. Every number the model is allowed to reference is INJECTED into
   the prompt as text ahead of time — never looked up, never computed, by
   the model itself.
2. Call the model with the explain prompt (`prompts/explain.md`), which
   states outright which constraints the candidate satisfies and instructs
   the model not to state any number absent from the injected facts.
3. Constrain the ask tightly: two sentences, prose only, no new claims about
   nutrition, safety, or constraint satisfaction beyond what was handed in.
4. Re-check the model's output against those same facts — the VERIFY state
   runs immediately after this skill, diffing every number the model wrote
   against catalog truth via the drift ledger. This skill's output is a
   draft until VERIFY passes it, not a final answer the moment it's
   generated.
5. On a verification failure — an invented number, an unsupported claim —
   the pipeline moves to ESCALATE, not back to this skill for a
   retry-and-hope. A model that drifted from its own inputs once in a turn
   is not trusted to self-correct within that same turn.

## Pitfalls

- **Letting the model compute or round a number itself.** "Roughly 30g of
  protein" from a model that was handed `protein_g: 31.4` is still drift —
  the prompt says don't invent, round, or alter any number, and VERIFY
  enforces that with an actual diff against catalog truth, not a vibe
  check on whether the sentence sounds plausible.
- **Treating this skill as a second chance to reconsider the choice.**
  `explain_choice` runs after ranking (folded into `retrieve_candidates`'s
  model-scored rerank step) and after TRUST has cleared the candidate; it
  has no authority to swap in a different candidate because the chosen one
  "explains awkwardly." If
  an explanation doesn't work, that's feedback for the rerank prompt or
  the trust weights, not license for this skill to overrule the pipeline
  in the moment.
- **Widening the prompt's inputs "for a richer explanation."** Every
  additional field handed to the model here is one more thing it could
  drift on. The prompt is deliberately narrow — name, verified facts,
  satisfied constraints — and that narrowness is exactly what keeps the
  post-hoc verification cheap and complete rather than open-ended.
- **Skipping VERIFY because the prompt "already told it not to invent
  numbers."** A prompt instruction is a request, not a guarantee — models
  comply with "don't invent numbers" at a high rate, not at 100%, and the
  entire reason a verification stage exists downstream of an LLM call is
  that 100% is the bar for anything a user might act on.
