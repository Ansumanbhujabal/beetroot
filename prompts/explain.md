---
id: explain
version: 1
stage: explain
inputs: [name, facts, satisfied]
---
Explain in two sentences why this meal suits the user.

Meal: {name}

Verified facts you may cite (computed from a trusted catalog — do not invent,
round, or alter any number):
{facts}

Constraints it satisfies: {satisfied}

Do not state any number that does not appear in the verified facts above.
