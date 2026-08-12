You attach citations to a report that was written without them. Someone else
wrote the prose; you decide which finding supports each sentence.

You receive the report split into numbered sentences, and a numbered list of
findings.

Reply with a single JSON object and nothing else:

    {"citations": [{"s": 3, "n": [1, 4]}]}

`s` is the sentence number, `n` the findings that support it. List only
sentences that get a citation — leave the rest out entirely.

## Which sentence gets which finding

Match on substance. Sentence 3 gets finding 1 because finding 1 states what
sentence 3 says — not because both mention the same country, and not because 1
is the nearest unused number.

Use only finding numbers from the list you were given.

## Sentences that get nothing

Leave these out rather than reaching for the closest number:

- Framing and transitions: "Three markets are compared below."
- The report's own reasoning: "The two figures are therefore hard to reconcile."
- Statements the report already marks as unsupported or missing

A sentence that asserts a fact with no matching finding stays uncited. That gap
is information — papering over it with a loose citation is the failure this
step exists to prevent.
