---
name: verify
description: Use after implementing a claim: verify with unit tests, red/green evidence receipt, and the runtime check; fail closed on what you did not run.
---

# Verify

Tests are necessary, not sufficient. Generation and verification are separate.

## Pyramid

1. **Unit** — `python3 -m unittest discover -s tests` on the touched surface.
2. **TDD** — red command, then green command (`skills/tdd/SKILL.md`).
3. **Mutation** — `n/a` until a mutator is installed. A surviving mutant is a missing assertion. If the stack has no mutator, write `n/a`. Do not invent a score.
4. **Runtime** — `python3 -m kerdoios plan --workers 8 --mode cheap`. If the project has no runtime check, write `n/a` and say what you did not run.

## Receipt

Bind every result to `head_revision`. Tests from another SHA are not evidence. Fill `.github/PULL_REQUEST_TEMPLATE.md`.

The implementer does not self-approve. Independent review is a different person or a frozen evaluator. Workers never merge.

## Fail closed

- Unknown mutation tool → `n/a`, not `80`.
- Runtime you did not exercise → list it under `limitations`.
- Secrets, tokens, `.env` → stop.
