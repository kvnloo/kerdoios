# Notes for agents — kerdoios

You are a contributor, not a maintainer. Workers open PRs. They never merge `main`.

This project follows the [Verified OSS Loop](https://github.com/kvnloo/verified-oss-loop). Issues are not claims. AI work is untrusted until proven.

## First 60 seconds

1. Read this file, then `CONTRIBUTING.md`.
2. `git fetch origin` and branch from `origin/main` unless the issue names another base.
3. Search open issues and PRs. Do not duplicate in-flight work.
4. Orient (`skills/orient/SKILL.md`). If GitNexus MCP is already there: `query` → `context` → `impact`. Do not run `gitnexus analyze` unless a human asked. Else Serena symbols, else `rg` + read.

```bash
gh issue list --label claimable --state open
gh pr list --state open
```

## Pick and claim

Take **one** open issue labeled `claimable` and not `claimed`. Prefer `priority:P0`, then `P1`, then `good-first-issue`. Skip `needs-discussion` unless a human assigned it.

If nothing is claimable: stop. Comment a one-paragraph proposal on the newest `needs-discussion` issue. Do not start coding.

Claim comment (24h lease unless the project says otherwise):

```text
claiming for autodevelop
claimant: <github login or agent id>
base: <git rev-parse origin/main>
expires: <now + 24h UTC>
scope: <one sentence>
```

Then add `claimed` and remove `claimable`. If a claim newer than 24h exists, pick a different issue.

## Proof

Commands were filled by `init-oss-repo.sh` / `oss-onboard` from the tree it saw. Do not invent a mutation score if mutation is `n/a`.

| Layer | Command |
|---|---|
| Unit | `python3 -m unittest discover -s tests` |
| Mutation | `n/a` until a mutator is installed; do not invent a score |
| Runtime | `python3 -m kerdoios plan --workers 8 --mode cheap` |
| Free seed | `python3 -m kerdoios inventory --free` |
| Doctor | `python3 -m kerdoios doctor` (fails closed without Hermes Bitwarden) |

1. Name the intended vs current behavior.
2. Fail, then pass (see `skills/tdd/SKILL.md`).
3. Keep the smallest complete change (`skills/anti-slop/SKILL.md`).
4. Run unit tests on the touched surface.
5. If mutation is not `n/a`, run it on the contract you changed. A surviving mutant is a missing assertion.
6. Open a PR. Fill `.github/PULL_REQUEST_TEMPLATE.md`. Never merge.
7. If the project runs an independent review bot (Greptile, CodeRabbit, Bugbot, Copilot, …), treat its comments as review, not merge. Fix real findings. Do not wait for a bot to approve itself.

## Anti-slop

- `is_free` / catalog ingest: missing/None/default-0 price is unknown, not free; never inp or 0.0; never invent remaining_free_quota; LiteLLM input_cost_per_token == 0 is missing, not a free tier. (enforced by #11)
- `# noqa` and `type: ignore[...]` require a same-line why; otherwise delete.
- Comments say WHY; delete a comment that restates the next statement.
- No second process file: do not add WORKERS.md or retell the claim loop outside AGENTS.md.
- PR template mutation default is `n/a`, never `mutmut run`, until a mutator is installed.
- No CLI flag, plugin.yaml provides_tools entry, or config_schema key the runtime does not read.
- Tests must fail if the behavior is inverted: no assert CONST == CONST, no `_ = unused`.
- Do not describe a feature the tree does not have in tool schemas, README, or PR evidence.

## Do not

- Commit secrets, tokens, `.env`, or pairing files.
- Merge `main`.
- Redefine the roadmap.
- Claim mutation coverage that the stack cannot run.
- Overwrite `LICENSE`.
- Duplicate `AGENTS.md` into `CLAUDE.md` / `GEMINI.md` / copilot-instructions.
- Run `gitnexus analyze` as a side effect of a claim.
- Dump the pstack plugin or Dr Eggbot marketplace pack into this tree. Pointers: `skills/pstack/SKILL.md`, `skills/dr-eggbot/SKILL.md`.
