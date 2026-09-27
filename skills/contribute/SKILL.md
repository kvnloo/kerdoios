# Contribute

You forked or cloned Kerdoios in a coding harness (Cursor, Codex, Claude Code, Hermes). This is the whole first pass. Do not invent a second loop.

**Find the advantage. POWER WAS NEVER SCARCE. ONLY SCATTERED.**

Kerdoios plans. Hermes executes. LiteLLM routes commodity.

## 90 seconds

1. Confirm this tree: `plugin.yaml` at the repo root and an importable `kerdoios/` package.
2. Green before you edit:

```bash
python3 -m unittest discover -s tests
python3 -m kerdoios plan --workers 8 --mode cheap
```

3. `git fetch` **origin main of kvnloo/kerdoios** if your copy might be stale. Read `ROADMAP.md` from that tip, not a leftover local draft.
4. Search open PRs on https://github.com/kvnloo/kerdoios. Stop if the leftover is already in flight.
5. One leftover. One branch from `origin/main` (or your fork's default that tracks it). Fail, then pass (`skills/tdd/SKILL.md`). Smallest complete change (`skills/anti-slop/SKILL.md`).
6. Open a PR. Fill `.github/PULL_REQUEST_TEMPLATE.md`. **Never merge `main`.**

## Hard no

- Never merge `main`.
- Never run an empty-queue dispatcher or timer polling.
- Never `POST` `/chat/completions` (or any execute CLI) as a gateway. Kerdoios returns `ExecutionPlan` only.
- Never treat LiteLLM `input_cost_per_token == 0` as a free chat tier. Missing price is unknown.
- Do not grow a second 100-provider SDK. Upstream pricing-table fixes belong in LiteLLM.

## Claiming kvnloo issues

If you have write access on `kvnloo/kerdoios`, the claim lease in root `AGENTS.md` still applies. This skill is the fork/harness on-ramp, not a second process.
