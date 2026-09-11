# Workers

This project follows `AGENTS.md` and the [Verified OSS Loop](https://github.com/kvnloo/verified-oss-loop).

## How to contribute

1. Read `AGENTS.md` and `CONTRIBUTING.md`.
2. Claim **one** open issue labeled `claimable` (not `claimed`). Prefer `priority:P0`, then `P1`, then `good-first-issue`.
3. Post the 24h claim comment, add `claimed`, and remove `claimable`.
4. Branch from `origin/main`. Fail, then pass. Fill the PR evidence template.

Workers open PRs. **They never merge.** Kevin tests locally and merges when the receipt is good.

```bash
gh issue list --label claimable --state open
gh pr list --state open
```
