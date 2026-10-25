# KERDOIOS

**Find the advantage.**

Kerdoios is a standalone [Hermes](https://github.com/NousResearch/hermes-agent) plugin that allots heterogeneous compute. Hermes still orchestrates, verifies, and retries. Kerdoios only answers: given this work, these offers, and these constraints, **what portfolio should run?**

It is not an LLM router that picks one endpoint. It builds an **execution portfolio**.

Cost is one dimension of advantage — not the project. Depending on the task, the advantageous route might optimize for cost, capability, latency, throughput, privacy, reliability, available capacity, or some combination. Modes today: `FREE`, `CHEAP`, `BALANCED`, `FAST`, `MAX`, `SCALE`, `PRIVATE`.

The MVP is LLM placement. The longer arc is any scarce computational resource that can be discovered, filtered, and combined.

<p align="center">
  <img alt="The Shrine of Kerdoios. Waterways converge on a mountain gate. The stone reads: POWER WAS NEVER SCARCE. ONLY SCATTERED." src="docs/art/shrine-the-scattered-rivers.png" width="100%">
</p>

---

**POWER WAS NEVER SCARCE. ONLY SCATTERED.**

A city once needed more power than any known river could give.

Kerdoios found no greater source. He walked into the mountains. He reopened forgotten aqueducts, redirected abandoned streams, restored old wells, borrowed excess flow, and released reservoirs only when the work actually required them.

For days, thousands of insignificant currents moved across the landscape. Then the water rose. Streams became rivers. Rivers became a torrent.

For one night the city possessed **the strength of an ocean without owning a single sea.**

That night is called the First Convergence.

The full story — the Rivers, the shrine, the glyph, the chapters — lives in [docs/lore.md](docs/lore.md).

**The rivers are compute.**

The world is already crossed by free quotas, idle local GPUs, promotional credits, rate-limited inference, cheap models, and expensive frontier ones. Most systems see incompatible scraps. Kerdoios finds a path through them. It does not create the water. It notices what others overlook, connects it, and uses it when that is advantageous.

---

## Upstream first

Do not grow a second LLM gateway here. The popular OSS project that already routes by cost, tracks spend, and speaks 100+ providers is [LiteLLM](https://github.com/BerriAI/litellm). See [UPSTREAM.md](UPSTREAM.md): small LiteLLM PRs first, then a slice of [quota pools #31823](https://github.com/BerriAI/litellm/issues/31823). Kerdoios keeps the Hermes/AODL **portfolio** planner (N workers, perishable quota, privacy hard-filter). Hermes still executes.

Closest-looking repos that are *not* popular enough to join instead: `ypollak2/llm-router` (78★) and `malda231125/free-llm-gateway` (7★). Steal ideas; do not fork them.

## Not Hermes core

Install into `~/.hermes/plugins/kerdoios/`. Do **not** PR this into `NousResearch/hermes-agent`. Vendor/compute plugins stay standalone. AODL remains the typed work spec; Kerdoios consumes a `WorkRequirement` JSON and returns an `ExecutionPlan` JSON.

## Install

```bash
git clone https://github.com/kvnloo/kerdoios.git ~/.hermes/plugins/kerdoios
hermes plugins doctor ~/.hermes/plugins/kerdoios --ci
hermes plugins enable kerdoios
```

`plugin.yaml` stays at the plugin root so Hermes can load the directory. The importable package is `kerdoios/` (plus `pyproject.toml`), so a checkout does not need to be named `kerdoios` and plugin cwd does not shadow stdlib `types`. Optional editable install from any folder name:

```bash
pip install -e .
```

The critical-path seed is **public free models**. No API keys are required:

```bash
python3 -m kerdoios inventory --free
```

`--free` queries OpenRouter's public `/models` catalog and keeps `:free` ids, remaining quota/credits that were actually populated, and local endpoints. Missing, `None`, or default-0 catalog prices are unknown, not free — including LiteLLM's `input_cost_per_token == 0` rows (missing prices, rerank, embeddings). It does **not** mix the 7-row demo fixture. If the network is empty, Kerdoios loads `kerdoios/providers/openrouter_free.snapshot.json`. When `GROQ_API_KEY` / `CEREBRAS_API_KEY` are set, keyed free-tier chat rows from those catalogs are overlaid; missing keys stay skipped.

Live adapters without the free filter overlay the full OpenRouter catalog on the fixture:

```bash
python3 -m kerdoios inventory --live
python3 -m kerdoios plan --live --workers 100 --budget 0.50 --mode cheap
```

`--live` queries OpenRouter's public `/models` catalog, any OpenAI-compatible local endpoint (`KERDOIOS_LOCAL_BASE_URL`, then `:11434` / `:1234`), and Groq/Cerebras only when those env keys already exist. Missing keys or a down network return an empty adapter result; planning still uses the fixture catalog. Do not ingest Tailscale or portal tokens into git.

See [ROADMAP.md](ROADMAP.md) for what is done, in progress, and out of MVP.

## CLI

```bash
python3 -m kerdoios inventory --free
python3 -m kerdoios inventory
python3 -m kerdoios inventory --live
python3 -m kerdoios plan --workers 100 --context 128000 --budget 0.50 --mode cheap
python3 -m kerdoios explain --workers 100 --budget 0.50
```

Once loaded by Hermes:

```text
hermes kerdoios inventory
hermes kerdoios plan
hermes kerdoios explain
```

## Model-routing objective (O14)

`kerdoios.routing_objective.plan_routing(offers, requirement, metrics,
lambda_value=1, lambda_evidence=...)` owns the approved single-call objective:

```text
latency_u = TTFT_u + out_tokens_t / tok/s_u
pace_u = burn_u * tokens_t / (remaining_u / time_to_reset_u)
score_u = latency_u * (1 + λ * pace_u) / p_accept[u, task_class(t)]
```

Minimize among offers that pass the existing capability, privacy, context, tool
and reliability hard filters. Source: OMP session
`01a16527-824c-7714-8205-9a2d0d5eff38`, ordinal 4409, 2026-10-24 03:32:22Z;
Kevin confirmed the agreement on 2026-10-24. The default lambda is 1 and
**unmeasured**, not an empirical calibration; other values require measured
source and timestamp. TTFT is seconds, throughput output tokens/second, reset
horizon hours; preserve the burn/balance quota unit rather than silently mixing
credits, tokens or account fractions.

Unknown TTFT, throughput, burn, balance, reset or task acceptance produces an
`unscored` entry, never a guessed zero. Task-class verified acceptance may update
the benchmark prior through the existing blend weight only after five
observations. HTTP completion and arbitrary benchmark scores are not acceptance
probabilities. The existing portfolio objectives are unchanged. z0intelligence's
compiled O14 policy enforces this output and owns quota admission and receipts;
Kerdoios does not become another gateway.

`observed.record_routing_attempt` records real transport status, measured token
counts and numeric/ISO quota headers in the existing observation ledger. Its
`routing_quota` records are deliberately excluded from task-success aggregates.
The model-routing catalog can capture the first actual OpenCode Go/DeepSeek 429
and any explicit daily reset. Partial logs do not prove a daily cap, and missing
account/window attribution stays unknown. No synthetic request is sent to reach
a limit.

## Architecture

```text
Hermes  →  AODL WorkRequirement  →  Kerdoios
                                      ├ inventory (ResourceOffer[])
                                      ├ hard-filter
                                      ├ effective cost + perishable quota
                                      ├ Pareto set
                                      ├ portfolio (N workers)
                                      └ ExecutionPlan + explain
Hermes executes. Kerdoios does not.
```

Small springs, private waterwheels, and expensive reservoir gates are the same diagram: free quota, local inference, and paid overflow. Kerdoios allots the portion. Hermes spends it. The budget is the cut.

## MVP claim

Hermes can execute an LLM workload more cheaply than naive single-provider routing without a material drop in verified success, by splitting workers across free quota, local inference, and a paid overflow slice — subject to hard constraints (context, tools, privacy).

Success target (when multiple viable providers exist):

- ≥ 50% monetary cost reduction vs naive paid-only
- ≥ 95% of baseline verified task success

Out of MVP: VM provisioning, GPU marketplaces, account creation, promotion scraping, RL, a public compute atlas writer.

## Contribute

[CONTRIBUTING.md](CONTRIBUTING.md). Workers open PRs. They never merge `main`.

## License

MIT
