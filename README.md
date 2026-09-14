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

`run` executes a plan through your own worker command with self-healing —
per-task isolation, retry with backoff, and failover through each
placement's fallback chain. Kerdoios never calls a model API itself; the
worker does, and reports the outcome on the worker contract
(see `kerdoios/heal.py`):

```bash
python3 -m kerdoios plan --workers 8 --mode cheap > plan.json
python3 -m kerdoios run --plan plan.json --tasks tasks.jsonl \
  --worker-cmd "python3 my_worker.py --model {model}"
```

Every attempt is auto-recorded with a failure reason
(`rate_limited|timeout|upstream_5xx|empty_response|worker_crash`), so the
next `--observed` plan routes around sick models — recent failures are
penalized, old ones decay away instead of blacklisting a model forever.

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
