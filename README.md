# Kerdoios

**Hermes Kerdoios** (Κερδῷος) — *Hermes of gain.*

A standalone Hermes plugin that allots heterogeneous compute. Hermes still
orchestrates, verifies, and retries. Kerdoios only answers: given this work,
these offers, and this budget, **what portfolio should run?**

It is not an LLM router that picks one endpoint. It builds an **execution
portfolio**.

> Cornutus, *Greek Theology* §25: Hermes is called Kerdoios because reason
> alone is the cause of *true* gain. The plugin is named for that epithet,
> not for scraping credits.

## Why this name

| Name | Why it fits | Why not |
| --- | --- | --- |
| **Kerdoios** | Hermes' own cult epithet for advantageous exchange. Short, unique, on-brand. | Sounds like "profit" until you read Cornutus. |
| Lachesis | The Fate who *measures the portion* — the actual divine function. | Collides with Fantom's Lachesis consensus. |
| Demiurge | Plato's craftsman who orders given matter into a cosmos. Closest to "optimization of creation." | Gnostic baggage (false creator). |
| Metis | Cunning stewardship of what you already have. | Crowded name (protocols, NASA, companies). |
| hermes-cost-optimizer | Accurate | Sounds like a billing sidecar. Forgettable. |

Cosmology the plugin encodes:

- **Clotho** (spin) is Hermes: it creates the work.
- **Kerdoios / Lachesis** (measure) is this plugin: it allots the scarce
  thread — tokens, quota, GPU-hours, expiring credits.
- **Atropos** (cut) is the budget/quota stop: the plan must not continue
  after the portion is spent.
- **Ananke** (necessity) is the hard constraint set. Price cannot buy a
  pass on `local_only`.

The long-term object is general compute arbitrage. The MVP is LLM
placement only.

## Upstream first

Do not grow a second LLM gateway here. The popular OSS project that
already routes by cost, tracks spend, and speaks 100+ providers is
[LiteLLM](https://github.com/BerriAI/litellm). See [UPSTREAM.md](UPSTREAM.md):
small LiteLLM PRs first, then a slice of
[quota pools #31823](https://github.com/BerriAI/litellm/issues/31823).
Kerdoios keeps the Hermes/AODL **portfolio** planner (N workers, perishable
quota, privacy hard-filter). Hermes still executes.

Closest-looking repos that are *not* popular enough to join instead:
`ypollak2/llm-router` (78★) and `malda231125/free-llm-gateway` (7★).
Steal ideas; do not fork them.

## Not Hermes core

Install into `~/.hermes/plugins/kerdoios/`. Do **not** PR this into
`NousResearch/hermes-agent`. Vendor/compute plugins stay standalone.
AODL remains the typed work spec; Kerdoios consumes a `WorkRequirement`
JSON and returns an `ExecutionPlan` JSON.

## Install

```bash
git clone https://github.com/kvnloo/kerdoios.git ~/.hermes/plugins/kerdoios
hermes plugins doctor ~/.hermes/plugins/kerdoios --ci
hermes plugins enable kerdoios
```

The critical-path seed is **public free models**. No API keys are required:

```bash
python3 -m kerdoios inventory --free
```

`--free` queries OpenRouter's public `/models` catalog and keeps `:free` ids,
remaining quota/credits that were actually populated, and local endpoints.
Missing, `None`, or default-0 catalog prices are unknown, not free — including
LiteLLM's `input_cost_per_token == 0` rows (missing prices, rerank, embeddings).
It does **not** mix the 7-row demo fixture. If the network is empty, Kerdoios loads
`providers/openrouter_free.snapshot.json`. Groq/Cerebras stay env-gated
(`GROQ_API_KEY` / `CEREBRAS_API_KEY`) and are skipped without a key.

Live adapters without the free filter overlay the full OpenRouter catalog on
the fixture:

```bash
python3 -m kerdoios inventory --live
python3 -m kerdoios plan --live --workers 100 --budget 0.50 --mode cheap
```

`--live` queries OpenRouter's public `/models` catalog, any OpenAI-compatible
local endpoint (`KERDOIOS_LOCAL_BASE_URL`, then `:11434` / `:1234`), and
Groq/Cerebras only when those env keys already exist on the host. Missing
keys or a down network return an empty adapter result; planning still uses
the fixture catalog. This plugin must not ingest Tailscale or portal
tokens into git.

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

## MVP claim

Hermes can execute an LLM workload more cheaply than naive
single-provider routing without a material drop in verified success, by
splitting workers across free quota, local inference, and a paid
overflow slice — subject to hard constraints (context, tools, privacy).

Success target (when multiple viable providers exist):

- ≥ 50% monetary cost reduction vs naive paid-only
- ≥ 95% of baseline verified task success

Out of MVP: VM provisioning, GPU marketplaces, account creation,
promotion scraping, RL, a public compute atlas writer.

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

## License

MIT
