# Kerdoios roadmap

Kerdoios returns ExecutionPlan. Hermes executes. Commodity routing belongs in LiteLLM.

**Execute:** Hermes (or LiteLLM) runs the plan. Do not add an execute CLI here.

**Done:** types, hard filters, perishable quota scoring, Pareto + N-worker allocate, mode presets, CLI, Hermes tools, 7-row fixture, live adapters, free-first OpenRouter seed, persist inventory cache (#5), origin-provider attribution (#6), observed-outcome JSONL + scoring blend (#8), unit CI (#12), pyproject.toml (#14), remaining-quota headers (#15), keyed Groq/Cerebras free-tier overlay (#16), anti-slop denylist (#17), missing catalog prices unknown-not-free (#18), OpenRouter origin ids without doubled origin (#27), openai_compat missing prices unknown not 0.0 (#28), Hermes record/observed tools (#30), AODL-shaped work spec ingest (#32), **capability_id + token receipts on Observation** (z0int bridge; hierarchical lookup exact→family→model), **WorkRequirement.capability_id** for residual plans.

**Boundary with z0int:** Kerdoios does **not** execute flies/MB. z0int preflight filters cognition that needs no frontier LLM; Kerdoios allocates residual WorkRequirement only and returns ExecutionPlan. Harness executes.

**In progress:** inventory every provider/model; keep free as the initial list. OpenRouter is the aggregator. Groq/Cerebras overlay needs keys. Premium-quota *reserve* shadow price (Astra) vs free *consume* policy. Observed aggregation volume per capability_id.

**Not done:** mutation; force host model switch from plan; counterfactual ledger dashboard.


**Trap:** LiteLLM `input_cost_per_token == 0` is missing price / rerank / embedding, not a usable free chat tier. `--free` must not ingest that dump. Contribute pricing-table fixes upstream; do not grow a second gateway here.

**Out of MVP:** VM provisioning, GPU marketplaces, account creation, promotion scraping, RL, public atlas writer. Do not grow a 100-provider SDK.
