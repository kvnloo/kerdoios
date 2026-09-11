# Kerdoios roadmap

Kerdoios plans portfolios. Hermes executes. LiteLLM owns commodity routing.

**Done:** types, hard filters, perishable quota scoring, Pareto + N-worker allocate, mode presets, CLI, Hermes tools, 7-row fixture, live adapters, free-first OpenRouter seed.

**In progress:** inventory every provider/model; keep free as the initial list. OpenRouter is the aggregator. Groq/Cerebras free tiers need keys.

**Not done:** persist inventory, live remaining-quota headers, execution receipts, AODL schema ingest, GPU/VM/mesh, execute-through-LiteLLM, pyproject.toml, unit CI, mutation.

**Trap:** LiteLLM `input_cost_per_token == 0` is missing price / rerank / embedding, not a usable free chat tier. `--free` must not ingest that dump. Contribute pricing-table fixes upstream; do not grow a second gateway here.

**Out of MVP:** VM provisioning, GPU marketplaces, account creation, promotion scraping, RL, public atlas writer. Do not grow a 100-provider SDK.
