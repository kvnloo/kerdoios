# Upstream first

Kerdoios is not a replacement for an LLM gateway.

The popular OSS project that already owns **per-request cost routing,
fallbacks, spend tracking, and 100+ provider adapters** is
[LiteLLM](https://github.com/BerriAI/litellm) (~58k stars). Contribute
there for commodity routing. Keep this repo for the piece LiteLLM does
not do: **workload-level execution portfolios** for Hermes/AODL
(N workers across perishable free quota, local GPUs, and paid overflow).

Do not send per-turn model routing into `NousResearch/hermes-agent`.
Hermes maintainers closed that under the `delegation-model-routing`
policy ([#41190](https://github.com/NousResearch/hermes-agent/issues/41190),
[#56655](https://github.com/NousResearch/hermes-agent/issues/56655)).

## What already exists

| Project | Stars | What it actually does | Relation to Kerdoios |
| --- | ---: | --- | --- |
| [BerriAI/litellm](https://github.com/BerriAI/litellm) | ~58k | Gateway: cost/latency/usage routing, budgets, RPM/TPM, fallbacks | **Contribute here.** Inventory + execute LLM offers. Open wedge: [quota pools #31823](https://github.com/BerriAI/litellm/issues/31823). |
| [Portkey-AI/gateway](https://github.com/Portkey-AI/gateway) | ~13k | MIT gateway, fallback / loadbalance / conditional | Second gravity well. Use if a TypeScript contribution is easier; do not duplicate both. |
| [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) | ~5.5k | Learned strong/weak model cascade | Capability scoring research, not a portfolio planner. |
| [weave-os/router](https://github.com/weave-os/router) | ~4.3k | Productized agentic prompt router | Watch; license is not a clean MIT/Apache contribution target. |
| [ypollak2/llm-router](https://github.com/ypollak2/llm-router) | 78 | Free-first coding-tool router | Closest *story*. Too small to be "the" upstream. |
| [malda231125/free-llm-gateway](https://github.com/malda231125/free-llm-gateway) | 7 | Groq/Cerebras/OpenRouter quota tracking | Closest *quota* implementation. Not popular. Steal ideas, do not fork. |

None of the popular repos construct a **100-worker portfolio** under a
budget. They pick **one endpoint per request**. That is the Kerdoios
delta. Reimplementing LiteLLM's provider catalog inside Kerdoios is the
trap.

## Contribution ladder (LiteLLM)

Sign the [LiteLLM CLA](https://cla-assistant.io/BerriAI/litellm) before
the first PR. Isolated scope, one mocked test, `make lint`.

1. **Small, unopinionated**
   - Pricing-table fixes for Groq / Cerebras / OpenRouter `:free`
   - Do **not** treat LiteLLM `input_cost_per_token == 0` as a free chat
     tier (~150 rows are missing prices, rerank, or embeddings). Fix those
     rows upstream; Kerdoios must not bulk-ingest them as `--free`.
   - Map remaining-quota / retry-after headers into router state
   - `insufficient_quota` vs retryable `429` (several prior PRs; finish a still-open slice)
   - Docs: `cost-based-routing` + provider budget examples for free tiers
2. **Medium, still LiteLLM-shaped**
   - Comment on and implement a *slice* of
     [#31823 provider quota pools](https://github.com/BerriAI/litellm/issues/31823)
     (multi-key packages with independent windows). That is perishable
     inventory in their vocabulary.
3. **Larger / opinionated (only after 1–2 merged PRs)**
   - Expiration urgency (credits that die tomorrow ≠ permanent $0 local)
   - Capability-aware eligibility before cost ranking
   - Optional: emit a Kerdoios `ResourceOffer[]` from LiteLLM deployments

## What Kerdoios keeps

- `WorkRequirement` / `ExecutionPlan` / `explain`
- Hard constraints (privacy, context, tools) that money cannot override
- Portfolio split across offers for `parallelism = N`
- Later: GPU / VM / Mesh offers LiteLLM will never model

Hermes should eventually **execute through LiteLLM** (or an
OpenAI-compatible proxy). Kerdoios should **plan**, then hand placements
back. Do not grow a second 100-provider SDK here.
