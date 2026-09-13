---
name: kerdoios
description: Use when deciding where scarce LLM compute should run: build a cost-aware execution portfolio (free quota, local inference, paid overflow) with the kerdoios CLI.
---

# Kerdoios

Kerdoios answers one question: given this work, these offers, and these constraints, **what portfolio should run?** It allots compute. It does not execute. Hermes (or your harness) executes the plan it returns.

## Run

From the repo root (`python3 -m kerdoios`), or after `pip install -e .` via the `kerdoios` entry point:

```bash
python3 -m kerdoios inventory --free        # public free models, no keys needed
python3 -m kerdoios plan --workers 8 --mode cheap
python3 -m kerdoios explain --workers 8 --mode cheap
```

Modes: `free`, `cheap`, `balanced`, `fast`, `max`, `scale`, `private`.
Common plan flags: `--budget`, `--context`, `--privacy public|confidential|local_only`,
`--coding`, `--reasoning`, `--no-tools`, `--aodl PATH`, `--live`, `--free`, `--observed`.

`plan` prints an `ExecutionPlan` JSON: the Pareto set of offers plus the per-worker
portfolio and expected cost. `explain` prints why each worker was placed there.
`record` logs an observed outcome (`--provider --model --task-type --completed --cost --retried`)
so `--observed` plans can blend in real results.

## Rules

- Missing, `None`, or default-0 catalog prices are **unknown**, not free. Never invent
  `remaining_free_quota`.
- `--free` seeds from OpenRouter's public `/models` catalog (or the bundled snapshot
  offline); it never mixes in the demo fixture.
- Free-tier keys (`GROQ_API_KEY`, `CEREBRAS_API_KEY`) are overlaid only when already set.
  Never commit secrets.
