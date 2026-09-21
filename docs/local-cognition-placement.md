# Local cognition placement

Resource-aware placement for the local cognition SLMs (JEV / tiny specialists /
orchestrator / general SLM). This document specifies the question Kerdoios answers for
those models, the inputs it reads, the inspectable recommendation it returns, and the
hard constraints that bound it.

## What this decides — and what it does not

Given an **already-selected capability/role** and the **current machine state**, Kerdoios
answers only:

1. can this model run locally **now**;
2. which **runtime/quant** fits;
3. should it be **co-resident** with what is already loaded?

**Kerdoios does NOT decide semantic suitability.** It does not choose which
capability or role the turn needs, which model should default to a role, or whether a
model's tool-calling behaviour is good enough. Those are z0intelligence decisions
(`manifests/local_cognition.v1.json`, `z0int cognition roles --json`) and remain there.
Kerdoios receives the selected capability and the candidate models for it, and returns a
placement. Semantic selection and placement stay separate on purpose — a placement result
must never be read as a promotion or as an endorsement of a model's quality.

Kerdoios also does not execute. Hermes (or the configured runtime) executes.

## Machine profile: `rtx3080ti-12gb` (first profile)

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12288 MiB |
| CPU | Intel Core i9-10900KF @ 3.70 GHz |
| Driver / CUDA | 610.57.04 / 13.3 |
| Runtime | llama.cpp 0.4.1-dev |
| Context | 4096 |
| Measurement date | 2026-09-21 |
| Receipt format | `z0int.serving_receipt.v1` |

### Measured serving table

**Every VRAM and latency value in this table is measured on the machine above, not
estimated.** They are not vendor figures and not interpolated between quantizations.

| model | quant | ctx | VRAM idle MiB | VRAM peak MiB | cold load ms | TTFT ms | decode tok/s | short decision ms | long decision ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| functiongemma_270m | Q8_0 | 4096 | 2223 | 2241 | 46 | 8.4 | 360.0 | 124 | 100 |
| hammer2.1_3b | Q4_K_M | 4096 | 3939 | 3957 | 66 | 5.6 | 281.6 | 137 | 214 |
| hammer2.1_7b | Q4_K_M | 4096 | 5994 | 6014 | 67 | 9.3 | 216.9 | 211 | 333 |
| nemotron_orchestrator_8b | Q4_K_M | 4096 | 7007 | 7023 | 122 | 9.7 | 113.7 | 733 | 855 |
| qwen3.5_9b | Q4_K_M | 4096 | 7372 | 7406 | 306 | 62.6 | 83.9 | 1275 | 1185 |

"Short / long decision ms" is end-to-end time for a realistic bounded choice over **5 legal
actions** with a short vs. a long situation prompt. All five fit at 4096 context; **none
were co-resident** while measuring, and precision beyond the values shown lives in the
receipts.

Two caveats that change placement, both measured:

- **Runtime is a placement variable.** The same FunctionGemma Q8_0 checkpoint decoded at
  **360 tok/s** through llama.cpp 0.4.1-dev and **10.5 tok/s** through Ollama 0.33.2 on the
  same GPU — a **~34x** difference for identical weights. A placement that names a model and
  a quant but not the runtime is incomplete, and runtime performance must be measured, not
  assumed from the weights.
- **A table entry without a runtime is not a placement.** `openjev_4b` (Qwen3.5-4B) carries
  declared `min_vram_gb` but **no local measurement yet**; it may only be placed by fallback
  to the declared constraint, and the placement must say so.

## Inputs

| Input | Source |
| --- | --- |
| Already-selected capability/role | z0intelligence (`z0int cognition roles --json`) |
| Candidate models for that role | z0intelligence (`z0int cognition manifest --json`) |
| Roles, licence, gated access, commercial flag | same manifest — carried through as constraints, not decided here |
| Declared `min_vram_gb`, dtype, `never_coreside` | `manifests/models.z0int.json` policies |
| Measured VRAM/latency per (model, runtime, quant, ctx) | `z0int.serving_receipt.v1` receipts |
| GPU model, total VRAM, driver, CUDA | machine probe |
| Free VRAM **now** and currently resident models | machine probe + serving registry |
| Served endpoints (`~/.z0int/config/serving.json`, `Z0INT_LOCAL_BASE_URL`) | z0intelligence serving map |
| Requested context, concurrency, latency budget class | caller |

## Inspectable placement recommendation (output)

A placement result is inspectable and must state every field below. An absent field is a
bug, not an implicit default.

| Field | Meaning |
| --- | --- |
| `candidate` | the model id offered for placement (already selected upstream) |
| `runtime` | `llama.cpp` / `ollama` / `vllm` / `transformers`, from the measured set |
| `quantization` | e.g. `Q4_K_M`, `Q8_0`, or `bfloat16` |
| `device` | `cuda:0`, `cpu`, … |
| `action` | `reuse` / `load` / `evict` / `defer` |
| `context_cap` | the context this placement will actually run at (not the advertised maximum) |
| `concurrency_cap` | how many workers may share this placement |
| `expected_latency_class` | derived from the **measured** decision ms for this (model, runtime, quant, ctx) |
| `constraint_violations` | explicit list — `[]` means nothing was violated |
| `fallback_candidate` | the next model/runtime/quant to try, or `none` with a reason |
| `receipt` | machine profile id, measurement ids used, and the inputs the decision was made from |

`action` semantics:

- **reuse** — the model is already resident on a serving endpoint; return the endpoint and
  do not reload. Preferred whenever the same (model, runtime, quant, ctx) is already up.
- **load** — resident, but not this (model, runtime, quant, ctx); free VRAM is sufficient
  for this placement plus what must stay resident.
- **evict** — loading requires unloading a named resident model first.
- **defer** — no placement satisfies the constraints now; return the reason and the fallback.

## Hard constraints

### Declared never-co-reside pairs (`manifests/models.z0int.json`, `policies.twelve_gb`)

These pairs must never be resident together on a 12 GB machine. They are declared policy,
not measured co-residency:

| Pair | Declared |
| --- | --- |
| `nanojev_06b` | `openjev_4b` |
| `nanojev_06b` | `openjev_06b` |
| `nanojev_06b` | `system_one_4b` |
| `openjev_06b` | `openjev_4b` |
| `openjev_06b` | `system_one_4b` |
| `decider_2b` | `system_one_4b` |

In policy terms: `nanojev_06b` must not co-reside with `openjev_4b`, `openjev_06b` or
`system_one_4b`; `openjev_06b` must not co-reside with `openjev_4b` or `system_one_4b`.

### New observation: one 7B–9B class model at a time on 12 GB

Derived from the measured peaks above at context 4096: every pair drawn from
`hammer2.1_7b`, `nemotron_orchestrator_8b` and `qwen3.5_9b` exceeds 12288 MiB.

| Pair (peak MiB) | Sum MiB | 12 GB |
| --- | ---: | --- |
| hammer2.1_7b (6014) + nemotron_orchestrator_8b (7023) | 13037 | does not fit |
| hammer2.1_7b (6014) + qwen3.5_9b (7406) | 13420 | does not fit |
| nemotron_orchestrator_8b (7023) + qwen3.5_9b (7406) | 14429 | does not fit |

So on 12 GB **only one 7B–9B class model is resident at a time.** Any placement that asks
for two of them must return `evict` or `defer`, never `load`.

Measured peak headroom for reference (12288 − peak): functiongemma_270m **10047 MiB**,
hammer2.1_3b **8331 MiB**, hammer2.1_7b **6274 MiB**, nemotron_orchestrator_8b **5265 MiB**,
qwen3.5_9b **4882 MiB**.

The smaller models can co-reside with one larger model at these numbers — e.g.
`hammer2.1_3b` (3957) + `qwen3.5_9b` (7406) = 11363 MiB. That is a measurement at
context 4096, not a guarantee: KV cache grows with context, peaks were taken with one model
resident, and a second endpoint adds its own runtime overhead. Treat measured peak as a
floor for the placement check, and re-measure before promising co-residency at a larger
context.

### Cap constraints

- **Context cap.** Place at the tested context (4096 for all rows above), not the advertised
  maximum. A model's advertised context is not a 12 GB budget.
- **Concurrency cap.** The measured table is single-stream; a placement must declare its
  concurrency cap rather than inheriting one.
- **Device.** `bfloat16` placements (decision-backend entries such as the JEV roster) are
  declared `cuda`-only where noted; placement may not silently switch dtype or device.

## Cold-load vs warm-resident cost

The same receipt that gives peak VRAM gives the cost of an eviction cycle. Cold load is the
time to bring the weights up from disk; TTFT is the first token with the model already
resident. Both are measured here.

| model | cold load ms | warm TTFT ms | warm short decision ms | cold load as % of one short decision | first call (cold + decision) vs warm |
| --- | ---: | ---: | ---: | ---: | ---: |
| functiongemma_270m | 46 | 8.4 | 124 | ~37% | 170 ms vs 124 ms |
| hammer2.1_3b | 66 | 5.6 | 137 | ~48% | 203 ms vs 137 ms |
| hammer2.1_7b | 67 | 9.3 | 211 | ~32% | 278 ms vs 211 ms |
| nemotron_orchestrator_8b | 122 | 9.7 | 733 | ~17% | 855 ms vs 733 ms |
| qwen3.5_9b | 306 | 62.6 | 1275 | ~24% | 1581 ms vs 1275 ms |

Reading, from these numbers only:

- **Reuse beats reload for the small models.** For `hammer2.1_3b`, a cold load costs ~48% of
  a warm decision — thrashing the small tiers roughly doubles their latency. Keep the tiny
  specialist resident when it is used repeatedly.
- **The large models amortize a load better.** For the 8B orchestrator a cold load is ~17% of
  one warm decision, so evicting it to make room and reloading later is comparatively cheap
  if it is not called every turn.
- **Warm TTFT is small next to a decision for every row** (8.4–62.6 ms against 124–1275 ms),
  so the placement's latency class is dominated by decode/decision time, not by the first
  token once the model is resident.
- **A swap is not free even when it fits.** Loading a 7B–9B model means evicting whichever
  one is resident (the exclusivity observation above), so a `load` action for one of them is
  always an `evict + load` in wall-clock terms: for `qwen3.5_9b`, 306 ms of cold load before
  any decision is made.

## Boundaries

- **Not here:** which capability the turn needs, which model is best for a role, promotion,
  serving selection, licence policy, tool-calling quality. Those are z0intelligence.
- **Not here:** execution. Kerdoios returns a placement; Hermes executes it.
- **Not here:** reranking or scoring of models beyond hard resource fit and licence/flags
  passed through as constraints.
- **External evidence** for the models themselves is frontier-kb, not this document.
