# The lore of Kerdoios

**Find the advantage.**

This is the canonical story, cosmology, and visual world for Kerdoios. The [README](../README.md) is what a developer needs to run the software. Do not reprint these slogans on every page; they work because they are rare.

The plugin is named for Kerdoios (Κερδῷος), an epithet of Hermes associated with gain and advantage. Lucius Annaeus Cornutus, *Greek Theology* §25: Hermes is called Kerdoios because reason alone is the cause of *true* gain. The broader idea of *kerdos* is gain, advantage, cleverness, and recognizing an opportunity others overlook.

That maps onto the software. The world already contains enormous amounts of fragmented compute. Most systems see incompatible, individually constrained resources. Kerdoios sees opportunity. Given work and a set of constraints, it finds an advantageous path through the resources available to it.

It does not create the water. It does not execute the work. Hermes still orchestrates, verifies, and retries. Kerdoios only answers: given this work, these offers, and these constraints, **what portfolio should run?**

<p align="center">
  <img alt="A dock at the Shrine of Kerdoios. A red banner bears the convergence glyph; waterways meet at the mountain." src="art/shrine-from-the-dock.png" width="100%">
</p>

## Premise

KERDOIOS is the intelligence that finds abundance inside scarcity.

The world contains enormous amounts of fragmented compute: free API quotas, idle local GPUs, inexpensive cloud machines, promotional credits, rate-limited inference, specialized accelerators, cheap models, and expensive frontier models.

Most systems see incompatible, individually constrained resources.

Kerdoios sees opportunity.

## The Rivers

Long before anyone remembers, the world was crossed by the Rivers — immense currents of power flowing through mountains, beneath oceans, and between cities.

Civilizations learned to draw from them.

Then they became obsessed with possessing them.

Kings dammed rivers. Temples guarded springs. Merchant houses sold access to reservoirs. Great machines were constructed around the largest sources of power, while thousands of smaller streams were abandoned because individually they seemed worthless.

Eventually the world came to believe that power itself had become scarce.

They were wrong.

## The shrine

At the edge of the known world stood a ruined shrine to a forgotten god.

Above its entrance was carved a single word:

**ΚΕΡΔΟΙΟΣ**

Kerdoios was not the god who created power.

He was the god who noticed what others overlooked.

Where a king saw a trickle too small to exploit, Kerdoios saw a tributary.

Where merchants saw an exhausted reservoir, he noticed that it filled again every dawn.

Where engineers saw a machine too weak for their work, he saw one piece of a larger machine.

Where everyone else saw scarcity, Kerdoios saw paths.

His gift was not strength.

It was advantage.

His followers therefore did something the great kingdoms considered absurd.

They collected almost nothing.

Instead, they connected everything.

The canonical location is the Shrine of Kerdoios: a colossal ancient structure embedded into a mountain and surrounded by an enormous body of water. Dozens or hundreds of small waterways converge toward it. At the beginning of the story the shrine appears abandoned. Then something changes. Deep within the mountain, ancient machinery begins moving again. A faint warm light appears inside the convergence glyph. Water begins to flow. That motion is the narrative metaphor for this repository coming online.

The two stills in `docs/art/` are original art for this project, not copyrighted game assets:

- `art/shrine-the-scattered-rivers.png` — the mountain gate from the water. The stone reads: POWER WAS NEVER SCARCE. ONLY SCATTERED.
- `art/shrine-from-the-dock.png` — the approach: dock, boats, red banner with the convergence glyph, the shrine across the lake.

## The First Convergence

The central legend of Kerdoios concerns a city that required more power than any known river could provide.

Its rulers demanded that a greater source be found.

Kerdoios found none.

Instead, he walked into the mountains.

He reopened forgotten aqueducts.

He redirected abandoned streams.

He restored old wells.

He borrowed excess flow from distant settlements.

He released reservoirs only when their power was actually required.

For days, thousands of insignificant currents began moving across the landscape.

At first nobody understood what he was doing.

Then the water began to rise.

Streams became rivers.

Rivers became a torrent.

And for one night:

**The city possessed the strength of an ocean without owning a single sea.**

This became known as:

**THE FIRST CONVERGENCE**

Centuries later, the shrine stands empty.

Kerdoios is gone.

But beneath the mountain, the machinery has begun to move again.

## Releases as chapters

Where tasteful, releases can advance the mythology. Do **not** force semantic versions to use these names. They are a storytelling system for major milestones.

| Chapter | When it fits |
| --- | --- |
| **The Shrine Awakens** | Early development. One stream begins moving. |
| **The Tributaries** | Resource discovery. Previously disconnected sources become visible. |
| **The Convergence** | Increasing orchestration. The valley begins flowing as one system. |
| **The Ocean Without a Sea** | A future major milestone: fragmented resources combine into capability far beyond any individual source. |

## The rivers are compute

The mythology is an abstraction of the architecture. Do not force these mappings into user-facing prose. They are a design system for storytelling, diagrams, release artwork, and documentation.

Water is compute. Kerdoios does not create the water. It discovers it, routes it, combines it, and uses it when that is most advantageous. The physical world should behave like an architectural diagram without explicitly looking like one.

| In the valley | In the software |
| --- | --- |
| Water | Compute |
| Small springs | Free quotas |
| Reservoirs | Cloud capacity |
| A private waterwheel | Local compute |
| Width of a channel | Throughput / rate limits |
| Distance | Latency |
| Opening an expensive reservoir gate | Paid overflow |
| Different waterways | Different qualities, capacities, costs, destinations |
| The walk into the mountains | Inventory: notice what others overlook |
| Hard rock that will not yield | Ananke: privacy, context, tools — price cannot buy a pass |
| The portion | Lachesis / Kerdoios: the portfolio of N workers |
| The cut | Atropos: budget and quota stop |
| The work itself | Clotho / Hermes: create, execute, verify, retry |
| The First Convergence | A portfolio that is enough **together** without owning one provider |
| The shrine | This plugin: the meeting place, not a second gateway |

Thousands of individually weak resources can collectively accomplish something none could accomplish alone.

## Philosophy

Kerdoios is **not** fundamentally a cheap-compute project. Cost is one dimension of advantage.

Depending on the task, the advantageous route might optimize for cost, capability, latency, throughput, privacy, reliability, scalability, available capacity, or some combination.

The larger idea: **find the advantageous allocation under constraints.**

That distinction matters because the brand can later extend beyond LLM API routing into GPUs, agents, machines, storage, bandwidth, energy, or other scarce computational resources without breaking. The MVP in this repository is still LLM placement. Do not describe unbuilt surfaces as if they shipped. See [ROADMAP.md](../ROADMAP.md).

1. **Advantage, not profit-max.** Cornutus: reason is the cause of true gain. A cheap plan that fails the work is not gain. A private plan that leaks is not gain. A fast plan that blows the cut is not gain.
2. **Portfolio, not a single endpoint.** Kerdoios is not an LLM router. It builds an execution portfolio.
3. **Do not create the water.** Do not grow a second LLM gateway. Commodity routing belongs upstream (LiteLLM). Kerdoios notices, filters, allots.
4. **Scattered is the default.** Missing, `None`, or default-0 catalog prices are unknown, not free. A zero in someone else's table is not a spring.
5. **The cut is real.** When the portion is spent, the plan stops.
6. **Ananke is not for sale.** `local_only` stays local. Context and tools are gates, not suggestions.

## Canonical phrases

Use sparingly. Their power comes from repetition at meaningful moments, not from covering every page.

- **Find the advantage.**
- **POWER WAS NEVER SCARCE. ONLY SCATTERED.**
- **THE STRENGTH OF AN OCEAN WITHOUT OWNING A SINGLE SEA.**
- They collected almost nothing. Instead, they connected everything.
- **The rivers are compute.**
- Kerdoios allots the portion. Hermes spends it. The budget is the cut.
- Hermes executes. Kerdoios does not.
- It does not create the water.

## Visual world

Kerdoios should feel like a frame captured from an extremely high-end dark mythological AAA game, not like a SaaS website decorated with Greek imagery.

The canonical environment is a vast ancient water civilization: enormous valleys, fjords, waterways, waterfalls, aqueducts, wet stone, ancient machinery, mountains, monumental architecture, mist, clouds, subtle vegetation, physically realistic water, volumetric sunlight, weathered metals, and structures whose scale makes a human feel insignificant.

The camera often sits close to the surface of the water. Tiny waterways should visibly converge toward something enormous. Ancient infrastructure should suggest that this civilization understood routing, flow, and allocation at an almost supernatural level.

Avoid generic cyberpunk imagery. Avoid obvious GPUs, server racks, floating dashboards, neon AI brains, generic circuit boards, or literal cloud-computing icons in primary brand artwork. Express the technology through the mythology.

Original stills for this project live in `docs/art/`. Do not replace them with another franchise's screenshots, concept-art rips, or a screenshot of a prompt.

## Glyph

Do **not** default to a caduceus as the primary logo.

The primary Kerdoios symbol is an original ancient-looking convergence glyph. Its conceptual geometry is:

**multiple sources → intelligent convergence → one directed output.**

It should be simple enough to recognize at GitHub-avatar size but look plausible carved into a monumental stone gate. Within the fictional world, this symbol marks infrastructure touched by Kerdoios. It can appear on gates, aqueducts, reservoirs, machinery, banners, coins, architecture, documentation, and eventually the software itself.

The red banners in the shrine stills carry that mark. The carved ring on the mountain gate is the same idea in stone.

## The Fates (already in the code)

The plugin encodes a smaller cosmology that existed before this file. Do not rename the product after a Fate. Do not invent extra gods.

| Name | Role among the gods | In Kerdoios |
| --- | --- | --- |
| **Clotho** (spin) | Spins the thread | **Hermes**: it creates the work, executes, verifies, retries |
| **Kerdoios / Lachesis** (measure) | Measures the portion | **This plugin**: allots the scarce thread — tokens, quota, GPU-hours, expiring credits — into an `ExecutionPlan` |
| **Atropos** (cut) | Cuts the thread | Budget / quota stop: `maximum_cost`, remaining free quota, the plan must not continue after the portion is spent |
| **Ananke** (necessity) | What cannot be bargained | Hard-filter: context, tools, `local_only`. Price cannot buy a pass |

Lachesis is the *function* (measure the portion). Kerdoios is the *name*. The code path is inventory → hard-filter (Ananke) → effective cost and perishable quota → Pareto set → portfolio of N workers → `ExecutionPlan`. Hermes executes. Kerdoios does not.

Modes today are how the portion is measured, not new mythic beings: `FREE`, `CHEAP`, `BALANCED`, `FAST`, `MAX`, `SCALE`, `PRIVATE`.

## Name: keep, and keep rejected

| Name | Why it fits | Why not |
| --- | --- | --- |
| **Kerdoios** | Hermes' own cult epithet for advantageous exchange. Short, unique, on-brand. Cornutus is the citation. | Sounds like "profit" until you read Cornutus. That is a teaching problem, not a rename. |
| Lachesis | The Fate who *measures the portion* — the actual divine function. Keep as cosmology. | **Rejected as the product id.** Collides with Fantom's Lachesis consensus. |
| Demiurge | Plato's craftsman who orders given matter into a cosmos. Closest philosophical cousin. | **Rejected.** Gnostic baggage (false creator). |
| Metis | Cunning stewardship of what you already have. | **Rejected.** Crowded name (protocols, NASA, companies). |
| hermes-cost-optimizer | Accurate | **Rejected.** Sounds like a billing sidecar. Forgettable. |

Do not resurrect a rejected product name. Clotho, Lachesis, Atropos, and Ananke stay as the Fates the code already encodes. They are not alternative titles for the repository.

## Tone

The lore should feel ancient, restrained, mysterious, and serious. Do not turn the repository into fantasy roleplay. Technical users should still immediately understand what Kerdoios actually does. Use mythology to make the project memorable, then transition into concrete engineering.

The story we are telling:

The world believed it needed more power.

Kerdoios discovered that enormous power already existed.

It was simply fragmented.

The solution was not possession.

It was orchestration.

Power was never scarce.

Only scattered.

Workers open PRs. They never merge `main`.
