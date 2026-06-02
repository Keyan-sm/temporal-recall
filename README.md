# temporal-recall

**Does your agent's memory return the right fact as the world changes — and can it tell
you what was true *last year*?** `temporal-recall` is a small, offline, LLM-free
benchmark that answers that on real evolving facts, plus a lightweight bitemporal memory
store that passes it.

On **6,000 queries** over real Wikidata facts that change over time (TempLAMA), three
memory designs that share the *same* retrieval differ enormously once time matters:

| memory design | as-of a past year | current value |
| --- | --- | --- |
| flat vector store (a pile of embeddings) | 0.667 | 0.593 |
| latest-wins (an update overwrites the old value) | **0.580** | 0.996 |
| **bitemporal** (keep every version with its valid time) | **0.998** | 0.996 |

All three share the *same* retrieval, and `latest`/`bitemporal` select within the
retrieved entity — so these gaps are *purely* about how memory handles time. (Retrieval
ceiling is **1.000**: the right answer is always among the candidates.)

The headline isn't that bitemporal scores ~1.0 — it does so by design, because it's the
*fix*. The headline is the shape of the **naive** failures: **latest-wins — what most
agent frameworks ship — answers "what is true now" almost perfectly (0.996) but only
58% of "what was true at time T" (0.580)**: it overwrites history. A flat vector memory
has no notion of time at all and is mediocre at both (0.59–0.67). That "treats change as
replacement, not evolution" failure is one the 2026 memory literature calls an open
problem — here it's measured on real data, and fixed.

## Why this exists

Agent memory is a crowded space (Mem0, Letta, Zep/Graphiti), but two things are still
missing:

- **A reproducible, offline benchmark for *temporal correctness*.** LoCoMo and
  LongMemEval test conversational recall and need an LLM judge; application-level
  temporal evaluation is, in practice, bespoke. `temporal-recall` scores any memory
  backend on real evolving facts with exact-match, no LLM, in seconds.
- **A *lightweight* bitemporal store.** Zep/Graphiti (Rasmussen et al., 2025) introduced
  bitemporal agent memory, but as a temporal **knowledge graph** that needs an LLM to
  build it and a graph database to hold it. `temporal-recall`'s store is the valid-time
  idea distilled to a drop-in Python object — no graph, no LLM, no database.

## Install

```bash
pip install temporal-recall            # core: numpy + scikit-learn
pip install "temporal-recall[bench]"   # + huggingface_hub (to load TempLAMA)
```

## Use

```python
from temporal_recall import Memory

mem = Memory(policy="bitemporal")
mem.remember("Lina works at Acme.",    valid_time=2019, object_value="Acme")
mem.remember("Lina works at Globex.",  valid_time=2022, object_value="Globex")
mem.remember("Lina works at Initech.", valid_time=2025, object_value="Initech")
mem.index()

mem.recall("Where does Lina work?", as_of=2020)[0].object_value   # 'Acme'
mem.recall("Where does Lina work?", as_of=2023)[0].object_value   # 'Globex'
mem.recall("Where does Lina work?", as_of=2026)[0].object_value   # 'Initech'
```

A `latest-wins` store would answer "Initech" for all three — it cannot tell you the past.

```bash
temporal-recall demo          # the three policies on one evolving fact (no download)
temporal-recall benchmark     # the full TempLAMA numbers above (~5 s)
```

## How it works

Every backend shares one retriever (default: TF-IDF + cosine; swap in embeddings via the
`Retriever` protocol) and differs only in how it **selects across time**:

- `flat` — return the most similar memory, ignore time.
- `latest` — among candidates, return the most recent. Change = replacement.
- `bitemporal` — keep every assertion with its valid time; on a query `as_of` time T,
  return the version in effect at T (latest assertion with `valid_time <= T`). Change =
  evolution. This is the valid-time dimension of Zep's bitemporal model, as a plain store.

Sharing the retriever is the point: it isolates the one variable that matters. And
because a real latest-wins/bitemporal store keys versions by entity, `latest` and
`bitemporal` select *within the retrieved entity* — so they aren't unfairly charged for
retrieval noise (a different entity the embedder happened to pull in).

## What's faithful, and what's mine

| Component | Source | Fidelity |
| --- | --- | --- |
| Bitemporal valid-time recall (`as_of`) | Zep / Graphiti (Rasmussen et al., 2025) | **Faithful** to the valid-time idea; lightweight (no KG/LLM/DB) |
| Evolving-facts data + alias scoring | TempLAMA (Dhingra et al., 2022) | **Faithful** — used as intended (facts that change over time) |
| Framing it as a *memory* temporal-correctness benchmark | — | **Mine.** TempLAMA was built to probe LMs; here it scores memory backends. |
| `flat` and `latest-wins` baselines | — | **Mine.** They stand in for common naive memory designs. |
| The open problem this targets | Mem0's 2026 memory landscape | Motivation: "change as replacement, not evolution"; "temporal queries are hardest" |

## Scope and honest limitations

- **Bitemporal's ~1.0 is by construction, and that's the point.** It is given each fact's
  valid time and the gold answer is always retrievable here (ceiling = 1.0), so it should
  win. The benchmark's job is to **quantify how far the naive designs fall** and to give
  a reproducible harness + a reference store — not to crown a SOTA number.
- **Retrieval is easy in TempLAMA** (the query contains the subject name), which is why
  the ceiling is 1.0 and the experiment cleanly isolates temporal selection. Paraphrase-
  robust retrieval is future work.
- **Valid time only.** Zep's full bitemporal model also tracks ingestion (transaction)
  time for audit and retroactive correction; the store records `system_time` but the
  benchmark does not yet exercise it.
- **Source-data noise.** A handful of TempLAMA rows list more than one entity for a year;
  scoring is exact-match over the gold alias set, so it inherits that small noise rather
  than claiming perfect labels.

## Module map

| Module | Responsibility |
| --- | --- |
| `data.py` | Load TempLAMA into statements + as-of/current queries |
| `retrieval.py` | TF-IDF cosine retriever (pluggable) |
| `memory.py` | `Memory` store + the three temporal policies |
| `metrics.py` | Alias-aware exact-match scoring |
| `benchmark.py` | Shared-retrieval evaluation harness |

## Papers

- **TempLAMA** — Dhingra et al., *Time-Aware Language Models as Temporal Knowledge Bases*, TACL 2022, [arXiv:2106.15110](https://arxiv.org/abs/2106.15110).
- **Zep** — Rasmussen et al., *A Temporal Knowledge Graph Architecture for Agent Memory*, [arXiv:2501.13956](https://arxiv.org/abs/2501.13956), 2025.

## Status

v0.1.0 — tested, CI configured for Python 3.9–3.12, runs offline for everything except
the TempLAMA download. Roadmap: ingestion-time experiments (retroactive corrections),
paraphrase-robust retrieval, and adapters to score external backends (Mem0, Letta).

## License

MIT
