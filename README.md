# temporal-recall

**A benchmark for whether agent memory returns the right fact as the world changes, and
whether it can tell you what was true last year.** Small, offline, and scored by exact match over
TF-IDF retrieval, it runs on TempLAMA's evolving Wikidata facts (Dhingra et al., 2022)
and ships a lightweight store that implements the valid-time dimension of Zep's
bitemporal model (Rasmussen et al., 2025), with no knowledge graph or database.

On 6,000 queries over real Wikidata facts that change over time (TempLAMA), three memory
designs share the same retrieval and split apart once time matters:

| memory design | as-of a past year | current value |
| --- | --- | --- |
| flat vector store (a pile of embeddings) | 0.667 | 0.593 |
| latest-wins (an update overwrites the old value) | **0.580** | 0.996 |
| **bitemporal** (keep every version with its valid time) | **0.998** | 0.996 |

All three share one retriever, and `latest`/`bitemporal` select within the retrieved
entity, so the gaps come down to how each design handles time. The retrieval ceiling is
1.000: the right answer always sits among the candidates.

Bitemporal scores about 1.0 by design, since it is the fix. The result worth reading is
the shape of the naive failures. Latest-wins, the design most agent frameworks ship,
answers "what is true now" at 0.996 but "what was true at time T" at only 0.580. It
overwrites history. A flat vector store has no model of time and lands at 0.59 to 0.67 on
both. The 2026 memory literature calls "treats change as replacement, not evolution" an
open problem; here you can measure it on real data, and fix it.

## Why this exists

Agent memory has many implementations (Mem0, Letta, Zep/Graphiti), but two things stay
missing:

- **A reproducible, offline benchmark for temporal correctness.** LoCoMo and LongMemEval
  test conversational recall and need an LLM judge; temporal evaluation at the application
  level stays bespoke. `temporal-recall` scores any memory backend on real evolving facts
  with exact match, no LLM, in seconds.
- **A lightweight bitemporal store.** Zep/Graphiti (Rasmussen et al., 2025) introduced
  bitemporal agent memory as a temporal knowledge graph that needs an LLM to build it and a
  graph database to hold it. The store here keeps the valid-time idea in one drop-in Python
  object, with no graph, no LLM, and no database.

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

A latest-wins store answers "Initech" for all three. It cannot tell you the past.

```bash
temporal-recall demo          # the three policies on one evolving fact (no download)
temporal-recall benchmark     # the full TempLAMA numbers above (~5 s)
```

## How it works

Every backend shares one retriever (default: TF-IDF + cosine; swap in embeddings through
the `Retriever` protocol) and differs only in how it selects across time:

- `flat` returns the most similar memory and ignores time.
- `latest` returns the most recent assertion. Change means replacement.
- `bitemporal` keeps every assertion with its valid time, and on a query `as_of` time T
  returns the version in effect at T (the latest assertion with `valid_time <= T`). Change
  means evolution. This is the valid-time dimension of Zep's bitemporal model, kept as a
  plain store.

Sharing the retriever isolates the one variable that matters. A real latest-wins or
bitemporal store keys its versions by entity, so `latest` and `bitemporal` select within
the retrieved entity rather than pay for retrieval noise (a different entity the embedder
pulled in).

## What's faithful, and what's mine

| Component | Source | Fidelity |
| --- | --- | --- |
| Bitemporal valid-time recall (`as_of`) | Zep / Graphiti (Rasmussen et al., 2025) | Faithful to the valid-time idea; lightweight (no KG/LLM/DB) |
| Evolving-facts data + alias scoring | TempLAMA (Dhingra et al., 2022) | Used as intended (facts that change over time) |
| Framing it as a memory temporal-correctness benchmark | mine | TempLAMA was built to probe LMs; here it scores memory backends |
| `flat` and `latest-wins` baselines | mine | They stand in for common naive memory designs |
| The open problem this targets | Mem0's 2026 memory landscape | Motivation: change as replacement, not evolution; temporal queries are hardest |

## Scope and honest limitations

- **Bitemporal's ~1.0 is by construction, and that is the point.** It carries each fact's
  valid time, and the gold answer always sits among the candidates (ceiling = 1.0), so it
  wins. The benchmark measures how far the naive designs fall and gives a reproducible
  harness plus a reference store, not a SOTA number.
- **Retrieval is easy in TempLAMA.** The query carries the subject name, which pushes the
  ceiling to 1.0 and isolates temporal selection. Paraphrase-robust retrieval is roadmap.
- **Valid time only.** Zep's full bitemporal model also tracks ingestion (transaction) time
  for audit and retroactive correction. The store records `system_time`, but the benchmark
  does not yet exercise it.
- **Source-data noise.** A handful of TempLAMA rows list more than one entity for a year.
  Scoring is exact match over the gold alias set, so it inherits that small noise rather than
  claiming perfect labels.

## Module map

| Module | Responsibility |
| --- | --- |
| `data.py` | Load TempLAMA into statements + as-of/current queries |
| `retrieval.py` | TF-IDF cosine retriever (pluggable) |
| `memory.py` | `Memory` store + the three temporal policies |
| `metrics.py` | Alias-aware exact-match scoring |
| `benchmark.py` | Shared-retrieval evaluation harness |

## Papers

- **TempLAMA**, Dhingra et al., *Time-Aware Language Models as Temporal Knowledge Bases*, TACL 2022, [arXiv:2106.15110](https://arxiv.org/abs/2106.15110).
- **Zep**, Rasmussen et al., *A Temporal Knowledge Graph Architecture for Agent Memory*, [arXiv:2501.13956](https://arxiv.org/abs/2501.13956), 2025.

## Status

v0.1.0. Tested, CI on Python 3.9 to 3.12, runs offline except for the TempLAMA download.
Roadmap: ingestion-time experiments (retroactive corrections), paraphrase-robust retrieval,
and adapters to score external backends (Mem0, Letta).

## License

MIT
