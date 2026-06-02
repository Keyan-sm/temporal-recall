"""Run the temporal-recall benchmark: three temporal policies on the same retrieved
candidates, scored against TempLAMA gold answers, split by query type.

All policies share one retrieval pass, so any accuracy difference is due solely to how
memory selects across time — not retrieval luck. We also report the *retrieval ceiling*
(how often a memory carrying the gold answer was even among the candidates), so the
policy accuracies can be read against the best any selector could do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import metrics
from .data import TempLama, TemporalQuery, build_queries
from .memory import POLICIES, Record, select
from .retrieval import Retriever, TfidfRetriever


@dataclass
class BenchmarkReport:
    n_statements: int
    n_queries: Dict[str, int]
    accuracy: Dict[str, Dict[str, float]]   # policy -> kind -> accuracy
    retrieval_ceiling: Dict[str, float]     # kind -> fraction gold answer was retrievable
    k: int


def run_benchmark(data: TempLama, queries: Optional[List[TemporalQuery]] = None,
                  k: int = 20, n_per_kind: int = 3000, seed: int = 0,
                  retriever: Optional[Retriever] = None) -> BenchmarkReport:
    if queries is None:
        queries = build_queries(data, n_per_kind=n_per_kind, seed=seed)

    records = [Record(text=s.text, valid_time=s.year, object_value=s.object_name,
                      aliases=s.aliases, fact_key=s.fact_key, id=i)
               for i, s in enumerate(data.statements)]
    retriever = retriever or TfidfRetriever()
    retriever.fit([r.text for r in records])
    candidate_lists = retriever.query_many([q.query for q in queries], k=k)

    correct = {p: {"as_of": 0, "current": 0} for p in POLICIES}
    totals = {"as_of": 0, "current": 0}
    ceiling = {"as_of": 0, "current": 0}

    for q, idxs in zip(queries, candidate_lists):
        candidates = [records[i] for i in idxs]
        totals[q.kind] += 1
        if any(metrics.matches(c.aliases, q.gold_aliases) for c in candidates):
            ceiling[q.kind] += 1
        for policy in POLICIES:
            chosen = select(policy, candidates, q.as_of)
            if chosen and metrics.matches(chosen[0].aliases, q.gold_aliases):
                correct[policy][q.kind] += 1

    accuracy = {
        p: {kind: (correct[p][kind] / totals[kind] if totals[kind] else 0.0) for kind in totals}
        for p in POLICIES
    }
    retrieval_ceiling = {kind: (ceiling[kind] / totals[kind] if totals[kind] else 0.0) for kind in totals}
    return BenchmarkReport(
        n_statements=len(records),
        n_queries=dict(totals),
        accuracy=accuracy,
        retrieval_ceiling=retrieval_ceiling,
        k=k,
    )
