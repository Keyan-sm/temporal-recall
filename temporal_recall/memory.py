"""Memory backends that share retrieval but differ in how they select across time.

Three temporal policies, each a one-line idea with very different temporal behavior:

* ``flat``       — return the single most semantically similar memory, ignoring time.
                   This is a plain vector store ("a pile of embeddings").
* ``latest``     — among the retrieved candidates for the same fact, return the most
                   recent one. This is the common "an update overwrites the old value"
                   design — change treated as *replacement*.
* ``bitemporal`` — keep every assertion with its valid time, and on a query ``as_of``
                   time T return the version in effect at T (the latest assertion with
                   valid_time <= T). Change treated as *evolution*. This is the
                   valid-time dimension of Zep's bitemporal model (Rasmussen et al.,
                   2025), implemented as a lightweight store — no knowledge graph, no
                   LLM, no graph database.

Because all three share the retriever, the benchmark isolates exactly one variable:
how memory handles time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from .retrieval import Retriever, TfidfRetriever

POLICIES = ("flat", "latest", "bitemporal")


@dataclass
class Record:
    text: str
    valid_time: float                 # when the asserted fact became true (e.g. a year)
    object_value: str = ""            # the answer this record asserts, if structured
    aliases: Set[str] = field(default_factory=set)
    fact_key: str = ""                # entity/slot this record is about (e.g. "<subject>_<relation>")
    system_time: float = 0.0          # when the record was ingested (transaction time)
    valid_until: Optional[float] = None  # explicit end of validity, if known
    id: int = -1


class Memory:
    """A temporal memory store. ``policy`` controls how `recall` selects across time."""

    def __init__(self, policy: str = "bitemporal", retriever: Optional[Retriever] = None):
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
        self.policy = policy
        self.retriever = retriever or TfidfRetriever()
        self.records: List[Record] = []
        self._indexed = False

    # --- write ------------------------------------------------------------
    def remember(self, text: str, valid_time: float, object_value: str = "",
                 aliases: Optional[Set[str]] = None, key: str = "",
                 valid_until: Optional[float] = None) -> int:
        """Store an assertion. ``key`` identifies the entity/slot it concerns; pass it
        when a store holds multiple entities so recall can group versions correctly."""
        rec = Record(text=text, valid_time=valid_time, object_value=object_value,
                     aliases=set(aliases or ()), fact_key=key,
                     system_time=float(len(self.records)),
                     valid_until=valid_until, id=len(self.records))
        self.records.append(rec)
        self._indexed = False
        return rec.id

    def index(self) -> "Memory":
        """Build the retrieval index. Call once after loading, before recall."""
        self.retriever.fit([r.text for r in self.records])
        self._indexed = True
        return self

    # --- read -------------------------------------------------------------
    def recall(self, query: str, as_of: Optional[float] = None, k: int = 10) -> List[Record]:
        """Return the memory selected by this policy for ``query`` (and ``as_of``)."""
        if not self._indexed:
            self.index()
        candidates = [self.records[i] for i in self.retriever.query(query, k=k)]
        return select(self.policy, candidates, as_of)


def select(policy: str, candidates: List[Record], as_of: Optional[float]) -> List[Record]:
    """Apply a temporal policy to retrieved candidates (most-similar first).

    Retrieval is shared, but `latest`/`bitemporal` then operate **within the retrieved
    entity** — the records sharing the top candidate's ``fact_key`` — because a real
    latest-wins or bitemporal store keys its versions by entity/slot and would never
    return a different entity's value. (When records carry no key, e.g. a single-entity
    store, this is a no-op.) This keeps the comparison about *time*, not retrieval noise.
    """
    if not candidates:
        return []
    if policy == "flat":
        return candidates[:1]
    target_key = candidates[0].fact_key
    same_entity = [c for c in candidates if c.fact_key == target_key]
    if policy == "latest":
        return [max(same_entity, key=lambda r: r.valid_time)]
    if policy == "bitemporal":
        if as_of is None:
            return [max(same_entity, key=lambda r: r.valid_time)]
        valid = [r for r in same_entity if r.valid_time <= as_of
                 and (r.valid_until is None or as_of < r.valid_until)]
        if not valid:
            return [min(same_entity, key=lambda r: r.valid_time)]
        return [max(valid, key=lambda r: r.valid_time)]
    raise ValueError(f"unknown policy: {policy}")
