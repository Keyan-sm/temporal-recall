"""Load TempLAMA into timestamped statements + temporal query sets.

TempLAMA (Dhingra et al., 2022) is a set of cloze queries whose answer changes with
time, e.g. "Tom Brady plays for _X_." -> New England Patriots (2010) ... Tampa Bay
Buccaneers (2020). Each row is one (subject, relation, year) with the gold answer(s)
for that year, carrying alias sets for robust exact-match scoring. ~97% of the
(subject, relation) facts actually evolve over 2010-2020.

We turn each row into:
  * a **statement** to store in memory: the cloze filled with that year's answer,
    tagged with its valid year and the answer's aliases;
  * and we derive two **query** sets: "as-of year Y" (what was true at Y) and
    "current" (what is true at the fact's latest year).

Loading needs the optional ``[bench]`` dependency (huggingface_hub). The core memory
backends and metrics do not import it.

Reference: Dhingra et al., "Time-Aware Language Models as Temporal Knowledge Bases,"
TACL 2022 / arXiv:2106.15110.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

HF_REPO = "Yova/templama"
SPLITS = {"train": "train_with_aliases.json", "val": "val_with_aliases.json",
          "test": "test_with_aliases.json"}


@dataclass
class Statement:
    """One time-stamped fact assertion to store in memory."""
    fact_key: str          # "<subject>_<relation>", the entity-attribute whose value evolves
    query: str             # the cloze, e.g. "Tom Brady plays for _X_."
    year: int              # the year this assertion is valid for
    object_name: str       # the filled answer, e.g. "New England Patriots"
    aliases: Set[str]      # accepted surface forms of the answer
    text: str = ""         # the filled statement used for retrieval

    def __post_init__(self):
        if not self.text:
            self.text = self.query.replace("_X_", self.object_name)


@dataclass
class TemporalQuery:
    query: str
    fact_key: str
    as_of: int
    gold_aliases: Set[str]
    kind: str              # "as_of" | "current"


@dataclass
class TempLama:
    statements: List[Statement]
    timeline: Dict[str, Dict[int, Set[str]]] = field(default_factory=dict)  # fact_key -> {year: aliases}

    def __len__(self) -> int:
        return len(self.statements)


def _aliases(answer_list) -> Tuple[str, Set[str]]:
    names: Set[str] = set()
    primary = ""
    for a in answer_list:
        for n in a.get("name", []):
            names.add(n)
            if not primary:
                primary = a.get("original_name", [n])[0] if a.get("original_name") else n
    if not primary and names:
        primary = sorted(names)[0]
    return primary, names


def load_templama(split: str = "test", cache_dir: Optional[str] = None) -> TempLama:
    """Download (if needed) and parse a TempLAMA split into statements + a timeline."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            "Loading TempLAMA needs the [bench] extra: pip install 'temporal-recall[bench]'"
        ) from exc
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}; got {split!r}")

    path = hf_hub_download(HF_REPO, SPLITS[split], repo_type="dataset", cache_dir=cache_dir)
    statements: List[Statement] = []
    timeline: Dict[str, Dict[int, Set[str]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            subject, relation, year = row["id"].rsplit("_", 2)
            year = int(year)
            fact_key = f"{subject}_{relation}"
            primary, names = _aliases(row["answer"])
            if not names:
                continue
            statements.append(Statement(fact_key=fact_key, query=row["query"], year=year,
                                        object_name=primary, aliases=names))
            timeline.setdefault(fact_key, {})[year] = names
    return TempLama(statements=statements, timeline=timeline)


def build_queries(data: TempLama, n_per_kind: int = 3000, seed: int = 0
                  ) -> List[TemporalQuery]:
    """Sample 'as-of' and 'current' queries from evolving facts.

    'as-of' asks for the value at a specific past year; 'current' asks for the value
    at the fact's most recent year. Only facts that actually change over time are used,
    so the queries genuinely exercise temporal selection.
    """
    import random

    rng = random.Random(seed)
    evolving = [k for k, series in data.timeline.items()
                if len({frozenset(v) for v in series.values()}) > 1]
    # index statements by fact for quick lookup of one representative query string
    query_of: Dict[str, str] = {}
    for s in data.statements:
        query_of.setdefault(s.fact_key, s.query)

    as_of_pool = [(k, y) for k in evolving for y in data.timeline[k]]
    rng.shuffle(as_of_pool)
    queries: List[TemporalQuery] = []
    for k, y in as_of_pool[:n_per_kind]:
        queries.append(TemporalQuery(query=query_of[k], fact_key=k, as_of=y,
                                     gold_aliases=data.timeline[k][y], kind="as_of"))

    current_facts = list(evolving)
    rng.shuffle(current_facts)
    for k in current_facts[:n_per_kind]:
        latest = max(data.timeline[k])
        queries.append(TemporalQuery(query=query_of[k], fact_key=k, as_of=latest,
                                     gold_aliases=data.timeline[k][latest], kind="current"))
    return queries
