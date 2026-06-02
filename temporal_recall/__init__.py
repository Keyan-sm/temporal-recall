"""temporal-recall: an offline benchmark for the temporal correctness of agent memory,
plus a lightweight bitemporal memory store."""
from __future__ import annotations

from .benchmark import BenchmarkReport, run_benchmark
from .memory import POLICIES, Memory, Record, select
from .retrieval import Retriever, TfidfRetriever

__version__ = "0.1.0"

__all__ = [
    "Memory",
    "Record",
    "select",
    "POLICIES",
    "run_benchmark",
    "BenchmarkReport",
    "TfidfRetriever",
    "Retriever",
    "__version__",
]


def __getattr__(name):
    # Lazily expose the TempLAMA loader so importing the package doesn't require the
    # optional [bench] dependency (huggingface_hub).
    if name in ("load_templama", "build_queries", "TempLama", "Statement", "TemporalQuery"):
        from . import data

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
