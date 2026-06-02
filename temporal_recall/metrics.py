"""Scoring: did the recalled memory carry the gold answer for the queried time?

Exact match against the answer's alias set, after light normalization (lowercase,
strip punctuation/articles). No LLM judge — scoring is deterministic and offline.
"""
from __future__ import annotations

import re
from typing import Iterable, Set

_PUNCT = re.compile(r"[^\w\s]")
_ARTICLES = re.compile(r"\b(the|a|an)\b")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    # Delete punctuation (so "A.C. Milan" == "AC Milan"), then strip leading articles.
    text = text.lower()
    text = _PUNCT.sub("", text)
    text = _WS.sub(" ", text).strip()
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def normalize_set(values: Iterable[str]) -> Set[str]:
    return {normalize(v) for v in values if normalize(v)}


def matches(predicted_aliases: Iterable[str], gold_aliases: Iterable[str]) -> bool:
    """True if any normalized predicted alias equals any normalized gold alias."""
    pred = normalize_set(predicted_aliases)
    gold = normalize_set(gold_aliases)
    return bool(pred & gold)
