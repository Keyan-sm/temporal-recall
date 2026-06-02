"""Retrieval over stored statements.

The default retriever is a dependency-light TF-IDF + cosine nearest-neighbor index
(scikit-learn). The point of this library is *temporal* selection, so retrieval is
deliberately a transparent baseline shared by every memory backend — they all see the
same candidates and differ only in how they pick among them across time. Swap in a
transformer/embedding retriever via the ``Retriever`` protocol without touching the
temporal logic.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    def fit(self, texts: List[str]) -> "Retriever": ...
    def query_many(self, texts: List[str], k: int) -> List[List[int]]: ...


class TfidfRetriever:
    """TF-IDF over word n-grams with brute-force cosine nearest neighbors."""

    def __init__(self, max_features: int = 50000, ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=max_features,
                                           ngram_range=ngram_range, sublinear_tf=True)
        self._nn = None
        self._n = 0

    def fit(self, texts: List[str]) -> "TfidfRetriever":
        from sklearn.neighbors import NearestNeighbors

        X = self._vectorizer.fit_transform(texts)
        self._n = X.shape[0]
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
        return self

    def query_many(self, texts: List[str], k: int) -> List[List[int]]:
        if self._nn is None:
            raise RuntimeError("TfidfRetriever.query_many called before fit().")
        k = min(k, self._n)
        Q = self._vectorizer.transform(texts)
        _, idx = self._nn.kneighbors(Q, n_neighbors=k)
        return idx.tolist()

    def query(self, text: str, k: int = 10) -> List[int]:
        return self.query_many([text], k)[0]
