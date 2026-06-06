"""Simple BM25 scorer — no external dependencies."""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25:
    """BM25Okapi scorer over a fixed corpus of tokenized documents."""

    def __init__(self, corpus_texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = [_tokenize(t) for t in corpus_texts]
        n = len(self.corpus)
        avg_dl = sum(len(d) for d in self.corpus) / max(n, 1)

        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df: dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        self._idf: dict[str, float] = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        self._avg_dl = avg_dl
        self._n = n

    def scores(self, query: str) -> list[float]:
        """Return a BM25 score for each document in the corpus."""
        q_tokens = _tokenize(query)
        result = [0.0] * self._n
        for token in q_tokens:
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue
            for i, doc in enumerate(self.corpus):
                tf = doc.count(token)
                if tf == 0:
                    continue
                dl = len(doc)
                score = idf * (
                    tf * (self.k1 + 1.0)
                    / (tf + self.k1 * (1.0 - self.b + self.b * dl / self._avg_dl))
                )
                result[i] += score
        return result
