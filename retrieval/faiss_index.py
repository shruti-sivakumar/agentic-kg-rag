"""
Dense passage index over a single question's retrieval corpus.

2WikiMultiHopQA questions are evaluated against a small, closed corpus —
the ten supporting and distractor passages shipped with each question —
rather than against one index spanning the entire dataset. Accordingly,
`PassageIndex` is built per question: it holds only that question's
passages and is queried by Systems A, B, and C's Vector-Retrieve step.

The index is a flat (exhaustive) inner-product FAISS index. Given a
corpus of ten passages, an exhaustive search is negligible in cost, and
the flat index guarantees exact nearest-neighbor results, so no
approximate-search structure is warranted here.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from data.load_2wiki import Passage
from retrieval.embed import encode


@dataclass
class ScoredPassage:
    passage: Passage
    score: float  # cosine similarity between the query and this passage


class PassageIndex:
    """A per-question flat inner-product index over `Passage.text` embeddings."""

    def __init__(self, passages: list[Passage]):
        if not passages:
            raise ValueError("PassageIndex requires at least one passage")
        self._passages = passages
        embeddings = encode([p.text for p in passages])
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def __len__(self) -> int:
        return len(self._passages)

    def search(self, query: str, k: int = 3) -> list[ScoredPassage]:
        """Return the k passages most similar to `query`, ranked by cosine similarity."""
        k = min(k, len(self._passages))
        query_embedding = encode([query])
        scores, indices = self._index.search(query_embedding, k)
        return [
            ScoredPassage(passage=self._passages[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
        ]


if __name__ == "__main__":
    from data.load_2wiki import load_2wiki

    example = load_2wiki(n=1, seed=0)[0]
    index = PassageIndex(example.passages)

    print(f"question: {example.question}")
    print(f"gold answer: {example.answer}")
    print(f"corpus size: {len(index)}")
    print()
    for rank, result in enumerate(index.search(example.question, k=3), start=1):
        print(f"{rank}. [{result.score:.3f}] {result.passage.title} (supporting={result.passage.is_supporting})")
