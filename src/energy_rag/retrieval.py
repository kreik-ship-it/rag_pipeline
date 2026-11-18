"""Retriever: dichte Suche, BM25, hybride Fusion (RRF) und Cross-Encoder-Reranking."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder
from .vector_store import ChromaVectorStore, ScoredChunk

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]: ...


# --------------------------------------------------------------------------- dense
class DenseRetriever:
    name = "dense"

    def __init__(self, store: ChromaVectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        return self.store.query(self.embedder.embed_query(query), top_k=top_k)


# ---------------------------------------------------------------------------- BM25
# Kleine deutsche Stoppwortliste; reicht, um Funktionswörter aus BM25 herauszuhalten.
_STOPWORDS_TEXT = """
    aber alle allem allen aller alles als also am an auch auf aus bei beim bis bzw da damit
    dann das dass dem den denen der des die dies diese diesem diesen dieser doch dort durch ein
    eine einem einen einer eines es für gegen hat hatte haben im in ins ist je jedoch kann kein
    keine man mit nach nicht noch nur ob oder ohne sehr sei sein seine sich sie sind so sowie
    über um und uns unter vom von vor war waren was wie wird wurde wurden zu zum zur zwischen
    welche welcher welches wieviel viele wo wann warum wer liegt lag liegen gibt
"""
GERMAN_STOPWORDS = frozenset(_STOPWORDS_TEXT.split())
_TOKEN = re.compile(r"[a-zäöüß0-9]+(?:[,.][0-9]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Kleinschreibung, Zahlen mit Dezimalkomma bleiben ein Token ("95,18"), Stoppwörter raus."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in GERMAN_STOPWORDS and len(t) > 1]


class BM25Retriever:
    """Lexikalische Suche. Stark bei exakten Begriffen, Abkürzungen und Zahlen."""

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk]):
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("BM25 benötigt mindestens einen Chunk.")
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            ScoredChunk(chunk=self.chunks[i], score=float(scores[i]), retriever="bm25")
            for i in order
            if scores[i] > 0
        ]


# -------------------------------------------------------------------------- hybrid
def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """RRF (Cormack et al., 2009): score = Σ 1/(k + rank).

    Rangbasiert, daher müssen die nicht vergleichbaren Skalen von Kosinus- und
    BM25-Scores nicht normalisiert werden.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk.id] = scores.get(item.chunk.id, 0.0) + 1.0 / (k + rank)
            chunks[item.chunk.id] = item.chunk
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [ScoredChunk(chunk=chunks[cid], score=s, retriever="hybrid") for cid, s in fused]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], candidate_k: int = 30, rrf_k: int = 60):
        self.retrievers = list(retrievers)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        rankings = [r.retrieve(query, top_k=max(top_k, self.candidate_k)) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, k=self.rrf_k)[:top_k]


# ------------------------------------------------------------------------ reranker
class CrossEncoderReranker:
    """Bewertet (Frage, Chunk)-Paare gemeinsam – genauer, aber langsamer als Bi-Encoder.

    Daher zweistufig: der Basis-Retriever liefert ``candidate_k`` Kandidaten, der
    Cross-Encoder sortiert sie neu.
    """

    def __init__(
        self,
        base: Retriever,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        candidate_k: int = 30,
        model=None,
    ):
        self.base = base
        self.candidate_k = candidate_k
        self.name = f"{base.name}+rerank"
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, max_length=512)
        self._model = model

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        candidates = self.base.retrieve(query, top_k=max(top_k, self.candidate_k))
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        order = np.argsort(np.asarray(scores))[::-1][:top_k]
        return [
            ScoredChunk(chunk=candidates[i].chunk, score=float(scores[i]), retriever=self.name)
            for i in order
        ]
