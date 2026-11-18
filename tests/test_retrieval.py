import numpy as np
import pytest

from energy_rag.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
    tokenize,
)
from energy_rag.vector_store import ScoredChunk

from .conftest import make_chunk


def test_tokenize_keeps_decimal_numbers_and_drops_stopwords():
    assert tokenize("Der Preis lag bei 95,18 Euro/MWh.") == ["preis", "95,18", "euro", "mwh"]


def test_cosine_scores_are_similarities(store, embedder):
    results = store.query(embedder.embed_query("Day-Ahead-Preis 2023 Euro/MWh"), top_k=5)
    scores = [r.score for r in results]
    assert results[0].chunk.page == 28
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_upsert_is_idempotent(store, corpus, embedder):
    before = store.count()
    store.upsert(corpus, embedder.embed_documents([c.text for c in corpus]))
    assert store.count() == before == len(corpus)


def test_upsert_rejects_length_mismatch(store, corpus):
    with pytest.raises(ValueError):
        store.upsert(corpus, np.zeros((1, 256), dtype=np.float32))


def test_all_chunks_roundtrip(store, corpus):
    assert {c.id for c in store.all_chunks()} == {c.id for c in corpus}


def test_dense_retriever(store, embedder):
    top = DenseRetriever(store, embedder).retrieve("Netzentgelte Haushaltskunden", top_k=1)
    assert top[0].chunk.page == 24


def test_bm25_finds_exact_numbers(corpus):
    top = BM25Retriever(corpus).retrieve("Wo steht 142 TWh?", top_k=2)
    assert top[0].chunk.source == "uba.pdf"
    assert all(r.score > 0 for r in top)


def test_bm25_empty_query_returns_nothing(corpus):
    assert BM25Retriever(corpus).retrieve("und der die", top_k=3) == []


def test_rrf_rewards_agreement():
    a, b, c = (make_chunk(t, page=i) for i, t in enumerate(["aaa", "bbb", "ccc"]))
    r1 = [ScoredChunk(a, 0.9), ScoredChunk(b, 0.8)]
    r2 = [ScoredChunk(b, 12.0), ScoredChunk(c, 3.0)]
    fused = reciprocal_rank_fusion([r1, r2], k=60)
    assert fused[0].chunk.id == b.id
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert {f.chunk.id for f in fused} == {a.id, b.id, c.id}


def test_hybrid_combines_dense_and_bm25(store, embedder, corpus):
    hybrid = HybridRetriever(
        [DenseRetriever(store, embedder), BM25Retriever(corpus)], candidate_k=5
    )
    results = hybrid.retrieve("Onshore-Windenergie Kosten", top_k=3)
    assert results[0].chunk.page == 6
    assert len(results) == 3


class FakeCrossEncoder:
    """Bevorzugt Kandidaten, die das Wort 'Power-to-Gas' enthalten."""

    def predict(self, pairs):
        return np.array([1.0 if "Power-to-Gas" in text else 0.0 for _, text in pairs])


def test_reranker_reorders_candidates(corpus):
    base = BM25Retriever(corpus)
    reranker = CrossEncoderReranker(base, model=FakeCrossEncoder(), candidate_k=5)
    top = reranker.retrieve("Strom TWh Prozent Cent", top_k=2)
    assert top[0].chunk.source == "uba.pdf"
    assert reranker.name == "bm25+rerank"
