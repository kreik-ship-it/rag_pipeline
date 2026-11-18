import json
import uuid
from types import SimpleNamespace

import pytest

from energy_rag.cli import DEFAULT_TESTSET, build_parser
from energy_rag.config import Settings
from energy_rag.evaluation import (
    EvalItem,
    LLMJudge,
    annotate_relevant_pages,
    evaluate_answers,
    evaluate_retriever,
    hit_rate_at_k,
    load_testset,
    mrr_at_k,
)
from energy_rag.generation import ExtractiveGenerator
from energy_rag.ingestion import Page
from energy_rag.pipeline import RAGPipeline, build_index, build_retriever
from energy_rag.vector_store import ChromaVectorStore, ScoredChunk

from .conftest import make_chunk


def test_hit_rate_and_mrr():
    ranks = [1, 3, None, 2]
    assert hit_rate_at_k(ranks, 1) == 0.25
    assert hit_rate_at_k(ranks, 3) == 0.75
    assert mrr_at_k(ranks, 10) == pytest.approx((1 + 1 / 3 + 1 / 2) / 4)
    assert mrr_at_k(ranks, 2) == pytest.approx((1 + 1 / 2) / 4)


def test_annotate_relevant_pages_uses_all_patterns():
    pages = [
        Page("a.pdf", 1, "Preis 95,18 Euro"),
        Page("a.pdf", 2, "Preis\n95,18 Euro/MWh im Jahr 2023"),
        Page("b.pdf", 2, "95,18 Euro/MWh 2023"),
    ]
    item = EvalItem("q1", "Preis?", "a.pdf", "95,18", evidence=[r"95,18 Euro", r"2023"])
    assert annotate_relevant_pages([item], pages)[0].relevant_pages == [2]
    with pytest.raises(ValueError):
        annotate_relevant_pages([EvalItem("q2", "?", "a.pdf", "-", evidence=["fehlt"])], pages)


class ListRetriever:
    name = "list"

    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=5):
        return self.results[query][:top_k]


def test_evaluate_retriever_ranks_on_page_level():
    hit = ScoredChunk(make_chunk("x", "a.pdf", 3), 1.0)
    miss = ScoredChunk(make_chunk("y", "a.pdf", 9), 1.0)
    items = [
        EvalItem("q1", "eins", "a.pdf", "", relevant_pages=[3]),
        EvalItem("q2", "zwei", "a.pdf", "", relevant_pages=[3]),
        EvalItem("neg", "drei", "a.pdf", "", answerable=False),
    ]
    retriever = ListRetriever({"eins": [hit, miss], "zwei": [miss, miss, hit]})
    report = evaluate_retriever(retriever, items, ks=(1, 3))
    assert report.n_questions == 2
    assert report.ranks == {"q1": 1, "q2": 3}
    assert report.metrics["Hit@1"] == 0.5
    assert report.metrics["Hit@3"] == 1.0
    assert report.metrics["MRR@3"] == pytest.approx((1 + 1 / 3) / 2)


def test_evaluate_answers_with_mocked_judge():
    hit = ScoredChunk(make_chunk("Preis 95,18", "a.pdf", 3), 1.0)
    items = [EvalItem("q1", "eins", "a.pdf", "95,18", relevant_pages=[3])]
    judge_json = '{"correct": true, "faithful": true, "reason": "ok"}'
    client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                content=[SimpleNamespace(type="text", text=judge_json)]
            )
        )
    )
    result = evaluate_answers(
        ListRetriever({"eins": [hit]}), ExtractiveGenerator(), items, judge=LLMJudge(client=client)
    )
    assert result["summary"]["Zitierquote"] == 1.0
    assert result["summary"]["Relevante Quelle zitiert"] == 1.0
    assert result["summary"]["Korrektheit"] == 1.0


def test_shipped_testset_is_consistent():
    items = load_testset(DEFAULT_TESTSET)
    assert len(items) >= 30
    assert len({i.id for i in items}) == len(items)
    for item in items:
        assert item.question.endswith("?")
        if item.answerable:
            assert item.relevant_pages and item.evidence, item.id
        else:
            assert not item.relevant_pages


def test_settings_collection_name_is_valid_and_model_specific():
    a = Settings().with_overrides(embedding_model="intfloat/multilingual-e5-small")
    b = a.with_overrides(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    assert a.collection_name != b.collection_name
    assert len(a.collection_name) <= 63
    assert a.collection_name.replace("-", "").isalnum()


def test_end_to_end_pipeline_offline(pdf_dir, embedder):
    settings = Settings().with_overrides(
        data_dir=pdf_dir, chunk_size=200, chunk_overlap=40, min_page_chars=20
    )
    store = ChromaVectorStore(None, f"e2e-{uuid.uuid4().hex[:8]}")
    stats = build_index(settings, embedder=embedder, store=store)
    assert (stats.documents, stats.pages) == (2, 3)

    # erneutes Indexieren erzeugt keine Duplikate
    build_index(settings, embedder=embedder, store=store)
    assert store.count() == stats.chunks

    retriever = build_retriever(settings, store, embedder, mode="hybrid", use_reranker=False)
    answer = RAGPipeline(retriever, ExtractiveGenerator(), top_k=3).ask(
        "Wie hoch war der Day-Ahead-Preis 2023?"
    )
    assert answer.sources[0].chunk.source == "strom.pdf"
    assert "95,18" in answer.text
    assert answer.cited_sources


def test_build_retriever_requires_index(embedder):
    empty = ChromaVectorStore(None, f"leer-{uuid.uuid4().hex[:8]}")
    with pytest.raises(RuntimeError, match="ingest"):
        build_retriever(Settings(), empty, embedder)


def test_pipeline_rejects_empty_question(store, embedder):
    pipeline = RAGPipeline(
        build_retriever(Settings(), store, embedder, "dense", False), ExtractiveGenerator()
    )
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_cli_parser():
    args = build_parser().parse_args(["evaluate", "--experiments", "baseline", "hybrid"])
    assert args.experiments == ["baseline", "hybrid"]
    assert json.dumps(vars(args), default=str)
