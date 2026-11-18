"""Kommandozeile: ``energy-rag ingest | ask | search | annotate | evaluate``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import PROJECT_ROOT, Settings

DEFAULT_TESTSET = PROJECT_ROOT / "data" / "eval" / "testset.jsonl"
DEFAULT_RESULTS = PROJECT_ROOT / "docs" / "results"


def _print_sources(sources) -> None:
    for i, s in enumerate(sources, start=1):
        snippet = " ".join(s.chunk.text.split())[:160]
        print(f"  [{i}] {s.chunk.citation}  (Score {s.score:.3f})\n      {snippet} …")


def cmd_ingest(args, settings: Settings) -> None:
    from .pipeline import build_index

    stats = build_index(settings, rebuild=args.rebuild)
    print(
        f"{stats.documents} Dokumente, {stats.pages} Seiten → {stats.chunks} Chunks "
        f"in '{settings.collection_name}' ({stats.seconds:.0f} s)"
    )


def cmd_ask(args, settings: Settings) -> None:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline.from_settings(settings)
    answer = pipeline.ask(args.question, top_k=args.top_k)
    print(f"\n{answer.text}\n\nQuellen ({answer.model}):")
    _print_sources(answer.sources)


def cmd_search(args, settings: Settings) -> None:
    from .embeddings import SentenceTransformerEmbedder
    from .pipeline import build_retriever
    from .vector_store import ChromaVectorStore

    store = ChromaVectorStore(settings.persist_dir, settings.collection_name)
    retriever = build_retriever(
        settings, store, SentenceTransformerEmbedder(settings.embedding_model)
    )
    _print_sources(retriever.retrieve(args.query, top_k=args.top_k))


def cmd_annotate(args, settings: Settings) -> None:
    from .evaluation import annotate_relevant_pages, load_testset, save_testset
    from .ingestion import load_pdfs

    items = load_testset(args.testset)
    annotated = annotate_relevant_pages(
        items, load_pdfs(settings.data_dir, settings.min_page_chars)
    )
    save_testset(annotated, args.testset)
    for item in annotated:
        print(f"{item.id:8s} {item.relevant_pages}")


# Experimentkonfigurationen für den Vergleich in der README.
EXPERIMENTS = {
    "baseline": dict(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        mode="dense",
        rerank=False,
        label="Ausgangsstand: MiniLM (englisch), 1000/200, dense",
    ),
    "e5-dense": dict(
        embedding_model="intfloat/multilingual-e5-small",
        chunk_size=800,
        chunk_overlap=150,
        mode="dense",
        rerank=False,
        label="Multilinguales E5, 800/150, dense",
    ),
    "bm25": dict(
        embedding_model="intfloat/multilingual-e5-small",
        chunk_size=800,
        chunk_overlap=150,
        mode="bm25",
        rerank=False,
        label="BM25 (lexikalisch), 800/150",
    ),
    "hybrid": dict(
        embedding_model="intfloat/multilingual-e5-small",
        chunk_size=800,
        chunk_overlap=150,
        mode="hybrid",
        rerank=False,
        label="Hybrid E5 + BM25 (RRF)",
    ),
    "hybrid-rerank": dict(
        embedding_model="intfloat/multilingual-e5-small",
        chunk_size=800,
        chunk_overlap=150,
        mode="hybrid",
        rerank=True,
        label="Hybrid + Cross-Encoder-Reranking",
    ),
}


def cmd_evaluate(args, settings: Settings) -> None:
    from .embeddings import SentenceTransformerEmbedder
    from .evaluation import evaluate_retriever, load_testset
    from .pipeline import build_index, build_retriever
    from .vector_store import ChromaVectorStore

    items = load_testset(args.testset)
    names = args.experiments or list(EXPERIMENTS)
    embedders: dict[str, SentenceTransformerEmbedder] = {}
    reports = []
    for name in names:
        exp = EXPERIMENTS[name]
        cfg = settings.with_overrides(
            embedding_model=exp["embedding_model"],
            chunk_size=exp["chunk_size"],
            chunk_overlap=exp["chunk_overlap"],
        )
        if cfg.embedding_model not in embedders:
            embedders[cfg.embedding_model] = SentenceTransformerEmbedder(cfg.embedding_model)
        embedder = embedders[cfg.embedding_model]
        store = ChromaVectorStore(cfg.persist_dir, cfg.collection_name)
        if store.count() == 0:
            print(f"Baue Index {cfg.collection_name} …")
            build_index(cfg, embedder=embedder, store=store)
        retriever = build_retriever(
            cfg, store, embedder, mode=exp["mode"], use_reranker=exp["rerank"]
        )
        report = evaluate_retriever(retriever, items, name=name)
        reports.append((exp["label"], report))
        m = report.metrics
        print(
            f"{name:14s} Hit@1 {m['Hit@1']:.2f}  Hit@3 {m['Hit@3']:.2f}  Hit@5 {m['Hit@5']:.2f}  "
            f"Hit@10 {m['Hit@10']:.2f}  MRR@10 {m['MRR@10']:.3f}  {report.mean_latency_ms:.0f} ms"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "retrieval_metrics.json"
    payload = [
        {
            "experiment": r.name,
            "label": label,
            "n_questions": r.n_questions,
            "metrics": r.metrics,
            "mean_latency_ms": r.mean_latency_ms,
            "ranks": r.ranks,
        }
        for label, r in reports
    ]
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nErgebnisse gespeichert: {out}")

    if args.answers:
        _evaluate_answers(settings, items, args.output)


def _evaluate_answers(settings: Settings, items, output: Path) -> None:
    from .evaluation import LLMJudge, evaluate_answers
    from .generation import has_anthropic_credentials
    from .pipeline import RAGPipeline

    if not has_anthropic_credentials():
        sys.exit("Für die Antwort-Evaluation wird ANTHROPIC_API_KEY benötigt (siehe .env.example).")
    pipeline = RAGPipeline.from_settings(settings)
    result = evaluate_answers(
        pipeline.retriever,
        pipeline.generator,
        items,
        judge=LLMJudge(model=settings.llm_model),
        top_k=settings.top_k,
    )
    (output / "answer_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="energy-rag", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="ausführliches Logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="PDFs laden, chunken, einbetten und indexieren")
    p.add_argument("--rebuild", action="store_true", help="Collection vorher leeren")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="Frage beantworten (mit Quellenangaben)")
    p.add_argument("question")
    p.add_argument("-k", "--top-k", type=int, default=None)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("search", help="Nur Retrieval, ohne Generierung")
    p.add_argument("query")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser(
        "annotate", help="Belegseiten des Testsets aus den Evidenz-Mustern bestimmen"
    )
    p.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("evaluate", help="Retrieval-Konfigurationen vergleichen")
    p.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    p.add_argument("--experiments", nargs="*", choices=list(EXPERIMENTS))
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    p.add_argument(
        "--answers", action="store_true", help="zusätzlich Antwortqualität (API-Key nötig)"
    )
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.func(args, Settings())


if __name__ == "__main__":
    main()
