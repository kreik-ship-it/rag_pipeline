"""Verdrahtung der Komponenten: Index aufbauen und Fragen beantworten."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .chunking import chunk_pages
from .config import Settings
from .embeddings import Embedder, SentenceTransformerEmbedder
from .generation import (
    Answer,
    ClaudeGenerator,
    ExtractiveGenerator,
    Generator,
    has_anthropic_credentials,
)
from .ingestion import load_pdfs
from .retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    Retriever,
)
from .vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    documents: int
    pages: int
    chunks: int
    seconds: float


def build_index(
    settings: Settings,
    embedder: Embedder | None = None,
    store: ChromaVectorStore | None = None,
    rebuild: bool = False,
) -> IndexStats:
    """Ingestion → Chunking → Embedding → Upsert in Chroma (idempotent)."""
    start = time.perf_counter()
    pages = load_pdfs(settings.data_dir, min_chars=settings.min_page_chars)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    logger.info("%d Seiten → %d Chunks", len(pages), len(chunks))

    embedder = embedder or SentenceTransformerEmbedder(
        settings.embedding_model, batch_size=settings.embedding_batch_size
    )
    store = store or ChromaVectorStore(settings.persist_dir, settings.collection_name)
    if rebuild:
        store.reset()
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.upsert(chunks, embeddings)
    return IndexStats(
        documents=len({p.source for p in pages}),
        pages=len(pages),
        chunks=len(chunks),
        seconds=time.perf_counter() - start,
    )


def build_retriever(
    settings: Settings,
    store: ChromaVectorStore,
    embedder: Embedder,
    mode: str | None = None,
    use_reranker: bool | None = None,
    reranker_model=None,
) -> Retriever:
    mode = mode or settings.retrieval_mode
    use_reranker = settings.use_reranker if use_reranker is None else use_reranker

    if store.count() == 0:
        raise RuntimeError(
            f"Collection '{store.collection_name}' ist leer. Zuerst `energy-rag ingest` ausführen."
        )
    dense = DenseRetriever(store, embedder)
    if mode == "dense":
        retriever: Retriever = dense
    elif mode == "bm25":
        retriever = BM25Retriever(store.all_chunks())
    elif mode == "hybrid":
        retriever = HybridRetriever(
            [dense, BM25Retriever(store.all_chunks())],
            candidate_k=settings.candidate_k,
            rrf_k=settings.rrf_k,
        )
    else:
        raise ValueError(f"Unbekannter Retrieval-Modus: {mode!r}")

    if use_reranker:
        retriever = CrossEncoderReranker(
            retriever,
            model_name=settings.reranker_model,
            candidate_k=settings.candidate_k,
            model=reranker_model,
        )
    return retriever


def default_generator(settings: Settings) -> Generator:
    if has_anthropic_credentials():
        return ClaudeGenerator(
            model=settings.llm_model,
            effort=settings.llm_effort,
            max_tokens=settings.llm_max_tokens,
            fallbacks=settings.llm_fallbacks,
        )
    logger.warning("Kein ANTHROPIC_API_KEY gesetzt – nutze extraktiven Fallback ohne LLM.")
    return ExtractiveGenerator()


class RAGPipeline:
    """Fassade für CLI, UI und Notebooks."""

    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    @classmethod
    def from_settings(cls, settings: Settings | None = None, generator: Generator | None = None):
        settings = settings or Settings()
        embedder = SentenceTransformerEmbedder(
            settings.embedding_model, batch_size=settings.embedding_batch_size
        )
        store = ChromaVectorStore(settings.persist_dir, settings.collection_name)
        retriever = build_retriever(settings, store, embedder)
        return cls(retriever, generator or default_generator(settings), top_k=settings.top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Die Frage darf nicht leer sein.")
        sources = self.retriever.retrieve(question, top_k=top_k or self.top_k)
        return self.generator.generate(question, sources)
