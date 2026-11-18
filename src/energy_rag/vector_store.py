"""Persistenter Vektorspeicher auf Basis von ChromaDB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    retriever: str = "dense"


def _to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        source=str(meta["source"]),
        page=int(meta["page"]),
        chunk_index=int(meta.get("chunk_index", 0)),
    )


class ChromaVectorStore:
    """Chroma-Collection mit **Kosinus-Distanz**.

    Chroma verwendet standardmäßig die quadrierte L2-Distanz. Die ursprüngliche
    Umsetzung rechnete trotzdem ``similarity = 1 - distance`` und erhielt so negative
    "Ähnlichkeiten", die der Schwellwert-Filter verwarf. Mit ``space=cosine`` gilt
    ``distance = 1 - cos_sim`` und die Umrechnung ist korrekt.
    """

    def __init__(self, persist_dir: Path | None, collection_name: str):
        import chromadb

        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"description": "Chunks der Energiemarkt-PDFs"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, configuration={"hnsw": {"space": "cosine"}}
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 1000) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Anzahl der Chunks und Embeddings stimmt nicht überein.")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
                embeddings=embeddings[start : start + batch_size].tolist(),
            )
        logger.info("%d Chunks gespeichert (Collection gesamt: %d)", len(chunks), self.count())

    def query(self, embedding: np.ndarray, top_k: int) -> list[ScoredChunk]:
        n = self.count()
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[np.asarray(embedding).tolist()],
            n_results=min(top_k, n),
            include=["documents", "metadatas", "distances"],
        )
        return [
            ScoredChunk(chunk=_to_chunk(cid, doc, meta), score=1.0 - float(dist), retriever="dense")
            for cid, doc, meta, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
                strict=True,
            )
        ]

    def all_chunks(self) -> list[Chunk]:
        """Alle gespeicherten Chunks (für den BM25-Index), deterministisch sortiert."""
        res = self._collection.get(include=["documents", "metadatas"])
        chunks = [
            _to_chunk(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]
        return sorted(chunks, key=lambda c: (c.source, c.page, c.chunk_index))
