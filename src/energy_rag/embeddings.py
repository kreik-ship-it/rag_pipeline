"""Embedding-Modelle hinter einer schmalen Schnittstelle (leicht mockbar in Tests)."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wrapper um ``sentence-transformers`` mit L2-normalisierten Vektoren.

    E5-Modelle wurden mit den Präfixen ``query:``/``passage:`` trainiert; ohne diese
    Präfixe sinkt die Retrieval-Qualität messbar. Sie werden daher automatisch gesetzt.
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        is_e5 = "e5" in model_name.lower()
        self._query_prefix = "query: " if is_e5 else ""
        self._passage_prefix = "passage: " if is_e5 else ""
        logger.info(
            "Embedding-Modell %s geladen (Dim. %d, max. %d Tokens)",
            model_name,
            self._model.get_embedding_dimension(),
            self._model.max_seq_length,
        )

    @property
    def max_seq_length(self) -> int:
        return int(self._model.max_seq_length)

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(
            [self._passage_prefix + t for t in texts], show_progress=len(texts) > 100
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]
