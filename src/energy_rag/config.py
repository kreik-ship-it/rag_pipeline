"""Zentrale Konfiguration.

Alle Werte haben sinnvolle Defaults und lassen sich über Umgebungsvariablen
(bzw. eine lokale ``.env``-Datei) überschreiben. API-Keys werden nie im Code
abgelegt: der Anthropic-Client liest ``ANTHROPIC_API_KEY`` selbst aus der Umgebung.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    """Parameter der gesamten Pipeline (Ingestion → Retrieval → Generierung)."""

    # Pfade
    data_dir: Path = field(
        default_factory=lambda: Path(_env("RAG_DATA_DIR", str(PROJECT_ROOT / "data")))
    )
    persist_dir: Path = field(
        default_factory=lambda: Path(
            _env("RAG_PERSIST_DIR", str(PROJECT_ROOT / "data" / "vector_store"))
        )
    )

    # Chunking (Zeichen). 800 Zeichen ≈ 200–250 Tokens: passt in das Kontextfenster
    # des Embedding-Modells (512 Tokens) ohne stilles Abschneiden.
    chunk_size: int = int(_env("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(_env("RAG_CHUNK_OVERLAP", "150"))
    min_page_chars: int = 80

    # Embeddings: mehrsprachiges Modell, da alle Dokumente deutsch sind.
    embedding_model: str = _env("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    embedding_batch_size: int = 32

    # Retrieval
    retrieval_mode: str = _env("RAG_RETRIEVAL_MODE", "hybrid")  # dense | bm25 | hybrid
    top_k: int = int(_env("RAG_TOP_K", "5"))
    candidate_k: int = 30  # Kandidaten je Retriever vor Fusion/Reranking
    rrf_k: int = 60
    use_reranker: bool = _env("RAG_USE_RERANKER", "true").lower() == "true"
    reranker_model: str = _env("RAG_RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    # Generierung (Anthropic Claude)
    llm_model: str = _env("RAG_LLM_MODEL", "claude-opus-5-5")
    llm_effort: str = _env("RAG_LLM_EFFORT", "medium")
    llm_max_tokens: int = int(_env("RAG_LLM_MAX_TOKENS", "16000"))
    # Server-seitiger Fallback bei Ablehnungen ("default" | "off"); manche Proxys
    # unterstützen den Beta-Parameter nicht.
    llm_fallbacks: str = _env("RAG_LLM_FALLBACKS", "default")

    @property
    def collection_name(self) -> str:
        """Eine Collection je Embedding-Modell/Chunking, damit Indizes nie vermischt werden."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.embedding_model.lower()).strip("-")
        return f"chunks-{slug}-{self.chunk_size}-{self.chunk_overlap}"[:63]

    def with_overrides(self, **kwargs) -> Settings:
        return replace(self, **kwargs)
