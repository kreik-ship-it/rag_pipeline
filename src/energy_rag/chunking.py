"""Zerlegung der Seiten in überlappende Chunks mit stabilen IDs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .ingestion import Page

# Absatz → Zeile → Satz → Wort. Der leere Separator ist nur die letzte Rückfallebene,
# damit nicht mitten im Wort geschnitten wird.
SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int
    chunk_index: int

    @property
    def citation(self) -> str:
        return f"{self.source}, S. {self.page}"

    def to_metadata(self) -> dict[str, str | int]:
        return {"source": self.source, "page": self.page, "chunk_index": self.chunk_index}


def make_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    """Deterministische ID: erneutes Indexieren erzeugt keine Duplikate (Upsert statt Add)."""
    digest = hashlib.sha1(f"{source}|{page}|{chunk_index}|{text}".encode()).hexdigest()
    return digest[:20]


def chunk_pages(
    pages: Iterable[Page], chunk_size: int = 800, chunk_overlap: int = 150
) -> list[Chunk]:
    """Teilt jede Seite separat. So bleibt jede Quelle eindeutig einer Seite zuordenbar."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap muss kleiner als chunk_size sein.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",
    )
    chunks: list[Chunk] = []
    for page in pages:
        for index, text in enumerate(splitter.split_text(page.text)):
            text = text.strip()
            if len(text) < 30:  # Seitenzahlen, Kopfzeilen-Reste
                continue
            chunks.append(
                Chunk(
                    id=make_chunk_id(page.source, page.page, index, text),
                    text=text,
                    source=page.source,
                    page=page.page,
                    chunk_index=index,
                )
            )
    return chunks
