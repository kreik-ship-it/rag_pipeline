"""Gemeinsame Fixtures. Alle Tests laufen ohne Netzwerk, Modelldownloads und API-Keys."""

from __future__ import annotations

import hashlib
import uuid

import numpy as np
import pytest

from energy_rag.chunking import Chunk, make_chunk_id
from energy_rag.retrieval import tokenize
from energy_rag.vector_store import ChromaVectorStore


class FakeEmbedder:
    """Deterministisches Bag-of-Words-Hashing statt eines neuronalen Modells."""

    model_name = "fake-embedder"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def make_chunk(text: str, source: str = "a.pdf", page: int = 1, index: int = 0) -> Chunk:
    return Chunk(make_chunk_id(source, page, index, text), text, source, page, index)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def corpus() -> list[Chunk]:
    return [
        make_chunk(
            "Die Stromgestehungskosten von PV-Freiflächenanlagen liegen bei 4,1 Cent.", "ise.pdf", 5
        ),
        make_chunk(
            "Onshore-Windenergie kostet zwischen 4,3 und 9,2 Cent pro Kilowattstunde.", "ise.pdf", 6
        ),
        make_chunk("Der Day-Ahead-Preis lag 2023 im Mittel bei 95,18 Euro/MWh.", "mon.pdf", 28),
        make_chunk(
            "Die Netzentgelte für Haushaltskunden stiegen 2024 um 24 Prozent.", "mon.pdf", 24
        ),
        make_chunk("Power-to-Gas erzeugt 2050 zwischen 36 und 142 TWh Strom.", "uba.pdf", 19),
    ]


@pytest.fixture
def store(corpus, embedder) -> ChromaVectorStore:
    s = ChromaVectorStore(None, f"test-{uuid.uuid4().hex[:8]}")
    s.upsert(corpus, embedder.embed_documents([c.text for c in corpus]))
    return s


@pytest.fixture
def pdf_dir(tmp_path):
    """Erzeugt zwei kleine PDFs mit bekanntem Inhalt."""
    import pymupdf

    pages = {
        "strom.pdf": [
            "Der durchschnittliche Day-Ahead-Preis betrug 2023 rund 95,18 Euro/MWh. "
            "Das ist ein Rückgang von rund 60 Prozent gegenüber dem Vorjahr.",
            "Die Netzentgelte für Haushaltskunden sind 2024 um gut 24 Prozent gestiegen. "
            "Der Mittelwert liegt bei 11,62 ct/kWh für typische Haushalte.",
        ],
        "kosten.pdf": [
            "Onshore-Windenergieanlagen erreichen Stromgestehungskosten zwischen 4,3 und "
            "9,2 Cent pro Kilowattstunde und gehören damit zu den günstigsten Technologien.",
            "x",  # fast leere Seite – soll beim Laden entfallen
        ],
    }
    for name, texts in pages.items():
        doc = pymupdf.open()
        for text in texts:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
        doc.save(tmp_path / name)
        doc.close()
    return tmp_path
