"""PDF-Ingestion: Seiten laden und Text bereinigen."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Page:
    """Eine PDF-Seite. ``page`` ist 1-basiert (wie in jedem PDF-Viewer)."""

    source: str
    page: int
    text: str


_HYPHEN_BREAK = re.compile(r"(\w)-\n([a-zäöüß])")
_COMPOUND_BREAK = re.compile(r"(\w)-\n([A-ZÄÖÜ0-9])")
_SOFT_HYPHEN = "­"
_INLINE_WS = re.compile("[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_DOT_LEADERS = re.compile(r"(?:\s*\.){5,}")


def clean_text(text: str) -> str:
    """Normalisiert PDF-Text, ohne Inhalte zu verändern.

    - Silbentrennung am Zeilenende wird zusammengeführt ("erneuer-\\nbaren" → "erneuerbaren").
      Folgt ein Großbuchstabe, handelt es sich um ein Bindestrich-Kompositum
      ("Onshore-\\nWindenergie" → "Onshore-Windenergie"); der Bindestrich bleibt erhalten.
    - Weiche Trennstriche, Tabs, Punktleisten aus Inhaltsverzeichnissen und Mehrfach-Leerzeichen
      werden entfernt.
    """
    text = text.replace("\r", "").replace(_SOFT_HYPHEN + "\n", "").replace(_SOFT_HYPHEN, "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _COMPOUND_BREAK.sub(r"\1-\2", text)
    text = _DOT_LEADERS.sub(" … ", text)
    text = _INLINE_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _line_key(line: str) -> str:
    """Seitenzahlen ignorieren, damit "Bericht | 23" und "Bericht | 24" gleich zählen."""
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_running_headers(texts: list[str], min_share: float = 0.3) -> list[str]:
    """Entfernt Kopf-/Fußzeilen, die auf mindestens ``min_share`` aller Seiten vorkommen.

    Solche Zeilen (Behördenname, Berichtstitel, Seitenzahl) tragen keine Information,
    landen aber sonst in jedem Chunk und verzerren Embeddings und BM25.
    """
    if len(texts) < 5:
        return texts
    counts: dict[str, int] = {}
    for text in texts:
        for key in {_line_key(line) for line in text.split("\n") if line.strip()}:
            counts[key] = counts.get(key, 0) + 1
    threshold = min_share * len(texts)
    repeated = {k for k, n in counts.items() if n >= threshold and len(k) < 150 and k != "#"}
    return [
        "\n".join(line for line in text.split("\n") if _line_key(line) not in repeated)
        for text in texts
    ]


def load_pdf(path: Path, min_chars: int = 80) -> list[Page]:
    """Lädt eine PDF-Datei seitenweise. Fast leere Seiten (Deckblätter, Abbildungen) entfallen."""
    import pymupdf

    with pymupdf.open(path) as doc:
        texts = remove_running_headers([clean_text(p.get_text()) for p in doc])
    return [
        Page(source=path.name, page=index + 1, text=text.strip())
        for index, text in enumerate(texts)
        if len(text.strip()) >= min_chars
    ]


def load_pdfs(directory: Path, min_chars: int = 80) -> list[Page]:
    """Lädt alle PDFs eines Verzeichnisses (defekte Dateien werden übersprungen)."""
    directory = Path(directory)
    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Keine PDF-Dateien in {directory} gefunden.")

    pages: list[Page] = []
    for pdf in pdf_files:
        try:
            doc_pages = load_pdf(pdf, min_chars=min_chars)
        except Exception as exc:  # noqa: BLE001 - eine defekte PDF soll den Lauf nicht abbrechen
            logger.error("PDF %s konnte nicht gelesen werden: %s", pdf.name, exc)
            continue
        logger.info("%s: %d Seiten mit Text", pdf.name, len(doc_pages))
        pages.extend(doc_pages)
    return pages
