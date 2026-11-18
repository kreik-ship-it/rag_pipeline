import pytest

from energy_rag.chunking import chunk_pages, make_chunk_id
from energy_rag.ingestion import Page, clean_text, load_pdfs, remove_running_headers


def test_clean_text_joins_hyphenated_line_breaks():
    assert clean_text("Anteil der erneuer-\nbaren Energien") == "Anteil der erneuerbaren Energien"


def test_clean_text_joins_soft_hyphen_at_line_end():
    assert clean_text("Messstellenbe­\ntrieb") == "Messstellenbetrieb"


def test_clean_text_keeps_hyphen_of_compound_split_at_line_end():
    assert clean_text("Onshore-\nWindenergieanlagen") == "Onshore-Windenergieanlagen"


def test_clean_text_keeps_compound_hyphens_and_removes_noise():
    raw = "PV-Anlagen­ und\t\tWind  \n\n\n\nInhalt ........ 12"
    assert clean_text(raw) == "PV-Anlagen und Wind\n\nInhalt … 12"


def test_load_pdfs_reads_pages_with_1_based_numbers_and_skips_empty(pdf_dir):
    pages = load_pdfs(pdf_dir, min_chars=20)
    assert {(p.source, p.page) for p in pages} == {
        ("strom.pdf", 1),
        ("strom.pdf", 2),
        ("kosten.pdf", 1),
    }
    assert any("95,18" in p.text for p in pages)


def test_load_pdfs_raises_without_pdfs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdfs(tmp_path)


def test_load_pdfs_skips_corrupt_file(pdf_dir):
    (pdf_dir / "kaputt.pdf").write_bytes(b"keine pdf")
    sources = {p.source for p in load_pdfs(pdf_dir, min_chars=20)}
    assert sources == {"strom.pdf", "kosten.pdf"}


def test_chunks_respect_size_and_keep_page_reference():
    text = " ".join(f"Satz Nummer {i} über Strompreise." for i in range(200))
    chunks = chunk_pages([Page("doc.pdf", 7, text)], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.page == 7 and c.source == "doc.pdf" for c in chunks)
    assert [c.chunk_index for c in chunks] == sorted(c.chunk_index for c in chunks)


def test_chunks_do_not_cut_words():
    text = " ".join(["Stromgestehungskosten"] * 100)
    chunks = chunk_pages([Page("doc.pdf", 1, text)], chunk_size=200, chunk_overlap=40)
    assert all(set(c.text.split()) == {"Stromgestehungskosten"} for c in chunks)


def test_chunk_ids_are_deterministic_and_unique():
    pages = [Page("a.pdf", 1, "Erster Absatz.\n\nZweiter Absatz mit mehr Text " * 30)]
    first, second = chunk_pages(pages, 200, 40), chunk_pages(pages, 200, 40)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert make_chunk_id("a.pdf", 1, 0, "x") != make_chunk_id("a.pdf", 2, 0, "x")


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=100, chunk_overlap=100)


def test_remove_running_headers_keeps_content():
    topics = ["Wind", "Solar", "Gas", "Kohle", "Netze", "Speicher", "Preise", "Wärme", "Biomasse"]
    texts = [f"BERICHT 2024 | {i}\n{t} ist Thema dieser Seite." for i, t in enumerate(topics)]
    texts[3] += "\nBERICHT 2024 wird im Fließtext erwähnt und bleibt erhalten."
    cleaned = remove_running_headers(texts)
    assert all("BERICHT 2024 |" not in t for t in cleaned)
    assert cleaned[0] == "Wind ist Thema dieser Seite."
    assert "bleibt erhalten" in cleaned[3]
