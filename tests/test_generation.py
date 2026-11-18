from types import SimpleNamespace

from energy_rag.generation import (
    NO_ANSWER,
    ClaudeGenerator,
    ExtractiveGenerator,
    build_user_prompt,
    extract_citations,
)
from energy_rag.vector_store import ScoredChunk

from .conftest import make_chunk


def _sources():
    return [
        ScoredChunk(make_chunk("Der Preis lag bei 95,18 Euro/MWh.", "mon.pdf", 28), 0.9),
        ScoredChunk(make_chunk("Netzentgelte stiegen um 24 Prozent.", "mon.pdf", 24), 0.7),
    ]


class FakeMessages:
    def __init__(self, text="Der Preis lag bei 95,18 Euro/MWh [1].", stop_reason="end_turn"):
        self.calls = []
        self.text = text
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.text),
            ],
            stop_reason=self.stop_reason,
            stop_details=None,
            model=kwargs["model"],
        )


def fake_client(**kwargs):
    messages = FakeMessages(**kwargs)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages)), messages


def test_extract_citations_handles_formats():
    assert extract_citations("A [2]. B [1][3]. C [1, 4]. D [2]") == [2, 1, 3, 4]
    assert extract_citations("keine Zitate") == []


def test_prompt_numbers_sources_with_document_and_page():
    prompt = build_user_prompt("Wie hoch war der Preis?", _sources())
    assert '<quelle nr="1" dokument="mon.pdf" seite="28">' in prompt
    assert '<quelle nr="2" dokument="mon.pdf" seite="24">' in prompt
    assert prompt.rstrip().endswith("[Nr].")


def test_claude_generator_uses_fallbacks_and_effort():
    client, messages = fake_client()
    answer = ClaudeGenerator(client=client).generate("Preis?", _sources())
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in call
    assert answer.text == "Der Preis lag bei 95,18 Euro/MWh [1]."
    assert answer.cited == [1]
    assert answer.cited_sources[0].chunk.page == 28


def test_claude_generator_without_fallbacks_uses_plain_endpoint():
    client, messages = fake_client()
    ClaudeGenerator(client=client, fallbacks="off").generate("Preis?", _sources())
    assert "fallbacks" not in messages.calls[0]


def test_claude_generator_handles_refusal_and_invalid_citations():
    client, _ = fake_client(text="", stop_reason="refusal")
    assert "abgelehnt" in ClaudeGenerator(client=client).generate("x", _sources()).text

    client, _ = fake_client(text="Antwort [7]")
    answer = ClaudeGenerator(client=client).generate("x", _sources())
    assert answer.invalid_citations == [7]
    assert answer.cited_sources == []


def test_no_sources_skips_llm_call():
    client, messages = fake_client()
    answer = ClaudeGenerator(client=client).generate("x", [])
    assert answer.text == NO_ANSWER
    assert messages.calls == []


def test_extractive_generator_cites_top_passages():
    answer = ExtractiveGenerator(n_passages=2).generate("Preis?", _sources())
    assert "95,18" in answer.text
    assert answer.cited == [1, 2]
