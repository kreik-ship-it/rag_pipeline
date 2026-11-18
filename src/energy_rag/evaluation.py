"""Evaluation von Retrieval (Hit-Rate, MRR, Recall) und Antwortqualität.

Relevanz wird auf **Seitenebene** bewertet: Ein Treffer liegt vor, wenn ein
abgerufener Chunk von einer Seite stammt, auf der die Antwort steht. Das ist
unabhängig von Chunk-Größe und Chunk-IDs und erlaubt so den fairen Vergleich
verschiedener Chunking-Konfigurationen.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .generation import Generator
from .ingestion import Page
from .retrieval import Retriever
from .vector_store import ScoredChunk


@dataclass
class EvalItem:
    id: str
    question: str
    source: str
    reference_answer: str
    evidence: list[str] = field(default_factory=list)  # Regex-Muster, die die Antwort belegen
    relevant_pages: list[int] = field(default_factory=list)
    answerable: bool = True

    def is_relevant(self, chunk_source: str, page: int) -> bool:
        return chunk_source == self.source and page in self.relevant_pages


def load_testset(path: Path) -> list[EvalItem]:
    with open(path, encoding="utf-8") as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def save_testset(items: Iterable[EvalItem], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def annotate_relevant_pages(items: Sequence[EvalItem], pages: Sequence[Page]) -> list[EvalItem]:
    """Bestimmt ``relevant_pages`` reproduzierbar: alle Seiten des Quelldokuments,
    auf denen **alle** Evidenz-Muster vorkommen."""
    by_source: dict[str, list[Page]] = {}
    for p in pages:
        by_source.setdefault(p.source, []).append(p)
    annotated = []
    for item in items:
        if not item.answerable:
            annotated.append(item)
            continue
        patterns = [re.compile(e, re.IGNORECASE) for e in item.evidence]
        hits = [
            p.page
            for p in by_source.get(item.source, [])
            if all(pat.search(_normalize(p.text)) for pat in patterns)
        ]
        if not hits:
            raise ValueError(f"Keine Belegseite für {item.id} gefunden – Evidenz prüfen.")
        annotated.append(EvalItem(**{**asdict(item), "relevant_pages": sorted(hits)}))
    return annotated


# ------------------------------------------------------------------ Retrieval-Metriken
def first_relevant_rank(item: EvalItem, results: Sequence[ScoredChunk]) -> int | None:
    for rank, r in enumerate(results, start=1):
        if item.is_relevant(r.chunk.source, r.chunk.page):
            return rank
    return None


def hit_rate_at_k(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    return sum(1.0 / r for r in ranks if r is not None and r <= k) / len(ranks)


@dataclass
class RetrievalReport:
    name: str
    n_questions: int
    metrics: dict[str, float]
    ranks: dict[str, int | None]
    mean_latency_ms: float

    def row(self) -> dict[str, float | str]:
        return {"Konfiguration": self.name, **self.metrics, "Latenz (ms)": self.mean_latency_ms}


def evaluate_retriever(
    retriever: Retriever,
    items: Sequence[EvalItem],
    ks: Sequence[int] = (1, 3, 5, 10),
    name: str | None = None,
) -> RetrievalReport:
    items = [i for i in items if i.answerable]
    max_k = max(ks)
    ranks: dict[str, int | None] = {}
    latencies = []
    for item in items:
        start = time.perf_counter()
        results = retriever.retrieve(item.question, top_k=max_k)
        latencies.append((time.perf_counter() - start) * 1000)
        ranks[item.id] = first_relevant_rank(item, results)

    rank_list = list(ranks.values())
    metrics = {f"Hit@{k}": hit_rate_at_k(rank_list, k) for k in ks}
    metrics[f"MRR@{max_k}"] = mrr_at_k(rank_list, max_k)
    return RetrievalReport(
        name=name or retriever.name,
        n_questions=len(items),
        metrics=metrics,
        ranks=ranks,
        mean_latency_ms=statistics.fmean(latencies) if latencies else 0.0,
    )


# ------------------------------------------------------------------- Antwortqualität
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "faithful": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "faithful", "reason"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """Bewerte die Antwort eines RAG-Systems.

<frage>{question}</frage>
<referenzantwort>{reference}</referenzantwort>
<quellen>{context}</quellen>
<antwort>{answer}</antwort>

- "correct": true, wenn die Antwort inhaltlich mit der Referenzantwort übereinstimmt
  (Zahlen müssen stimmen; zusätzliche korrekte Details sind erlaubt). Ist die Frage laut
  Referenz nicht beantwortbar, ist "correct" true genau dann, wenn die Antwort das erkennt.
- "faithful": true, wenn jede Aussage der Antwort durch die Quellen gedeckt ist.
- "reason": ein kurzer Satz Begründung."""


class LLMJudge:
    """LLM-as-a-Judge mit strukturierter JSON-Ausgabe."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def judge(self, question: str, reference: str, answer: str, context: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": JUDGE_SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question, reference=reference, answer=answer, context=context
                    ),
                }
            ],
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)


def evaluate_answers(
    retriever: Retriever,
    generator: Generator,
    items: Sequence[EvalItem],
    judge: LLMJudge | None = None,
    top_k: int = 5,
) -> dict:
    """Antwortqualität: Zitierquote, Anteil relevanter Zitate und (optional) LLM-Urteil."""
    from .generation import format_context

    rows = []
    for item in items:
        sources = retriever.retrieve(item.question, top_k=top_k)
        answer = generator.generate(item.question, sources)
        cited_relevant = any(
            item.is_relevant(s.chunk.source, s.chunk.page) for s in answer.cited_sources
        )
        row = {
            "id": item.id,
            "answerable": item.answerable,
            "has_citation": bool(answer.cited) and not answer.invalid_citations,
            "cited_relevant": cited_relevant,
            "answer": answer.text,
        }
        if judge is not None:
            row.update(
                judge.judge(
                    item.question, item.reference_answer, answer.text, format_context(sources)
                )
            )
        rows.append(row)

    answerable = [r for r in rows if r["answerable"]]
    summary = {
        "n": len(rows),
        "Zitierquote": statistics.fmean(r["has_citation"] for r in answerable)
        if answerable
        else 0.0,
        "Relevante Quelle zitiert": statistics.fmean(r["cited_relevant"] for r in answerable)
        if answerable
        else 0.0,
    }
    if judge is not None:
        summary["Korrektheit"] = statistics.fmean(r["correct"] for r in rows)
        summary["Faithfulness"] = statistics.fmean(r["faithful"] for r in rows)
    return {"summary": summary, "rows": rows}
