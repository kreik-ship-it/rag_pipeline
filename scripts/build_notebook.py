"""Erzeugt und führt ``notebooks/01_rag_evaluation.ipynb`` aus.

Voraussetzung: PDFs in ``data/`` und ``energy-rag evaluate`` wurde ausgeführt
(liefert ``docs/results/retrieval_metrics.json``).

    python scripts/build_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_rag_evaluation.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        "# Energy-RAG: Analyse & Evaluation\n\n"
        "Dieses Notebook dokumentiert die methodischen Entscheidungen des Projekts mit Zahlen:\n\n"
        "1. **Korpus & Chunking** – wie viele Seiten/Chunks, wie lang sind sie in Tokens?\n"
        "2. **Zwei Fehler des Ausgangsstands** – abgeschnittene Chunks und falsche Score-Umrechnung\n"
        "3. **Retrieval-Evaluation** – fünf Konfigurationen auf 34 handannotierten Fragen\n"
        "4. **Fehleranalyse** – welche Fragen bleiben schwierig?\n"
        "5. **Beispiel** – Antwort mit Quellenangaben"
    ),
    code(
        "import json\n"
        "import warnings\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "from energy_rag.chunking import chunk_pages\n"
        "from energy_rag.config import PROJECT_ROOT, Settings\n"
        "from energy_rag.evaluation import load_testset\n"
        "from energy_rag.ingestion import load_pdfs\n\n"
        "warnings.filterwarnings('ignore')\n"
        "pd.set_option('display.max_colwidth', 90)\n"
        "IMG = PROJECT_ROOT / 'docs' / 'images'\n"
        "IMG.mkdir(parents=True, exist_ok=True)\n"
        "settings = Settings()"
    ),
    md("## 1. Korpus und Chunking"),
    code(
        "pages = load_pdfs(settings.data_dir)\n"
        "corpus = pd.DataFrame([{'Dokument': p.source, 'Zeichen': len(p.text)} for p in pages])\n"
        "corpus.groupby('Dokument').agg(Seiten=('Zeichen', 'size'), Zeichen=('Zeichen', 'sum'))"
    ),
    code(
        "from transformers import AutoTokenizer\n\n"
        "configs = {\n"
        "    'Ausgangsstand (1000/200)': (1000, 200, 'sentence-transformers/all-MiniLM-L6-v2', 256),\n"
        "    'Neu (800/150)': (800, 150, 'intfloat/multilingual-e5-small', 512),\n"
        "}\n"
        "rows = []\n"
        "for name, (size, overlap, model, max_len) in configs.items():\n"
        "    chunks = chunk_pages(pages, size, overlap)\n"
        "    tok = AutoTokenizer.from_pretrained(model)\n"
        "    n_tokens = np.array([len(tok(c.text)['input_ids']) for c in chunks])\n"
        "    rows.append({\n"
        "        'Konfiguration': name, 'Modell': model.split('/')[-1], 'Chunks': len(chunks),\n"
        "        'Ø Tokens': round(n_tokens.mean()), 'Max. Tokens Modell': max_len,\n"
        "        'Anteil abgeschnitten': f'{(n_tokens > max_len).mean():.0%}',\n"
        "    })\n"
        "pd.DataFrame(rows).set_index('Konfiguration')"
    ),
    md(
        "**Fehler 1 – stilles Abschneiden:** `all-MiniLM-L6-v2` verarbeitet höchstens 256 "
        "Wordpiece-Tokens. Deutscher Fachtext zerfällt im englischen Vokabular in sehr viele "
        "Teilwörter, sodass ein Großteil der 1000-Zeichen-Chunks abgeschnitten wurde – der "
        "Rest des Chunks war für die Suche unsichtbar. Das multilinguale E5-Modell hat ein "
        "passendes Vokabular und 512 Tokens Kontext."
    ),
    md("## 2. Fehler 2 – L2-Distanz als Kosinus interpretiert"),
    code(
        "# Chroma nutzt standardmäßig die quadrierte L2-Distanz. Für normierte Vektoren gilt\n"
        "# d = 2 - 2·cos  →  der alte Code rechnete 'similarity = 1 - d = 2·cos - 1'.\n"
        "cos = np.linspace(0, 1, 6)\n"
        "pd.DataFrame({'echte Kosinus-Ähnlichkeit': cos, 'alter Score (1 - L2²)': 2 * cos - 1}).round(2)"
    ),
    md(
        "Alle Treffer mit einer Kosinus-Ähnlichkeit unter 0,5 bekamen einen **negativen** Score "
        "und wurden vom Filter `score >= 0` verworfen – deshalb lieferte die Beispielanfrage im "
        "alten Notebook nur 1 statt 5 Treffer. Lösung: Collection mit `space='cosine'`."
    ),
    md(
        "## 3. Retrieval-Evaluation\n\n"
        "Testset: 34 beantwortbare Fragen (12 ISE, 14 Monitoringbericht, 8 UBA), in "
        "Alltagssprache formuliert, mit Belegseiten, die reproduzierbar aus Evidenz-Mustern "
        "bestimmt wurden (`energy-rag annotate`). Ein Treffer zählt, wenn ein Chunk von einer "
        "Belegseite des richtigen Dokuments stammt."
    ),
    code(
        "results = json.loads((PROJECT_ROOT / 'docs/results/retrieval_metrics.json').read_text(encoding='utf-8'))\n"
        "metrics = pd.DataFrame([{'Konfiguration': r['label'], **r['metrics'],\n"
        "                         'Latenz (ms)': round(r['mean_latency_ms'])} for r in results]).set_index('Konfiguration')\n"
        "metrics.style.format({c: '{:.2f}' for c in metrics.columns if c != 'Latenz (ms)'})"
    ),
    code(
        "labels = [r['experiment'] for r in results]\n"
        "hit1 = [r['metrics']['Hit@1'] for r in results]\n"
        "hit5 = [r['metrics']['Hit@5'] for r in results]\n"
        "y = np.arange(len(labels))[::-1]\n"
        "fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)\n"
        "ax.barh(y + 0.19, hit5, height=0.36, color='#2a78d6', label='Hit@5', edgecolor='white', linewidth=2)\n"
        "ax.barh(y - 0.19, hit1, height=0.36, color='#eb6834', label='Hit@1', edgecolor='white', linewidth=2)\n"
        "for yi, v in zip(y, hit5):\n"
        "    ax.text(v + 0.01, yi + 0.19, f'{v:.0%}', va='center', fontsize=9, color='#333')\n"
        "ax.set_yticks(y, labels)\n"
        "ax.set_xlim(0, 1.08)\n"
        "ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))\n"
        "ax.grid(axis='x', color='#e5e5e5'); ax.set_axisbelow(True)\n"
        "for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)\n"
        "ax.set_title('Anteil der Fragen mit Belegseite unter den Top-k (n = 34)', loc='left', fontsize=11)\n"
        "ax.legend(frameon=False, loc='lower right')\n"
        "fig.tight_layout(); fig.savefig(IMG / 'retrieval_comparison.png', bbox_inches='tight')\n"
        "plt.show()"
    ),
    md("## 4. Fehleranalyse je Frage"),
    code(
        "items = {i.id: i for i in load_testset(PROJECT_ROOT / 'data/eval/testset.jsonl') if i.answerable}\n"
        "ranks = pd.DataFrame({r['experiment']: r['ranks'] for r in results})\n"
        "ranks.insert(0, 'Frage', [items[i].question for i in ranks.index])\n"
        "best = results[-1]['experiment']\n"
        "print('Rang der ersten Belegseite (leer = nicht in Top-10)')\n"
        "ranks.sort_values(best, na_position='first').head(12)"
    ),
    code(
        "by_doc = ranks.assign(Dokument=[i.split('-')[0] for i in ranks.index])\n"
        "(by_doc.drop(columns='Frage').groupby('Dokument')\n"
        "      .agg(lambda s: (s.fillna(99) <= 5).mean()).round(2)\n"
        "      .rename(index={'ise': 'Fraunhofer ISE', 'mon': 'Monitoringbericht', 'uba': 'UBA 2050'}))"
    ),
    md("## 5. Beispiel: Antwort mit Quellenangaben"),
    code(
        "from energy_rag.pipeline import RAGPipeline\n\n"
        "pipeline = RAGPipeline.from_settings(settings)\n"
        "answer = pipeline.ask('Wie teuer ist Strom aus neuen Windkraftanlagen an Land?')\n"
        "print(f'Generator: {answer.model}\\n')\n"
        "print(answer.text, '\\n')\n"
        "for i, s in enumerate(answer.sources, 1):\n"
        "    print(f'[{i}] {s.chunk.citation}  (Score {s.score:.2f})')"
    ),
    md(
        "Ohne `ANTHROPIC_API_KEY` zeigt die Pipeline die besten Fundstellen extraktiv an; mit Key "
        "formuliert Claude eine Antwort, die jede Aussage mit `[Nr]` belegt."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(exist_ok=True)
NotebookClient(
    nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(OUT.parent)}}
).execute()
nbformat.write(nb, OUT)
print(f"geschrieben: {OUT}")
