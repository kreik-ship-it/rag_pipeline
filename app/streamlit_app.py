"""Streamlit-Oberfläche: Frage stellen, Antwort mit Quellen anzeigen.

Start: ``streamlit run app/streamlit_app.py`` (vorher ``energy-rag ingest``).
"""

from __future__ import annotations

import streamlit as st

from energy_rag.config import Settings
from energy_rag.generation import has_anthropic_credentials
from energy_rag.pipeline import RAGPipeline

EXAMPLES = [
    "Wie teuer ist Strom aus neuen PV-Freiflächenanlagen im Jahr 2024?",
    "Wie stark sind die Netzentgelte für Haushalte 2024 gestiegen?",
    "Welche Rolle spielen Kapazitätsmärkte für die Transformation bis 2050?",
]

st.set_page_config(page_title="Energy-RAG", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner="Lade Modelle und Index …")
def load_pipeline(mode: str, rerank: bool) -> RAGPipeline:
    settings = Settings().with_overrides(retrieval_mode=mode, use_reranker=rerank)
    return RAGPipeline.from_settings(settings)


with st.sidebar:
    st.header("Einstellungen")
    mode = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"], index=0)
    rerank = st.toggle("Cross-Encoder-Reranking", value=True)
    top_k = st.slider("Anzahl Quellen (top-k)", 3, 10, 5)
    if has_anthropic_credentials():
        st.success(f"Generierung: {Settings().llm_model}")
    else:
        st.warning("Kein ANTHROPIC_API_KEY – es werden nur Fundstellen angezeigt.")

st.title("⚡ Energy-RAG")
st.caption(
    "Fragen an drei Studien zum deutschen Strommarkt (Monitoringbericht 2024, "
    "Fraunhofer-ISE-Stromgestehungskosten 2024, UBA-Transformationsstudie 2050)."
)

example = st.selectbox("Beispielfrage", ["–"] + EXAMPLES)
question = st.text_input("Frage", value="" if example == "–" else example)

if question:
    try:
        pipeline = load_pipeline(mode, rerank)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    with st.spinner("Suche und generiere …"):
        answer = pipeline.ask(question, top_k=top_k)

    st.subheader("Antwort")
    st.markdown(answer.text)

    st.subheader("Quellen")
    cited = set(answer.cited)
    for i, src in enumerate(answer.sources, start=1):
        marker = "✅ zitiert" if i in cited else ""
        with st.expander(f"[{i}] {src.chunk.citation} · Score {src.score:.3f} {marker}"):
            st.write(src.chunk.text)
