"""
Module 10: Streamlit Interface

Interactive web dashboard for Research Novelty Detector.
"""

from pathlib import Path

import streamlit as st

from src.config import (MAX_QUERIES, RESULTS_PER_QUERY, TOP_K_COMPARE,
                        api_key_status, get_groq_model)
from src.pdf_processor import OCRRequiredError, PDFProcessingError
from src.pipeline import analyze_paper, run_research

st.set_page_config(
    page_title="Research Novelty Detector",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Research Novelty Detector")
st.markdown(
    "Analyze uploaded research papers, extract key sections locally, retrieve related work, "
    "compute semantic similarity, and generate evidence-grounded comparison reports."
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    keys = api_key_status()
    st.write(
        f"**Tavily API Key (.env):** {'✅ Configured' if keys['TAVILY_API_KEY'] else '⚠️ Not set / Placeholder'}"
    )
    st.write(
        f"**Groq API Key (.env):** {'✅ Configured' if keys['GROQ_API_KEY'] else '⚠️ Not set / Placeholder'}"
    )

    with st.expander("🔑 Enter API Keys (Optional Override)", expanded=not (keys["TAVILY_API_KEY"] and keys["GROQ_API_KEY"])):
        st.caption("Keys entered here are used only for this session and are never saved to disk.")
        ui_tavily_key = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")
        ui_groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        ui_groq_model = st.text_input("Groq Model", value=get_groq_model())

    st.markdown("---")
    st.subheader("Search & Ranking Parameters")
    max_queries = st.slider("Max Search Queries", 1, 5, MAX_QUERIES)
    results_per_query = st.slider("Results per Query", 1, 10, RESULTS_PER_QUERY)
    top_k = st.slider("Top Papers for LLM Comparison", 1, 5, TOP_K_COMPARE)
    use_llm = st.checkbox("Run Groq LLM Comparison", value=True)

    st.markdown("---")
    st.caption(
        "Note: Semantic similarity reflects topical closeness and is NOT a novelty percentage. "
        "Web search is scoped and conservative to avoid unnecessary API costs."
    )

# File upload
uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "research_results" not in st.session_state:
    st.session_state.research_results = None

# Sample Paper Quick Load Option
sample_path = Path("data/uploaded_papers/sample.pdf")
if sample_path.exists() and not uploaded_file:
    if st.button("📄 Load Included Sample Paper (Toward User-Aware Virtual Agents)"):
        pdf_bytes = sample_path.read_bytes()
        try:
            with st.spinner("Extracting text and analyzing sections locally..."):
                st.session_state.analysis = analyze_paper(
                    pdf_bytes=pdf_bytes,
                    file_name="sample.pdf",
                    max_queries=max_queries
                )
                st.session_state.research_results = None
            st.success("Sample paper processed locally!")
        except Exception as e:
            st.error(f"Error processing sample: {e}")

if uploaded_file is not None:
    if st.button("1️⃣ Extract Sections & Generate Queries", type="primary"):
        pdf_bytes = uploaded_file.read()
        try:
            with st.spinner("Processing PDF locally (PyMuPDF & KeyBERT)..."):
                st.session_state.analysis = analyze_paper(
                    pdf_bytes=pdf_bytes,
                    file_name=uploaded_file.name,
                    max_queries=max_queries
                )
                st.session_state.research_results = None
            st.success("Extraction complete!")
        except OCRRequiredError:
            st.error("No extractable text found. This PDF may be a scanned document requiring OCR.")
        except PDFProcessingError as e:
            st.error(f"PDF Processing Error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Display Stage 1 results if available
if st.session_state.analysis:
    analysis = st.session_state.analysis
    paper = analysis["paper"]
    sections = analysis["sections"]
    headings = analysis["headings"]
    queries = analysis["queries"]

    st.markdown("---")
    st.header("📄 Uploaded Paper Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Detected Title:** {paper.get('title') or 'N/A'}")
        st.write(f"**Total Pages:** {paper.get('total_pages')}")
    with col2:
        st.write(f"**File Name:** {paper.get('file_name')}")
        if paper.get("warnings"):
            for w in paper["warnings"]:
                st.warning(w)

    with st.expander("📑 Extracted Paper Sections", expanded=False):
        for name, text in sections.items():
            st.subheader(name.replace("_", " ").title())
            if text:
                st.caption(f"Source heading(s): {', '.join(headings.get(name, [])) or 'Inferred'}")
                st.text_area(
                    label=f"{name} content",
                    value=text[:1500] + ("..." if len(text) > 1500 else ""),
                    height=150,
                    key=f"sec_{name}",
                    label_visibility="collapsed",
                )
            else:
                st.write("*Not found in document.*")

    st.subheader("🔍 Generated Academic Search Queries")
    editable_queries = []
    for idx, q in enumerate(queries):
        col_q, col_type = st.columns([4, 1])
        with col_q:
            new_q = st.text_input(f"Query {idx+1}", value=q["query"], key=f"q_{idx}")
            editable_queries.append({
                "query": new_q,
                "type": q.get("type", "custom"),
                "keyphrases": q.get("keyphrases", []),
            })
        with col_type:
            st.write(f"Type: `{q.get('type')}`")

    st.markdown("---")
    st.header("2️⃣ Literature Search & Novelty Analysis")
    if st.button("🚀 Run Related Work Search & Comparison", type="primary"):
        status_box = st.empty()

        def update_progress(msg):
            status_box.info(f"⏳ {msg}")

        try:
            with st.spinner("Executing pipeline..."):
                res = run_research(
                    analysis,
                    queries=editable_queries,
                    max_queries=max_queries,
                    results_per_query=results_per_query,
                    top_k=top_k,
                    use_llm=use_llm,
                    tavily_api_key=ui_tavily_key.strip() or None,
                    groq_api_key=ui_groq_key.strip() or None,
                    groq_model=ui_groq_model.strip() or None,
                    progress=update_progress,
                    save_report=True,
                )
                st.session_state.research_results = res
            if res.get("errors"):
                status_box.warning("Analysis finished with warnings/errors (see below).")
            else:
                status_box.success("✅ Analysis completed successfully!")
        except Exception as e:
            status_box.error(f"Error during analysis: {e}")

# Display Stage 2 results if available
if st.session_state.research_results:
    res = st.session_state.research_results
    ranked = res.get("ranked", [])
    comparison = res.get("comparison") or {}

    if res.get("errors"):
        for err in res["errors"]:
            st.warning(f"⚠️ {err}")

    st.markdown("---")
    st.header("📊 Similarity Ranking & Retrieved Papers")
    st.info(
        "Semantic similarity uses local embeddings (`all-MiniLM-L6-v2`) to estimate topical relevance. "
        "It does NOT indicate originality or produce novelty percentages."
    )

    if ranked:
        table_data = []
        for r in ranked:
            title_val = r.get("title") or ""
            table_data.append({
                "Rank": str(r.get("rank")) if r.get("rank") else "Excluded (Self)",
                "Title": title_val[:70] + ("..." if len(title_val) > 70 else ""),
                "Similarity": f"{r.get('overall_similarity'):.2f}" if r.get("overall_similarity") is not None else "N/A",
                "Band": r.get("band"),
                "Evidence": r.get("evidence_description"),
                "URL": r.get("url"),
            })
        st.dataframe(table_data, width="stretch")

        with st.expander("🔬 Detailed Aspect Breakdown & Source Passages"):
            for item in ranked:
                st.markdown(f"### {item.get('title')}")
                st.write(f"**URL:** [{item.get('url')}]({item.get('url')})")
                st.write(f"**Evidence Level:** `{item.get('evidence_level')}` ({item.get('evidence_description')})")
                st.write(f"**Explanation:** {item.get('explanation')}")
                st.write("**Aspect Similarity:**")
                for asp_name, asp_data in item.get("aspects", {}).items():
                    score_str = f"{asp_data.get('score'):.2f}" if asp_data.get("score") is not None else "N/A"
                    st.write(f"- **{asp_data.get('label')}:** {score_str} ({asp_data.get('band')})")
                    if asp_data.get("evidence_passage"):
                        st.caption(f"> {asp_data['evidence_passage'][:300]}...")
                st.divider()
    else:
        st.write("*No related papers were retrieved. Check your Tavily API key or search queries.*")

    st.header("🤖 Research Comparison & Novelty Insights")
    if comparison.get("markdown"):
        st.markdown(comparison["markdown"])
    else:
        st.write("No comparison output generated.")

    if comparison.get("warnings"):
        for w in comparison["warnings"]:
            st.warning(f"⚠️ {w}")

    # Report Download
    st.markdown("---")
    st.header("📥 Download Full Report")
    report_md = res.get("report_markdown", "")
    st.download_button(
        label="Download Markdown Report (.md)",
        data=report_md,
        file_name="research_novelty_report.md",
        mime="text/markdown",
    )
