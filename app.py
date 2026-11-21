from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from course_rag import CourseRAG, RAGConfig


st.set_page_config(page_title="Course Q&A", page_icon="📚", layout="wide")
PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading the course search model...")
def load_rag() -> CourseRAG:
    return CourseRAG(
        RAGConfig(
            course_dir=PROJECT_DIR / "data" / "courses",
            chroma_dir=PROJECT_DIR / "data" / "chroma_db",
        )
    )


def course_chunk_counts(rag: CourseRAG) -> Counter:
    if rag.collection.count() == 0:
        return Counter()
    result = rag.collection.get(include=["metadatas"], limit=rag.collection.count())
    return Counter(
        metadata["course"]
        for metadata in result["metadatas"]
        if metadata and metadata.get("course")
    )


st.title("Course Q&A")
st.write("Ask questions about the course materials. Answers use retrieved passages and show their source files.")

try:
    rag = load_rag()
except Exception as exc:
    st.error(f"Could not start the RAG system: {exc}")
    st.stop()

with st.sidebar:
    st.header("Course index")
    counts = course_chunk_counts(rag)
    indexed_chunks = sum(counts.values())
    st.metric("Indexed chunks", indexed_chunks)
    if counts:
        chart_data = pd.DataFrame({"Course": list(counts), "Chunks": list(counts.values())}).set_index("Course")
        st.bar_chart(chart_data)
    else:
        st.info("Build the index to see course coverage.")

    if st.button("Build or refresh index", width="stretch"):
        with st.spinner("Reading course files and building the vector database..."):
            try:
                st.session_state["last_ingestion_report"] = rag.ingest()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    last_report = st.session_state.get("last_ingestion_report")
    if last_report:
        st.success(f"Indexed {last_report['chunks']} text chunks from {last_report['files']} files.")
        if last_report["warnings"]:
            with st.expander("Files needing attention"):
                for warning in last_report["warnings"]:
                    st.write(f"- {warning}")

courses = rag.list_courses()
if rag.collection.count() == 0:
    st.warning("The course database is empty. Use **Build or refresh index** in the sidebar first.")

with st.form("question_form"):
    question = st.text_area("Your question", placeholder="Ask in English or Arabic", height=100)
    selected_course = st.selectbox("Search in", ["All courses"] + courses)
    top_k = st.slider("Number of source passages", min_value=1, max_value=8, value=4)
    submitted = st.form_submit_button("Get answer", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    elif rag.collection.count() == 0:
        st.error("Build the course index before asking questions.")
    else:
        with st.spinner("Searching the course materials and writing an answer..."):
            try:
                result = rag.answer(
                    question.strip(),
                    top_k=top_k,
                    course=None if selected_course == "All courses" else selected_course,
                )
                st.subheader("Answer")
                st.markdown(result["answer"])
                if result["sources"]:
                    st.subheader("Source relevance")
                    similarity_data = pd.DataFrame(
                        {
                            "Similarity": [source["similarity"] for source in result["sources"]],
                        },
                        index=[source["citation"] for source in result["sources"]],
                    )
                    st.bar_chart(similarity_data)
                    st.subheader("Sources")
                    for source in result["sources"]:
                        with st.expander(
                            f"{source['citation']} · {source['course']} · {source['file']} · {source['location']}"
                        ):
                            st.write(f"Similarity: {source['similarity']:.3f}")
                            st.code(source["relative_path"], language=None)
            except Exception as exc:
                st.error(f"The question could not be answered: {exc}")

st.divider()
st.caption("The system is designed to answer from the indexed course materials and cite retrieved sources.")
