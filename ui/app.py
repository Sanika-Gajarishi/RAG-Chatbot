import streamlit as st
import requests

# ==========================================================
# Configuration
# ==========================================================

API_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide"
)

# ==========================================================
# Header
# ==========================================================

st.title("📄 Multilingual Document Q&A with Citations")

st.markdown(
"""
Ask questions in **English, Hindi, Marathi**, or any supported language.

The assistant answers **only from the uploaded documents** and cites the relevant sources.
"""
)

# ==========================================================
# Fetch available documents (for dropdowns in the compare tab)
# ==========================================================

try:
    _docs_response = requests.get(f"{API_URL}/documents", timeout=5)
    AVAILABLE_DOCUMENTS = _docs_response.json().get("documents", []) if _docs_response.status_code == 200 else []
except requests.exceptions.RequestException:
    AVAILABLE_DOCUMENTS = []

# ==========================================================
# Tabs
# ==========================================================

tab1, tab2 = st.tabs(
    [
        "📄 Ask Question",
        "⚖️ Compare Documents"
    ]
)

# ==========================================================
# ASK QUESTION
# ==========================================================

with tab1:

    st.subheader("Ask a Question")

    question = st.text_area(
        "Enter your question",
        height=120,
        placeholder="Example: What is Generative AI?"
    )

    use_rerank = st.checkbox(
        "Use cross-encoder reranker (slower, more precise)",
        value=False
    )

    if st.button(
        "🔍 Search Documents",
        use_container_width=True
    ):

        if question.strip() == "":

            st.warning("Please enter a question.")

            st.stop()

        with st.spinner("Searching..."):

            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "rerank": use_rerank
                }
            )

        if response.status_code != 200:

            st.error("Unable to connect to the API.")

            st.stop()

        result = response.json()

        st.markdown("---")

        confidence = result.get("confidence")
        needs_review = result.get("needs_review", False)

        if confidence is not None:

            if needs_review:
                st.warning(
                    f"⚠️ Low retrieval confidence ({confidence:.2f}). "
                    "This answer may be unreliable — please verify it against "
                    "the cited sources before trusting it, or ask a human reviewer."
                )
            else:
                st.caption(f"Retrieval confidence: {confidence:.2f}")

        st.markdown("## 📖 Answer")

        st.write(result["answer"])

        # -----------------------------------
        # Sources (Collapsed)
        # -----------------------------------

        with st.expander("📚 View Sources"):

            if len(result["citations"]) == 0:

                st.write("No citations available.")

            else:

                for citation in result["citations"]:

                    st.markdown(
                        f"### 📄 {citation['document']}"
                    )

                    st.write(
                        f"**Page:** {citation['page']}"
                    )

                    st.write(
                        f"**Chunk:** {citation['chunk_id']}"
                    )

                    st.write("**Snippet:**")

                    st.info(
                        citation["snippet"]
                    )

                    st.divider()
    # ==========================================================
# COMPARE DOCUMENTS
# ==========================================================

with tab2:

    st.subheader("Compare Two Documents")

    st.write(
        "Select two documents and provide a topic to check whether they agree, conflict, or partially agree."
    )

    if AVAILABLE_DOCUMENTS:

        document1 = st.selectbox("Document 1", AVAILABLE_DOCUMENTS, index=0)

        document2 = st.selectbox(
            "Document 2",
            AVAILABLE_DOCUMENTS,
            index=min(1, len(AVAILABLE_DOCUMENTS) - 1)
        )

    else:

        document1 = st.text_input(
            "Document 1",
            placeholder="Example: ICSICE.pdf"
        )

        document2 = st.text_input(
            "Document 2",
            placeholder="Example: Foundations of GenAI & Tools.pdf"
        )

    topic = st.text_input(
        "Topic",
        placeholder="Example: Generative AI"
    )

    if st.button(
        "⚖️ Compare Documents",
        use_container_width=True
    ):

        if (
            document1.strip() == ""
            or document2.strip() == ""
            or topic.strip() == ""
        ):

            st.warning("Please complete all fields.")
            st.stop()

        with st.spinner("Comparing documents..."):

            response = requests.post(

                f"{API_URL}/contradict",

                json={

                    "document1": document1,

                    "document2": document2,

                    "topic": topic

                }

            )

        if response.status_code != 200:

            st.error("Unable to compare documents.")
            st.stop()

        result = response.json()

        st.markdown("---")

        st.markdown("## ⚖️ Comparison Result")

        comparison = result["comparison"]

        if isinstance(comparison, dict):

            status = comparison.get("status", "Unknown")

            badge = {
                "Agree": "🟢",
                "Conflict": "🔴",
                "Partial Agreement": "🟡",
                "Not Enough Information": "⚪"
            }.get(status, "⚪")

            st.markdown(f"**Status:** {badge} {status}")

            if comparison.get("reason"):
                st.markdown(f"**Reason:** {comparison['reason']}")

            if comparison.get("summary"):
                st.markdown("**Details:**")
                st.write(comparison["summary"])

        else:
            st.write(comparison)

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "Built using FastAPI • FAISS • Sentence Transformers • Gemini • Streamlit"
)