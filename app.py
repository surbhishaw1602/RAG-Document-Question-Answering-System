import os
import time
import tempfile
import html

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not configured in your .env file.")
    st.stop()


# ============================================================
# 2. Page configuration
# ============================================================

st.set_page_config(
    page_title="DocuMind AI · RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. Custom CSS — dark, gradient, "sophisticated AI product" look
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* App background */
    .stApp {
        background: radial-gradient(circle at 15% 10%, #1a1440 0%, #0d0b1e 45%, #0a0918 100%);
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Hero header */
    .hero {
        text-align: center;
        padding: 34px 20px 18px 20px;
    }
    .hero-badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(124,58,237,0.18), rgba(56,189,248,0.18));
        border: 1px solid rgba(167,139,250,0.35);
        color: #c4b5fd;
        font-size: 12.5px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #f472b6 0%, #a78bfa 35%, #38bdf8 70%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 16.5px;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Section card */
    .glass-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px 26px;
        backdrop-filter: blur(6px);
        margin-bottom: 18px;
    }

    .section-label {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 10px;
    }

    /* Metric pills */
    .stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 6px; }
    .stat-pill {
        flex: 1;
        min-width: 130px;
        background: linear-gradient(145deg, rgba(124,58,237,0.14), rgba(56,189,248,0.08));
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 14px;
        padding: 14px 16px;
    }
    .stat-num {
        font-size: 26px;
        font-weight: 800;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-label {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Chat bubbles */
    .qa-block { margin-bottom: 22px; }
    .q-bubble {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 16px 16px 4px 16px;
        display: inline-block;
        max-width: 85%;
        font-weight: 500;
        font-size: 15px;
        box-shadow: 0 4px 16px rgba(124,58,237,0.25);
    }
    .q-wrap { text-align: right; margin-bottom: 10px; }
    .a-bubble {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        color: #e2e8f0;
        padding: 16px 20px;
        border-radius: 16px 16px 16px 4px;
        font-size: 15.5px;
        line-height: 1.65;
        max-width: 92%;
    }
    .a-label {
        font-size: 11.5px;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: rgba(52,211,153,0.1);
        border: 1px solid rgba(52,211,153,0.3);
        color: #6ee7b7;
        font-size: 11.5px;
        font-family: 'JetBrains Mono', monospace;
        padding: 3px 10px;
        border-radius: 8px;
        margin: 3px 6px 0 0;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #38bdf8);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 10px 22px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(124,58,237,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124,58,237,0.45);
    }

    /* Text input */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        color: #f1f5f9;
        padding: 12px 14px;
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(124,58,237,0.05);
        border: 1.5px dashed rgba(167,139,250,0.4);
        border-radius: 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #140f2e 0%, #0d0b1e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .suggested-q {
        background: rgba(56,189,248,0.08);
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 13.5px;
        color: #7dd3fc;
        margin-bottom: 6px;
        cursor: pointer;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">✦ Gemini-Powered Retrieval</div>
        <div class="hero-title">DocuMind AI</div>
        <div class="hero-subtitle">
            Upload a PDF, and have a real conversation with it.
            Grounded answers, zero hallucination guesswork, full source transparency.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. Session state
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: question, answer, sources, latency
if "db" not in st.session_state:
    st.session_state.db = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# 6. Sidebar — settings + live stats
# ============================================================

with st.sidebar:

    st.markdown("### ⚙️ Configuration")

    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
    top_k = st.slider("Chunks Retrieved (k)", 2, 10, 4, 1)
    temperature = st.slider("Answer Creativity", 0.0, 1.0, 0.0, 0.1)

    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.caption("Embeddings: `gemini-embedding-2`")
    st.caption("LLM: `gemini-3.1-flash-lite`")

    if st.session_state.doc_stats:
        st.markdown("---")
        st.markdown("### 📊 Document Stats")
        s = st.session_state.doc_stats
        st.caption(f"📄 **{s['file_name']}**")
        st.caption(f"Pages: {s['pages']}  ·  Chunks: {s['chunks']}")
        st.caption(f"Indexed in {s['index_time']:.1f}s")

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        history_text = "\n\n".join(
            f"Q: {h['question']}\nA: {h['answer']}"
            for h in st.session_state.chat_history
        )
        st.download_button(
            "⬇️ Export Chat (.txt)",
            data=history_text,
            file_name="documind_chat_history.txt",
            mime="text/plain",
        )


# ============================================================
# 7. Upload section
# ============================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📄 Upload Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 8. Process uploaded PDF (only re-index on new file or new settings)
# ============================================================

if uploaded_file is not None:

    needs_indexing = (
        st.session_state.db is None
        or st.session_state.file_name != uploaded_file.name
    )

    if needs_indexing:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            pdf_path = temp_file.name

        start_time = time.time()

        try:
            with st.spinner("📖 Reading PDF..."):
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
        except Exception as e:
            st.error(f"❌ Error while reading PDF: {str(e)}")
            st.stop()

        with st.spinner("✂️ Splitting document into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            docs = text_splitter.split_documents(documents)

        with st.spinner("🧠 Creating Gemini embeddings & building index..."):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-2",
                    google_api_key=GEMINI_API_KEY,
                )
                db = FAISS.from_documents(docs, embeddings)
            except Exception as e:
                st.error("❌ Error while creating Gemini embeddings.")
                st.error(str(e))
                st.stop()

        index_time = time.time() - start_time

        st.session_state.db = db
        st.session_state.file_name = uploaded_file.name
        st.session_state.doc_stats = {
            "file_name": uploaded_file.name,
            "pages": len(documents),
            "chunks": len(docs),
            "index_time": index_time,
        }
        st.session_state.chat_history = []

    # --------------------------------------------------------
    # Stats row
    # --------------------------------------------------------

    s = st.session_state.doc_stats
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-label">✅ Document Indexed — {html.escape(s['file_name'])}</div>
            <div class="stat-row">
                <div class="stat-pill"><div class="stat-num">{s['pages']}</div><div class="stat-label">Pages</div></div>
                <div class="stat-pill"><div class="stat-num">{s['chunks']}</div><div class="stat-label">Chunks</div></div>
                <div class="stat-pill"><div class="stat-num">{s['index_time']:.1f}s</div><div class="stat-label">Index Time</div></div>
                <div class="stat-pill"><div class="stat-num">{top_k}</div><div class="stat-label">Retrieved / Query</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    db = st.session_state.db
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=temperature,
            google_api_key=GEMINI_API_KEY,
        )
    except Exception as e:
        st.error("❌ Error while initializing Gemini.")
        st.error(str(e))
        st.stop()

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful, precise document question-answering assistant.

        Answer the question using ONLY the information provided in the context below.
        Do not use outside knowledge. Be concise but complete.

        If the answer cannot be found in the context, say exactly:
        "The information is not available in the document."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(d.page_content for d in retrieved_docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------------
    # Suggested questions
    # --------------------------------------------------------

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">💬 Ask a Question</div>', unsafe_allow_html=True)

    suggestions = [
        "What is this document about?",
        "Summarize the key points.",
        "What are the main conclusions?",
    ]
    cols = st.columns(len(suggestions))
    clicked_suggestion = None
    for col, s_text in zip(cols, suggestions):
        with col:
            if st.button(s_text, key=f"sugg_{s_text}", use_container_width=True):
                clicked_suggestion = s_text

    query = st.text_input(
        "Enter your question:",
        value=clicked_suggestion or "",
        placeholder="Ask anything about the document...",
        label_visibility="collapsed",
    )

    ask_col, _ = st.columns([1, 4])
    with ask_col:
        ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if ask_clicked or clicked_suggestion:
        final_query = query.strip()

        if final_query == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤔 Searching the document..."):
                try:
                    t0 = time.time()
                    retrieved = retriever.invoke(final_query)
                    answer = qa_chain.invoke(final_query)
                    latency = time.time() - t0
                except Exception as e:
                    st.error("❌ Error while asking Gemini.")
                    st.error(str(e))
                    st.stop()

            sources = []
            for d in retrieved:
                page = d.metadata.get("page", None)
                if page is not None:
                    sources.append(f"page {page + 1}")
            sources = sorted(set(sources), key=lambda x: int(x.split()[-1]))

            st.session_state.chat_history.append({
                "question": final_query,
                "answer": answer,
                "sources": sources,
                "latency": latency,
            })

    # --------------------------------------------------------
    # Chat history display (most recent first)
    # --------------------------------------------------------

    if st.session_state.chat_history:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🗨️ Conversation</div>', unsafe_allow_html=True)

        for turn in reversed(st.session_state.chat_history):
            safe_q = html.escape(turn["question"])
            safe_a = html.escape(turn["answer"])
            source_chips = "".join(
                f'<span class="source-chip">📎 {html.escape(s)}</span>'
                for s in turn["sources"]
            )

            st.markdown(
                f"""
                <div class="qa-block">
                    <div class="q-wrap"><span class="q-bubble">{safe_q}</span></div>
                    <div class="a-bubble">
                        <div class="a-label">DocuMind · {turn['latency']:.1f}s</div>
                        {safe_a}
                        <div style="margin-top:10px;">{source_chips}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding: 46px 20px;">
            <div style="font-size: 44px; margin-bottom: 10px;">📤</div>
            <div style="color:#cbd5e1; font-size:16px; font-weight:600;">
                Upload a PDF above to start the conversation
            </div>
            <div style="color:#64748b; font-size:13.5px; margin-top:6px;">
                Your document is chunked, embedded, and indexed locally in this session.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )