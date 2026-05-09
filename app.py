"""
IntelliBot 🧠 — Hierarchical RAG-Based Document Q&A System
Main Streamlit application with premium UI.
"""

import streamlit as st
import time
import base64
import os
from rag_engine import RAGPipeline
from llm_client import generate_answer, stream_to_string

# ─────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliBot 🧠 | AI Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Logo Helper
# ─────────────────────────────────────────────────────────
def get_logo_base64():
    """Load logo as base64 for embedding in HTML."""
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = get_logo_base64()

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #7C3AED;
    --primary-light: #A78BFA;
    --primary-dark: #5B21B6;
    --accent: #06B6D4;
    --bg-dark: #0F0F1A;
    --bg-card: #1A1A2E;
    --bg-card-hover: #222240;
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --border: rgba(124, 58, 237, 0.2);
    --glow: rgba(124, 58, 237, 0.4);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 1200px; }

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(6,182,212,0.1) 100%);
    border-radius: 20px;
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(124,58,237,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 60%, rgba(6,182,212,0.06) 0%, transparent 60%);
    animation: pulse-bg 8s ease-in-out infinite alternate;
}
@keyframes pulse-bg { 0% { transform: scale(1); } 100% { transform: scale(1.05); } }

.hero-logo {
    width: 80px;
    height: 80px;
    border-radius: 18px;
    margin: 0 auto 0.8rem;
    display: block;
    box-shadow: 0 8px 32px rgba(124,58,237,0.35);
    border: 2px solid rgba(124,58,237,0.3);
    position: relative;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #A78BFA 0%, #06B6D4 50%, #7C3AED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    position: relative;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
    position: relative;
}

/* Stat cards */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1rem 0; }
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}
.stat-card:hover {
    border-color: var(--primary-light);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,58,237,0.15);
}
.stat-number { font-size: 1.6rem; font-weight: 700; color: var(--primary-light); }
.stat-label { font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.5px; }

/* Chat messages */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 0.5rem;
    scrollbar-width: thin;
    scrollbar-color: var(--primary-dark) transparent;
}
.user-msg, .bot-msg {
    padding: 1rem 1.2rem;
    border-radius: 16px;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.user-msg {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(124,58,237,0.08));
    border: 1px solid rgba(124,58,237,0.3);
    margin-left: 2rem;
}
.bot-msg {
    background: var(--bg-card);
    border: 1px solid var(--border);
    margin-right: 2rem;
}

/* Hierarchy tree */
.hierarchy-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
    color: var(--text-primary);
}

/* Source cards */
.source-card {
    background: rgba(6,182,212,0.05);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    transition: all 0.3s ease;
}
.source-card:hover { border-color: var(--accent); background: rgba(6,182,212,0.1); }

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stFileUploader {
    border: 2px dashed rgba(124,58,237,0.4);
    border-radius: 14px;
    padding: 1rem;
    transition: border-color 0.3s;
}
section[data-testid="stSidebar"] .stFileUploader:hover {
    border-color: var(--primary-light);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(124,58,237,0.5) !important;
}

/* Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* Processing animation */
.processing-text {
    background: linear-gradient(90deg, var(--primary-light), var(--accent), var(--primary-light));
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 2s linear infinite;
    font-weight: 600;
}
@keyframes shimmer { to { background-position: 200% center; } }

/* Expander styling */
.streamlit-expanderHeader { font-weight: 600 !important; font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "rag_pipeline": None,
        "messages": [],
        "doc_processed": False,
        "doc_stats": {},
        "doc_title": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Auto-load API key from .env
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        st.success("🔑 API Key loaded", icon="✅")
    else:
        st.warning("⚠️ GROQ_API_KEY not found in .env file")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # PDF Upload
    st.markdown("## 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        help="Upload any PDF document to start asking questions.",
    )

    # Chunking settings
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("Chunk Size (words)", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap (words)", 20, 200, 80, 10)
        top_k = st.slider("Retrieved Chunks (top-k)", 3, 10, 5)
        bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.35, 0.05)

    # Process button
    if uploaded_file:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner(""):
                st.markdown(
                    '<p class="processing-text">⚡ Processing your document...</p>',
                    unsafe_allow_html=True,
                )

                pipeline = RAGPipeline(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                try:
                    stats = pipeline.process_pdf(uploaded_file, title=uploaded_file.name)
                except ValueError as e:
                    st.error(f"⚠️ {e}")
                    st.stop()

                st.session_state.rag_pipeline = pipeline
                st.session_state.doc_stats = stats
                st.session_state.doc_processed = True
                st.session_state.doc_title = uploaded_file.name
                st.session_state.messages = []

                st.success(f"✅ Processed {stats['total_pages']} pages → {stats['total_chunks']} chunks")
                st.rerun()

    elif not uploaded_file:
        st.info("📤 Upload a PDF to get started.")

    # Reset button
    if st.session_state.doc_processed:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        if st.button("🗑️ Reset", use_container_width=True):
            for key in ["rag_pipeline", "messages", "doc_processed", "doc_stats", "doc_title"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# ─────────────────────────────────────────────────────────
# Main Content Area
# ─────────────────────────────────────────────────────────

# Hero Header
logo_html = ""
if LOGO_B64:
    logo_html = f'<img src="data:image/png;base64,{LOGO_B64}" class="hero-logo" alt="IntelliBot Logo"/>'

st.markdown(f"""
<div class="hero-header">
    {logo_html}
    <h1 class="hero-title">🧠 IntelliBot</h1>
    <p class="hero-subtitle">Hierarchical RAG-Powered Document Intelligence</p>
</div>
""", unsafe_allow_html=True)


if not st.session_state.doc_processed:
    # Landing state — show instructions
    col1, col2, col3 = st.columns(3)
    features = [
        ("📤 Upload", "Upload any PDF document to start exploring its content with AI."),
        ("🌳 Hierarchical Index", "Auto-builds a topic tree for structured navigation."),
        ("🔍 Hybrid Retrieval", "BM25 + Semantic search for precise, relevant answers."),
    ]
    for col, (title, desc) in zip([col1, col2, col3], features):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="min-height:140px;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{title.split()[0]}</div>
                <div style="font-weight:600; color:var(--text-primary); margin-bottom:0.4rem;">{title.split(maxsplit=1)[1]}</div>
                <div style="font-size:0.82rem; color:var(--text-secondary);">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:2rem; color:var(--text-secondary);">
        <p style="font-size:1.1rem;">👈 Upload a PDF to get started</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Document is processed — show dashboard + chat
    pipeline: RAGPipeline = st.session_state.rag_pipeline
    stats = st.session_state.doc_stats
    doc_title = st.session_state.get("doc_title", "Document")

    # Stats bar
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card"><div class="stat-number">{stats['total_pages']}</div><div class="stat-label">Pages</div></div>
        <div class="stat-card"><div class="stat-number">{stats['total_chunks']}</div><div class="stat-label">Chunks</div></div>
        <div class="stat-card"><div class="stat-number">{stats['total_words']:,}</div><div class="stat-label">Words</div></div>
        <div class="stat-card"><div class="stat-number">{stats['hierarchy_nodes']}</div><div class="stat-label">Topics</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs: Chat | Hierarchy | Sources
    tab_chat, tab_hierarchy, tab_about = st.tabs(["💬 Chat", "🌳 Document Hierarchy", "ℹ️ About"])

    with tab_chat:
        # Display chat messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                if "sources" in msg:
                    with st.expander("📎 View Sources"):
                        for src in msg["sources"]:
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Pages {src['pages']}</strong> | Relevance: {src['score']}<br/>
                                <span style="color:var(--text-secondary);">{src['preview']}</span>
                            </div>
                            """, unsafe_allow_html=True)

        # Chat input
        user_query = st.chat_input("Ask anything about your document...")

        if user_query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.markdown(f'<div class="user-msg">🧑 {user_query}</div>', unsafe_allow_html=True)

            with st.spinner("🔍 Retrieving & generating..."):
                # Retrieve
                results = pipeline.query(user_query, top_k=top_k if 'top_k' in dir() else 5)

                # Convert retrieval results to the dict format expected by generate_answer
                context_chunks = []
                for r in results:
                    pages = r.chunk.page_numbers
                    context_chunks.append({
                        "text": str(r.chunk.text),
                        "start_page": min(pages) - 1 if pages else 0,
                        "end_page": max(pages) - 1 if pages else 0,
                        "topic_words": [r.chunk.section_title] if r.chunk.section_title else [],
                    })

                # Generate response via streaming
                try:
                    stream = generate_answer(
                        query=user_query,
                        context_chunks=context_chunks,
                        chat_history=st.session_state.messages,
                        book_title=doc_title,
                    )
                    response = stream_to_string(stream)
                except Exception as e:
                    response = f"❌ Error generating response: {str(e)}"

                # Build sources for display
                sources = []
                for r in results:
                    sources.append({
                        "pages": ", ".join(str(p) for p in r.chunk.page_numbers),
                        "score": f"{r.score:.2f}",
                        "preview": r.chunk.text[:150] + "...",
                    })

                # Add bot message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                })

            st.rerun()

    with tab_hierarchy:
        st.markdown("### 🌳 Document Topic Hierarchy")
        st.markdown(
            '<p style="color:var(--text-secondary);font-size:0.85rem;">'
            "Auto-detected structure from your document</p>",
            unsafe_allow_html=True,
        )

        hierarchy_text = pipeline.get_hierarchy_text()
        st.markdown(f'<div class="hierarchy-box">{hierarchy_text}</div>', unsafe_allow_html=True)



    with tab_about:
        st.markdown("""
        ### 🛠️ How IntelliBot Works

        **IntelliBot** uses a **Hierarchical Retrieval-Augmented Generation (RAG)** pipeline:

        1. **📄 PDF Parsing** — Extracts text from your uploaded PDF using PyPDF2
        2. **✂️ Smart Chunking** — Splits text into overlapping chunks with page tracking
        3. **🌳 Hierarchy Detection** — Automatically builds a topic tree from chapter/section patterns
        4. **🔍 Hybrid Retrieval** — Combines BM25 (lexical) + Semantic (dense) search with score fusion
        5. **🤖 LLM Generation** — Sends retrieved context to GROQ's ultra-fast LLM inference

        ---

        **Tech Stack:**
        | Component | Technology |
        |-----------|-----------|
        | UI | Streamlit |
        | PDF Parsing | PyPDF2 |
        | Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
        | Lexical Search | Rank-BM25 |
        | LLM API | GROQ (Llama 3.3 70B) |
        | Text Processing | NLTK |
        """)
