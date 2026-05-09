<div align="center">

# 🧠 IntelliBot

### Hierarchical RAG-Powered Document Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://intellibot-pwakjcfunbxrmzrg5zrovs.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![GROQ API](https://img.shields.io/badge/LLM-GROQ%20API-orange?logo=lightning&logoColor=white)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Upload any PDF. Ask anything. Get detailed, cited answers in seconds.**

IntelliBot is a production-ready RAG (Retrieval-Augmented Generation) system that combines hierarchical document indexing, hybrid BM25 + semantic retrieval, and ultra-fast LLM inference to deliver precise, contextually rich answers from your documents.

[🚀 **Try the Live Demo**](https://intellibot-pwakjcfunbxrmzrg5zrovs.streamlit.app/) · [🐛 Report Bug](https://github.com/Avisweta-De/IntelliBot/issues) · [✨ Request Feature](https://github.com/Avisweta-De/IntelliBot/issues)

</div>

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 📄 Universal PDF Processing
Upload **any PDF** — textbooks, research papers, reports, manuals — and IntelliBot automatically extracts, cleans, and structures the content for intelligent querying.

</td>
<td width="50%">

### 🌳 Hierarchical Topic Tree
Automatically detects **chapters, sections, and subsections** from document structure, building a navigable topic hierarchy for organized content exploration.

</td>
</tr>
<tr>
<td width="50%">

### 🔍 Hybrid Retrieval Engine
Combines **BM25 lexical search** with **semantic dense retrieval** (sentence-transformers) using weighted score fusion — delivering superior relevance over either method alone.

</td>
<td width="50%">

### 🤖 Detailed AI Summaries
Powered by **Llama 3.3 70B** on GROQ's ultra-fast inference, responses include structured headings, bullet points, practical examples, source citations, and key takeaways.

</td>
</tr>
<tr>
<td width="50%">

### 📎 Source Citations
Every answer is grounded in the document with **page-level citations** and relevance scores, so you can always verify and trace back to the original content.

</td>
<td width="50%">

### 💬 Conversation Memory
Maintains **chat history context** across questions, enabling natural follow-up queries and deeper exploration of topics without repeating context.

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     IntelliBot Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📄 PDF Upload                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────┐                                           │
│  │  PDF Parser   │  PyPDF2 → Text extraction + cleaning     │
│  │  (PyPDF2)     │  Unicode sanitization for tokenizer      │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ Smart Chunker │  Overlapping chunks with page tracking   │
│  │ (500w / 80w)  │  Section-aware splitting                 │
│  └──────┬───────┘                                           │
│         │                                                   │
│    ┌────┴────┐                                              │
│    ▼         ▼                                              │
│ ┌───────┐ ┌────────────┐ ┌──────────────┐                  │
│ │ BM25  │ │  Semantic   │ │  Hierarchy   │                  │
│ │ Index │ │  Embeddings │ │   Builder    │                  │
│ │(Okapi)│ │ (MiniLM-L6) │ │ (Auto-detect)│                  │
│ └───┬───┘ └─────┬──────┘ └──────────────┘                  │
│     │           │                                           │
│     ▼           ▼                                           │
│  ┌─────────────────┐                                        │
│  │  Hybrid Fusion   │  Weighted score combination           │
│  │ (0.35 BM25 +     │  Min-max normalization                │
│  │  0.65 Semantic)   │  Top-k selection                     │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │   GROQ LLM API  │  Llama 3.3 70B Versatile              │
│  │   (Streaming)    │  Context-grounded generation          │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  💬 Detailed Answer + Source Citations + Key Takeaway        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | [Streamlit](https://streamlit.io) | Interactive UI with dark glassmorphism theme |
| **PDF Parsing** | [PyPDF2](https://pypdf2.readthedocs.io) | Text extraction with unicode sanitization |
| **Embeddings** | [sentence-transformers](https://sbert.net) (`all-MiniLM-L6-v2`) | 384-dim dense vectors for semantic search |
| **Lexical Search** | [Rank-BM25](https://github.com/dorianbrown/rank_bm25) | Okapi BM25 for keyword matching |
| **LLM Inference** | [GROQ API](https://groq.com) (`llama-3.3-70b-versatile`) | Ultra-fast streaming generation |
| **NLP** | [NLTK](https://nltk.org) | Tokenization, stopword removal |
| **Environment** | [python-dotenv](https://github.com/theskumar/python-dotenv) | Secure API key management |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A free [GROQ API Key](https://console.groq.com/keys)

### Installation

```bash
# Clone the repository
git clone https://github.com/Avisweta-De/IntelliBot.git
cd IntelliBot

# Install dependencies
pip install -r requirements.txt

# Configure your API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Launch IntelliBot
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 📖 Usage

1. **Upload** any PDF document using the sidebar
2. **Click** "🚀 Process Document" — IntelliBot will chunk, embed, and index your PDF
3. **Ask questions** in the chat — get detailed answers with source citations
4. **Explore** the 🌳 Document Hierarchy tab to see the auto-detected topic structure
5. **Fine-tune** retrieval settings in Advanced Settings (chunk size, top-k, BM25 weight)

---

## 📁 Project Structure

```
IntelliBot/
├── app.py                 # Streamlit UI — premium dark theme with chat interface
├── rag_engine.py          # Core RAG pipeline — parsing, chunking, hierarchy, retrieval
├── embeddings.py          # Embedding engine — sentence-transformers with sanitization
├── llm_client.py          # GROQ API client — streaming responses with conversation memory
├── requirements.txt       # Python dependencies
├── .env                   # API key (not tracked by git)
├── .gitignore             # Git exclusions
├── README.md              # This file
├── assets/
│   └── logo.png           # IntelliBot logo
└── .streamlit/
    └── config.toml        # Streamlit dark theme configuration
```

---

## ⚙️ Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk Size | 500 words | Target size for text chunks |
| Chunk Overlap | 80 words | Overlap between consecutive chunks |
| Top-K | 5 | Number of chunks retrieved per query |
| BM25 Weight | 0.35 | Weight for lexical search (semantic = 1 - BM25) |
| LLM Temperature | 0.2 | Response creativity (lower = more focused) |
| Max Tokens | 2048 | Maximum response length |

---

## 🔬 How the Hybrid Retrieval Works

IntelliBot's retrieval engine combines two complementary approaches:

1. **BM25 (Lexical)** — Matches exact keywords and terms. Excels at finding specific names, numbers, and technical terms.

2. **Semantic (Dense)** — Matches meaning using neural embeddings. Excels at understanding paraphrased queries and conceptual questions.

3. **Hybrid Fusion** — Both scores are min-max normalized to `[0, 1]` and combined:
   ```
   final_score = 0.35 × BM25_normalized + 0.65 × Semantic_normalized
   ```

This hybrid approach consistently outperforms either method alone across diverse query types.

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [GROQ](https://groq.com) for ultra-fast LLM inference
- [Sentence-Transformers](https://sbert.net) for the embedding model
- [Streamlit](https://streamlit.io) for the beautiful UI framework

---

<div align="center">

**Built with ❤️ by [Avisweta De](https://github.com/Avisweta-De)**

⭐ Star this repo if you found it useful!

</div>
