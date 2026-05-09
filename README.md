# IntelliBot 🧠 — Hierarchical RAG Document Q&A

An intelligent RAG-based document assistant that lets you upload any PDF and ask questions with AI-powered answers.

## 🔍 Features

- **Universal PDF Upload** — Works with any PDF document
- **Hierarchical Indexing** — Auto-detects chapters, sections, and topics
- **Hybrid Retrieval** — BM25 (lexical) + Semantic (dense) search with score fusion
- **Source Citations** — Every answer includes page numbers and relevance scores
- **Detailed Summaries** — Structured responses with headings, examples, and key takeaways
- **Streaming Responses** — Real-time token streaming via GROQ API

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Lexical Search | Rank-BM25 |
| LLM API | GROQ (Llama 3.3 70B) |
| Text Processing | NLTK |

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/IntelliBot.git
cd IntelliBot

# Install dependencies
pip install -r requirements.txt

# Create .env with your GROQ API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run
streamlit run app.py
```

## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://intellibot.streamlit.app)

## 📝 License

MIT
