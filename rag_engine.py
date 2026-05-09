"""
RAG Engine — Core retrieval pipeline with hierarchical indexing.
Handles PDF parsing, chunking, BM25 + semantic hybrid retrieval,
and hierarchical topic tree construction.
"""

import re
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi
from embeddings import EmbeddingEngine
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# ─────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    page_numbers: List[int]
    chunk_id: int
    section_title: str = ""
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class HierarchyNode:
    """A node in the hierarchical topic tree."""
    title: str
    level: int  # 0 = root, 1 = chapter, 2 = section, 3 = subsection
    children: List["HierarchyNode"] = field(default_factory=list)
    chunk_ids: List[int] = field(default_factory=list)
    summary: str = ""


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    chunk: Chunk
    score: float
    retrieval_method: str  # "bm25", "semantic", or "hybrid"


# ─────────────────────────────────────────────────────────
# PDF Parser
# ─────────────────────────────────────────────────────────

class PDFParser:
    """Extracts and cleans text from PDF files."""

    @staticmethod
    def extract_text(pdf_file) -> Tuple[str, List[Dict]]:
        """
        Extract text from a PDF file.

        Args:
            pdf_file: File-like object or path to the PDF.

        Returns:
            Tuple of (full_text, page_texts) where page_texts is a list of
            dicts with 'page' and 'text' keys.
        """
        reader = PdfReader(pdf_file)
        page_texts = []
        full_text_parts = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Clean the text
            text = PDFParser._clean_text(text)
            if text.strip():
                page_texts.append({"page": i + 1, "text": text})
                full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)
        return full_text, page_texts

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted PDF text."""
        # Remove surrogates and invalid unicode from PDF extraction
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        # Remove control characters except newlines and tabs
        text = "".join(ch if ch in ("\n", "\t") or (ord(ch) >= 32) else " " for ch in text)
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove page number patterns
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        # Fix common OCR/extraction artifacts
        text = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)
        # Normalize line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# ─────────────────────────────────────────────────────────
# Text Chunker
# ─────────────────────────────────────────────────────────

class TextChunker:
    """Smart text chunking with overlap and section awareness."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
        min_chunk_size: int = 50,
    ):
        """
        Args:
            chunk_size: Target number of words per chunk.
            chunk_overlap: Number of overlapping words between chunks.
            min_chunk_size: Minimum chunk size in words (smaller chunks are merged).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, page_texts: List[Dict]) -> List[Chunk]:
        """
        Break document into overlapping chunks with page number tracking.

        Args:
            page_texts: List of dicts with 'page' and 'text' keys.

        Returns:
            List of Chunk objects.
        """
        chunks = []
        chunk_id = 0

        # Build a list of (word, page_number) pairs
        word_page_map = []
        for pt in page_texts:
            words = pt["text"].split()
            for w in words:
                word_page_map.append((w, pt["page"]))

        if not word_page_map:
            return chunks

        total_words = len(word_page_map)
        start = 0

        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            chunk_words = word_page_map[start:end]

            text = " ".join([w for w, _ in chunk_words])
            pages = sorted(set([p for _, p in chunk_words]))

            # Detect section title if present
            section_title = self._detect_section_title(text)

            chunk = Chunk(
                text=text,
                page_numbers=pages,
                chunk_id=chunk_id,
                section_title=section_title,
            )

            if chunk.word_count >= self.min_chunk_size:
                chunks.append(chunk)
                chunk_id += 1

            # Move forward with overlap
            start = end - self.chunk_overlap
            if start >= total_words:
                break
            if end == total_words:
                break

        return chunks

    @staticmethod
    def _detect_section_title(text: str) -> str:
        """Try to detect chapter/section titles from text patterns."""
        patterns = [
            r"(?:Chapter|CHAPTER)\s+\d+[\.:]\s*(.+?)(?:\n|$)",
            r"(?:Section|SECTION)\s+[\d\.]+[\.:]\s*(.+?)(?:\n|$)",
            r"^(\d+[\.\d]*\s+[A-Z][^\n]{5,50})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:200], re.MULTILINE)
            if match:
                return match.group(1).strip()
        return ""


# ─────────────────────────────────────────────────────────
# Hierarchy Builder
# ─────────────────────────────────────────────────────────

class HierarchyBuilder:
    """Builds a hierarchical topic tree from document chunks."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def build_from_chunks(
        self, chunks: List[Chunk], document_title: str = "Document"
    ) -> HierarchyNode:
        """
        Build a hierarchical tree from chunks by detecting structural patterns.

        Args:
            chunks: List of document chunks.
            document_title: Title for the root node.

        Returns:
            Root HierarchyNode of the topic tree.
        """
        root = HierarchyNode(title=document_title, level=0)

        current_chapter = None
        current_section = None

        for chunk in chunks:
            text = chunk.text

            # Detect chapter-level headings
            chapter_match = re.search(
                r"(?:Chapter|CHAPTER|Part|PART)\s+(\d+|[IVXLC]+)[\.:,\s]+(.+?)(?:\.|$)",
                text[:200],
            )

            # Detect section-level headings
            section_match = re.search(
                r"(\d+\.\d+[\.\d]*)\s+([A-Z][^\n]{3,60})", text[:200]
            )

            if chapter_match:
                chapter_title = f"Chapter {chapter_match.group(1)}: {chapter_match.group(2).strip()}"
                current_chapter = HierarchyNode(
                    title=chapter_title, level=1, chunk_ids=[chunk.chunk_id]
                )
                root.children.append(current_chapter)
                current_section = None

            elif section_match:
                section_title = (
                    f"{section_match.group(1)} {section_match.group(2).strip()}"
                )
                current_section = HierarchyNode(
                    title=section_title, level=2, chunk_ids=[chunk.chunk_id]
                )
                if current_chapter:
                    current_chapter.children.append(current_section)
                else:
                    root.children.append(current_section)

            else:
                # Add chunk to current section or chapter
                if current_section:
                    current_section.chunk_ids.append(chunk.chunk_id)
                elif current_chapter:
                    current_chapter.chunk_ids.append(chunk.chunk_id)
                else:
                    # No structure detected — add to root
                    if not root.children or root.children[-1].level != 1:
                        intro_node = HierarchyNode(
                            title="Introduction / Preamble",
                            level=1,
                            chunk_ids=[chunk.chunk_id],
                        )
                        root.children.append(intro_node)
                    else:
                        root.children[-1].chunk_ids.append(chunk.chunk_id)

        return root

    def render_tree(self, node: HierarchyNode, prefix: str = "") -> str:
        """
        Render the hierarchy tree as a formatted string.

        Args:
            node: The root node to render.
            prefix: Current indentation prefix.

        Returns:
            Formatted tree string.
        """
        icons = {0: "📖", 1: "📑", 2: "📄", 3: "🔹"}
        icon = icons.get(node.level, "•")

        lines = [f"{prefix}{icon} {node.title} ({len(node.chunk_ids)} chunks)"]

        for i, child in enumerate(node.children):
            is_last = i == len(node.children) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            child_lines = self.render_tree(child, prefix="")
            first_line = f"{prefix}{connector}{child_lines.split(chr(10))[0].strip()}"
            lines.append(first_line)
            for line in child_lines.split("\n")[1:]:
                if line.strip():
                    lines.append(f"{prefix}{extension}{line}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Hybrid Retriever
# ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 (lexical) and semantic (dense) retrieval
    with reciprocal rank fusion for superior results.
    """

    def __init__(self, chunks: List[Chunk], embedding_engine: EmbeddingEngine):
        """
        Initialize the hybrid retriever.

        Args:
            chunks: List of document chunks.
            embedding_engine: Initialized EmbeddingEngine instance.
        """
        self.chunks = chunks
        self.embedding_engine = embedding_engine
        self.stop_words = set(stopwords.words("english"))

        if not chunks:
            # Nothing to index — retrieval will return empty results
            self.bm25 = None
            self.chunk_embeddings = np.array([])
            return

        # Build BM25 index
        tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Build semantic index
        chunk_texts = [str(c.text) for c in chunks]
        # Filter out any problematic entries
        chunk_texts = [t if isinstance(t, str) and len(t.strip()) > 0 else "empty" for t in chunk_texts]
        self.chunk_embeddings = embedding_engine.encode(chunk_texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.35,
        semantic_weight: float = 0.65,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant chunks using hybrid retrieval.

        Args:
            query: The user's query.
            top_k: Number of results to return.
            bm25_weight: Weight for BM25 scores in fusion.
            semantic_weight: Weight for semantic scores in fusion.

        Returns:
            List of RetrievalResult objects, sorted by relevance.
        """
        if not self.chunks:
            return []

        # BM25 retrieval
        bm25_scores = self.bm25.get_scores(self._tokenize(query))

        # Semantic retrieval
        query_embedding = self.embedding_engine.encode([query])
        semantic_scores = self.embedding_engine.compute_similarity(
            query_embedding[0], self.chunk_embeddings
        )

        # Normalize scores to [0, 1]
        bm25_norm = self._normalize_scores(bm25_scores)
        semantic_norm = self._normalize_scores(semantic_scores)

        # Hybrid fusion
        hybrid_scores = (bm25_weight * bm25_norm) + (semantic_weight * semantic_norm)

        # Rank and return top-k
        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            if hybrid_scores[idx] > 0.01:  # Filter near-zero scores
                results.append(
                    RetrievalResult(
                        chunk=self.chunks[idx],
                        score=float(hybrid_scores[idx]),
                        retrieval_method="hybrid",
                    )
                )

        return results

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s == 0:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25, removing stopwords."""
        words = word_tokenize(text.lower())
        return [w for w in words if w.isalnum() and w not in self.stop_words]


# ─────────────────────────────────────────────────────────
# RAG Pipeline Orchestrator
# ─────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline orchestrator.
    Ties together PDF parsing, chunking, hierarchy building,
    indexing, and retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_engine = EmbeddingEngine()
        self.hierarchy_builder = HierarchyBuilder()
        self.retriever: Optional[HybridRetriever] = None
        self.chunks: List[Chunk] = []
        self.hierarchy: Optional[HierarchyNode] = None
        self.full_text: str = ""
        self.doc_title: str = "Document"
        self.total_pages: int = 0
        self.doc_hash: str = ""

    def process_pdf(self, pdf_file, title: str = "Uploaded Document") -> Dict:
        """
        Process a PDF file through the entire pipeline.

        Args:
            pdf_file: File-like object of the uploaded PDF.
            title: Document title.

        Returns:
            Dict with processing stats.
        """
        self.doc_title = title

        # 1. Extract text
        self.full_text, page_texts = PDFParser.extract_text(pdf_file)
        self.total_pages = len(page_texts)

        # Generate a hash for caching
        self.doc_hash = hashlib.md5(self.full_text[:5000].encode()).hexdigest()

        # 2. Chunk the document
        self.chunks = self.chunker.chunk_document(page_texts)

        if not self.chunks:
            raise ValueError(
                "No text could be extracted from this PDF. "
                "The file may be scanned images, empty, or contain "
                "only non-extractable content (e.g. pure graphics)."
            )

        # 3. Build hierarchy
        self.hierarchy = self.hierarchy_builder.build_from_chunks(
            self.chunks, document_title=title
        )

        # 4. Build retriever index
        self.retriever = HybridRetriever(self.chunks, self.embedding_engine)

        return {
            "total_pages": self.total_pages,
            "total_chunks": len(self.chunks),
            "total_words": sum(c.word_count for c in self.chunks),
            "hierarchy_nodes": self._count_nodes(self.hierarchy),
        }

    def query(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Query the processed document.

        Args:
            question: User's question.
            top_k: Number of chunks to retrieve.

        Returns:
            List of RetrievalResult objects.
        """
        if self.retriever is None:
            raise ValueError("No document has been processed. Upload a PDF first.")
        return self.retriever.retrieve(question, top_k=top_k)

    def get_context_string(self, results: List[RetrievalResult]) -> str:
        """
        Build a context string from retrieval results for the LLM.

        Args:
            results: List of RetrievalResult objects.

        Returns:
            Formatted context string with page references.
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            pages = ", ".join(str(p) for p in result.chunk.page_numbers)
            section = f" | Section: {result.chunk.section_title}" if result.chunk.section_title else ""
            context_parts.append(
                f"[Source {i} | Pages: {pages}{section} | Relevance: {result.score:.2f}]\n"
                f"{result.chunk.text}"
            )
        return "\n\n---\n\n".join(context_parts)

    def get_hierarchy_text(self) -> str:
        """Get a formatted string of the document hierarchy."""
        if self.hierarchy is None:
            return "No hierarchy built yet."
        return self.hierarchy_builder.render_tree(self.hierarchy)

    def _count_nodes(self, node: HierarchyNode) -> int:
        """Recursively count hierarchy nodes."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
