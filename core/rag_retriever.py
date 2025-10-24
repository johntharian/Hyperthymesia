"""
RAG (Retrieval Augmented Generation) retriever.
Finds relevant document chunks for answering questions.
"""

from typing import Dict, List, Optional

from core.chunker import DocumentChunker
from storage.db import Database
from storage.vector_store import VectorStore


class RAGRetriever:
    """
    Retrieves relevant document chunks for Q&A.

    Uses hybrid approach:
    1. Semantic search (vector similarity)
    2. Keyword search (BM25)
    3. Re-ranking and deduplication
    """

    def __init__(
        self, db: Optional[Database] = None, vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize RAG retriever.

        Args:
            db: Database instance
            vector_store: Vector store instance
        """
        self.db = db or Database()
        self.vector_store = vector_store or VectorStore()
        self.chunker = DocumentChunker()

    def retrieve_context(
        self, question: str, num_chunks: int = 5, max_tokens: int = 3000
    ) -> Dict:
        """
        Retrieve relevant context for a question.

        Args:
            question: User's question
            num_chunks: Number of chunks to retrieve
            max_tokens: Maximum context tokens

        Returns:
            Dictionary with context and source information
        """
        # 1. Search for relevant documents
        semantic_results = self.vector_store.search(question, limit=num_chunks * 2)
        keyword_results = self.db.search_keyword(question, limit=num_chunks * 2)

        # 2. Merge and deduplicate results
        all_doc_ids = set()
        for r in semantic_results:
            all_doc_ids.add(r["id"])
        for r in keyword_results:
            all_doc_ids.add(r["id"])

        if not all_doc_ids:
            return {"context": "", "sources": [], "chunks_used": 0}

        # 3. Get full document content and chunk
        chunks = []
        sources = []

        for doc_id in list(all_doc_ids)[:num_chunks]:
            # Get document content
            content = self.db.get_document_content(doc_id)
            if not content:
                continue

            # Get document metadata
            cursor = self.db.conn.cursor()
            cursor.execute(
                """
                SELECT path, filename, file_type
                FROM documents
                WHERE id = ?
            """,
                (doc_id,),
            )

            row = cursor.fetchone()
            if not row:
                continue

            # Chunk the document
            doc_chunks = self.chunker.chunk_document(
                content, row["path"], row["file_type"]
            )

            # Find most relevant chunks using vector similarity
            best_chunk = self._find_best_chunk(question, doc_chunks)

            if best_chunk:
                chunks.append(best_chunk)
                sources.append(
                    {
                        "file": row["filename"],
                        "path": row["path"],
                        "chunk_index": best_chunk["chunk_index"],
                    }
                )

        # 4. Build context within token limit
        context = self._build_context(chunks, max_tokens)

        return {
            "context": context,
            "sources": sources[: len(chunks)],
            "chunks_used": len(chunks),
        }

    def _find_best_chunk(self, question: str, chunks: List[Dict]) -> Optional[Dict]:
        """
        Find the most relevant chunk from a document.

        Uses semantic similarity between question and chunk.

        Args:
            question: User's question
            chunks: List of document chunks

        Returns:
            Best matching chunk or None
        """
        if not chunks:
            return None

        # For now, simple heuristic:
        # - Prefer chunks that contain question keywords
        # - Later: use cross-encoder for better re-ranking

        question_words = set(question.lower().split())
        best_chunk = None
        best_score = 0

        for chunk in chunks:
            content_lower = chunk["content"].lower()

            # Count matching words
            score = sum(1 for word in question_words if word in content_lower)

            # Boost for longer content (more context)
            score += len(chunk["content"]) / 1000

            if score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk

    def _build_context(self, chunks: List[Dict], max_tokens: int) -> str:
        """
        Build context string from chunks within token limit.

        Args:
            chunks: List of chunks
            max_tokens: Maximum tokens allowed

        Returns:
            Formatted context string
        """
        context_parts = []
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            # Format chunk with source info
            chunk_text = f"Document {i+1} ({chunk['file_path']}):\n{chunk['content']}\n"

            # Estimate tokens
            chunk_tokens = self.chunker.estimate_tokens(chunk_text)

            # Check if adding this chunk exceeds limit
            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens

        return "\n---\n".join(context_parts)

    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        doc_count = self.db.get_stats()["document_count"]

        return {
            "indexed_documents": doc_count,
            "vector_count": self.vector_store.get_count(),
        }
