"""
RAG (Retrieval Augmented Generation) retriever.
Finds relevant document chunks for answering questions.
"""

from typing import Dict, List, Optional

from core.chunker import DocumentChunker
from storage.db import Database
from storage.vector_store import VectorStore

# Dependency directories to filter out
DEPENDENCY_DIRS = {
    'node_modules', 
    '__pycache__', 
    '.venv', 
    'venv', 
    'env', 
    'site-packages', 
    'vendor',
    'packages'
}

class RAGRetriever:
    """
    Retrieves relevant document chunks for Q&A.

    Uses hybrid approach:
    1. Semantic search (vector similarity)
    2. Keyword search (BM25)
    3. Re-ranking and deduplication
    """

    def __init__(
        self, db: Optional[Database] = None, vector_store: Optional[VectorStore] = None,
        searcher: Optional[IntelligentSearcher] = None
    ):
        """
        Initialize RAG retriever.

        Args:
            db: Database instance
            vector_store: Vector store instance
            searcher: IntelligentSearcher instance
        """
        self.db = db or Database()
        self.vector_store = vector_store or VectorStore()
        self.searcher = searcher or IntelligentSearcher()
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
    
        # 1. Extract key terms (simple, no LLM)
        key_terms = self._extract_key_terms(question)

        #  2. Semantic search (embeddings understand questions)
        semantic_results = self.vector_store.search(
            question,  # Use full question for semantic
            limit=num_chunks * 3
        )
        
        # 3. Keyword search with extracted terms
        keyword_results = self.db.search_keyword(
            key_terms,  # Use extracted terms for keyword
            limit=num_chunks * 2
        )

        # def is_user_code(result):
        #     """Check if result is from user code (not dependencies)."""
        #     # For semantic results
        #     if 'metadata' in result:
        #         path = result.get('metadata', {}).get('path', '')
        #     else:
        #         # For keyword results
        #         path = result.get('path', '')
        
        #     return not any(dep in path for dep in DEPENDENCY_DIRS)

        # # Filter both result sets
        # semantic_results = [r for r in semantic_results if is_user_code(r)]
        # keyword_results = [r for r in keyword_results if is_user_code(r)]

        # # 3. BOOST USER CODE (in case some deps slip through)
        # for result in semantic_results:
        #     # User code gets 2x score boost
        #     if 'score' in result:
        #         result['score'] *= 2.0
    
        # for result in keyword_results:
        #     if 'score' in result:
        #         result['score'] *= 2.0


        # 4. Merge and deduplicate results
        all_doc_ids = set()
        for r in semantic_results:
            all_doc_ids.add(r["id"])
        for r in keyword_results:
            all_doc_ids.add(r["id"])

        if not all_doc_ids:
            return {"context": "", "sources": [], "chunks_used": 0}

        # 5. Get full document content and chunk
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

    def _extract_key_terms(self, question: str) -> str:
        """Simple keyword extraction without LLM."""
        stop_words = {
            "how",
            "what",
            "where",
            "when",
            "why",
            "who",
            "which",
            "does",
            "do",
            "did",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "we",
            "i",
            "you",
            "it",
            "this",
            "that",
            "use",
        }

        words = question.lower().split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        return " ".join(key_terms)

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
