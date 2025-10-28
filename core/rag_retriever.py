"""
RAG (Retrieval Augmented Generation) retriever.
Finds relevant document chunks for answering questions.
"""

from typing import Dict, List, Optional

from core.chunker import DocumentChunker
from core.local_llm import LocalLLM
from core.query_analyzer import QueryAnalyzer
from core.query_expander import QueryExpander
from storage.db import Database
from storage.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

# Dependency directories to filter out
DEPENDENCY_DIRS = {
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "site-packages",
    "vendor",
    "packages",
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
        self.query_analyzer = QueryAnalyzer()
        self.query_expander = QueryExpander()
        self.llm = LocalLLM()

    def retrieve_context(
        self, question: str, num_chunks: int = 5, max_tokens: int = 3000
    ) -> Dict:
        """
        Retrieve relevant context for a question.

        Implements retry mechanism: if initial search returns no results,
        falls back to simpler keyword extraction.

        Args:
            question: User's question
            num_chunks: Number of chunks to retrieve
            max_tokens: Maximum context tokens

        Returns:
            Dictionary with context and source information
        """

        analysis = self.query_analyzer.analyze(question)

        search_terms = question
        if analysis["is_complex"]:
            search_terms = self._rewrite_query_local(question)

        logger.debug(f"Initial search terms: {search_terms}")

        #  2. Semantic search (embeddings understand questions)
        logger.debug("Performing semantic search")
        semantic_results = self.vector_store.search(
            search_terms, limit=num_chunks * 3  # Use full question for semantic
        )
        logger.debug(f"Semantic search returned {len(semantic_results)} results")

        # 3. Keyword search with extracted terms
        logger.debug("Performing keyword search")
        keyword_results = self.db.search_keyword(
            search_terms, limit=num_chunks * 2  # Use extracted terms for keyword
        )
        logger.debug(f"Keyword search returned {len(keyword_results)} results")

        merged_results = self._reciprocal_rank_fusion(
            keyword_results,
            semantic_results,
            search_query=search_terms
        )

        # Log debug info about search results
        logger.debug(f"Semantic results ({len(semantic_results)}):")
        for i, r in enumerate(semantic_results[:5]):
            # Handle different result formats
            if 'metadata' in r:
                path = r['metadata'].get('path', 'unknown')
            else:
                path = r.get('path', 'unknown')
            logger.debug(f"  {i+1}. {path}")

        logger.debug(f"Keyword results ({len(keyword_results)}):")
        for i, r in enumerate(keyword_results[:5]):
            path = r.get('path', 'unknown')
            logger.debug(f"  {i+1}. {path}")

        logger.debug("Merged results:")
        for i, r in enumerate(merged_results[:5]):
            # Handle different result formats
            if 'metadata' in r:
                path = r['metadata'].get('path', 'unknown')
            else:
                path = r.get('path', 'unknown')
            logger.debug(f"  {i+1}. {path}")


        # 4. Merge and deduplicate results
        all_doc_ids = []
        logger.debug("Final merged results for retrieval:")
        for i, r in enumerate(merged_results):
            doc_id = r.get("id", "unknown")
            source = r.get("metadata", {}).get("path", r.get("path", "unknown"))
            logger.debug(f"{i+1}. ID: {doc_id}, Source: {source}")
            all_doc_ids.append(doc_id)

        # RETRY MECHANISM: If no results found, try simpler extraction
        if not all_doc_ids:
            logger.info(f"No results for '{search_terms}', retrying with simple extraction")
            simple_terms = self._simple_extract(question)
            logger.debug(f"Retry search terms: {simple_terms}")

            # Retry semantic search
            semantic_results = self.vector_store.search(
                simple_terms, limit=num_chunks * 3
            )
            logger.debug(f"Retry semantic search returned {len(semantic_results)} results")

            # Retry keyword search
            keyword_results = self.db.search_keyword(
                simple_terms, limit=num_chunks * 2
            )
            logger.debug(f"Retry keyword search returned {len(keyword_results)} results")

            # Merge retry results
            merged_results = self._reciprocal_rank_fusion(
                keyword_results,
                semantic_results,
                search_query=simple_terms
            )

            for i, r in enumerate(merged_results):
                doc_id = r.get("id", "unknown")
                all_doc_ids.append(doc_id)

            logger.info(f"Retry found {len(all_doc_ids)} documents")

        if not all_doc_ids:
            logger.warning(f"No documents found for question: {question}")
            return {"context": "", "sources": [], "chunks_used": 0}

        # 5. Get full document content and chunk
        chunks = []
        sources = []
        logger.debug(f"Processing top {num_chunks} documents")

        for doc_id in all_doc_ids[:num_chunks]:
            logger.debug(f"Processing document ID: {doc_id}")
            # Get document content
            content = self.db.get_document_content(doc_id)
            if not content:
                logger.debug(f"No content found for document {doc_id}")
                continue
            logger.debug(f"Retrieved content ({len(content)} chars)")

            # Get document metadata
            cursor = self.db.conn.cursor()
            cursor.execute(
                """
                SELECT path, filename, file_type, indexed_at
                FROM documents
                WHERE id = ?
            """,
                (doc_id,),
            )

            row = cursor.fetchone()
            if not row:
                logger.debug(f"No metadata found for document {doc_id}")
                continue
            logger.debug(f"Metadata: path={row['path']}, filename={row['filename']}, type={row['file_type']}")

            # Chunk the document
            doc_chunks = self.chunker.chunk_document(
                content, row["path"], row["file_type"]
            )
            logger.debug(f"Split into {len(doc_chunks)} chunks")

            # Find most relevant chunks using vector similarity
            best_chunk = self._find_best_chunk(search_terms, doc_chunks)
            if best_chunk:
                logger.debug(f"Selected chunk {best_chunk.get('chunk_index')} (score: {best_chunk.get('score', 'N/A')})")
            else:
                logger.debug("No suitable chunk found")

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

    def _rewrite_query_local(self, question: str) -> str:
        """
        Use local LLM to rewrite question into better search query.
        Developer-focused to extract code, function, and class names.
        Also uses query expander for abstract question patterns.
        """
        # First, try query expander for abstract questions
        if self.query_expander.is_abstract_query(question):
            logger.debug(f"Detected abstract query, using expander")
            expanded = self.query_expander.expand_query(question)
            logger.debug(f"Expanded to: {expanded}")
        else:
            expanded = question

        if not self.llm.is_available():
            return self._simple_extract(expanded)

        try:
            prompt = f"""You are a code search optimizer. Rewrite this developer question into effective search keywords to find relevant code.

Developer question context examples:
- "how do I handle authentication?" → "authentication handler login verify"
- "where are database queries?" → "database query connection execute"
- "how do retries work?" → "retry mechanism exponential backoff"
- "what is error handling like?" → "error exception try catch handler"

Keep keywords focused on:
• Class names (Session, Request, Response)
• Function names (authenticate, parse, validate)
• Concepts (caching, retry, timeout, authentication)

Developer question: {expanded}

Optimized search keywords (2-6 terms):"""

            response = self.llm._generate_ollama(prompt, max_tokens=50)
            rewritten = response.strip()

            # Validate response is reasonable
            if len(rewritten.split()) > 10 or len(rewritten) < 3:
                return self._simple_extract(expanded)

            return rewritten
        except:
            return self._simple_extract(expanded)

    def _simple_extract(self, question: str) -> str:
        """Simple keyword extraction (fallback)."""
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

    def _calculate_filename_boost_rag(self, filename: str, query: str) -> float:
        """
        Calculate a boost factor based on filename match quality for RAG.

        Boosts scoring when query terms match the filename:
        - Exact filename match: 1.0x (100% boost)
        - Partial filename match: 0.5x (50% boost)
        - Extension match (e.g., .py, .js): 0.1x (10% boost)
        - No filename match: 0.0x (no boost)

        Args:
            filename: The document filename
            query: The search query

        Returns:
            Boost multiplier (0.0 to 1.0)
        """
        import os

        filename_lower = filename.lower()
        query_lower = query.lower()

        # Check for exact filename match (case-insensitive)
        if filename_lower == query_lower or filename_lower == query_lower + os.path.splitext(filename)[1]:
            return 1.0

        # Check if filename contains all query terms (phrase match)
        query_words = query_lower.split()
        if all(word in filename_lower for word in query_words):
            return 0.8

        # Check for partial filename match (at least one significant word)
        for word in query_words:
            if len(word) > 2 and word in filename_lower:  # Ignore short words like "a", "to", etc.
                return 0.5

        # Check if query contains filename or vice versa
        if query_lower in filename_lower or filename_lower in query_lower:
            return 0.3

        # Check file extension match
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext and file_ext[1:] in query_lower:  # Remove the leading dot
            return 0.1

        # No filename match
        return 0.0

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

    def _reciprocal_rank_fusion(self, keyword_results: List[Dict],
                                semantic_results: List[Dict],
                                keyword_weight: float = 0.5,
                                semantic_weight: float = 0.5,
                                k: int = 60,
                                search_query: str = "") -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion to combine results.

        RRF formula: score = Σ(weight / (k + rank))

        Args:
            keyword_results: Results from keyword search
            semantic_results: Results from semantic search
            keyword_weight: Weight for keyword results
            semantic_weight: Weight for semantic results
            k: RRF constant (typically 60)
            search_query: Original search query for filename boosting

        Returns:
            Fused and ranked results
        """
        scores = {}
        doc_info = {}

        # Process keyword results
        for rank,result in enumerate(keyword_results,1):
            doc_id = result['id']
            rrf_score = keyword_weight / (k + rank)

            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            # Preserve all original metadata
            doc_info[doc_id] = {
                'id': doc_id,
                'keyword_rank': rank,
                'keyword_score': result.get('raw_score', 0),
                'in_keyword': True,
                'in_semantic': doc_id in [r['id'] for r in semantic_results],
                # Preserve metadata from the result
                'metadata': result.get('metadata', {}),
                'path': result.get('path')
            }

        # Process semantic results
        for rank,result in enumerate(semantic_results,1):
            doc_id = result['id']
            rrf_score = semantic_weight / (k + rank)

            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    'id': doc_id,
                    'in_keyword': False,
                    'in_semantic': True,
                    # Preserve metadata from the result
                    'metadata': result.get('metadata', {}),
                    'path': result.get('path')
                }

            doc_info[doc_id]['semantic_rank'] = rank
            doc_info[doc_id]['semantic_score'] = result.get('raw_score', 0)

        # Combine and sort
        ranked_results = []
        for doc_id, score in scores.items():
            result = doc_info[doc_id].copy()
            result['final_score'] = score
            ranked_results.append(result)

        # Sort by final RRF score
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Apply filename boosting for better matching
        if search_query:
            for result in ranked_results:
                # Try to get filename from path
                if 'path' in result and result['path']:
                    import os
                    filename = os.path.basename(result['path'])
                    filename_boost = self._calculate_filename_boost_rag(filename, search_query)
                    result['final_score'] = result['final_score'] * (1 + filename_boost)

            # Re-sort after boosting
            ranked_results.sort(key=lambda x: x['final_score'], reverse=True)

        return ranked_results

    def _build_context(self, chunks: List[Dict], max_tokens: int) -> str:
        """
        Build context string from chunks within token limit.

        Includes filename metadata prominently for developer context.

        Args:
            chunks: List of chunks
            max_tokens: Maximum tokens allowed

        Returns:
            Formatted context string
        """
        context_parts = []
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            # Extract filename from path
            file_path = chunk.get('file_path', 'unknown')
            import os
            filename = os.path.basename(file_path)
            file_type = chunk.get('file_type', '')

            # Format chunk with prominent source info for better LLM understanding
            chunk_text = f"""📄 {filename} [{file_type}] ({file_path})
---
{chunk['content']}
---"""

            # Estimate tokens
            chunk_tokens = self.chunker.estimate_tokens(chunk_text)

            # Check if adding this chunk exceeds limit
            if total_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
            logger.debug(f"Added chunk {i+1} from {filename} ({chunk_tokens} tokens)")

        result = "\n\n".join(context_parts)
        logger.debug(f"Total context built: {total_tokens} tokens from {len(context_parts)} chunks")
        return result

    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        doc_count = self.db.get_stats()["document_count"]

        return {
            "indexed_documents": doc_count,
            "vector_count": self.vector_store.get_count(),
        }
