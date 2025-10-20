"""
Hybrid search implementation using RRF (Reciprocal Rank Fusion).
"""
from pathlib import Path
from typing import Dict, List, Optional

from storage.db import Database
from storage.vector_store import VectorStore


class Searcher:
    """Handles hybrid search combining keyword and semantic search."""
    
    def __init__(self, db: Optional[Database] = None, 
                 vector_store: Optional[VectorStore] = None):
        """
        Initialize searcher.
        
        Args:
            db: Database instance for keyword search
            vector_store: VectorStore instance for semantic search
        """
        self.db = db or Database()
        self.vector_store = vector_store or VectorStore()
    
    def search(self, query: str, limit: int = 10, 
               file_type: Optional[str] = None,
               path_filter: Optional[str] = None,
               keyword_weight: float = 0.5,
               semantic_weight: float = 0.5,
               min_score: float = 0.00) -> List[Dict]:
        """
        Hybrid search using RRF to combine keyword and semantic search.
        
        Args:
            query: Search query
            limit: Maximum results to return
            file_type: Optional file type filter (e.g., 'pdf', 'txt')
            path_filter: Optional path prefix filter
            keyword_weight: Weight for keyword search (0-1)
            semantic_weight: Weight for semantic search (0-1)
            min_score: Minimum RRF score threshold for results
        
        Returns:
            List of search results with scores and metadata
        """
        # Fetch more results than needed for better fusion
        fetch_limit = min(limit * 5, 100)
        
        # Perform both searches
        keyword_results = self._keyword_search(query, fetch_limit)
        semantic_results = self._semantic_search(query, fetch_limit, file_type, path_filter)
        
        # If both searches returned nothing, return empty
        if not keyword_results and not semantic_results:
            return []
        
        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            keyword_results, 
            semantic_results,
            keyword_weight,
            semantic_weight
        )
        
        # Filter by minimum score threshold
        fused_results = [r for r in fused_results if r['final_score'] >= 0] #TODO: fix min score
        
        # If no results above threshold, return empty
        if not fused_results:
            return []
        
        # Apply filters if needed
        if file_type or path_filter:
            fused_results = self._apply_filters(fused_results, file_type, path_filter)
        
        # Get full metadata for top results
        final_results = self._enrich_results(fused_results[:limit], query)
        
        return final_results
    
    def _keyword_search(self, query: str, limit: int) -> List[Dict]:
        """
        Perform keyword search using SQLite FTS5.
        
        Args:
            query: Search query
            limit: Max results
        
        Returns:
            List of results with document IDs and ranks
        """
        try:
            results = self.db.search_keyword(query, limit)
            return [{'id': r['id'], 'rank': idx + 1, 'raw_score': r.get('score', 0)} 
                    for idx, r in enumerate(results)]
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []
    
    def _semantic_search(self, query: str, limit: int,
                        file_type: Optional[str] = None,
                        path_filter: Optional[str] = None) -> List[Dict]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Search query
            limit: Max results
            file_type: Optional file type filter
            path_filter: Optional path filter
        
        Returns:
            List of results with document IDs and ranks
        """
        try:
            # Build metadata filter for ChromaDB
            where_filter = {}
            if file_type:
                where_filter['file_type'] = f'.{file_type.lower()}'
            if path_filter:
                # Note: ChromaDB doesn't support partial string matching easily
                # We'll filter after retrieval
                pass
            
            results = self.vector_store.search(
                query, 
                limit=limit,
                filter_metadata=where_filter if where_filter else None
            )
            
            return [{'id': r['id'], 'rank': idx + 1, 'raw_score': r.get('score', 0)} 
                    for idx, r in enumerate(results)]
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, keyword_results: List[Dict], 
                                semantic_results: List[Dict],
                                keyword_weight: float = 0.5,
                                semantic_weight: float = 0.5,
                                k: int = 60) -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion to combine results.
        
        RRF formula: score = Σ(weight / (k + rank))
        
        Args:
            keyword_results: Results from keyword search
            semantic_results: Results from semantic search
            keyword_weight: Weight for keyword results
            semantic_weight: Weight for semantic results
            k: RRF constant (typically 60)
        
        Returns:
            Fused and ranked results
        """
        scores = {}
        doc_info = {}
        
        # Process keyword results
        for result in keyword_results:
            doc_id = result['id']
            rank = result['rank']
            rrf_score = keyword_weight / (k + rank)
            
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_info[doc_id] = {
                'id': doc_id,
                'keyword_rank': rank,
                'keyword_score': result.get('raw_score', 0),
                'in_keyword': True,
                'in_semantic': doc_id in [r['id'] for r in semantic_results]
            }
        
        # Process semantic results
        for result in semantic_results:
            doc_id = result['id']
            rank = result['rank']
            rrf_score = semantic_weight / (k + rank)
            
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    'id': doc_id,
                    'in_keyword': False,
                    'in_semantic': True
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
        
        return ranked_results
    
    def _apply_filters(self, results: List[Dict], 
                      file_type: Optional[str],
                      path_filter: Optional[str]) -> List[Dict]:
        """
        Apply post-search filters to results.
        
        Args:
            results: Search results
            file_type: File type to filter by
            path_filter: Path prefix to filter by
        
        Returns:
            Filtered results
        """
        filtered = results
        
        if file_type:
            file_type = f'.{file_type.lower()}'
            filtered = [r for r in filtered if r.get('file_type') == file_type]
        
        if path_filter:
            path_filter = str(Path(path_filter).resolve())
            filtered = [r for r in filtered if r.get('path', '').startswith(path_filter)]
        
        return filtered
    
    def _enrich_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Enrich results with full metadata and snippets.
        
        Args:
            results: Fused search results
            query: Original search query for snippet generation
        
        Returns:
            Enriched results with metadata
        """
        enriched = []
        
        for result in results:
            doc_id = result['id']
            
            # Get document metadata from DB
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT d.path, d.filename, d.file_type, d.size, d.modified_at
                FROM documents d
                WHERE d.id = ?
            """, (doc_id,))
            
            row = cursor.fetchone()
            if not row:
                continue
            
            # Get content snippet
            content = self.db.get_document_content(doc_id)
            snippet = self._generate_snippet(content, query) if content else ""
            
            enriched.append({
                'id': doc_id,
                'path': row['path'],
                'filename': row['filename'],
                'file_type': row['file_type'],
                'size': row['size'],
                'modified': row['modified_at'],
                'score': result['final_score'],
                'snippet': snippet,
                'matched_in': self._get_match_source(result)
            })
        
        return enriched
    
    def _generate_snippet(self, content: str, query: str, 
                         snippet_length: int = 200) -> str:
        """
        Generate a text snippet showing query context.
        
        Args:
            content: Full document content
            query: Search query
            snippet_length: Approximate snippet length
        
        Returns:
            Snippet with query context
        """
        if not content:
            return ""
        
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find first occurrence of any query word
        query_words = query_lower.split()
        best_pos = -1
        
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
        
        if best_pos == -1:
            # No match found, return beginning
            return content[:snippet_length] + "..."
        
        # Extract snippet around the match
        start = max(0, best_pos - snippet_length // 2)
        end = min(len(content), best_pos + snippet_length // 2)
        
        snippet = content[start:end]
        
        # Add ellipsis
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet.strip()
    
    def _get_match_source(self, result: Dict) -> str:
        """
        Determine which search method(s) found this result.
        
        Args:
            result: Search result with match information
        
        Returns:
            String describing match source
        """
        in_keyword = result.get('in_keyword', False)
        in_semantic = result.get('in_semantic', False)
        
        if in_keyword and in_semantic:
            return "keyword+semantic"
        elif in_keyword:
            return "keyword"
        elif in_semantic:
            return "semantic"
        else:
            return "unknown"