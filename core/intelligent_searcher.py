"""
Intelligent searcher that adapts to query complexity.
"""
from typing import List, Dict, Optional
from core.search import Searcher
from core.query_analyzer import get_analyzer
from core.llm_helper import get_rewriter


class IntelligentSearcher:
    """
    Smart search that automatically uses LLM for complex queries.
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize intelligent searcher.
        
        Args:
            use_llm: Whether to enable LLM features (requires API key)
        """
        self.base_searcher = Searcher()
        self.query_analyzer = get_analyzer()
        
        # Try to initialize LLM (may be None if not configured)
        self.llm_rewriter = get_rewriter() if use_llm else None
        self.llm_enabled = self.llm_rewriter is not None and self.llm_rewriter.is_available()
    
    def search(self, query: str, limit: int = 10, 
               file_type: Optional[str] = None,
               path_filter: Optional[str] = None,
               force_llm: bool = False,
               verbose: bool = False) -> Dict:
        """
        Intelligent search that adapts to query complexity.
        
        Args:
            query: User's search query
            limit: Maximum results to return
            file_type: Optional file type filter
            path_filter: Optional path filter
            force_llm: Force LLM usage even for simple queries
            verbose: Print detailed info about search strategy
        
        Returns:
            Dictionary with results and metadata
        """
        # Analyze query complexity
        analysis = self.query_analyzer.analyze(query)
        
        # Determine if we should use LLM
        use_llm_for_query = (
            self.llm_enabled and 
            (analysis['is_complex'] or force_llm)
        )
        
        # Prepare response metadata
        response = {
            'query': query,
            'analysis': analysis,
            'used_llm': False,
            'rewritten_query': None,
            'results': []
        }
        
        if verbose:
            self._print_analysis(analysis, use_llm_for_query)
        
        # Execute search based on complexity
        if use_llm_for_query:
            # Complex query - use LLM to optimize
            response['used_llm'] = True
            rewritten = self.llm_rewriter.rewrite_query(query)
            response['rewritten_query'] = rewritten
            
            if verbose:
                print(f"💡 Rewritten query: '{rewritten}'")
            
            # Search with rewritten query
            results = self.base_searcher.search(
                query=rewritten,
                limit=limit,
                file_type=file_type,
                path_filter=path_filter
            )
        else:
            # Simple query - direct search
            results = self.base_searcher.search(
                query=query,
                limit=limit,
                file_type=file_type,
                path_filter=path_filter
            )
        
        response['results'] = results
        return response
    
    def _print_analysis(self, analysis: Dict, will_use_llm: bool):
        """Print analysis information."""
        print(f"📊 Query Analysis:")
        print(f"   Complexity Score: {analysis['complexity_score']}")
        if analysis['reason']:
            print(f"   Factors: {', '.join(analysis['reason'])}")
        print(f"   Strategy: {'LLM-enhanced' if will_use_llm else 'Direct'} search")
        print()
    
    def is_llm_available(self) -> bool:
        """Check if LLM features are available."""
        return self.llm_enabled
    
    def get_search_strategy(self, query: str) -> str:
        """
        Get recommended search strategy for a query without executing search.
        
        Args:
            query: Search query
        
        Returns:
            'direct' or 'llm_enhanced'
        """
        analysis = self.query_analyzer.analyze(query)
        return analysis['suggested_strategy']


# Example usage and testing
if __name__ == "__main__":
    # Test queries
    test_queries = [
        "python tutorial",  # Simple
        "machine learning",  # Simple
        "that PDF about neural networks I read last week",  # Complex
        "how do I implement async in python?",  # Complex (question)
        "show me files related to the Chicago project",  # Complex (conversational)
        "project_proposal.pdf",  # Simple (filename)
    ]
    
    searcher = IntelligentSearcher(use_llm=False)  # Test without LLM
    
    print("Query Complexity Analysis:")
    print("=" * 60)
    
    for query in test_queries:
        analysis = searcher.query_analyzer.analyze(query)
        print(f"\nQuery: '{query}'")
        print(f"  Complexity Score: {analysis['complexity_score']}")
        print(f"  Is Complex: {analysis['is_complex']}")
        print(f"  Strategy: {analysis['suggested_strategy']}")
        if analysis['reason']:
            print(f"  Reasons: {', '.join(analysis['reason'])}")