"""
SearchTool for agent to find relevant code in indexed documents.
"""
from typing import Dict, Any, List, Optional
from core.tools.base_tool import BaseTool, ToolResult
from core.search import Searcher
from core.query_expander import QueryExpander


class SearchTool(BaseTool):
    """
    Tool for searching indexed code.

    Capabilities:
    - Keyword + semantic hybrid search
    - File type filtering
    - Path-based filtering
    - Query expansion for abstract questions
    """

    def __init__(self):
        """Initialize SearchTool."""
        super().__init__(
            name='search',
            description='''Search for relevant code and documents.
Use this to find files, functions, classes, or patterns.
Parameters:
  - query (str, required): What to search for
  - limit (int, optional): Max results to return (default: 10)
  - file_type (str, optional): Filter by file extension (e.g., '.py')
  - path_filter (str, optional): Filter by path prefix
Returns: List of matching files with code snippets and scores'''
        )
        self.searcher = Searcher()
        self.expander = QueryExpander()

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute search.

        Args:
            query (str): Search query
            limit (int): Max results (default: 10)
            file_type (str): Optional file type filter
            path_filter (str): Optional path filter

        Returns:
            ToolResult with search results
        """
        # Validate required params
        if not self.validate_params(['query'], kwargs):
            return ToolResult(
                success=False,
                data=None,
                message='Missing required parameter: query'
            )

        query = kwargs['query']
        limit = kwargs.get('limit', 10)
        file_type = kwargs.get('file_type')
        path_filter = kwargs.get('path_filter')

        try:
            self.logger.debug(f"SearchTool: Searching for '{query}'")

            # Expand abstract queries
            if self.expander.is_abstract_query(query):
                expanded_query = self.expander.expand_query(query)
                self.logger.debug(f"Expanded query: {expanded_query}")
            else:
                expanded_query = query

            # Perform search
            results = self.searcher.search(
                expanded_query,
                limit=limit,
                file_type=file_type,
                path_filter=path_filter
            )

            if not results:
                return ToolResult(
                    success=True,
                    data=[],
                    message=f'No results found for "{query}"'
                )

            # Format results for agent
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    'rank': i,
                    'filename': result.get('filename', 'unknown'),
                    'path': result.get('path', 'unknown'),
                    'file_type': result.get('file_type', ''),
                    'score': result.get('final_score', 0),
                    'snippet': result.get('snippet', '')[:200],  # First 200 chars
                    'match_type': result.get('match_type', 'both'),
                })

            self.logger.debug(f"SearchTool: Found {len(results)} results")

            result = ToolResult(
                success=True,
                data=formatted_results,
                message=f'Found {len(results)} relevant files',
                metadata={
                    'original_query': query,
                    'expanded_query': expanded_query if self.expander.is_abstract_query(query) else None,
                    'result_count': len(results),
                }
            )

            self.log_execution(kwargs, result)
            return result

        except Exception as e:
            self.logger.error(f"SearchTool error: {e}")
            return ToolResult(
                success=False,
                data=None,
                message=f'Search error: {str(e)}'
            )

    def search_by_concept(self, concept: str, limit: int = 5) -> ToolResult:
        """
        Search for a programming concept.

        Args:
            concept: Concept to search for (e.g., 'error handling', 'retry')
            limit: Max results

        Returns:
            ToolResult with search results
        """
        # Use query expander to get related patterns
        expanded = self.expander.expand_query(concept)
        return self.execute(query=expanded, limit=limit)

    def search_by_pattern(self, pattern: str, limit: int = 5) -> ToolResult:
        """
        Search for code matching a pattern.

        Args:
            pattern: Pattern to search for (e.g., 'class.*Handler')
            limit: Max results

        Returns:
            ToolResult with search results
        """
        return self.execute(query=pattern, limit=limit)
