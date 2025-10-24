"""
Query complexity analysis to determine if LLM assistance is needed.
"""
from typing import Dict
import re


class QueryAnalyzer:
    """Analyzes queries to determine complexity and optimal search strategy."""
    
    # Indicators of complex queries
    QUESTION_WORDS = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'whose', 'whom']
    VAGUE_WORDS = ['that', 'thing', 'stuff', 'something', 'about', 'related', 'similar']
    TEMPORAL_WORDS = ['recent', 'latest', 'last', 'yesterday', 'ago', 'old', 'new']
    CONVERSATIONAL_PHRASES = ['show me', 'find me', 'looking for', 'can you', 'i need', 'i want']
    
    def __init__(self):
        self.complexity_threshold = 3  # Score >= 3 means complex query
    
    def analyze(self, query: str) -> Dict:
        """
        Analyze query complexity and characteristics.
        
        Args:
            query: User's search query
        
        Returns:
            Dictionary with analysis results
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        word_count = len(words)
        
        analysis = {
            'original_query': query,
            'word_count': word_count,
            'is_question': False,
            'is_conversational': False,
            'has_vague_language': False,
            'has_temporal_reference': False,
            'has_multiple_concepts': False,
            'complexity_score': 0,
            'is_complex': False,
            'reason': [],
            'suggested_strategy': 'direct'  # 'direct' or 'llm_enhanced'
        }
        
        # Check for question format
        if any(query_lower.startswith(q) for q in self.QUESTION_WORDS):
            analysis['is_question'] = True
            analysis['complexity_score'] += 2
            analysis['reason'].append('question format')
        
        # Check for conversational language
        if any(phrase in query_lower for phrase in self.CONVERSATIONAL_PHRASES):
            analysis['is_conversational'] = True
            analysis['complexity_score'] += 1
            analysis['reason'].append('conversational language')
        
        # Check for vague language
        if any(word in words for word in self.VAGUE_WORDS):
            analysis['has_vague_language'] = True
            analysis['complexity_score'] += 2
            analysis['reason'].append('vague terms')
        
        # Check for temporal references
        if any(word in words for word in self.TEMPORAL_WORDS):
            analysis['has_temporal_reference'] = True
            analysis['complexity_score'] += 1
            analysis['reason'].append('time reference')
        
        # Check for multiple distinct concepts (heuristic: varied vocabulary)
        unique_word_ratio = len(set(words)) / len(words) if words else 0
        if word_count > 5 and unique_word_ratio > 0.8:
            analysis['has_multiple_concepts'] = True
            analysis['complexity_score'] += 1
            analysis['reason'].append('multiple concepts')
        
        # Long queries are often complex
        if word_count >= 5: 
            analysis['complexity_score'] += 2
            analysis['reason'].append('long query')
        elif word_count <= 3:
            # Short, specific queries are usually simple
            analysis['complexity_score'] -= 1
        
        # Check if query looks like a filename
        if self._looks_like_filename(query):
            analysis['complexity_score'] -= 2
            analysis['reason'].append('filename pattern')
        
        # Check for technical terms (simple queries often have technical terms)
        if self._has_technical_terms(query):
            analysis['complexity_score'] -= 1
        
        # Determine if complex
        analysis['is_complex'] = analysis['complexity_score'] >= self.complexity_threshold
        
        # Suggest strategy
        if analysis['is_complex']:
            analysis['suggested_strategy'] = 'llm_enhanced'
        else:
            analysis['suggested_strategy'] = 'direct'
        
        return analysis
    
    def _looks_like_filename(self, query: str) -> bool:
        """Check if query looks like a filename."""
        # Has file extension
        if re.search(r'\.\w{2,4}$', query):
            return True
        
        # Has path separators
        if '/' in query or '\\' in query:
            return True
        
        # Has underscores/hyphens (common in filenames)
        words = query.split()
        if len(words) <= 2 and any('_' in w or '-' in w for w in words):
            return True
        
        return False
    
    def _has_technical_terms(self, query: str) -> bool:
        """Check if query contains technical terms (programming, etc.)."""
        technical_patterns = [
            r'\b(python|java|javascript|js|cpp|c\+\+|ruby|go|rust|sql)\b',
            r'\b(api|sdk|http|json|xml|csv|pdf)\b',
            r'\b(function|class|method|variable|algorithm|code)\b',
            r'\b(ml|ai|nlp|cv|nn|cnn|rnn|lstm|bert|gpt)\b',
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in technical_patterns)
    
    def should_use_llm(self, query: str) -> bool:
        """
        Quick check if LLM should be used for this query.
        
        Args:
            query: Search query
        
        Returns:
            True if LLM enhancement is recommended
        """
        analysis = self.analyze(query)
        return analysis['is_complex']
    
    def get_complexity_explanation(self, query: str) -> str:
        """
        Get human-readable explanation of why query is complex/simple.
        
        Args:
            query: Search query
        
        Returns:
            Explanation string
        """
        analysis = self.analyze(query)
        
        if analysis['is_complex']:
            reasons = ', '.join(analysis['reason'])
            return f"Complex query detected ({reasons}). Using LLM to optimize search."
        else:
            return "Simple query. Using direct search."


# Singleton instance
_analyzer = None

def get_analyzer() -> QueryAnalyzer:
    """Get or create QueryAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = QueryAnalyzer()
    return _analyzer