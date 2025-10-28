"""
Query expander for abstract developer questions.
Converts vague questions into concrete search keywords using pattern knowledge.
"""
from typing import List, Dict, Set
from utils.logger import get_logger

logger = get_logger(__name__)


class QueryExpander:
    """
    Expands abstract developer questions into concrete search keywords.

    Uses a knowledge base of common code patterns to help find relevant code
    when users ask abstract questions like "how do retries work?"
    """

    # Knowledge base: abstract concepts → concrete patterns/keywords
    CONCEPT_PATTERNS = {
        # Error handling and exceptions
        'error': {
            'keywords': ['exception', 'try', 'catch', 'handler', 'error', 'raise', 'logging'],
            'class_patterns': ['Error', 'Exception', 'Handler'],
            'function_patterns': ['handle', 'catch', 'log', 'raise'],
        },
        'exception': {
            'keywords': ['exception', 'error', 'try', 'catch', 'raise', 'handling'],
            'class_patterns': ['Exception', 'Error'],
            'function_patterns': ['raise', 'except', 'handle'],
        },
        'retry': {
            'keywords': ['retry', 'backoff', 'exponential', 'timeout', 'attempt', 'repeat'],
            'class_patterns': ['Retry', 'Retrier', 'Backoff'],
            'function_patterns': ['retry', 'backoff', 'attempt'],
        },
        'timeout': {
            'keywords': ['timeout', 'deadline', 'duration', 'timeout', 'wait', 'patience'],
            'class_patterns': ['Timeout', 'Timer'],
            'function_patterns': ['timeout', 'wait'],
        },

        # Authentication and security
        'auth': {
            'keywords': ['authentication', 'auth', 'login', 'token', 'verify', 'password', 'session'],
            'class_patterns': ['Auth', 'Authenticator', 'Session', 'Token'],
            'function_patterns': ['authenticate', 'login', 'verify', 'logout'],
        },
        'authentication': {
            'keywords': ['authentication', 'auth', 'login', 'credential', 'password', 'verify'],
            'class_patterns': ['Authenticator', 'Auth', 'Login'],
            'function_patterns': ['authenticate', 'login', 'verify'],
        },
        'login': {
            'keywords': ['login', 'authenticate', 'password', 'credential', 'auth', 'session'],
            'class_patterns': ['Login', 'Authenticator'],
            'function_patterns': ['login', 'authenticate'],
        },
        'token': {
            'keywords': ['token', 'jwt', 'bearer', 'api_key', 'session', 'cookie'],
            'class_patterns': ['Token', 'JWT', 'APIKey'],
            'function_patterns': ['generate_token', 'validate_token'],
        },

        # Caching and performance
        'cache': {
            'keywords': ['cache', 'caching', 'cached', 'memoize', 'memo', 'store'],
            'class_patterns': ['Cache', 'Cacher', 'LRU'],
            'function_patterns': ['cache', 'get_cached', 'invalidate'],
        },
        'performance': {
            'keywords': ['performance', 'optimization', 'optimize', 'fast', 'speed', 'efficient'],
            'class_patterns': [],
            'function_patterns': ['optimize', 'cache', 'batch'],
        },

        # Database operations
        'database': {
            'keywords': ['database', 'db', 'query', 'connection', 'pool', 'transaction'],
            'class_patterns': ['Database', 'DB', 'Connection', 'Pool'],
            'function_patterns': ['query', 'execute', 'transaction'],
        },
        'query': {
            'keywords': ['query', 'sql', 'select', 'where', 'filter', 'search'],
            'class_patterns': ['Query', 'QueryBuilder'],
            'function_patterns': ['query', 'select', 'filter'],
        },
        'connection': {
            'keywords': ['connection', 'connect', 'pool', 'connector', 'driver'],
            'class_patterns': ['Connection', 'Pool', 'Connector'],
            'function_patterns': ['connect', 'disconnect'],
        },

        # HTTP and APIs
        'http': {
            'keywords': ['http', 'request', 'response', 'rest', 'api', 'endpoint'],
            'class_patterns': ['Request', 'Response', 'HTTP', 'Client'],
            'function_patterns': ['request', 'get', 'post', 'put', 'delete'],
        },
        'api': {
            'keywords': ['api', 'endpoint', 'request', 'response', 'rest', 'client'],
            'class_patterns': ['API', 'Client', 'Server'],
            'function_patterns': ['request', 'call', 'invoke'],
        },
        'request': {
            'keywords': ['request', 'http', 'client', 'send', 'api', 'endpoint'],
            'class_patterns': ['Request', 'Client', 'Session'],
            'function_patterns': ['request', 'send', 'get'],
        },
        'response': {
            'keywords': ['response', 'http', 'status', 'code', 'body', 'header'],
            'class_patterns': ['Response', 'Result'],
            'function_patterns': ['parse', 'decode', 'handle'],
        },

        # Configuration and settings
        'config': {
            'keywords': ['config', 'configuration', 'settings', 'setup', 'parameter'],
            'class_patterns': ['Config', 'Settings', 'Configuration'],
            'function_patterns': ['configure', 'setup', 'initialize'],
        },
        'settings': {
            'keywords': ['settings', 'config', 'configuration', 'parameter', 'option'],
            'class_patterns': ['Settings', 'Config'],
            'function_patterns': ['get_setting', 'set_setting'],
        },

        # Data processing
        'parse': {
            'keywords': ['parse', 'parsing', 'deserialize', 'decode', 'convert'],
            'class_patterns': ['Parser', 'Decoder'],
            'function_patterns': ['parse', 'decode', 'deserialize'],
        },
        'serialize': {
            'keywords': ['serialize', 'encode', 'convert', 'format', 'dump'],
            'class_patterns': ['Serializer', 'Encoder'],
            'function_patterns': ['serialize', 'encode', 'dump'],
        },
        'filter': {
            'keywords': ['filter', 'filtering', 'search', 'match', 'condition'],
            'class_patterns': ['Filter', 'Matcher'],
            'function_patterns': ['filter', 'match', 'check'],
        },

        # Logging and monitoring
        'logging': {
            'keywords': ['logging', 'log', 'logger', 'debug', 'info', 'warning'],
            'class_patterns': ['Logger', 'Log'],
            'function_patterns': ['log', 'debug', 'info'],
        },
        'debug': {
            'keywords': ['debug', 'logging', 'trace', 'verbose', 'output'],
            'class_patterns': ['Logger', 'Debugger'],
            'function_patterns': ['debug', 'trace', 'log'],
        },

        # Concurrency
        'thread': {
            'keywords': ['thread', 'threading', 'concurrent', 'lock', 'mutex', 'async'],
            'class_patterns': ['Thread', 'Lock', 'Semaphore'],
            'function_patterns': ['thread', 'lock', 'spawn'],
        },
        'async': {
            'keywords': ['async', 'await', 'asynchronous', 'promise', 'future', 'callback'],
            'class_patterns': ['Task', 'Future', 'Promise'],
            'function_patterns': ['async', 'await', 'callback'],
        },
    }

    def __init__(self):
        """Initialize query expander with pattern knowledge."""
        self.concepts = set(self.CONCEPT_PATTERNS.keys())

    def expand_query(self, query: str) -> str:
        """
        Expand an abstract query into concrete keywords.

        Args:
            query: Original user query

        Returns:
            Expanded query with additional keywords
        """
        logger.debug(f"Expanding query: {query}")

        # Find matching concepts
        matched_concepts = self._detect_concepts(query)
        logger.debug(f"Detected concepts: {matched_concepts}")

        if not matched_concepts:
            logger.debug("No concepts matched, returning original query")
            return query

        # Collect all keywords from matched concepts
        expanded_keywords = set()
        for concept in matched_concepts:
            pattern = self.CONCEPT_PATTERNS[concept]
            expanded_keywords.update(pattern['keywords'])
            logger.debug(f"Added keywords from '{concept}': {pattern['keywords']}")

        # Combine original query with expanded keywords
        expanded = f"{query} {' '.join(expanded_keywords)}"
        logger.debug(f"Expanded query: {expanded}")

        return expanded

    def get_class_patterns(self, query: str) -> List[str]:
        """
        Get class name patterns that might be relevant to this query.

        Args:
            query: User query

        Returns:
            List of class patterns to search for
        """
        matched_concepts = self._detect_concepts(query)
        patterns = []

        for concept in matched_concepts:
            pattern_list = self.CONCEPT_PATTERNS[concept].get('class_patterns', [])
            patterns.extend(pattern_list)

        return patterns

    def get_function_patterns(self, query: str) -> List[str]:
        """
        Get function name patterns that might be relevant to this query.

        Args:
            query: User query

        Returns:
            List of function patterns to search for
        """
        matched_concepts = self._detect_concepts(query)
        patterns = []

        for concept in matched_concepts:
            pattern_list = self.CONCEPT_PATTERNS[concept].get('function_patterns', [])
            patterns.extend(pattern_list)

        return patterns

    def _detect_concepts(self, query: str) -> Set[str]:
        """
        Detect which concepts are relevant to this query.

        Args:
            query: User query

        Returns:
            Set of matching concept names
        """
        query_lower = query.lower()
        matched = set()

        # Direct keyword matching
        for concept in self.concepts:
            # Check if concept appears in query
            if concept in query_lower:
                matched.add(concept)
                continue

            # Check if any concept keywords appear in query
            keywords = self.CONCEPT_PATTERNS[concept]['keywords']
            if any(keyword in query_lower for keyword in keywords):
                matched.add(concept)

        return matched

    def is_abstract_query(self, query: str) -> bool:
        """
        Check if a query is abstract (would benefit from expansion).

        Abstract queries typically:
        - Start with question words (how, what, why)
        - Use vague language (stuff, thing, work)
        - Have multiple words but no specific keywords

        Args:
            query: User query

        Returns:
            True if query is abstract
        """
        query_lower = query.lower()

        # Question format is abstract
        question_words = ['how', 'what', 'why', 'when', 'where']
        if any(query_lower.startswith(w) for w in question_words):
            return True

        # Contains vague language
        vague_words = ['work', 'thing', 'stuff', 'happens', 'goes']
        if any(word in query_lower for word in vague_words):
            return True

        # Long query with varied concepts but no specific terms
        words = query_lower.split()
        if len(words) > 4 and len(set(words)) > len(words) * 0.7:
            # High vocabulary diversity suggests abstract query
            return True

        return False


# Singleton instance
_expander = None

def get_expander() -> QueryExpander:
    """Get or create QueryExpander instance."""
    global _expander
    if _expander is None:
        _expander = QueryExpander()
    return _expander
