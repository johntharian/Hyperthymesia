# tests/test_query_analyzer.py

from core.query_analyzer import QueryAnalyzer

def test_query_analyzer():
    """Test the QueryAnalyzer with sample queries."""
    analyzer = QueryAnalyzer()

    test_queries = [
        'python tutorial',
        'machine learning',
        'that document about neural networks I read last week',
        'how do I implement async functions in python?',
        'show me files related to the Chicago project',
        'project_proposal.pdf',
    ]

    print('Query Complexity Analysis:')
    print('=' * 70)

    for query in test_queries:
        analysis = analyzer.analyze(query)
        print(f'\nQuery: "{query}"')
        print(f'  Complexity Score: {analysis["complexity_score"]}')
        print(f'  Is Complex: {analysis["is_complex"]}')
        print(f'  Strategy: {analysis["suggested_strategy"]}')
        if analysis.get('reason'):
            print(f'  Reasons: {", ".join(analysis["reason"])}')

if __name__ == "__main__":
    test_query_analyzer()