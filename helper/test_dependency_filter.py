"""
Dependency Filter Visualization

This script demonstrates how dependencies are filtered out from search results,
similar to how the search command handles it.
"""
from typing import List, Dict, Set
from dataclasses import dataclass, asdict
import json

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

@dataclass
class SearchResult:
    """Represents a search result item."""
    id: str
    path: str
    content: str = ""
    score: float = 0.0

def is_dependency(result: Dict) -> bool:
    """
    Check if a search result is from a dependency directory.
    
    Args:
        result: Search result dictionary with a 'path' key
        
    Returns:
        bool: True if the result is from a dependency directory
    """
    path = result.get('path', '').lower()
    return any(f'/{dep}/' in path or path.endswith(f'/{dep}') for dep in DEPENDENCY_DIRS)

def generate_sample_data() -> List[Dict]:
    """Generate sample search results with various paths."""
    return [
        {"id": "1", "path": "/project/src/main.py", "content": "Main application code"},
        {"id": "2", "path": "/project/node_modules/express/index.js", "content": "Express framework"},
        {"id": "3", "path": "/project/docs/README.md", "content": "Project documentation"},
        {"id": "4", "path": "/project/venv/lib/python3.9/site-packages/requests/__init__.py", "content": "Requests library"},
        {"id": "5", "path": "/project/static/js/app.js", "content": "Frontend JavaScript"},
        {"id": "6", "path": "/project/vendor/package/helper.py", "content": "Vendor package"},
        {"id": "7", "path": "/project/tests/test_main.py", "content": "Test cases"},
        {"id": "8", "path": "/project/node_modules/lodash/lodash.js", "content": "Lodash utility"},
    ]

def print_results(title: str, results: List[Dict]):
    """Print search results in a formatted way."""
    print(f"\n{title}:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['path']}")
        print(f"   ID: {result['id']}")
        print(f"   Content: {result['content']}")
        print("-" * 80)

def main():
    # Generate sample search results
    all_results = generate_sample_data()
    
    # Print all results
    print("🔍 ALL SEARCH RESULTS")
    print("=" * 80)
    print_results("All Results", all_results)
    
    # Filter out dependencies
    filtered_results = [r for r in all_results if not is_dependency(r)]
    
    # Print filtered results
    print("\n✅ FILTERED RESULTS (No Dependencies)")
    print("=" * 80)
    if filtered_results:
        print_results("Filtered Results", filtered_results)
    else:
        print("No results after filtering")
    
    # Show what was filtered out
    filtered_out = [r for r in all_results if is_dependency(r)]
    if filtered_out:
        print("\n❌ FILTERED OUT (Dependencies)")
        print("=" * 80)
        for i, result in enumerate(filtered_out, 1):
            print(f"{i}. {result['path']} (ID: {result['id']})")
    
    # Print summary
    print("\n📊 SUMMARY")
    print("=" * 80)
    print(f"Total results: {len(all_results)}")
    print(f"After filtering: {len(filtered_results)}")
    print(f"Filtered out: {len(all_results) - len(filtered_results)}")

if __name__ == "__main__":
    main()
