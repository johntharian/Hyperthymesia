"""
CLI commands for SearchAll.
"""
import click
from pathlib import Path
from cli.formatters import print_success, print_error, print_info, print_results, format_size
from core.indexer import Indexer
from core.search import Searcher
from storage.db import Database


# Create global instances (lazy-loaded)
_indexer = None
_db = None


def get_indexer():
    """Get or create indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer


def get_db():
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


@click.group()
def index():
    """Manage indexed folders and files."""
    pass


@index.command('add')
@click.argument('path', type=click.Path(exists=True))
@click.option('--name', '-n', help='Optional name for this indexed location')
@click.option('--recursive/--no-recursive', '-r/-R', default=True, 
              help='Recursively index subdirectories')
def index_add(path, name, recursive):
    """
    Add a folder or file to the index.
    
    PATH: The folder or file path to index
    """
    abs_path = Path(path).resolve()
    
    print_info(f"Indexing: {abs_path}")
    print_info(f"Recursive: {recursive}\n")
    
    try:
        indexer = get_indexer()
        stats = indexer.index_path(str(abs_path), recursive=recursive, name=name)
        
        print()  # New line after progress bar
        print_success(f"✓ Successfully indexed: {abs_path}")
        print_info(f"  Files indexed: {stats['indexed']}")
        if stats['failed'] > 0:
            print_info(f"  Files failed: {stats['failed']}")
        print_info(f"  Total size: {format_size(stats['total_size'])}")
        
    except Exception as e:
        print_error(f"✗ Error indexing: {e}")


@index.command('list')
def index_list():
    """List all indexed locations."""
    try:
        db = get_db()
        sources = db.get_sources()
        
        if not sources:
            print_info("No indexed locations yet. Use 'hyperthymesia index add <path>' to get started.")
        else:
            print_info(f"Indexed locations ({len(sources)}):\n")
            for source in sources:
                print(f"  • {source['path']}")
                if source['name']:
                    print(f"    Name: {source['name']}")
                print(f"    Last indexed: {source['last_indexed'] or 'Never'}")
                print(f"    Files: {source['file_count']}")
                print(f"    Size: {format_size(source['total_size'])}")
                print()
    except Exception as e:
        print_error(f"Error listing sources: {e}")


@index.command('remove')
@click.argument('path', type=click.Path())
def index_remove(path):
    """
    Remove a location from the index.
    
    PATH: The folder or file path to remove
    """
    abs_path = Path(path).resolve()
    
    if click.confirm(f'Remove {abs_path} from index?'):
        try:
            indexer = get_indexer()
            indexer.remove_source(str(abs_path))
            print_success(f"✓ Removed {abs_path} from index")
        except Exception as e:
            print_error(f"✗ Error removing source: {e}")
    else:
        print_info("Cancelled")


@index.command('refresh')
@click.option('--path', '-p', type=click.Path(exists=True), 
              help='Refresh specific path only')
def index_refresh(path):
    """
    Refresh the index for all or specific locations.
    """
    try:
        indexer = get_indexer()
        
        if path:
            abs_path = Path(path).resolve()
            print_info(f"Refreshing index for: {abs_path}\n")
            stats = indexer.index_path(str(abs_path), recursive=True)
            print()
            print_success(f"✓ Refreshed {abs_path}")
            print_info(f"  Files indexed: {stats['indexed']}")
        else:
            print_info("Refreshing all indexed locations...\n")
            indexer.reindex_all()
            print()
            print_success("✓ Index refreshed successfully")
    except Exception as e:
        print_error(f"✗ Error refreshing index: {e}")


@index.command('stats')
def index_stats():
    """Show statistics about the indexed data."""
    try:
        indexer = get_indexer()
        stats = indexer.get_stats()
        
        print_info("Index Statistics:\n")
        print(f"  Indexed locations: {stats['sources']}")
        print(f"  Total files indexed: {stats['documents']}")
        print(f"  Total size: {format_size(stats['total_size'])}")
        print(f"  Vector embeddings: {stats['vector_count']}")
        
    except Exception as e:
        print_error(f"Error fetching stats: {e}")


@click.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--limit', '-l', default=10, help='Maximum number of results')
@click.option('--file-type', '-t', help='Filter by file type (e.g., pdf, txt)')
@click.option('--path', '-p', type=click.Path(), help='Search only in specific path')
@click.option('--keyword-weight', default=0.5, help='Weight for keyword search (0-1)')
@click.option('--semantic-weight', default=0.5, help='Weight for semantic search (0-1)')
@click.option('--min-score', default=0.01, help='Minimum score threshold')
def search(query, limit, file_type, path, keyword_weight, semantic_weight, min_score):
    """
    Search indexed files using hybrid search (keyword + semantic).
    
    QUERY: The search terms (can be multiple words)
    """
    
    query_string = ' '.join(query)
    
    print_info(f"🔍 Searching for: '{query_string}'")
    if file_type:
        print_info(f"   File type filter: {file_type}")
    if path:
        print_info(f"   Path filter: {path}")
    print()
    
    try:
        searcher = Searcher()
        results = searcher.search(
            query=query_string,
            limit=limit,
            file_type=file_type,
            path_filter=path,
            keyword_weight=keyword_weight,
            semantic_weight=semantic_weight,
            min_score=min_score
        )
        
        if not results:
            print_info("❌ No good matches found for your query.")
            print()
            print_info("💡 Suggestions:")
            print("  • Try different or more general keywords")
            print("  • Check your spelling")
            print("  • Make sure relevant files are indexed")
            print("  • Lower the minimum score threshold with --min-score")
            print()
            print_info("ℹ️  Use 'hyperthymesia index add <path>' to index more files")
        else:
            print_results(results, query_string)
            
    except Exception as e:
        print_error(f"Error searching: {e}")
        import traceback
        traceback.print_exc()