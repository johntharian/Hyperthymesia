"""
CLI commands for SearchAll.
"""
import click
from pathlib import Path
from cli.formatters import print_success, print_error, print_info, print_results, format_size
from core.indexer import Indexer
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
@click.option('--include-deps-docs', is_flag=True,
              help='Include documentation from dependencies (node_modules, venv, etc.)')
def index_add(path, name, recursive, include_deps_docs):
    """
    Add a folder or file to the index.
    
    PATH: The folder or file path to index
    """
    abs_path = Path(path).resolve()
    
    print_info(f"Indexing: {abs_path}")
    print_info(f"Recursive: {recursive}")
    if include_deps_docs:
        print_info("Including documentation from dependencies")
    print()
    
    try:
        indexer = get_indexer()
        # Pass the include_deps_docs flag through indexer options
        # (We'll need to update indexer.index_path to accept this)
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
@click.option('--detailed', '-d', is_flag=True, help='Show detailed breakdown by file type')
def index_stats(detailed):
    """Show statistics about the indexed data."""
    try:
        indexer = get_indexer()
        stats = indexer.get_stats()
        
        print_info("Index Statistics:\n")
        print(f"  Indexed locations: {stats['sources']}")
        print(f"  Total files indexed: {stats['documents']}")
        print(f"  Total size: {format_size(stats['total_size'])}")
        print(f"  Vector embeddings: {stats['vector_count']}")
        
        if detailed:
            # Show breakdown by file type
            db = get_db()
            cursor = db.conn.cursor()
            
            cursor.execute("""
                SELECT 
                    file_type,
                    COUNT(*) as count,
                    SUM(size) as total_size
                FROM documents
                GROUP BY file_type
                ORDER BY count DESC
                LIMIT 20
            """)
            
            results = cursor.fetchall()
            if results:
                print("\n  Breakdown by file type:")
                for row in results:
                    file_type = row['file_type'] or 'unknown'
                    count = row['count']
                    size = format_size(row['total_size'])
                    print(f"    {file_type:15} {count:6} files  {size}")
        
    except Exception as e:
        print_error(f"Error fetching stats: {e}")


@click.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--limit', '-l', default=10, help='Maximum number of results')
@click.option('--file-type', '-t', help='Filter by file type (e.g., pdf, txt)')
@click.option('--path', '-p', type=click.Path(), help='Search only in specific path')
@click.option('--include-deps', is_flag=True, 
              help='Include results from dependencies (node_modules, venv, etc.)')
@click.option('--force-llm', is_flag=True, help='Force LLM usage even for simple queries')
@click.option('--no-llm', is_flag=True, help='Disable LLM, use only direct search')
@click.option('--llm-provider', default='gemini', 
              type=click.Choice(['gemini', 'openai', 'anthropic']),
              help='LLM provider to use (default: gemini)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed search strategy info')
def search(query, limit, file_type, path, include_deps, force_llm, no_llm, llm_provider, verbose):
    """
    Search indexed files using intelligent hybrid search.
    
    Automatically uses LLM to optimize complex queries like questions
    or conversational searches. Simple keyword searches use fast direct search.
    
    QUERY: The search terms (can be multiple words)
    """
    from core.intelligent_searcher import IntelligentSearcher
    
    query_string = ' '.join(query)
    
    print_info(f"🔍 Searching for: '{query_string}'")
    if file_type:
        print_info(f"   File type filter: {file_type}")
    if path:
        print_info(f"   Path filter: {path}")
    if include_deps:
        print_info(f"   Including dependencies")
    print()
    
    try:
        # Check for database migration
        from utils.migration import needs_migration, migrate_for_filename_search
        from storage.db import Database
        
        db = Database()
        if needs_migration(db):
            migrate_for_filename_search(db)
        
        # Initialize intelligent searcher
        use_llm = not no_llm
        searcher = IntelligentSearcher(use_llm=use_llm)
        
        # Check if LLM is available
        if use_llm and not searcher.is_llm_available():
            print_info(f"ℹ️  LLM features not available (set {llm_provider.upper()}_API_KEY to enable)")
            print_info("   Using direct search mode\n")
        
        # Perform intelligent search
        response = searcher.search(
            query=query_string,
            limit=limit,
            file_type=file_type,
            path_filter=path,
            force_llm=force_llm,
            verbose=verbose
        )
        
        results = response['results']
        
        # Filter out dependencies if not requested
        if not include_deps:
            dep_dirs = {'node_modules', '__pycache__', '.venv', 'venv', 'env', 'site-packages', 'vendor'}
            original_count = len(results)
            results = [r for r in results if not any(dep in r['path'] for dep in dep_dirs)]
            
            if original_count > len(results):
                excluded = original_count - len(results)
                print_info(f"ℹ️  Excluded {excluded} results from dependencies. Use --include-deps to see them.\n")
        
        # Show if query was rewritten
        if response['used_llm'] and response['rewritten_query']:
            print_info(f"💡 Optimized to: '{response['rewritten_query']}'\n")
        
        if not results:
            print_info("❌ No good matches found for your query.")
            print()
            print_info("💡 Suggestions:")
            print("  • Try different or more general keywords")
            print("  • Check your spelling")
            print("  • Make sure relevant files are indexed")
            if not include_deps:
                print("  • Use --include-deps to search in dependencies")
            print()
            print_info("ℹ️  Use 'hyperthymesia index add <path>' to index more files")
        else:
            print_results(results, query_string)
            
    except Exception as e:
        print_error(f"Error searching: {e}")
        import traceback
        traceback.print_exc()


@click.command()
@click.argument('question', nargs=-1, required=True)
@click.option('--use-cloud', is_flag=False, 
              help='Use cloud LLM (requires API key) instead of local')
@click.option('--provider', default='gemini',
              type=click.Choice(['openai', 'anthropic', 'gemini']),
              help='Cloud provider if using --use-cloud')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed process')
def ask(question, use_cloud, provider, verbose):
    """
    Ask a question about your indexed documents.
    
    Uses RAG (Retrieval Augmented Generation) to answer questions based on
    your documents. Runs locally by default, or use --use-cloud for better quality.
    
    QUESTION: Your question (can be multiple words)
    
    Examples:
      hyperthymesia ask "how did I implement authentication?"
      hyperthymesia ask "what does the config file say about database?"
      hyperthymesia ask "explain the main algorithm" --use-cloud
    """
    from core.rag_retriever import RAGRetriever
    from core.local_llm import LocalLLM, CloudLLM
    
    question_string = ' '.join(question)
    
    print_info(f"💭 Question: {question_string}\n")
    
    try:
        # Initialize retriever
        if verbose:
            print_info("📚 Searching for relevant documents...")
        
        retriever = RAGRetriever()
        
        # Retrieve context
        result = retriever.retrieve_context(question_string, num_chunks=5)
        
        if not result['context']:
            print_info("❌ No relevant documents found for your question.")
            print()
            print_info("💡 Tips:")
            print("  • Make sure relevant files are indexed")
            print("  • Try rephrasing your question")
            print("  • Use more specific keywords")
            print()
            print_info("ℹ️  Index more files with: hyperthymesia index add <path>")
            return
        
        if verbose:
            print_success(f"✓ Found {result['chunks_used']} relevant chunks")
            print_info(f"   Context size: ~{len(result['context'])} characters\n")
        
        # Initialize LLM
        if use_cloud:
            if verbose:
                print_info(f"🌐 Using {provider} API...")
            
            try:
                llm = CloudLLM(provider=provider)
            except ValueError as e:
                print_error(f"\n❌ {e}")
                print_info(f"   Set {provider.upper()}_API_KEY environment variable")
                print_info(f"   Example: export {provider.upper()}_API_KEY='your-key'")
                return
        else:
            if verbose:
                print_info("🤖 Using local LLM...")
            
            llm = LocalLLM()
            
            if not llm.is_available():
                print_error("\n❌ No local LLM available.")
                print()
                print_info("Install Ollama (easiest): https://ollama.ai")
                print("  Then run: ollama pull llama3.2:3b")
                print()
                print_info("Or use cloud: --use-cloud")
                return
        
        # Generate answer
        if verbose:
            print_info("⏳ Generating answer...\n")
        else:
            print_info("⏳ Thinking...\n")
        
        answer = llm.answer_question(
            question_string,
            result['context'],
            max_tokens=500
        )
        
        # Display answer
        print_success("🤖 Answer:\n")
        print(answer)
        print()
        
        # Display sources
        if result['sources']:
            print_info("📄 Sources:\n")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['file']}")
                print(f"     {source['path']}")
                print()
        
        # Show backend info if verbose
        if verbose:
            if hasattr(llm, 'get_backend_info'):
                info = llm.get_backend_info()
                print_info(f"Backend: {info['backend']}")
        
    except Exception as e:
        print_error(f"Error answering question: {e}")
        import traceback
        if verbose:
            traceback.print_exc()


@click.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--verbose', '-v', is_flag=True, help='Show agent reasoning steps')
def agent(query, verbose):
    """
    Ask a question using intelligent agentic reasoning.

    The agent uses local LLM to:
    1. Understand your question
    2. Reason about what to do
    3. Plan and execute tools
    4. Synthesize comprehensive answers

    For complex questions, this provides more intelligent results than simple search.
    For simple queries, it automatically falls back to fast direct search.

    QUERY: Your question (can be multiple words)

    Examples:
      hyperthymesia agent "how do retries work?"
      hyperthymesia agent "where is error handling used?" --verbose
      hyperthymesia agent "explain the authentication flow"
    """
    from core.agent import HyperthymesiaAgent

    query_string = ' '.join(query)

    print_info(f"🤖 Agent Mode: {query_string}\n")

    try:
        # Initialize agent
        agent_instance = HyperthymesiaAgent()

        # Process query
        response = agent_instance.process_query(query_string, verbose=verbose)

        if not response.success:
            print_error(f"❌ {response.answer}")
            print()
            print_info("ℹ️  Make sure files are indexed: hyperthymesia index add <path>")
            return

        # Display answer
        print_success("✨ Agent Analysis:\n")
        print(response.answer)
        print()

        # Display detailed explanation if available
        if response.detailed_explanation:
            print_success("📚 Detailed Explanation:\n")
            print(response.detailed_explanation)
            print()

        # Display sources
        all_sources = set(response.sources or [])
        if response.explanation_sources:
            all_sources.update(response.explanation_sources)

        if all_sources:
            print_info("📄 Sources:\n")
            for i, source in enumerate(sorted(all_sources), 1):
                print(f"  {i}. {source}")
            print()

        # Show reasoning if verbose
        if verbose and response.reasoning_chain:
            print_info("💭 Reasoning Chain:\n")
            print(response.reasoning_chain)
            print()

            if response.steps:
                print_info("📋 Execution Steps:\n")
                for step in response.steps:
                    if step.action:
                        print(f"  Step {step.step_num}: {step.action.tool_name}")
                        print(f"    └─ {step.action.reasoning}")
                        if step.result:
                            print(f"    └─ {step.result.message}")
                print()

    except Exception as e:
        print_error(f"Error in agent processing: {e}")
        import traceback
        if verbose:
            traceback.print_exc()