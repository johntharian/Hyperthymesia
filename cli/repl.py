"""
Interactive REPL shell for Hyperthymesia.
"""
import click
import sys
from pathlib import Path

from core.indexer import Indexer
from core.rag_retriever import RAGRetriever
from core.local_llm import LocalLLM
from core.intelligent_searcher import IntelligentSearcher
from storage.db import Database
from cli.formatters import print_success, print_error, print_info, print_results


class HyperthymesiaShell:
    """Interactive shell for Hyperthymesia."""
    
    def __init__(self):
        """Initialize shell and load models."""
        self.running = True
        
        # Print welcome
        print_success("🧠 Welcome to Hyperthymesia!")
        print_info("Loading models... (this takes ~10s first time)\n")
        
        # Pre-load everything (so commands are instant)
        self.indexer = Indexer()
        self.searcher = IntelligentSearcher(use_llm=True)  # No LLM for search in REPL
        self.rag_retriever = RAGRetriever()
        
        # Try to initialize local LLM
        self.llm = LocalLLM()
        if self.llm.is_available():
            print_success("✓ Local LLM ready")
        else:
            print_info("ℹ️  Local LLM not available (Ollama not running)")
        
        self.db = Database()

        print()
        print_success("Ready! Just ask your questions naturally or type 'help' for commands.\n")
    
    def run(self):
        """Run the interactive shell."""
        while self.running:
            try:
                # Get user input
                user_input = input("mesia> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                self.execute_command(user_input)
                
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to leave")
                continue
            except EOFError:
                break
    
    def execute_command(self, user_input: str):
        """Execute a command."""
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Handle exit/quit (with or without /)
        if command in ['exit', 'quit', 'q', '/exit', '/quit', '/q']:
            print_info("Goodbye! 👋")
            self.running = False

        # Handle help (with or without /)
        elif command in ['help', '/help']:
            self.show_help()

        # Handle clear (with or without /)
        elif command in ['clear', 'cls', '/clear', '/cls']:
            import os
            os.system('clear' if os.name != 'nt' else 'cls')

        # Handle slash commands
        elif command.startswith('/'):
            # Remove the / for processing
            cmd_name = command[1:]

            if cmd_name == 'search' or cmd_name == 's':
                if not args:
                    print_error("Usage: /search <query>")
                    return
                self.cmd_search(args)

            elif cmd_name == 'index':
                self.cmd_index(args)

            elif cmd_name == 'stats':
                self.cmd_stats()

            else:
                print_error(f"Unknown command: {command}")
                print_info("Type 'help' for available commands")

        # Treat everything else as a natural question/ask command
        else:
            # The entire input is the question
            self.cmd_ask(user_input)
    
    def show_help(self):
        """Show help message."""
        print()
        print_success("Available Commands:\n")
        print("  Natural Questions (Just Ask!):")
        print("    <question>       Any natural language question")
        print("    Example: 'what does this function do?'\n")
        print("  Indexing (Requires /):")
        print("    /index add <path>   Index a directory")
        print("    /index list         List indexed locations\n")
        print("  Searching (Optional /):")
        print("    /search <query>     Search for files\n")
        print("  Other Commands (With or Without /):")
        print("    /stats or stats             Show index statistics")
        print("    /help or help               Show this help")
        print("    /clear or clear             Clear screen")
        print("    /exit or exit, /quit or quit, /q or q   Exit shell\n")
    
    def cmd_search(self, query: str):
        """Execute search command."""
        print()
        
        try:
            response = self.searcher.search(
                query=query,
                limit=10,
                verbose=False
            )
            
            results = response['results']
            
            # Filter dependencies
            dep_dirs = {'node_modules', '__pycache__', '.venv', 'venv', 'vendor'}
            results = [r for r in results 
                      if not any(dep in r['path'] for dep in dep_dirs)]
            
            if not results:
                print_info("No results found")
            else:
                print_results(results, query)
        
        except Exception as e:
            print_error(f"Search error: {e}")
        
        print()
    
    def cmd_ask(self, question: str):
        """Execute ask command."""
        print()
        
        if not self.llm.is_available():
            print_error("Local LLM not available. Start Ollama first:")
            print_info("  ollama serve")
            print()
            return
        
        try:
            print_info("⏳ Thinking...\n")
            
            # Retrieve context
            result = self.rag_retriever.retrieve_context(question, num_chunks=5)
            
            if not result['context']:
                print_info("No relevant documents found")
                print()
                return
            
            # Generate answer
            answer = self.llm.answer_question(
                question,
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
                print()
        
        except Exception as e:
            print_error(f"Error: {e}")
        
        print()
    
    def cmd_index(self, args: str):
        """Execute index commands."""
        print()
        
        if not args:
            print_error("Usage: index <add|list|stats> [path]")
            return
        
        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower()
        
        if subcommand == 'add':
            if len(parts) < 2:
                print_error("Usage: index add <path>")
                return
            
            path = parts[1]
            try:
                print_info(f"Indexing: {path}")
                stats = self.indexer.index_path(path, recursive=True)
                print_success(f"✓ Indexed {stats['indexed']} files")
            except Exception as e:
                print_error(f"Error: {e}")
        
        elif subcommand == 'list':
            sources = self.db.get_sources()
            if not sources:
                print_info("No indexed locations")
            else:
                print_info(f"Indexed locations ({len(sources)}):\n")
                for source in sources:
                    print(f"  • {source['path']}")
                    print(f"    Files: {source['file_count']}")
        
        else:
            print_error(f"Unknown subcommand: {subcommand}")
        
        print()
    
    def cmd_stats(self):
        """Show statistics."""
        print()
        
        try:
            stats = self.indexer.get_stats()
            print_info("Index Statistics:\n")
            print(f"  Indexed locations: {stats['sources']}")
            print(f"  Total files: {stats['documents']}")
            print(f"  Vector embeddings: {stats['vector_count']}")
        except Exception as e:
            print_error(f"Error: {e}")
        
        print()


def start_repl():
    """Start the interactive shell."""
    shell = HyperthymesiaShell()
    shell.run()


if __name__ == "__main__":
    start_repl()