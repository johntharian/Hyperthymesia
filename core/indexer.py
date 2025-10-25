"""
Core indexer for scanning and indexing files.
"""

from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from parsers import code as code_parser
from parsers import pdf as pdf_parser
from parsers import text as text_parser
from utils.gitignore import GitignoreFilter 
from storage.db import Database
from storage.vector_store import VectorStore
from tqdm import tqdm


class Indexer:
    """Handles file scanning and indexing for both keyword and semantic search."""

    def __init__(
        self, db: Optional[Database] = None, vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize indexer.

        Args:
            db: Database instance (creates new if None)
            vector_store: VectorStore instance (creates new if None)
        """
        self.db = db or Database()
        self.vector_store = vector_store or VectorStore()

        # Supported parsers mapping
        self.parsers = {"text": text_parser, "pdf": pdf_parser, "code": code_parser}

    def index_path(
        self,
        path: str,
        recursive: bool = True,
        name: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Index a file or directory.

        Args:
            path: Path to file or directory
            recursive: Whether to recursively index subdirectories
            name: Optional name for this source
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with indexing statistics
        """
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Add source to database
        source_id = self.db.add_source(str(path_obj), name)

        # Collect files to index
        if path_obj.is_file():
            files = [path_obj]
        else:
            pattern = "**/*" if recursive else "*"
            all_files = [f for f in path_obj.glob(pattern) if f.is_file()]

            # Apply gitignore filtering
            gitignore = GitignoreFilter(path_obj)
            files = gitignore.filter_files(all_files)

            ignored_count = len(all_files) - len(files)
            
            if ignored_count > 0:
                print(f"Filtered out {ignored_count} files (build artifacts, dependencies, etc.)")


        # Filter to only supported files
        supported_files = [f for f in files if self._is_supported(f)]

        print(f"\nFound {len(supported_files)} supported files to index...")

        # Index files
        indexed_count = 0
        failed_count = 0
        total_size = 0

        # Batch processing for vector store
        batch_ids = []
        batch_contents = []
        batch_metadatas = []
        batch_size = 10  # Process 10 files at a time

        for file_path in tqdm(supported_files, desc="Indexing files"):
            try:
                result = self._index_file(source_id, file_path)
                if result:
                    indexed_count += 1
                    total_size += result["size"]

                    # Add to batch
                    batch_ids.append(result["doc_id"])
                    batch_contents.append(result["content"])
                    batch_metadatas.append(result["metadata"])

                    # Process batch if full
                    if len(batch_ids) >= batch_size:
                        self.vector_store.add_documents_batch(
                            batch_ids, batch_contents, batch_metadatas
                        )
                        batch_ids = []
                        batch_contents = []
                        batch_metadatas = []
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
                failed_count += 1

        # Process remaining batch
        if batch_ids:
            self.vector_store.add_documents_batch(
                batch_ids, batch_contents, batch_metadatas
            )

        # Update source statistics
        self.db.update_source_stats(source_id)

        return {
            "source_id": source_id,
            "indexed": indexed_count,
            "failed": failed_count,
            "total_size": total_size,
        }

    def _index_file(self, source_id: int, file_path: Path) -> Optional[dict]:
        """
        Index a single file.

        Args:
            source_id: ID of the source this file belongs to
            file_path: Path to the file

        Returns:
            Dictionary with indexing results or None if failed
        """
        # Get file metadata
        stat = file_path.stat()
        file_type = file_path.suffix.lower()
        modified_at = datetime.fromtimestamp(stat.st_mtime)

        # Parse file content
        content = self._parse_file(file_path)
        if not content:
            return None

        # Enhance content with filename for better semantic search
        # The filename often contains important keywords
        filename_without_ext = file_path.stem.replace("_", " ").replace("-", " ")
        enhanced_content = f"Filename: {filename_without_ext}\n\n{content}"

        # Store in SQLite (keyword search)
        doc_id = self.db.add_document(
            source_id=source_id,
            path=str(file_path),
            content=content,
            file_type=file_type,
            size=stat.st_size,
            modified_at=modified_at,
        )

        # Prepare metadata for vector store
        metadata = {
            "path": str(file_path),
            "filename": file_path.name,
            "file_type": file_type,
            "size": stat.st_size,
            "source_id": source_id,
        }

        return {
            "doc_id": doc_id,
            "content": enhanced_content,  # Use enhanced content for embeddings
            "metadata": metadata,
            "size": stat.st_size,
        }

    def _parse_file(self, file_path: Path) -> Optional[str]:
        """
        Parse a file and extract its content using the appropriate parser.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content or None
        """
        # Find and use the appropriate parser
        for parser_name, parser in self.parsers.items():
            if parser.is_supported(file_path):
                return parser.parse(file_path)

        # No parser found for this file type
        return None

    def _is_supported(self, file_path: Path) -> bool:
        """Check if a file type is supported by any parser."""
        return any(parser.is_supported(file_path) for parser in self.parsers.values())

    def reindex_all(self):
        """Re-index all sources."""
        sources = self.db.get_sources()

        for source in sources:
            print(f"\nRe-indexing: {source['path']}")
            try:
                self.index_path(source["path"], recursive=True)
            except Exception as e:
                print(f"Error re-indexing {source['path']}: {e}")

    def remove_source(self, path: str):
        """
        Remove a source and all its indexed documents.

        Args:
            path: Path to the source to remove
        """
        abs_path = str(Path(path).resolve())

        # Remove from vector store
        self.vector_store.delete_by_source(abs_path)

        # Remove from database
        self.db.remove_source(abs_path)

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        db_stats = self.db.get_stats()
        vector_count = self.vector_store.get_count()

        return {
            "sources": db_stats["source_count"],
            "documents": db_stats["document_count"],
            "total_size": db_stats["total_size"],
            "vector_count": vector_count,
        }
