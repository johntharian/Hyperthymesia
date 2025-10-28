"""
Database management for SQLite with FTS5 (keyword search).
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import platformdirs
from utils.logger import get_logger

logger = get_logger(__name__)


class Database:
    """Manages SQLite database with FTS5 for keyword search."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to database file. If None, uses default location.
        """
        if db_path is None:
            # Use platform-appropriate data directory
            app_dir = platformdirs.user_data_dir("hyperthymesia", "hyperthymesia")
            Path(app_dir).mkdir(parents=True, exist_ok=True)
            db_path = Path(app_dir) / "hyperthymesia.db"

        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Access columns by name

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()

        # Sources table - tracks indexed folders
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                name TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_indexed TIMESTAMP,
                file_count INTEGER DEFAULT 0,
                total_size INTEGER DEFAULT 0
            )
        """
        )

        # Documents table - stores file metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT,
                size INTEGER,
                created_at TIMESTAMP,
                modified_at TIMESTAMP,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
            )
        """
        )

        # FTS5 virtual table for full-text search
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                document_id UNINDEXED,
                content,
                tokenize='porter unicode61'
            )
        """
        )

        self.conn.commit()

    def add_source(self, path: str, name: Optional[str] = None) -> int:
        """
        Add a new source to index.

        Args:
            path: Path to the source directory
            name: Optional name for the source

        Returns:
            Source ID
        """
        cursor = self.conn.cursor()
        abs_path = str(Path(path).resolve())

        cursor.execute(
            """
            INSERT OR IGNORE INTO sources (path, name)
            VALUES (?, ?)
        """,
            (abs_path, name or Path(path).name),
        )

        self.conn.commit()

        # Get the source ID
        cursor.execute("SELECT id FROM sources WHERE path = ?", (abs_path,))
        return cursor.fetchone()[0]

    def get_sources(self) -> List[Dict]:
        """Get all indexed sources."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, path, name, added_at, last_indexed, file_count, total_size
            FROM sources
            ORDER BY added_at DESC
        """
        )

        return [dict(row) for row in cursor.fetchall()]

    def remove_source(self, path: str):
        """Remove a source and all its documents."""
        cursor = self.conn.cursor()
        abs_path = str(Path(path).resolve())

        # Get source ID
        cursor.execute("SELECT id FROM sources WHERE path = ?", (abs_path,))
        result = cursor.fetchone()

        if result:
            source_id = result[0]

            # Delete associated document content from FTS
            cursor.execute(
                """
                DELETE FROM documents_fts 
                WHERE document_id IN (
                    SELECT id FROM documents WHERE source_id = ?
                )
            """,
                (source_id,),
            )

            # Delete source (CASCADE will delete documents)
            cursor.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            self.conn.commit()

    def add_document(
        self,
        source_id: int,
        path: str,
        content: str,
        file_type: str,
        size: int,
        modified_at: datetime,
    ) -> int:
        """
        Add or update a document in the database.

        Args:
            source_id: ID of the source this document belongs to
            path: Full path to the document
            content: Extracted text content
            file_type: File extension/type
            size: File size in bytes
            modified_at: File modification timestamp

        Returns:
            Document ID
        """
        cursor = self.conn.cursor()
        abs_path = str(Path(path).resolve())
        filename = Path(path).name

        # Insert or replace document metadata
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents 
            (source_id, path, filename, file_type, size, modified_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (source_id, abs_path, filename, file_type, size, modified_at),
        )

        doc_id = cursor.lastrowid

        # Insert or replace in FTS table
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents_fts (document_id, content)
            VALUES (?, ?)
        """,
            (doc_id, content),
        )

        self.conn.commit()
        return doc_id

    def update_source_stats(self, source_id: int):
        """Update file count and size for a source."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            UPDATE sources
            SET file_count = (
                SELECT COUNT(*) FROM documents WHERE source_id = ?
            ),
            total_size = (
                SELECT COALESCE(SUM(size), 0) FROM documents WHERE source_id = ?
            ),
            last_indexed = CURRENT_TIMESTAMP
            WHERE id = ?
        """,
            (source_id, source_id, source_id),
        )

        self.conn.commit()

    def search_keyword(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search documents using keyword/FTS search.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching documents with metadata
        """
        cursor = self.conn.cursor()

        # FTS5 query with ranking
        # Escape single quotes and wrap in quotes to treat as phrase search
        # This prevents FTS5 syntax errors and ensures predictable search behavior
        escaped_query = query.replace("'", "''")

        cursor.execute(
            f"""
            SELECT
                d.id,
                d.path,
                d.filename,
                d.file_type,
                d.size,
                d.modified_at,
                fts.rank as score
            FROM documents_fts fts
            JOIN documents d ON fts.document_id = d.id
            WHERE documents_fts MATCH '"{escaped_query}"'
            ORDER BY fts.rank
            LIMIT ?
        """,
            (limit,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_document_content(self, doc_id: int) -> Optional[str]:
        """Get the full content of a document by ID."""
        cursor = self.conn.cursor()
        
        # First try to get the document path
        cursor.execute(
            """
            SELECT path FROM documents WHERE id = ?
            """,
            (doc_id,),
        )
        
        result = cursor.fetchone()
        if not result:
            logger.debug(f"No document found with ID: {doc_id}")
            return None
            
        doc_path = result[0]

        try:
            # Try to read the file directly
            with open(doc_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error reading file {doc_path}: {str(e)}")
            # Fallback to FTS content if file reading fails
            cursor.execute(
                """
                SELECT content FROM documents_fts WHERE document_id = ?
                """,
                (doc_id,),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_stats(self) -> Dict:
        """Get overall database statistics."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            SELECT 
                COUNT(DISTINCT s.id) as source_count,
                COUNT(d.id) as document_count,
                COALESCE(SUM(d.size), 0) as total_size
            FROM sources s
            LEFT JOIN documents d ON s.id = d.source_id
        """
        )

        return dict(cursor.fetchone())

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
