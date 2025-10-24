
"""
Database migration utilities.
"""
from storage.db import Database
import sqlite3


def needs_migration(db: Database) -> bool:
    """
    Check if FTS table needs migration to support filename search.
    
    Returns:
        True if migration is needed
    """
    cursor = db.conn.cursor()
    
    try:
        # Check if filename column exists
        cursor.execute("SELECT filename FROM documents_fts LIMIT 1")
        return False  # Schema is up to date
    except sqlite3.OperationalError:
        return True  # Old schema - needs migration


def migrate_for_filename_search(db: Database):
    """
    Migrate database to support filename search.
    Drops and recreates FTS table with new schema.
    """
    cursor = db.conn.cursor()
    
    print("\n🔧 Upgrading database for filename search...")
    
    # Drop old FTS table
    cursor.execute("DROP TABLE IF EXISTS documents_fts")
    
    # Create new FTS table with filename column
    cursor.execute("""
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            document_id UNINDEXED,
            filename,
            content,
            tokenize='porter unicode61'
        )
    """)
    
    db.conn.commit()
    
    print("✅ Database upgraded successfully!")
    print("\n⚠️  Please re-index your files to enable filename search:")
    print("   hyperthymesia index refresh")
    print()