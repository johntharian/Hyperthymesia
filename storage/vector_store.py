"""
Vector database management using ChromaDB for semantic search.
"""
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
import platformdirs
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Manages ChromaDB for semantic/vector search."""
    
    def __init__(self, persist_dir: Optional[Path] = None, 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store with ChromaDB.
        
        Args:
            persist_dir: Directory to persist ChromaDB data
            model_name: Sentence transformer model to use for embeddings
        """
        if persist_dir is None:
            # Use platform-appropriate data directory
            app_dir = platformdirs.user_data_dir("hyperthymesia", "hyperthymesia")
            persist_dir = Path(app_dir) / "vectordb"
        
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    
    def add_document(self, doc_id: int, content: str, metadata: Dict):
        """
        Add a document to the vector store.
        
        Args:
            doc_id: Document ID (from SQLite)
            content: Text content to embed
            metadata: Document metadata (path, filename, etc.)
        """
        # Generate embedding
        embedding = self.model.encode(content).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
            ids=[str(doc_id)]
        )
    
    def add_documents_batch(self, doc_ids: List[int], contents: List[str], 
                           metadatas: List[Dict]):
        """
        Add multiple documents in batch (more efficient).
        
        Args:
            doc_ids: List of document IDs
            contents: List of text contents
            metadatas: List of metadata dicts
        """
        if not doc_ids:
            return
        
        # Generate embeddings in batch
        embeddings = self.model.encode(contents).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=[str(doc_id) for doc_id in doc_ids]
        )
    
    def search(self, query: str, limit: int = 10, 
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Semantic search using vector similarity.
        
        Args:
            query: Search query (natural language)
            limit: Maximum results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of matching documents with similarity scores
        """
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        if results['ids'][0]:  # Check if we got results
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': int(doc_id),
                    'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i]
                })
        
        return formatted_results
    
    def delete_document(self, doc_id: int):
        """Delete a document from the vector store."""
        try:
            self.collection.delete(ids=[str(doc_id)])
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
    
    def delete_by_source(self, source_path: str):
        """Delete all documents from a specific source."""
        try:
            self.collection.delete(
                where={"source_path": source_path}
            )
        except Exception as e:
            print(f"Error deleting documents from source {source_path}: {e}")
    
    def get_count(self) -> int:
        """Get total number of documents in vector store."""
        return self.collection.count()
    
    def clear(self):
        """Clear all documents from the vector store."""
        self.client.delete_collection(name="documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )