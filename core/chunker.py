"""
Document chunking for RAG (Retrieval Augmented Generation).
Splits documents into meaningful chunks for better context retrieval.
"""
from typing import List, Dict
from pathlib import Path
import re


class DocumentChunker:
    """
    Intelligent document chunking for RAG.
    
    Strategies:
    - Code files: Split by function/class
    - Markdown: Split by headers
    - Text: Split by paragraphs
    - PDFs: Split by pages/paragraphs
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            overlap: Overlap between chunks (for context continuity)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, content: str, file_path: str, 
                      file_type: str) -> List[Dict]:
        """
        Chunk a document based on its type.
        
        Args:
            content: Document content
            file_path: Path to file (for metadata)
            file_type: File extension
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        # Choose chunking strategy based on file type
        if file_type in {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c'}:
            chunks = self._chunk_code(content)
        elif file_type in {'.md', '.markdown'}:
            chunks = self._chunk_markdown(content)
        else:
            chunks = self._chunk_text(content)
        
        # Add metadata to each chunk
        result = []
        for i, chunk_text in enumerate(chunks):
            result.append({
                'content': chunk_text,
                'file_path': file_path,
                'file_type': file_type,
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
        
        return result
    
    def _chunk_code(self, content: str) -> List[str]:
        """
        Chunk code by functions and classes.
        
        Tries to keep functions/classes together for better context.
        Falls back to simple chunking if no clear boundaries.
        """
        chunks = []
        
        # Try to split by function/class definitions
        # Pattern matches: def, class, func, function, etc.
        pattern = r'((?:^|\n)(?:def|class|func|function|fn|struct|impl)\s+\w+.*?(?=\n(?:def|class|func|function|fn|struct|impl)\s+|\Z))'
        
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        code_blocks = [m.group(1) for m in matches]
        
        if code_blocks:
            # Found function/class boundaries
            current_chunk = ""
            
            for block in code_blocks:
                if len(current_chunk) + len(block) < self.chunk_size:
                    current_chunk += block
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = block
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            # Fallback to simple chunking
            chunks = self._chunk_text(content)
        
        return chunks
    
    def _chunk_markdown(self, content: str) -> List[str]:
        """
        Chunk markdown by headers.
        
        Keeps sections together for better context.
        """
        chunks = []
        
        # Split by headers (# ## ### etc.)
        sections = re.split(r'\n(#{1,6}\s+.*)\n', content)
        
        current_chunk = ""
        current_header = ""
        
        for i, section in enumerate(sections):
            # Check if this is a header
            if re.match(r'^#{1,6}\s+', section):
                current_header = section
                continue
            
            section_text = current_header + "\n" + section if current_header else section
            
            if len(current_chunk) + len(section_text) < self.chunk_size:
                current_chunk += section_text
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section_text
            
            current_header = ""
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else self._chunk_text(content)
    
    def _chunk_text(self, content: str) -> List[str]:
        """
        Generic text chunking with overlap.
        
        Splits on paragraph boundaries when possible.
        """
        chunks = []
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too large, split it
                if len(para) > self.chunk_size:
                    # Split large paragraph into sentences
                    sentences = re.split(r'[.!?]+\s+', para)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) < self.chunk_size:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    current_chunk = temp_chunk
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of token count.
        Rule of thumb: 1 token ≈ 4 characters for English.
        
        Args:
            text: Text to estimate
        
        Returns:
            Estimated token count
        """
        return len(text) // 4


# Example usage and testing
if __name__ == "__main__":
    chunker = DocumentChunker(chunk_size=500, overlap=50)
    
    # Test with code
    code = """
def authenticate_user(username, password):
    '''Authenticate user with credentials.'''
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return generate_token(user)
    return None

def generate_token(user):
    '''Generate JWT token for user.'''
    payload = {
        'user_id': user.id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')
"""
    
    chunks = chunker.chunk_document(code, 'auth.py', '.py')
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Tokens: ~{chunker.estimate_tokens(chunk['content'])}")
        print(f"  Content preview: {chunk['content'][:100]}...")