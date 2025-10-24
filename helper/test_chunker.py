"""
Test script for DocumentChunker._chunk_code function.

This script helps visualize how the DocumentChunker splits code into chunks.
"""
import sys
from pathlib import Path

# Add the parent directory to path to import the module
sys.path.append(str(Path(__file__).parent.parent))

from hyperthymesia_cli.core.chunker import DocumentChunker

def test_code_chunking(code_sample, chunk_size=500, overlap=100):
    """
    Test the _chunk_code function with a given code sample.
    
    Args:
        code_sample (str): The code to be chunked
        chunk_size (int): Maximum size of each chunk
        overlap (int): Overlap between chunks
    """
    print("\n" + "="*80)
    print("TESTING CODE CHUNKING")
    print("="*80)
    
    # Initialize the chunker with specified parameters
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    
    # Get the chunks
    chunks = chunker._chunk_code(code_sample)
    
    # Print results
    print(f"\nSplit into {len(chunks)} chunks:")
    print("-" * 40)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nCHUNK {i} (length: {len(chunk)}):")
        print("-" * 30)
        print(chunk.strip())
        print("\n" + "-" * 30)

if __name__ == "__main__":
    # Example Python code to test
    sample_python = """
class ExampleClass:
    def __init__(self, name):
        self.name = name
        self.count = 0
    
    def increment(self):
        self.count += 1
        return self.count

def calculate_sum(a, b):
    return a + b

def greet(name: str) -> str:
    return f"Hello, {name}!"
    """
    
    # Run the test with the sample code
    test_code_chunking(
        code_sample=sample_python,
        chunk_size=200,  # Smaller chunk size for testing
        overlap=50       # Smaller overlap for testing
    )
    
    # You can also test with code from a file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
                test_code_chunking(
                    code_sample=file_content,
                    chunk_size=500,
                    overlap=100
                )
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("\nTip: You can also test with your own file by running:")
        print(f"python {__file__} path/to/your/file.py")
