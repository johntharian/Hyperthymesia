"""
Text file parser.
"""
from pathlib import Path
from typing import Optional


def parse(file_path: Path) -> Optional[str]:
    """
    Extract text content from a plain text file.
    
    Args:
        file_path: Path to the text file
    
    Returns:
        Extracted text content or None if parsing fails
    """
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def is_supported(file_path: Path) -> bool:
    """Check if this parser supports the given file."""
    text_extensions = {'.txt', '.md', '.markdown', '.log', '.csv', 
                      '.json', '.xml', '.yaml', '.yml', '.ini', '.cfg'}
    return file_path.suffix.lower() in text_extensions