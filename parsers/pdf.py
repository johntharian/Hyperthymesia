"""
PDF file parser.
"""
from pathlib import Path
from typing import Optional
import PyPDF2


def parse(file_path: Path) -> Optional[str]:
    """
    Extract text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Extracted text content or None if parsing fails
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text_parts = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            # Combine all pages
            full_text = '\n\n'.join(text_parts)
            
            # Return None if no text extracted
            if not full_text.strip():
                print(f"Warning: No text extracted from {file_path} (might be image-based PDF)")
                return None
            
            return full_text
            
    except PyPDF2.errors.PdfReadError as e:
        print(f"PDF read error for {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
        return None


def is_supported(file_path: Path) -> bool:
    """Check if this parser supports the given file."""
    return file_path.suffix.lower() == '.pdf'