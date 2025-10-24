# helper/pdf/parse_pdf.py
import sys
from pathlib import Path
from parsers.pdf import parse as parse_pdf  # Import the existing PDF parser

def preview_pdf(file_path: str, max_chars: int = 1000):
    """
    Parse a PDF file and print a preview of its extracted text content.
    
    Args:
        file_path: Path to the PDF file
        max_chars: Maximum characters to print (to avoid overwhelming output)
    
    Prints the extracted text or an error message if parsing fails.
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    if not path.suffix.lower() == '.pdf':
        print(f"Error: '{file_path}' is not a PDF file.")
        return
    
    print(f"Parsing PDF: {file_path}")
    print("=" * 50)
    
    # Use the existing parser
    content = parse_pdf(path)
    
    if content:
        # Truncate if too long
        preview = content[:max_chars]
        if len(content) > max_chars:
            preview += "\n... [truncated]"
        
        print("Extracted Content:")
        print(preview)
        print(f"\nTotal characters extracted: {len(content)}")
    else:
        print("No text content could be extracted (PDF might be image-based or corrupted).")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_pdf.py <pdf_file_path>")
        print("Example: python parse_pdf.py /path/to/document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]  # Get PDF path from first argument
    preview_pdf(pdf_path)