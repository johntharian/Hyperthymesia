"""
Code file parser with basic symbol extraction.
"""
from pathlib import Path
from typing import Optional
import re


# Code file extensions
CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
    '.sh', '.bash', '.zsh', '.sql', '.r', '.m', '.mm',
    '.html', '.css', '.scss', '.sass', '.less',
}


def parse(file_path: Path) -> Optional[str]:
    """
    Extract text content from code files with metadata.
    
    Args:
        file_path: Path to the code file
    
    Returns:
        Extracted content with symbols and code
    """
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract symbols (functions, classes) for better search
        symbols = extract_symbols(content, file_path.suffix)
        
        # Build enhanced content
        enhanced_parts = []
        
        # Add symbols section if found
        if symbols:
            enhanced_parts.append("Symbols: " + ", ".join(symbols))
            enhanced_parts.append("")  # Empty line
        
        # Add original content
        enhanced_parts.append(content)
        
        return "\n".join(enhanced_parts)
        
    except UnicodeDecodeError:
        # Not a text file
        return None
    except Exception as e:
        print(f"Error parsing code file {file_path}: {e}")
        return None


def extract_symbols(content: str, extension: str) -> list:
    """
    Extract function and class names from code.
    
    Args:
        content: File content
        extension: File extension
    
    Returns:
        List of symbol names
    """
    symbols = []
    
    # Python
    if extension == '.py':
        # Find functions: def function_name(
        symbols.extend(re.findall(r'def\s+(\w+)\s*\(', content))
        # Find classes: class ClassName
        symbols.extend(re.findall(r'class\s+(\w+)\s*[:\(]', content))
    
    # JavaScript/TypeScript
    elif extension in {'.js', '.jsx', '.ts', '.tsx'}:
        # Find functions: function name(, const name = function(, const name = (
        symbols.extend(re.findall(r'function\s+(\w+)\s*\(', content))
        symbols.extend(re.findall(r'const\s+(\w+)\s*=\s*(?:function|\()', content))
        symbols.extend(re.findall(r'let\s+(\w+)\s*=\s*(?:function|\()', content))
        # Find classes: class Name
        symbols.extend(re.findall(r'class\s+(\w+)\s*[{]', content))
    
    # Java/C#/C++
    elif extension in {'.java', '.cs', '.cpp', '.c', '.h', '.hpp'}:
        # Find functions: type name(
        symbols.extend(re.findall(r'\w+\s+(\w+)\s*\([^)]*\)\s*{', content))
        # Find classes: class Name
        symbols.extend(re.findall(r'class\s+(\w+)\s*[{:]', content))
    
    # Go
    elif extension == '.go':
        # Find functions: func name(
        symbols.extend(re.findall(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(', content))
        # Find types: type Name struct
        symbols.extend(re.findall(r'type\s+(\w+)\s+struct', content))
    
    # Rust
    elif extension == '.rs':
        # Find functions: fn name(
        symbols.extend(re.findall(r'fn\s+(\w+)\s*[<\(]', content))
        # Find structs: struct Name
        symbols.extend(re.findall(r'struct\s+(\w+)\s*[{<]', content))
    
    # Remove duplicates and common false positives
    symbols = list(set(symbols))
    symbols = [s for s in symbols if len(s) > 1 and not s.startswith('_')]
    
    return sorted(symbols)[:50]  # Limit to top 50


def is_supported(file_path: Path) -> bool:
    """Check if this parser supports the given file."""
    return file_path.suffix.lower() in CODE_EXTENSIONS