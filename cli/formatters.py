"""
Output formatting utilities for CLI.
"""
import click
from pathlib import Path


def print_success(message):
    """Print success message in green."""
    click.secho(message, fg='green')


def print_error(message):
    """Print error message in red."""
    click.secho(message, fg='red', err=True)


def print_info(message):
    """Print info message in blue."""
    click.secho(message, fg='blue')


def print_warning(message):
    """Print warning message in yellow."""
    click.secho(message, fg='yellow')


def print_results(results, query):
    """
    Format and print search results.
    
    Args:
        results: List of search result dictionaries
        query: The search query string
    """
    print_success(f"✨ Found {len(results)} result(s):\n")
    
    for idx, result in enumerate(results, 1):
        # File path and filename
        path = result.get('path', 'Unknown')
        filename = result.get('filename', Path(path).name)
        
        # Show filename prominently, path dimmed
        click.secho(f"{idx}. ", fg='white', nl=False)
        click.secho(f"{filename}", fg='cyan', bold=True)
        click.secho(f"   📁 {path}", fg='white', dim=True)
        
        # Score and match source
        score = result.get('score', 0)
        matched_in = result.get('matched_in', 'unknown')
        
        # Color code based on match source
        if matched_in == 'keyword+semantic':
            match_color = 'green'
            match_label = '🎯 Both'
        elif matched_in == 'keyword':
            match_color = 'yellow'
            match_label = '🔤 Keyword'
        elif matched_in == 'semantic':
            match_color = 'magenta'
            match_label = '🧠 Semantic'
        else:
            match_color = 'white'
            match_label = '❓ Unknown'
        
        click.secho(f"   Score: {score:.4f} | Match: ", fg='white', dim=True, nl=False)
        click.secho(match_label, fg=match_color, dim=True)
        
        # File metadata
        size = result.get('size', 0)
        modified = result.get('modified', 'Unknown')
        file_type = result.get('file_type', '')
        click.secho(f"   {file_type} | {format_size(size)} | Modified: {modified}", 
                   fg='white', dim=True)
        
        # Context snippet
        snippet = result.get('snippet', '')
        if snippet:
            # Highlight query terms in snippet
            highlighted = highlight_text(snippet, query)
            click.secho("   ", nl=False)
            click.echo(highlighted)
        
        click.echo()  # Empty line between results


def highlight_text(text, query):
    """
    Highlight query terms in text snippet.
    
    Args:
        text: The text to highlight
        query: The search query
    
    Returns:
        Text with highlighted query terms
    """
    # Simple highlighting - can be improved
    words = query.lower().split()
    result = text
    
    for word in words:
        # Case-insensitive replace with styled version
        import re
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        result = pattern.sub(
            lambda m: click.style(m.group(), fg='yellow', bold=True),
            result
        )
    
    return result


def format_size(size_bytes):
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_path(path, max_length=60):
    """
    Format long paths by truncating the middle.
    
    Args:
        path: File path
        max_length: Maximum length before truncation
    
    Returns:
        Formatted path string
    """
    path_str = str(path)
    if len(path_str) <= max_length:
        return path_str
    
    # Truncate middle
    half = (max_length - 3) // 2
    return f"{path_str[:half]}...{path_str[-half:]}"


def print_progress_bar(current, total, prefix='', suffix='', length=50):
    """
    Print a progress bar.
    
    Args:
        current: Current progress value
        total: Total value
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
    """
    if total == 0:
        percent = 100
        filled = length
    else:
        percent = int(100 * (current / total))
        filled = int(length * current // total)
    
    bar = '█' * filled + '-' * (length - filled)
    
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    
    if current >= total:
        print()  # New line when complete