"""
Git ignore pattern matching for excluding files from indexing.
"""
from pathlib import Path
from typing import List, Set
import fnmatch


class GitignoreFilter:
    """Filter files based on .gitignore patterns and common exclusions."""
    
    # Common patterns to always ignore (even without .gitignore)
    DEFAULT_IGNORES = {
        # Version control
        '.git', '.svn', '.hg',
        
        # Build outputs
        'dist', 'build', 'target', 'out', '.next', '.nuxt',
        '*.pyc', '*.pyo', '*.so', '*.dll', '*.dylib',
        '*.class', '*.o', '*.obj',
        
        # IDE
        '.idea', '.vscode', '.vs', '*.swp', '*.swo', '*~',
        '.DS_Store', 'Thumbs.db',
        
        # Logs and temp
        '*.log', '*.tmp', '*.temp',
        '.cache', '.pytest_cache', '.mypy_cache',
        
        # Large binary files
        '*.zip', '*.tar', '*.gz', '*.rar', '*.7z',
        '*.mp4', '*.mov', '*.avi', '*.mkv',
        '*.mp3', '*.wav', '*.flac',
    }
    
    # Directories that are dependencies (index docs only)
    DEPENDENCY_DIRS = {
        'node_modules', '__pycache__', '.venv', 'venv', 'env',
        'site-packages', 'vendor', 'packages'
    }
    
    # Files to keep from dependency directories
    DEPENDENCY_DOCS = {
        'README.md', 'README.rst', 'README.txt', 'README',
        'CHANGELOG.md', 'CHANGES.md', 'HISTORY.md',
        'LICENSE', 'LICENSE.md', 'LICENSE.txt',
        'CONTRIBUTING.md', 'API.md', 'USAGE.md'
    }
    
    def __init__(self, root_path: Path, index_dependency_docs: bool = True):
        """
        Initialize gitignore filter.
        
        Args:
            root_path: Root directory to start searching for .gitignore
            index_dependency_docs: If True, index documentation from dependencies
        """
        self.root_path = Path(root_path)
        self.index_dependency_docs = index_dependency_docs
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> List[str]:
        """Load patterns from .gitignore files."""
        patterns = list(self.DEFAULT_IGNORES)
        
        # Look for .gitignore in root
        gitignore_path = self.root_path / '.gitignore'
        if gitignore_path.exists():
            try:
                with open(gitignore_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception as e:
                print(f"Warning: Could not read .gitignore: {e}")
        
        return patterns
    
    def _is_in_dependency_dir(self, file_path: Path) -> bool:
        """Check if file is inside a dependency directory."""
        try:
            rel_path = file_path.relative_to(self.root_path)
            parts = rel_path.parts
            return any(dep_dir in parts for dep_dir in self.DEPENDENCY_DIRS)
        except ValueError:
            return False
    
    def _is_dependency_doc(self, file_path: Path) -> bool:
        """Check if file is a documentation file in dependencies."""
        filename = file_path.name
        
        # Check if it's a known doc file
        if filename in self.DEPENDENCY_DOCS:
            return True
        
        # Check if it's in a docs directory
        try:
            rel_path = file_path.relative_to(self.root_path)
            parts = rel_path.parts
            if 'docs' in parts or 'documentation' in parts:
                return True
        except ValueError:
            pass
        
        return False
    
    def should_ignore(self, file_path: Path) -> bool:
        """
        Check if a file should be ignored based on patterns.
        
        Args:
            file_path: Path to check
        
        Returns:
            True if file should be ignored
        """
        # Special handling for dependency directories
        if self._is_in_dependency_dir(file_path):
            if self.index_dependency_docs and self._is_dependency_doc(file_path):
                # Keep documentation from dependencies
                return False
            else:
                # Skip other files in dependencies
                return True
        
        try:
            # Get relative path from root
            rel_path = file_path.relative_to(self.root_path)
            path_str = str(rel_path)
            
            # Check each part of the path
            parts = rel_path.parts
            
            for pattern in self.patterns:
                # Directory pattern (ends with /)
                if pattern.endswith('/'):
                    dir_pattern = pattern[:-1]
                    if dir_pattern in parts:
                        return True
                
                # File pattern
                if fnmatch.fnmatch(file_path.name, pattern):
                    return True
                
                # Full path pattern
                if fnmatch.fnmatch(path_str, pattern):
                    return True
            
            return False
            
        except ValueError:
            # File is not relative to root
            return False
    
    def filter_files(self, files: List[Path]) -> List[Path]:
        """
        Filter a list of files, removing ignored ones.
        
        Args:
            files: List of file paths
        
        Returns:
            Filtered list of files
        """
        return [f for f in files if not self.should_ignore(f)]