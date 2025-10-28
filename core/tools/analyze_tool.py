"""
AnalyzeTool for agent to understand and analyze code.
"""
from typing import Dict, Any, List, Optional
import re
from core.tools.base_tool import BaseTool, ToolResult
from storage.db import Database


class AnalyzeTool(BaseTool):
    """
    Tool for analyzing code and extracting patterns.

    Capabilities:
    - Extract functions and classes
    - Find patterns in code
    - Analyze dependencies
    - Understand code structure
    """

    def __init__(self):
        """Initialize AnalyzeTool."""
        super().__init__(
            name='analyze',
            description='''Analyze code to extract structure and patterns.
Use this to understand how code works, find functions/classes, etc.
Parameters:
  - file_path (str, required): Path to file to analyze
  - analysis_type (str, optional): 'structure', 'functions', 'classes', 'imports' (default: 'structure')
Returns: Analysis results with code elements found'''
        )
        self.db = Database()

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute code analysis.

        Args:
            file_path (str): Path to file to analyze
            analysis_type (str): Type of analysis to perform

        Returns:
            ToolResult with analysis
        """
        # Validate required params
        if not self.validate_params(['file_path'], kwargs):
            return ToolResult(
                success=False,
                data=None,
                message='Missing required parameter: file_path'
            )

        file_path = kwargs['file_path']
        analysis_type = kwargs.get('analysis_type', 'structure')

        try:
            self.logger.debug(f"AnalyzeTool: Analyzing {file_path} ({analysis_type})")

            # Get file content
            content = self.db.get_document_content(file_path)
            if not content:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f'Could not read file: {file_path}'
                )

            # Perform requested analysis
            if analysis_type == 'functions':
                analysis = self._analyze_functions(content, file_path)
            elif analysis_type == 'classes':
                analysis = self._analyze_classes(content, file_path)
            elif analysis_type == 'imports':
                analysis = self._analyze_imports(content, file_path)
            else:  # structure
                analysis = self._analyze_structure(content, file_path)

            result = ToolResult(
                success=True,
                data=analysis,
                message=f'Analyzed {file_path}',
                metadata={'analysis_type': analysis_type, 'file_path': file_path}
            )

            self.log_execution(kwargs, result)
            return result

        except Exception as e:
            self.logger.error(f"AnalyzeTool error: {e}")
            return ToolResult(
                success=False,
                data=None,
                message=f'Analysis error: {str(e)}'
            )

    def _analyze_structure(self, content: str, file_path: str) -> Dict:
        """Analyze overall code structure."""
        return {
            'functions': self._extract_functions(content),
            'classes': self._extract_classes(content),
            'imports': self._extract_imports(content),
            'lines_of_code': len(content.split('\n')),
            'file_size': len(content),
        }

    def _analyze_functions(self, content: str, file_path: str) -> Dict:
        """Extract and analyze functions."""
        functions = self._extract_functions(content)
        return {
            'count': len(functions),
            'functions': functions,
            'public_functions': [f for f in functions if not f['name'].startswith('_')],
        }

    def _analyze_classes(self, content: str, file_path: str) -> Dict:
        """Extract and analyze classes."""
        classes = self._extract_classes(content)
        return {
            'count': len(classes),
            'classes': classes,
        }

    def _analyze_imports(self, content: str, file_path: str) -> Dict:
        """Extract and analyze imports."""
        imports = self._extract_imports(content)
        return {
            'count': len(imports),
            'imports': imports,
            'external': [i for i in imports if not i.startswith('.')],
            'internal': [i for i in imports if i.startswith('.')],
        }

    def _extract_functions(self, content: str) -> List[Dict]:
        """Extract function definitions."""
        functions = []

        # Python function pattern
        pattern = r'^\s*def\s+(\w+)\s*\((.*?)\).*?:'
        lines = content.split('\n')

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                name = match.group(1)
                params = match.group(2)

                # Get docstring if exists
                docstring = ''
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        docstring = next_line.strip('"\' ')

                functions.append({
                    'name': name,
                    'params': [p.strip() for p in params.split(',') if p.strip()],
                    'line': i + 1,
                    'docstring': docstring,
                })

        return functions

    def _extract_classes(self, content: str) -> List[Dict]:
        """Extract class definitions."""
        classes = []

        # Python class pattern
        pattern = r'^\s*class\s+(\w+)\s*(?:\((.*?)\))?.*?:'
        lines = content.split('\n')

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                name = match.group(1)
                base = match.group(2) or ''

                # Get docstring if exists
                docstring = ''
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        docstring = next_line.strip('"\' ')

                classes.append({
                    'name': name,
                    'base': base,
                    'line': i + 1,
                    'docstring': docstring,
                })

        return classes

    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements."""
        imports = []

        # Pattern for various import styles
        patterns = [
            r'^\s*import\s+([\w.]+)',  # import x
            r'^\s*from\s+([\w.]+)\s+import',  # from x import
        ]

        for line in content.split('\n'):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    imports.append(match.group(1))

        return list(set(imports))  # Remove duplicates
