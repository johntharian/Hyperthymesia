"""
Base class for all agent tools.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    data: Any
    message: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'data': self.data,
            'message': self.message,
            'metadata': self.metadata,
        }


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Tools are reusable components that agents can invoke to:
    - Search for information
    - Analyze code
    - Synthesize answers
    - Retrieve context
    """

    def __init__(self, name: str, description: str):
        """
        Initialize a tool.

        Args:
            name: Tool name (e.g., 'search', 'analyze')
            description: Tool description for agent reasoning
        """
        self.name = name
        self.description = description
        self.logger = logger

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with success status and data
        """
        pass

    def validate_params(self, required_params: List[str], provided_params: Dict) -> bool:
        """
        Validate that all required parameters are provided.

        Args:
            required_params: List of required parameter names
            provided_params: Dictionary of provided parameters

        Returns:
            True if all required params present, False otherwise
        """
        missing = set(required_params) - set(provided_params.keys())
        if missing:
            self.logger.warning(f"Missing parameters for {self.name}: {missing}")
            return False
        return True

    def get_info(self) -> Dict[str, Any]:
        """
        Get tool information for agent reasoning.

        Returns:
            Dictionary with tool metadata
        """
        return {
            'name': self.name,
            'description': self.description,
            'type': self.__class__.__name__,
        }

    def log_execution(self, params: Dict, result: ToolResult):
        """Log tool execution for debugging."""
        self.logger.debug(f"Tool '{self.name}' executed")
        self.logger.debug(f"  Params: {params}")
        self.logger.debug(f"  Result: success={result.success}, message={result.message}")
