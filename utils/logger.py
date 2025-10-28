"""
Logging module for Hyperthymesia.

Handles debug logging to file while maintaining clean CLI output.
Provides both file logging and optional console debug output.
"""

import logging
import logging.handlers
from pathlib import Path
import platformdirs
import os
from datetime import datetime


# Get platform-specific data directory
DATA_DIR = Path(platformdirs.user_data_dir("hyperthymesia"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Log file path
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"hyperthymesia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class HyperthymesiaLogger:
    """
    Centralized logger for Hyperthymesia.

    Features:
    - File logging with rotation
    - Structured logging with modules and levels
    - Optional console debug output
    - Performance tracking
    """

    def __init__(self, name: str = "hyperthymesia", debug: bool = False):
        """
        Initialize logger.

        Args:
            name: Logger name (usually __name__)
            debug: Enable console debug output
        """
        self.name = name
        self.debug_mode = debug
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        self.logger.handlers = []

        # File handler - always log everything
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler - only if debug mode enabled
        if debug:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter(
                '[%(levelname)s] %(name)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)

    def log_search(self, query: str, num_results: int, took_seconds: float):
        """Log search operation."""
        self.logger.info(
            f"Search executed",
            extra={
                'query': query,
                'results': num_results,
                'duration_sec': round(took_seconds, 3)
            }
        )

    def log_rag_retrieval(self, question: str, chunks_found: int, sources: int, took_seconds: float):
        """Log RAG retrieval operation."""
        self.logger.info(
            f"RAG retrieval executed",
            extra={
                'question': question,
                'chunks': chunks_found,
                'sources': sources,
                'duration_sec': round(took_seconds, 3)
            }
        )

    def log_indexing(self, path: str, files_indexed: int, failed: int, took_seconds: float):
        """Log indexing operation."""
        self.logger.info(
            f"Indexing completed",
            extra={
                'path': path,
                'indexed': files_indexed,
                'failed': failed,
                'duration_sec': round(took_seconds, 3)
            }
        )

    def get_log_file_path(self) -> Path:
        """Get current log file path."""
        return LOG_FILE

    def get_log_directory(self) -> Path:
        """Get log directory path."""
        return LOG_DIR

    @staticmethod
    def get_recent_logs(max_files: int = 10) -> list:
        """
        Get list of recent log files.

        Args:
            max_files: Maximum number of recent logs to return

        Returns:
            List of log file paths, sorted by newest first
        """
        log_files = sorted(LOG_DIR.glob("*.log"), key=os.path.getmtime, reverse=True)
        return log_files[:max_files]


# Global logger instances (one per module)
_loggers = {}


def get_logger(name: str, debug: bool = False) -> HyperthymesiaLogger:
    """
    Get or create a logger for a module.

    Args:
        name: Module name (usually __name__)
        debug: Enable console debug output

    Returns:
        HyperthymesiaLogger instance
    """
    if name not in _loggers:
        _loggers[name] = HyperthymesiaLogger(name, debug=debug)
    return _loggers[name]


def enable_debug_logging():
    """Enable debug output to console for all loggers."""
    for logger in _loggers.values():
        logger.debug_mode = True
        # Add console handler if not already present
        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in logger.logger.handlers
        )
        if not has_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter('[%(levelname)s] %(name)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.logger.addHandler(console_handler)


def disable_debug_logging():
    """Disable debug output to console for all loggers."""
    for logger in _loggers.values():
        logger.debug_mode = False
        # Remove console handlers
        logger.logger.handlers = [
            h for h in logger.logger.handlers
            if not isinstance(h, logging.StreamHandler) or isinstance(h, logging.FileHandler)
        ]


# Convenience function for quick debugging
def log_debug(module_name: str, message: str):
    """Quick debug logging without creating logger instance."""
    logger = get_logger(module_name)
    logger.debug(message)
