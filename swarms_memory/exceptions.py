"""
Custom exceptions for swarms-memory package.

This module provides standardized exception types for consistent error handling
across all swarms-memory components.
"""

from typing import Optional, Any


class SwarmsMemoryError(Exception):
    """Base exception for all swarms-memory related errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class VectorDatabaseError(SwarmsMemoryError):
    """Raised when vector database operations fail."""
    pass


class EmbeddingError(SwarmsMemoryError):
    """Raised when embedding operations fail."""
    pass


class ConfigurationError(SwarmsMemoryError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(SwarmsMemoryError):
    """Raised when input validation fails."""
    pass


class DatabaseConnectionError(SwarmsMemoryError):
    """Raised when database connection fails."""
    pass


class QueryError(SwarmsMemoryError):
    """Raised when query operations fail."""
    pass


class DocumentProcessingError(SwarmsMemoryError):
    """Raised when document processing operations fail."""
    pass


def handle_error(
    error: Exception,
    operation: str,
    component: str,
    reraise_as: type = SwarmsMemoryError
) -> None:
    """
    Standardized error handling function.
    
    Args:
        error: The original exception
        operation: Description of the operation that failed
        component: Name of the component where error occurred
        reraise_as: Exception type to reraise as
        
    Raises:
        reraise_as: The standardized exception type
    """
    error_msg = f"{component} failed during {operation}: {str(error)}"
    raise reraise_as(error_msg, error)