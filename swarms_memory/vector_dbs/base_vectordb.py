from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from loguru import logger


class BaseVectorDatabase(ABC):
    """
    Abstract base class for vector databases.

    This class defines the standardized interface for all vector database implementations
    in the swarms-memory package. It provides common functionality and enforces
    consistent method signatures across different vector database providers.
    
    All vector database wrappers should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self):
        """Initialize the base vector database."""
        self.logger = logger
    
    @abstractmethod
    def add(self, doc: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the vector database.
        
        Args:
            doc (str): The document text to add
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document
            
        Returns:
            str: The unique identifier for the added document
        """
        pass
    
    @abstractmethod
    def query(
        self, 
        query_text: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar documents.
        
        Args:
            query_text (str): The query text to search for
            top_k (int): Number of results to return (default: 5)
            filter_dict (Optional[Dict[str, Any]]): Metadata filters to apply
            
        Returns:
            List[Dict[str, Any]]: List of matching documents with metadata and scores
        """
        pass
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            doc_id (str): The unique identifier of the document
            
        Returns:
            Optional[Dict[str, Any]]: The document data if found, None otherwise
        """
        # Default implementation - can be overridden by subclasses
        self.logger.warning(f"{self.__class__.__name__}.get() not implemented")
        return None
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id (str): The unique identifier of the document to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        self.logger.warning(f"{self.__class__.__name__}.delete() not implemented")
        return False
    
    def clear(self) -> bool:
        """
        Clear all documents from the vector database.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        # Default implementation - can be overridden by subclasses
        self.logger.warning(f"{self.__class__.__name__}.clear() not implemented")
        return False
    
    def count(self) -> int:
        """
        Get the number of documents in the vector database.
        
        Returns:
            int: Number of documents, -1 if not supported
        """
        # Default implementation - can be overridden by subclasses
        self.logger.warning(f"{self.__class__.__name__}.count() not implemented")
        return -1
    
    # Optional methods that can be implemented by subclasses
    def update(self, doc_id: str, doc: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id (str): The unique identifier of the document to update
            doc (str): The new document content
            metadata (Optional[Dict[str, Any]]): Updated metadata
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Default implementation: delete and re-add
        if self.delete(doc_id):
            new_id = self.add(doc, metadata)
            return new_id is not None
        return False
    
    # Utility methods for consistent behavior
    def log_query(self, query: str) -> None:
        """
        Log a query operation.
        
        Args:
            query (str): The query text that was executed
        """
        self.logger.info(f"Query executed: {query[:100]}..." if len(query) > 100 else f"Query executed: {query}")
    
    def log_add_operation(self, doc: str, doc_id: str) -> None:
        """
        Log an add operation.
        
        Args:
            doc (str): The document that was added
            doc_id (str): The ID assigned to the document
        """
        doc_preview = doc[:50] + "..." if len(doc) > 50 else doc
        self.logger.info(f"Document added with ID {doc_id}: {doc_preview}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector database.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        return {
            "status": "unknown",
            "message": f"{self.__class__.__name__}.health_check() not implemented",
            "timestamp": None
        }
    
    # Legacy method compatibility - with deprecation warnings
    def connect(self) -> None:
        """
        Legacy method for backward compatibility.
        Modern vector databases handle connections automatically.
        """
        self.logger.warning(
            f"{self.__class__.__name__}.connect() is deprecated and will be removed. "
            "Modern vector databases handle connections automatically during initialization."
        )
    
    def close(self) -> None:
        """
        Legacy method for backward compatibility.
        Modern vector databases handle connections automatically.
        """
        self.logger.warning(
            f"{self.__class__.__name__}.close() is deprecated and will be removed. "
            "Modern vector databases handle cleanup automatically."
        )