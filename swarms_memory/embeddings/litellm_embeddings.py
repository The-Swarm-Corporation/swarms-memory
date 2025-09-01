import os
from typing import List, Optional, Dict, Any, Union
from loguru import logger
from swarms_memory.exceptions import EmbeddingError, ValidationError, handle_error

try:
    from litellm import embedding
except ImportError:
    raise ImportError(
        "litellm is required for LiteLLMEmbeddings. Install with: pip install litellm"
    )


class LiteLLMEmbeddings:
    """
    A flexible embeddings wrapper that uses LiteLLM to support multiple embedding providers.
    
    This class provides a unified interface for generating embeddings using various
    providers like OpenAI, Azure, Cohere, Voyage AI, AWS Bedrock, and more through LiteLLM.
    
    Attributes:
        model (str): The embedding model name with optional provider prefix 
            (e.g., "text-embedding-3-small", "azure/my-deployment", "cohere/embed-english-v3.0")
        api_key (Optional[str]): API key for the provider. Can also be set via environment variables.
        api_base (Optional[str]): API base URL for providers like Azure.
        api_version (Optional[str]): API version for providers like Azure.
        dimensions (Optional[int]): Output dimension for models that support it (e.g., text-embedding-3).
        input_type (Optional[str]): Input type for dual-mode models ("query" or "passage").
        metadata (Optional[Dict[str, Any]]): Additional metadata to pass with requests.
        **kwargs: Additional provider-specific parameters.
    
    Examples:
        >>> # OpenAI embeddings
        >>> embedder = LiteLLMEmbeddings(model="text-embedding-3-small")
        >>> vectors = embedder.embed_documents(["Hello world", "Test document"])
        
        >>> # Azure OpenAI embeddings  
        >>> embedder = LiteLLMEmbeddings(
        ...     model="azure/my-embedding-deployment",
        ...     api_key="your-api-key",
        ...     api_base="https://your-resource.openai.azure.com",
        ...     api_version="2023-07-01-preview"
        ... )
        
        >>> # Cohere embeddings with input type
        >>> embedder = LiteLLMEmbeddings(
        ...     model="cohere/embed-english-v3.0",
        ...     input_type="search_document"
        ... )
        
        >>> # AWS Bedrock embeddings
        >>> embedder = LiteLLMEmbeddings(
        ...     model="bedrock/amazon.titan-embed-text-v1"
        ... )
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        dimensions: Optional[int] = None,
        input_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the LiteLLM embeddings wrapper.
        
        Args:
            model: The embedding model name with optional provider prefix
            api_key: API key for the provider (optional, can use env vars)
            api_base: API base URL for providers like Azure
            api_version: API version for providers like Azure
            dimensions: Output dimension for models that support it
            input_type: Input type for dual-mode models ("query" or "passage")
            metadata: Additional metadata to pass with requests
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.dimensions = dimensions
        self.input_type = input_type
        self.metadata = metadata or {}
        self.kwargs = kwargs
        
        # Safely log model name without exposing potential API keys
        safe_model_name = self._sanitize_model_for_logging(model)
        logger.info(f"Initialized LiteLLMEmbeddings with model: {safe_model_name}")
    
    def _sanitize_model_for_logging(self, model_name: str) -> str:
        """Sanitize model names for safe logging without exposing API keys"""
        if "/" in model_name:
            parts = model_name.split("/")
            # If the part after "/" is suspiciously long (like an API key), mask it
            if len(parts) > 1 and len(parts[-1]) > 20:
                return f"{parts[0]}/***{parts[-1][-4:]}"
        return model_name
    
    def _prepare_params(self) -> Dict[str, Any]:
        """Prepare parameters for the embedding call."""
        params = {
            "model": self.model,
            "metadata": self.metadata,
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_version:
            params["api_version"] = self.api_version
        if self.dimensions:
            params["dimensions"] = self.dimensions
        if self.input_type:
            params["input_type"] = self.input_type
            
        # Add any additional kwargs
        params.update(self.kwargs)
        
        return params
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
            
        Raises:
            Exception: If the embedding generation fails
        """
        # Input validation
        if not texts or not isinstance(texts, list):
            raise ValidationError("Texts must be a non-empty list")
        
        if not all(isinstance(text, str) for text in texts):
            raise ValidationError("All texts must be strings")
            
        if any(len(text.strip()) == 0 for text in texts):
            raise ValidationError("All texts must contain non-whitespace content")
        
        try:
            params = self._prepare_params()
            params["input"] = texts
            
            response = embedding(**params)
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in response.data]
            
            logger.debug(f"Generated {len(embeddings)} embeddings for documents")
            return embeddings
            
        except ValidationError:
            raise
        except Exception as e:
            handle_error(e, "document embedding generation", "LiteLLMEmbeddings", EmbeddingError)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        For models that support dual modes (query vs document), this method
        can automatically set the input_type to "query" if not already specified.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
            
        Raises:
            Exception: If the embedding generation fails
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValidationError("Text must be a non-empty string")
        
        if len(text.strip()) == 0:
            raise ValidationError("Text cannot be just whitespace")
            
        try:
            params = self._prepare_params()
            params["input"] = [text]
            
            # For query embedding, we might want to set input_type to "query"
            # if the model supports it and it's not already set
            if not self.input_type and self._supports_input_type():
                params["input_type"] = "query"
            
            response = embedding(**params)
            
            # Extract the first (and only) embedding
            query_embedding = response.data[0]["embedding"]
            
            logger.debug(f"Generated query embedding with dimension {len(query_embedding)}")
            return query_embedding
            
        except ValidationError:
            raise
        except Exception as e:
            handle_error(e, "query embedding generation", "LiteLLMEmbeddings", EmbeddingError)
    
    def _supports_input_type(self) -> bool:
        """
        Check if the model supports input_type parameter.
        
        Returns:
            True if the model supports input_type, False otherwise
        """
        # Models known to support input_type
        dual_mode_models = [
            "nvidia", "e5", "cohere", "voyage"
        ]
        
        model_lower = self.model.lower()
        return any(provider in model_lower for provider in dual_mode_models)
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of the embeddings produced by the model.
        
        This method attempts to determine the embedding dimension by generating
        a sample embedding. If dimensions parameter is set, returns that instead.
        
        Returns:
            The embedding dimension, or None if it cannot be determined
        """
        if self.dimensions:
            return self.dimensions
        
        try:
            # Generate a sample embedding to determine dimension
            sample_embedding = self.embed_query("sample text")
            dimension = len(sample_embedding)
            logger.info(f"Detected embedding dimension: {dimension}")
            return dimension
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {str(e)}")
            return None
    
    def batch_embed_documents(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for documents in batches.
        
        Useful for processing large numbers of documents to avoid rate limits
        or memory issues.
        
        Args:
            texts: List of text documents to embed
            batch_size: Number of documents to process in each batch
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Processed batch {i//batch_size + 1}, total embeddings: {len(all_embeddings)}")
        
        return all_embeddings