"""
Utility functions for unified embedding model handling across vector databases.
"""

from typing import Union, Callable, List, Any, Optional
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning("SentenceTransformers not installed. Some embedding features may not work.")

try:
    from swarms_memory.embeddings.litellm_embeddings import LiteLLMEmbeddings
except ImportError:
    logger.error("LiteLLMEmbeddings not found. LiteLLM functionality will be disabled.")
    LiteLLMEmbeddings = None


def detect_litellm_model(model_name: str) -> bool:
    """
    Detect if a model name corresponds to a LiteLLM provider model.
    
    Args:
        model_name: The model name to check
        
    Returns:
        True if the model is a LiteLLM provider model, False otherwise
    """
    if not isinstance(model_name, str):
        return False
    
    model_lower = model_name.lower()
    
    # Common SentenceTransformer patterns - these should use SentenceTransformers
    sentence_transformer_patterns = [
        "all-minilm", "all-mpnet", "multi-qa", "distiluse", "paraphrase-",
        "sentence-t5", "gtr-t5", "sup-simcse"
    ]
    
    # Check if it's a SentenceTransformer model
    for pattern in sentence_transformer_patterns:
        if pattern in model_lower:
            return False  # Use SentenceTransformers instead of LiteLLM
    
    # LiteLLM provider prefixes and patterns
    litellm_patterns = [
        # Provider prefixes
        "openai/", "azure/", "cohere/", "voyage/", "bedrock/", "nvidia/", "huggingface/",
        # OpenAI model names
        "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002",
        # Common patterns
        "text-similarity-", "text-search-", "embed-",
    ]
    
    # Check for exact matches or prefix matches
    for pattern in litellm_patterns:
        if pattern.endswith("/"):
            # Provider prefix check
            if model_lower.startswith(pattern):
                return True
        else:
            # Exact match or contains check
            if model_lower == pattern or pattern in model_lower:
                return True
    
    # Additional checks for common provider patterns
    provider_indicators = ["gpt-", "claude-", "gemini-", "palm-"]
    if any(indicator in model_lower for indicator in provider_indicators):
        return True
    
    return False


def setup_unified_embedding(
    embedding_model: Union[str, Any, Callable],
    embedding_function: Optional[Callable] = None,
    dimension: Optional[int] = None,
    **embedding_kwargs
) -> tuple[Any, Callable, int]:
    """
    Set up unified embedding model handling for vector databases.
    
    Args:
        embedding_model: The embedding model specification:
            - str: LiteLLM model name or SentenceTransformer model name
            - LiteLLMEmbeddings: Pre-configured LiteLLM instance
            - Callable: Custom embedding function
        embedding_function: Legacy embedding function parameter (for backward compatibility)
        dimension: Expected embedding dimension
        **embedding_kwargs: Additional arguments for embedding model setup
        
    Returns:
        Tuple of (embedder_instance, embedding_function, dimension)
    """
    
    # Handle legacy embedding_function parameter first (backward compatibility)
    if embedding_function is not None and callable(embedding_function):
        logger.info("Using legacy embedding_function parameter")
        return None, embedding_function, dimension
    
    # Handle different embedding_model types
    if isinstance(embedding_model, str):
        return _setup_string_embedding_model(embedding_model, dimension, **embedding_kwargs)
    
    elif LiteLLMEmbeddings and isinstance(embedding_model, LiteLLMEmbeddings):
        logger.info("Using provided LiteLLMEmbeddings instance")
        detected_dim = embedding_model.get_embedding_dimension()
        final_dimension = dimension or detected_dim
        return embedding_model, None, final_dimension
    
    elif callable(embedding_model):
        logger.info("Using custom embedding function from embedding_model parameter")
        return None, embedding_model, dimension
    
    else:
        raise ValueError(
            f"Unsupported embedding_model type: {type(embedding_model)}. "
            "Must be a string (model name), LiteLLMEmbeddings instance, or callable function."
        )


def _setup_string_embedding_model(
    model_name: str, 
    dimension: Optional[int] = None, 
    **embedding_kwargs
) -> tuple[Any, Callable, int]:
    """Set up embedding model from string name."""
    
    if detect_litellm_model(model_name):
        # Use LiteLLM
        if LiteLLMEmbeddings is None:
            raise ImportError(
                f"LiteLLM model '{model_name}' detected but LiteLLMEmbeddings not available. "
                "Please install litellm: pip install litellm"
            )
        
        logger.info(f"Using LiteLLM model: {model_name}")
        embedder = LiteLLMEmbeddings(model=model_name, **embedding_kwargs)
        detected_dim = embedder.get_embedding_dimension()
        final_dimension = dimension or detected_dim
        
        return embedder, None, final_dimension
    
    else:
        # Use SentenceTransformer
        if SentenceTransformer is None:
            raise ImportError(
                f"SentenceTransformer model '{model_name}' detected but sentence-transformers not available. "
                "Please install sentence-transformers: pip install sentence-transformers"
            )
        
        logger.info(f"Using SentenceTransformer model: {model_name}")
        embedder = SentenceTransformer(model_name)
        
        # Get dimension from SentenceTransformer
        if hasattr(embedder, 'get_sentence_embedding_dimension'):
            detected_dim = embedder.get_sentence_embedding_dimension()
        elif hasattr(embedder, 'encode'):
            # Fallback: encode a sample to get dimension
            try:
                sample_embedding = embedder.encode("sample text")
                detected_dim = len(sample_embedding)
            except Exception as e:
                logger.warning(f"Could not detect SentenceTransformer dimension: {e}")
                detected_dim = None
        else:
            detected_dim = None
        
        final_dimension = dimension or detected_dim
        return embedder, None, final_dimension


def get_embedding_function(embedder: Any, embedding_function: Optional[Callable] = None) -> Callable:
    """
    Get the appropriate embedding function for the configured embedder.
    
    Args:
        embedder: The embedding model instance (LiteLLMEmbeddings, SentenceTransformer, etc.)
        embedding_function: Custom embedding function if provided
        
    Returns:
        Callable that takes text and returns embedding vector
    """
    
    if embedding_function is not None:
        return embedding_function
    
    if LiteLLMEmbeddings and isinstance(embedder, LiteLLMEmbeddings):
        return embedder.embed_query
    
    elif SentenceTransformer and isinstance(embedder, SentenceTransformer):
        return embedder.encode
    
    else:
        raise ValueError(f"No embedding function available for embedder type: {type(embedder)}")


def get_batch_embedding_function(embedder: Any, embedding_function: Optional[Callable] = None) -> Optional[Callable]:
    """
    Get the batch embedding function if available.
    
    Args:
        embedder: The embedding model instance
        embedding_function: Custom embedding function if provided
        
    Returns:
        Callable that takes list of texts and returns list of embedding vectors, or None
    """
    
    if embedding_function is not None:
        # Custom functions don't have batch support by default
        return None
    
    if LiteLLMEmbeddings and isinstance(embedder, LiteLLMEmbeddings):
        return embedder.embed_documents
    
    elif SentenceTransformer and isinstance(embedder, SentenceTransformer):
        return embedder.encode
    
    return None