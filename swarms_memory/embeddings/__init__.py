"""
Embeddings module for swarms-memory.

This module provides flexible embedding functionality through various providers
using LiteLLM and custom embedding functions.
"""

from .litellm_embeddings import LiteLLMEmbeddings
from .embedding_utils import (
    detect_litellm_model,
    setup_unified_embedding,
    get_embedding_function,
    get_batch_embedding_function,
)

__all__ = [
    "LiteLLMEmbeddings",
    "detect_litellm_model", 
    "setup_unified_embedding",
    "get_embedding_function",
    "get_batch_embedding_function",
]