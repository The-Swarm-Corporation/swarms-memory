import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from pinecone import Pinecone, ServerlessSpec
from loguru import logger
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase
from swarms_memory.embeddings.litellm_embeddings import LiteLLMEmbeddings


class PineconeMemory(BaseVectorDatabase):
    """
    A highly customizable wrapper class for Pinecone-based vector database with modern embedding support.

    This class provides methods to add documents to the Pinecone index and query the index
    for similar documents. It supports any embedding model through LiteLLM or custom embedding functions,
    providing maximum flexibility for text vectorization.
    
    Args:
        api_key (str): Pinecone API key.
        index_name (str): Name of the Pinecone index to use.
        embedding_model (Union[str, LiteLLMEmbeddings, Callable]): 
            The embedding model or function to use. Can be:
            - A string model name for LiteLLM (e.g., "text-embedding-3-small", 
              "azure/my-deployment", "cohere/embed-english-v3.0")
            - A LiteLLMEmbeddings instance with custom configuration
            - A custom embedding function that takes text and returns a vector
        dimension (int): Dimension of the document embeddings. Defaults to 1536.
        metric (str): Distance metric for Pinecone index. Defaults to 'cosine'.
        cloud (str): Cloud provider for serverless index. Defaults to 'aws'.
        region (str): Region for serverless index. Defaults to 'us-east-1'.
        namespace (str): Pinecone namespace. Defaults to ''.
        api_key_embedding (Optional[str]): API key for the embedding provider.
        api_base (Optional[str]): API base URL for providers like Azure.
        api_version (Optional[str]): API version for providers like Azure.
        embedding_kwargs (Optional[Dict]): Additional kwargs for embedding model.
        logger_config (Optional[Dict[str, Any]]): Configuration for the logger. Defaults to None.

    Examples:
        >>> # Example 1: Using OpenAI embeddings
        >>> pinecone_db = PineconeMemory(
        >>>     api_key="your-pinecone-key",
        >>>     index_name="my-index",
        >>>     embedding_model="text-embedding-3-small",
        >>>     dimension=1536
        >>> )
        >>>
        >>> # Example 2: Using Azure OpenAI embeddings
        >>> pinecone_db = PineconeMemory(
        >>>     api_key="your-pinecone-key",
        >>>     index_name="my-index", 
        >>>     embedding_model="azure/my-embedding-deployment",
        >>>     api_key_embedding="your-azure-key",
        >>>     api_base="https://your-resource.openai.azure.com",
        >>>     api_version="2023-07-01-preview",
        >>>     dimension=1536
        >>> )
        >>>
        >>> # Example 3: Using Cohere embeddings
        >>> pinecone_db = PineconeMemory(
        >>>     api_key="your-pinecone-key",
        >>>     index_name="my-index",
        >>>     embedding_model="cohere/embed-english-v3.0",
        >>>     api_key_embedding="your-cohere-key",
        >>>     dimension=1024,
        >>>     embedding_kwargs={"input_type": "search_document"}
        >>> )
        >>>
        >>> # Example 4: Using custom embedding function
        >>> def my_embedding_func(text: str) -> List[float]:
        >>>     # Your custom embedding logic here
        >>>     return [0.1] * 1536
        >>> 
        >>> pinecone_db = PineconeMemory(
        >>>     api_key="your-pinecone-key",
        >>>     index_name="my-index",
        >>>     embedding_model=my_embedding_func,
        >>>     dimension=1536
        >>> )
        >>>
        >>> # Add documents
        >>> doc_id = pinecone_db.add("This is a sample document for testing")
        >>>
        >>> # Query similar documents
        >>> results = pinecone_db.query("sample query", top_k=5)
        >>> print(results)
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding_model: Union[str, LiteLLMEmbeddings, Callable],
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws", 
        region: str = "us-east-1",
        namespace: str = "",
        api_key_embedding: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        logger_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PineconeMemory with modern embedding support.
        """
        super().__init__()
        self._setup_logger(logger_config)
        logger.info("Initializing PineconeMemory with modern embedding support")

        # Store configuration
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)

        # Setup embedding model
        self._setup_embedding_model(
            embedding_model,
            api_key_embedding,
            api_base,
            api_version, 
            embedding_kwargs
        )

        # Create or get index
        self._setup_index()

        logger.info("PineconeMemory initialized successfully")

    def _setup_logger(self, config: Optional[Dict[str, Any]] = None):
        """Set up the logger with the given configuration."""
        if config:
            logger.configure(**config)
        else:
            # Use default logger configuration
            pass

    def _setup_embedding_model(
        self,
        embedding_model: Union[str, LiteLLMEmbeddings, Callable],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Setup the embedding model based on the provided configuration."""
        
        if isinstance(embedding_model, str):
            # Use LiteLLM for the model
            kwargs = embedding_kwargs or {}
            self.embedder = LiteLLMEmbeddings(
                model=embedding_model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                **kwargs
            )
            self.embedding_func = None
            self.model_name = embedding_model
            logger.info(f"Using LiteLLM model: {embedding_model}")
            
            # Auto-detect embedding dimension if not provided
            try:
                detected_dim = self.embedder.get_embedding_dimension()
                if detected_dim and detected_dim != self.dimension:
                    logger.warning(
                        f"Detected embedding dimension ({detected_dim}) differs from "
                        f"specified dimension ({self.dimension}). Using detected dimension."
                    )
                    self.dimension = detected_dim
            except Exception as e:
                logger.warning(f"Could not auto-detect embedding dimension: {e}")
                
        elif isinstance(embedding_model, LiteLLMEmbeddings):
            # Direct LiteLLMEmbeddings instance
            self.embedder = embedding_model
            self.embedding_func = None
            self.model_name = getattr(embedding_model, 'model', 'custom_litellm')
            logger.info("Using provided LiteLLMEmbeddings instance")
            
            # Auto-detect embedding dimension
            try:
                detected_dim = self.embedder.get_embedding_dimension()
                if detected_dim and detected_dim != self.dimension:
                    logger.warning(
                        f"Detected embedding dimension ({detected_dim}) differs from "
                        f"specified dimension ({self.dimension}). Using detected dimension."
                    )
                    self.dimension = detected_dim
            except Exception as e:
                logger.warning(f"Could not auto-detect embedding dimension: {e}")
                
        elif callable(embedding_model):
            # Custom embedding function
            self.embedder = None
            self.embedding_func = embedding_model
            self.model_name = 'custom_function'
            logger.info("Using custom embedding function")
        else:
            raise ValueError(
                "embedding_model must be a string (LiteLLM model name), "
                "a LiteLLMEmbeddings instance, or a callable function"
            )

    def _setup_index(self):
        """Create or get the Pinecone index."""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")

            # Get index handle
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone index: {str(e)}")
            raise Exception(f"Failed to setup Pinecone index: {str(e)}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using the configured embedding method."""
        try:
            if self.embedder:
                # Use LiteLLM embedder
                return self.embedder.embed_query(text)
            elif self.embedding_func:
                # Use custom embedding function
                return self.embedding_func(text)
            else:
                raise ValueError("No embedding method configured")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def add(self, doc: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the Pinecone index.

        Args:
            doc (str): The document to be added.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document.

        Returns:
            str: The unique identifier for the added document.
        """
        try:
            logger.info(f"Adding document: {doc[:50]}...")
            
            # Generate embedding
            embedding = self._get_embedding(doc)
            
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata = metadata or {}
            metadata["text"] = doc
            metadata["model"] = self.model_name
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=[(doc_id, embedding, metadata)],
                namespace=self.namespace,
            )
            
            logger.success(f"Document added successfully with ID: {doc_id}")
            self.log_add_operation(doc, doc_id)
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise Exception(f"Failed to add document: {str(e)}")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ):
        """
        Query the Pinecone index for similar documents.

        Args:
            query_text (str): The query string.
            top_k (int): The number of top results to return. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]]): Metadata filter for the query.
            return_metadata (bool): If True, return full metadata. If False, return concatenated text.

        Returns:
            Union[List[Dict[str, Any]], str]: Formatted results based on return_metadata flag.
        """
        try:
            logger.info(f"Querying with: {query_text}")
            self.log_query(query_text)
            
            # Generate query embedding
            query_embedding = self._get_embedding(query_text)
            
            # Perform search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict,
            )

            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append(
                    {
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata,
                    }
                )

            logger.success(f"Query completed. Found {len(formatted_results)} results.")
            
            if return_metadata:
                return formatted_results
            else:
                # Return concatenated text for Swarms agent compatibility
                return "\n\n".join(
                    result["metadata"].get("text", str(result["metadata"]))
                    for result in formatted_results
                )
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise Exception(f"Query failed: {str(e)}")

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            doc_id (str): The unique identifier of the document
            
        Returns:
            Optional[Dict[str, Any]]: The document data if found, None otherwise
        """
        try:
            results = self.index.fetch(
                ids=[doc_id],
                namespace=self.namespace
            )
            
            if doc_id in results.vectors:
                vector_data = results.vectors[doc_id]
                return {
                    "id": doc_id,
                    "metadata": vector_data.metadata,
                    "values": vector_data.values
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {str(e)}")
            return None

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id (str): The unique identifier of the document to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.index.delete(
                ids=[doc_id],
                namespace=self.namespace
            )
            logger.success(f"Document {doc_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        Clear all documents from the vector database namespace.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.success(f"Cleared all documents from namespace '{self.namespace}'")
            return True
        except Exception as e:
            logger.error(f"Failed to clear namespace '{self.namespace}': {str(e)}")
            return False

    def count(self) -> int:
        """
        Get the number of documents in the vector database.
        
        Returns:
            int: Number of documents, -1 if not supported
        """
        try:
            stats = self.index.describe_index_stats()
            if self.namespace:
                return stats.namespaces.get(self.namespace, {}).get("vector_count", 0)
            else:
                return stats.total_vector_count
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            return -1

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector database.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Test basic connectivity
            stats = self.index.describe_index_stats()
            
            # Test embedding generation
            test_embedding = self._get_embedding("test")
            
            return {
                "status": "healthy",
                "index_name": self.index_name,
                "namespace": self.namespace,
                "total_vectors": stats.total_vector_count,
                "dimension": self.dimension,
                "embedding_model": self.model_name,
                "embedding_test": len(test_embedding) == self.dimension,
                "timestamp": stats.dimension  # Using this as a proxy for last update
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "index_name": self.index_name,
                "namespace": self.namespace,
                "timestamp": None
            }