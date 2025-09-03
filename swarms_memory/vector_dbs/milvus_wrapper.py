import uuid
import inspect
from typing import List, Dict, Any, Callable, Optional, Union
import numpy as np
from loguru import logger
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase
from swarms_memory.embeddings.litellm_embeddings import LiteLLMEmbeddings


class MilvusDB(BaseVectorDatabase):
    """
    A comprehensive wrapper class for Milvus vector database with LiteLLM embedding support.
    
    This class provides methods to add documents to Milvus collections and query them
    for similar documents. It supports any embedding model through LiteLLM or custom 
    embedding functions, and works with both Milvus Lite and Milvus Server deployments.
    
    Args:
        embedding_model (Union[str, LiteLLMEmbeddings, Callable]): 
            The embedding model or function to use. Can be:
            - A string model name for LiteLLM (e.g., "text-embedding-3-small", 
              "azure/my-deployment", "cohere/embed-english-v3.0")
            - A LiteLLMEmbeddings instance with custom configuration
            - A custom embedding function that takes text and returns a vector
        collection_name (str): Name of the Milvus collection. Defaults to "documents".
        uri (Optional[str]): Milvus server URI. If None, uses Milvus Lite with local file.
        token (Optional[str]): Authentication token for Milvus server.
        db_file (Optional[str]): Local database file for Milvus Lite. Defaults to "milvus_lite.db".
        dimension (Optional[int]): Dimension of embeddings. Auto-detected if not provided.
        metric (str): Distance metric for similarity search. Defaults to "COSINE".
        index_type (str): Vector index type. Defaults to "FLAT".
        api_key_embedding (Optional[str]): API key for the embedding provider.
        api_base (Optional[str]): API base URL for providers like Azure.
        api_version (Optional[str]): API version for providers like Azure.
        embedding_kwargs (Optional[Dict]): Additional kwargs for embedding model.
        
    Examples:
        >>> # Example 1: Using OpenAI embeddings with Milvus Lite
        >>> milvus_db = MilvusDB(
        >>>     embedding_model="text-embedding-3-small",
        >>>     collection_name="my_docs"
        >>> )
        >>>
        >>> # Example 2: Using Milvus Server with custom configuration
        >>> milvus_db = MilvusDB(
        >>>     embedding_model="text-embedding-3-small",
        >>>     uri="http://localhost:19530",
        >>>     token="root:Milvus",
        >>>     collection_name="my_docs"
        >>> )
        >>>
        >>> # Example 3: Using custom embedding function
        >>> def my_embedding_func(text: str) -> List[float]:
        >>>     # Your custom embedding logic here
        >>>     return [0.1] * 768
        >>> 
        >>> milvus_db = MilvusDB(
        >>>     embedding_model=my_embedding_func,
        >>>     dimension=768,
        >>>     collection_name="my_docs"
        >>> )
        >>>
        >>> # Example 4: Using LiteLLMEmbeddings instance
        >>> embedder = LiteLLMEmbeddings(
        >>>     model="text-embedding-3-small",
        >>>     api_key="your-api-key"
        >>> )
        >>> milvus_db = MilvusDB(
        >>>     embedding_model=embedder,
        >>>     collection_name="my_docs"
        >>> )
        
    Methods:
        add(doc: str, metadata: Optional[Dict[str, Any]] = None) -> str:
            Add a document with optional metadata to the collection.
            
        query(query_text: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> Union[List[Dict[str, Any]], str]:
            Query the collection for similar documents. Returns formatted string for RAG context, list otherwise.
            
        get(doc_id: str) -> Optional[Dict[str, Any]]:
            Retrieve a document by its ID.
            
        delete(doc_id: str) -> bool:
            Delete a document by its ID.
            
        clear() -> bool:
            Clear all documents from the collection.
            
        count() -> int:
            Get the number of documents in the collection.
            
        health_check() -> Dict[str, Any]:
            Check the health status of the Milvus connection.
    """
    
    def __init__(
        self,
        embedding_model: Union[str, LiteLLMEmbeddings, Callable],
        collection_name: str = "documents",
        uri: Optional[str] = None,
        token: Optional[str] = None,
        db_file: str = "milvus_lite.db",
        dimension: Optional[int] = None,
        metric: str = "COSINE",
        index_type: str = "FLAT",
        api_key_embedding: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_kwargs: Optional[Dict] = None,
    ):
        """Initialize the MilvusDB wrapper."""
        super().__init__()
        
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "pymilvus is required for MilvusDB. Install it with: pip install pymilvus"
            )
        
        self.collection_name = collection_name
        self.metric = metric.upper()
        self.index_type = index_type.upper()
        self.dimension = dimension
        
        # Initialize embedding model
        self.embedder = None
        self.embedding_func = None
        self.model_name = None
        
        self._setup_embedding_model(
            embedding_model, api_key_embedding, api_base, 
            api_version, embedding_kwargs
        )
        
        # Initialize Milvus client
        if uri is None:
            # Use Milvus Lite with local file
            self.client = MilvusClient(db_file)
            logger.info(f"Initialized Milvus Lite with database file: {db_file}")
        else:
            # Use Milvus Server
            self.client = MilvusClient(uri=uri, token=token)
            logger.info(f"Initialized Milvus Server connection to: {uri}")
        
        # Auto-detect dimension if not provided
        if self.dimension is None:
            self._detect_embedding_dimension()
        
        # Setup collection
        self._setup_collection()
        
        logger.info(f"MilvusDB initialized successfully with collection '{self.collection_name}'")
    
    def _setup_embedding_model(
        self,
        embedding_model: Union[str, LiteLLMEmbeddings, Callable],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_kwargs: Optional[Dict] = None,
    ):
        """Set up the embedding model based on the provided input."""
        embedding_kwargs = embedding_kwargs or {}
        
        if isinstance(embedding_model, str):
            # String model name - create LiteLLMEmbeddings instance
            self.model_name = embedding_model
            self.embedder = LiteLLMEmbeddings(
                model=embedding_model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                **embedding_kwargs
            )
            logger.info(f"Using LiteLLM model: {embedding_model}")
            
        elif isinstance(embedding_model, LiteLLMEmbeddings):
            # Pre-configured LiteLLMEmbeddings instance
            self.embedder = embedding_model
            self.model_name = embedding_model.model
            logger.info(f"Using pre-configured LiteLLMEmbeddings: {self.model_name}")
            
        elif callable(embedding_model):
            # Custom embedding function
            self.embedding_func = embedding_model
            self.model_name = "custom_function"
            logger.info("Using custom embedding function")
            
        else:
            raise ValueError(
                "embedding_model must be a string, LiteLLMEmbeddings instance, or callable function"
            )
    
    def _detect_embedding_dimension(self):
        """Auto-detect the embedding dimension by testing with sample text."""
        try:
            sample_embedding = self._get_embedding("Sample text for dimension detection")
            self.dimension = len(sample_embedding)
            logger.info(f"Detected embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to detect embedding dimension: {str(e)}")
            raise ValueError("Could not detect embedding dimension. Please provide dimension parameter.")
    
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
            raise
    
    def _setup_collection(self):
        """Set up the Milvus collection with appropriate schema and index."""
        try:
            # Drop existing collection if it exists (for clean setup)
            if self.client.has_collection(collection_name=self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
            else:
                # Create new collection with specified dimension
                # Use auto_id=True to let Milvus generate integer IDs
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=self.dimension,
                    metric_type=self.metric,
                    index_type=self.index_type,
                    auto_id=True  # Let Milvus generate IDs automatically
                )
                logger.info(f"Created new collection '{self.collection_name}' with dimension {self.dimension}")
                
        except Exception as e:
            logger.error(f"Failed to setup collection: {str(e)}")
            raise
    
    def add(self, doc: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the Milvus collection.
        
        Args:
            doc (str): The document text to add
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document
            
        Returns:
            str: The unique identifier for the added document
        """
        try:
            logger.info(f"Adding document: {doc[:50]}...")
            
            # Generate embedding
            embedding = self._get_embedding(doc)
            
            # Prepare metadata
            metadata = metadata or {}
            metadata["text"] = doc
            metadata["model"] = self.model_name
            
            # Prepare data for insertion (no ID since auto_id=True)
            data = [{
                "vector": embedding,
                **metadata
            }]
            
            # Insert into collection
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            # Get the generated ID from the result
            doc_id = str(result['ids'][0]) if result.get('ids') else "unknown"
            
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
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Query the Milvus collection for similar documents.
        
        Args:
            query_text (str): The query string
            top_k (int): The number of top results to return. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]]): Metadata filter for the query
            
        Returns:
            Union[List[Dict[str, Any]], str]: List of dictionaries for normal use, 
            formatted string for RAG operations
        """
        try:
            logger.info(f"Querying with: {query_text}")
            self.log_query(query_text)
            
            # Generate query embedding
            query_embedding = self._get_embedding(query_text)
            
            # Prepare filter expression
            filter_expr = None
            if filter_dict:
                # Convert filter dict to Milvus expression format
                conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, str):
                        conditions.append(f'{key} == "{value}"')
                    else:
                        conditions.append(f'{key} == {value}')
                filter_expr = " and ".join(conditions)
            
            # Search in collection
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr,
                output_fields=["*"]  # Return all fields including metadata
            )
            
            # Format results
            formatted_results = []
            if search_results and len(search_results) > 0:
                for hit in search_results[0]:  # search_results[0] contains results for first query
                    result_dict = {
                        "id": hit.get("id"),
                        "score": 1.0 - hit.get("distance", 0),  # Convert distance to similarity
                        "metadata": hit.get("entity", {})
                    }
                    formatted_results.append(result_dict)
            
            logger.success(f"Query completed. Found {len(formatted_results)} results.")
            
            # Check if this is being called in a RAG context by examining the stack
            import inspect
            frame = inspect.currentframe()
            try:
                # Look for signs this is a RAG query from swarms framework
                for i in range(10):  # Check up to 10 frames up
                    if frame is None:
                        break
                    frame = frame.f_back
                    if frame and frame.f_code:
                        func_name = frame.f_code.co_name
                        filename = frame.f_code.co_filename
                        # Check if we're being called from swarms RAG handling
                        if ('rag' in func_name.lower() or 
                            'handle' in func_name.lower() or 
                            'dynamic_auto_chunking' in filename or
                            'swarms' in filename):
                            # Return formatted text instead of list for RAG context
                            text_chunks = []
                            for result in formatted_results:
                                metadata = result.get("metadata", {})
                                text_content = metadata.get("text", "")
                                if text_content:
                                    text_chunks.append(text_content)
                            formatted_text = "\n\n".join(text_chunks)
                            logger.info(f"Detected RAG context, returning formatted text with {len(text_chunks)} chunks")
                            return formatted_text
            except:
                # If stack inspection fails, continue with normal behavior
                pass
            finally:
                del frame
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise Exception(f"Query failed: {str(e)}")
    
    def query_as_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        separator: str = "\n\n"
    ) -> str:
        """
        Query the collection and return results formatted as text string.
        
        This method is specifically designed for RAG operations where the retrieved
        documents need to be concatenated into a single string for context.
        
        Args:
            query_text (str): The query string
            top_k (int): The number of top results to return. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]]): Metadata filter for the query
            separator (str): Separator between documents. Defaults to double newline.
            
        Returns:
            str: Concatenated text content of retrieved documents
        """
        try:
            # Temporarily disable RAG context detection for this call
            results = self._query_raw(query_text, top_k, filter_dict)
            
            if not results:
                logger.warning("No results found for query")
                return ""
            
            # Extract text content from each result
            text_chunks = []
            for result in results:
                metadata = result.get("metadata", {})
                text_content = metadata.get("text", "")
                if text_content:
                    text_chunks.append(text_content)
            
            # Join all text chunks with separator
            formatted_text = separator.join(text_chunks)
            logger.success(f"Formatted {len(text_chunks)} documents as text string")
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"Query as text failed: {str(e)}")
            raise Exception(f"Query as text failed: {str(e)}")
    
    def _query_raw(
        self,
        query_text: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Internal query method that always returns list format (no RAG context detection).
        """
        try:
            logger.info(f"Raw querying with: {query_text}")
            self.log_query(query_text)
            
            # Generate query embedding
            query_embedding = self._get_embedding(query_text)
            
            # Prepare filter expression
            filter_expr = None
            if filter_dict:
                # Convert filter dict to Milvus expression format
                conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, str):
                        conditions.append(f'{key} == "{value}"')
                    else:
                        conditions.append(f'{key} == {value}')
                filter_expr = " and ".join(conditions)
            
            # Search in collection
            search_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                filter=filter_expr,
                output_fields=["*"]  # Return all fields including metadata
            )
            
            # Format results
            formatted_results = []
            if search_results and len(search_results) > 0:
                for hit in search_results[0]:  # search_results[0] contains results for first query
                    result_dict = {
                        "id": hit.get("id"),
                        "score": 1.0 - hit.get("distance", 0),  # Convert distance to similarity
                        "metadata": hit.get("entity", {})
                    }
                    formatted_results.append(result_dict)
            
            logger.success(f"Raw query completed. Found {len(formatted_results)} results.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Raw query failed: {str(e)}")
            raise Exception(f"Raw query failed: {str(e)}")
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            doc_id (str): The unique identifier of the document
            
        Returns:
            Optional[Dict[str, Any]]: The document data if found, None otherwise
        """
        try:
            # Convert string ID to integer for Milvus
            int_id = int(doc_id)
            
            results = self.client.query(
                collection_name=self.collection_name,
                ids=[int_id],
                output_fields=["*"]
            )
            
            if results and len(results) > 0:
                doc_data = results[0]
                return {
                    "id": str(doc_data.get("id")),  # Convert back to string
                    "metadata": {k: v for k, v in doc_data.items() if k not in ["id", "vector"]}
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
            # Convert string ID to integer for Milvus
            int_id = int(doc_id)
            
            result = self.client.delete(
                collection_name=self.collection_name,
                ids=[int_id]
            )
            
            logger.info(f"Document {doc_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            # Drop and recreate collection
            if self.client.has_collection(collection_name=self.collection_name):
                self.client.drop_collection(collection_name=self.collection_name)
            
            self._setup_collection()
            logger.info(f"Collection '{self.collection_name}' cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False
    
    def count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            int: Number of documents
        """
        try:
            # Use collection stats to get count
            stats = self.client.describe_collection(collection_name=self.collection_name)
            
            # Try to get count from collection stats
            if hasattr(stats, 'num_entities'):
                return stats.num_entities
            
            # Fallback: Query with reasonable limit and estimate
            results = self.client.query(
                collection_name=self.collection_name,
                filter="",  # No filter to get all
                output_fields=["id"],
                limit=16384  # Maximum allowed limit
            )
            
            return len(results) if results else 0
            
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            return -1
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Milvus connection.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Try to list collections to test connection
            collections = self.client.list_collections()
            
            return {
                "status": "healthy",
                "message": "Milvus connection is working",
                "collections": collections,
                "timestamp": None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Milvus connection failed: {str(e)}",
                "timestamp": None
            }