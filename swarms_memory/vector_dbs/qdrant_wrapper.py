import uuid
from typing import Optional, Union, Callable, List, Dict, Any

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from loguru import logger
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase
from swarms_memory.embeddings.litellm_embeddings import LiteLLMEmbeddings

# Load environment variables
load_dotenv()


class QdrantDB(BaseVectorDatabase):
    """
    A memory vector database wrapper for [Qdrant](https://qdrant.tech/) with flexible embedding support.

    This wrapper now supports any embedding model through LiteLLM or custom embedding functions,
    providing maximum flexibility for text vectorization.

    Args:
        client (QdrantClient): A pre-configured Qdrant client instance for
            database connection.
        collection_name (str, optional): The name of the collection to store
            vectors in. Defaults to "swarms_collection".
        embedding_model (Union[str, LiteLLMEmbeddings, Callable]): 
            The embedding model or function to use. Can be:
            - A string model name for LiteLLM (e.g., "text-embedding-3-small", 
              "azure/my-deployment", "cohere/embed-english-v3.0")
            - A LiteLLMEmbeddings instance with custom configuration
            - A custom embedding function that takes text and returns a vector
            - Uses Qdrant's built-in models if "qdrant:" prefix is used
        embedding_dim (Optional[int]): The dimension of embeddings. If not provided,
            will be auto-detected from the embedding model.
        distance (models.Distance, optional): The distance metric to use for
            similarity search. Defaults to models.Distance.COSINE.
        n_results (int, optional): The number of results to retrieve during
            similarity search. Defaults to 10.
        api_key (Optional[str]): API key for the embedding provider.
        api_base (Optional[str]): API base URL for providers like Azure.
        api_version (Optional[str]): API version for providers like Azure.
        embedding_kwargs (Optional[Dict]): Additional kwargs for embedding model.
        *args: Additional positional arguments passed to the parent class.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:
        >>> from qdrant_client import QdrantClient
        >>>
        >>> # Initialize Qdrant client
        >>> client = QdrantClient("localhost", port=6333)
        >>>
        >>> # Example 1: Using OpenAI embeddings
        >>> qdrant_db = QdrantDB(
        >>>     client=client,
        >>>     collection_name="my_collection",
        >>>     embedding_model="text-embedding-3-small",
        >>>     embedding_dim=1536,
        >>>     distance=models.Distance.COSINE
        >>> )
        >>>
        >>> # Example 2: Using Azure OpenAI embeddings
        >>> qdrant_db = QdrantDB(
        >>>     client=client,
        >>>     embedding_model="azure/my-embedding-deployment",
        >>>     api_key="your-api-key",
        >>>     api_base="https://your-resource.openai.azure.com",
        >>>     api_version="2023-07-01-preview"
        >>> )
        >>>
        >>> # Example 3: Using Cohere embeddings
        >>> qdrant_db = QdrantDB(
        >>>     client=client,
        >>>     embedding_model="cohere/embed-english-v3.0",
        >>>     embedding_kwargs={"input_type": "search_document"}
        >>> )
        >>>
        >>> # Example 4: Using custom LiteLLMEmbeddings instance
        >>> from swarms_memory.embeddings.litellm_embeddings import LiteLLMEmbeddings
        >>> embedder = LiteLLMEmbeddings(
        >>>     model="voyage/voyage-3-large",
        >>>     dimensions=1024
        >>> )
        >>> qdrant_db = QdrantDB(
        >>>     client=client,
        >>>     embedding_model=embedder
        >>> )
        >>>
        >>> # Example 5: Using Qdrant's built-in models
        >>> qdrant_db = QdrantDB(
        >>>     client=client,
        >>>     embedding_model="qdrant:sentence-transformers/all-MiniLM-L6-v2"
        >>> )
        >>>
        >>> # Example 6: Using custom embedding function
        >>> def my_embedding_func(text: str) -> List[float]:
        >>>     # Your custom embedding logic here
        >>>     return [0.1] * 384
        >>> 
        >>> qdrant_db = QdrantDB(
        >>>     client=client,
        >>>     embedding_model=my_embedding_func,
        >>>     embedding_dim=384
        >>> )
        >>>
        >>> # Add documents
        >>> doc_id = qdrant_db.add("This is a sample document for testing")
        >>>
        >>> # Query similar documents
        >>> results = qdrant_db.query("sample query")
        >>> print(results)
    """

    def __init__(
        self,
        client: QdrantClient,
        embedding_model: Union[str, 'LiteLLMEmbeddings', Callable],
        collection_name: str = "swarms_collection",
        embedding_dim: Optional[int] = None,
        distance: models.Distance = models.Distance.COSINE,
        n_results: int = 10,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.collection_name = collection_name
        self.distance = distance
        self.n_results = n_results
        self.client = client
        self.embedding_dim = embedding_dim
        self._use_qdrant_builtin = False
        
        # Setup embedding model
        self._setup_embedding_model(
            embedding_model, 
            api_key, 
            api_base, 
            api_version, 
            embedding_kwargs
        )
        
        # Get or create collection
        self._setup_collection()

    def _setup_embedding_model(
        self,
        embedding_model: Union[str, 'LiteLLMEmbeddings', Callable],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Setup the embedding model based on the provided configuration."""
        
        if isinstance(embedding_model, str):
            # Check if it's a Qdrant built-in model
            if embedding_model.startswith("qdrant:"):
                self._use_qdrant_builtin = True
                self.model_name = embedding_model[7:]  # Remove "qdrant:" prefix
                self.embedding_func = None
                logger.info(f"Using Qdrant built-in model: {self.model_name}")
            else:
                # Use LiteLLM for the model
                self._use_qdrant_builtin = False
                kwargs = embedding_kwargs or {}
                self.embedder = LiteLLMEmbeddings(
                    model=embedding_model,
                    api_key=api_key,
                    api_base=api_base,
                    api_version=api_version,
                    **kwargs
                )
                self.embedding_func = None
                logger.info(f"Using LiteLLM model: {embedding_model}")
                
                # Auto-detect embedding dimension if not provided
                if self.embedding_dim is None:
                    self.embedding_dim = self.embedder.get_embedding_dimension()
                    
        elif isinstance(embedding_model, LiteLLMEmbeddings):
            # Direct LiteLLMEmbeddings instance
            self._use_qdrant_builtin = False
            self.embedder = embedding_model
            self.embedding_func = None
            logger.info("Using provided LiteLLMEmbeddings instance")
            
            # Auto-detect embedding dimension if not provided
            if self.embedding_dim is None:
                self.embedding_dim = self.embedder.get_embedding_dimension()
                
        elif callable(embedding_model):
            # Custom embedding function
            self._use_qdrant_builtin = False
            self.embedding_func = embedding_model
            self.embedder = None
            logger.info("Using custom embedding function")
            
            # For custom functions, embedding_dim must be provided
            if self.embedding_dim is None:
                raise ValueError(
                    "embedding_dim must be provided when using a custom embedding function"
                )
        else:
            raise ValueError(
                f"Invalid embedding_model type: {type(embedding_model)}. "
                "Must be a string, LiteLLMEmbeddings instance, or callable."
            )
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using the configured model."""
        if self._use_qdrant_builtin:
            # This will be handled by Qdrant's models.Document
            return None
        elif self.embedding_func:
            return self.embedding_func(text)
        elif self.embedder:
            return self.embedder.embed_query(text)
        else:
            raise RuntimeError("No embedding model configured")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using the configured model."""
        if self._use_qdrant_builtin:
            # This will be handled by Qdrant's models.Document
            return None
        elif self.embedding_func:
            return [self.embedding_func(text) for text in texts]
        elif self.embedder:
            return self.embedder.embed_documents(texts)
        else:
            raise RuntimeError("No embedding model configured")
    
    def _setup_collection(self):
        """Set up the Qdrant collection with proper configuration."""
        if not self.client.collection_exists(self.collection_name):
            # Determine vector size
            if self._use_qdrant_builtin:
                vector_size = self.client.get_embedding_size(self.model_name)
            elif self.embedding_dim:
                vector_size = self.embedding_dim
            else:
                # Try to detect dimension from a sample embedding
                try:
                    sample_embedding = self._get_embedding("sample text")
                    vector_size = len(sample_embedding)
                    self.embedding_dim = vector_size
                except Exception as e:
                    logger.error(f"Failed to detect embedding dimension: {e}")
                    raise ValueError(
                        "Could not determine embedding dimension. "
                        "Please provide embedding_dim parameter."
                    )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=self.distance,
                ),
            )
            logger.info(
                f"Created Qdrant collection: {self.collection_name} with vector size {vector_size}"
            )
        else:
            logger.info(
                f"Using existing collection: {self.collection_name}"
            )

    def add(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a document to the Qdrant collection.

        Args:
            document (str): The document to be added.
            metadata (Optional[Dict[str, Any]]): Additional metadata to store with the document.

        Returns:
            str: The ID of the added document.
        """
        doc_id = str(uuid.uuid4())
        
        # Prepare payload
        payload = {"document": document}
        if metadata:
            payload.update(metadata)
        
        # Create point based on embedding type
        if self._use_qdrant_builtin:
            # Use Qdrant's built-in model
            point = models.PointStruct(
                id=doc_id,
                vector=models.Document(
                    text=document, model=self.model_name
                ),
                payload=payload,
            )
        else:
            # Use custom embedding
            embedding = self._get_embedding(document)
            point = models.PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload,
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        
        logger.debug(f"Added document with ID: {doc_id}")
        return doc_id

    def query(
        self,
        query_text: str,
        n_results: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Query documents from the Qdrant collection.

        Args:
            query_text (str): The query string.
            n_results (Optional[int]): Number of results to retrieve. Uses self.n_results if not provided.
            return_metadata (bool): If True, returns list of dicts with documents and metadata.

        Returns:
            Union[str, List[Dict[str, Any]]]: Either concatenated documents as string,
                or list of dicts with 'document' and metadata if return_metadata=True.
        """
        n_results = n_results or self.n_results
        
        # Prepare query based on embedding type
        if self._use_qdrant_builtin:
            query = models.Document(
                text=query_text, model=self.model_name
            )
        else:
            query = self._get_embedding(query_text)
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            limit=n_results,
            with_payload=True,
            with_vectors=False,
        )
        
        if return_metadata:
            # Return list of documents with metadata
            results = []
            for point in search_result.points:
                result = point.payload.copy()
                result['score'] = point.score if hasattr(point, 'score') else None
                result['id'] = point.id
                results.append(result)
            return results
        else:
            # Return concatenated documents as string
            docs = [
                point.payload.get("document", "")
                for point in search_result.points
            ]
            return "\n".join(docs)
    
    def batch_add(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> List[str]:
        """
        Add multiple documents to the Qdrant collection in batches.

        Args:
            documents (List[str]): List of documents to be added.
            metadata (Optional[List[Dict[str, Any]]]): List of metadata dicts for each document.
            batch_size (int): Number of documents to process in each batch.

        Returns:
            List[str]: List of IDs for the added documents.
        """
        if metadata and len(metadata) != len(documents):
            raise ValueError("Length of metadata must match length of documents")
        
        doc_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size] if metadata else [{}] * len(batch_docs)
            batch_ids = [str(uuid.uuid4()) for _ in batch_docs]
            
            points = []
            
            if self._use_qdrant_builtin:
                # Use Qdrant's built-in model
                for doc_id, doc, meta in zip(batch_ids, batch_docs, batch_metadata):
                    payload = {"document": doc}
                    payload.update(meta)
                    
                    points.append(
                        models.PointStruct(
                            id=doc_id,
                            vector=models.Document(
                                text=doc, model=self.model_name
                            ),
                            payload=payload,
                        )
                    )
            else:
                # Use custom embeddings - batch process for efficiency
                embeddings = self._get_embeddings(batch_docs)
                
                for doc_id, doc, embedding, meta in zip(
                    batch_ids, batch_docs, embeddings, batch_metadata
                ):
                    payload = {"document": doc}
                    payload.update(meta)
                    
                    points.append(
                        models.PointStruct(
                            id=doc_id,
                            vector=embedding,
                            payload=payload,
                        )
                    )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            doc_ids.extend(batch_ids)
            logger.debug(f"Added batch of {len(batch_ids)} documents")
        
        logger.info(f"Added total of {len(doc_ids)} documents to collection")
        return doc_ids
