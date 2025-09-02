import uuid
from typing import List, Dict, Any, Callable, Optional, Union
import faiss
import numpy as np
import pickle
import os
from loguru import logger
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase
from swarms_memory.embeddings.litellm_embeddings import LiteLLMEmbeddings


class FAISSDB(BaseVectorDatabase):
    """
    A highly customizable wrapper class for FAISS-based vector database with modern embedding support.

    This class provides methods to add documents to the FAISS index and query the index
    for similar documents. It supports any embedding model through LiteLLM or custom embedding functions,
    providing maximum flexibility for text vectorization.
    
    Args:
        embedding_model (Union[str, LiteLLMEmbeddings, Callable]): 
            The embedding model or function to use. Can be:
            - A string model name for LiteLLM (e.g., "text-embedding-3-small", 
              "azure/my-deployment", "cohere/embed-english-v3.0")
            - A LiteLLMEmbeddings instance with custom configuration
            - A custom embedding function that takes text and returns a vector
        dimension (int): Dimension of the document embeddings. Defaults to 1536.
        index_type (str): Type of FAISS index to use. Defaults to 'Flat'.
        metric (str): Distance metric for FAISS index. Defaults to 'cosine'.
        index_file (Optional[str]): Path to save/load FAISS index. If provided, will persist index to disk.
        api_key_embedding (Optional[str]): API key for the embedding provider.
        api_base (Optional[str]): API base URL for providers like Azure.
        api_version (Optional[str]): API version for providers like Azure.
        embedding_kwargs (Optional[Dict]): Additional kwargs for embedding model.
        logger_config (Optional[Dict[str, Any]]): Configuration for the logger. Defaults to None.

    Examples:
        >>> # Example 1: Using OpenAI embeddings
        >>> faiss_db = FAISSDB(
        >>>     embedding_model="text-embedding-3-small",
        >>>     dimension=1536
        >>> )
        >>>
        >>> # Example 2: Using Azure OpenAI embeddings
        >>> faiss_db = FAISSDB(
        >>>     embedding_model="azure/my-embedding-deployment",
        >>>     api_key_embedding="your-azure-key",
        >>>     api_base="https://your-resource.openai.azure.com",
        >>>     api_version="2023-07-01-preview",
        >>>     dimension=1536
        >>> )
        >>>
        >>> # Example 3: Using Cohere embeddings
        >>> faiss_db = FAISSDB(
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
        >>> faiss_db = FAISSDB(
        >>>     embedding_model=my_embedding_func,
        >>>     dimension=1536
        >>> )
        >>>
        >>> # Example 5: With persistent index
        >>> faiss_db = FAISSDB(
        >>>     embedding_model="text-embedding-3-small",
        >>>     dimension=1536,
        >>>     index_file="my_faiss_index.bin"
        >>> )
        >>>
        >>> # Add documents
        >>> doc_id = faiss_db.add("This is a sample document for testing")
        >>>
        >>> # Query similar documents
        >>> results = faiss_db.query("sample query", top_k=5)
        >>> print(results)
    """

    def __init__(
        self,
        embedding_model: Union[str, LiteLLMEmbeddings, Callable],
        dimension: int = 1536,
        index_type: str = "Flat",
        metric: str = "cosine",
        index_file: Optional[str] = None,
        api_key_embedding: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        logger_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the FAISSDB with modern embedding support.
        """
        super().__init__()
        self._setup_logger(logger_config)
        logger.info("Initializing FAISSDB with modern embedding support")

        # Store configuration
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.index_file = index_file

        # Initialize document storage
        self.documents = {}  # doc_id -> metadata mapping
        self.doc_id_to_index = {}  # doc_id -> faiss_index mapping
        self.index_to_doc_id = {}  # faiss_index -> doc_id mapping
        self.next_index = 0

        # Setup embedding model
        self._setup_embedding_model(
            embedding_model,
            api_key_embedding,
            api_base,
            api_version, 
            embedding_kwargs
        )

        # Create or load FAISS index
        self._setup_index()

        logger.info("FAISSDB initialized successfully")

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
        """Create or load the FAISS index."""
        try:
            # Try to load existing index if file specified
            if self.index_file and os.path.exists(self.index_file):
                logger.info(f"Loading existing FAISS index from: {self.index_file}")
                self.index = faiss.read_index(self.index_file)
                
                # Load metadata
                metadata_file = self.index_file + ".metadata"
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                        self.documents = metadata.get('documents', {})
                        self.doc_id_to_index = metadata.get('doc_id_to_index', {})
                        self.index_to_doc_id = metadata.get('index_to_doc_id', {})
                        self.next_index = metadata.get('next_index', 0)
                        
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            else:
                # Create new index
                logger.info(f"Creating new FAISS index: {self.index_type}, metric: {self.metric}")
                self.index = self._create_index()
                
        except Exception as e:
            logger.error(f"Failed to setup FAISS index: {str(e)}")
            raise Exception(f"Failed to setup FAISS index: {str(e)}")

    def _create_index(self):
        """Create and return a FAISS index based on the specified type and metric."""
        if self.metric == "cosine":
            base_index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            base_index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "inner_product":
            base_index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}. Use 'cosine', 'l2', or 'inner_product'")

        if self.index_type == "Flat":
            return base_index
        elif self.index_type == "IVF":
            nlist = 100  # number of clusters
            if self.metric == "cosine":
                quantizer = faiss.IndexFlatIP(self.dimension)
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World index
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 is M parameter
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}. Use 'Flat', 'IVF', or 'HNSW'")

        return index

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

    def _normalize_embedding(self, embedding: List[float]) -> np.ndarray:
        """Normalize embedding for cosine similarity if needed."""
        embedding_array = np.array([embedding], dtype=np.float32)
        
        if self.metric == "cosine":
            # Normalize for cosine similarity
            norms = np.linalg.norm(embedding_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embedding_array = embedding_array / norms
            
        return embedding_array

    def _save_index(self):
        """Save the FAISS index and metadata to disk if index_file is specified."""
        if not self.index_file:
            return
            
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save metadata
            metadata_file = self.index_file + ".metadata"
            metadata = {
                'documents': self.documents,
                'doc_id_to_index': self.doc_id_to_index,
                'index_to_doc_id': self.index_to_doc_id,
                'next_index': self.next_index,
            }
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.debug(f"Saved index to {self.index_file}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")

    def add(self, doc: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the FAISS index.

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
            normalized_embedding = self._normalize_embedding(embedding)
            
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata = metadata or {}
            metadata["text"] = doc
            metadata["model"] = self.model_name
            
            # Add to FAISS index
            self.index.add(normalized_embedding)
            
            # Store document metadata and mappings
            self.documents[doc_id] = metadata
            self.doc_id_to_index[doc_id] = self.next_index
            self.index_to_doc_id[self.next_index] = doc_id
            self.next_index += 1
            
            # Save if persistence enabled
            self._save_index()
            
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
    ) -> List[Dict[str, Any]]:
        """
        Query the FAISS index for similar documents.

        Args:
            query_text (str): The query string.
            top_k (int): The number of top results to return. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]]): Metadata filter for the query.
                Note: FAISS doesn't support native filtering, so this is applied post-search.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the top_k most similar documents.
        """
        try:
            logger.info(f"Querying with: {query_text}")
            self.log_query(query_text)
            
            if self.index.ntotal == 0:
                logger.warning("Index is empty, no documents to search")
                return []
            
            # Generate query embedding
            query_embedding = self._get_embedding(query_text)
            normalized_query = self._normalize_embedding(query_embedding)
            
            # Search FAISS index (search more than needed if filtering is required)
            search_k = top_k * 5 if filter_dict else top_k
            search_k = min(search_k, self.index.ntotal)
            
            scores, indices = self.index.search(normalized_query, search_k)

            # Format results
            formatted_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS uses -1 for empty slots
                    continue
                    
                if idx not in self.index_to_doc_id:
                    continue
                    
                doc_id = self.index_to_doc_id[idx]
                doc_metadata = self.documents.get(doc_id, {})
                
                # Apply metadata filter if provided
                if filter_dict:
                    if not all(
                        doc_metadata.get(key) == value 
                        for key, value in filter_dict.items()
                    ):
                        continue
                
                # Convert score based on metric
                if self.metric == "cosine":
                    similarity_score = float(score)  # Already normalized similarity
                elif self.metric == "l2":
                    similarity_score = 1.0 / (1.0 + float(score))  # Convert distance to similarity
                else:
                    similarity_score = float(score)
                
                formatted_results.append({
                    "id": doc_id,
                    "score": similarity_score,
                    "metadata": doc_metadata,
                })
                
                # Stop when we have enough results
                if len(formatted_results) >= top_k:
                    break

            logger.success(f"Query completed. Found {len(formatted_results)} results.")
            
            # Check if this is being called in a RAG context by examining the stack
            # If so, return formatted text instead of raw results
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
        Query the vector database and return results formatted as text string.
        
        This method is specifically designed for RAG operations where the retrieved
        documents need to be concatenated into a single string for context.
        
        Args:
            query_text (str): The query string.
            top_k (int): The number of top results to return. Defaults to 5.
            filter_dict (Optional[Dict[str, Any]]): Metadata filter for the query.
            separator (str): Separator between documents. Defaults to double newline.
            
        Returns:
            str: Concatenated text content of retrieved documents.
        """
        try:
            results = self.query(query_text, top_k, filter_dict)
            
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

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            doc_id (str): The unique identifier of the document
            
        Returns:
            Optional[Dict[str, Any]]: The document data if found, None otherwise
        """
        try:
            if doc_id not in self.documents:
                return None
                
            metadata = self.documents[doc_id]
            faiss_idx = self.doc_id_to_index.get(doc_id)
            
            return {
                "id": doc_id,
                "metadata": metadata,
                "faiss_index": faiss_idx
            }
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {str(e)}")
            return None

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Note: FAISS doesn't support efficient deletion. This marks the document 
        as deleted but doesn't remove it from the index. Consider rebuilding 
        the index periodically for optimal performance.
        
        Args:
            doc_id (str): The unique identifier of the document to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            if doc_id not in self.documents:
                logger.warning(f"Document {doc_id} not found")
                return False
                
            # Remove from metadata
            del self.documents[doc_id]
            
            # Remove from mappings
            if doc_id in self.doc_id_to_index:
                faiss_idx = self.doc_id_to_index[doc_id]
                del self.doc_id_to_index[doc_id]
                del self.index_to_doc_id[faiss_idx]
            
            # Save if persistence enabled
            self._save_index()
            
            logger.success(f"Document {doc_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        Clear all documents from the vector database.
        
        Returns:
            bool: True if clearing was successful, False otherwise
        """
        try:
            # Reset all data structures
            self.documents = {}
            self.doc_id_to_index = {}
            self.index_to_doc_id = {}
            self.next_index = 0
            
            # Recreate empty index
            self.index = self._create_index()
            
            # Save if persistence enabled
            self._save_index()
            
            logger.success("Cleared all documents from FAISS index")
            return True
        except Exception as e:
            logger.error(f"Failed to clear FAISS index: {str(e)}")
            return False

    def count(self) -> int:
        """
        Get the number of documents in the vector database.
        
        Returns:
            int: Number of documents
        """
        return len(self.documents)

    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index from scratch, removing deleted documents.
        This can improve performance after many deletions.
        
        Returns:
            bool: True if rebuild was successful, False otherwise
        """
        try:
            logger.info("Rebuilding FAISS index...")
            
            # Store current documents
            current_docs = list(self.documents.items())
            
            if not current_docs:
                logger.info("No documents to rebuild")
                return True
            
            # Clear and recreate
            self.clear()
            
            # Re-add all documents
            for doc_id, metadata in current_docs:
                doc_text = metadata.get('text', '')
                if doc_text:
                    # Remove 'text' and 'model' from metadata before re-adding
                    clean_metadata = {k: v for k, v in metadata.items() 
                                    if k not in ['text', 'model']}
                    self.add(doc_text, clean_metadata)
            
            logger.success(f"Index rebuilt successfully with {len(current_docs)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the vector database.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Test basic functionality
            test_embedding = self._get_embedding("test")
            
            return {
                "status": "healthy",
                "total_documents": len(self.documents),
                "index_total": self.index.ntotal,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric": self.metric,
                "embedding_model": self.model_name,
                "embedding_test": len(test_embedding) == self.dimension,
                "persistence_enabled": self.index_file is not None,
                "index_file": self.index_file
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "total_documents": len(self.documents) if hasattr(self, 'documents') else 0,
                "timestamp": None
            }