from typing import Any, Callable, Dict, List, Optional, Union
import os
import uuid
import json
import numpy as np
from loguru import logger
import singlestoredb as s2
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase
from swarms_memory.embeddings.embedding_utils import setup_unified_embedding, get_embedding_function


class SingleStoreDB(BaseVectorDatabase):
    """
    A highly customizable wrapper class for SingleStore-based Retrieval-Augmented Generation (RAG) system.

    This class provides methods to add documents to SingleStore and query for similar documents
    using vector similarity search. It supports custom embedding models, preprocessing functions,
    and other customizations.
    """

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        table_name: str,
        dimension: int = 768,
        port: int = 3306,
        ssl: bool = True,
        ssl_verify: bool = True,
        embedding_model: Union[str, Any, Callable] = "text-embedding-3-small",
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        preprocess_function: Optional[Callable[[str], str]] = None,
        postprocess_function: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        namespace: str = "",
        logger_config: Optional[Dict[str, Any]] = None,
        **embedding_kwargs
    ):
        """
        Initialize the SingleStoreDB wrapper.

        Args:
            host (str): SingleStore host address
            user (str): SingleStore username
            password (str): SingleStore password
            database (str): Database name
            table_name (str): Table name for vector storage
            dimension (int): Dimension of vectors to store. Defaults to 768
            port (int): SingleStore port number. Defaults to 3306
            ssl (bool): Whether to use SSL for connection. Defaults to True
            ssl_verify (bool): Whether to verify SSL certificate. Defaults to True
            embedding_model (Union[str, Any, Callable]): Embedding model specification:
                - str: LiteLLM model name (e.g., "text-embedding-3-small") or SentenceTransformer model
                - LiteLLMEmbeddings: Pre-configured instance
                - Callable: Custom embedding function
            embedding_function (Optional[Callable]): Legacy embedding function (for backward compatibility)
            preprocess_function (Optional[Callable]): Custom function for preprocessing documents. Defaults to None
            postprocess_function (Optional[Callable]): Custom function for postprocessing query results. Defaults to None
            namespace (str): Namespace for document organization. Defaults to ""
            logger_config (Optional[Dict]): Configuration for the logger. Defaults to None
        """
        super().__init__()
        self._setup_logger(logger_config)
        logger.info("Initializing SingleStoreDB")

        # Store connection parameters
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.ssl = ssl
        self.ssl_verify = ssl_verify
        
        self.table_name = table_name
        self.dimension = dimension
        self.namespace = namespace

        # Setup unified embedding system
        self.embedder, custom_embedding_function, detected_dimension = setup_unified_embedding(
            embedding_model=embedding_model,
            embedding_function=embedding_function,
            dimension=dimension,
            **embedding_kwargs
        )
        
        # Use detected dimension if available, otherwise use provided dimension
        self.dimension = detected_dimension or dimension
        
        self.embedding_function = (
            custom_embedding_function or 
            get_embedding_function(self.embedder) if self.embedder else
            self._default_embedding_function
        )
        self.preprocess_function = preprocess_function or self._default_preprocess_function
        self.postprocess_function = postprocess_function or self._default_postprocess_function

        # Initialize database and create table if needed
        self._initialize_database()
        logger.info("SingleStoreDB initialized successfully")

    def _setup_logger(self, config: Optional[Dict[str, Any]] = None):
        """Set up the logger with the given configuration."""
        default_config = {
            "handlers": [
                {"sink": "singlestore_wrapper.log", "rotation": "500 MB"},
                {"sink": lambda msg: print(msg, end="")},
            ],
        }
        logger.configure(**(config or default_config))

    def _default_embedding_function(self, text: str) -> np.ndarray:
        """Default embedding function - should not be called if embedding setup is correct."""
        raise NotImplementedError(
            "Default embedding function called - this indicates an issue with embedding setup. "
            "Please check your embedding_model configuration."
        )

    def _default_preprocess_function(self, text: str) -> str:
        """Default preprocessing function."""
        return text.strip()

    def _default_postprocess_function(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Default postprocessing function."""
        return results

    def _initialize_database(self):
        """Initialize the database and create the vector table if it doesn't exist."""
        # Build connection string with SSL options
        ssl_params = []
        if self.ssl:
            ssl_params.append("ssl=true")
            if not self.ssl_verify:
                ssl_params.append("ssl_verify=false")
        
        ssl_string = "&".join(ssl_params)
        if ssl_string:
            ssl_string = "?" + ssl_string

        # Use standard connection URL format as per documentation
        self.connection_string = f"{self.user}:{self.password}@{self.host}:{self.port}/{self.database}{ssl_string}"

        with s2.connect(self.connection_string) as conn:
            with conn.cursor() as cursor:
                # Create table with optimized settings for vector operations
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        document TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
                        embedding BLOB,
                        metadata JSON,
                        namespace VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        KEY idx_namespace (namespace),
                        VECTOR INDEX vec_idx (embedding) DIMENSION = {self.dimension} DISTANCE_TYPE = DOT_PRODUCT
                    ) ENGINE = columnstore;
                """)
                logger.info(f"Table {self.table_name} initialized")

    def add(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a document to the vector database.

        Args:
            document (str): The document text to add
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document

        Returns:
            str: Document ID of the added document
        """
        logger.info(f"Adding document: {document[:50]}...")
        
        # Process document and generate embedding
        processed_doc = self.preprocess_function(document)
        embedding_result = self.embedding_function(processed_doc)
        
        # Convert to numpy array if needed
        if isinstance(embedding_result, list):
            embedding = np.array(embedding_result, dtype=np.float32)
        else:
            embedding = embedding_result
        
        # Prepare metadata
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata_json = json.dumps(metadata)

        # Insert into database
        with s2.connect(self.connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name}
                    (id, document, embedding, metadata, namespace)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (doc_id, processed_doc, embedding, metadata_json, self.namespace)
                )

        logger.success(f"Document added successfully with ID: {doc_id}")
        return doc_id

    def query(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Query the vector database for similar documents.

        Args:
            query (str): Query text
            top_k (int): Number of results to return. Defaults to 5
            metadata_filter (Optional[Dict[str, Any]]): Filter results by metadata
            return_metadata (bool): Whether to return detailed metadata. Defaults to False.

        Returns:
            Union[str, List[Dict[str, Any]]]: If return_metadata=False, returns concatenated text.
                If return_metadata=True, returns list of dictionaries with detailed results.
        """
        logger.info(f"Querying with: {query}")

        # Process query and generate embedding
        processed_query = self.preprocess_function(query)
        embedding_result = self.embedding_function(processed_query)
        
        # Convert to numpy array if needed
        if isinstance(embedding_result, list):
            query_embedding = np.array(embedding_result, dtype=np.float32)
        else:
            query_embedding = embedding_result

        # Construct metadata filter if provided
        filter_clause = ""
        if metadata_filter:
            filter_conditions = []
            for key, value in metadata_filter.items():
                filter_conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = '{value}'")
            if filter_conditions:
                filter_clause = "AND " + " AND ".join(filter_conditions)

        # Query database
        with s2.connect(self.connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id, document, metadata, DOT_PRODUCT(embedding, %s) as similarity
                    FROM {self.table_name}
                    WHERE namespace = %s {filter_clause}
                    ORDER BY similarity DESC
                    LIMIT %s
                    """,
                    (query_embedding, self.namespace, top_k)
                )
                results = cursor.fetchall()

        # Format and process results
        formatted_results = []
        for doc_id, document, metadata_json, similarity in results:
            metadata = json.loads(metadata_json) if metadata_json else {}
            metadata["text"] = document  # Add text to metadata for consistency
            formatted_results.append({
                "id": doc_id,
                "score": float(similarity),
                "metadata": metadata,
            })

        processed_results = self.postprocess_function(formatted_results)
        logger.success(f"Query completed. Found {len(processed_results)} results.")
        
        if return_metadata:
            return processed_results
        else:
            # Return concatenated text for backward compatibility
            return "\n\n".join(
                result["metadata"].get("text", str(result["metadata"]))
                for result in processed_results
            )

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the database.

        Args:
            doc_id (str): ID of the document to delete

        Returns:
            bool: True if deletion was successful
        """
        logger.info(f"Deleting document with ID: {doc_id}")
        
        with s2.connect(self.connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE id = %s AND namespace = %s",
                    (doc_id, self.namespace)
                )
                deleted = cursor.rowcount > 0

        if deleted:
            logger.success(f"Document {doc_id} deleted successfully")
        else:
            logger.warning(f"Document {doc_id} not found")
        
        return deleted
    
    def clear(self) -> bool:
        """
        Clear all documents from the namespace.

        Returns:
            bool: True if clearing was successful, False otherwise.
        """
        try:
            logger.info(f"Clearing all documents from namespace: {self.namespace}")
            
            with s2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"DELETE FROM {self.table_name} WHERE namespace = %s",
                        (self.namespace,)
                    )
                    deleted_count = cursor.rowcount
            
            logger.success(f"Cleared {deleted_count} documents successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear documents: {str(e)}")
            return False
