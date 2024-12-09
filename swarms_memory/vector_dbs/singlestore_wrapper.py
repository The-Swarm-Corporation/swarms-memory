from typing import Any, Callable, Dict, List, Optional
import os
import uuid
import json
import numpy as np
from loguru import logger
import singlestoredb as s2
from sentence_transformers import SentenceTransformer
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase


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
        embedding_model: Optional[Any] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        preprocess_function: Optional[Callable[[str], str]] = None,
        postprocess_function: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        namespace: str = "",
        logger_config: Optional[Dict[str, Any]] = None,
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
            embedding_model (Optional[Any]): Model for generating embeddings. Defaults to None
            embedding_function (Optional[Callable]): Custom function for generating embeddings. Defaults to None
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

        # Set up embedding model and functions
        self.embedding_model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_function = embedding_function or self._default_embedding_function
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
        """Default embedding function using the SentenceTransformer model."""
        return self.embedding_model.encode(text)

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
        embedding = self.embedding_function(processed_doc)
        
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
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar documents.

        Args:
            query (str): Query text
            top_k (int): Number of results to return. Defaults to 5
            metadata_filter (Optional[Dict[str, Any]]): Filter results by metadata

        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata and similarity scores
        """
        logger.info(f"Querying with: {query}")

        # Process query and generate embedding
        processed_query = self.preprocess_function(query)
        query_embedding = self.embedding_function(processed_query)

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
            formatted_results.append({
                "id": doc_id,
                "document": self.postprocess_function(document) if document else None,
                "metadata": metadata,
                "similarity": float(similarity)
            })

        logger.success(f"Query completed. Found {len(formatted_results)} results.")
        return formatted_results

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
