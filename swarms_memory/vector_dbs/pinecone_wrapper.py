from typing import Any, Callable, Dict, List, Optional

import pinecone
from loguru import logger
from sentence_transformers import SentenceTransformer
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase

class PineconeMemory(BaseVectorDatabase):
    """
    A highly customizable wrapper class for Pinecone-based Retrieval-Augmented Generation (RAG) system.

    This class provides methods to add documents to the Pinecone index and query the index
    for similar documents. It allows for custom embedding models, preprocessing functions,
    and other customizations.
    """

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 768,
        embedding_model: Optional[Any] = None,
        embedding_function: Optional[
            Callable[[str], List[float]]
        ] = None,
        preprocess_function: Optional[Callable[[str], str]] = None,
        postprocess_function: Optional[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
        ] = None,
        metric: str = "cosine",
        pod_type: str = "p1",
        namespace: str = "",
        logger_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the PineconeMemory.

        Args:
            api_key (str): Pinecone API key.
            environment (str): Pinecone environment.
            index_name (str): Name of the Pinecone index to use.
            dimension (int): Dimension of the document embeddings. Defaults to 768.
            embedding_model (Optional[Any]): Custom embedding model. Defaults to None.
            embedding_function (Optional[Callable]): Custom embedding function. Defaults to None.
            preprocess_function (Optional[Callable]): Custom preprocessing function. Defaults to None.
            postprocess_function (Optional[Callable]): Custom postprocessing function. Defaults to None.
            metric (str): Distance metric for Pinecone index. Defaults to 'cosine'.
            pod_type (str): Pinecone pod type. Defaults to 'p1'.
            namespace (str): Pinecone namespace. Defaults to ''.
            logger_config (Optional[Dict]): Configuration for the logger. Defaults to None.
        """
        super().__init__()
        self._setup_logger(logger_config)
        logger.info("Initializing PineconeMemory")

        pinecone.init(api_key=api_key, environment=environment)

        if index_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pinecone.create_index(
                index_name,
                dimension=dimension,
                metric=metric,
                pod_type=pod_type,
            )

        self.index = pinecone.Index(index_name)
        self.namespace = namespace

        self.embedding_model = embedding_model or SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.embedding_function = (
            embedding_function or self._default_embedding_function
        )
        self.preprocess_function = (
            preprocess_function or self._default_preprocess_function
        )
        self.postprocess_function = (
            postprocess_function or self._default_postprocess_function
        )

        logger.info("PineconeMemory initialized successfully")

    def _setup_logger(self, config: Optional[Dict[str, Any]] = None):
        """Set up the logger with the given configuration."""
        default_config = {
            "handlers": [
                {"sink": "rag_wrapper.log", "rotation": "500 MB"},
                {"sink": lambda msg: print(msg, end="")},
            ],
        }
        logger.configure(**(config or default_config))

    def _default_embedding_function(self, text: str) -> List[float]:
        """Default embedding function using the SentenceTransformer model."""
        return self.embedding_model.encode(text).tolist()

    def _default_preprocess_function(self, text: str) -> str:
        """Default preprocessing function."""
        return text.strip()

    def _default_postprocess_function(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Default postprocessing function."""
        return results

    def add(
        self, doc: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document to the Pinecone index.

        Args:
            doc (str): The document to be added.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document.

        Returns:
            None
        """
        logger.info(f"Adding document: {doc[:50]}...")
        processed_doc = self.preprocess_function(doc)
        embedding = self.embedding_function(processed_doc)
        id = str(abs(hash(doc)))
        metadata = metadata or {}
        metadata["text"] = processed_doc
        self.index.upsert(
            vectors=[(id, embedding, metadata)],
            namespace=self.namespace,
        )
        logger.success(f"Document added successfully with ID: {id}")

    def query(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the Pinecone index for similar documents.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return. Defaults to 5.
            filter (Optional[Dict[str, Any]]): Metadata filter for the query.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the top_k most similar documents.
        """
        logger.info(f"Querying with: {query}")
        processed_query = self.preprocess_function(query)
        query_embedding = self.embedding_function(processed_query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter,
        )

        formatted_results = []
        for match in results.matches:
            formatted_results.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                }
            )

        processed_results = self.postprocess_function(
            formatted_results
        )
        logger.success(
            f"Query completed. Found {len(processed_results)} results."
        )
        return processed_results
