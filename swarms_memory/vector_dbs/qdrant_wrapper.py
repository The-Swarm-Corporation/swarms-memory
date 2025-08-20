import uuid

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from loguru import logger
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase

# Load environment variables
load_dotenv()


class QdrantDB(BaseVectorDatabase):
    """
    A memory vector database wrapper for [Qdrant](https://qdrant.tech/).

    Args:
        client (QdrantClient): A pre-configured Qdrant client instance for
            database connection.
        collection_name (str, optional): The name of the collection to store
            vectors in.
            Defaults to "swarms_collection".
        model_name (str, optional): The embedding model to use for text
            vectorization.
            Defaults to "sentence-transformers/all-MiniLM-L6-v2".
        distance (models.Distance, optional): The distance metric to use for
            similarity search.
            Defaults to models.Distance.COSINE.
        n_results (int, optional): The number of results to retrieve during
            similarity search.
            Defaults to 10.
        *args: Additional positional arguments passed to the parent
            class.
        **kwargs: Additional keyword arguments passed to the parent
            class.

    Examples:
        >>> from qdrant_client import QdrantClient
        >>>
        >>> # Initialize Qdrant client
        >>> client = QdrantClient("localhost", port=6333)
        >>>
        >>> # Create QdrantDB instance
        >>> qdrant_db = QdrantDB(
        >>>     client=client,
        >>>     collection_name="my_collection",
        >>>     model_name=(
        >>>         "sentence-transformers/all-MiniLM-L6-v2"
        >>>     ),
        >>>     distance=models.Distance.COSINE,
        >>>     n_results=5
        >>> )
        >>>
        >>> # Add documents
        >>> doc_id = qdrant_db.add(
        >>>     "This is a sample document for testing"
        >>> )
        >>>
        >>> # Query similar documents
        >>> results = qdrant_db.query("sample query")
        >>> print(results)
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "swarms_collection",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance: models.Distance = models.Distance.COSINE,
        n_results: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.collection_name = collection_name
        self.model_name = model_name
        self.distance = distance
        self.n_results = n_results
        self.client = client

        # Get or create collection
        self._setup_collection()

    def _setup_collection(self):
        """Set up the Qdrant collection with proper configuration."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.client.get_embedding_size(
                        self.model_name
                    ),
                    distance=self.distance,
                ),
            )
            logger.info(
                f"Created Qdrant collection: {self.collection_name} "
            )
        else:
            logger.info(
                f"Using existing collection: {self.collection_name}"
            )

    def add(
        self,
        document: str,
        *args,
        **kwargs,
    ) -> str:
        """
        Add a document to the Qdrant collection.

        Args:
            document (str): The document to be added.

        Returns:
            str: The ID of the added document.
        """

        doc_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=doc_id,
                    vector=models.Document(
                        text=document, model=self.model_name
                    ),
                    payload={
                        "document": document,
                    },
                )
            ],
        )
        return doc_id

    def query(
        self,
        query_text: str,
        *args,
        **kwargs,
    ) -> str:
        """
        Query documents from the Qdrant collection.

        Args:
            query_text (str): The query string.

        Returns:
            str: The retrieved documents as a concatenated string.
        """
        query_doc = models.Document(
            text=query_text, model=self.model_name
        )

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_doc,
            limit=self.n_results,
            with_payload=True,
            with_vectors=False,
        )

        docs = [
            point.payload.get("document", "")
            for point in search_result.points
        ]

        return "\n".join(docs)
