from typing import List, Dict, Any, Callable, Optional
import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase


class FAISSDB(BaseVectorDatabase):
    """
    A highly customizable wrapper class for FAISS-based Retrieval-Augmented Generation (RAG) system.

    This class provides methods to add documents to the FAISS index and query the index
    for similar documents. It allows for custom embedding models, preprocessing functions,
    and other customizations.
    """

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "Flat",
        embedding_model: Optional[Any] = None,
        embedding_function: Optional[
            Callable[[str], List[float]]
        ] = None,
        preprocess_function: Optional[Callable[[str], str]] = None,
        postprocess_function: Optional[
            Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]
        ] = None,
        metric: str = "cosine",
        logger_config: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the FAISSDB.

        Args:
            dimension (int): Dimension of the document embeddings. Defaults to 768.
            index_type (str): Type of FAISS index to use. Defaults to 'Flat'.
            embedding_model (Optional[Any]): Custom embedding model. Defaults to None.
            embedding_function (Optional[Callable]): Custom embedding function. Defaults to None.
            preprocess_function (Optional[Callable]): Custom preprocessing function. Defaults to None.
            postprocess_function (Optional[Callable]): Custom postprocessing function. Defaults to None.
            metric (str): Distance metric for FAISS index. Defaults to 'cosine'.
            logger_config (Optional[Dict]): Configuration for the logger. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self._setup_logger(logger_config)
        logger.info("Initializing FAISSDB")

        self.dimension = dimension
        self.index = self._create_index(index_type, metric)
        self.documents = []

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

        logger.info("FAISSDB initialized successfully")

    def _setup_logger(self, config: Optional[Dict[str, Any]] = None):
        """Set up the logger with the given configuration."""
        default_config = {
            "handlers": [
                {
                    "sink": "faiss_rag_wrapper.log",
                    "rotation": "500 MB",
                },
                {"sink": lambda msg: print(msg, end="")},
            ],
        }
        logger.configure(**(config or default_config))

    def _create_index(self, index_type: str, metric: str):
        """Create and return a FAISS index based on the specified type and metric."""
        if metric == "cosine":
            index = faiss.IndexFlatIP(self.dimension)
        elif metric == "l2":
            index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if index_type == "Flat":
            return index
        elif index_type == "IVF":
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist
            )
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        return index

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
        Add a document to the FAISS index.

        Args:
            doc (str): The document to be added.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document.

        Returns:
            None
        """
        logger.info(f"Adding document: {doc[:50]}...")
        processed_doc = self.preprocess_function(doc)
        embedding = self.embedding_function(processed_doc)

        self.index.add(np.array([embedding], dtype=np.float32))
        metadata = metadata or {}
        metadata["text"] = processed_doc
        self.documents.append(metadata)

        logger.success(
            f"Document added successfully. Total documents: {len(self.documents)}"
        )

    def query(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the FAISS index for similar documents.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the top_k most similar documents.
        """
        logger.info(f"Querying with: {query}")
        processed_query = self.preprocess_function(query)
        query_embedding = self.embedding_function(processed_query)

        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), top_k
        )

        results = []
        for i, (distance, idx) in enumerate(
            zip(distances[0], indices[0])
        ):
            if idx != -1:  # FAISS uses -1 for empty slots
                result = {
                    "id": idx,
                    "score": 1
                    - distance,  # Convert distance to similarity score
                    "metadata": self.documents[idx],
                }
                results.append(result)

        processed_results = self.postprocess_function(results)
        logger.success(
            f"Query completed. Found {len(processed_results)} results."
        )
        return processed_results


# # Example usage
# if __name__ == "__main__":
#     from transformers import AutoTokenizer, AutoModel
#     import torch

#     # Custom embedding function using a HuggingFace model
#     def custom_embedding_function(text: str) -> List[float]:
#         tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         model = AutoModel.from_pretrained("bert-base-uncased")
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512,
#         )
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings = (
#             outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
#         )
#         return embeddings

#     # Custom preprocessing function
#     def custom_preprocess(text: str) -> str:
#         return text.lower().strip()

#     # Custom postprocessing function
#     def custom_postprocess(
#         results: List[Dict[str, Any]],
#     ) -> List[Dict[str, Any]]:
#         for result in results:
#             result["custom_score"] = (
#                 result["score"] * 2
#             )  # Example modification
#         return results

#     # Initialize the wrapper with custom functions
#     wrapper = FAISSDB(
#         dimension=768,
#         index_type="Flat",
#         embedding_function=custom_embedding_function,
#         preprocess_function=custom_preprocess,
#         postprocess_function=custom_postprocess,
#         metric="cosine",
#         logger_config={
#             "handlers": [
#                 {
#                     "sink": "custom_faiss_rag_wrapper.log",
#                     "rotation": "1 GB",
#                 },
#                 {
#                     "sink": lambda msg: print(
#                         f"Custom log: {msg}", end=""
#                     )
#                 },
#             ],
#         },
#     )

#     # Adding documents
#     wrapper.add(
#         "This is a sample document about artificial intelligence.",
#         {"category": "AI"},
#     )
#     wrapper.add(
#         "Python is a popular programming language for data science.",
#         {"category": "Programming"},
#     )

#     # Querying
#     results = wrapper.query("What is AI?")
#     for result in results:
#         print(
#             f"Score: {result['score']}, Custom Score: {result['custom_score']}, Text: {result['metadata']['text']}"
#         )
