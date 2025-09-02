import numpy as np
import torch
from typing import List, Tuple
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from concurrent.futures import ProcessPoolExecutor


class SparseEmbedder:
    """
    Generates sparse embeddings for text chunks using a simple word count method.

    This class provides a fast, CPU-friendly method for creating sparse embeddings.
    It uses a hash function to map words to indices in a fixed-size vector.

    Attributes:
        vocab_size (int): The size of the vocabulary (embedding dimension).
    """

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a text string into a sparse vector.

        Args:
            text (str): The input text to embed.

        Returns:
            np.ndarray: A sparse embedding vector.
        """
        words = text.lower().split()
        embedding = np.zeros(self.vocab_size, dtype=np.float32)
        for word in words:
            hash_value = hash(word) % self.vocab_size
            embedding[hash_value] += 1
        return embedding / (np.linalg.norm(embedding) + 1e-8)


class GraphBuilder:
    """
    Builds a graph representation of document chunks based on embedding similarities.

    This class constructs an adjacency matrix representing the similarity between
    document chunks.

    Attributes:
        similarity_threshold (float): The minimum similarity for an edge to be created.
    """

    def __init__(self, similarity_threshold: float = 0.1):
        self.similarity_threshold = similarity_threshold

    def build_graph(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Build a graph from a list of embeddings.

        Args:
            embeddings (List[np.ndarray]): List of embedding vectors.

        Returns:
            np.ndarray: An adjacency matrix representing the graph.
        """
        num_nodes = len(embeddings)
        embeddings_matrix = np.vstack(embeddings)

        # Compute pairwise similarities efficiently
        similarities = np.dot(embeddings_matrix, embeddings_matrix.T)

        # Create adjacency matrix
        adjacency_matrix = np.where(
            similarities > self.similarity_threshold, similarities, 0
        )
        np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops

        return adjacency_matrix


class PPRRetriever:
    """
    Implements the Personalized PageRank algorithm for retrieving relevant chunks.

    This class performs a fast approximation of Personalized PageRank to identify
    the most relevant document chunks given a query.

    Attributes:
        alpha (float): The damping factor in the PageRank algorithm.
        num_iterations (int): The maximum number of iterations for the algorithm.
    """

    def __init__(self, alpha: float = 0.85, num_iterations: int = 20):
        self.alpha = alpha
        self.num_iterations = num_iterations

    def retrieve(
        self,
        query_embedding: np.ndarray,
        graph: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[int]:
        """
        Retrieve the most relevant document chunks for a given query.

        Args:
            query_embedding (np.ndarray): The embedding of the query.
            graph (np.ndarray): The adjacency matrix representing the document graph.
            embeddings (np.ndarray): The embeddings of all document chunks.
            top_k (int): The number of chunks to retrieve.

        Returns:
            List[int]: Indices of the top-k most relevant chunks.
        """
        num_nodes = graph.shape[0]
        personalization = np.dot(query_embedding, embeddings.T)
        
        # Prevent division by zero
        personalization_sum = personalization.sum()
        if np.abs(personalization_sum) < 1e-10 or np.isnan(personalization_sum):
            # If all similarities are effectively zero, use uniform distribution
            personalization = np.ones(len(personalization)) / len(personalization)
        else:
            # Normalize to create a proper probability distribution
            personalization = np.abs(personalization)  # Take absolute values to handle negative similarities
            personalization_sum = personalization.sum()
            if personalization_sum > 0:
                personalization = personalization / personalization_sum
            else:
                personalization = np.ones(len(personalization)) / len(personalization)

        scores = personalization.copy()
        for _ in range(self.num_iterations):
            new_scores = (
                1 - self.alpha
            ) * personalization + self.alpha * (graph @ scores)
            if np.allclose(new_scores, scores):
                break
            scores = new_scores

        return np.argsort(scores)[-top_k:][::-1].tolist()


class RAGSystem:
    """
    Retrieval-Augmented Generation (RAG) system for processing documents and answering queries.

    This class combines sparse embedding, graph-based retrieval, and a language model
    to provide context-aware answers to queries based on a large document.

    Attributes:
        embedder (SparseEmbedder): The embedding system for creating sparse representations.
        graph_builder (GraphBuilder): The system for building a graph from embeddings.
        retriever (PPRRetriever): The retrieval system for finding relevant chunks.
        llm (torch.nn.Module): The language model for generating answers.
    """

    def __init__(self, llm: torch.nn.Module, vocab_size: int = 10000):
        self.embedder = SparseEmbedder(vocab_size)
        self.graph_builder = GraphBuilder()
        self.retriever = PPRRetriever()
        self.llm = llm

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
    )
    def process_document(
        self, document: str, chunk_size: int = 100
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Process a document by chunking, embedding, and building a graph.

        Args:
            document (str): The input document to process.
            chunk_size (int): The size of each document chunk.

        Returns:
            Tuple[List[str], np.ndarray, np.ndarray]: Chunks, embeddings, and graph.
        """
        logger.info("Processing document")
        chunks = [
            document[i : i + chunk_size]
            for i in range(0, len(document), chunk_size)
        ]

        # Parallel embedding
        with ProcessPoolExecutor() as executor:
            embeddings = list(
                executor.map(self.embedder.embed, chunks)
            )

        embeddings = np.vstack(embeddings)
        graph = self.graph_builder.build_graph(embeddings)
        return chunks, embeddings, graph

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
    )
    def answer_query(
        self,
        query: str,
        chunks: List[str],
        embeddings: np.ndarray,
        graph: np.ndarray,
    ) -> str:
        """
        Answer a query based on the processed document.

        Args:
            query (str): The query to answer.
            chunks (List[str]): The document chunks.
            embeddings (np.ndarray): The embeddings of all chunks.
            graph (np.ndarray): The graph representation of the document.

        Returns:
            str: The generated answer to the query.
        """
        logger.info(f"Answering query: {query}")
        query_embedding = self.embedder.embed(query)
        retrieved_indices = self.retriever.retrieve(
            query_embedding, graph, embeddings
        )
        context = " ".join([chunks[i] for i in retrieved_indices])

        # Simplified LLM usage (replace with actual LLM integration)
        answer = self.llm(f"Query: {query}\nContext: {context}")
        return answer


# Example usage
if __name__ == "__main__":
    # Dummy LLM for illustration
    class DummyLLM(torch.nn.Module):
        def forward(self, x):
            return f"Answer based on: {x[:100]}..."

    llm = DummyLLM()
    rag_system = RAGSystem(llm)

    document = (
        "Long document text..." * 1000
    )  # Simulating a very long document
    chunks, embeddings, graph = rag_system.process_document(document)

    query = "What is the main topic of the document?"
    answer = rag_system.answer_query(query, chunks, embeddings, graph)
    print(f"Answer: {answer}")
