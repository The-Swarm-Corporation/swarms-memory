from typing import List, Dict, Any
import pinecone
from loguru import logger
from sentence_transformers import SentenceTransformer
from swarms.memory.base_vectordb import BaseVectorDatabase

class PineconeMemory(BaseVectorDatabase):
    """
    A wrapper class for Pinecone-based Retrieval-Augmented Generation (RAG) system.
    
    This class provides methods to add documents to the Pinecone index and query the index
    for similar documents.
    """

    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int = 768):
        """
        Initialize the PineconeMemory.

        Args:
            api_key (str): Pinecone API key.
            environment (str): Pinecone environment.
            index_name (str): Name of the Pinecone index to use.
            dimension (int): Dimension of the document embeddings. Defaults to 768.
        """
        logger.info("Initializing PineconeMemory")
        pinecone.init(api_key=api_key, environment=environment)
        
        if index_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pinecone.create_index(index_name, dimension=dimension)
        
        self.index = pinecone.Index(index_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("PineconeMemory initialized successfully")

    def add(self, doc: str) -> None:
        """
        Add a document to the Pinecone index.

        Args:
            doc (str): The document to be added.

        Returns:
            None
        """
        logger.info(f"Adding document: {doc[:50]}...")
        embedding = self.model.encode(doc).tolist()
        id = str(abs(hash(doc)))
        self.index.upsert([(id, embedding, {"text": doc})])
        logger.success(f"Document added successfully with ID: {id}")

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the Pinecone index for similar documents.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the top_k most similar documents.
        """
        logger.info(f"Querying with: {query}")
        query_embedding = self.model.encode(query).tolist()
        results = self.index.query(query_embedding, top_k=top_k, include_metadata=True)
        
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata["text"]
            })
        
        logger.success(f"Query completed. Found {len(formatted_results)} results.")
        return formatted_results

# # Example usage
# if __name__ == "__main__":
#     logger.add("rag_wrapper.log", rotation="500 MB")
    
#     wrapper = PineconeMemory(
#         api_key="your-api-key",
#         environment="your-environment",
#         index_name="your-index-name"
#     )
    
#     # Adding documents
#     wrapper.add("This is a sample document about artificial intelligence.")
#     wrapper.add("Python is a popular programming language for data science.")
    
#     # Querying
#     results = wrapper.query("What is AI?")
#     for result in results:
#         print(f"Score: {result['score']}, Text: {result['text']}")