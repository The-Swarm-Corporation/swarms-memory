"""
Qdrant Vector Database Wrapper Example
"""

import os
from qdrant_client import QdrantClient, models
from swarms_memory.vector_dbs import QdrantDB


def main():
    """Main function demonstrating QdrantDB usage."""

    # Example 1: Basic usage in-memory (for experimentation)
    print("=== Example 1: In-memory QdrantDB ===")

    # Create in-memory client
    in_memory_client = QdrantClient(":memory:")

    qdrant_memory = QdrantDB(
        client=in_memory_client,
        collection_name="demo_collection",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        distance=models.Distance.COSINE,
        n_results=3,
    )

    # Add some documents
    documents = [
        "Qdrant has Langchain integrations",
        "Qdrant also has Llama Index integrations",
        "Qdrant is a vector similarity search engine",
        "Qdrant supports multiple distance metrics like cosine and"
        " euclidean",
        "Qdrant can be deployed as a cloud service or self-hosted",
    ]

    for i, doc in enumerate(documents):
        doc_id = qdrant_memory.add(doc)
        print(f"Added document {i + 1} with ID: {doc_id}")

    # Query the documents
    query = "What integrations does Qdrant have?"
    results = qdrant_memory.query(query)
    print(f"\nQuery: {query}")
    print(f"Results:\n{results}")

    # Example 2: Production usage with local Qdrant server
    print("\n=== Example 2: Local Qdrant Server ===")

    # Note: This requires a running Qdrant server
    # You can start one with: docker run -p 6333:6333 qdrant/qdrant
    try:
        production_client = QdrantClient("localhost", port=6333)

        qdrant_prod = QdrantDB(
            client=production_client,
            collection_name="production_collection",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            distance=models.Distance.COSINE,
            n_results=1,
        )

        # Add a document
        doc_id = qdrant_prod.add("This is a production document")
        print(f"Added production document with ID: {doc_id}")

        # Query
        results = qdrant_prod.query("production document")
        print(f"Query results: {results}")
    except Exception as e:
        print(f"Local server not available: {e}")
        print("Start a local Qdrant server with: docker run -p 6333:6333 qdrant/qdrant")

    # Example 3: Qdrant Cloud usage
    print("\n=== Example 3: Qdrant Cloud ===")
    
    # Use environment variables for secure credential handling
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if qdrant_url and qdrant_api_key:
        # Initialize Qdrant Cloud client
        cloud_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Example with OpenAI embeddings
        cloud_db = QdrantDB(
            client=cloud_client,
            embedding_model="text-embedding-3-small",
            collection_name="cloud_collection",
            distance=models.Distance.COSINE,
            n_results=3,
        )
        
        # Add documents
        cloud_docs = [
            "Qdrant Cloud provides managed vector database hosting.",
            "Vector search enables semantic similarity matching.",
            "Qdrant supports multiple embedding models and distances.",
        ]
        
        for doc in cloud_docs:
            cloud_db.add(doc)
        
        # Query
        results = cloud_db.query("What is Qdrant Cloud?")
        print(f"Cloud query results: {results}")
    else:
        print("Qdrant Cloud credentials not found.")
        print("Set QDRANT_URL and QDRANT_API_KEY environment variables to use cloud features.")


if __name__ == "__main__":
    main()
