# Example usage of the PineconeMemory class

# Import necessary libraries
import os
from swarms_memory import PineconeMemory

# Define your Pinecone API key and environment
API_KEY = os.getenv(
    "PINECONE_API_KEY"
)  # Make sure to set your Pinecone API key in the environment
ENVIRONMENT = "us-west1-gcp"  # Example environment
INDEX_NAME = "my-pinecone-index"  # Name of the Pinecone index to use

# Create an instance of the PineconeMemory class
pinecone_memory = PineconeMemory(
    api_key=API_KEY,
    environment=ENVIRONMENT,
    index_name=INDEX_NAME,
    dimension=768,  # Dimension of the embeddings
)


# Function to add documents to the Pinecone index
def add_documents():
    """Add sample documents to the Pinecone index."""
    documents = [
        "This is the first document about machine learning.",
        "This document discusses deep learning techniques.",
        "Here we talk about natural language processing.",
        "This is another document related to artificial intelligence.",
        "This document covers supervised and unsupervised learning.",
    ]

    for doc in documents:
        # Adding documents to the Pinecone index with optional metadata
        pinecone_memory.add(doc, metadata={"source": "example"})


# Function to query documents from the Pinecone index
def query_documents(query: str):
    """Query the Pinecone index for similar documents."""
    results = pinecone_memory.query(
        query, top_k=3
    )  # Retrieve the top 3 similar documents
    print("\nQuery Results:")
    for result in results:
        print(
            f"ID: {result['id']}, Score: {result['score']}, Metadata: {result['metadata']}"
        )


# Adding documents to the Pinecone index
add_documents()

# Querying the Pinecone index
query_documents("Tell me about machine learning techniques.")

# Close the Pinecone index (if needed, cleanup)
# Note: Pinecone does not require a close method as it is a managed service.
