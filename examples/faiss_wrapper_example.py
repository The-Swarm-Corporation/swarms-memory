# Example usage of the FAISSDB class
from swarms_memory import FAISSDB


# Create an instance of the FAISSDB class
faiss_db = FAISSDB(
    dimension=768,  # Dimension of the embeddings
    index_type="Flat",  # Type of FAISS index to use
)


# Function to add documents to the FAISS index
def add_documents():
    """Add sample documents to the FAISS index."""
    documents = [
        "This is the first document about machine learning.",
        "This document discusses deep learning techniques.",
        "Here we talk about natural language processing.",
        "This is another document related to artificial intelligence.",
        "This document covers supervised and unsupervised learning.",
    ]

    for doc in documents:
        # Adding documents to the FAISS index with optional metadata
        faiss_db.add(doc, metadata={"source": "example"})


# Function to query documents from the FAISS index
def query_documents(query: str):
    """Query the FAISS index for similar documents."""
    results = faiss_db.query(
        query, top_k=3
    )  # Retrieve the top 3 similar documents
    print("\nQuery Results:")
    for result in results:
        print(
            f"ID: {result['id']}, Score: {result['score']}, Metadata: {result['metadata']}"
        )


# Adding documents to the FAISS index
add_documents()

# Querying the FAISS index
query_documents("Tell me about machine learning techniques.")

# The FAISS index will now contain the added documents, and the query will return the most relevant ones based on the embeddings.
