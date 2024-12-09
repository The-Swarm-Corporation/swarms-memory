import os
from dotenv import load_dotenv
from swarms_memory.vector_dbs.singlestore_wrapper import SingleStoreDB

# Load environment variables
load_dotenv()

def main():
    # Initialize SingleStore with environment variables
    db = SingleStoreDB(
        host=os.getenv("SINGLESTORE_HOST"),
        port=int(os.getenv("SINGLESTORE_PORT", "3306")),
        user=os.getenv("SINGLESTORE_USER"),
        password=os.getenv("SINGLESTORE_PASSWORD"),
        database=os.getenv("SINGLESTORE_DATABASE"),
        table_name="example_vectors",
        dimension=768,  # Default dimension for all-MiniLM-L6-v2
        namespace="example"
    )

    # Example documents
    documents = [
        "SingleStore is a distributed SQL database that combines the horizontal scalability of NoSQL systems with the ACID guarantees of traditional RDBMSs.",
        "Vector similarity search in SingleStore uses DOT_PRODUCT distance type for efficient nearest neighbor queries.",
        "SingleStore supports both row and column store formats, making it suitable for both transactional and analytical workloads."
    ]

    # Add documents to the database
    doc_ids = []
    for doc in documents:
        doc_id = db.add(
            document=doc,
            metadata={"source": "example", "type": "documentation"}
        )
        doc_ids.append(doc_id)
        print(f"Added document with ID: {doc_id}")

    # Query similar documents
    query = "How does SingleStore handle vector similarity search?"
    results = db.query(
        query=query,
        top_k=2,
        metadata_filter={"source": "example"}
    )

    print("\nQuery:", query)
    print("\nResults:")
    for result in results:
        print(f"\nDocument: {result['document']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Metadata: {result['metadata']}")

    # Clean up - delete documents
    print("\nCleaning up...")
    for doc_id in doc_ids:
        db.delete(doc_id)
        print(f"Deleted document with ID: {doc_id}")

if __name__ == "__main__":
    main()
