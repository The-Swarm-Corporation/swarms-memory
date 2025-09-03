# Modern Pinecone Wrapper Example
# This example demonstrates the modernized PineconeMemory class with LiteLLM embedding support

import os
from swarms_memory import PineconeMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    Demonstrate the modernized PineconeMemory usage.
    """
    
    # Modern initialization - no deprecated environment parameter
    pinecone_memory = PineconeMemory(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="swarms-modern-example",
        embedding_model="text-embedding-3-small",  # LiteLLM model
        dimension=1536,  # Will be auto-detected
        metric="cosine",
        cloud="aws",
        region="us-east-1",
        namespace="example",
        api_key_embedding=os.getenv("OPENAI_API_KEY")  # For embedding provider
    )
    
    print("🚀 Modern Pinecone Memory Example")
    print(f"Health Check: {pinecone_memory.health_check()}")
    
    # Sample documents to add
    documents = [
        {
            "text": "Pinecone is a vector database that makes it easy to add semantic search to applications.",
            "metadata": {"category": "database", "source": "documentation"}
        },
        {
            "text": "LiteLLM provides a unified interface for using different embedding models from various providers.",
            "metadata": {"category": "embeddings", "source": "documentation"}
        },
        {
            "text": "Vector databases store high-dimensional vectors and enable similarity search at scale.",
            "metadata": {"category": "concepts", "source": "explanation"}
        },
        {
            "text": "RAG systems combine retrieval and generation for more accurate AI responses.",
            "metadata": {"category": "architecture", "source": "explanation"}
        }
    ]
    
    # Add documents
    print("\nAdding documents...")
    doc_ids = []
    for doc in documents:
        doc_id = pinecone_memory.add(doc["text"], metadata=doc["metadata"])
        doc_ids.append(doc_id)
        print(f"Added: {doc['text'][:50]}... (ID: {doc_id})")
    
    # Query examples
    queries = [
        "What is Pinecone?",
        "How do embedding models work?",
        "What are vector databases?",
        "Tell me about RAG systems"
    ]
    
    print("\nQuerying documents...")
    for query in queries:
        print(f"\n❓ Query: {query}")
        results = pinecone_memory.query(query_text=query, top_k=2)
        
        for i, result in enumerate(results, 1):
            score = result["score"]
            text = result["metadata"]["text"]
            category = result["metadata"].get("category", "unknown")
            print(f"   {i}. [{category}] Score: {score:.3f}")
            print(f"      {text[:100]}...")
    
    # Demonstrate CRUD operations
    print("\n🛠️  CRUD Operations...")
    
    # Get a specific document
    first_doc_id = doc_ids[0]
    retrieved_doc = pinecone_memory.get(first_doc_id)
    if retrieved_doc:
        print(f"📖 Retrieved doc: {retrieved_doc['metadata']['text'][:50]}...")
    
    # Count documents
    doc_count = pinecone_memory.count()
    print(f"Total documents: {doc_count}")
    
    # Delete a document
    if len(doc_ids) > 2:
        delete_id = doc_ids[-1]
        success = pinecone_memory.delete(delete_id)
        if success:
            print(f"🗑️  Deleted document: {delete_id}")
            print(f"Documents after deletion: {pinecone_memory.count()}")
    
    # Health check after operations
    print(f"\n🏥 Final Health Check: {pinecone_memory.health_check()}")
    
    print("\nModern Pinecone example completed!")

if __name__ == "__main__":
    main()