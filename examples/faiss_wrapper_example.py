# Modern FAISS Wrapper Example
# This example demonstrates the modernized FAISSDB class with LiteLLM embedding support

import os
from swarms_memory import FAISSDB
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """
    Demonstrate the modernized FAISSDB usage.
    """
    
    print("🚀 Modern FAISS DB Example")
    
    # Modern initialization with LiteLLM embeddings
    faiss_db = FAISSDB(
        embedding_model="text-embedding-3-small",  # LiteLLM model
        dimension=1536,  # Will be auto-detected
        index_type="Flat",  # Can also use "IVF" or "HNSW"
        metric="cosine",
        index_file="example_faiss_index.bin",  # Persist to disk
        api_key_embedding=os.getenv("OPENAI_API_KEY")
    )
    
    print(f"📊 Health Check: {faiss_db.health_check()}")
    
    # Sample documents with metadata
    documents = [
        {
            "text": "FAISS is a library for efficient similarity search and clustering of dense vectors, developed by Facebook AI Research.",
            "metadata": {"category": "library", "source": "documentation", "technology": "faiss"}
        },
        {
            "text": "Machine learning algorithms can be supervised, unsupervised, or reinforcement-based depending on the learning approach.",
            "metadata": {"category": "concepts", "source": "textbook", "technology": "ml"}
        },
        {
            "text": "Deep learning neural networks use multiple layers to learn hierarchical representations of data.",
            "metadata": {"category": "concepts", "source": "research", "technology": "deep-learning"}
        },
        {
            "text": "Natural language processing involves computational techniques for analyzing and understanding human language.",
            "metadata": {"category": "field", "source": "textbook", "technology": "nlp"}
        },
        {
            "text": "Vector databases enable semantic search by storing high-dimensional embeddings and performing similarity queries.",
            "metadata": {"category": "database", "source": "documentation", "technology": "vector-db"}
        }
    ]
    
    # Add documents
    print("\n📚 Adding documents...")
    doc_ids = []
    for doc in documents:
        doc_id = faiss_db.add(doc["text"], metadata=doc["metadata"])
        doc_ids.append(doc_id)
        print(f"✅ Added: {doc['text'][:50]}... (ID: {doc_id})")
    
    # Query examples
    queries = [
        "What is FAISS?",
        "machine learning techniques",
        "neural networks and deep learning",
        "language processing",
        "vector database search"
    ]
    
    print("\n🔍 Querying documents...")
    for query in queries:
        print(f"\n❓ Query: {query}")
        results = faiss_db.query(query_text=query, top_k=2)
        
        for i, result in enumerate(results, 1):
            score = result["score"]
            text = result["metadata"]["text"]
            category = result["metadata"].get("category", "unknown")
            technology = result["metadata"].get("technology", "unknown")
            print(f"   {i}. [{category}/{technology}] Score: {score:.3f}")
            print(f"      {text[:80]}...")
    
    # Demonstrate filtering
    print("\n🔎 Filtered search (category='concepts')...")
    filter_results = faiss_db.query(
        query_text="learning algorithms", 
        top_k=5,
        filter_dict={"category": "concepts"}
    )
    
    for i, result in enumerate(filter_results, 1):
        text = result["metadata"]["text"]
        print(f"   {i}. {text[:60]}...")
    
    # CRUD operations
    print("\n🛠️  CRUD Operations...")
    
    # Get a specific document
    first_doc_id = doc_ids[0]
    retrieved_doc = faiss_db.get(first_doc_id)
    if retrieved_doc:
        print(f"📖 Retrieved doc: {retrieved_doc['metadata']['text'][:50]}...")
    
    # Count documents
    doc_count = faiss_db.count()
    print(f"📊 Total documents: {doc_count}")
    
    # Delete a document
    if len(doc_ids) > 2:
        delete_id = doc_ids[-1]
        success = faiss_db.delete(delete_id)
        if success:
            print(f"🗑️  Deleted document: {delete_id}")
            print(f"📊 Documents after deletion: {faiss_db.count()}")
    
    # Rebuild index for better performance
    print("\n🔄 Rebuilding index...")
    faiss_db.rebuild_index()
    
    # Final health check
    final_health = faiss_db.health_check()
    print(f"\n🏥 Final Health Check:")
    for key, value in final_health.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Example completed! Index saved to: {faiss_db.index_file}")

def custom_embedding_example():
    """
    Demonstrate using custom embedding functions with FAISS.
    """
    print("\n🔧 Custom Embedding Function Example")
    
    # Simple custom embedding function (for demonstration)
    def custom_embedder(text: str):
        # This is just a demo - use proper embeddings in production
        import hashlib
        import numpy as np
        
        # Create a simple hash-based embedding
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        return np.random.normal(0, 1, 384).tolist()
    
    # Initialize with custom function
    custom_faiss = FAISSDB(
        embedding_model=custom_embedder,
        dimension=384,
        index_type="Flat",
        metric="cosine"
    )
    
    # Add some documents
    custom_faiss.add("Custom embedding example document", {"type": "demo"})
    custom_faiss.add("Another test document", {"type": "demo"})
    
    # Query
    results = custom_faiss.query("test document", top_k=1)
    print(f"Custom embedding result: {results[0]['metadata']['text']}")

if __name__ == "__main__":
    main()
    custom_embedding_example()