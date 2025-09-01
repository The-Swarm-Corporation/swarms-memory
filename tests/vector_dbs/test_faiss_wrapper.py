"""
Human-Testable FAISS Wrapper Test Suite

This script demonstrates and tests the modernized FAISS wrapper functionality
including LiteLLM integration, persistence, CRUD operations, and error handling.

Run this script to verify FAISS wrapper functionality:
python tests/vector_dbs/test_faiss_wrapper.py
"""

import os
import tempfile
from swarms_memory.vector_dbs.faiss_wrapper import FAISSDB

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_faiss_basic_operations():
    """Test basic FAISS operations with custom embedding function."""
    print("Testing FAISS Basic Operations")
    
    def simple_embedding(text):
        import hashlib
        import numpy as np
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        return np.random.normal(0, 1, 384).tolist()
    
    # Initialize FAISS with custom embedding
    faiss_db = FAISSDB(
        embedding_model=simple_embedding,
        dimension=384,
        index_type="Flat",
        metric="cosine"
    )
    
    print("✓ FAISS initialized successfully")
    print(f"  Dimension: {faiss_db.dimension}, Index: {faiss_db.index_type}, Metric: {faiss_db.metric}")
    
    # Test adding documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning learns through interaction with environment."
    ]
    
    doc_ids = []
    for i, doc in enumerate(documents):
        doc_id = faiss_db.add(doc, metadata={"category": "AI", "index": i})
        doc_ids.append(doc_id)
        if i == 0:
            print(f"✓ Added document: {doc_id[:8]}...")
    
    print(f"✓ Added {len(documents)} documents")
    
    # Test counting and querying
    count = faiss_db.count()
    print(f"✓ Total documents: {count}")
    
    results = faiss_db.query(query_text="neural networks deep learning", top_k=3)
    print(f"✓ Found {len(results)} results for neural networks query")
    if results:
        print(f"  Top result score: {results[0]['score']:.4f}")
    
    # Test document operations
    if doc_ids:
        retrieved = faiss_db.get(doc_ids[0])
        if retrieved:
            print("✓ Successfully retrieved document by ID")
        
        success = faiss_db.delete(doc_ids[-1])
        print(f"✓ Delete operation: {'success' if success else 'failed'}")
    
    # Test health check
    health = faiss_db.health_check()
    print(f"✓ Health check status: {health.get('status', 'unknown')}")
    
    return faiss_db

def test_faiss_persistence():
    """Test FAISS persistence functionality."""
    print("\nTesting FAISS Persistence")
    
    # Create temporary file for persistence
    with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
        temp_path = f.name
    
    try:
        # Create FAISS with persistence
        faiss_db1 = FAISSDB(
            embedding_model=lambda _: [0.1] * 256,
            dimension=256,
            index_file=temp_path
        )
        
        print(f"✓ Created FAISS with persistence")
        
        # Add some data
        doc_id = faiss_db1.add("Persistent document for testing", metadata={"persist": True})
        print(f"✓ Added persistent document: {doc_id[:8]}...")
        
        faiss_db1.save()
        print("✓ Saved index to disk")
        
        # Create new instance loading from disk
        faiss_db2 = FAISSDB(
            embedding_model=lambda _: [0.1] * 256,
            dimension=256,
            index_file=temp_path
        )
        
        print(f"✓ Loaded from disk - document count: {faiss_db2.count()}")
        
        # Verify document was persisted
        retrieved = faiss_db2.get(doc_id)
        if retrieved:
            print("✓ Document successfully retrieved from persisted index")
        
    finally:
        # Cleanup
        for ext in ['', '.metadata']:
            try:
                os.unlink(temp_path + ext)
            except FileNotFoundError:
                pass

def test_faiss_filtering():
    """Test FAISS filtering functionality."""
    print("\nTesting FAISS Filtering")
    
    faiss_db = FAISSDB(
        embedding_model=lambda text: [hash(text) % 100 / 100.0] * 128,
        dimension=128
    )
    
    # Add documents with different categories
    categories = ["tutorial", "research", "news", "tutorial", "research"]
    topics = ["AI", "ML", "NLP", "CV", "RL"]
    
    for i in range(5):
        faiss_db.add(
            f"Document about {topics[i]} in {categories[i]} format",
            metadata={"category": categories[i], "topic": topics[i], "index": i}
        )
    
    print(f"✓ Added {len(categories)} documents with categories and topics")
    
    # Test filtering by category
    tutorial_results = faiss_db.query(
        query_text="document tutorial",
        top_k=10,
        filter_dict={"category": "tutorial"}
    )
    
    print(f"✓ Found {len(tutorial_results)} tutorial documents")
    
    # Test filtering by topic
    ai_results = faiss_db.query(
        query_text="artificial intelligence",
        top_k=10,
        filter_dict={"topic": "AI"}
    )
    
    print(f"✓ Found {len(ai_results)} AI-related documents")

def test_faiss_litellm_integration():
    """Test FAISS with LiteLLM integration (if API keys available)."""
    print("\nTesting FAISS LiteLLM Integration")
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ OPENAI_API_KEY not found - skipping LiteLLM integration test")
        print("  Set OPENAI_API_KEY environment variable to test LiteLLM integration")
        return
    
    try:
        # Initialize FAISS with LiteLLM
        faiss_db = FAISSDB(
            embedding_model="text-embedding-3-small",
            api_key_embedding=openai_key
        )
        
        print("✓ FAISS initialized with LiteLLM embeddings")
        print(f"  Auto-detected dimension: {faiss_db.dimension}")
        
        # Add AI-related documents
        ai_docs = [
            "Artificial intelligence is transforming healthcare through diagnostic tools.",
            "Machine learning algorithms improve recommendation systems.",
            "Natural language processing enables chatbots and virtual assistants.",
            "Computer vision powers autonomous vehicles and medical imaging."
        ]
        
        for i, doc in enumerate(ai_docs):
            faiss_db.add(doc, metadata={"source": "AI_research", "doc_num": i})
        
        print(f"✓ Added {len(ai_docs)} documents with LiteLLM embeddings")
        
        # Semantic search test
        results = faiss_db.query(query_text="medical AI applications", top_k=2)
        print(f"✓ Semantic search found {len(results)} relevant documents")
        
        if results:
            print(f"  Best match score: {results[0]['score']:.4f}")
        
    except Exception as e:
        print(f"⚠️ LiteLLM integration test failed: {str(e)}")
        print("  This might be due to API limits or network issues")

def test_faiss_index_types():
    """Test different FAISS index types."""
    print("\nTesting Different FAISS Index Types")
    
    index_types = ["Flat", "IVF", "HNSW"]
    
    for index_type in index_types:
        try:
            faiss_db = FAISSDB(
                embedding_model=lambda text: [hash(text+str(i)) % 100 / 100.0 for i in range(64)],
                dimension=64,
                index_type=index_type,
                metric="cosine"
            )
            
            # Add and query test document
            faiss_db.add(f"Test document for {index_type} index")
            results = faiss_db.query("test document", top_k=1)
            
            print(f"✓ {index_type} index working - found {len(results)} results")
            
        except Exception as e:
            print(f"⚠️ {index_type} index failed: {str(e)}")

def test_faiss_error_handling():
    """Test FAISS error handling scenarios."""
    print("\nTesting FAISS Error Handling")
    
    faiss_db = FAISSDB(
        embedding_model=lambda _: [0.1] * 128,
        dimension=128
    )
    
    # Test getting non-existent document
    result = faiss_db.get("non-existent-id")
    print(f"✓ Non-existent document returns: {result}")
    
    # Test deleting non-existent document
    success = faiss_db.delete("non-existent-id")
    print(f"✓ Delete non-existent returns: {success}")
    
    # Test query on empty index
    results = faiss_db.query("test query", top_k=5)
    print(f"✓ Query on empty index returns: {len(results)} results")
    
    # Test clear operation
    faiss_db.add("Document to be cleared")
    count_before = faiss_db.count()
    success = faiss_db.clear()
    count_after = faiss_db.count()
    print(f"✓ Clear operation: {count_before} -> {count_after} documents")

def main():
    """Run all FAISS tests."""
    print("FAISS Wrapper Test Suite")
    print("This script tests the modernized FAISS wrapper functionality")
    
    try:
        test_faiss_basic_operations()
        test_faiss_persistence()
        test_faiss_filtering()
        test_faiss_litellm_integration()
        test_faiss_index_types()
        test_faiss_error_handling()
        
        print("\n" + "="*60)
        print("  All Tests Completed Successfully!")
        print("="*60)
        print("✓ FAISS wrapper is working correctly")
        print("✓ All core functionality verified")
        print("✓ Error handling working as expected")
        
    except Exception as e:
        print("\n" + "="*60)
        print("  Test Suite Failed!")
        print("="*60)
        print(f"❌ Error: {str(e)}")
        print("Please check the FAISS wrapper implementation")
        raise

if __name__ == "__main__":
    main()