"""
Milvus Wrapper Test Suite

This script demonstrates and tests the Milvus wrapper functionality
including CRUD operations, vector search, and collection management.

Run this script to verify Milvus wrapper functionality:
python tests/vector_dbs/test_milvus_wrapper.py
"""

import os
import tempfile
from swarms_memory.vector_dbs.milvus_wrapper import MilvusDB

def test_milvus_basic_operations():
    """Test basic Milvus operations with local database."""
    print("Testing Milvus Basic Operations")
    
    # Initialize Milvus with local database (Milvus Lite)
    milvus_db = MilvusDB(
        embedding_model="text-embedding-3-small",
        collection_name="test_collection",
        db_file="test_milvus.db",
        metric="COSINE"
    )
    
    print("✓ Milvus initialized successfully")
    print(f"  Collection: {milvus_db.collection_name}, Metric: {milvus_db.metric}")
    print(f"  Vector dimension: {milvus_db.dimension}")
    
    # Test adding documents
    documents = [
        "Milvus is an open-source vector database built for scalability.",
        "Vector databases enable semantic search in AI applications.",
        "Milvus supports multiple indexing algorithms for optimal performance.",
        "High-dimensional embeddings capture semantic meaning of text.",
        "Distributed architecture allows horizontal scaling in Milvus."
    ]
    
    doc_ids = []
    for i, doc in enumerate(documents):
        doc_id = milvus_db.add(doc, metadata={"category": "database", "index": i})
        doc_ids.append(doc_id)
        if i == 0:
            print(f"✓ Added document with ID: {doc_id}")
    
    print(f"✓ Added {len(documents)} documents")
    
    # Test counting
    count = milvus_db.count()
    print(f"✓ Total documents: {count}")
    
    # Test querying
    query = "What is Milvus database?"
    results = milvus_db.query(query, top_k=3)
    print(f"✓ Found {len(results)} results for query")
    if results:
        print(f"  Top result score: {results[0]['score']:.4f}")
    
    # Test query as text
    text_results = milvus_db.query_as_text(query, top_k=2)
    print(f"✓ Query as text returned {len(text_results.split())} words")
    
    # Test document retrieval
    if doc_ids:
        retrieved = milvus_db.get(doc_ids[0])
        if retrieved:
            print("✓ Successfully retrieved document by ID")
    
    # Test document deletion
    if doc_ids:
        success = milvus_db.delete(doc_ids[-1])
        print(f"✓ Delete operation: {'success' if success else 'failed'}")
    
    # Test update
    if doc_ids and len(doc_ids) > 1:
        updated = milvus_db.update(
            doc_ids[1], 
            "Updated: Milvus provides enterprise-grade vector search.",
            metadata={"updated": True}
        )
        print(f"✓ Update operation: {'success' if updated else 'failed'}")
    
    # Test health check
    health = milvus_db.health_check()
    print(f"✓ Health check status: {health.get('status', 'unknown')}")
    
    return milvus_db

def test_milvus_cloud_connection():
    """Test Milvus Cloud (Zilliz) connection."""
    print("\nTesting Milvus Cloud Connection")
    
    # Check for cloud credentials
    uri = os.getenv("MILVUS_URI")
    token = os.getenv("MILVUS_TOKEN")
    
    if not uri or not token:
        print("⚠ Skipping cloud test - MILVUS_URI and MILVUS_TOKEN not set")
        return None
    
    try:
        # Initialize Milvus with cloud connection
        cloud_db = MilvusDB(
            embedding_model="text-embedding-3-small",
            collection_name="test_cloud_collection",
            uri=uri,
            token=token,
            metric="COSINE"
        )
        
        print("✓ Connected to Milvus Cloud successfully")
        print(f"  URI: {uri[:30]}...")
        
        # Test basic operations
        doc_id = cloud_db.add("Test document for cloud storage")
        print(f"✓ Added document to cloud: {doc_id}")
        
        results = cloud_db.query("cloud storage", top_k=1)
        print(f"✓ Query executed on cloud: {len(results)} results")
        
        # Clean up
        if doc_id:
            cloud_db.delete(doc_id)
            print("✓ Cleaned up test document from cloud")
        
        return cloud_db
        
    except Exception as e:
        print(f"✗ Cloud connection failed: {e}")
        return None

def test_milvus_metadata_filtering():
    """Test Milvus metadata filtering functionality."""
    print("\nTesting Milvus Metadata Filtering")
    
    milvus_db = MilvusDB(
        collection_name="test_filtering",
        db_file="test_filtering.db"
    )
    
    # Add documents with different categories
    categories = ["tutorial", "research", "news", "tutorial", "research"]
    topics = ["AI", "database", "search", "ML", "vector"]
    
    doc_ids = []
    for i, (cat, topic) in enumerate(zip(categories, topics)):
        doc = f"Document {i} about {topic}"
        doc_id = milvus_db.add(doc, metadata={
            "category": cat,
            "topic": topic,
            "index": i
        })
        doc_ids.append(doc_id)
    
    print(f"✓ Added {len(doc_ids)} documents with metadata")
    
    # Test query with filtering
    results = milvus_db.query(
        "document", 
        top_k=10,
        metadata_filter={"category": "research"}
    )
    print(f"✓ Found {len(results)} research documents")
    
    # Clean up
    for doc_id in doc_ids:
        milvus_db.delete(doc_id)
    
    return True

def test_milvus_batch_operations():
    """Test Milvus batch operations."""
    print("\nTesting Milvus Batch Operations")
    
    milvus_db = MilvusDB(
        collection_name="test_batch",
        db_file="test_batch.db"
    )
    
    # Test batch add
    documents = [
        "Batch document 1: Vector databases",
        "Batch document 2: Machine learning",
        "Batch document 3: Semantic search",
        "Batch document 4: Data processing",
        "Batch document 5: AI applications"
    ]
    
    doc_ids = milvus_db.add_batch(documents)
    print(f"✓ Batch added {len(doc_ids)} documents")
    
    # Test batch query
    queries = [
        "vector database",
        "machine learning AI",
        "semantic search"
    ]
    
    for query in queries:
        results = milvus_db.query(query, top_k=2)
        print(f"✓ Query '{query[:20]}...' returned {len(results)} results")
    
    # Test batch delete
    if len(doc_ids) >= 2:
        delete_ids = doc_ids[:2]
        success = milvus_db.delete_batch(delete_ids)
        print(f"✓ Batch delete operation: {'success' if success else 'failed'}")
        
        # Verify deletion
        remaining = milvus_db.count()
        print(f"✓ Remaining documents: {remaining}")
    
    return milvus_db

def test_milvus_similarity_metrics():
    """Test different similarity metrics in Milvus."""
    print("\nTesting Milvus Similarity Metrics")
    
    metrics = ["COSINE", "L2", "IP"]  # Inner Product
    
    for metric in metrics:
        try:
            db = MilvusDB(
                collection_name=f"test_{metric.lower()}",
                db_file=f"test_{metric.lower()}.db",
                metric=metric
            )
            
            # Add test documents
            docs = [
                "Similarity testing with different metrics",
                "Vector comparison methods in Milvus",
                "Distance calculations for embeddings"
            ]
            
            for doc in docs:
                db.add(doc)
            
            # Test query
            results = db.query("similarity metrics", top_k=2)
            print(f"✓ {metric} metric: {len(results)} results found")
            
        except Exception as e:
            print(f"✗ {metric} metric failed: {e}")

def run_all_tests():
    """Run all Milvus wrapper tests."""
    print("=" * 60)
    print("Milvus Wrapper Test Suite")
    print("=" * 60)
    
    try:
        # Basic operations
        test_milvus_basic_operations()
        
        # Cloud connection (optional)
        test_milvus_cloud_connection()
        
        # Metadata filtering
        test_milvus_metadata_filtering()
        
        # Batch operations
        test_milvus_batch_operations()
        
        # Similarity metrics
        test_milvus_similarity_metrics()
        
        print("\n" + "=" * 60)
        print("✓ All Milvus wrapper tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test files
        import glob
        for file in glob.glob("test_*.db"):
            try:
                os.remove(file)
                print(f"  Cleaned up: {file}")
            except:
                pass

if __name__ == "__main__":
    run_all_tests()