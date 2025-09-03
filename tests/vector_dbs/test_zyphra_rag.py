"""
Zyphra RAG System Test Suite

This script demonstrates and tests the Zyphra RAG system functionality
including document processing, graph-based retrieval, and query answering.

Run this script to verify Zyphra RAG functionality:
python tests/vector_dbs/test_zyphra_rag.py
"""

import torch
from swarms_memory.vector_dbs.zyphra_rag import RAGSystem

class MockLLM(torch.nn.Module):
    """Mock LLM for testing purposes."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, prompt: str) -> str:
        """Generate a simple response based on the prompt."""
        if "summarize" in prompt.lower():
            return "This is a summary of the retrieved documents about the query topic."
        elif "explain" in prompt.lower():
            return "This explains the concept based on the retrieved context."
        else:
            return f"Generated response for: {prompt[:100]}..."
    
    def __call__(self, prompt: str) -> str:
        return self.forward(prompt)

def test_zyphra_basic_operations():
    """Test basic Zyphra RAG operations."""
    print("Testing Zyphra RAG Basic Operations")
    
    # Initialize mock LLM
    llm = MockLLM()
    
    # Initialize RAG system
    rag_system = RAGSystem(
        llm=llm,
        vocab_size=10000
    )
    
    print("✓ Zyphra RAG system initialized")
    print(f"  Vocab size: {rag_system.vocab_size}")
    print(f"  Graph builder: {rag_system.graph_builder is not None}")
    
    # Test document
    document = """
    Zyphra RAG is an advanced retrieval system that uses graph-based algorithms.
    It employs Personalized PageRank for finding relevant document chunks.
    The system creates sparse embeddings for efficient similarity computation.
    Graph representations capture relationships between document segments.
    This approach enables context-aware retrieval for better results.
    """
    
    # Process the document
    chunks, embeddings, graph = rag_system.process_document(
        document, 
        chunk_size=50
    )
    
    print(f"✓ Document processed successfully")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Graph nodes: {graph.shape[0]}")
    
    # Test query
    query = "How does Zyphra RAG work?"
    answer = rag_system.answer_query(query, chunks, embeddings, graph)
    
    print(f"✓ Query answered successfully")
    print(f"  Query: {query}")
    print(f"  Answer length: {len(answer)} characters")
    
    return rag_system, chunks, embeddings, graph

def test_zyphra_chunking():
    """Test Zyphra document chunking functionality."""
    print("\nTesting Zyphra Document Chunking")
    
    llm = MockLLM()
    rag_system = RAGSystem(llm=llm)
    
    # Test with different chunk sizes
    document = "This is a test document. " * 20  # Create a longer document
    
    chunk_sizes = [20, 50, 100]
    for chunk_size in chunk_sizes:
        chunks, _, _ = rag_system.process_document(document, chunk_size=chunk_size)
        print(f"✓ Chunk size {chunk_size}: Created {len(chunks)} chunks")
        if chunks:
            avg_chunk_len = sum(len(c) for c in chunks) / len(chunks)
            print(f"  Average chunk length: {avg_chunk_len:.1f} characters")

def test_zyphra_sparse_embeddings():
    """Test Zyphra sparse embeddings functionality."""
    print("\nTesting Zyphra Sparse Embeddings")
    
    llm = MockLLM()
    rag_system = RAGSystem(llm=llm, vocab_size=5000)
    
    # Test embedding generation
    test_texts = [
        "Vector databases store embeddings",
        "Machine learning models process data",
        "Natural language processing is useful",
        "Artificial intelligence transforms industries"
    ]
    
    for text in test_texts:
        embedding = rag_system._get_sparse_embedding(text)
        print(f"✓ Embedding for '{text[:30]}...'")
        print(f"  Shape: {embedding.shape}")
        print(f"  Non-zero elements: {embedding.count_nonzero()}")
        print(f"  Max value: {embedding.max():.4f}")

def test_zyphra_graph_building():
    """Test Zyphra graph building functionality."""
    print("\nTesting Zyphra Graph Building")
    
    llm = MockLLM()
    rag_system = RAGSystem(llm=llm)
    
    # Create a document with clear structure
    document = """
    Chapter 1: Introduction to AI.
    Artificial intelligence is transforming technology.
    Machine learning is a subset of AI.
    
    Chapter 2: Deep Learning.
    Neural networks power deep learning.
    Deep learning requires large datasets.
    
    Chapter 3: Applications.
    AI is used in healthcare.
    AI powers recommendation systems.
    """
    
    chunks, embeddings, graph = rag_system.process_document(document, chunk_size=40)
    
    print(f"✓ Graph built successfully")
    print(f"  Graph shape: {graph.shape}")
    print(f"  Graph density: {graph.count_nonzero() / (graph.shape[0] * graph.shape[1]):.4f}")
    
    # Check graph properties
    if graph.shape[0] > 0:
        row_sums = graph.sum(dim=1)
        print(f"  Average connections per node: {row_sums.mean():.2f}")
        print(f"  Max connections: {row_sums.max()}")

def test_zyphra_retrieval_quality():
    """Test Zyphra retrieval quality with different queries."""
    print("\nTesting Zyphra Retrieval Quality")
    
    llm = MockLLM()
    rag_system = RAGSystem(llm=llm)
    
    # Create a knowledge base
    document = """
    Python is a high-level programming language.
    Python supports multiple programming paradigms.
    JavaScript is used for web development.
    JavaScript runs in browsers and Node.js.
    Java is a statically typed language.
    Java is popular for enterprise applications.
    C++ provides low-level memory control.
    C++ is used in system programming.
    """
    
    chunks, embeddings, graph = rag_system.process_document(document, chunk_size=50)
    
    # Test different queries
    queries = [
        "What is Python used for?",
        "Tell me about JavaScript",
        "How does Java compare to C++?",
        "What are programming languages?"
    ]
    
    for query in queries:
        answer = rag_system.answer_query(query, chunks, embeddings, graph)
        print(f"✓ Query: '{query}'")
        print(f"  Answer generated ({len(answer)} chars)")

def test_zyphra_large_document():
    """Test Zyphra with a larger document."""
    print("\nTesting Zyphra with Large Document")
    
    llm = MockLLM()
    rag_system = RAGSystem(llm=llm, vocab_size=15000)
    
    # Create a larger document
    sections = [
        "Machine learning revolutionizes data analysis.",
        "Deep learning models process complex patterns.",
        "Natural language processing understands text.",
        "Computer vision analyzes visual information.",
        "Reinforcement learning optimizes decisions.",
        "Transfer learning leverages pre-trained models.",
        "Federated learning preserves data privacy.",
        "Meta-learning enables fast adaptation."
    ]
    
    large_document = " ".join(sections * 5)  # Repeat to make it larger
    
    print(f"  Document size: {len(large_document)} characters")
    
    # Process the large document
    chunks, embeddings, graph = rag_system.process_document(
        large_document, 
        chunk_size=100
    )
    
    print(f"✓ Large document processed")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Graph edges: {graph.count_nonzero()}")
    
    # Test retrieval on large document
    test_query = "What are the different types of learning in AI?"
    answer = rag_system.answer_query(test_query, chunks, embeddings, graph)
    print(f"✓ Query on large document successful")

def test_zyphra_edge_cases():
    """Test Zyphra edge cases and error handling."""
    print("\nTesting Zyphra Edge Cases")
    
    llm = MockLLM()
    rag_system = RAGSystem(llm=llm)
    
    # Test empty document
    try:
        chunks, embeddings, graph = rag_system.process_document("", chunk_size=50)
        print(f"✓ Empty document handled: {len(chunks)} chunks")
    except Exception as e:
        print(f"⚠ Empty document error: {e}")
    
    # Test very small chunk size
    try:
        chunks, embeddings, graph = rag_system.process_document(
            "Small test", 
            chunk_size=5
        )
        print(f"✓ Small chunk size handled: {len(chunks)} chunks")
    except Exception as e:
        print(f"⚠ Small chunk size error: {e}")
    
    # Test single word document
    try:
        chunks, embeddings, graph = rag_system.process_document("Word", chunk_size=50)
        print(f"✓ Single word handled: {len(chunks)} chunks")
    except Exception as e:
        print(f"⚠ Single word error: {e}")
    
    # Test special characters
    try:
        special_doc = "Test @#$%^&* with special chars!"
        chunks, embeddings, graph = rag_system.process_document(special_doc, chunk_size=50)
        print(f"✓ Special characters handled: {len(chunks)} chunks")
    except Exception as e:
        print(f"⚠ Special characters error: {e}")

def test_zyphra_performance():
    """Test Zyphra performance with timing."""
    print("\nTesting Zyphra Performance")
    
    import time
    
    llm = MockLLM()
    rag_system = RAGSystem(llm=llm, vocab_size=10000)
    
    # Create test document
    document = "Performance testing document. " * 100
    
    # Time document processing
    start_time = time.time()
    chunks, embeddings, graph = rag_system.process_document(document, chunk_size=75)
    process_time = time.time() - start_time
    
    print(f"✓ Document processing time: {process_time:.3f} seconds")
    print(f"  Processing rate: {len(document) / process_time:.0f} chars/sec")
    
    # Time query answering
    query = "What is this document about?"
    start_time = time.time()
    answer = rag_system.answer_query(query, chunks, embeddings, graph)
    query_time = time.time() - start_time
    
    print(f"✓ Query answering time: {query_time:.3f} seconds")
    
    # Calculate metrics
    chunks_per_second = len(chunks) / process_time if process_time > 0 else 0
    print(f"✓ Performance metrics:")
    print(f"  Chunks processed per second: {chunks_per_second:.1f}")
    print(f"  Total processing time: {process_time + query_time:.3f} seconds")

def run_all_tests():
    """Run all Zyphra RAG tests."""
    print("=" * 60)
    print("Zyphra RAG System Test Suite")
    print("=" * 60)
    
    try:
        # Basic operations
        rag_system, chunks, embeddings, graph = test_zyphra_basic_operations()
        
        # Chunking
        test_zyphra_chunking()
        
        # Sparse embeddings
        test_zyphra_sparse_embeddings()
        
        # Graph building
        test_zyphra_graph_building()
        
        # Retrieval quality
        test_zyphra_retrieval_quality()
        
        # Large document
        test_zyphra_large_document()
        
        # Edge cases
        test_zyphra_edge_cases()
        
        # Performance
        test_zyphra_performance()
        
        print("\n" + "=" * 60)
        print("✓ All Zyphra RAG tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()