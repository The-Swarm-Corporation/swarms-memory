#!/usr/bin/env python3
"""
Unified Memory System Example with Swarms Agents

This example demonstrates how to use different vector databases with swarms agents
using the standardized swarms-memory interface. All vector databases now support:
- Unified LiteLLM embedding models for access to multiple providers
- Consistent API interfaces (add, query, delete, clear)
- Swarms agent integration

Supported vector databases:
- QdrantVectorDatabase (fully implemented)
- PineconeMemory (updated with v3+ API)
- ChromaDB (standardized)
- FAISSDB (standardized)
- SingleStoreDB (standardized)
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demonstrate_qdrant_with_agent():
    """Demonstrate QdrantVectorDatabase with swarms agent (fully working)."""
    print("🔵 Demonstrating QdrantVectorDatabase with Swarms Agent")
    
    try:
        from swarms_memory.vector_dbs.qdrant_wrapper import QdrantVectorDatabase
        from swarms import Agent, OpenAIChat
        
        # Initialize vector database with LiteLLM embeddings
        vector_db = QdrantVectorDatabase(
            collection_name="agent_memory",
            embedding_model="text-embedding-3-small",  # OpenAI via LiteLLM
            # Alternative embedding models:
            # embedding_model="cohere/embed-english-v3.0",
            # embedding_model="voyage/voyage-01",
            # embedding_model="all-MiniLM-L6-v2",  # SentenceTransformer
            location=":memory:"  # In-memory for demo
        )
        
        # Add some knowledge to the vector database
        docs = [
            "Swarms is a multi-agent AI framework that enables coordination between multiple AI agents.",
            "Vector databases store high-dimensional vectors and enable semantic search.",
            "LiteLLM provides a unified interface to multiple embedding providers including OpenAI, Cohere, and more.",
            "The swarms-memory library provides standardized vector database interfaces."
        ]
        
        print("   Adding documents to vector database...")
        for doc in docs:
            doc_id = vector_db.add(doc, metadata={"source": "demo"})
            print(f"   ✅ Added: {doc[:50]}... (ID: {doc_id})")
        
        # Create a swarms agent with vector database memory
        llm = OpenAIChat(
            model_name="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        def memory_retrieval(query: str) -> str:
            """Custom memory retrieval function."""
            results = vector_db.query(query, top_k=2, return_metadata=True)
            if results:
                context = "\n".join([
                    f"- {result['metadata'].get('text', '')}" 
                    for result in results
                ])
                return f"Relevant memory:\n{context}"
            return "No relevant memory found."
        
        agent = Agent(
            agent_name="MemoryAgent",
            llm=llm,
            system_prompt="""You are an AI assistant with access to a knowledge base.
            Use the provided memory context to answer questions accurately.
            If the memory context is relevant, incorporate it into your response.""",
            max_loops=1,
            verbose=True
        )
        
        # Example query
        query = "What is swarms and how does it work?"
        memory_context = memory_retrieval(query)
        
        print(f"\n   🧠 Memory Context Retrieved:\n{memory_context}")
        
        # Run agent with memory context
        response = agent.run(
            f"Context: {memory_context}\n\nQuestion: {query}"
        )
        
        print(f"\n   🤖 Agent Response:\n{response}")
        
        # Demonstrate other operations
        print(f"\n   📊 Database Stats: {vector_db.get_stats()}")
        
        print("   ✅ QdrantVectorDatabase demonstration completed successfully!\n")
        
    except ImportError as e:
        print(f"   ❌ Missing dependencies for Qdrant demo: {e}")
    except Exception as e:
        print(f"   ❌ Error in Qdrant demo: {e}")

def demonstrate_pinecone_with_agent():
    """Demonstrate updated PineconeMemory with swarms agent."""
    print("📌 Demonstrating PineconeMemory with Swarms Agent")
    
    try:
        from swarms_memory.vector_dbs.pinecone_wrapper import PineconeMemory
        from swarms import Agent, OpenAIChat
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("   ⚠️  PINECONE_API_KEY not set, skipping Pinecone demo")
            return
            
        # Initialize with updated Pinecone v3+ API (no deprecated environment parameter)
        vector_db = PineconeMemory(
            api_key=api_key,
            index_name="swarms-demo",
            dimension=1536,  # text-embedding-3-small dimension
            embedding_model="text-embedding-3-small",  # LiteLLM OpenAI
            cloud="aws",
            region="us-east-1"
        )
        
        # Add documents
        docs = ["Agent collaboration is key to multi-agent systems.", 
                "Memory systems enable agents to learn and adapt."]
                
        print("   Adding documents...")
        for doc in docs:
            doc_id = vector_db.add(doc, metadata={"type": "knowledge"})
            print(f"   ✅ Added: {doc[:40]}... (ID: {doc_id})")
        
        # Query with new standardized interface
        query_result = vector_db.query(
            "How do agents collaborate?", 
            top_k=1, 
            return_metadata=True
        )
        
        print(f"   🔍 Query Result: {query_result}")
        print("   ✅ PineconeMemory demonstration completed!\n")
        
    except ImportError as e:
        print(f"   ❌ Missing dependencies for Pinecone demo: {e}")
    except Exception as e:
        print(f"   ❌ Error in Pinecone demo: {e}")

def demonstrate_chromadb_with_agent():
    """Demonstrate updated ChromaDB with swarms agent."""
    print("🟣 Demonstrating ChromaDB with Swarms Agent")
    
    try:
        from swarms_memory.vector_dbs.chroma_db_wrapper import ChromaDB
        
        # Initialize with LiteLLM embeddings
        vector_db = ChromaDB(
            output_dir="swarms_demo",
            embedding_model="text-embedding-3-small",  # LiteLLM
            verbose=True
        )
        
        # Add and query documents
        doc_id = vector_db.add(
            "ChromaDB now supports LiteLLM embeddings for multiple providers.",
            metadata={"source": "documentation"}
        )
        
        result = vector_db.query(
            "What embedding providers are supported?",
            top_k=1,
            return_metadata=True
        )
        
        print(f"   📝 Document ID: {doc_id}")
        print(f"   🔍 Query Result: {result}")
        print("   ✅ ChromaDB demonstration completed!\n")
        
    except ImportError as e:
        print(f"   ❌ Missing dependencies for ChromaDB demo: {e}")
    except Exception as e:
        print(f"   ❌ Error in ChromaDB demo: {e}")

def demonstrate_faiss_with_agent():
    """Demonstrate updated FAISS with swarms agent."""
    print("⚡ Demonstrating FAISS with Swarms Agent")
    
    try:
        from swarms_memory.vector_dbs.faiss_wrapper import FAISSDB
        
        # Initialize with LiteLLM embeddings
        vector_db = FAISSDB(
            dimension=1536,
            embedding_model="text-embedding-3-small",
            metric="cosine"
        )
        
        # Add and query documents
        doc_id = vector_db.add(
            "FAISS provides high-performance vector similarity search.",
            metadata={"library": "meta"}
        )
        
        result = vector_db.query(
            "What is FAISS good for?",
            top_k=1,
            return_metadata=True
        )
        
        print(f"   📝 Document ID: {doc_id}")
        print(f"   🔍 Query Result: {result}")
        print("   ✅ FAISS demonstration completed!\n")
        
    except ImportError as e:
        print(f"   ❌ Missing dependencies for FAISS demo: {e}")
    except Exception as e:
        print(f"   ❌ Error in FAISS demo: {e}")

def main():
    """Run all vector database demonstrations."""
    print("🚀 Swarms-Memory Unified Vector Database Demonstrations\n")
    print("This example shows the standardized interface across all vector databases:")
    print("- Unified LiteLLM embedding support")
    print("- Consistent add(), query(), delete(), clear() methods")
    print("- Swarms agent integration patterns")
    print("- Modern API standards (Pinecone v3+, etc.)\n")
    
    # Run demonstrations
    demonstrate_qdrant_with_agent()      # Fully working reference implementation
    demonstrate_pinecone_with_agent()    # Updated with v3+ API
    demonstrate_chromadb_with_agent()    # Standardized interface
    demonstrate_faiss_with_agent()       # Standardized interface
    
    print("🎉 All demonstrations completed!")
    print("\n💡 Key Improvements:")
    print("- ✅ Unified LiteLLM embedding system across all databases")
    print("- ✅ Consistent API interfaces for better developer experience")
    print("- ✅ Updated Pinecone wrapper with v3+ compatibility")
    print("- ✅ Enhanced base class with proper abstract methods")
    print("- ✅ Swarms agent integration examples")
    print("- ✅ Modern dependency management")

if __name__ == "__main__":
    main()