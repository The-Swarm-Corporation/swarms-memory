"""
Agent with Milvus Cloud RAG (Retrieval-Augmented Generation)

This example demonstrates using Milvus Cloud (Zilliz) as a vector database for RAG operations,
allowing agents to store and retrieve documents from your cloud-hosted Milvus account.
"""

import os
from dotenv import load_dotenv
from swarms import Agent
from swarms_memory import MilvusDB

# Load environment variables
load_dotenv()

def main():
    print("🌐 Milvus Cloud RAG Integration with Swarms")
    print("=" * 60)
    
    # Get Milvus Cloud credentials
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    
    if not milvus_uri or not milvus_token:
        print("❌ Missing Milvus Cloud credentials!")
        print("Please set MILVUS_URI and MILVUS_TOKEN in your .env file")
        return
    
    print(f"🔗 Connecting to: {milvus_uri}")
    print(f"🔑 Using token: {milvus_token[:20]}...")
    print()
    
    try:
        # Initialize Milvus Cloud wrapper for RAG operations
        rag_db = MilvusDB(
            embedding_model="text-embedding-3-small",  # OpenAI embedding model
            collection_name="swarms_cloud_knowledge",  # Cloud collection name
            uri=milvus_uri,                           # Your Zilliz Cloud URI
            token=milvus_token,                       # Your Zilliz Cloud token
            metric="COSINE",                          # Distance metric for similarity search
        )
        
        print("✅ Successfully connected to Milvus Cloud!")
        print(f"   Collection: {rag_db.collection_name}")
        print(f"   Dimension: {rag_db.dimension}")
        print(f"   Model: {rag_db.model_name}")
        print()
        
        # Add documents to the cloud knowledge base
        documents = [
            "Milvus Cloud is a fully managed vector database service provided by Zilliz.",
            "Zilliz Cloud offers serverless vector database hosting with automatic scaling.",
            "Milvus supports multiple indexing algorithms including HNSW, IVF, and DiskANN for optimal performance.",
            "Vector databases enable semantic search by storing high-dimensional embeddings.",
            "Retrieval-Augmented Generation (RAG) combines vector search with language model generation.",
            "The Swarms framework enables sophisticated multi-agent coordination and workflow orchestration.",
            "Swarms agents can utilize cloud-hosted vector databases for persistent knowledge storage.",
            "LiteLLM provides unified access to multiple embedding providers and language models.",
            "Semantic similarity search finds the most relevant documents based on meaning rather than keywords.",
            "Cloud-hosted vector databases offer scalability, reliability, and managed infrastructure.",
            "Kye Gomez is the founder and CEO of Swarms Corporation."
        ]
        
        # Add documents individually with metadata
        print("📝 Building cloud knowledge base...")
        for i, doc in enumerate(documents):
            rag_db.add(
                doc, 
                metadata={
                    "source": "swarms_documentation",
                    "category": "ai_technology", 
                    "index": i,
                    "timestamp": "2024",
                    "environment": "cloud"
                }
            )
            print(f"   ✅ Added document {i+1}: {doc[:60]}...")
        
        print(f"\n🎉 Successfully added {len(documents)} documents to cloud collection!")
        
        # Create agent with cloud-based RAG capabilities
        agent = Agent(
            agent_name="Cloud-RAG-Agent",
            agent_description="Swarms Agent with Milvus Cloud-powered RAG for scalable knowledge retrieval",
            model_name="gpt-4o",
            max_loops=1,
            dynamic_temperature_enabled=True,
            long_term_memory=rag_db
        )
        
        print("✅ Swarms agent created with cloud-hosted memory!")
        
        # Test queries with cloud-based RAG
        test_queries = [
            "What is Milvus Cloud and how does it differ from self-hosted Milvus?",
            "Who founded Swarms and what does the framework enable?", 
            "Explain RAG and how it enhances AI applications with vector databases.",
            "What indexing algorithms does Milvus support for performance optimization?",
            "How do cloud-hosted vector databases benefit AI applications?"
        ]
        
        print("\n" + "="*80)
        print("🤖 Testing Swarms Agent with Cloud-Hosted Milvus RAG")
        print("="*80)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 70)
            
            try:
                response = agent.run(query)
                print(f"🎯 Response: {response}")
                
            except Exception as e:
                print(f"❌ Query Error: {e}")
            
            print("-" * 70)
        
        print("\n🎉 Cloud RAG integration test completed successfully!")
        
        # Display cloud database statistics
        print("\n📊 Cloud Database Statistics:")
        print(f"   📦 Collection: {rag_db.collection_name}")
        print(f"   📏 Vector dimension: {rag_db.dimension}")
        print(f"   🧠 Embedding model: {rag_db.model_name}")
        print(f"   📈 Distance metric: {rag_db.metric}")
        print(f"   🌐 Cloud URI: {milvus_uri}")
        
        # Test document count
        try:
            count = rag_db.count()
            print(f"   📄 Total documents: {count}")
        except Exception as e:
            print(f"   📄 Total documents: Unable to count ({e})")
        
        # Health check
        health = rag_db.health_check()
        print(f"   🏥 Health status: {health['status']}")
        
        # Optional: Test direct cloud query
        print("\n🔍 Direct Cloud Query Test:")
        try:
            direct_query = "What are the benefits of using cloud-hosted vector databases?"
            results = rag_db.query_as_text(direct_query, top_k=3)
            print(f"Query: {direct_query}")
            print("Retrieved context from cloud:")
            print("-" * 50)
            print(results[:300] + "..." if len(results) > 300 else results)
            print("-" * 50)
        except Exception as e:
            print(f"❌ Direct query failed: {e}")
        
        print("\n✨ Your collections are now visible in your Zilliz Cloud dashboard!")
        print(f"   Dashboard: https://cloud.zilliz.com/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify MILVUS_URI and MILVUS_TOKEN in .env file")
        print("2. Check your Zilliz Cloud account status")
        print("3. Ensure proper network connectivity")
        print("4. Validate API token permissions")


if __name__ == "__main__":
    main()