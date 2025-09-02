"""
Milvus Cloud Example - Using Your Zilliz Cloud Account

This example demonstrates how to use the MilvusDB wrapper with your actual
Milvus Cloud (Zilliz) account instead of local Milvus Lite.
"""

import os
from dotenv import load_dotenv
from swarms_memory import MilvusDB

# Load environment variables
load_dotenv()

def test_milvus_cloud():
    """Test connection to Milvus Cloud and create collections."""
    print("🌐 Testing Milvus Cloud Connection")
    print("=" * 50)
    
    # Get credentials from environment
    milvus_uri = os.getenv("MILVUS_URI")
    milvus_token = os.getenv("MILVUS_TOKEN")
    
    if not milvus_uri or not milvus_token:
        print("❌ Missing Milvus Cloud credentials!")
        print("Please set MILVUS_URI and MILVUS_TOKEN in your .env file")
        return
    
    print(f"URI: {milvus_uri}")
    print(f"Token: {milvus_token[:20]}...")
    print()
    
    try:
        # Initialize MilvusDB with Cloud credentials
        milvus_db = MilvusDB(
            embedding_model="text-embedding-3-small",
            collection_name="cloud_test_collection",
            uri=milvus_uri,      # Your Zilliz Cloud URI
            token=milvus_token,  # Your Zilliz Cloud token
            # No db_file parameter for cloud
        )
        
        print("✅ Successfully connected to Milvus Cloud!")
        print(f"   Collection: {milvus_db.collection_name}")
        print(f"   Dimension: {milvus_db.dimension}")
        print(f"   Model: {milvus_db.model_name}")
        print()
        
        # Add some test documents
        test_docs = [
            "Milvus Cloud provides managed vector database services.",
            "Zilliz is the company behind Milvus vector database.",
            "Vector search enables semantic similarity matching.",
            "AI applications benefit from vector database capabilities.",
            "Embeddings convert text into numerical representations."
        ]
        
        print("📝 Adding documents to cloud collection...")
        doc_ids = []
        for i, doc in enumerate(test_docs):
            doc_id = milvus_db.add(
                doc, 
                metadata={
                    "source": "cloud_test",
                    "category": "technology", 
                    "index": i
                }
            )
            doc_ids.append(doc_id)
            print(f"   Added: {doc[:50]}... (ID: {doc_id})")
        
        print(f"\n✅ Added {len(doc_ids)} documents to cloud collection")
        
        # Test search functionality
        print("\n🔍 Testing search functionality...")
        query = "What is vector search?"
        results = milvus_db.query(query, top_k=3)
        
        print(f"Query: '{query}'")
        print(f"Found {len(results)} results:")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Text: {result['metadata']['text']}")
            print(f"     ID: {result['id']}")
        
        # Test document count
        count = milvus_db.count()
        print(f"\n📊 Total documents in cloud collection: {count}")
        
        # Health check
        health = milvus_db.health_check()
        print(f"🏥 Database health: {health['status']}")
        
        return milvus_db
        
    except Exception as e:
        print(f"❌ Error connecting to Milvus Cloud: {e}")
        print("\nTroubleshooting:")
        print("1. Check your MILVUS_URI and MILVUS_TOKEN in .env file")
        print("2. Verify your Zilliz Cloud account is active")
        print("3. Ensure your API token has proper permissions")
        return None


def test_swarms_with_cloud():
    """Test Swarms integration with Milvus Cloud."""
    print("\n🤖 Testing Swarms Integration with Milvus Cloud")
    print("=" * 60)
    
    try:
        from swarms import Agent
        
        # Initialize cloud-based MilvusDB
        rag_db = MilvusDB(
            embedding_model="text-embedding-3-small",
            collection_name="swarms_cloud_rag",
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN")
        )
        
        print("✅ Cloud RAG database initialized")
        
        # Add knowledge base documents
        knowledge_docs = [
            "Milvus Cloud offers serverless vector database hosting.",
            "Swarms framework enables sophisticated multi-agent workflows.",
            "RAG (Retrieval-Augmented Generation) enhances LLM responses with external knowledge.",
            "Vector databases store high-dimensional embeddings for similarity search.",
            "Kye Gomez is the founder and CEO of Swarms Corporation."
        ]
        
        print("📚 Building cloud knowledge base...")
        for doc in knowledge_docs:
            rag_db.add(doc, metadata={"type": "knowledge", "source": "swarms_docs"})
            
        print(f"✅ Added {len(knowledge_docs)} documents to cloud RAG database")
        
        # Create Swarms agent with cloud memory
        agent = Agent(
            agent_name="Cloud-RAG-Agent",
            agent_description="Swarms Agent with cloud-hosted Milvus RAG capabilities",
            model_name="gpt-4o",
            max_loops=1,
            dynamic_temperature_enabled=True,
            long_term_memory=rag_db
        )
        
        print("✅ Swarms agent created with cloud memory")
        
        # Test RAG query
        query = "Who founded Swarms and what does RAG stand for?"
        print(f"\n💬 Query: {query}")
        
        response = agent.run(query)
        print(f"🎯 Agent Response: {response}")
        
        return True
        
    except ImportError:
        print("❌ Swarms not available for testing")
        return False
    except Exception as e:
        print(f"❌ Error in Swarms integration: {e}")
        return False


def list_cloud_collections():
    """List all collections in your Milvus Cloud account."""
    print("\n📋 Listing Cloud Collections")
    print("=" * 40)
    
    try:
        from pymilvus import MilvusClient
        
        # Connect directly to cloud
        client = MilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN")
        )
        
        # List all collections
        collections = client.list_collections()
        print(f"Found {len(collections)} collection(s):")
        
        for collection in collections:
            print(f"  - {collection}")
            
            # Get collection stats
            try:
                stats = client.describe_collection(collection_name=collection)
                print(f"    Schema: {len(stats.get('fields', []))} fields")
            except:
                print("    (Unable to get detailed stats)")
        
        return collections
        
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []


def main():
    """Run all Milvus Cloud examples."""
    print("🚀 Milvus Cloud Integration Examples")
    print("=" * 60)
    
    # Test basic cloud connection
    milvus_db = test_milvus_cloud()
    
    if milvus_db:
        # Test Swarms integration
        test_swarms_with_cloud()
        
        # List all collections
        list_cloud_collections()
        
        print("\n🎉 All Milvus Cloud examples completed successfully!")
        print("\nYou can now see your collections in your Zilliz Cloud dashboard!")
    else:
        print("\n❌ Could not connect to Milvus Cloud. Please check your credentials.")


if __name__ == "__main__":
    main()