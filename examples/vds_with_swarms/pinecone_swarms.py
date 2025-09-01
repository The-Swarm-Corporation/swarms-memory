"""
Pinecone + Swarms Agent Integration Example

This example demonstrates how to use PineconeMemory with Swarms agents
for building a knowledge-augmented AI assistant with persistent memory.

Features:
- Pinecone v3+ API (no deprecated parameters)
- LiteLLM embeddings (text-embedding-3-small)
- Swarms agent with RAG capabilities
- Dynamic memory updates from conversations
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from swarms import Agent
from swarms_memory.vector_dbs.pinecone_wrapper import PineconeMemory
from swarms_memory.embeddings.litellm_embeddings import LiteLLMEmbeddings
from loguru import logger

# Load environment variables
load_dotenv()

class PineconeSwarmAgent:
    """
    A Swarms agent enhanced with Pinecone vector memory for RAG capabilities.
    """
    
    def __init__(
        self,
        agent_name: str = "PineconeAgent",
        pinecone_api_key: str = None,
        openai_api_key: str = None,
        index_name: str = "swarms-agent-memory",
        model_name: str = "gpt-4",
        embedding_model: str = "text-embedding-3-small",
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize the Pinecone-powered Swarms agent.
        
        Args:
            agent_name: Name of the agent
            pinecone_api_key: Pinecone API key
            openai_api_key: OpenAI API key
            index_name: Name of Pinecone index
            model_name: LLM model to use
            embedding_model: Embedding model (LiteLLM)
            cloud: Cloud provider for Pinecone
            region: Region for Pinecone
        """
        # Store initialization parameters
        self.index_name = index_name
        self.embedding_model_name = embedding_model
        
        # Initialize Pinecone vector database with modern API
        self.vector_db = PineconeMemory(
            api_key=pinecone_api_key or os.getenv("PINECONE_API_KEY"),
            index_name=index_name,
            embedding_model=embedding_model,
            dimension=1536,  # Dimension for text-embedding-3-small
            metric="cosine",
            cloud=cloud,
            region=region,
            namespace="agent_memory",
            api_key_embedding=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Swarms agent
        
        self.agent = Agent(
            agent_name=agent_name,
            model_name="gpt-4o",
            system_prompt="""You are an intelligent AI assistant with access to a vector memory system.
            
            Your capabilities:
            1. Store and retrieve information from your Pinecone memory
            2. Learn from conversations and update your knowledge base
            3. Provide accurate answers using retrieved context
            4. Maintain conversation history for better responses
            
            When answering questions:
            - First check your memory for relevant information
            - Use retrieved context to provide accurate, detailed answers
            - If no relevant context exists, clearly state you're using general knowledge
            - Learn from new information and store it for future use
            
            Be helpful, accurate, and continuously improve your knowledge base.""",
            max_loops=1,
            verbose=True
        )
        
        self.agent_name = agent_name
        logger.info(f"Initialized {agent_name} with Pinecone memory")
    
    def add_knowledge(self, documents: List[str], metadata_list: List[Dict[str, Any]] = None):
        """
        Add knowledge documents to the agent's memory.
        
        Args:
            documents: List of documents to add
            metadata_list: Optional metadata for each document
        """
        if metadata_list is None:
            metadata_list = [{"source": "knowledge_base"} for _ in documents]
        
        doc_ids = []
        for doc, metadata in zip(documents, metadata_list):
            doc_id = self.vector_db.add(doc, metadata=metadata)
            doc_ids.append(doc_id)
            logger.info(f"Added document to memory: {doc[:50]}... (ID: {doc_id})")
        
        return doc_ids
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context from Pinecone memory.
        
        Args:
            query: Query string
            top_k: Number of results to retrieve
            
        Returns:
            Formatted context string
        """
        results = self.vector_db.query(
            query_text=query,
            top_k=top_k
        )
        
        if not results:
            return "No relevant context found in memory."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result['metadata'].get('text', '')
            score = result.get('score', 0)
            source = result['metadata'].get('source', 'unknown')
            
            context_parts.append(
                f"[Context {i} - Relevance: {score:.3f} - Source: {source}]\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def process_query(self, user_query: str, store_interaction: bool = True) -> str:
        """
        Process a user query with memory augmentation.
        
        Args:
            user_query: The user's question
            store_interaction: Whether to store the interaction in memory
            
        Returns:
            Agent's response
        """
        # Retrieve relevant context
        context = self.retrieve_context(user_query)
        
        # Construct augmented prompt
        augmented_prompt = f"""
        Retrieved Context from Memory:
        {context}
        
        User Question: {user_query}
        
        Please provide a comprehensive answer using the retrieved context when relevant.
        If the context is helpful, reference it in your response.
        """
        
        # Get agent response
        response = self.agent.run(augmented_prompt)
        
        # Store the interaction in memory for future reference
        if store_interaction:
            interaction_doc = f"Q: {user_query}\nA: {response}"
            self.vector_db.add(
                interaction_doc,
                metadata={
                    "source": "conversation",
                    "agent": self.agent_name,
                    "type": "qa_pair"
                }
            )
            logger.info("Stored interaction in memory")
        
        return response
    
    def clear_memory(self, namespace: str = None):
        """Clear all documents from memory."""
        success = self.vector_db.clear()
        if success:
            logger.info("Memory cleared successfully")
        return success
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        # Note: Pinecone doesn't have built-in stats, so we'll return basic info
        return {
            "database": "Pinecone",
            "index_name": self.index_name,
            "namespace": self.vector_db.namespace,
            "embedding_model": self.embedding_model_name,
            "dimension": 1536
        }


def main():
    """
    Demonstration of Pinecone + Swarms integration.
    """
    print("🚀 Pinecone + Swarms Agent Demo\n")
    
    # Initialize the agent
    agent = PineconeSwarmAgent(
        agent_name="KnowledgeExpert",
        index_name="swarms-demo",
        embedding_model="text-embedding-3-small",
        model_name="gpt-4"
    )
    
    # Add initial knowledge base
    print("📚 Adding knowledge to agent's memory...\n")
    knowledge_docs = [
        "Pinecone is a managed vector database designed for production-ready AI applications. It offers serverless and pod-based deployments.",
        "Vector databases enable semantic search by storing and querying high-dimensional vectors that represent the meaning of data.",
        "RAG (Retrieval Augmented Generation) combines the power of retrieval systems with generative AI models for more accurate responses.",
        "Swarms is a framework for building multi-agent AI systems that can collaborate to solve complex problems.",
        "LiteLLM provides a unified interface for accessing multiple embedding providers like OpenAI, Cohere, and Voyage AI.",
        "The text-embedding-3-small model from OpenAI produces 1536-dimensional vectors and costs $0.00002 per 1k tokens."
    ]
    
    agent.add_knowledge(
        knowledge_docs,
        metadata_list=[{"source": "documentation", "topic": topic} 
                       for topic in ["pinecone", "vector_db", "rag", "swarms", "litellm", "embeddings"]]
    )
    
    # Interactive Q&A examples
    queries = [
        "What is Pinecone and how does it work?",
        "How can vector databases improve AI applications?",
        "What are the benefits of using RAG in production?",
        "How do swarms agents collaborate?",
        "What embedding models are available through LiteLLM?"
    ]
    
    print("💬 Starting Q&A Session:\n")
    for query in queries:
        print(f"❓ Question: {query}")
        response = agent.process_query(query)
        print(f"🤖 Answer: {response}\n")
        print("-" * 80 + "\n")
    
    # Demonstrate learning from conversation
    print("📝 Teaching the agent new information...\n")
    new_info = "Pinecone recently introduced serverless indexes that automatically scale and only charge for usage, making it cost-effective for variable workloads."
    agent.process_query(
        f"Please remember this information: {new_info}",
        store_interaction=True
    )
    
    # Query about the new information
    print("❓ Testing recall of new information...")
    response = agent.process_query("What do you know about Pinecone's serverless indexes?")
    print(f"🤖 Answer: {response}\n")
    
    # Show memory stats
    stats = agent.get_memory_stats()
    print(f"📊 Memory Stats: {stats}\n")
    
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    main()