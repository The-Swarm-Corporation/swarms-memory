"""
FAISS + Swarms Agent Integration Example

This example demonstrates how to use FAISSDB with Swarms agents
for building a knowledge-augmented AI assistant with persistent memory.

Features:
- Modern FAISS v1+ API with LiteLLM embeddings
- Multiple embedding provider support (OpenAI, Azure, Cohere, etc.)
- Swarms agent with RAG capabilities
- Dynamic memory updates from conversations
- Optional index persistence to disk
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from swarms import Agent
from swarms_memory.vector_dbs.faiss_wrapper import FAISSDB
from loguru import logger

# Load environment variables
load_dotenv()

class FAISSSwarmAgent:
    """
    A Swarms agent enhanced with FAISS vector memory for RAG capabilities.
    """
    
    def __init__(
        self,
        agent_name: str = "FAISSAgent",
        openai_api_key: str = None,
        embedding_model: str = "text-embedding-3-small",
        model_name: str = "gpt-4",
        index_file: str = None,
        index_type: str = "Flat",
        metric: str = "cosine"
    ):
        """
        Initialize the FAISS-powered Swarms agent.
        
        Args:
            agent_name: Name of the agent
            openai_api_key: OpenAI API key for embeddings and LLM
            embedding_model: Embedding model to use (LiteLLM format)
            model_name: LLM model to use
            index_file: Optional path to persist FAISS index
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW')
            metric: Distance metric ('cosine', 'l2', 'inner_product')
        """
        # Store initialization parameters
        self.agent_name = agent_name
        self.embedding_model_name = embedding_model
        self.index_file = index_file
        
        # Initialize FAISS vector database with modern API
        self.vector_db = FAISSDB(
            embedding_model=embedding_model,
            dimension=1536,  # Will be auto-detected
            index_type=index_type,
            metric=metric,
            index_file=index_file,
            api_key_embedding=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Swarms agent
        self.agent = Agent(
            agent_name=agent_name,
            model_name="gpt-4o",
            system_prompt="""You are an intelligent AI assistant with access to a FAISS vector memory system.
            
            Your capabilities:
            1. Store and retrieve information from your FAISS memory
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
        
        logger.info(f"Initialized {agent_name} with FAISS memory")
    
    def add_knowledge(self, documents: List[str], metadata_list: List[Dict[str, Any]] = None):
        """
        Add knowledge documents to the agent's memory.
        
        Args:
            documents: List of documents to add
            metadata_list: Optional metadata for each document
        """
        if metadata_list is None:
            metadata_list = [{}] * len(documents)
        
        logger.info(f"Adding {len(documents)} documents to agent memory...")
        
        for i, (doc, metadata) in enumerate(zip(documents, metadata_list)):
            try:
                doc_id = self.vector_db.add(doc, metadata=metadata)
                logger.info(f"Added document to memory: {doc[:50]}... (ID: {doc_id})")
            except Exception as e:
                logger.error(f"Failed to add document {i}: {str(e)}")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context from the vector database.
        
        Args:
            query: The query to search for
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
        
        # Format context for the agent
        context_parts = []
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            text = result.get('metadata', {}).get('text', 'No text available')
            source = result.get('metadata', {}).get('source', 'unknown')
            
            context_parts.append(f"[Context {i} - Relevance: {score:.3f} - Source: {source}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def process_query(self, user_query: str) -> str:
        """
        Process a user query using RAG (Retrieval Augmented Generation).
        
        Args:
            user_query: The user's question
            
        Returns:
            The agent's response
        """
        logger.info(f"Processing query: {user_query}")
        
        # Retrieve relevant context
        context = self.retrieve_context(user_query)
        
        # Create enhanced prompt with context
        enhanced_prompt = f"""
        Retrieved Context from Memory:
        {context}
        
        User Question: {user_query}
        
        Please provide a comprehensive answer using the retrieved context when relevant.
        If the context is helpful, reference it in your response.
        """
        
        # Get response from the agent
        try:
            response = self.agent.run(enhanced_prompt)
            
            # Store the interaction in memory for future reference
            interaction_text = f"Q: {user_query}\nA: {response}"
            self.vector_db.add(interaction_text, metadata={
                "type": "interaction",
                "query": user_query,
                "response": response
            })
            logger.info("Stored interaction in memory")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Sorry, I encountered an error processing your query: {str(e)}"
    
    def clear_memory(self) -> bool:
        """Clear all documents from the agent's memory."""
        success = self.vector_db.clear()
        if success:
            logger.info("Memory cleared successfully")
        return success
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return self.vector_db.health_check()
    
    def rebuild_index(self) -> bool:
        """Rebuild the FAISS index for better performance after deletions."""
        success = self.vector_db.rebuild_index()
        if success:
            logger.info("Index rebuilt successfully")
        return success


def main():
    """
    Main example demonstrating the FAISS + Swarms integration.
    """
    print("🚀 FAISS + Swarms Agent Demo\n")
    
    # Initialize the agent
    agent = FAISSSwarmAgent(
        agent_name="KnowledgeExpert",
        embedding_model="text-embedding-3-small",
        model_name="gpt-4",
        index_file="knowledge_base.faiss",  # Persist to disk
        index_type="Flat",  # Can also use "IVF" or "HNSW"
        metric="cosine"
    )
    
    # Sample knowledge base
    knowledge_docs = [
        "FAISS is a library for efficient similarity search and clustering of dense vectors. It is developed by Facebook AI Research.",
        "Vector databases enable semantic search by storing and querying high-dimensional vectors that represent the meaning of data.",
        "RAG (Retrieval Augmented Generation) combines the power of retrieval systems with generative AI models for more accurate responses.",
        "Swarms is a framework for building multi-agent AI systems that can collaborate to solve complex problems.",
        "LiteLLM provides a unified interface for accessing multiple embedding providers like OpenAI, Cohere, and Voyage AI.",
        "The text-embedding-3-small model from OpenAI produces 1536-dimensional vectors and is optimized for efficiency and performance."
    ]
    
    # Add knowledge to the agent's memory
    print("Adding knowledge to agent's memory...")
    agent.add_knowledge(
        knowledge_docs,
        metadata_list=[{"source": "documentation", "topic": topic} 
                      for topic in ["faiss", "vector_db", "rag", "swarms", "litellm", "embeddings"]]
    )
    
    # Example queries
    queries = [
        "What is FAISS and how does it work?",
        "How can vector databases improve AI applications?",
        "What are the benefits of using RAG in production?",
        "How do swarms agents collaborate?",
        "What embedding models are available through LiteLLM?",
        "Please remember this information: FAISS supports multiple index types like Flat, IVF, and HNSW for different performance characteristics.",
        "What do you know about FAISS index types?"
    ]
    
    print("💬 Starting Q&A Session:\n")
    
    for query in queries:
        print(f"❓ Question: {query}")
        response = agent.process_query(query)
        print(f"🤖 Answer: {response}")
        print("-" * 80)
    
    # Show memory statistics
    print("\nMemory Statistics:")
    stats = agent.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nDemo completed! Index saved to: {agent.index_file}")
    
    # Optional: Rebuild index for better performance
    print("\n🔄 Rebuilding index for optimal performance...")
    agent.rebuild_index()


if __name__ == "__main__":
    main()