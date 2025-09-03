"""
Agent with Zyphra RAG (Retrieval-Augmented Generation)

This example demonstrates using Zyphra RAG system for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
Note: Zyphra RAG is a complete RAG system with graph-based retrieval.
"""

import torch
from swarms import Agent
from swarms_memory.vector_dbs.zyphra_rag import RAGSystem


# Simple LLM wrapper that uses the agent's model
class AgentLLMWrapper(torch.nn.Module):
    """
    LLM wrapper that integrates with the Swarms Agent's model.
    """
    def __init__(self):
        super().__init__()
        self.agent = None
        
    def set_agent(self, agent):
        """Set the agent reference for LLM calls"""
        self.agent = agent
        
    def forward(self, prompt: str) -> str:
        if self.agent:
            return self.agent.llm(prompt)
        return f"Generated response for: {prompt[:100]}..."
    
    def __call__(self, prompt: str) -> str:
        return self.forward(prompt)


# Create a wrapper class to make Zyphra RAG compatible with Swarms Agent
class ZyphraRAGWrapper:
    """
    Wrapper to make Zyphra RAG system compatible with Swarms Agent memory interface.
    """
    def __init__(self, rag_system, chunks, embeddings, graph):
        self.rag_system = rag_system
        self.chunks = chunks
        self.embeddings = embeddings
        self.graph = graph
    
    def add(self, doc: str):
        """Add method for compatibility - Zyphra processes entire documents at once"""
        print(f"Note: Zyphra RAG processes entire documents. Document already processed: {doc[:50]}...")
    
    def query(self, query_text: str, **kwargs) -> str:
        """Query the RAG system"""
        return self.rag_system.answer_query(query_text, self.chunks, self.embeddings, self.graph)


if __name__ == '__main__':
    # Create LLM wrapper
    llm = AgentLLMWrapper()
    
    # Initialize Zyphra RAG System
    rag_db = RAGSystem(
        llm=llm,
        vocab_size=10000  # Vocabulary size for sparse embeddings
    )

    # Add documents to the knowledge base
    documents = [
        "Zyphra RAG is an advanced retrieval system that combines sparse embeddings with graph-based retrieval algorithms.",
        "Zyphra RAG uses Personalized PageRank (PPR) to identify the most relevant document chunks for a given query.",
        "The system builds a graph representation of document chunks based on embedding similarities between text segments.",
        "Zyphra RAG employs sparse embeddings using word count methods for fast, CPU-friendly text representation.",
        "The graph builder creates adjacency matrices representing similarity relationships between document chunks.",
        "Zyphra RAG excels at context-aware document retrieval through its graph-based approach to semantic search.",
        "Kye Gomez is the founder of Swarms."
    ]
    
    document_text = " ".join(documents)

    # Process the document (creates chunks, embeddings, and graph)
    chunks, embeddings, graph = rag_db.process_document(document_text, chunk_size=100)

    # Create the wrapper
    rag_wrapper = ZyphraRAGWrapper(rag_db, chunks, embeddings, graph)

    # Create agent with RAG capabilities
    agent = Agent(
        agent_name="RAG-Agent",
        agent_description="Swarms Agent with Zyphra RAG-powered graph-based retrieval for enhanced knowledge retrieval",
        model_name="gpt-4o",
        max_loops=1,
        dynamic_temperature_enabled=True,
        long_term_memory=rag_wrapper
    )
    
    # Connect the LLM wrapper to the agent
    llm.set_agent(agent)

    # Query with RAG
    response = agent.run("What is Zyphra RAG and who is the founder of Swarms?")
    print(response)