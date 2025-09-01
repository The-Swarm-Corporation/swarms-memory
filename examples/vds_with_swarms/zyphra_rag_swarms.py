"""
Agent with Zyphra RAG (Retrieval-Augmented Generation)

This example demonstrates using Zyphra RAG system for RAG operations,
allowing agents to store and retrieve documents for enhanced context.
Note: Zyphra RAG is a complete RAG system with graph-based retrieval.
"""

import torch
from swarms import Agent
from swarms_memory.vector_dbs.zyphra_rag import RAGSystem


# Simple LLM wrapper for demonstration
class SimpleLLM(torch.nn.Module):
    """
    Simple LLM wrapper for demonstration purposes.
    In production, replace this with your actual language model.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, prompt: str) -> str:
        # This is a placeholder - in production you would use a real LLM
        # like OpenAI's GPT, Hugging Face models, etc.
        return f"Based on the context: {prompt[:200]}... [Generated response would appear here]"
    
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
    # Initialize Zyphra RAG System
    llm = SimpleLLM()
    rag_db = RAGSystem(
        llm=llm,
        vocab_size=10000  # Vocabulary size for sparse embeddings
    )

    # Prepare document text - Zyphra RAG processes entire documents
    document_text = " ".join([
        "Zyphra RAG is an advanced retrieval system that combines sparse embeddings with graph-based retrieval algorithms.",
        "Zyphra RAG uses Personalized PageRank (PPR) to identify the most relevant document chunks for a given query.",
        "The system builds a graph representation of document chunks based on embedding similarities between text segments.",
        "Zyphra RAG employs sparse embeddings using word count methods for fast, CPU-friendly text representation.",
        "The graph builder creates adjacency matrices representing similarity relationships between document chunks.",
        "Zyphra RAG excels at context-aware document retrieval through its graph-based approach to semantic search."
    ])

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

    # Query with RAG
    response = agent.run("What is Zyphra RAG and how does its graph-based retrieval work?")
    print(response)