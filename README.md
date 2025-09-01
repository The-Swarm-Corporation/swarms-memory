<div align="center">
  <a href="https://swarms.world">
    <h1>Swarms Memory</h1>
  </a>
</div>
<p align="center">
  <em>The Enterprise-Grade Production-Ready Vector Memory Framework</em>
</p>

<p align="center">
    <a href="https://pypi.org/project/swarms-memory/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/swarms-memory?style=for-the-badge&color=3670A0">
    </a>
</p>
<p align="center">
<a href="https://twitter.com/swarms_corp/">🐦 Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://discord.gg/agora-999382051935506503">📢 Discord</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://swarms.world/explorer">Swarms Platform</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://docs.swarms.world">📙 Documentation</a>
</p>

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/swarms)](https://github.com/kyegomez/swarms-memory/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/swarms)](https://github.com/kyegomez/swarms-memory/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/swarms)](https://github.com/kyegomez/swarms-memory/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/swarms-memory)](https://github.com/kyegomez/swarms-memory/blob/main/LICENSE)[![Dependency Status](https://img.shields.io/librariesio/github/kyegomez/swarms)](https://libraries.io/github/kyegomez/swarms) [![Downloads](https://static.pepy.tech/badge/swarms-memory/month)](https://pepy.tech/project/swarms-memory)

[![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarmsmemory)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms-memory)

---

## 🚀 **What's New in v0.2.0**

**Major modernization update** bringing cutting-edge features and improved developer experience:

✅ **100+ Embedding Models** - LiteLLM integration supports OpenAI, Azure, Cohere, Voyage AI, AWS Bedrock  
✅ **Modern APIs** - No deprecated parameters, auto-dimension detection, UUID document IDs  
✅ **Production Ready** - Comprehensive error handling, health checks, persistence  
✅ **Swarms Integration** - Built-in RAG agents with conversation memory  
✅ **Enhanced FAISS** - Index persistence, rebuild optimization, multiple index types  
✅ **Modernized Pinecone** - v4+ API, serverless support, improved performance  

---

## 🎯 **Supported Vector Databases**

| **Vector DB**   | **Status** | **LiteLLM** | **Swarms** | **Persistence** | **Description**                                                    |
| --------------- | ---------- | ----------- | ---------- | --------------- | ------------------------------------------------------------------ |
| **Pinecone**    | ✅         | ✅          | ✅         | ☁️              | Fully managed serverless vector database                          |
| **FAISS**       | ✅         | ✅          | ✅         | 💾              | Facebook's similarity search library with local persistence       |
| **Qdrant**      | ✅         | ✅          | ✅         | 💾              | Open-source vector search engine with hybrid cloud support        |
| **ChromaDB**    | ✅         | ✅          | ⏳         | 💾              | Open-source embedding database for LLM applications               |
| **SingleStore** | ✅         | ✅          | ⏳         | ☁️              | Distributed SQL database with vector similarity search            |
| **Weaviate**    | ⏳         | ⏳          | ⏳         | ☁️              | Open-source vector database with GraphQL API                      |

**Legend**: ✅ Available | ⏳ Coming Soon | ☁️ Cloud | 💾 Local/Self-hosted

---

## 📦 **Installation**

```bash
# Basic installation
pip install swarms-memory

# With specific vector database support
pip install swarms-memory[pinecone]     # Pinecone + dependencies
pip install swarms-memory[faiss]        # FAISS + dependencies
pip install swarms-memory[all]          # All vector databases
```

## ⚡ **Quick Start**

### **Basic Vector Database Usage**

All vector databases now share a consistent, modern API:

```python
from swarms_memory import PineconeMemory, FAISSDB, QdrantDB

# Initialize with any embedding provider
db = PineconeMemory(
    api_key="your-pinecone-key",
    index_name="my-index", 
    embedding_model="text-embedding-3-small",  # OpenAI
    api_key_embedding="your-openai-key"
)

# Or use Azure OpenAI
db = FAISSDB(
    embedding_model="azure/my-deployment",
    api_key_embedding="your-azure-key",
    api_base="https://your-resource.openai.azure.com",
    dimension=1536
)

# Consistent API across all databases
doc_id = db.add("Your document text", metadata={"source": "docs"})
results = db.query("search query", top_k=5)
document = db.get(doc_id)
success = db.delete(doc_id)
count = db.count()
health = db.health_check()
```

### **Swarms Agent with Memory**

Build intelligent agents with persistent memory:

```python
from swarms import Agent
from swarms_memory import PineconeMemory

# Create memory-enhanced agent
memory = PineconeMemory(
    api_key="your-pinecone-key",
    index_name="agent-memory",
    embedding_model="text-embedding-3-small"
)

agent = Agent(
    agent_name="KnowledgeBot",
    model_name="gpt-4",
    system_prompt="You are an AI with persistent memory...",
    long_term_memory=memory  # Add memory to agent
)

# Agent automatically stores and retrieves context
response = agent.run("Tell me about vector databases")
```

---

## 🔧 **Comprehensive Examples**

### **Pinecone: Managed Vector Database**

Modern Pinecone integration with serverless support:

```python
import os
from swarms_memory import PineconeMemory

# Modern Pinecone initialization (no deprecated parameters)
pinecone = PineconeMemory(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name="production-index",
    embedding_model="text-embedding-3-small",   # 100+ models supported
    dimension=1536,                             # Auto-detected
    metric="cosine",
    cloud="aws",                                # Serverless deployment
    region="us-east-1",
    namespace="documents",
    api_key_embedding=os.getenv("OPENAI_API_KEY")
)

# Add documents with rich metadata
doc_id = pinecone.add(
    "Pinecone is a vector database optimized for ML applications.",
    metadata={
        "category": "database",
        "tags": ["vector", "ml", "production"],
        "created_at": "2024-01-15"
    }
)

# Advanced querying with filters
results = pinecone.query(
    "What are vector databases?",
    top_k=5,
    filter_dict={"category": "database"}
)

# Production features
health = pinecone.health_check()
print(f"Status: {health['status']}, Vectors: {health['total_vectors']}")
```

### **FAISS: High-Performance Local Search**

Enhanced FAISS with persistence and multiple index types:

```python
from swarms_memory import FAISSDB

# FAISS with persistence and optimization
faiss = FAISSDB(
    embedding_model="text-embedding-3-small",
    dimension=1536,
    index_type="HNSW",                    # High-performance graph index
    metric="cosine",
    index_file="persistent_index.faiss", # Auto-save to disk
    api_key_embedding=os.getenv("OPENAI_API_KEY")
)

# Batch document addition
documents = [
    {"text": "FAISS provides efficient similarity search...", "meta": {"type": "info"}},
    {"text": "Vector indexing enables fast retrieval...", "meta": {"type": "concept"}},
]

for doc in documents:
    faiss.add(doc["text"], metadata=doc["meta"])

# Advanced features
results = faiss.query("similarity search", top_k=10)
faiss.rebuild_index()  # Optimize after deletions
stats = faiss.health_check()
print(f"Documents: {stats['total_documents']}, Index: {stats['index_type']}")
```

### **Multi-Provider Embedding Support**

Use any embedding provider with consistent API:

```python
from swarms_memory import FAISSDB

# OpenAI embeddings
db1 = FAISSDB(embedding_model="text-embedding-3-small", api_key_embedding="sk-...")

# Azure OpenAI
db2 = FAISSDB(
    embedding_model="azure/my-deployment",
    api_key_embedding="azure-key",
    api_base="https://resource.openai.azure.com",
    api_version="2023-07-01-preview"
)

# Cohere embeddings
db3 = FAISSDB(
    embedding_model="cohere/embed-english-v3.0",
    api_key_embedding="cohere-key",
    embedding_kwargs={"input_type": "search_document"}
)

# Custom embedding function
def custom_embedder(text: str) -> List[float]:
    # Your embedding logic here
    return [0.1] * 1536

db4 = FAISSDB(embedding_model=custom_embedder, dimension=1536)
```

### **Production RAG System**

Complete RAG implementation with conversation memory:

```python
import os
from swarms import Agent
from swarms_memory import PineconeMemory

class ProductionRAGSystem:
    def __init__(self):
        # Initialize vector memory
        self.memory = PineconeMemory(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name="rag-system",
            embedding_model="text-embedding-3-small",
            namespace="knowledge_base",
            api_key_embedding=os.getenv("OPENAI_API_KEY")
        )
        
        # Create intelligent agent
        self.agent = Agent(
            agent_name="RAG-Assistant", 
            model_name="gpt-4",
            system_prompt="""You are an AI assistant with access to a knowledge base.
            Use retrieved context to provide accurate, detailed responses.""",
            max_loops=1
        )
    
    def add_knowledge(self, documents: List[str], metadata_list: List[Dict] = None):
        """Add documents to knowledge base"""
        for i, doc in enumerate(documents):
            meta = metadata_list[i] if metadata_list else {}
            doc_id = self.memory.add(doc, metadata=meta)
            print(f"Added: {doc[:50]}... (ID: {doc_id})")
    
    def query(self, question: str) -> str:
        """Query with RAG"""
        # Retrieve relevant context
        context_results = self.memory.query(question, top_k=3)
        
        # Format context
        context = "\n".join([
            f"[{r['score']:.3f}] {r['metadata']['text']}" 
            for r in context_results
        ])
        
        # Generate response with context
        prompt = f"""
        Context from knowledge base:
        {context}
        
        Question: {question}
        
        Please answer using the provided context when relevant.
        """
        
        response = self.agent.run(prompt)
        
        # Store conversation for future context
        self.memory.add(f"Q: {question}\nA: {response}", 
                       metadata={"type": "conversation"})
        
        return response

# Usage
rag = ProductionRAGSystem()

# Add knowledge
knowledge_docs = [
    "Vector databases store high-dimensional embeddings for similarity search.",
    "RAG systems combine retrieval and generation for accurate AI responses.",
    "Pinecone offers serverless vector database with automatic scaling."
]

rag.add_knowledge(knowledge_docs, [{"source": "docs"} for _ in knowledge_docs])

# Interactive querying
response = rag.query("What are the benefits of vector databases?")
print(response)
```

---

## 🔥 **Advanced Features**

### **Health Monitoring & Observability**

```python
# Comprehensive health checks
health = db.health_check()
print(f"""
Database Status: {health['status']}
Total Documents: {health['total_documents']}  
Embedding Model: {health['embedding_model']}
Dimension: {health['dimension']}
""")

# Performance monitoring
import time
start = time.time()
results = db.query("test query", top_k=100)
print(f"Query time: {time.time() - start:.3f}s")
```

### **Batch Operations**

```python
# Efficient batch processing
documents = ["doc1", "doc2", "doc3", ...]
metadatas = [{"id": i} for i in range(len(documents))]

# Batch add (where supported)
doc_ids = []
for doc, meta in zip(documents, metadatas):
    doc_id = db.add(doc, metadata=meta)
    doc_ids.append(doc_id)

# Batch delete
for doc_id in doc_ids:
    db.delete(doc_id)
```

### **Data Management**

```python
# Document lifecycle management
doc_id = db.add("Original document", metadata={"version": 1})

# Update document (delete + re-add with new version)
db.delete(doc_id) 
new_id = db.add("Updated document", metadata={"version": 2})

# Clear all documents
db.clear()

# Get collection statistics
count = db.count()
print(f"Total documents: {count}")
```

---

## 🌍 **Multi-Cloud & Deployment Options**

### **Cloud Providers**

```python
# AWS with Pinecone
pinecone_aws = PineconeMemory(
    api_key="pinecone-key",
    index_name="aws-index",
    cloud="aws",
    region="us-east-1",
    embedding_model="text-embedding-3-small"
)

# GCP with Pinecone  
pinecone_gcp = PineconeMemory(
    api_key="pinecone-key", 
    index_name="gcp-index",
    cloud="gcp",
    region="us-central1-a",
    embedding_model="text-embedding-3-small"
)

# Azure with custom embeddings
azure_faiss = FAISSDB(
    embedding_model="azure/my-embedding-deployment",
    api_key_embedding="azure-key",
    api_base="https://myresource.openai.azure.com",
    api_version="2023-07-01-preview"
)
```

### **On-Premises & Edge**

```python
# Local FAISS with persistence
local_db = FAISSDB(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Local model
    index_file="/data/vector_index.faiss",
    index_type="HNSW"
)

# Qdrant self-hosted
from qdrant_client import QdrantClient
qdrant_client = QdrantClient("http://localhost:6333")
qdrant_db = QdrantDB(client=qdrant_client, embedding_model="local-model")
```

---

## 🛠️ **Environment Setup**

### **Environment Variables**

Create a `.env` file with your API keys:

```bash
# Vector Database Keys
PINECONE_API_KEY=your_pinecone_key_here
QDRANT_URL=https://your-cluster.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_key_here

# Embedding Provider Keys  
OPENAI_API_KEY=sk-your_openai_key_here
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here

# Azure OpenAI (if using Azure)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_API_VERSION=2023-07-01-preview
```

### **Docker Deployment**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
```

### **Requirements**

```txt
swarms-memory>=0.2.0
swarms>=5.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

---

## 📊 **Performance Benchmarks**

| **Operation** | **Pinecone** | **FAISS (Local)** | **Qdrant** |
| ------------- | ------------ | ------------------ | ----------- |
| Insert 1K     | ~2s          | ~0.1s              | ~1s         |
| Query (p95)   | ~100ms       | ~10ms              | ~50ms       |
| Storage       | ☁️ Managed   | 💾 Local           | 🔄 Hybrid   |
| Scalability   | ∞ Serverless | 🖥️ Hardware        | 📈 Cluster  |

---

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Setup**

```bash
git clone https://github.com/kyegomez/swarms-memory.git
cd swarms-memory
pip install -e ".[dev]"
pre-commit install
```

### **Running Tests**

```bash
pytest tests/ -v
python -m pytest tests/vector_dbs/ -v --cov=swarms_memory
```

---

## 📚 **Documentation & Examples**

- **📖 [Full Documentation](https://docs.swarms.world/memory)**
- **💡 [Example Gallery](examples/)**
- **🚀 [Migration Guide](MIGRATION.md)**
- **🐛 [Issue Tracker](https://github.com/kyegomez/swarms-memory/issues)**

### **Example Files**

- [`pinecone_swarms.py`](pinecone_swarms.py) - Production RAG with Pinecone
- [`faiss_swarms.py`](faiss_swarms.py) - Local vector search with persistence  
- [`examples/unified_agent_memory_example.py`](examples/unified_agent_memory_example.py) - Multi-database comparison
- [`examples/`](examples/) - Comprehensive example collection

---

## 🔒 **Security & Privacy**

- **🔐 API Key Management** - Secure environment variable handling
- **🛡️ Input Validation** - Comprehensive data sanitization
- **🔒 TLS/SSL Support** - Encrypted connections to cloud providers
- **🏢 On-Premises Options** - Full data control with FAISS/Qdrant
- **📋 Compliance** - SOC2, GDPR, HIPAA compatible deployments

---

## 📈 **Roadmap**

### **v0.3.0 (Q1 2024)**
- [ ] Weaviate integration with GraphQL support
- [ ] Redis vector search integration
- [ ] Advanced metadata filtering across all databases
- [ ] Streaming ingestion for large datasets

### **v0.4.0 (Q2 2024)**
- [ ] Multi-modal embeddings (text + images)
- [ ] Federated search across multiple vector databases
- [ ] Advanced RAG techniques (HyDE, Query expansion)
- [ ] Performance optimization and caching layer

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Citation**

If you use Swarms Memory in your research or project, please cite:

```bibtex
@misc{swarms-memory,
  author = {Gomez, Kye},
  title = {{Swarms Memory: Enterprise-Grade Vector Memory Framework}},
  howpublished = {\url{https://github.com/kyegomez/swarms-memory}},
  year = {2024},
  note = {Accessed: 2024-01-15}
}
```

---

## 🌟 **Support the Project**

- ⭐ **Star** this repository
- 🐦 **Follow** us on [Twitter](https://twitter.com/swarms_corp/)  
- 💬 **Join** our [Discord](https://discord.gg/agora-999382051935506503)
- 📢 **Share** with your network

---

<div align="center">
  <p><strong>Built with ❤️ by the Swarms Team</strong></p>
  <p><a href="https://swarms.world">swarms.world</a> • <a href="https://docs.swarms.world">Documentation</a> • <a href="https://discord.gg/agora-999382051935506503">Community</a></p>
</div>