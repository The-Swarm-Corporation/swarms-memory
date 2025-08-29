# Migration Guide: Swarms-Memory v0.1.3+

This guide helps you migrate from older versions of swarms-memory to the new standardized interface with unified LiteLLM embeddings and consistent APIs.

## 🚀 What's New

### Major Improvements
- ✅ **Unified LiteLLM Embeddings**: Access to 100+ embedding providers (OpenAI, Cohere, Voyage, Azure, etc.)
- ✅ **Consistent APIs**: Standardized `add()`, `query()`, `delete()`, `clear()` methods across all databases
- ✅ **Pinecone v3+ Support**: Updated with latest Pinecone API (removed deprecated `environment` parameter)
- ✅ **Enhanced Base Class**: Proper abstract methods with type hints
- ✅ **Better Error Handling**: Consistent exception patterns
- ✅ **Modern Dependencies**: Updated package versions and constraints

### Breaking Changes
- 🔄 **PineconeMemory**: Removed deprecated `environment` parameter
- 🔄 **All Wrappers**: Changed `embedding_model` parameter behavior
- 🔄 **Query Methods**: Added `return_metadata` parameter for consistent return types
- 🔄 **Add Methods**: Now return document ID strings instead of None

## 📋 Migration Steps

### 1. Update Dependencies

**Old pyproject.toml:**
```toml
[tool.poetry.dependencies]
pinecone = "*"
qdrant-client = "*"
# Missing litellm
```

**New pyproject.toml:**
```toml
[tool.poetry.dependencies]
pinecone = "^3.0.0"           # Pinecone v3+
qdrant-client = "^1.7.0"      # Updated version
litellm = "^1.0.0"            # New: Unified embeddings
python-dotenv = "^1.0.0"      # New: Environment management
swarms = {version = "*", optional = true}  # New: Agent integration
```

### 2. Migrate PineconeMemory

**Old Code:**
```python
from swarms_memory.vector_dbs.pinecone_wrapper import PineconeMemory

# ❌ DEPRECATED: environment parameter
vector_db = PineconeMemory(
    api_key="your-key",
    environment="us-east-1-gcp",  # ❌ DEPRECATED
    index_name="my-index",
    embedding_model=None  # Used SentenceTransformer by default
)

# Old add method returned None
vector_db.add("document text")

# Old query method always returned List[Dict]
results = vector_db.query("query text", top_k=5)
```

**New Code:**
```python
from swarms_memory.vector_dbs.pinecone_wrapper import PineconeMemory

# ✅ NEW: No environment parameter, LiteLLM embeddings
vector_db = PineconeMemory(
    api_key="your-key",
    index_name="my-index",
    cloud="aws",           # ✅ Use cloud instead of environment
    region="us-east-1",    # ✅ Specify region
    embedding_model="text-embedding-3-small",  # ✅ LiteLLM model
    # Alternative models:
    # embedding_model="cohere/embed-english-v3.0",
    # embedding_model="voyage/voyage-01", 
    # embedding_model="all-MiniLM-L6-v2",  # SentenceTransformer still supported
)

# New add method returns document ID
doc_id = vector_db.add("document text", metadata={"source": "migration"})
print(f"Added document: {doc_id}")

# New query method has flexible return types
# For backward compatibility (returns string):
text_result = vector_db.query("query text", top_k=5)

# For detailed results (returns List[Dict]):
detailed_results = vector_db.query("query text", top_k=5, return_metadata=True)
```

### 3. Migrate ChromaDB

**Old Code:**
```python
from swarms_memory.vector_dbs.chroma_db_wrapper import ChromaDB

# Old ChromaDB with limited embedding options
vector_db = ChromaDB(
    output_dir="my_collection"
    # No embedding model configuration
)

vector_db.add("document")  # Returns UUID
results = vector_db.query("query")  # Returns string
```

**New Code:**
```python
from swarms_memory.vector_dbs.chroma_db_wrapper import ChromaDB

# ✅ NEW: LiteLLM embeddings support
vector_db = ChromaDB(
    output_dir="my_collection",
    embedding_model="text-embedding-3-small",  # ✅ LiteLLM
    verbose=True  # Better logging
)

# Standardized interface
doc_id = vector_db.add("document", metadata={"type": "text"})

# Flexible return types
text_result = vector_db.query("query", top_k=5)
detailed_results = vector_db.query("query", top_k=5, return_metadata=True)

# New methods
success = vector_db.delete(doc_id)
cleared = vector_db.clear()
```

### 4. Migrate FAISS

**Old Code:**
```python
from swarms_memory.vector_dbs.faiss_wrapper import FAISSDB

vector_db = FAISSDB(
    dimension=768,
    embedding_model=None  # Used SentenceTransformer
)

vector_db.add("document")  # Returned None
results = vector_db.query("query")  # Returned List[Dict]
```

**New Code:**
```python
from swarms_memory.vector_dbs.faiss_wrapper import FAISSDB

# ✅ NEW: Auto-detect dimension from embedding model
vector_db = FAISSDB(
    embedding_model="text-embedding-3-small",  # ✅ LiteLLM
    # dimension will be auto-detected as 1536
    index_type="Flat",
    metric="cosine"
)

# Standardized interface
doc_id = vector_db.add("document", metadata={"source": "migration"})
text_result = vector_db.query("query", top_k=5)
detailed_results = vector_db.query("query", top_k=5, return_metadata=True)

# Note: FAISS doesn't support individual delete (database limitation)
cleared = vector_db.clear()  # This works
```

### 5. Migrate SingleStoreDB

**Old Code:**
```python
from swarms_memory.vector_dbs.singlestore_wrapper import SingleStoreDB

vector_db = SingleStoreDB(
    host="host",
    user="user", 
    password="password",
    database="db",
    table_name="vectors",
    embedding_model=None  # Used SentenceTransformer
)
```

**New Code:**
```python
from swarms_memory.vector_dbs.singlestore_wrapper import SingleStoreDB

# ✅ NEW: LiteLLM embeddings with auto-dimension detection
vector_db = SingleStoreDB(
    host="host",
    user="user",
    password="password", 
    database="db",
    table_name="vectors",
    embedding_model="text-embedding-3-small",  # ✅ LiteLLM
    # dimension auto-detected as 1536
)

# Enhanced query with metadata filtering
results = vector_db.query(
    "query text",
    top_k=5,
    metadata_filter={"source": "important"},
    return_metadata=True
)

# Full CRUD operations
doc_id = vector_db.add("document", metadata={"type": "knowledge"})
success = vector_db.delete(doc_id)
cleared = vector_db.clear()
```

## 🎯 LiteLLM Embedding Models

### Supported Providers

```python
# OpenAI (default)
embedding_model="text-embedding-3-small"      # 1536 dimensions
embedding_model="text-embedding-3-large"      # 3072 dimensions
embedding_model="text-embedding-ada-002"      # 1536 dimensions

# Azure OpenAI
embedding_model="azure/my-deployment"
# Additional params: api_base, api_version, api_key

# Cohere
embedding_model="cohere/embed-english-v3.0"   # 1024 dimensions
embedding_model="cohere/embed-multilingual-v3.0"

# Voyage AI
embedding_model="voyage/voyage-01"             # 1536 dimensions
embedding_model="voyage/voyage-large-2"       # 1536 dimensions

# AWS Bedrock
embedding_model="bedrock/amazon.titan-embed-text-v1"

# Hugging Face
embedding_model="huggingface/sentence-transformers/all-MiniLM-L6-v2"

# SentenceTransformers (still supported)
embedding_model="all-MiniLM-L6-v2"           # 384 dimensions
embedding_model="all-mpnet-base-v2"          # 768 dimensions
```

### Environment Variables

Set these for different providers:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Cohere
COHERE_API_KEY=your_cohere_key

# Voyage AI
VOYAGE_API_KEY=your_voyage_key

# Azure OpenAI
AZURE_API_KEY=your_azure_key
AZURE_API_BASE=https://your-resource.openai.azure.com
AZURE_API_VERSION=2023-07-01-preview
```

## 🔧 Advanced Configuration

### Custom Embedding Configuration

```python
# Provider-specific settings
vector_db = PineconeMemory(
    api_key="your-key",
    index_name="my-index",
    embedding_model="cohere/embed-english-v3.0",
    # LiteLLM provider-specific params:
    input_type="search_document",  # For Cohere
    api_key="your-cohere-key",     # If not in env
)

# Azure OpenAI with custom settings
vector_db = ChromaDB(
    output_dir="azure_collection",
    embedding_model="azure/my-embedding-deployment",
    api_base="https://your-resource.openai.azure.com",
    api_version="2023-07-01-preview",
    api_key="your-azure-key"
)
```

### Swarms Agent Integration

```python
from swarms import Agent, OpenAIChat
from swarms_memory.vector_dbs.qdrant_wrapper import QdrantVectorDatabase

# Setup vector database
vector_db = QdrantVectorDatabase(
    collection_name="agent_memory",
    embedding_model="text-embedding-3-small",
    location=":memory:"
)

# Add knowledge
vector_db.add("Swarms enables multi-agent coordination.")

# Create agent with memory
def memory_retrieval(query: str) -> str:
    results = vector_db.query(query, top_k=3, return_metadata=True)
    return "\\n".join([r['metadata']['text'] for r in results])

agent = Agent(
    agent_name="MemoryAgent", 
    llm=OpenAIChat(model_name="gpt-4"),
    system_prompt="Use the memory context to answer questions."
)

# Use agent with memory
query = "What is swarms?"
context = memory_retrieval(query)
response = agent.run(f"Context: {context}\\n\\nQuestion: {query}")
```

## 🚨 Common Issues & Solutions

### Issue: Import Errors
```python
# ❌ Error: ModuleNotFoundError: No module named 'litellm'
ImportError: litellm is required for LiteLLMEmbeddings

# ✅ Solution: Install required dependencies
pip install litellm python-dotenv
```

### Issue: Pinecone Environment Parameter
```python
# ❌ Error: Using deprecated 'environment' parameter
TypeError: __init__() got an unexpected keyword argument 'environment'

# ✅ Solution: Use cloud and region instead
vector_db = PineconeMemory(
    api_key="key",
    index_name="index", 
    cloud="aws",        # Instead of environment
    region="us-east-1"
)
```

### Issue: Embedding Dimension Mismatch
```python
# ❌ Error: Dimension mismatch between embedding model and vector database
ValueError: Expected 768 dimensions, got 1536

# ✅ Solution: Let the system auto-detect dimensions
vector_db = FAISSDB(
    embedding_model="text-embedding-3-small"
    # Don't specify dimension - it will be auto-detected as 1536
)
```

### Issue: API Key Not Found
```python
# ❌ Error: No API key found for provider
AuthenticationError: Invalid API key provided

# ✅ Solution: Set environment variables or pass explicitly
import os
os.environ["OPENAI_API_KEY"] = "your-key"

# Or pass directly:
vector_db = PineconeMemory(
    api_key="pinecone-key",
    embedding_model="text-embedding-3-small",
    api_key="openai-key"  # For LiteLLM
)
```

## 📚 Additional Resources

- [LiteLLM Documentation](https://docs.litellm.ai/docs/embedding/supported_embedding)
- [Pinecone v3 Migration Guide](https://docs.pinecone.io/docs/migrate)
- [Swarms Framework Documentation](https://github.com/kyegomez/swarms)

## 🤝 Need Help?

If you encounter issues during migration:

1. Check the [examples/](examples/) directory for working code
2. Review the [tests/](tests/) for expected behavior
3. Open an issue on GitHub with your specific error and code

---

**Happy migrating! 🎉**