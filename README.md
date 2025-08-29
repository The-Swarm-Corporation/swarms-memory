<div align="center">
  <a href="https://swarms.world">
    <h1>Swarms Memory</h1>
  </a>
</div>
<p align="center">
  <em>The Enterprise-Grade Production-Ready RAG Framework</em>
</p>

<p align="center">
    <a href="https://pypi.org/project/swarms/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/swarms?style=for-the-badge&color=3670A0">
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


Here's a more detailed and larger table with descriptions and website links for each RAG system:

| **RAG System**  | **Status**  | **Description**                                                                                      | **Website**                                |
| --------------- | ----------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **ChromaDB**    | Available   | A high-performance, distributed database optimized for handling large-scale AI tasks.                | [ChromaDB](https://chromadb.com)           |
| **Pinecone**    | Available   | A fully managed vector database that makes it easy to add vector search to your applications.        | [Pinecone](https://pinecone.io)            |
| **Redis**       | Coming Soon | An open-source, in-memory data structure store, used as a database, cache, and message broker.       | [Redis](https://redis.io)                  |
| **Faiss**       | Available   | A library for efficient similarity search and clustering of dense vectors, developed by Facebook AI. | [Faiss](https://faiss.ai)                  |
| **SingleStore** | Available   | A distributed SQL database that provides high-performance vector similarity search.                  | [SingleStore](https://www.singlestore.com) |
| **Qdrant**      | Available   | An open-source, massive scale vector search engine written in Rust                                   | [Qdrant](https://qdrant.tech/)             |
| **HNSW**        | Coming Soon | A graph-based algorithm for approximate nearest neighbor search.                                     | [HNSW](https://github.com/nmslib/hnswlib)  |


This table includes a brief description of each system, their current status, links to their documentation, and their respective websites for further information.


### Requirements:
- `python 3.10` 
- `.env` with your respective keys like `PINECONE_API_KEY` can be found in the `.env.examples`

## Install
```bash
$ pip install swarms-memory
```




## Usage

### Pinecone
```python
from typing import List, Dict, Any
from swarms_memory import PineconeMemory


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel
    import torch

    # Custom embedding function using a HuggingFace model
    def custom_embedding_function(text: str) -> List[float]:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = (
            outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        )
        return embeddings

    # Custom preprocessing function
    def custom_preprocess(text: str) -> str:
        return text.lower().strip()

    # Custom postprocessing function
    def custom_postprocess(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        for result in results:
            result["custom_score"] = (
                result["score"] * 2
            )  # Example modification
        return results

    # Initialize the wrapper with custom functions
    wrapper = PineconeMemory(
        api_key="your-api-key",
        environment="your-environment",
        index_name="your-index-name",
        embedding_function=custom_embedding_function,
        preprocess_function=custom_preprocess,
        postprocess_function=custom_postprocess,
        logger_config={
            "handlers": [
                {
                    "sink": "custom_rag_wrapper.log",
                    "rotation": "1 GB",
                },
                {
                    "sink": lambda msg: print(
                        f"Custom log: {msg}", end=""
                    )
                },
            ],
        },
    )

    # Adding documents
    wrapper.add(
        "This is a sample document about artificial intelligence.",
        {"category": "AI"},
    )
    wrapper.add(
        "Python is a popular programming language for data science.",
        {"category": "Programming"},
    )

    # Querying
    results = wrapper.query("What is AI?", filter={"category": "AI"})
    for result in results:
        print(
            f"Score: {result['score']}, Custom Score: {result['custom_score']}, Text: {result['metadata']['text']}"
        )



```


### ChromaDB
```python
from swarms_memory import ChromaDB

chromadb = ChromaDB(
    metric="cosine",
    output_dir="results",
    limit_tokens=1000,
    n_results=2,
    docs_folder="path/to/docs",
    verbose=True,
)

# Add a document
doc_id = chromadb.add("This is a test document.")

# Query the document
result = chromadb.query("This is a test query.")

# Traverse a directory
chromadb.traverse_directory()

# Display the result
print(result)

```


### Faiss

```python
from typing import List, Dict, Any
from swarms_memory.faiss_wrapper import FAISSDB


from transformers import AutoTokenizer, AutoModel
import torch


# Custom embedding function using a HuggingFace model
def custom_embedding_function(text: str) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = (
        outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    )
    return embeddings


# Custom preprocessing function
def custom_preprocess(text: str) -> str:
    return text.lower().strip()


# Custom postprocessing function
def custom_postprocess(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for result in results:
        result["custom_score"] = (
            result["score"] * 2
        )  # Example modification
    return results


# Initialize the wrapper with custom functions
wrapper = FAISSDB(
    dimension=768,
    index_type="Flat",
    embedding_function=custom_embedding_function,
    preprocess_function=custom_preprocess,
    postprocess_function=custom_postprocess,
    metric="cosine",
    logger_config={
        "handlers": [
            {
                "sink": "custom_faiss_rag_wrapper.log",
                "rotation": "1 GB",
            },
            {"sink": lambda msg: print(f"Custom log: {msg}", end="")},
        ],
    },
)

# Adding documents
wrapper.add(
    "This is a sample document about artificial intelligence.",
    {"category": "AI"},
)
wrapper.add(
    "Python is a popular programming language for data science.",
    {"category": "Programming"},
)

# Querying
results = wrapper.query("What is AI?")
for result in results:
    print(
        f"Score: {result['score']}, Custom Score: {result['custom_score']}, Text: {result['metadata']['text']}"
    )
```


### SingleStore
```python
from swarms_memory.vector_dbs.singlestore_wrapper import SingleStoreDB

# Initialize SingleStore with environment variables
db = SingleStoreDB(
    host="your_host",
    port=3306,
    user="your_user",
    password="your_password",
    database="your_database",
    table_name="example_vectors",
    dimension=768,  # Default dimension for all-MiniLM-L6-v2
    namespace="example"
)

# Custom embedding function example (optional)
def custom_embedding_function(text: str) -> List[float]:
    # Your custom embedding logic here
    return embeddings

# Initialize with custom functions
db = SingleStoreDB(
    host="your_host",
    port=3306,
    user="your_user",
    password="your_password",
    database="your_database",
    table_name="example_vectors",
    dimension=768,
    namespace="example",
    embedding_function=custom_embedding_function,
    preprocess_function=lambda x: x.lower(),  # Simple preprocessing
    postprocess_function=lambda x: sorted(x, key=lambda k: k['similarity'], reverse=True)  # Sort by similarity
)

# Add documents with metadata
doc_id = db.add(
    document="SingleStore is a distributed SQL database that combines horizontal scalability with ACID guarantees.",
    metadata={"source": "docs", "category": "database"}
)

# Query similar documents
results = db.query(
    query="How does SingleStore scale?",
    top_k=3,
    metadata_filter={"source": "docs"}
)

# Process results
for result in results:
    print(f"Document: {result['document']}")
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Metadata: {result['metadata']}\n")

# Delete a document
db.delete(doc_id)

# Key features:
# - Built on SingleStore's native vector similarity search
# - Supports custom embedding models and functions
# - Automatic table creation with optimized vector indexing
# - Metadata filtering for refined searches
# - Document preprocessing and postprocessing
# - Namespace support for document organization
# - SSL support for secure connections

# For more examples, see the [SingleStore example](examples/singlestore_wrapper_example.py).
```

### Qdrant with Flexible Embeddings

Qdrant now supports any embedding model through LiteLLM, providing maximum flexibility for local, cloud, and in-memory deployments:

```python
import os
from qdrant_client import QdrantClient, models
from swarms_memory.vector_dbs import QdrantDB

# Example 1: In-memory Qdrant (for experimentation)
in_memory_client = QdrantClient(":memory:")
qdrant_memory = QdrantDB(
    client=in_memory_client,
    collection_name="demo_collection",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    distance=models.Distance.COSINE,
    n_results=3,
)

# Example 2: Qdrant Cloud with OpenAI embeddings
cloud_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),  # Your cluster URL
    api_key=os.getenv("QDRANT_API_KEY"),  # Your API key
)

rag_db = QdrantDB(
    client=cloud_client,
    embedding_model="text-embedding-3-small",
    collection_name="openai_collection",
    distance=models.Distance.COSINE,
    n_results=1,  # Return only most relevant result
)

# Example 3: Local Qdrant server
# Start with: docker run -p 6333:6333 qdrant/qdrant
local_client = QdrantClient("localhost", port=6333)
local_db = QdrantDB(
    client=local_client,
    collection_name="local_collection",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    distance=models.Distance.COSINE,
    n_results=3,
)

# Supports Azure OpenAI, Cohere, Voyage AI, AWS Bedrock, and custom embedding functions
# See examples/qdrant_wrapper_example.py for additional embedding model examples

# Add test documents
knowledge_docs = [
    "Qdrant is a vector database for similarity search.",
    "Vector embeddings represent text as numerical vectors.",
    "Similarity search finds related documents using vector distance.",
    "Qdrant supports in-memory, local, and cloud deployments.",
]

for doc in knowledge_docs:
    rag_db.add(doc)

# Query similar documents
results = rag_db.query("What is vector search?")
print(f"Results: {results}")
```

#### Qdrant Deployment Options

**In-Memory**: Perfect for testing and experimentation
```python
client = QdrantClient(":memory:")
```

**Local Server**: For production workloads with full control
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Qdrant Cloud**: Managed service with scalability and reliability
```python
client = QdrantClient(url="https://your-cluster.qdrant.tech", api_key="your-key")
```

For comprehensive examples including all deployment types, see [`examples/qdrant_wrapper_example.py`](examples/qdrant_wrapper_example.py).

#### Supported Embedding Models

| **Provider**         | **Model(s) / Integration**                                                                                 |
|----------------------|-----------------------------------------------------------------------------------------------------------|
| **OpenAI**           | `"text-embedding-3-small"`, `"text-embedding-3-large"`                                                    |
| **Azure OpenAI**     | `"azure/your-deployment-name"`                                                                            |
| **Cohere**           | `"cohere/embed-english-v3.0"`, `"cohere/embed-multilingual-v3.0"`                                         |
| **Voyage AI**        | `"voyage/voyage-3-large"`, `"voyage/voyage-code-2"`                                                       |
| **AWS Bedrock**      | `"bedrock/amazon.titan-embed-text-v1"`                                                                    |
| **Custom Functions** | Any callable that takes text and returns a vector                                                         |
| **Qdrant Built-in**  | `"qdrant:sentence-transformers/all-MiniLM-L6-v2"`                                                        |

For other vector databases (Pinecone, FAISS, SingleStore), you can use LiteLLM embeddings through their `embedding_function` parameter:

```python
from swarms_memory.embeddings import LiteLLMEmbeddings

# Create embedder
embedder = LiteLLMEmbeddings(model="text-embedding-3-small")

# Use with any vector DB
faiss_db = FAISSDB(embedding_function=embedder.embed_query)
pinecone_db = PineconeMemory(embedding_function=embedder.embed_query)
```

# License
MIT


# Citation
Please cite Swarms in your paper or your project if you found it beneficial in any way! Appreciate you.

```bibtex
@misc{swarms,
  author = {Gomez, Kye},
  title = {{Swarms: The Multi-Agent Collaboration Framework}},
  howpublished = {\url{https://github.com/kyegomez/swarms}},
  year = {2023},
  note = {Accessed: Date}
}
```
