# SingleStore Vector Database

SingleStore is a distributed SQL database that provides high-performance vector similarity search capabilities. This implementation uses the official SingleStore Python library to provide efficient vector storage and similarity search for your RAG (Retrieval-Augmented Generation) system.

## Features

- High-performance vector similarity search using SingleStore's native vector operations
- Automatic vector indexing for fast similarity search
- Support for custom embedding models and functions
- Document preprocessing and postprocessing capabilities
- Namespace support for document organization
- Comprehensive error handling and logging
- Built-in connection management using the official SingleStore Python client

## Installation

```bash
pip install singlestoredb sentence-transformers numpy
```

## Configuration

Set up your SingleStore credentials in your environment variables:

```bash
SINGLESTORE_HOST=your_host
SINGLESTORE_USER=your_user
SINGLESTORE_PASSWORD=your_password
```

## Usage

### Basic Usage

```python
from swarms_memory.vector_dbs import SingleStoreDB

# Initialize the database
db = SingleStoreDB(
    database="vectordb",
    table_name="embeddings",
    dimension=768  # matches the embedding model's dimension
)

# Add a document
doc_id = db.add("This is a sample document")

# Query similar documents
results = db.query("Find similar documents", top_k=3)

# Get a specific document
doc = db.get(doc_id)

# Delete a document
db.delete(doc_id)
```

### Advanced Usage

#### Custom Embedding Model

```python
from sentence_transformers import SentenceTransformer

# Use a different embedding model
db = SingleStoreDB(
    database="vectordb",
    embedding_model=SentenceTransformer("all-mpnet-base-v2"),
    dimension=768
)
```

#### Custom Embedding Function

```python
def custom_embedding_function(text: str) -> List[float]:
    # Your custom embedding logic here
    return [0.1, 0.2, ...]  # Must match dimension

db = SingleStoreDB(
    database="vectordb",
    embedding_function=custom_embedding_function,
    dimension=768
)
```

#### Document Preprocessing

```python
def preprocess_text(text: str) -> str:
    # Custom preprocessing logic
    return text.lower().strip()

db = SingleStoreDB(
    database="vectordb",
    preprocess_function=preprocess_text
)
```

#### Result Postprocessing

```python
def postprocess_results(results: List[Dict]) -> List[Dict]:
    # Custom postprocessing logic
    return sorted(results, key=lambda x: x["similarity"], reverse=True)

db = SingleStoreDB(
    database="vectordb",
    postprocess_function=postprocess_results
)
```

#### Using Namespaces

```python
# Initialize with a default namespace
db = SingleStoreDB(
    database="vectordb",
    namespace="project1"
)

# Add document to the namespace
db.add("Document in project1")

# Query within the namespace
results = db.query("Query in project1")

# Query in a different namespace
results = db.query("Query in project2", namespace="project2")
```

## API Reference

### SingleStoreDB Class

```python
class SingleStoreDB:
    def __init__(
        self,
        host: str = None,
        port: int = 3306,
        user: str = None,
        password: str = None,
        database: str = "vectordb",
        table_name: str = "embeddings",
        dimension: int = 768,
        embedding_model: Optional[Any] = None,
        embedding_function: Optional[Callable] = None,
        preprocess_function: Optional[Callable] = None,
        postprocess_function: Optional[Callable] = None,
        namespace: str = "",
        verbose: bool = False
    )
```

#### Methods

- `add(document: str, metadata: Dict = None, embedding: List[float] = None, doc_id: str = None) -> str`
  Add a document to the database.

- `query(query: str, top_k: int = 5, embedding: List[float] = None, namespace: str = None) -> List[Dict]`
  Query similar documents.

- `delete(doc_id: str, namespace: str = None) -> bool`
  Delete a document from the database.

- `get(doc_id: str, namespace: str = None) -> Optional[Dict]`
  Retrieve a specific document.

## Performance Optimization

### Vector Indexing

The implementation automatically creates a vector index on the embedding column:

```sql
VECTOR INDEX vec_idx (embedding) DIMENSION = {dimension}
```

This index significantly improves the performance of similarity search queries.

### Connection Management

The implementation uses the official SingleStore Python client with proper connection management:
- Automatic connection pooling
- Context managers for cursor operations
- Proper cleanup of resources

### Query Optimization

- Uses native SingleStore vector operations for similarity search
- Efficient handling of vector data using SingleStore's array type
- Proper indexing for fast lookups and filtering

## Error Handling

The implementation includes comprehensive error handling for:
- Connection issues
- Query execution errors
- Invalid embeddings
- Missing documents
- Authentication failures

All errors are logged using the `loguru` logger for easy debugging.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.
