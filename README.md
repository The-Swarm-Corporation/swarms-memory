
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


[![GitHub issues](https://img.shields.io/github/issues/kyegomez/swarms)](https://github.com/kyegomez/swarms-memory/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/swarms)](https://github.com/kyegomez/swarms-memory/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/swarms)](https://github.com/kyegomez/swarms-memory/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/swarms-memory)](https://github.com/kyegomez/swarms-memory/blob/main/LICENSE)[![GitHub star chart](https://img.shields.io/github/stars/kyegomez/swarms-memory?style=social)](https://star-history.com/#kyegomez/swarms)[![Dependency Status](https://img.shields.io/librariesio/github/kyegomez/swarms)](https://libraries.io/github/kyegomez/swarms) [![Downloads](https://static.pepy.tech/badge/swarms-memory/month)](https://pepy.tech/project/swarms-memory)

[![Join the Agora discord](https://img.shields.io/discord/1110910277110743103?label=Discord&logo=discord&logoColor=white&style=plastic&color=d7b023)![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/swarmsmemory)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&title=Swarms%20-%20the%20future%20of%20AI) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&t=Swarms%20-%20the%20future%20of%20AI) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Swarms%20-%20the%20future%20of%20AI) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Swarms%20-%20the%20future%20of%20AI%20%23swarms%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fswarms)


Here's a more detailed and larger table with descriptions and website links for each RAG system:

| **RAG System** | **Status**  | **Description**                                                                 | **Documentation**                                     | **Website**                      |
|----------------|-------------|---------------------------------------------------------------------------------|-------------------------------------------------------|----------------------------------|
| **ChromaDB**   | Available   | A high-performance, distributed database optimized for handling large-scale AI tasks. | [ChromaDB Documentation](swarms_memory/memory/chromadb.md) | [ChromaDB](https://chromadb.com) |
| **Pinecone**   | Available   | A fully managed vector database that makes it easy to add vector search to your applications. | [Pinecone Documentation](swarms_memory/memory/pinecone.md) | [Pinecone](https://pinecone.io) |
| **Redis**      | Coming Soon | An open-source, in-memory data structure store, used as a database, cache, and message broker. | [Redis Documentation](swarms_memory/memory/redis.md) | [Redis](https://redis.io)       |
| **Faiss**      | Coming Soon | A library for efficient similarity search and clustering of dense vectors, developed by Facebook AI. | [Faiss Documentation](swarms_memory/memory/faiss.md) | [Faiss](https://faiss.ai)       |
| **HNSW**       | Coming Soon | A graph-based algorithm for approximate nearest neighbor search, known for its speed and accuracy. | [HNSW Documentation](swarms_memory/memory/hnsw.md)   | [HNSW](https://github.com/nmslib/hnswlib) |

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

