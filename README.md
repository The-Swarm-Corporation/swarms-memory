[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Swarms Memory
A collection of memory systems for `swarms` such as Pinecone, ChromaDB, Weaviate


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
```

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

