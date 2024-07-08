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
