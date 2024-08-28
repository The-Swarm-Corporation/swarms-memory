# Example usage of the ChromaDB class

# Import necessary libraries
from swarms_memory import ChromaDB


# Define the folder where documents are stored (optional)
docs_folder = "path/to/docs"  # Replace with your actual folder path

# Create an instance of the ChromaDB class
chroma_db = ChromaDB(
    metric="cosine",  # Similarity metric to use
    output_dir="results",  # Name of the collection to store results
    docs_folder=docs_folder,  # Optional folder to traverse for documents
)


# Function to add a document to the ChromaDB
def add_document(document: str):
    """Add a sample document to the ChromaDB."""
    doc_id = chroma_db.add(
        document
    )  # Add the document and get its ID
    print(f"Added document with ID: {doc_id}")


# Function to query documents from ChromaDB
def query_documents(query_text: str):
    """Query documents from the ChromaDB."""
    results = chroma_db.query(query_text)  # Query the database
    print(
        "Query Results:\n", results
    )  # Print the retrieved documents


# Example document to add
sample_document = "This is a sample document containing information about machine learning."
add_document(sample_document)  # Adding the sample document

# Querying the database for similar documents
query_text = "information about machine learning"
query_documents(query_text)  # Perform the query

# If you want to traverse a directory and add all supported files to the ChromaDB
if docs_folder:
    chroma_db.traverse_directory(
        docs_folder
    )  # Traverse and add documents from the specified folder
