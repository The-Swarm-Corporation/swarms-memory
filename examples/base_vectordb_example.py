# Example usage of the BaseVectorDatabase class

# First, we need to create a concrete implementation of the BaseVectorDatabase class.
# Let's create a simple in-memory vector database for demonstration purposes.

from loguru import logger
from swarms_memory import BaseVectorDatabase


class InMemoryVectorDatabase(BaseVectorDatabase):
    """Concrete implementation of the BaseVectorDatabase using an in-memory dictionary."""

    def __init__(self):
        """Initialize an empty in-memory database."""
        self.database = {}
        self.counter = 0  # To simulate auto-incrementing IDs

    def connect(self):
        """Establish a connection to the in-memory database (no action needed here)."""
        logger.info("Connected to in-memory vector database.")

    def close(self):
        """Close the connection to the database (no action needed here)."""
        logger.info("Closed connection to in-memory vector database.")

    def query(self, query: str):
        """Simulate querying the database by returning stored entries that match the query."""
        logger.info(f"Executing query: {query}")
        # For demonstration, we'll just return all entries since we don't have actual query parsing
        return self.fetch_all()

    def fetch_all(self):
        """Fetch all records from the in-memory database."""
        logger.info("Fetching all records from the database.")
        return list(self.database.values())

    def fetch_one(self):
        """Fetch one record from the in-memory database (not implemented for simplicity)."""
        logger.warning(
            "fetch_one not implemented in in-memory database."
        )

    def add(self, doc: str):
        """Add a new document to the in-memory database."""
        self.counter += 1
        self.database[self.counter] = {"id": self.counter, "doc": doc}
        logger.info(f"Added document with ID {self.counter}.")

    def get(self, query: str):
        """Retrieve a record based on the query (ID in this case)."""
        doc_id = int(query)
        logger.info(f"Retrieving document with ID: {doc_id}")
        return self.database.get(doc_id)

    def update(self, doc):
        """Update a record in the in-memory database."""
        doc_id = doc.get("id")
        if doc_id in self.database:
            self.database[doc_id] = doc
            logger.info(f"Updated document with ID {doc_id}.")
        else:
            logger.warning(
                f"Document with ID {doc_id} not found for update."
            )

    def delete(self, message):
        """Delete a record from the in-memory database based on the ID."""
        doc_id = int(message)
        if doc_id in self.database:
            del self.database[doc_id]
            logger.info(f"Deleted document with ID {doc_id}.")
        else:
            logger.warning(
                f"Document with ID {doc_id} not found for deletion."
            )

    def print_all(self):
        """Print all records in the database."""
        for entry in self.database.values():
            print(entry)


# Example usage of the InMemoryVectorDatabase class
db = InMemoryVectorDatabase()
db.connect()

# Add some documents to the database
db.add("This is the first document.")
db.add("This is the second document.")
db.add("This is the third document.")

# Query the database to fetch all entries
all_entries = db.fetch_all()
print("All entries in the database:")
for entry in all_entries:
    print(entry)

# Retrieve a specific document by ID
doc_id_to_retrieve = 2
retrieved_doc = db.get(str(doc_id_to_retrieve))
print(
    f"\nRetrieved document with ID {doc_id_to_retrieve}: {retrieved_doc}"
)

# Update a document
db.update({"id": 2, "doc": "This is the updated second document."})

# Retrieve the updated document
updated_doc = db.get("2")
print(f"\nUpdated document with ID 2: {updated_doc}")

# Delete a document
db.delete("1")

# Print all entries after deletion
print("\nEntries after deletion:")
db.print_all()

# Close the database connection
db.close()
