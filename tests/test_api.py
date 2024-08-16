import pytest
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_URL = os.getenv("BASE_SWARMS_MEMORY_URL")


@pytest.fixture(scope="module")
def collection_name():
    """
    Fixture for creating a unique collection name.
    """
    return "my_collection"


@pytest.fixture(scope="module")
def collection_url():
    """
    Fixture to return the base collection URL.
    """
    return f"{BASE_URL}/collections"


@pytest.fixture(scope="module")
def setup_collection(collection_url, collection_name):
    """
    Fixture to create a collection before tests and delete it after.
    """
    # Create the collection
    data = {"name": collection_name}
    response = requests.post(collection_url, json=data)
    assert response.status_code == 201
    collection_id = response.json().get("collection_id")

    yield collection_id

    # Delete the collection after tests
    requests.delete(f"{collection_url}/{collection_id}")


def test_create_collection(setup_collection):
    """
    Test the creation of a new collection.
    """
    # The setup_collection fixture already creates the collection and returns the ID.
    assert setup_collection is not None


def test_add_documents(collection_url, setup_collection):
    """
    Test adding documents to a collection.
    """
    url = f"{collection_url}/{setup_collection}/documents"
    data = {
        "documents": [
            "This is a document about pineapples",
            "This is a document about oranges",
        ],
    }
    response = requests.post(url, json=data)
    assert response.status_code == 200
    response_json = response.json()
    assert "document_ids" in response_json

    # Return the first document ID for use in other tests
    return response_json["document_ids"][0]


def test_query_documents(collection_url, setup_collection):
    """
    Test querying documents in a collection.
    """
    url = f"{collection_url}/{setup_collection}/documents"
    data = {
        "query_texts": ["This is a query document about Hawaii"],
        "n_results": 2,
    }
    response = requests.get(url, json=data)
    assert response.status_code == 200
    response_json = response.json()
    assert "results" in response_json


def test_delete_document(
    collection_url, setup_collection, test_add_documents
):
    """
    Test deleting a document from a collection.
    """
    document_id = test_add_documents  # Use the document ID from the add documents test
    url = (
        f"{collection_url}/{setup_collection}/documents/{document_id}"
    )
    response = requests.delete(url)
    assert response.status_code == 200
    response_json = response.json()
    assert "deleted" in response_json
