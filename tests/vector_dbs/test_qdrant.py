"""
Tests for QdrantDB wrapper
"""

import pytest
from unittest.mock import patch, MagicMock
from qdrant_client import models
from swarms_memory.vector_dbs.qdrant_wrapper import QdrantDB


@pytest.fixture
def mock_qdrant_client():
    with patch("qdrant_client.QdrantClient") as mock_client:
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_client_instance.collection_exists.return_value = False

        mock_client_instance.get_embedding_size.return_value = 384

        yield mock_client_instance


def test_init(mock_qdrant_client):
    """Test QdrantDB initialization."""
    qdrant_db = QdrantDB(
        client=mock_qdrant_client,
        collection_name="test_collection",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        distance=models.Distance.COSINE,
        n_results=1,
    )

    assert qdrant_db.collection_name == "test_collection"
    assert (
        qdrant_db.model_name
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert qdrant_db.distance == models.Distance.COSINE
    assert qdrant_db.n_results == 1
    assert qdrant_db.client == mock_qdrant_client


def test_setup_collection_new(mock_qdrant_client):
    """Test collection setup when collection doesn't exist."""
    QdrantDB(
        client=mock_qdrant_client, collection_name="test_collection"
    )

    mock_qdrant_client.collection_exists.assert_called_once_with(
        "test_collection"
    )

    mock_qdrant_client.create_collection.assert_called_once()
    create_call = mock_qdrant_client.create_collection.call_args
    assert create_call[1]["collection_name"] == "test_collection"
    assert (
        create_call[1]["vectors_config"].distance
        == models.Distance.COSINE
    )


def test_setup_collection_existing(mock_qdrant_client):
    """Test collection setup when collection already exists."""
    mock_qdrant_client.collection_exists.return_value = True

    QdrantDB(
        client=mock_qdrant_client, collection_name="test_collection"
    )

    mock_qdrant_client.collection_exists.assert_called_once_with(
        "test_collection"
    )

    mock_qdrant_client.create_collection.assert_not_called()


def test_add(mock_qdrant_client):
    """Test adding a document."""
    qdrant_db = QdrantDB(
        client=mock_qdrant_client, collection_name="test_collection"
    )

    doc_id = qdrant_db.add("test document")

    mock_qdrant_client.upsert.assert_called_once()
    upsert_call = mock_qdrant_client.upsert.call_args

    assert upsert_call[1]["collection_name"] == "test_collection"
    assert len(upsert_call[1]["points"]) == 1

    point = upsert_call[1]["points"][0]
    assert point.payload["document"] == "test document"
    assert isinstance(doc_id, str)


def test_query(mock_qdrant_client):
    """Test querying documents."""
    qdrant_db = QdrantDB(
        client=mock_qdrant_client, collection_name="test_collection"
    )

    mock_point = MagicMock()
    mock_point.payload = {"document": "test document"}
    mock_qdrant_client.query_points.return_value.points = [mock_point]

    result = qdrant_db.query("test query")

    mock_qdrant_client.query_points.assert_called_once()
    query_call = mock_qdrant_client.query_points.call_args

    assert query_call[1]["collection_name"] == "test_collection"
    assert query_call[1]["limit"] == 1
    assert query_call[1]["with_payload"] is True
    assert query_call[1]["with_vectors"] is False

    assert result == "test document"


def test_query_multiple_results(mock_qdrant_client):
    """Test querying with multiple results."""
    qdrant_db = QdrantDB(
        client=mock_qdrant_client,
        collection_name="test_collection",
        n_results=3,
    )

    mock_points = [
        MagicMock(payload={"document": "doc1"}),
        MagicMock(payload={"document": "doc2"}),
        MagicMock(payload={"document": "doc3"}),
    ]
    mock_qdrant_client.query_points.return_value.points = mock_points

    result = qdrant_db.query("test query")

    query_call = mock_qdrant_client.query_points.call_args
    assert query_call[1]["limit"] == 3

    assert result == "doc1\ndoc2\ndoc3"


def test_query_no_results(mock_qdrant_client):
    """Test querying when no results are found."""
    qdrant_db = QdrantDB(
        client=mock_qdrant_client, collection_name="test_collection"
    )

    mock_qdrant_client.query_points.return_value.points = []

    result = qdrant_db.query("test query")

    assert result == ""


def test_query_missing_document_field(mock_qdrant_client):
    """Test querying when payload doesn't contain document field."""
    qdrant_db = QdrantDB(
        client=mock_qdrant_client, collection_name="test_collection"
    )

    mock_point = MagicMock()
    mock_point.payload = {"other_field": "value"}
    mock_qdrant_client.query_points.return_value.points = [mock_point]

    result = qdrant_db.query("test query")

    assert result == ""


def test_different_distance_metrics(mock_qdrant_client):
    """Test initialization with different distance metrics."""
    distances = [
        models.Distance.COSINE,
        models.Distance.EUCLIDEAN,
        models.Distance.DOT,
    ]

    for distance in distances:
        qdrant_db = QdrantDB(
            client=mock_qdrant_client,
            collection_name="test_collection",
            distance=distance,
        )
        assert qdrant_db.distance == distance


def test_custom_model_name(mock_qdrant_client):
    """Test initialization with custom model name."""
    custom_model = "sentence-transformers/all-mpnet-base-v2"
    qdrant_db = QdrantDB(
        client=mock_qdrant_client,
        collection_name="test_collection",
        model_name=custom_model,
    )

    assert qdrant_db.model_name == custom_model


@patch("qdrant_client.QdrantClient")
def test_client_initialization(mock_client_class):
    """Test that client is properly initialized."""

    mock_client_instance = MagicMock()
    mock_client_class.return_value = mock_client_instance
    mock_client_instance.collection_exists.return_value = False
    mock_client_instance.get_embedding_size.return_value = 384

    qdrant_db = QdrantDB(
        client=mock_client_instance, collection_name="test_collection"
    )

    assert qdrant_db.client == mock_client_instance
