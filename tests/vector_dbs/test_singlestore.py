import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import singlestoredb as s2
from swarms_memory.vector_dbs.singlestore_wrapper import SingleStoreDB


@pytest.fixture
def mock_singlestore():
    with patch('singlestoredb.connect') as mock_connect:
        # Create mock connection and cursor
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_connection

        # Initialize DB with test configuration
        db = SingleStoreDB(
            host="localhost",
            port=3306,
            user="test_user",
            password="test_password",
            database="test_db",
            table_name="test_embeddings",
            dimension=4,
            namespace="test",
            ssl=True,
            ssl_verify=True
        )
        yield db, mock_cursor, mock_connect


def test_initialization(mock_singlestore):
    db, mock_cursor, mock_connect = mock_singlestore
    
    # Verify connection string format with SSL
    expected_conn_string = "test_user:test_password@localhost:3306/test_db?ssl=true"
    assert mock_connect.call_args[0][0] == expected_conn_string
    
    # Verify table creation with new schema
    create_table_call = mock_cursor.execute.call_args_list[0]
    create_table_sql = create_table_call[0][0]
    
    # Check for new schema elements
    assert "CREATE TABLE IF NOT EXISTS" in create_table_sql
    assert "test_embeddings" in create_table_sql
    assert "document TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci" in create_table_sql
    assert "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in create_table_sql
    assert "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP" in create_table_sql
    assert "KEY idx_namespace (namespace)" in create_table_sql
    assert "VECTOR INDEX vec_idx (embedding) DIMENSION = 4 DISTANCE_TYPE = DOT_PRODUCT" in create_table_sql
    assert "ENGINE = columnstore" in create_table_sql


@patch('singlestoredb.connect')
def test_ssl_configuration(mock_connect):
    # Setup mock
    mock_cursor = MagicMock()
    mock_connection = MagicMock()
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value.__enter__.return_value = mock_connection
    
    # Test with SSL disabled
    db_no_ssl = SingleStoreDB(
        host="localhost",
        port=3306,
        user="test_user",
        password="test_password",
        database="test_db",
        table_name="test_embeddings",
        dimension=4,
        ssl=False
    )
    
    # Verify connection string without SSL
    expected_conn_string = "test_user:test_password@localhost:3306/test_db"
    assert mock_connect.call_args[0][0] == expected_conn_string

    # Reset mock
    mock_connect.reset_mock()

    # Test with SSL enabled but no verification
    db_ssl_no_verify = SingleStoreDB(
        host="localhost",
        port=3306,
        user="test_user",
        password="test_password",
        database="test_db",
        table_name="test_embeddings",
        dimension=4,
        ssl=True,
        ssl_verify=False
    )
    
    # Verify connection string with SSL no verify
    expected_conn_string = "test_user:test_password@localhost:3306/test_db?ssl=true&ssl_verify=false"
    assert mock_connect.call_args[0][0] == expected_conn_string


def test_add_document(mock_singlestore):
    db, mock_cursor, _ = mock_singlestore
    
    # Mock embedding function
    test_embedding = np.array([0.1, 0.2, 0.3, 0.4])
    db.embedding_function = MagicMock(return_value=test_embedding)
    
    # Add document with metadata
    metadata = {"key": "value"}
    doc_id = db.add("test document", metadata=metadata)
    
    # Verify the insert query
    insert_call = mock_cursor.execute.call_args_list[-1]
    insert_sql = insert_call[0][0]
    
    # Verify SQL includes all columns
    assert "INSERT INTO test_embeddings" in insert_sql
    assert "id" in insert_sql
    assert "document" in insert_sql
    assert "embedding" in insert_sql
    assert "metadata" in insert_sql
    assert "namespace" in insert_sql
    
    # Verify parameters
    params = insert_call[0][1]
    assert len(params) == 5  # id, document, embedding, metadata, namespace
    assert params[1] == "test document"  # document
    np.testing.assert_array_equal(params[2], test_embedding)  # embedding
    assert '"key": "value"' in params[3]  # metadata
    assert params[4] == "test"  # namespace


def test_query(mock_singlestore):
    db, mock_cursor, _ = mock_singlestore
    
    # Mock embedding function and query results
    test_embedding = np.array([0.1, 0.2, 0.3, 0.4])
    db.embedding_function = MagicMock(return_value=test_embedding)
    
    mock_cursor.fetchall.return_value = [
        ("doc1", "content1", '{"key": "value1"}', 0.9),
        ("doc2", "content2", '{"key": "value2"}', 0.8)
    ]
    
    # Query with metadata filter
    results = db.query(
        "test query",
        top_k=2,
        metadata_filter={"category": "test"}
    )
    
    # Verify query execution
    query_call = mock_cursor.execute.call_args_list[-1]
    assert "SELECT" in query_call[0][0]
    assert "DOT_PRODUCT" in query_call[0][0]
    assert "JSON_EXTRACT" in query_call[0][0]
    
    # Verify results
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["similarity"] == 0.9
    assert results[0]["metadata"]["key"] == "value1"


def test_delete(mock_singlestore):
    db, mock_cursor, _ = mock_singlestore
    
    # Mock successful deletion
    mock_cursor.rowcount = 1
    
    # Delete document
    success = db.delete("test_id")
    
    # Verify deletion query
    delete_call = mock_cursor.execute.call_args_list[-1]
    assert "DELETE FROM test_embeddings" in delete_call[0][0]
    assert delete_call[0][1][0] == "test_id"
    assert success is True
    
    # Test unsuccessful deletion
    mock_cursor.rowcount = 0
    success = db.delete("nonexistent_id")
    assert success is False


def test_preprocessing(mock_singlestore):
    db, mock_cursor, _ = mock_singlestore
    
    # Define custom preprocessing function
    def custom_preprocess(text: str) -> str:
        return text.strip().lower()
    
    db.preprocess_function = custom_preprocess
    
    # Mock embedding function
    test_embedding = np.array([0.1, 0.2, 0.3, 0.4])
    db.embedding_function = MagicMock(return_value=test_embedding)
    
    # Add document with preprocessing
    db.add("  TEST DOCUMENT  ")
    
    # Verify preprocessed document
    insert_call = mock_cursor.execute.call_args_list[-1]
    assert insert_call[0][1][1] == "test document"


def test_postprocessing(mock_singlestore):
    db, mock_cursor, _ = mock_singlestore
    
    # Define custom postprocessing function
    def custom_postprocess(document: str) -> str:
        return document.upper()
    
    db.postprocess_function = custom_postprocess
    
    # Mock embedding function and query results
    test_embedding = np.array([0.1, 0.2, 0.3, 0.4])
    db.embedding_function = MagicMock(return_value=test_embedding)
    
    mock_cursor.fetchall.return_value = [
        ("doc1", "test document", "{}", 0.9)
    ]
    
    # Query with postprocessing
    results = db.query("test")
    
    # Verify postprocessed results
    assert results[0]["document"] == "TEST DOCUMENT"


def test_logger_setup(mock_singlestore):
    db, _, _ = mock_singlestore
    
    # Test with custom logger config
    custom_config = {
        "handlers": [
            {"sink": "custom.log", "rotation": "1 MB"},
        ],
    }
    
    with patch('loguru.logger.configure') as mock_configure:
        db._setup_logger(custom_config)
        mock_configure.assert_called_once_with(**custom_config)
    
    # Test with default config
    with patch('loguru.logger.configure') as mock_configure:
        db._setup_logger(None)
        called_config = mock_configure.call_args[1]
        assert "handlers" in called_config
        assert len(called_config["handlers"]) == 2
        assert called_config["handlers"][0]["sink"] == "singlestore_wrapper.log"


def test_default_embedding_function(mock_singlestore):
    db, _, _ = mock_singlestore
    
    # Mock the embedding model
    test_embedding = np.array([0.1, 0.2, 0.3, 0.4])
    db.embedding_model = MagicMock()
    db.embedding_model.encode.return_value = test_embedding
    
    # Test the default embedding function
    result = db._default_embedding_function("test text")
    
    # Verify the embedding model was called correctly
    db.embedding_model.encode.assert_called_once_with("test text")
    np.testing.assert_array_equal(result, test_embedding)
    assert isinstance(result, np.ndarray)


def test_default_preprocess_function(mock_singlestore):
    db, _, _ = mock_singlestore
    
    # Test with spaces to trim
    result = db._default_preprocess_function("  test text  ")
    assert result == "test text"
    
    # Test with no spaces to trim
    result = db._default_preprocess_function("test text")
    assert result == "test text"
    
    # Test with empty string
    result = db._default_preprocess_function("")
    assert result == ""


def test_default_postprocess_function(mock_singlestore):
    db, _, _ = mock_singlestore
    
    # Test with empty list
    test_results = []
    result = db._default_postprocess_function(test_results)
    assert result == []
    
    # Test with list of results
    test_results = [{"id": "1", "text": "test"}]
    result = db._default_postprocess_function(test_results)
    assert result == test_results
    assert result[0]["id"] == "1"
