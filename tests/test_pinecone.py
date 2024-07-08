from unittest.mock import patch
from swarms_memory.pinecone_wrapper import PineconeMemory


@patch("pinecone.init")
@patch("pinecone.list_indexes")
@patch("pinecone.create_index")
@patch("pinecone.Index")
@patch("sentence_transformers.SentenceTransformer")
def test_init(
    mock_st,
    mock_index,
    mock_create_index,
    mock_list_indexes,
    mock_init,
):
    mock_list_indexes.return_value = []
    PineconeMemory("api_key", "environment", "index_name")
    mock_init.assert_called_once_with(
        api_key="api_key", environment="environment"
    )
    mock_create_index.assert_called_once()
    mock_index.assert_called_once_with("index_name")
    mock_st.assert_called_once_with("all-MiniLM-L6-v2")


@patch("loguru.logger.configure")
def test_setup_logger(mock_configure):
    PineconeMemory._setup_logger(None)
    mock_configure.assert_called_once()


@patch("sentence_transformers.SentenceTransformer.encode")
def test_default_embedding_function(mock_encode):
    pm = PineconeMemory("api_key", "environment", "index_name")
    pm._default_embedding_function("text")
    mock_encode.assert_called_once_with("text")


def test_default_preprocess_function():
    pm = PineconeMemory("api_key", "environment", "index_name")
    assert pm._default_preprocess_function(" text ") == "text"


def test_default_postprocess_function():
    pm = PineconeMemory("api_key", "environment", "index_name")
    assert pm._default_postprocess_function("results") == "results"


@patch("pinecone.Index.upsert")
def test_add(mock_upsert):
    pm = PineconeMemory("api_key", "environment", "index_name")
    pm.add("doc")
    mock_upsert.assert_called_once()


@patch("pinecone.Index.query")
def test_query(mock_query):
    pm = PineconeMemory("api_key", "environment", "index_name")
    pm.query("query")
    mock_query.assert_called_once()
