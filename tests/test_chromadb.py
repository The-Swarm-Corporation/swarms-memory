from unittest.mock import patch, MagicMock
from swarms_memory.chroma_db_wrapper import ChromaDB


@patch("chromadb.PersistentClient")
@patch("chromadb.Client")
def test_init(mock_client, mock_persistent_client):
    chroma_db = ChromaDB(
        metric="cosine",
        output_dir="swarms",
        limit_tokens=1000,
        n_results=1,
        docs_folder=None,
        verbose=False,
    )
    assert chroma_db.metric == "cosine"
    assert chroma_db.output_dir == "swarms"
    assert chroma_db.limit_tokens == 1000
    assert chroma_db.n_results == 1
    assert chroma_db.docs_folder is None
    assert chroma_db.verbose is False
    mock_persistent_client.assert_called_once()
    mock_client.assert_called_once()


@patch("chromadb.PersistentClient")
@patch("chromadb.Client")
def test_add(mock_client, mock_persistent_client):
    chroma_db = ChromaDB()
    mock_collection = MagicMock()
    chroma_db.collection = mock_collection
    doc_id = chroma_db.add("test document")
    mock_collection.add.assert_called_once_with(
        ids=[doc_id], documents=["test document"]
    )
    assert isinstance(doc_id, str)


@patch("chromadb.PersistentClient")
@patch("chromadb.Client")
def test_query(mock_client, mock_persistent_client):
    chroma_db = ChromaDB()
    mock_collection = MagicMock()
    chroma_db.collection = mock_collection
    mock_collection.query.return_value = {
        "documents": ["test document"]
    }
    result = chroma_db.query("test query")
    mock_collection.query.assert_called_once_with(
        query_texts=["test query"], n_results=1
    )
    assert result == "test document\n"


@patch("chromadb.PersistentClient")
@patch("chromadb.Client")
@patch("os.walk")
@patch("swarms_memory.chroma_db_wrapper.ChromaDB.add")
def test_traverse_directory(
    mock_add, mock_walk, mock_client, mock_persistent_client
):
    chroma_db = ChromaDB(docs_folder="test_folder")
    mock_walk.return_value = [("root", "dirs", ["file1", "file2"])]
    chroma_db.traverse_directory()
    assert mock_add.call_count == 2
