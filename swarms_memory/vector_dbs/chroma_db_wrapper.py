import os
import uuid
from typing import Optional, Dict, Any, Union, Callable, List

import chromadb
from dotenv import load_dotenv
from loguru import logger
from swarms_memory.vector_dbs.base_vectordb import BaseVectorDatabase
from swarms_memory.embeddings.embedding_utils import setup_unified_embedding, get_embedding_function
from swarms_memory.exceptions import VectorDatabaseError, ConfigurationError, ValidationError, handle_error
from swarms.utils.data_to_text import data_to_text

# Load environment variables
load_dotenv()


class EmbeddingFunctionWrapper:
    """Wrapper class to provide ChromaDB-compatible embedding function with name attribute."""
    
    def __init__(self, embedding_function: Callable, name: str = "custom_embedding"):
        self.embedding_function = embedding_function
        self._name = name
    
    def __call__(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Call the underlying embedding function with ChromaDB-compatible signature."""
        if isinstance(input, str):
            return self.embedding_function(input)
        else:
            # Handle batch processing
            return [self.embedding_function(text) for text in input]
    
    def name(self) -> str:
        """Return the name of the embedding function."""
        return self._name


# Results storage using local ChromaDB
class ChromaDB(BaseVectorDatabase):
    """

    ChromaDB database

    Args:
        metric (str): The similarity metric to use.
        output (str): The name of the collection to store the results in.
        limit_tokens (int, optional): The maximum number of tokens to use for the query. Defaults to 1000.
        n_results (int, optional): The number of results to retrieve. Defaults to 2.

    Methods:
        add: _description_
        query: _description_

    Examples:
        >>> chromadb = ChromaDB(
        >>>     metric="cosine",
        >>>     output="results",
        >>>     llm="gpt3",
        >>>     openai_api_key=OPENAI_API_KEY,
        >>> )
        >>> chromadb.add(task, result, result_id)
    """

    def __init__(
        self,
        metric: str = "cosine",
        output_dir: str = "swarms",
        limit_tokens: Optional[int] = 1000,
        n_results: int = 1,
        docs_folder: str = None,
        verbose: bool = False,
        embedding_model: Union[str, Any, Callable] = "text-embedding-3-small",
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        dimension: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.output_dir = output_dir
        self.limit_tokens = limit_tokens
        self.n_results = n_results
        self.docs_folder = docs_folder
        self.verbose = verbose
        
        # Setup unified embedding system
        embedding_kwargs = {k: v for k, v in kwargs.items() if k.startswith('embedding_')}
        self.embedder, custom_embedding_function, self.dimension = setup_unified_embedding(
            embedding_model=embedding_model,
            embedding_function=embedding_function,
            dimension=dimension,
            **embedding_kwargs
        )
        
        self.embedding_function = (
            custom_embedding_function or 
            get_embedding_function(self.embedder) if self.embedder else None
        )

        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.PersistentClient(
            settings=chromadb.config.Settings(
                persist_directory=chroma_persist_dir,
            ),
            *args,
            **kwargs,
        )

        # Create ChromaDB client
        self.client = chromadb.Client()

        # Create Chroma collection with custom embedding function if available
        collection_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('embedding_')}
        if self.embedding_function:
            # Wrap the embedding function to provide the required name() method
            wrapped_embedding_function = EmbeddingFunctionWrapper(
                self.embedding_function, 
                name=f"{embedding_model}_embedding"
            )
            collection_kwargs["embedding_function"] = wrapped_embedding_function
            
        self.collection = chroma_client.get_or_create_collection(
            name=output_dir,
            metadata={"hnsw:space": metric},
            **collection_kwargs,
        )
        logger.info(
            "ChromaDB collection created:"
            f" {self.collection.name} with metric: {self.metric} and"
            f" output directory: {self.output_dir}"
        )

        # If docs
        if docs_folder:
            logger.info(f"Traversing directory: {docs_folder}")
            self.traverse_directory()

    def add(
        self,
        document: str,
        metadata: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Add a document to the ChromaDB collection.

        Args:
            document (str): The document to be added.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the document.

        Returns:
            str: The ID of the added document.
        """
        # Input validation
        if not document or not isinstance(document, str):
            raise ValidationError("Document must be a non-empty string")
        
        try:
            doc_id = str(uuid.uuid4())
            
            add_kwargs = {k: v for k, v in kwargs.items()}
            if metadata:
                add_kwargs["metadatas"] = [metadata]
                
            self.collection.add(
                ids=[doc_id],
                documents=[document],
                **add_kwargs,
            )
            
            if self.verbose:
                logger.success(f"Document added successfully with ID: {doc_id}")
            return doc_id
        except ValidationError:
            raise
        except Exception as e:
            handle_error(e, "document addition", "ChromaDB", VectorDatabaseError)

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        return_metadata: bool = False,
        *args,
        **kwargs,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Query documents from the ChromaDB collection.

        Args:
            query_text (str): The query string.
            top_k (Optional[int]): The number of documents to retrieve. Defaults to n_results.
            return_metadata (bool): Whether to return detailed metadata. Defaults to False.

        Returns:
            Union[str, List[Dict[str, Any]]]: If return_metadata=False, returns concatenated text.
                If return_metadata=True, returns list of dictionaries with detailed results.
        """
        # Input validation
        if not query_text or not isinstance(query_text, str):
            raise ValidationError("Query text must be a non-empty string")
        
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValidationError("top_k must be a positive integer")
            
        try:
            n_results = top_k or self.n_results
            
            if self.verbose:
                logger.info(f"Querying documents for: {query_text}")
                
            query_kwargs = {k: v for k, v in kwargs.items()}
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                **query_kwargs,
            )

            # Format results
            formatted_results = []
            documents = results.get("documents", [[]])[0]
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            for i, doc in enumerate(documents):
                result = {
                    "id": ids[i] if i < len(ids) else None,
                    "score": 1.0 - distances[i] if i < len(distances) else None,  # Convert distance to similarity
                    "metadata": {"text": doc}
                }
                if i < len(metadatas) and metadatas[i]:
                    result["metadata"].update(metadatas[i])
                formatted_results.append(result)

            if self.verbose:
                logger.success(f"Query completed. Found {len(formatted_results)} results.")
            
            if return_metadata:
                return formatted_results
            else:
                # Return concatenated text for backward compatibility
                return "\n\n".join(
                    result["metadata"].get("text", str(result["metadata"]))
                    for result in formatted_results
                )

        except ValidationError:
            raise
        except Exception as e:
            handle_error(e, "document query", "ChromaDB", VectorDatabaseError)

    def traverse_directory(
        self, docs_folder: str = None, *args, **kwargs
    ):
        """
        Traverse through every file in the given directory and its subdirectories,
        and return the paths of all files.
        Parameters:
        - directory_name (str): The name of the directory to traverse.
        Returns:
        - list: A list of paths to each file in the directory and its subdirectories.
        """
        try:
            logger.info(f"Traversing directory: {self.docs_folder}")
            added_to_db = False
            allowed_extensions = [
                "txt",
                "pdf",
                "docx",
                "doc",
                "md",
                "yaml",
                "json",
                "csv",
                "tsv",
                "xls",
                "xlsx",
                "xml",
                "yml",
            ]

            for root, dirs, files in os.walk(self.docs_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() in allowed_extensions:
                        data = data_to_text(file_path)
                        added_to_db = self.add(str(data))
                        print(f"{file_path} added to Database")
                    else:
                        print(
                            f"Skipped {file_path} due to unsupported file extension"
                        )

            return added_to_db

        except Exception as error:
            logger.error(
                f"Failed to traverse directory: {str(error)}"
            )
            raise error
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the ChromaDB collection.

        Args:
            doc_id (str): The document ID to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            if self.verbose:
                logger.info(f"Deleting document with ID: {doc_id}")
            self.collection.delete(ids=[doc_id])
            if self.verbose:
                logger.success(f"Document deleted successfully: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            bool: True if clearing was successful, False otherwise.
        """
        try:
            if self.verbose:
                logger.info("Clearing all documents from collection")
            # Get all document IDs and delete them
            all_docs = self.collection.get()
            if all_docs.get('ids'):
                self.collection.delete(ids=all_docs['ids'])
            if self.verbose:
                logger.success("All documents cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear documents: {str(e)}")
            return False
