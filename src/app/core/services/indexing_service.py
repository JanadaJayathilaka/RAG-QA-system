"""Service functions for indexing documents into the vector database."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from src.app.core.retrieval.vector_store import index_documents

def index_pdf_file(file_path: Path) -> int:
    """load the pdf file and chunk it and return the number of chunks indexed."""
    return index_documents(file_path)


