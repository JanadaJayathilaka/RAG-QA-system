from langchain_core.tools import tool
from src.app.core.retrieval.vector_store import retrieve


from ..retrieval.vector_store import retrieve
from ..retrieval.serialization import serialize_chunks


@tool(response_format="content_and_artifact")
def retrieve_tool(query: str):
    """Search the vector store for relevant information based on the query.
    This tool retrieve 5 most relevant chunks from the vector store and returns them as content and artifacts.
    The chunks are formatted with page numbers and indices for easy reference.

    Args:
        query (str): The search query to retrieve relevant information.
    
    Returns:
        Tuple of (serialized_content, artifacts) where:
        - serialized_content: A formatted string containing the retrieved chunks
          with metadata. Format: "Chunk 1 (page=X): ...\n\nChunk 2 (page=Y): ..."
        - artifact: List of Document objects with full metadata for reference
    """

    docs = retrieve(query, top_k=5)

    context = serialize_chunks(docs)

 #the context goes to the agent as content and the docs go as artifacts for reference
    return context, docs  


