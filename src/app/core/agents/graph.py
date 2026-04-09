"""LangGraph orchestration for the linear multi-agent QA flow."""
from typing import Any
from .state import QAState

from functools import lru_cache
from langgraph.graph import StateGraph
from langgraph.graph import START,END
def create_qa_graph() -> Any:
    """Create and compile the linear multi-agent QA graph.

    The graph executes in order:
    1. Retrieval Agent: gathers context from vector store
    2. Summarization Agent: generates draft answer from context
    3. Verification Agent: verifies and corrects the answer

    Returns:
        Compiled graph ready for execution.
    """
    builder = StateGraph(QAState)
    # Define the graph structure
    builder.add_node("retrieval", retrieval_node)  # Placeholder, actual function will be set in execution
    builder.add_node("summarization", summarization_node)  # Placeholder
    builder.add_node("verification", verification_node)  # Placeholder

    
    builder.add_edge(START, "retrieval")
    builder.add_edge("retrieval", "summarization")
    builder.add_edge("summarization", "verification")
    builder.add_edge("verification", END)

    return builder.compile()

@lru_cache(maxsize=1)
def get_qa_graph():
    """Get the compiled QA graph, cached for efficiency."""
    return create_qa_graph()

