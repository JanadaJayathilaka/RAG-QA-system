
from typing import Any, Dict
from src.app.core.agents.graph import run_qa_flow

def answer_question(question: str) -> Dict[str, Any]:
    """Run the multi-agent QA flow for a given question.

    Args:
        question: User's natural language question about the vector databases paper.

    Returns:
        Dictionary containing at least `answer` and `context` keys.
    """
    return run_qa_flow(question)
