from pydantic import BaseModel

class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the vector databases paper.
    """
    question: str

class QAResponse(BaseModel):
    """Response schema for the `/qa` endpoint.

    The response includes:
    - `answer`: The final verified answer from the verification agent.
    - `context`: The retrieved context from the vector store used to generate the answers.
    """
    answer: str
    context: str

