"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""
from langchain.agents import create_agent
from src.app.core.llm.factory import create_chat_model
from typing import List
from .tools import retrieve_tool
from .state import QAState
from langchain.messages import AIMessage, ToolMessage, HumanMessage

from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)

def _extract_last_ai_content(messages: List[AIMessage]) -> str:
    """Helper function to extract the content of the last AIMessage from a list of messages."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""

retrieval_agent = create_agent(
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    model=create_chat_model(),
    tools=[retrieve_tool],
)

summarization_agent = create_agent(
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
    model=create_chat_model(),
    tools=[],
)


verification_agent = create_agent(
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
    model=create_chat_model(),
    tools=[],
)


def retrieval_node(state: QAState) -> QAState:
    """Retrieval agent node: gathers context from vector store based on the question.
    This node:
    - sends the user's question to the retrieval agent
    - The agent uses the attached retrieval tool to fetch document chunks.
    - Extracts the tool's content (CONTEXT string) from the ToolMessage.
    - Stores the consolidated context string in `state["context"]`.
    """
    question = state["question"]

    result = retrieval_agent.invoke({"messages": HumanMessage(content=question)})

    messages =result.get("messages", [])
    context = ""

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            context = str(msg.content)
            break
    return {
        "context": context,
    }

def summarization_node(state: QAState) -> QAState:
    """Summarization agent node: generates a draft answer from the question and retrieved context.
    This node:
    - sends the question and retrieved context to the summarization agent
    - The agent processes this information and generates a draft answer.
    - The draft answer is stored in `state["draft_answer"]` for the next node to use.
    """
    question = state["question"]
    context = state.get("context")

    user_content = f"Question: {question}\n\nContext: {context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )

    messages = result.get("messages", [])
    

    draft_answer = _extract_last_ai_content(messages)
    return {
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer in `state["answer"]`.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"Question: {question}\n\nContext: {context}\n\nDraft Answer: {draft_answer}"

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    final_answer = _extract_last_ai_content(messages)
    return {
        "answer": final_answer,
    }