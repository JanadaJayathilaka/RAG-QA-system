"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""
from langchain.agents import create_agent
from src.app.core.llm.factory import create_chat_model
from typing import List
from .tools import retrieve_tool
from .state import QAState
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)

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

    