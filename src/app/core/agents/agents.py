"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""
from langchain.agents import create_agent
from src.app.core.llm.factory import create_chat_model
from typing import List
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)

retrieval_agent = create_agent(
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    model=create_chat_model(),
    tools=[],
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


