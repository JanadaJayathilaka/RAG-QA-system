"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""

from typing import List
from .prompts import (
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)

from langchain.agents import create_agent




# # Define agents at module level for reuse
# retrieval_agent = create_agent(
#     model=create_chat_model(),
#     tools=[retrieval_tool],
#     system_prompt=RETRIEVAL_SYSTEM_PROMPT,
# )
