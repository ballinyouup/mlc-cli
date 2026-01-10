"""
MLC-LLM provider implementation.

This module provides LLM inference using the MLC-LLM engine.
"""

from __future__ import annotations

from typing import AsyncIterator, TYPE_CHECKING

from src.core.domain import ChatMessage

if TYPE_CHECKING:
    from mlc_llm import MLCEngine


class MLCLLMProvider:
    """
    LLMProvider implementation using MLC-LLM.

    Implements the LLMProvider protocol for local LLM inference.
    Can be swapped with OpenAI, Anthropic, or other providers.
    """

    def __init__(self, engine: MLCEngine) -> None:
        """
        Initialize the LLM provider.

        Args:
            engine: Initialized MLC-LLM engine instance.
        """
        self._engine = engine

    async def generate(
        self,
        messages: list[ChatMessage],
        model: str,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of chat messages forming the conversation.
            model: Model identifier to use.
            stream: Whether to stream the response.

        Yields:
            Response text chunks if streaming, or full response if not.
        """
        formatted_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]

        if stream:
            for response in self._engine.chat.completions.create(
                messages=formatted_messages,
                model=model,
                stream=True,
            ):
                for choice in response.choices:
                    if choice.delta.content:
                        yield choice.delta.content
        else:
            response = self._engine.chat.completions.create(
                messages=formatted_messages,
                model=model,
                stream=False,
            )
            if response.choices:
                yield response.choices[0].message.content or ""

    def terminate(self) -> None:
        """Clean up LLM resources."""
        self._engine.terminate()
