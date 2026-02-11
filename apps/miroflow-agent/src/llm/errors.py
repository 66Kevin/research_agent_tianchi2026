# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

"""
Custom exception types for LLM error handling.
"""


class NonRetriableLLMError(Exception):
    """Base class for non-retriable LLM errors."""


class PolicyBlockedError(NonRetriableLLMError):
    """Raised when provider safety/policy blocks the request."""

