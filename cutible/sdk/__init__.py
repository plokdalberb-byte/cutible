"""Python SDK for Cutible (plan §8.3).

Client library for embedding Cutible into Python-based agent pipelines.
Provides a high-level interface over the REST API or direct in-process use.
"""

from .client import CutibleClient

__all__ = ["CutibleClient"]
