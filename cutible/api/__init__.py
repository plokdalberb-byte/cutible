"""REST API for Cutible (plan §8.2).

FastAPI-based HTTP interface for products/backends that want
to create projects, upload assets, run editing, and get renders.
"""

from .app import create_app

__all__ = ["create_app"]
