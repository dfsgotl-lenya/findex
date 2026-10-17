"""Lab 7: findex over HTTP — FastAPI API and a Jinja2 search page.

``uvicorn findex.web:app`` (or ``findex serve``) starts the service. ``app`` is
created lazily on first access so importing :mod:`findex.web` in tests does not
read the environment or touch the index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .app import create_app

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> FastAPI | object:
    if name == "create_app":
        from .app import create_app

        return create_app
    if name == "app":
        from .app import create_app

        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(name)
