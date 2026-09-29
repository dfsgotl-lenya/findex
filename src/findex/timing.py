"""Timing decorator used by the Lab 3 build/load/search paths."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
logger = logging.getLogger(__name__)


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Log elapsed time while preserving function metadata with ``wraps``."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("timed %s: %.3f ms", fn.__name__, elapsed_ms)

    return wrapper
