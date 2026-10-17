"""Configuration read from the environment (12-factor, rule III)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every value can be overridden with an environment variable of the same name.

    ``INDEX_PATH=/data/peps.bin LOG_LEVEL=DEBUG findex serve`` — nothing is
    hard-coded in the application, so one Docker image runs in dev and prod.
    A ``.env`` file in the working directory is read in development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    index_path: Path = Field(
        default=Path("data/peps.bin"),
        description="Saved findex index (pickle or JSON) loaded once at startup.",
    )
    semantic_path: Path | None = Field(
        default=None,
        description=(
            "Directory written by `findex embed`; defaults to "
            "<index_path>.semantic. Missing directory = keyword search only."
        ),
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = "text"
    site_name: str = "findex"
    # Lab 7 M2 experiment switch: register /search as ``def`` (thread pool) or
    # as a naive ``async def`` that runs the CPU-bound scorer on the event loop.
    # The default is the variant that does not block the loop.
    search_route: Literal["def", "async"] = "def"

    @property
    def semantic_dir(self) -> Path:
        return self.semantic_path or self.index_path.with_suffix(".semantic")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """FastAPI dependency; cached so the environment is parsed once."""
    return Settings()
