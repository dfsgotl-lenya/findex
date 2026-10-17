"""Allow ``python -m findex ...`` (handy for cProfile / py-spy)."""

from .cli import app

if __name__ == "__main__":
    app()
