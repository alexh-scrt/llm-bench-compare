"""llm_bench_compare package.

This package provides a Flask web application that aggregates and visualizes
benchmark scores (MMLU, HumanEval, MATH, GSM8K, MBPP) for major open-source
LLMs, with cost-per-token overlays for capability-vs-cost trade-off analysis.

Usage::

    from llm_bench_compare import create_app

    app = create_app()
    app.run()

Or via the CLI entry-point::

    llm-bench-compare
"""

from __future__ import annotations

import os
from typing import Any

__version__ = "0.1.0"
__all__ = ["create_app", "main"]


def create_app(config: dict[str, Any] | None = None) -> "Flask":  # type: ignore[name-defined]  # noqa: F821
    """Application factory for the llm_bench_compare Flask app.

    Creates and configures a Flask application instance.  An optional
    *config* mapping can override any default configuration values, which
    is particularly useful for testing (e.g. passing
    ``{"TESTING": True}``).

    Args:
        config: Optional dictionary of Flask configuration overrides.

    Returns:
        A fully configured :class:`flask.Flask` application instance.

    Raises:
        ImportError: If Flask or required modules are not installed.
        RuntimeError: If the data files cannot be located.
    """
    # Import here so that the package can be imported even if Flask is not yet
    # installed (e.g. during initial setup introspection).
    from llm_bench_compare.app import create_app as _create_app  # noqa: PLC0415

    return _create_app(config=config)


def main() -> None:
    """Entry-point for the ``llm-bench-compare`` CLI command.

    Reads ``FLASK_HOST``, ``FLASK_PORT``, and ``FLASK_DEBUG`` environment
    variables to control the development server, then starts Flask's built-in
    server.  This is intended for local development only; use a production
    WSGI server (e.g. gunicorn) for deployment.

    Environment variables:
        FLASK_HOST: Hostname to bind (default ``"127.0.0.1"``).
        FLASK_PORT: Port number to bind (default ``5000``).
        FLASK_DEBUG: Set to ``"1"`` or ``"true"`` to enable debug mode.
    """
    app = create_app()
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug_env = os.environ.get("FLASK_DEBUG", "0").lower()
    debug = debug_env in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
