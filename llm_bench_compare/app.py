"""Flask application factory and route definitions for llm_bench_compare.

This module defines the Flask application factory :func:`create_app` and all
web endpoints:

* ``GET /`` — Main comparison page (HTML).
* ``GET /api/models`` — JSON list of models with optional query-param filters.
* ``GET /api/models/<model_id>`` — JSON detail for a single model.
* ``GET /api/filter-options`` — JSON object of available filter values.
* ``GET /api/pricing/<model_id>`` — JSON pricing detail for a single model.
* ``GET /api/compare`` — JSON payload for radar-chart comparison of up to 4 models.

All JSON responses use snake_case keys.  Error responses follow the shape::

    {"error": "<human-readable message>", "status": <http_code>}

Usage::

    from llm_bench_compare.app import create_app

    app = create_app()
    app.run(debug=True)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from llm_bench_compare.data_loader import (
    clear_cache,
    get_filter_options,
    get_merged_df,
    load_pricing_df,
)
from llm_bench_compare.filters import (
    apply_filters,
    get_model_by_id,
    get_models_by_ids,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Benchmark keys exposed in the API.
_BENCHMARK_KEYS: tuple[str, ...] = ("mmlu", "humaneval", "math", "gsm8k", "mbpp")

#: Maximum number of models allowed in a single /api/compare request.
_MAX_COMPARE_MODELS: int = 4

#: Columns to include in the list API response (subset of merged df).
_LIST_COLUMNS: tuple[str, ...] = (
    "model_id",
    "display_name",
    "family",
    "version",
    "parameter_size_b",
    "parameter_size_bucket",
    "architecture",
    "context_length_k",
    "license",
    "open_weights",
    "release_date",
    "task_categories",
    "benchmark_mmlu",
    "benchmark_humaneval",
    "benchmark_math",
    "benchmark_gsm8k",
    "benchmark_mbpp",
    "cheapest_input_per_1m",
    "cheapest_output_per_1m",
    "self_hosted_hourly_usd",
    "self_hosted_gpu_setup",
    "self_hosted_min_vram_gb",
    "self_hosted_throughput_tps",
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _nan_to_none(value: Any) -> Any:
    """Convert float NaN / infinity to ``None`` so JSON serialisation works.

    :class:`float` ``NaN`` and ``inf`` are not valid JSON values; this helper
    replaces them with ``None`` (which serialises to JSON ``null``).

    Args:
        value: Any scalar value.

    Returns:
        The original *value* unless it is a non-finite float, in which case
        ``None`` is returned.
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _row_to_dict(row: pd.Series, columns: tuple[str, ...]) -> dict[str, Any]:
    """Serialise selected columns of a :class:`~pandas.Series` to a dict.

    Only columns present in both *columns* and *row.index* are included.
    NaN values are converted to ``None``.

    Args:
        row: A single model row as a pandas Series.
        columns: Tuple of column names to include.

    Returns:
        Dictionary mapping column names to Python-native values.
    """
    result: dict[str, Any] = {}
    for col in columns:
        if col not in row.index:
            result[col] = None
            continue
        val = row[col]
        # Pandas NA / NaN → None
        try:
            if pd.isna(val):
                val = None
        except (TypeError, ValueError):
            # pd.isna raises TypeError for lists etc.
            pass
        if val is not None:
            val = _nan_to_none(val)
        # Convert numpy types to native Python
        if hasattr(val, "item"):
            val = val.item()
        result[col] = val
    return result


def _df_to_records(df: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Convert a filtered DataFrame to a list of serialisable dicts.

    Args:
        df: The DataFrame to serialise.
        columns: Column names to include in each record.

    Returns:
        List of dictionaries, one per row.
    """
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(_row_to_dict(row, columns))
    return records


def _error_response(message: str, status_code: int) -> tuple[Response, int]:
    """Build a JSON error response tuple.

    Args:
        message: Human-readable error description.
        status_code: HTTP status code.

    Returns:
        A ``(Response, int)`` tuple suitable for returning from a Flask view.
    """
    return jsonify({"error": message, "status": status_code}), status_code


# ---------------------------------------------------------------------------
# Query-parameter parsing helpers
# ---------------------------------------------------------------------------


def _parse_list_param(param_name: str) -> list[str]:
    """Extract a multi-value query parameter from the current request.

    Handles both comma-separated single values
    (``?categories=coding,math``) and repeated keys
    (``?categories=coding&categories=math``).

    Args:
        param_name: The query-string parameter name.

    Returns:
        A (possibly empty) list of non-empty, stripped string values.
    """
    raw_values: list[str] = request.args.getlist(param_name)
    result: list[str] = []
    for raw in raw_values:
        for part in raw.split(","):
            stripped = part.strip()
            if stripped:
                result.append(stripped)
    return result


def _parse_bool_param(param_name: str, default: bool = False) -> bool:
    """Parse a boolean query parameter.

    Treats ``"1"``, ``"true"`` (case-insensitive), and ``"yes"`` as
    ``True``; all other non-empty values as ``False``.

    Args:
        param_name: The query-string parameter name.
        default: Value to return when the parameter is absent.

    Returns:
        Parsed boolean value.
    """
    raw = request.args.get(param_name, "").lower().strip()
    if not raw:
        return default
    return raw in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application.

    This factory function follows the Flask application-factory pattern so
    that multiple isolated instances can be created (useful for testing).

    Configuration precedence (highest to lowest):

    1. Values passed via the *config* argument.
    2. Hard-coded defaults set inside this function.

    Args:
        config: Optional dictionary of Flask configuration overrides.
            Commonly used keys:

            * ``TESTING`` — set to ``True`` in test suites.
            * ``DATA_CACHE_ENABLED`` — set to ``False`` to disable the
              module-level data cache (not recommended in production).

    Returns:
        A configured :class:`flask.Flask` application instance with all
        routes registered.

    Raises:
        RuntimeError: If the data files cannot be located during eager
            loading (only when ``EAGER_LOAD_DATA`` is ``True``).
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # --- Default configuration ---
    app.config.update(
        {
            "JSON_SORT_KEYS": False,
            "JSONIFY_PRETTYPRINT_REGULAR": False,
            "EAGER_LOAD_DATA": True,
        }
    )

    # --- Override with caller-supplied config ---
    if config:
        app.config.update(config)

    # --- Configure logging ---
    if not app.debug:
        logging.basicConfig(level=logging.INFO)

    # --- Eagerly warm the data cache (unless disabled for testing) ---
    if app.config.get("EAGER_LOAD_DATA", True) and not app.config.get("TESTING", False):
        try:
            get_merged_df()
            logger.info("Data cache warmed successfully.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not pre-warm data cache: %s", exc)

    # -----------------------------------------------------------------------
    # Register routes
    # -----------------------------------------------------------------------

    @app.route("/", methods=["GET"])
    def index() -> str:
        """Render the main comparison page.

        Loads the full (unfiltered) model list and filter options and
        passes them to the Jinja2 template as initial page data.

        Returns:
            Rendered HTML string for the main page.
        """
        try:
            df = get_merged_df()
            filter_opts = get_filter_options()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load data for index page.")
            # Render with empty data so the page still loads
            df = pd.DataFrame()
            filter_opts = {
                "task_categories": [],
                "size_buckets": [],
                "licenses": [],
                "families": [],
            }

        models = _df_to_records(df, _LIST_COLUMNS) if not df.empty else []

        return render_template(
            "index.html",
            models=models,
            filter_options=filter_opts,
            benchmark_keys=list(_BENCHMARK_KEYS),
            total_model_count=len(models),
        )

    # -----------------------------------------------------------------------

    @app.route("/api/models", methods=["GET"])
    def api_models() -> tuple[Response, int] | Response:
        """Return a JSON array of models, optionally filtered and sorted.

        Query Parameters:
            categories (str, repeatable): Task categories to filter by
                (``reasoning``, ``coding``, ``math``).  Multiple values
                may be comma-separated or supplied as repeated params.
            buckets (str, repeatable): Parameter size buckets to filter by
                (``≤7B``, ``8–34B``, ``35B+``).
            licenses (str, repeatable): License types to filter by
                (``Apache-2.0``, ``MIT``, ``custom/commercial``).
            families (str, repeatable): Model family names to filter by.
            open_weights (bool): When ``true``/``1``, restrict to models
                with open weights.
            sort (str): Benchmark key to sort by
                (``mmlu``, ``humaneval``, ``math``, ``gsm8k``, ``mbpp``).
            asc (bool): When ``true``/``1``, sort ascending (lowest first).
                Default is descending (highest first).

        Returns:
            JSON object::

                {
                    "count": <int>,
                    "models": [ { ...model fields... }, ... ]
                }

            or a JSON error object with status 400 / 500.
        """
        try:
            df = get_merged_df()
        except FileNotFoundError as exc:
            logger.error("Data file missing: %s", exc)
            return _error_response("Data files not found. Check server configuration.", 500)
        except ValueError as exc:
            logger.error("Data validation error: %s", exc)
            return _error_response(f"Data validation error: {exc}", 500)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error loading data.")
            return _error_response("Internal server error.", 500)

        # Parse query parameters
        task_categories = _parse_list_param("categories") or None
        size_buckets = _parse_list_param("buckets") or None
        licenses = _parse_list_param("licenses") or None
        families = _parse_list_param("families") or None
        open_weights_only = _parse_bool_param("open_weights", default=False)
        sort_benchmark = request.args.get("sort", "").strip() or None
        sort_ascending = _parse_bool_param("asc", default=False)

        # Validate sort_benchmark early so we can return 400 rather than 500
        valid_benchmarks = set(_BENCHMARK_KEYS)
        if sort_benchmark and sort_benchmark not in valid_benchmarks:
            return _error_response(
                f"Invalid sort benchmark '{sort_benchmark}'. "
                f"Valid options: {sorted(valid_benchmarks)}",
                400,
            )

        try:
            filtered = apply_filters(
                df,
                task_categories=task_categories,
                size_buckets=size_buckets,
                licenses=licenses,
                families=families,
                open_weights_only=open_weights_only,
                sort_benchmark=sort_benchmark,
                sort_ascending=sort_ascending,
            )
        except ValueError as exc:
            return _error_response(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error applying filters.")
            return _error_response("Internal server error while applying filters.", 500)

        records = _df_to_records(filtered, _LIST_COLUMNS)
        return jsonify({"count": len(records), "models": records})

    # -----------------------------------------------------------------------

    @app.route("/api/models/<string:model_id>", methods=["GET"])
    def api_model_detail(model_id: str) -> tuple[Response, int] | Response:
        """Return full detail for a single model.

        Path Parameters:
            model_id: The unique model identifier (e.g. ``deepseek-r1``).

        Returns:
            JSON object with all model fields, or a 404 JSON error if the
            model is not found.
        """
        try:
            df = get_merged_df()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error loading data for model detail.")
            return _error_response("Internal server error.", 500)

        row = get_model_by_id(df, model_id)
        if row is None:
            return _error_response(f"Model '{model_id}' not found.", 404)

        record = _row_to_dict(row, _LIST_COLUMNS)
        return jsonify(record)

    # -----------------------------------------------------------------------

    @app.route("/api/filter-options", methods=["GET"])
    def api_filter_options() -> tuple[Response, int] | Response:
        """Return the available filter option values from the data store.

        This endpoint is called by the client-side JavaScript on page load
        to populate the filter controls dynamically.

        Returns:
            JSON object::

                {
                    "task_categories": ["coding", "math", "reasoning"],
                    "size_buckets":    ["≤7B", "8–34B", "35B+"],
                    "licenses":        ["Apache-2.0", "MIT", "custom/commercial"],
                    "families":        ["DeepSeek", "Gemma", ...]
                }
        """
        try:
            options = get_filter_options()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error retrieving filter options.")
            return _error_response("Internal server error.", 500)

        return jsonify(options)

    # -----------------------------------------------------------------------

    @app.route("/api/pricing/<string:model_id>", methods=["GET"])
    def api_pricing_detail(model_id: str) -> tuple[Response, int] | Response:
        """Return detailed pricing information for a single model.

        Returns all provider entries for the given model, plus self-hosted
        cost estimates.

        Path Parameters:
            model_id: The unique model identifier.

        Returns:
            JSON object::

                {
                    "model_id": "deepseek-r1",
                    "api_providers": [
                        {
                            "provider": "DeepSeek Platform",
                            "provider_url": "https://platform.deepseek.com",
                            "input_per_1m": 0.55,
                            "output_per_1m": 2.19,
                            "notes": "..."
                        },
                        ...
                    ],
                    "self_hosted": {
                        "hourly_cost_usd": 28.0,
                        "gpu_setup": "8× H100 80GB SXM",
                        "min_vram_gb": 640,
                        "throughput_tps": 2000
                    }
                }

            or a 404 JSON error if the model is not found in pricing data.
        """
        try:
            pricing_df = load_pricing_df()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error loading pricing data.")
            return _error_response("Internal server error.", 500)

        model_rows = pricing_df[pricing_df["model_id"] == model_id]

        # Also check if model exists in benchmark data (may exist without pricing)
        try:
            merged_df = get_merged_df()
        except Exception:  # noqa: BLE001
            merged_df = pd.DataFrame()

        model_in_benchmarks = (
            not merged_df.empty
            and "model_id" in merged_df.columns
            and (merged_df["model_id"] == model_id).any()
        )

        if model_rows.empty and not model_in_benchmarks:
            return _error_response(f"Model '{model_id}' not found.", 404)

        # Build API providers list
        api_providers: list[dict[str, Any]] = []
        for _, row in model_rows.iterrows():
            provider_entry: dict[str, Any] = {}
            for col in ("provider", "provider_url", "input_per_1m", "output_per_1m", "notes"):
                if col in row.index:
                    val = row[col]
                    try:
                        is_na = pd.isna(val)
                    except (TypeError, ValueError):
                        is_na = False
                    provider_entry[col] = None if is_na else _nan_to_none(
                        val.item() if hasattr(val, "item") else val
                    )
                else:
                    provider_entry[col] = None
            api_providers.append(provider_entry)

        # Build self-hosted summary (take first row; should be identical across providers)
        self_hosted: dict[str, Any] = {}
        if not model_rows.empty:
            first_row = model_rows.iloc[0]
            sh_col_map = {
                "hourly_cost_usd": "self_hosted_hourly_usd",
                "gpu_setup": "self_hosted_gpu_setup",
                "min_vram_gb": "self_hosted_min_vram_gb",
                "throughput_tps": "self_hosted_throughput_tps",
            }
            for out_key, src_col in sh_col_map.items():
                if src_col in first_row.index:
                    val = first_row[src_col]
                    try:
                        is_na = pd.isna(val)
                    except (TypeError, ValueError):
                        is_na = False
                    self_hosted[out_key] = None if is_na else _nan_to_none(
                        val.item() if hasattr(val, "item") else val
                    )
                else:
                    self_hosted[out_key] = None
        else:
            # Model exists in benchmarks but has no pricing data
            self_hosted = {
                "hourly_cost_usd": None,
                "gpu_setup": None,
                "min_vram_gb": None,
                "throughput_tps": None,
            }

        return jsonify(
            {
                "model_id": model_id,
                "api_providers": api_providers,
                "self_hosted": self_hosted,
            }
        )

    # -----------------------------------------------------------------------

    @app.route("/api/compare", methods=["GET"])
    def api_compare() -> tuple[Response, int] | Response:
        """Return benchmark data for up to 4 models for radar-chart display.

        Query Parameters:
            ids (str, repeatable): Model IDs to compare.  Up to
                :data:`_MAX_COMPARE_MODELS` values are accepted.  May be
                comma-separated or repeated (``?ids=a&ids=b``).

        Returns:
            JSON object::

                {
                    "benchmark_keys": ["mmlu", "humaneval", "math", "gsm8k", "mbpp"],
                    "models": [
                        {
                            "model_id": "deepseek-r1",
                            "display_name": "DeepSeek R1",
                            "family": "DeepSeek",
                            "scores": {
                                "mmlu": 90.8,
                                "humaneval": 92.3,
                                "math": 79.8,
                                "gsm8k": 95.9,
                                "mbpp": 87.6
                            }
                        },
                        ...
                    ]
                }

            Returns HTTP 400 if no IDs are supplied or if more than
            :data:`_MAX_COMPARE_MODELS` IDs are requested.
        """
        model_ids = _parse_list_param("ids")

        if not model_ids:
            return _error_response(
                "At least one model ID is required. Use ?ids=<model_id>.", 400
            )

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids: list[str] = []
        for mid in model_ids:
            if mid not in seen:
                seen.add(mid)
                unique_ids.append(mid)

        if len(unique_ids) > _MAX_COMPARE_MODELS:
            return _error_response(
                f"Too many models requested. Maximum is {_MAX_COMPARE_MODELS}.", 400
            )

        try:
            df = get_merged_df()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error loading data for compare endpoint.")
            return _error_response("Internal server error.", 500)

        selected = get_models_by_ids(df, unique_ids)

        if selected.empty:
            return _error_response(
                f"None of the requested model IDs were found: {unique_ids}", 404
            )

        # Warn about IDs that were not found (but still return what we have)
        found_ids = set(selected["model_id"].tolist())
        missing_ids = [mid for mid in unique_ids if mid not in found_ids]
        if missing_ids:
            logger.warning(
                "api_compare: the following model IDs were not found: %s", missing_ids
            )

        # Build per-model score dicts
        compare_models: list[dict[str, Any]] = []
        for mid in unique_ids:
            row_matches = selected[selected["model_id"] == mid]
            if row_matches.empty:
                continue
            row = row_matches.iloc[0]

            scores: dict[str, Any] = {}
            for key in _BENCHMARK_KEYS:
                col = f"benchmark_{key}"
                if col in row.index:
                    val = row[col]
                    try:
                        is_na = pd.isna(val)
                    except (TypeError, ValueError):
                        is_na = False
                    scores[key] = None if is_na else _nan_to_none(
                        val.item() if hasattr(val, "item") else val
                    )
                else:
                    scores[key] = None

            entry: dict[str, Any] = {
                "model_id": mid,
                "display_name": _safe_str(row, "display_name"),
                "family": _safe_str(row, "family"),
                "parameter_size_bucket": _safe_str(row, "parameter_size_bucket"),
                "license": _safe_str(row, "license"),
                "scores": scores,
            }
            compare_models.append(entry)

        return jsonify(
            {
                "benchmark_keys": list(_BENCHMARK_KEYS),
                "models": compare_models,
                "missing_ids": missing_ids,
            }
        )

    # -----------------------------------------------------------------------
    # Error handlers
    # -----------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(error: Exception) -> tuple[Response, int]:
        """Handle 404 Not Found errors with a JSON response."""
        return _error_response("The requested resource was not found.", 404)

    @app.errorhandler(405)
    def method_not_allowed(error: Exception) -> tuple[Response, int]:
        """Handle 405 Method Not Allowed errors with a JSON response."""
        return _error_response("Method not allowed.", 405)

    @app.errorhandler(500)
    def internal_server_error(error: Exception) -> tuple[Response, int]:
        """Handle 500 Internal Server Error with a JSON response."""
        return _error_response("An internal server error occurred.", 500)

    return app


# ---------------------------------------------------------------------------
# Utility helpers (private)
# ---------------------------------------------------------------------------


def _safe_str(row: pd.Series, col: str) -> str | None:
    """Safely retrieve a string value from a pandas Series row.

    Args:
        row: A pandas Series representing one model row.
        col: Column name to look up.

    Returns:
        String value, or ``None`` if the column is absent or contains NA.
    """
    if col not in row.index:
        return None
    val = row[col]
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val)
