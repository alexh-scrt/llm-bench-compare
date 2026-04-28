"""Data loader module for llm_bench_compare.

This module is responsible for loading, validating, merging, and caching
benchmark scores and pricing data from the JSON data store into pandas
DataFrames.  A module-level cache ensures that JSON files are parsed only
once per process lifetime, keeping HTTP request latency low.

Public API::

    from llm_bench_compare.data_loader import get_merged_df, get_benchmarks_df, get_pricing_df

    df = get_merged_df()          # full merged DataFrame
    benchmarks = get_benchmarks_df()  # benchmark scores only
    pricing = get_pricing_df()        # flattened pricing data
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"
_BENCHMARKS_PATH = _DATA_DIR / "benchmarks.json"
_PRICING_PATH = _DATA_DIR / "pricing.json"

# ---------------------------------------------------------------------------
# Required schema fields
# ---------------------------------------------------------------------------

_REQUIRED_MODEL_FIELDS: frozenset[str] = frozenset(
    {
        "model_id",
        "display_name",
        "family",
        "parameter_size_bucket",
        "license",
        "benchmarks",
        "task_categories",
    }
)

_BENCHMARK_KEYS: tuple[str, ...] = ("mmlu", "humaneval", "math", "gsm8k", "mbpp")

_VALID_SIZE_BUCKETS: frozenset[str] = frozenset({"≤7B", "8–34B", "35B+"})

_VALID_LICENSES: frozenset[str] = frozenset(
    {"Apache-2.0", "MIT", "custom/commercial"}
)

# ---------------------------------------------------------------------------
# Module-level thread-safe cache
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_benchmarks_df_cache: pd.DataFrame | None = None
_pricing_df_cache: pd.DataFrame | None = None
_merged_df_cache: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file from *path*.

    Args:
        path: Absolute or relative :class:`~pathlib.Path` to the JSON file.

    Returns:
        Parsed JSON as a Python dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed as valid JSON.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. "
            "Ensure the package data files are installed correctly."
        )
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _validate_model_record(record: dict[str, Any], index: int) -> list[str]:
    """Validate a single model record from benchmarks.json.

    Checks that all required top-level fields are present and that the
    ``parameter_size_bucket`` and ``license`` values are drawn from the
    known-good sets.  Missing benchmark values (``null``) are acceptable.

    Args:
        record: A single model dictionary from the ``models`` array.
        index: Zero-based position in the array (used in error messages).

    Returns:
        A (possibly empty) list of human-readable validation error strings.
        An empty list means the record is valid.
    """
    errors: list[str] = []
    model_id = record.get("model_id", f"<index {index}>")

    # Check required fields
    for field in _REQUIRED_MODEL_FIELDS:
        if field not in record:
            errors.append(f"[{model_id}] Missing required field: '{field}'")

    # Validate parameter_size_bucket
    bucket = record.get("parameter_size_bucket")
    if bucket is not None and bucket not in _VALID_SIZE_BUCKETS:
        errors.append(
            f"[{model_id}] Unknown parameter_size_bucket '{bucket}'. "
            f"Expected one of: {sorted(_VALID_SIZE_BUCKETS)}"
        )

    # Validate license
    license_val = record.get("license")
    if license_val is not None and license_val not in _VALID_LICENSES:
        # Warn only — new licenses may be added over time
        logger.warning(
            "[%s] Unrecognised license value '%s'. "
            "Consider adding it to _VALID_LICENSES.",
            model_id,
            license_val,
        )

    # Validate benchmarks sub-dict
    benchmarks = record.get("benchmarks")
    if benchmarks is not None:
        if not isinstance(benchmarks, dict):
            errors.append(f"[{model_id}] 'benchmarks' must be a dict, got {type(benchmarks)}")
        else:
            for key in _BENCHMARK_KEYS:
                if key not in benchmarks:
                    errors.append(
                        f"[{model_id}] Missing benchmark key '{key}' in 'benchmarks' dict"
                    )
                else:
                    val = benchmarks[key]
                    if val is not None and not isinstance(val, (int, float)):
                        errors.append(
                            f"[{model_id}] Benchmark '{key}' must be a number or null, "
                            f"got {type(val)}"
                        )

    # Validate task_categories
    categories = record.get("task_categories")
    if categories is not None:
        if not isinstance(categories, list):
            errors.append(
                f"[{model_id}] 'task_categories' must be a list, got {type(categories)}"
            )

    return errors


def _build_benchmarks_df(raw: dict[str, Any]) -> pd.DataFrame:
    """Build a flat :class:`~pandas.DataFrame` from the raw benchmarks JSON.

    Each row corresponds to one model.  Benchmark scores from the nested
    ``benchmarks`` sub-dict are promoted to top-level columns.  The
    ``task_categories`` list is kept as a Python list column so that
    filtering code can use ``.apply()`` or ``.explode()`` as needed.

    Args:
        raw: Parsed content of *benchmarks.json* as a dictionary.

    Returns:
        DataFrame with one row per model and columns for all metadata
        and benchmark scores.

    Raises:
        ValueError: If *raw* does not contain a ``models`` key or if any
            model record fails schema validation.
    """
    if "models" not in raw:
        raise ValueError("benchmarks.json must have a top-level 'models' key.")

    models: list[dict[str, Any]] = raw["models"]
    if not isinstance(models, list):
        raise ValueError("'models' in benchmarks.json must be a JSON array.")

    all_errors: list[str] = []
    for idx, model in enumerate(models):
        errs = _validate_model_record(model, idx)
        all_errors.extend(errs)

    if all_errors:
        error_summary = "\n  ".join(all_errors)
        raise ValueError(
            f"Schema validation failed for benchmarks.json:\n  {error_summary}"
        )

    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {
            "model_id": model.get("model_id"),
            "display_name": model.get("display_name"),
            "family": model.get("family"),
            "version": model.get("version"),
            "parameter_size_b": model.get("parameter_size_b"),
            "parameter_size_bucket": model.get("parameter_size_bucket"),
            "architecture": model.get("architecture"),
            "context_length_k": model.get("context_length_k"),
            "license": model.get("license"),
            "open_weights": model.get("open_weights"),
            "release_date": model.get("release_date"),
            "task_categories": model.get("task_categories", []),
        }
        # Flatten benchmark scores
        bench = model.get("benchmarks") or {}
        for key in _BENCHMARK_KEYS:
            row[f"benchmark_{key}"] = bench.get(key)  # may be None
        rows.append(row)

    df = pd.DataFrame(rows)

    # Enforce sensible dtypes
    df["model_id"] = df["model_id"].astype("string")
    df["display_name"] = df["display_name"].astype("string")
    df["family"] = df["family"].astype("string")
    df["parameter_size_bucket"] = df["parameter_size_bucket"].astype("string")
    df["license"] = df["license"].astype("string")

    for key in _BENCHMARK_KEYS:
        col = f"benchmark_{key}"
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.debug("Built benchmarks DataFrame with %d rows and %d columns.", len(df), len(df.columns))
    return df


def _build_pricing_df(raw: dict[str, Any]) -> pd.DataFrame:
    """Build a flat :class:`~pandas.DataFrame` from the raw pricing JSON.

    The pricing JSON stores per-model lists of provider entries.  This
    function flattens the nested structure so that each row represents a
    single (model, provider) pair.

    Additionally, a ``cheapest_api_input_per_1m`` and
    ``cheapest_api_output_per_1m`` column are computed per model, and a
    ``self_hosted_hourly_usd`` column is joined from the ``self_hosted``
    section of the JSON.

    Args:
        raw: Parsed content of *pricing.json* as a dictionary.

    Returns:
        DataFrame with columns: ``model_id``, ``provider``,
        ``input_per_1m``, ``output_per_1m``, and
        ``self_hosted_hourly_usd``.

    Raises:
        ValueError: If *raw* is missing ``api_providers`` or ``self_hosted`` keys.
    """
    for required_key in ("api_providers", "self_hosted"):
        if required_key not in raw:
            raise ValueError(
                f"pricing.json must have a top-level '{required_key}' key."
            )

    # --- Flatten api_providers ---
    api_rows: list[dict[str, Any]] = []
    for model_entry in raw["api_providers"]:
        model_id: str = model_entry.get("model_id", "")
        if not model_id:
            logger.warning("Skipping api_providers entry with missing model_id.")
            continue
        providers: list[dict[str, Any]] = model_entry.get("providers", [])
        for prov in providers:
            api_rows.append(
                {
                    "model_id": model_id,
                    "provider": prov.get("provider"),
                    "provider_url": prov.get("provider_url"),
                    "input_per_1m": prov.get("input_per_1m"),
                    "output_per_1m": prov.get("output_per_1m"),
                    "notes": prov.get("notes"),
                }
            )

    api_df = pd.DataFrame(
        api_rows,
        columns=["model_id", "provider", "provider_url", "input_per_1m", "output_per_1m", "notes"],
    )
    if not api_df.empty:
        api_df["model_id"] = api_df["model_id"].astype("string")
        api_df["input_per_1m"] = pd.to_numeric(api_df["input_per_1m"], errors="coerce")
        api_df["output_per_1m"] = pd.to_numeric(api_df["output_per_1m"], errors="coerce")

    # --- Build self_hosted lookup ---
    self_hosted_rows: list[dict[str, Any]] = []
    for sh in raw["self_hosted"]:
        model_id = sh.get("model_id", "")
        if not model_id:
            logger.warning("Skipping self_hosted entry with missing model_id.")
            continue
        self_hosted_rows.append(
            {
                "model_id": model_id,
                "self_hosted_hourly_usd": sh.get("estimated_hourly_cost_usd"),
                "self_hosted_gpu_setup": sh.get("recommended_gpu_setup"),
                "self_hosted_min_vram_gb": sh.get("min_vram_gb"),
                "self_hosted_throughput_tps": sh.get("throughput_tokens_per_sec_approx"),
            }
        )

    sh_df = pd.DataFrame(
        self_hosted_rows,
        columns=[
            "model_id",
            "self_hosted_hourly_usd",
            "self_hosted_gpu_setup",
            "self_hosted_min_vram_gb",
            "self_hosted_throughput_tps",
        ],
    )
    if not sh_df.empty:
        sh_df["model_id"] = sh_df["model_id"].astype("string")
        sh_df["self_hosted_hourly_usd"] = pd.to_numeric(
            sh_df["self_hosted_hourly_usd"], errors="coerce"
        )

    # --- Join self_hosted info into api_df ---
    if api_df.empty:
        pricing_df = sh_df.rename(columns={})  # keep as-is
    else:
        pricing_df = api_df.merge(sh_df, on="model_id", how="left")

    logger.debug(
        "Built pricing DataFrame with %d rows and %d columns.",
        len(pricing_df),
        len(pricing_df.columns),
    )
    return pricing_df


def _compute_cheapest_api(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the cheapest API input and output prices per model.

    Args:
        pricing_df: The full pricing DataFrame (one row per model-provider pair).

    Returns:
        DataFrame with columns ``model_id``, ``cheapest_input_per_1m``,
        ``cheapest_output_per_1m``, indexed by ``model_id``.
    """
    if pricing_df.empty or "input_per_1m" not in pricing_df.columns:
        return pd.DataFrame(
            columns=["model_id", "cheapest_input_per_1m", "cheapest_output_per_1m"]
        ).set_index("model_id")

    agg = (
        pricing_df.groupby("model_id", as_index=False)
        .agg(
            cheapest_input_per_1m=("input_per_1m", "min"),
            cheapest_output_per_1m=("output_per_1m", "min"),
        )
        .set_index("model_id")
    )
    return agg


def _build_merged_df(
    benchmarks_df: pd.DataFrame, pricing_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge benchmark and pricing DataFrames into a single wide DataFrame.

    The merge is a left join on ``model_id`` so that models without pricing
    data still appear (with ``NaN`` for pricing columns).  Cheapest API
    prices are pre-computed and appended as convenience columns.

    Args:
        benchmarks_df: Output of :func:`_build_benchmarks_df`.
        pricing_df: Output of :func:`_build_pricing_df`.

    Returns:
        Merged DataFrame with one row per model.
    """
    # Get per-model cheapest API prices
    cheapest = _compute_cheapest_api(pricing_df)

    # Get first self_hosted entry per model (should be unique anyway)
    sh_cols = [
        "model_id",
        "self_hosted_hourly_usd",
        "self_hosted_gpu_setup",
        "self_hosted_min_vram_gb",
        "self_hosted_throughput_tps",
    ]
    available_sh_cols = [c for c in sh_cols if c in pricing_df.columns]

    if available_sh_cols and not pricing_df.empty:
        sh_unique = (
            pricing_df[available_sh_cols]
            .drop_duplicates(subset="model_id")
            .set_index("model_id")
        )
    else:
        sh_unique = pd.DataFrame()

    merged = benchmarks_df.set_index("model_id")

    if not cheapest.empty:
        merged = merged.join(cheapest, how="left")

    if not sh_unique.empty:
        # Drop columns already present in merged to avoid duplicates
        sh_cols_to_join = [c for c in sh_unique.columns if c not in merged.columns]
        if sh_cols_to_join:
            merged = merged.join(sh_unique[sh_cols_to_join], how="left")

    merged = merged.reset_index()

    logger.debug(
        "Built merged DataFrame with %d rows and %d columns.",
        len(merged),
        len(merged.columns),
    )
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_benchmarks_df(force_reload: bool = False) -> pd.DataFrame:
    """Load and return the benchmarks DataFrame, using an in-memory cache.

    The first call parses ``data/benchmarks.json`` and performs schema
    validation.  Subsequent calls return the cached DataFrame instantly.

    Args:
        force_reload: If ``True``, bypass the cache and reload from disk.

    Returns:
        DataFrame with one row per model and columns for benchmark scores
        and model metadata.

    Raises:
        FileNotFoundError: If ``benchmarks.json`` cannot be found.
        ValueError: If the JSON fails schema validation.
    """
    global _benchmarks_df_cache  # noqa: PLW0603

    with _cache_lock:
        if _benchmarks_df_cache is None or force_reload:
            logger.info("Loading benchmarks data from %s", _BENCHMARKS_PATH)
            raw = _load_json(_BENCHMARKS_PATH)
            _benchmarks_df_cache = _build_benchmarks_df(raw)
        return _benchmarks_df_cache.copy()


def load_pricing_df(force_reload: bool = False) -> pd.DataFrame:
    """Load and return the pricing DataFrame, using an in-memory cache.

    The first call parses ``data/pricing.json`` and flattens the nested
    provider structure.  Subsequent calls return the cached DataFrame.

    Args:
        force_reload: If ``True``, bypass the cache and reload from disk.

    Returns:
        DataFrame with one row per (model, provider) pair and columns for
        API pricing and self-hosted cost estimates.

    Raises:
        FileNotFoundError: If ``pricing.json`` cannot be found.
        ValueError: If the JSON fails schema validation.
    """
    global _pricing_df_cache  # noqa: PLW0603

    with _cache_lock:
        if _pricing_df_cache is None or force_reload:
            logger.info("Loading pricing data from %s", _PRICING_PATH)
            raw = _load_json(_PRICING_PATH)
            _pricing_df_cache = _build_pricing_df(raw)
        return _pricing_df_cache.copy()


def get_merged_df(force_reload: bool = False) -> pd.DataFrame:
    """Return the fully merged benchmark + pricing DataFrame.

    This is the primary entry-point for application code.  The result is
    cached after the first call.  Each public column is described below:

    Benchmark columns (float, ``NaN`` if not reported):
        - ``benchmark_mmlu``
        - ``benchmark_humaneval``
        - ``benchmark_math``
        - ``benchmark_gsm8k``
        - ``benchmark_mbpp``

    Pricing columns (float, ``NaN`` if no data):
        - ``cheapest_input_per_1m``  — lowest API input price (USD/1M tokens)
        - ``cheapest_output_per_1m`` — lowest API output price (USD/1M tokens)
        - ``self_hosted_hourly_usd`` — estimated GPU rental cost (USD/hour)
        - ``self_hosted_gpu_setup``  — recommended GPU configuration string
        - ``self_hosted_min_vram_gb``— minimum VRAM required (GB)
        - ``self_hosted_throughput_tps`` — approximate tokens per second

    Args:
        force_reload: If ``True``, bypass all caches and reload from disk.

    Returns:
        Wide DataFrame with one row per model.

    Raises:
        FileNotFoundError: If either JSON data file cannot be found.
        ValueError: If JSON validation fails.
    """
    global _merged_df_cache  # noqa: PLW0603

    with _cache_lock:
        if _merged_df_cache is None or force_reload:
            benchmarks_df = load_benchmarks_df(force_reload=force_reload)
            pricing_df = load_pricing_df(force_reload=force_reload)
            _merged_df_cache = _build_merged_df(benchmarks_df, pricing_df)
        return _merged_df_cache.copy()


# Convenience aliases used by other modules
get_benchmarks_df = load_benchmarks_df
get_pricing_df = load_pricing_df


def clear_cache() -> None:
    """Evict all cached DataFrames, forcing a fresh reload on next access.

    Primarily useful in tests or when data files have been updated at
    runtime.
    """
    global _benchmarks_df_cache, _pricing_df_cache, _merged_df_cache  # noqa: PLW0603

    with _cache_lock:
        _benchmarks_df_cache = None
        _pricing_df_cache = None
        _merged_df_cache = None
    logger.debug("Data loader cache cleared.")


def get_filter_options() -> dict[str, list[str]]:
    """Return the unique filter option values available in the data.

    Reads the merged DataFrame and extracts the distinct values for the
    three main filter dimensions: task category, parameter size bucket,
    and license type.  The task-category list is derived by exploding the
    per-model ``task_categories`` list column.

    Returns:
        A dictionary with keys:

        - ``"task_categories"`` — sorted list of distinct category strings
        - ``"size_buckets"`` — ordered list of size bucket labels
        - ``"licenses"`` — sorted list of distinct license strings
        - ``"families"`` — sorted list of distinct model family strings
    """
    df = get_merged_df()

    # Explode task_categories list column to get unique values
    all_categories: list[str] = sorted(
        {
            cat
            for cats in df["task_categories"]
            if isinstance(cats, list)
            for cat in cats
        }
    )

    # Size buckets in a meaningful display order
    bucket_order = ["≤7B", "8–34B", "35B+"]
    available_buckets = df["parameter_size_bucket"].dropna().unique().tolist()
    size_buckets = [b for b in bucket_order if b in available_buckets]
    # Append any unexpected buckets at the end
    size_buckets += sorted(b for b in available_buckets if b not in bucket_order)

    licenses = sorted(df["license"].dropna().unique().tolist())
    families = sorted(df["family"].dropna().unique().tolist())

    return {
        "task_categories": all_categories,
        "size_buckets": size_buckets,
        "licenses": licenses,
        "families": families,
    }
