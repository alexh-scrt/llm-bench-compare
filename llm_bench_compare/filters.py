"""Filter business logic for llm_bench_compare.

This module provides pure functions that filter the merged benchmark +
pricing DataFrame produced by :mod:`llm_bench_compare.data_loader`.  Every
function accepts a :class:`~pandas.DataFrame` as its first argument and
returns a *new* filtered DataFrame, leaving the input unchanged.

All functions are deliberately free of side-effects (no global state, no I/O)
so they are trivially unit-testable and can be composed freely::

    from llm_bench_compare.data_loader import get_merged_df
    from llm_bench_compare.filters import (
        filter_by_task_category,
        filter_by_size_bucket,
        filter_by_license,
        apply_filters,
    )

    df = get_merged_df()

    # Compose individual filters
    df = filter_by_task_category(df, ["coding", "math"])
    df = filter_by_size_bucket(df, ["≤7B", "8–34B"])

    # Or apply all filters at once
    df = apply_filters(
        df,
        task_categories=["coding"],
        size_buckets=["35B+"],
        licenses=["Apache-2.0", "MIT"],
    )
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical parameter-size bucket labels in display order.
SIZE_BUCKET_ORDER: tuple[str, ...] = ("≤7B", "8–34B", "35B+")

#: Known task category values used in the data store.
KNOWN_TASK_CATEGORIES: frozenset[str] = frozenset({"reasoning", "coding", "math"})

#: Known license values used in the data store.
KNOWN_LICENSES: frozenset[str] = frozenset({"Apache-2.0", "MIT", "custom/commercial"})


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------


def filter_by_task_category(
    df: pd.DataFrame,
    categories: Sequence[str],
) -> pd.DataFrame:
    """Return rows whose ``task_categories`` list intersects *categories*.

    A model is included if **at least one** of its task categories appears in
    the requested *categories* sequence (i.e. the filter uses OR logic across
    the supplied values, not AND).

    If *categories* is empty or ``None``-equivalent, the original DataFrame is
    returned unchanged (no filtering applied).

    Args:
        df: The merged model DataFrame from
            :func:`~llm_bench_compare.data_loader.get_merged_df`.
        categories: A sequence of task-category strings to keep, e.g.
            ``["coding", "math"]``.  Case-sensitive.

    Returns:
        A filtered copy of *df*.  Index is preserved but not reset.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.

    Examples:
        >>> filtered = filter_by_task_category(df, ["coding"])
        >>> filtered = filter_by_task_category(df, [])  # no-op
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    # Normalise: convert to a plain list and strip whitespace
    cats: list[str] = [str(c).strip() for c in categories if c is not None]

    if not cats:
        logger.debug("filter_by_task_category: no categories supplied — returning all %d rows.", len(df))
        return df.copy()

    cats_set: frozenset[str] = frozenset(cats)

    # Warn about unknown categories so callers get early feedback
    unknown = cats_set - KNOWN_TASK_CATEGORIES
    if unknown:
        logger.warning(
            "filter_by_task_category: unknown category value(s) %s. "
            "Known categories: %s",
            sorted(unknown),
            sorted(KNOWN_TASK_CATEGORIES),
        )

    def _has_category(cell: object) -> bool:
        """Return True if the cell (list of categories) intersects cats_set."""
        if not isinstance(cell, list):
            return False
        return bool(frozenset(cell) & cats_set)

    if "task_categories" not in df.columns:
        logger.warning(
            "filter_by_task_category: 'task_categories' column not found — returning empty DataFrame."
        )
        return df.iloc[0:0].copy()

    mask = df["task_categories"].apply(_has_category)
    result = df[mask].copy()
    logger.debug(
        "filter_by_task_category: categories=%s → %d/%d rows.",
        cats,
        len(result),
        len(df),
    )
    return result


def filter_by_size_bucket(
    df: pd.DataFrame,
    buckets: Sequence[str],
) -> pd.DataFrame:
    """Return rows whose ``parameter_size_bucket`` is in *buckets*.

    If *buckets* is empty, the original DataFrame is returned unchanged.

    Args:
        df: The merged model DataFrame.
        buckets: A sequence of size-bucket strings to keep, e.g.
            ``["≤7B", "8–34B"]``.  Valid values are ``"≤7B"``,
            ``"8–34B"``, and ``"35B+"``.

    Returns:
        A filtered copy of *df*.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.

    Examples:
        >>> small_models = filter_by_size_bucket(df, ["≤7B"])
        >>> mid_models   = filter_by_size_bucket(df, ["8–34B", "35B+"])
        >>> all_models   = filter_by_size_bucket(df, [])  # no-op
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    bucket_list: list[str] = [str(b).strip() for b in buckets if b is not None]

    if not bucket_list:
        logger.debug("filter_by_size_bucket: no buckets supplied — returning all %d rows.", len(df))
        return df.copy()

    buckets_set: frozenset[str] = frozenset(bucket_list)

    unknown = buckets_set - frozenset(SIZE_BUCKET_ORDER)
    if unknown:
        logger.warning(
            "filter_by_size_bucket: unknown bucket value(s) %s. "
            "Known buckets: %s",
            sorted(unknown),
            list(SIZE_BUCKET_ORDER),
        )

    if "parameter_size_bucket" not in df.columns:
        logger.warning(
            "filter_by_size_bucket: 'parameter_size_bucket' column not found — returning empty DataFrame."
        )
        return df.iloc[0:0].copy()

    mask = df["parameter_size_bucket"].isin(buckets_set)
    result = df[mask].copy()
    logger.debug(
        "filter_by_size_bucket: buckets=%s → %d/%d rows.",
        bucket_list,
        len(result),
        len(df),
    )
    return result


def filter_by_license(
    df: pd.DataFrame,
    licenses: Sequence[str],
) -> pd.DataFrame:
    """Return rows whose ``license`` field is in *licenses*.

    If *licenses* is empty, the original DataFrame is returned unchanged.

    Args:
        df: The merged model DataFrame.
        licenses: A sequence of license strings to keep, e.g.
            ``["Apache-2.0", "MIT"]``.  Valid values are
            ``"Apache-2.0"``, ``"MIT"``, and ``"custom/commercial"``.

    Returns:
        A filtered copy of *df*.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.

    Examples:
        >>> open_models = filter_by_license(df, ["Apache-2.0", "MIT"])
        >>> all_models  = filter_by_license(df, [])  # no-op
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    license_list: list[str] = [str(lic).strip() for lic in licenses if lic is not None]

    if not license_list:
        logger.debug("filter_by_license: no licenses supplied — returning all %d rows.", len(df))
        return df.copy()

    licenses_set: frozenset[str] = frozenset(license_list)

    unknown = licenses_set - KNOWN_LICENSES
    if unknown:
        logger.warning(
            "filter_by_license: unknown license value(s) %s. "
            "Known licenses: %s",
            sorted(unknown),
            sorted(KNOWN_LICENSES),
        )

    if "license" not in df.columns:
        logger.warning(
            "filter_by_license: 'license' column not found — returning empty DataFrame."
        )
        return df.iloc[0:0].copy()

    mask = df["license"].isin(licenses_set)
    result = df[mask].copy()
    logger.debug(
        "filter_by_license: licenses=%s → %d/%d rows.",
        license_list,
        len(result),
        len(df),
    )
    return result


def filter_by_family(
    df: pd.DataFrame,
    families: Sequence[str],
) -> pd.DataFrame:
    """Return rows whose ``family`` field is in *families*.

    If *families* is empty, the original DataFrame is returned unchanged.

    Args:
        df: The merged model DataFrame.
        families: A sequence of model family strings to keep, e.g.
            ``["DeepSeek", "Llama"]``.

    Returns:
        A filtered copy of *df*.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.

    Examples:
        >>> deepseek = filter_by_family(df, ["DeepSeek"])
        >>> all_models = filter_by_family(df, [])  # no-op
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    family_list: list[str] = [str(f).strip() for f in families if f is not None]

    if not family_list:
        logger.debug("filter_by_family: no families supplied — returning all %d rows.", len(df))
        return df.copy()

    families_set: frozenset[str] = frozenset(family_list)

    if "family" not in df.columns:
        logger.warning(
            "filter_by_family: 'family' column not found — returning empty DataFrame."
        )
        return df.iloc[0:0].copy()

    mask = df["family"].isin(families_set)
    result = df[mask].copy()
    logger.debug(
        "filter_by_family: families=%s → %d/%d rows.",
        family_list,
        len(result),
        len(df),
    )
    return result


def filter_by_open_weights(
    df: pd.DataFrame,
    open_weights_only: bool,
) -> pd.DataFrame:
    """Optionally restrict results to models with publicly available weights.

    Args:
        df: The merged model DataFrame.
        open_weights_only: If ``True``, only rows where ``open_weights`` is
            ``True`` are returned.  If ``False``, all rows are returned
            unchanged.

    Returns:
        A filtered copy of *df*.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    if not open_weights_only:
        return df.copy()

    if "open_weights" not in df.columns:
        logger.warning(
            "filter_by_open_weights: 'open_weights' column not found — returning empty DataFrame."
        )
        return df.iloc[0:0].copy()

    mask = df["open_weights"] == True  # noqa: E712 — intentional: handles pandas NA safely
    result = df[mask].copy()
    logger.debug(
        "filter_by_open_weights: open_weights_only=True → %d/%d rows.",
        len(result),
        len(df),
    )
    return result


def sort_by_benchmark(
    df: pd.DataFrame,
    benchmark: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """Sort *df* by a benchmark score column.

    Models with ``NaN`` scores for the chosen benchmark are placed last
    regardless of sort direction.

    Args:
        df: The merged model DataFrame.
        benchmark: The benchmark short-name to sort by.  One of
            ``"mmlu"``, ``"humaneval"``, ``"math"``, ``"gsm8k"``,
            ``"mbpp"``.
        ascending: If ``True``, sort lowest score first.  Default is
            ``False`` (highest first).

    Returns:
        A sorted copy of *df*.  The integer index is reset so that it
        runs 0 … n-1.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.
        ValueError: If *benchmark* is not a recognised benchmark key.

    Examples:
        >>> best_coders = sort_by_benchmark(df, "humaneval")
        >>> worst_first = sort_by_benchmark(df, "mmlu", ascending=True)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    col = f"benchmark_{benchmark}"
    if col not in df.columns:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Expected one of: mmlu, humaneval, math, gsm8k, mbpp."
        )

    result = df.sort_values(col, ascending=ascending, na_position="last").reset_index(drop=True)
    logger.debug(
        "sort_by_benchmark: benchmark=%s, ascending=%s → %d rows.",
        benchmark,
        ascending,
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Composite apply_filters function
# ---------------------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    *,
    task_categories: Sequence[str] | None = None,
    size_buckets: Sequence[str] | None = None,
    licenses: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    open_weights_only: bool = False,
    sort_benchmark: str | None = None,
    sort_ascending: bool = False,
) -> pd.DataFrame:
    """Apply all active filters to *df* and optionally sort the result.

    This is the primary entry-point for route handlers.  Each filter
    argument is optional; passing ``None`` or an empty sequence for a
    filter dimension disables that filter (i.e. all values pass through).

    Filters are applied in the following order:

    1. Task category  (OR logic within the dimension)
    2. Parameter size bucket  (OR logic)
    3. License type  (OR logic)
    4. Model family  (OR logic)
    5. Open-weights flag
    6. Sort by benchmark score  (optional)

    Args:
        df: The merged model DataFrame from
            :func:`~llm_bench_compare.data_loader.get_merged_df`.
        task_categories: Sequence of task-category strings.  Models are
            kept when **any** of their categories appears in this list.
            Pass ``None`` or ``[]`` to disable.
        size_buckets: Sequence of size-bucket strings (``"≤7B"``,
            ``"8–34B"``, ``"35B+"``).  Pass ``None`` or ``[]`` to
            disable.
        licenses: Sequence of license strings.  Pass ``None`` or ``[]``
            to disable.
        families: Sequence of model family strings.  Pass ``None`` or
            ``[]`` to disable.
        open_weights_only: If ``True``, only models with publicly
            available weights are included.
        sort_benchmark: Optional benchmark key to sort by.  One of
            ``"mmlu"``, ``"humaneval"``, ``"math"``, ``"gsm8k"``,
            ``"mbpp"``.
        sort_ascending: Sort direction; only meaningful when
            *sort_benchmark* is provided.  Default ``False`` (highest
            first).

    Returns:
        A filtered (and optionally sorted) copy of *df*.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.
        ValueError: If *sort_benchmark* is not a recognised benchmark key.

    Examples:
        >>> result = apply_filters(
        ...     df,
        ...     task_categories=["coding"],
        ...     size_buckets=["35B+"],
        ...     licenses=["Apache-2.0", "MIT"],
        ...     sort_benchmark="humaneval",
        ... )
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    result = df.copy()

    # --- Task category filter ---
    if task_categories is not None:
        result = filter_by_task_category(result, task_categories)

    # --- Size bucket filter ---
    if size_buckets is not None:
        result = filter_by_size_bucket(result, size_buckets)

    # --- License filter ---
    if licenses is not None:
        result = filter_by_license(result, licenses)

    # --- Family filter ---
    if families is not None:
        result = filter_by_family(result, families)

    # --- Open weights filter ---
    if open_weights_only:
        result = filter_by_open_weights(result, open_weights_only=True)

    # --- Optional sort ---
    if sort_benchmark is not None:
        result = sort_by_benchmark(result, sort_benchmark, ascending=sort_ascending)
    else:
        result = result.reset_index(drop=True)

    logger.debug(
        "apply_filters: final result has %d/%d rows.",
        len(result),
        len(df),
    )
    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_model_by_id(df: pd.DataFrame, model_id: str) -> pd.Series | None:
    """Look up a single model row by its ``model_id``.

    Args:
        df: The merged model DataFrame.
        model_id: The unique model identifier string.

    Returns:
        A :class:`~pandas.Series` representing the first matching row, or
        ``None`` if no match is found.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    if "model_id" not in df.columns:
        return None

    matches = df[df["model_id"] == model_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def get_models_by_ids(df: pd.DataFrame, model_ids: Sequence[str]) -> pd.DataFrame:
    """Return rows matching any of the given *model_ids*.

    Useful for the radar chart comparison feature where a user selects up
    to 4 specific models.

    Args:
        df: The merged model DataFrame.
        model_ids: A sequence of model identifier strings.  Duplicates are
            ignored.  Order is preserved relative to *df*.

    Returns:
        A filtered copy of *df* containing only the requested models, in
        the same order they appear in *df*.  If none of the IDs match,
        an empty DataFrame with the same columns is returned.

    Raises:
        TypeError: If *df* is not a :class:`~pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}")

    if not model_ids:
        return df.iloc[0:0].copy()

    ids_set: frozenset[str] = frozenset(str(mid) for mid in model_ids if mid is not None)

    if "model_id" not in df.columns:
        return df.iloc[0:0].copy()

    mask = df["model_id"].isin(ids_set)
    return df[mask].copy()
