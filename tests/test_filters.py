"""Unit tests for llm_bench_compare.filters.

Covers all filter functions, edge cases (empty input, no matches, multi-value
filters), the composite apply_filters function, and utility helpers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from llm_bench_compare.data_loader import clear_cache, get_merged_df
from llm_bench_compare.filters import (
    KNOWN_LICENSES,
    KNOWN_TASK_CATEGORIES,
    SIZE_BUCKET_ORDER,
    apply_filters,
    filter_by_family,
    filter_by_license,
    filter_by_open_weights,
    filter_by_size_bucket,
    filter_by_task_category,
    get_model_by_id,
    get_models_by_ids,
    sort_by_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_df() -> pd.DataFrame:
    """Load the real merged DataFrame once per test module."""
    clear_cache()
    return get_merged_df()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A small synthetic DataFrame for fast, isolated unit tests."""
    rows: list[dict[str, Any]] = [
        {
            "model_id": "model-alpha",
            "display_name": "Model Alpha",
            "family": "Alpha",
            "parameter_size_bucket": "≤7B",
            "license": "Apache-2.0",
            "open_weights": True,
            "task_categories": ["reasoning", "coding"],
            "benchmark_mmlu": 70.0,
            "benchmark_humaneval": 65.0,
            "benchmark_math": 40.0,
            "benchmark_gsm8k": 80.0,
            "benchmark_mbpp": 55.0,
        },
        {
            "model_id": "model-beta",
            "display_name": "Model Beta",
            "family": "Beta",
            "parameter_size_bucket": "8–34B",
            "license": "MIT",
            "open_weights": True,
            "task_categories": ["math"],
            "benchmark_mmlu": 80.0,
            "benchmark_humaneval": None,
            "benchmark_math": 60.0,
            "benchmark_gsm8k": 88.0,
            "benchmark_mbpp": None,
        },
        {
            "model_id": "model-gamma",
            "display_name": "Model Gamma",
            "family": "Gamma",
            "parameter_size_bucket": "35B+",
            "license": "custom/commercial",
            "open_weights": False,
            "task_categories": ["reasoning", "math", "coding"],
            "benchmark_mmlu": 90.0,
            "benchmark_humaneval": 88.0,
            "benchmark_math": 75.0,
            "benchmark_gsm8k": 95.0,
            "benchmark_mbpp": 82.0,
        },
        {
            "model_id": "model-delta",
            "display_name": "Model Delta",
            "family": "Alpha",
            "parameter_size_bucket": "35B+",
            "license": "Apache-2.0",
            "open_weights": True,
            "task_categories": ["coding"],
            "benchmark_mmlu": 85.0,
            "benchmark_humaneval": 91.0,
            "benchmark_math": None,
            "benchmark_gsm8k": None,
            "benchmark_mbpp": 89.0,
        },
    ]
    df = pd.DataFrame(rows)
    df["benchmark_humaneval"] = pd.to_numeric(df["benchmark_humaneval"], errors="coerce")
    df["benchmark_math"] = pd.to_numeric(df["benchmark_math"], errors="coerce")
    df["benchmark_gsm8k"] = pd.to_numeric(df["benchmark_gsm8k"], errors="coerce")
    df["benchmark_mbpp"] = pd.to_numeric(df["benchmark_mbpp"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# filter_by_task_category
# ---------------------------------------------------------------------------


class TestFilterByTaskCategory:
    """Tests for filter_by_task_category."""

    def test_single_category_returns_matching_rows(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_task_category(sample_df, ["math"])
        # model-beta and model-gamma have "math"
        assert set(result["model_id"]) == {"model-beta", "model-gamma"}

    def test_coding_category(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_task_category(sample_df, ["coding"])
        assert set(result["model_id"]) == {"model-alpha", "model-gamma", "model-delta"}

    def test_multiple_categories_uses_or_logic(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_task_category(sample_df, ["math", "coding"])
        # model-alpha (coding), model-beta (math), model-gamma (both), model-delta (coding)
        assert set(result["model_id"]) == {"model-alpha", "model-beta", "model-gamma", "model-delta"}

    def test_empty_categories_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_task_category(sample_df, [])
        assert len(result) == len(sample_df)

    def test_nonexistent_category_returns_empty(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_task_category(sample_df, ["nonexistent"])
        assert len(result) == 0

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_task_category(sample_df, ["coding"])
        result["_new_col"] = 1
        assert "_new_col" not in sample_df.columns

    def test_input_df_not_mutated(self, sample_df: pd.DataFrame) -> None:
        original_len = len(sample_df)
        filter_by_task_category(sample_df, ["coding"])
        assert len(sample_df) == original_len

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pandas DataFrame"):
            filter_by_task_category("not a df", ["coding"])  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        df_no_col = sample_df.drop(columns=["task_categories"])
        result = filter_by_task_category(df_no_col, ["coding"])
        assert len(result) == 0

    def test_reasoning_category_correct_count(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_task_category(sample_df, ["reasoning"])
        # model-alpha and model-gamma have "reasoning"
        assert set(result["model_id"]) == {"model-alpha", "model-gamma"}

    def test_all_three_categories_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_task_category(sample_df, ["reasoning", "coding", "math"])
        assert len(result) == len(sample_df)

    def test_none_values_in_sequence_ignored(self, sample_df: pd.DataFrame) -> None:
        # Should not raise; None entries are stripped
        result = filter_by_task_category(sample_df, ["coding", None])  # type: ignore[list-item]
        assert len(result) > 0

    def test_real_data_coding_filter(self, full_df: pd.DataFrame) -> None:
        result = filter_by_task_category(full_df, ["coding"])
        assert len(result) > 0
        for cats in result["task_categories"]:
            assert "coding" in cats

    def test_real_data_math_filter(self, full_df: pd.DataFrame) -> None:
        result = filter_by_task_category(full_df, ["math"])
        assert len(result) > 0
        for cats in result["task_categories"]:
            assert "math" in cats

    def test_real_data_multi_category(self, full_df: pd.DataFrame) -> None:
        result_single = filter_by_task_category(full_df, ["coding"])
        result_multi = filter_by_task_category(full_df, ["coding", "math"])
        assert len(result_multi) >= len(result_single)


# ---------------------------------------------------------------------------
# filter_by_size_bucket
# ---------------------------------------------------------------------------


class TestFilterBySizeBucket:
    """Tests for filter_by_size_bucket."""

    def test_single_bucket_small(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(sample_df, ["≤7B"])
        assert list(result["model_id"]) == ["model-alpha"]

    def test_single_bucket_mid(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(sample_df, ["8–34B"])
        assert list(result["model_id"]) == ["model-beta"]

    def test_single_bucket_large(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(sample_df, ["35B+"])
        assert set(result["model_id"]) == {"model-gamma", "model-delta"}

    def test_multiple_buckets_or_logic(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(sample_df, ["≤7B", "35B+"])
        assert set(result["model_id"]) == {"model-alpha", "model-gamma", "model-delta"}

    def test_all_buckets_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(sample_df, ["≤7B", "8–34B", "35B+"])
        assert len(result) == len(sample_df)

    def test_empty_buckets_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(sample_df, [])
        assert len(result) == len(sample_df)

    def test_nonexistent_bucket_returns_empty(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(sample_df, ["1000B+"])
        assert len(result) == 0

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(sample_df, ["≤7B"])
        result["_extra"] = 99
        assert "_extra" not in sample_df.columns

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pandas DataFrame"):
            filter_by_size_bucket(42, ["≤7B"])  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        df_no_col = sample_df.drop(columns=["parameter_size_bucket"])
        result = filter_by_size_bucket(df_no_col, ["≤7B"])
        assert len(result) == 0

    def test_real_data_small_models(self, full_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(full_df, ["≤7B"])
        assert len(result) > 0
        assert all(result["parameter_size_bucket"] == "≤7B")

    def test_real_data_large_models(self, full_df: pd.DataFrame) -> None:
        result = filter_by_size_bucket(full_df, ["35B+"])
        assert len(result) > 0
        assert all(result["parameter_size_bucket"] == "35B+")

    def test_real_data_combined_buckets_superset(self, full_df: pd.DataFrame) -> None:
        r1 = filter_by_size_bucket(full_df, ["≤7B"])
        r2 = filter_by_size_bucket(full_df, ["8–34B"])
        r_combined = filter_by_size_bucket(full_df, ["≤7B", "8–34B"])
        assert len(r_combined) == len(r1) + len(r2)


# ---------------------------------------------------------------------------
# filter_by_license
# ---------------------------------------------------------------------------


class TestFilterByLicense:
    """Tests for filter_by_license."""

    def test_apache_license_filter(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_license(sample_df, ["Apache-2.0"])
        assert set(result["model_id"]) == {"model-alpha", "model-delta"}

    def test_mit_license_filter(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_license(sample_df, ["MIT"])
        assert list(result["model_id"]) == ["model-beta"]

    def test_custom_license_filter(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_license(sample_df, ["custom/commercial"])
        assert list(result["model_id"]) == ["model-gamma"]

    def test_multiple_licenses_or_logic(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_license(sample_df, ["Apache-2.0", "MIT"])
        assert set(result["model_id"]) == {"model-alpha", "model-beta", "model-delta"}

    def test_all_licenses_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_license(sample_df, ["Apache-2.0", "MIT", "custom/commercial"])
        assert len(result) == len(sample_df)

    def test_empty_licenses_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_license(sample_df, [])
        assert len(result) == len(sample_df)

    def test_nonexistent_license_returns_empty(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_license(sample_df, ["GPL-3.0"])
        assert len(result) == 0

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_license(sample_df, ["MIT"])
        result["_extra"] = 1
        assert "_extra" not in sample_df.columns

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pandas DataFrame"):
            filter_by_license(None, ["MIT"])  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        df_no_col = sample_df.drop(columns=["license"])
        result = filter_by_license(df_no_col, ["MIT"])
        assert len(result) == 0

    def test_real_data_apache_filter(self, full_df: pd.DataFrame) -> None:
        result = filter_by_license(full_df, ["Apache-2.0"])
        assert len(result) > 0
        assert all(result["license"] == "Apache-2.0")

    def test_real_data_open_licenses(self, full_df: pd.DataFrame) -> None:
        result = filter_by_license(full_df, ["Apache-2.0", "MIT"])
        assert len(result) > 0
        assert all(result["license"].isin(["Apache-2.0", "MIT"]))

    def test_real_data_custom_commercial(self, full_df: pd.DataFrame) -> None:
        result = filter_by_license(full_df, ["custom/commercial"])
        assert len(result) > 0
        assert all(result["license"] == "custom/commercial")


# ---------------------------------------------------------------------------
# filter_by_family
# ---------------------------------------------------------------------------


class TestFilterByFamily:
    """Tests for filter_by_family."""

    def test_single_family(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_family(sample_df, ["Alpha"])
        assert set(result["model_id"]) == {"model-alpha", "model-delta"}

    def test_multiple_families(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_family(sample_df, ["Alpha", "Beta"])
        assert set(result["model_id"]) == {"model-alpha", "model-beta", "model-delta"}

    def test_empty_families_returns_all(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_family(sample_df, [])
        assert len(result) == len(sample_df)

    def test_nonexistent_family_returns_empty(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_family(sample_df, ["NoSuchFamily"])
        assert len(result) == 0

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            filter_by_family([], ["Alpha"])  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        df_no_col = sample_df.drop(columns=["family"])
        result = filter_by_family(df_no_col, ["Alpha"])
        assert len(result) == 0

    def test_real_data_deepseek_family(self, full_df: pd.DataFrame) -> None:
        result = filter_by_family(full_df, ["DeepSeek"])
        assert len(result) >= 3  # V2, V3, R1
        assert all(result["family"] == "DeepSeek")

    def test_real_data_llama_and_mistral(self, full_df: pd.DataFrame) -> None:
        result = filter_by_family(full_df, ["Llama", "Mistral"])
        assert len(result) > 0
        assert all(result["family"].isin(["Llama", "Mistral"]))


# ---------------------------------------------------------------------------
# filter_by_open_weights
# ---------------------------------------------------------------------------


class TestFilterByOpenWeights:
    """Tests for filter_by_open_weights."""

    def test_open_weights_only_excludes_closed(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_open_weights(sample_df, open_weights_only=True)
        # model-gamma has open_weights=False
        assert "model-gamma" not in set(result["model_id"])

    def test_open_weights_only_includes_open(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_open_weights(sample_df, open_weights_only=True)
        assert set(result["model_id"]) == {"model-alpha", "model-beta", "model-delta"}

    def test_false_flag_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = filter_by_open_weights(sample_df, open_weights_only=False)
        assert len(result) == len(sample_df)

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            filter_by_open_weights("bad", open_weights_only=True)  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        df_no_col = sample_df.drop(columns=["open_weights"])
        result = filter_by_open_weights(df_no_col, open_weights_only=True)
        assert len(result) == 0

    def test_real_data_open_weights(self, full_df: pd.DataFrame) -> None:
        result = filter_by_open_weights(full_df, open_weights_only=True)
        assert len(result) > 0
        assert all(result["open_weights"] == True)  # noqa: E712


# ---------------------------------------------------------------------------
# sort_by_benchmark
# ---------------------------------------------------------------------------


class TestSortByBenchmark:
    """Tests for sort_by_benchmark."""

    def test_default_sort_descending(self, sample_df: pd.DataFrame) -> None:
        result = sort_by_benchmark(sample_df, "mmlu")
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)

    def test_ascending_sort(self, sample_df: pd.DataFrame) -> None:
        result = sort_by_benchmark(sample_df, "mmlu", ascending=True)
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores)

    def test_nan_values_placed_last(self, sample_df: pd.DataFrame) -> None:
        result = sort_by_benchmark(sample_df, "humaneval")
        # model-beta has NaN humaneval → should appear last
        last_row = result.iloc[-1]
        assert pd.isna(last_row["benchmark_humaneval"])

    def test_index_is_reset(self, sample_df: pd.DataFrame) -> None:
        result = sort_by_benchmark(sample_df, "mmlu")
        assert list(result.index) == list(range(len(result)))

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        result = sort_by_benchmark(sample_df, "mmlu")
        result["_extra"] = 0
        assert "_extra" not in sample_df.columns

    def test_raises_value_error_for_unknown_benchmark(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown benchmark"):
            sort_by_benchmark(sample_df, "nonexistent")

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            sort_by_benchmark({}, "mmlu")  # type: ignore[arg-type]

    def test_all_benchmark_keys_work(self, sample_df: pd.DataFrame) -> None:
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            result = sort_by_benchmark(sample_df, key)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_df)

    def test_real_data_sort_by_mmlu(self, full_df: pd.DataFrame) -> None:
        result = sort_by_benchmark(full_df, "mmlu")
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# apply_filters (composite)
# ---------------------------------------------------------------------------


class TestApplyFilters:
    """Tests for the composite apply_filters function."""

    def test_no_filters_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df)
        assert len(result) == len(sample_df)

    def test_none_filters_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(
            sample_df,
            task_categories=None,
            size_buckets=None,
            licenses=None,
        )
        assert len(result) == len(sample_df)

    def test_single_task_category_filter(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df, task_categories=["math"])
        assert set(result["model_id"]) == {"model-beta", "model-gamma"}

    def test_single_size_bucket_filter(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df, size_buckets=["≤7B"])
        assert list(result["model_id"]) == ["model-alpha"]

    def test_single_license_filter(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df, licenses=["MIT"])
        assert list(result["model_id"]) == ["model-beta"]

    def test_combined_category_and_bucket_filters(self, sample_df: pd.DataFrame) -> None:
        # coding models that are ≤7B or 8–34B
        result = apply_filters(
            sample_df,
            task_categories=["coding"],
            size_buckets=["≤7B", "8–34B"],
        )
        # model-alpha: coding + ≤7B ✓
        # model-beta: math only (no coding) → excluded
        # model-gamma: coding + 35B+ → excluded by bucket
        # model-delta: coding + 35B+ → excluded by bucket
        assert set(result["model_id"]) == {"model-alpha"}

    def test_combined_bucket_and_license_filters(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(
            sample_df,
            size_buckets=["35B+"],
            licenses=["Apache-2.0"],
        )
        # model-gamma is 35B+ but custom/commercial → excluded
        # model-delta is 35B+ and Apache-2.0 ✓
        assert set(result["model_id"]) == {"model-delta"}

    def test_all_filters_combined_narrows_results(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(
            sample_df,
            task_categories=["coding"],
            size_buckets=["≤7B"],
            licenses=["Apache-2.0"],
        )
        assert list(result["model_id"]) == ["model-alpha"]

    def test_conflicting_filters_return_empty(self, sample_df: pd.DataFrame) -> None:
        # No model is both ≤7B and 35B+
        result = apply_filters(
            sample_df,
            size_buckets=["≤7B"],
            licenses=["MIT"],  # model-beta is MIT but 8-34B, not ≤7B
        )
        assert len(result) == 0

    def test_sort_benchmark_applied(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df, sort_benchmark="mmlu")
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)

    def test_sort_ascending(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df, sort_benchmark="mmlu", sort_ascending=True)
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores)

    def test_sort_with_filter_combined(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(
            sample_df,
            size_buckets=["35B+"],
            sort_benchmark="humaneval",
        )
        assert len(result) == 2  # gamma and delta
        scores = result["benchmark_humaneval"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)

    def test_open_weights_only_filter(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df, open_weights_only=True)
        assert "model-gamma" not in set(result["model_id"])
        assert len(result) == 3

    def test_open_weights_false_no_effect(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df, open_weights_only=False)
        assert len(result) == len(sample_df)

    def test_family_filter_applied(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df, families=["Alpha"])
        assert set(result["model_id"]) == {"model-alpha", "model-delta"}

    def test_family_filter_with_other_filters(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(
            sample_df,
            families=["Alpha"],
            size_buckets=["35B+"],
        )
        assert set(result["model_id"]) == {"model-delta"}

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(sample_df)
        result["_extra"] = 99
        assert "_extra" not in sample_df.columns

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            apply_filters("not a df")  # type: ignore[arg-type]

    def test_raises_value_error_for_invalid_sort_benchmark(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown benchmark"):
            apply_filters(sample_df, sort_benchmark="invalid")

    def test_empty_input_df_returns_empty(self) -> None:
        empty_df = pd.DataFrame(
            columns=[
                "model_id", "family", "parameter_size_bucket",
                "license", "open_weights", "task_categories",
                "benchmark_mmlu", "benchmark_humaneval",
                "benchmark_math", "benchmark_gsm8k", "benchmark_mbpp",
            ]
        )
        result = apply_filters(
            empty_df,
            task_categories=["coding"],
            size_buckets=["≤7B"],
            licenses=["MIT"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_real_data_no_filters(self, full_df: pd.DataFrame) -> None:
        result = apply_filters(full_df)
        assert len(result) == len(full_df)

    def test_real_data_coding_apache(self, full_df: pd.DataFrame) -> None:
        result = apply_filters(
            full_df,
            task_categories=["coding"],
            licenses=["Apache-2.0"],
        )
        assert len(result) > 0
        assert all(result["license"] == "Apache-2.0")
        for cats in result["task_categories"]:
            assert "coding" in cats

    def test_real_data_small_math_models(self, full_df: pd.DataFrame) -> None:
        result = apply_filters(
            full_df,
            task_categories=["math"],
            size_buckets=["≤7B"],
        )
        # Should be a subset; may be empty if no such model
        assert isinstance(result, pd.DataFrame)
        assert all(result["parameter_size_bucket"] == "≤7B")

    def test_real_data_sort_by_humaneval(self, full_df: pd.DataFrame) -> None:
        result = apply_filters(full_df, sort_benchmark="humaneval")
        valid_scores = result["benchmark_humaneval"].dropna().tolist()
        assert valid_scores == sorted(valid_scores, reverse=True)

    def test_index_always_reset(self, sample_df: pd.DataFrame) -> None:
        result = apply_filters(
            sample_df,
            size_buckets=["≤7B"],
        )
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# get_model_by_id
# ---------------------------------------------------------------------------


class TestGetModelById:
    """Tests for get_model_by_id."""

    def test_returns_series_for_known_id(self, sample_df: pd.DataFrame) -> None:
        row = get_model_by_id(sample_df, "model-alpha")
        assert row is not None
        assert isinstance(row, pd.Series)
        assert row["model_id"] == "model-alpha"

    def test_returns_none_for_unknown_id(self, sample_df: pd.DataFrame) -> None:
        row = get_model_by_id(sample_df, "nonexistent-model")
        assert row is None

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            get_model_by_id("bad", "model-alpha")  # type: ignore[arg-type]

    def test_missing_model_id_column_returns_none(self, sample_df: pd.DataFrame) -> None:
        df_no_col = sample_df.drop(columns=["model_id"])
        result = get_model_by_id(df_no_col, "model-alpha")
        assert result is None

    def test_real_data_known_model(self, full_df: pd.DataFrame) -> None:
        row = get_model_by_id(full_df, "deepseek-r1")
        assert row is not None
        assert row["display_name"] == "DeepSeek R1"

    def test_real_data_unknown_model(self, full_df: pd.DataFrame) -> None:
        row = get_model_by_id(full_df, "gpt-5-turbo-ultra")
        assert row is None


# ---------------------------------------------------------------------------
# get_models_by_ids
# ---------------------------------------------------------------------------


class TestGetModelsByIds:
    """Tests for get_models_by_ids."""

    def test_returns_matching_rows(self, sample_df: pd.DataFrame) -> None:
        result = get_models_by_ids(sample_df, ["model-alpha", "model-gamma"])
        assert set(result["model_id"]) == {"model-alpha", "model-gamma"}

    def test_empty_ids_returns_empty_df(self, sample_df: pd.DataFrame) -> None:
        result = get_models_by_ids(sample_df, [])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == list(sample_df.columns)

    def test_unknown_ids_return_empty(self, sample_df: pd.DataFrame) -> None:
        result = get_models_by_ids(sample_df, ["does-not-exist"])
        assert len(result) == 0

    def test_partial_match(self, sample_df: pd.DataFrame) -> None:
        result = get_models_by_ids(sample_df, ["model-alpha", "does-not-exist"])
        assert set(result["model_id"]) == {"model-alpha"}

    def test_duplicate_ids_not_duplicated_in_result(self, sample_df: pd.DataFrame) -> None:
        result = get_models_by_ids(sample_df, ["model-alpha", "model-alpha"])
        assert len(result) == 1

    def test_all_ids_returns_full_df(self, sample_df: pd.DataFrame) -> None:
        all_ids = list(sample_df["model_id"])
        result = get_models_by_ids(sample_df, all_ids)
        assert len(result) == len(sample_df)

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        result = get_models_by_ids(sample_df, ["model-alpha"])
        result["_extra"] = 1
        assert "_extra" not in sample_df.columns

    def test_raises_type_error_for_non_dataframe(self) -> None:
        with pytest.raises(TypeError):
            get_models_by_ids(None, ["model-alpha"])  # type: ignore[arg-type]

    def test_missing_model_id_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        df_no_col = sample_df.drop(columns=["model_id"])
        result = get_models_by_ids(df_no_col, ["model-alpha"])
        assert len(result) == 0

    def test_real_data_select_two_models(self, full_df: pd.DataFrame) -> None:
        result = get_models_by_ids(full_df, ["deepseek-r1", "qwen2.5-72b-instruct"])
        assert len(result) == 2
        assert set(result["model_id"]) == {"deepseek-r1", "qwen2.5-72b-instruct"}


# ---------------------------------------------------------------------------
# Constants integrity
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are consistent with the data store."""

    def test_size_bucket_order_has_three_entries(self) -> None:
        assert len(SIZE_BUCKET_ORDER) == 3

    def test_size_bucket_values_match_data(self, full_df: pd.DataFrame) -> None:
        data_buckets = set(full_df["parameter_size_bucket"].dropna().unique())
        assert data_buckets.issubset(frozenset(SIZE_BUCKET_ORDER))

    def test_known_task_categories_non_empty(self) -> None:
        assert len(KNOWN_TASK_CATEGORIES) >= 3

    def test_known_licenses_non_empty(self) -> None:
        assert len(KNOWN_LICENSES) >= 3
