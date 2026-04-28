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
            "parameter_size_bucket": "\u22647B",
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
            "parameter_size_bucket": "8\u201334B",
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

    def test_single_category_math_returns_matching_rows(self, sample_df: pd.DataFrame) -> None:
        """Only models with 'math' in task_categories should be returned."""
        result = filter_by_task_category(sample_df, ["math"])
        # model-beta (math) and model-gamma (reasoning, math, coding)
        assert set(result["model_id"]) == {"model-beta", "model-gamma"}

    def test_single_category_coding_returns_matching_rows(self, sample_df: pd.DataFrame) -> None:
        """Only models with 'coding' should be returned."""
        result = filter_by_task_category(sample_df, ["coding"])
        assert set(result["model_id"]) == {"model-alpha", "model-gamma", "model-delta"}

    def test_single_category_reasoning_returns_matching_rows(self, sample_df: pd.DataFrame) -> None:
        """Only models with 'reasoning' should be returned."""
        result = filter_by_task_category(sample_df, ["reasoning"])
        assert set(result["model_id"]) == {"model-alpha", "model-gamma"}

    def test_multiple_categories_uses_or_logic(self, sample_df: pd.DataFrame) -> None:
        """A model matching ANY of the categories should be included."""
        result = filter_by_task_category(sample_df, ["math", "coding"])
        # All four models have at least one of math or coding
        assert set(result["model_id"]) == {
            "model-alpha", "model-beta", "model-gamma", "model-delta"
        }

    def test_all_three_categories_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """Requesting all three categories should match every model."""
        result = filter_by_task_category(sample_df, ["reasoning", "coding", "math"])
        assert len(result) == len(sample_df)

    def test_empty_categories_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """An empty filter list should be a no-op (return all rows)."""
        result = filter_by_task_category(sample_df, [])
        assert len(result) == len(sample_df)

    def test_nonexistent_category_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """A category not present in any model should yield an empty DataFrame."""
        result = filter_by_task_category(sample_df, ["nonexistent"])
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        """Mutating the result should not affect the original DataFrame."""
        result = filter_by_task_category(sample_df, ["coding"])
        result["_new_col"] = 1
        assert "_new_col" not in sample_df.columns

    def test_input_df_not_mutated(self, sample_df: pd.DataFrame) -> None:
        """The source DataFrame should have the same row count after filtering."""
        original_len = len(sample_df)
        filter_by_task_category(sample_df, ["coding"])
        assert len(sample_df) == original_len

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Passing a non-DataFrame should raise TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame"):
            filter_by_task_category("not a df", ["coding"])  # type: ignore[arg-type]

    def test_raises_type_error_for_list_input(self) -> None:
        """Passing a list instead of a DataFrame should raise TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame"):
            filter_by_task_category([], ["coding"])  # type: ignore[arg-type]

    def test_missing_column_returns_empty_dataframe(self, sample_df: pd.DataFrame) -> None:
        """A DataFrame without task_categories column should return empty."""
        df_no_col = sample_df.drop(columns=["task_categories"])
        result = filter_by_task_category(df_no_col, ["coding"])
        assert len(result) == 0

    def test_none_values_in_sequence_ignored(self, sample_df: pd.DataFrame) -> None:
        """None entries in the categories list should be silently skipped."""
        result = filter_by_task_category(sample_df, ["coding", None])  # type: ignore[list-item]
        assert len(result) > 0

    def test_whitespace_stripped_from_category_values(self, sample_df: pd.DataFrame) -> None:
        """Category strings with surrounding whitespace should still match."""
        result = filter_by_task_category(sample_df, ["  coding  "])
        assert len(result) > 0
        assert set(result["model_id"]) == {"model-alpha", "model-gamma", "model-delta"}

    def test_result_column_set_preserved(self, sample_df: pd.DataFrame) -> None:
        """Filtered DataFrame should have the same columns as the input."""
        result = filter_by_task_category(sample_df, ["coding"])
        assert set(result.columns) == set(sample_df.columns)

    def test_real_data_coding_filter_correct_categories(self, full_df: pd.DataFrame) -> None:
        """Every model in the result must have 'coding' in its task_categories."""
        result = filter_by_task_category(full_df, ["coding"])
        assert len(result) > 0
        for cats in result["task_categories"]:
            assert "coding" in cats

    def test_real_data_math_filter_correct_categories(self, full_df: pd.DataFrame) -> None:
        """Every model in the result must have 'math' in its task_categories."""
        result = filter_by_task_category(full_df, ["math"])
        assert len(result) > 0
        for cats in result["task_categories"]:
            assert "math" in cats

    def test_real_data_reasoning_filter_non_empty(self, full_df: pd.DataFrame) -> None:
        """The reasoning filter should return at least one model."""
        result = filter_by_task_category(full_df, ["reasoning"])
        assert len(result) > 0

    def test_real_data_multi_category_superset_of_single(self, full_df: pd.DataFrame) -> None:
        """Filtering by [coding, math] should return >= rows than coding alone."""
        result_coding = filter_by_task_category(full_df, ["coding"])
        result_multi = filter_by_task_category(full_df, ["coding", "math"])
        assert len(result_multi) >= len(result_coding)

    def test_real_data_all_categories_returns_full_df(self, full_df: pd.DataFrame) -> None:
        """Filtering by all known categories should return all rows."""
        result = filter_by_task_category(full_df, ["reasoning", "coding", "math"])
        assert len(result) == len(full_df)


# ---------------------------------------------------------------------------
# filter_by_size_bucket
# ---------------------------------------------------------------------------


class TestFilterBySizeBucket:
    """Tests for filter_by_size_bucket."""

    def test_small_bucket_returns_only_small_models(self, sample_df: pd.DataFrame) -> None:
        """The ≤7B bucket should return only model-alpha."""
        result = filter_by_size_bucket(sample_df, ["\u22647B"])
        assert list(result["model_id"]) == ["model-alpha"]

    def test_mid_bucket_returns_only_mid_models(self, sample_df: pd.DataFrame) -> None:
        """The 8–34B bucket should return only model-beta."""
        result = filter_by_size_bucket(sample_df, ["8\u201334B"])
        assert list(result["model_id"]) == ["model-beta"]

    def test_large_bucket_returns_large_models(self, sample_df: pd.DataFrame) -> None:
        """The 35B+ bucket should return model-gamma and model-delta."""
        result = filter_by_size_bucket(sample_df, ["35B+"])
        assert set(result["model_id"]) == {"model-gamma", "model-delta"}

    def test_multiple_buckets_or_logic(self, sample_df: pd.DataFrame) -> None:
        """Passing ≤7B and 35B+ should return their combined models."""
        result = filter_by_size_bucket(sample_df, ["\u22647B", "35B+"])
        assert set(result["model_id"]) == {"model-alpha", "model-gamma", "model-delta"}

    def test_all_buckets_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """Passing all three bucket labels should return every row."""
        result = filter_by_size_bucket(sample_df, ["\u22647B", "8\u201334B", "35B+"])
        assert len(result) == len(sample_df)

    def test_empty_buckets_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """An empty bucket list is a no-op."""
        result = filter_by_size_bucket(sample_df, [])
        assert len(result) == len(sample_df)

    def test_nonexistent_bucket_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """An unknown bucket label should return an empty DataFrame."""
        result = filter_by_size_bucket(sample_df, ["1000B+"])
        assert len(result) == 0

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        """Mutating the result should not affect the original DataFrame."""
        result = filter_by_size_bucket(sample_df, ["\u22647B"])
        result["_extra"] = 99
        assert "_extra" not in sample_df.columns

    def test_input_df_not_mutated(self, sample_df: pd.DataFrame) -> None:
        """Filtering must not modify the source DataFrame."""
        original_len = len(sample_df)
        filter_by_size_bucket(sample_df, ["35B+"])
        assert len(sample_df) == original_len

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame"):
            filter_by_size_bucket(42, ["\u22647B"])  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """A DataFrame without parameter_size_bucket should return empty."""
        df_no_col = sample_df.drop(columns=["parameter_size_bucket"])
        result = filter_by_size_bucket(df_no_col, ["\u22647B"])
        assert len(result) == 0

    def test_result_column_set_preserved(self, sample_df: pd.DataFrame) -> None:
        """Result columns should be identical to input columns."""
        result = filter_by_size_bucket(sample_df, ["35B+"])
        assert set(result.columns) == set(sample_df.columns)

    def test_none_values_in_sequence_ignored(self, sample_df: pd.DataFrame) -> None:
        """None entries in the bucket list should be silently skipped."""
        result = filter_by_size_bucket(sample_df, ["\u22647B", None])  # type: ignore[list-item]
        assert len(result) == 1
        assert result.iloc[0]["model_id"] == "model-alpha"

    def test_real_data_small_models_have_correct_bucket(self, full_df: pd.DataFrame) -> None:
        """All returned models must have parameter_size_bucket == ≤7B."""
        result = filter_by_size_bucket(full_df, ["\u22647B"])
        assert len(result) > 0
        assert all(result["parameter_size_bucket"] == "\u22647B")

    def test_real_data_large_models_have_correct_bucket(self, full_df: pd.DataFrame) -> None:
        """All returned models must have parameter_size_bucket == 35B+."""
        result = filter_by_size_bucket(full_df, ["35B+"])
        assert len(result) > 0
        assert all(result["parameter_size_bucket"] == "35B+")

    def test_real_data_mid_models_have_correct_bucket(self, full_df: pd.DataFrame) -> None:
        """All returned models must have parameter_size_bucket == 8–34B."""
        result = filter_by_size_bucket(full_df, ["8\u201334B"])
        assert len(result) > 0
        assert all(result["parameter_size_bucket"] == "8\u201334B")

    def test_real_data_combined_buckets_additive(self, full_df: pd.DataFrame) -> None:
        """Combining two non-overlapping buckets returns their union."""
        r_small = filter_by_size_bucket(full_df, ["\u22647B"])
        r_mid = filter_by_size_bucket(full_df, ["8\u201334B"])
        r_combined = filter_by_size_bucket(full_df, ["\u22647B", "8\u201334B"])
        assert len(r_combined) == len(r_small) + len(r_mid)

    def test_real_data_all_three_buckets_equals_full_df(self, full_df: pd.DataFrame) -> None:
        """Passing all three buckets should return the same number of rows as the full df."""
        result = filter_by_size_bucket(full_df, ["\u22647B", "8\u201334B", "35B+"])
        assert len(result) == len(full_df)


# ---------------------------------------------------------------------------
# filter_by_license
# ---------------------------------------------------------------------------


class TestFilterByLicense:
    """Tests for filter_by_license."""

    def test_apache_license_filter(self, sample_df: pd.DataFrame) -> None:
        """Apache-2.0 filter should return model-alpha and model-delta."""
        result = filter_by_license(sample_df, ["Apache-2.0"])
        assert set(result["model_id"]) == {"model-alpha", "model-delta"}

    def test_mit_license_filter(self, sample_df: pd.DataFrame) -> None:
        """MIT filter should return only model-beta."""
        result = filter_by_license(sample_df, ["MIT"])
        assert list(result["model_id"]) == ["model-beta"]

    def test_custom_license_filter(self, sample_df: pd.DataFrame) -> None:
        """custom/commercial filter should return only model-gamma."""
        result = filter_by_license(sample_df, ["custom/commercial"])
        assert list(result["model_id"]) == ["model-gamma"]

    def test_multiple_licenses_or_logic(self, sample_df: pd.DataFrame) -> None:
        """Apache-2.0 OR MIT should return alpha, beta, and delta."""
        result = filter_by_license(sample_df, ["Apache-2.0", "MIT"])
        assert set(result["model_id"]) == {"model-alpha", "model-beta", "model-delta"}

    def test_all_licenses_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """Passing all three known licenses should return every model."""
        result = filter_by_license(sample_df, ["Apache-2.0", "MIT", "custom/commercial"])
        assert len(result) == len(sample_df)

    def test_empty_licenses_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """An empty license list is a no-op."""
        result = filter_by_license(sample_df, [])
        assert len(result) == len(sample_df)

    def test_nonexistent_license_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """An unknown license should yield an empty DataFrame."""
        result = filter_by_license(sample_df, ["GPL-3.0"])
        assert len(result) == 0

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        """Mutating the result must not affect the original DataFrame."""
        result = filter_by_license(sample_df, ["MIT"])
        result["_extra"] = 1
        assert "_extra" not in sample_df.columns

    def test_input_df_not_mutated(self, sample_df: pd.DataFrame) -> None:
        """Filtering must not modify the source DataFrame."""
        original_len = len(sample_df)
        filter_by_license(sample_df, ["MIT"])
        assert len(sample_df) == original_len

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame"):
            filter_by_license(None, ["MIT"])  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """A DataFrame without a license column should return empty."""
        df_no_col = sample_df.drop(columns=["license"])
        result = filter_by_license(df_no_col, ["MIT"])
        assert len(result) == 0

    def test_none_values_in_sequence_ignored(self, sample_df: pd.DataFrame) -> None:
        """None entries in the license list should be silently skipped."""
        result = filter_by_license(sample_df, ["MIT", None])  # type: ignore[list-item]
        assert len(result) == 1
        assert result.iloc[0]["model_id"] == "model-beta"

    def test_result_column_set_preserved(self, sample_df: pd.DataFrame) -> None:
        """Result columns should be identical to input columns."""
        result = filter_by_license(sample_df, ["Apache-2.0"])
        assert set(result.columns) == set(sample_df.columns)

    def test_real_data_apache_filter_all_correct(self, full_df: pd.DataFrame) -> None:
        """Every returned model must have license == Apache-2.0."""
        result = filter_by_license(full_df, ["Apache-2.0"])
        assert len(result) > 0
        assert all(result["license"] == "Apache-2.0")

    def test_real_data_mit_filter_all_correct(self, full_df: pd.DataFrame) -> None:
        """Every returned model must have license == MIT."""
        result = filter_by_license(full_df, ["MIT"])
        assert len(result) > 0
        assert all(result["license"] == "MIT")

    def test_real_data_custom_commercial_filter(self, full_df: pd.DataFrame) -> None:
        """Every returned model must have license == custom/commercial."""
        result = filter_by_license(full_df, ["custom/commercial"])
        assert len(result) > 0
        assert all(result["license"] == "custom/commercial")

    def test_real_data_open_licenses_combined(self, full_df: pd.DataFrame) -> None:
        """Apache-2.0 OR MIT should return only models with those licenses."""
        result = filter_by_license(full_df, ["Apache-2.0", "MIT"])
        assert len(result) > 0
        assert all(result["license"].isin(["Apache-2.0", "MIT"]))

    def test_real_data_all_licenses_equals_full_df(self, full_df: pd.DataFrame) -> None:
        """Passing all three license values should return the full DataFrame."""
        result = filter_by_license(full_df, ["Apache-2.0", "MIT", "custom/commercial"])
        assert len(result) == len(full_df)


# ---------------------------------------------------------------------------
# filter_by_family
# ---------------------------------------------------------------------------


class TestFilterByFamily:
    """Tests for filter_by_family."""

    def test_single_family_alpha(self, sample_df: pd.DataFrame) -> None:
        """Filtering by Alpha family should return alpha and delta."""
        result = filter_by_family(sample_df, ["Alpha"])
        assert set(result["model_id"]) == {"model-alpha", "model-delta"}

    def test_single_family_beta(self, sample_df: pd.DataFrame) -> None:
        """Filtering by Beta family should return only model-beta."""
        result = filter_by_family(sample_df, ["Beta"])
        assert set(result["model_id"]) == {"model-beta"}

    def test_single_family_gamma(self, sample_df: pd.DataFrame) -> None:
        """Filtering by Gamma family should return only model-gamma."""
        result = filter_by_family(sample_df, ["Gamma"])
        assert set(result["model_id"]) == {"model-gamma"}

    def test_multiple_families_or_logic(self, sample_df: pd.DataFrame) -> None:
        """Alpha OR Beta should return alpha, beta, and delta."""
        result = filter_by_family(sample_df, ["Alpha", "Beta"])
        assert set(result["model_id"]) == {"model-alpha", "model-beta", "model-delta"}

    def test_all_families_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """Passing all families should return every row."""
        result = filter_by_family(sample_df, ["Alpha", "Beta", "Gamma"])
        assert len(result) == len(sample_df)

    def test_empty_families_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """An empty family list is a no-op."""
        result = filter_by_family(sample_df, [])
        assert len(result) == len(sample_df)

    def test_nonexistent_family_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """An unknown family name should yield an empty DataFrame."""
        result = filter_by_family(sample_df, ["NoSuchFamily"])
        assert len(result) == 0

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        """Mutating the result should not affect the original DataFrame."""
        result = filter_by_family(sample_df, ["Alpha"])
        result["_extra"] = 99
        assert "_extra" not in sample_df.columns

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError):
            filter_by_family([], ["Alpha"])  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """A DataFrame without a family column should return empty."""
        df_no_col = sample_df.drop(columns=["family"])
        result = filter_by_family(df_no_col, ["Alpha"])
        assert len(result) == 0

    def test_none_values_in_sequence_ignored(self, sample_df: pd.DataFrame) -> None:
        """None entries in the families list should be silently skipped."""
        result = filter_by_family(sample_df, ["Beta", None])  # type: ignore[list-item]
        assert len(result) == 1
        assert result.iloc[0]["model_id"] == "model-beta"

    def test_result_column_set_preserved(self, sample_df: pd.DataFrame) -> None:
        """Result columns should match input columns."""
        result = filter_by_family(sample_df, ["Gamma"])
        assert set(result.columns) == set(sample_df.columns)

    def test_real_data_deepseek_family(self, full_df: pd.DataFrame) -> None:
        """DeepSeek family filter should return only DeepSeek models."""
        result = filter_by_family(full_df, ["DeepSeek"])
        assert len(result) >= 3  # V2, V3, R1
        assert all(result["family"] == "DeepSeek")

    def test_real_data_llama_family(self, full_df: pd.DataFrame) -> None:
        """Llama family filter should return only Llama models."""
        result = filter_by_family(full_df, ["Llama"])
        assert len(result) > 0
        assert all(result["family"] == "Llama")

    def test_real_data_mistral_family(self, full_df: pd.DataFrame) -> None:
        """Mistral family filter should return only Mistral models."""
        result = filter_by_family(full_df, ["Mistral"])
        assert len(result) > 0
        assert all(result["family"] == "Mistral")

    def test_real_data_qwen_family(self, full_df: pd.DataFrame) -> None:
        """Qwen family filter should return only Qwen models."""
        result = filter_by_family(full_df, ["Qwen"])
        assert len(result) > 0
        assert all(result["family"] == "Qwen")

    def test_real_data_llama_and_mistral_combined(self, full_df: pd.DataFrame) -> None:
        """Combining Llama and Mistral should return only those families."""
        result = filter_by_family(full_df, ["Llama", "Mistral"])
        assert len(result) > 0
        assert all(result["family"].isin(["Llama", "Mistral"]))

    def test_real_data_unknown_family_returns_empty(self, full_df: pd.DataFrame) -> None:
        """A family name not in the data should return an empty DataFrame."""
        result = filter_by_family(full_df, ["UnknownFamily9000"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_by_open_weights
# ---------------------------------------------------------------------------


class TestFilterByOpenWeights:
    """Tests for filter_by_open_weights."""

    def test_open_weights_only_excludes_closed(self, sample_df: pd.DataFrame) -> None:
        """model-gamma (open_weights=False) should be excluded."""
        result = filter_by_open_weights(sample_df, open_weights_only=True)
        assert "model-gamma" not in set(result["model_id"])

    def test_open_weights_only_includes_open(self, sample_df: pd.DataFrame) -> None:
        """Only models with open_weights=True should remain."""
        result = filter_by_open_weights(sample_df, open_weights_only=True)
        assert set(result["model_id"]) == {"model-alpha", "model-beta", "model-delta"}

    def test_open_weights_true_all_returned_are_open(self, sample_df: pd.DataFrame) -> None:
        """Every returned model must have open_weights == True."""
        result = filter_by_open_weights(sample_df, open_weights_only=True)
        assert all(result["open_weights"] == True)  # noqa: E712

    def test_false_flag_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """open_weights_only=False should be a no-op."""
        result = filter_by_open_weights(sample_df, open_weights_only=False)
        assert len(result) == len(sample_df)

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        """Mutating the result must not affect the original DataFrame."""
        result = filter_by_open_weights(sample_df, open_weights_only=True)
        result["_extra"] = 42
        assert "_extra" not in sample_df.columns

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError):
            filter_by_open_weights("bad", open_weights_only=True)  # type: ignore[arg-type]

    def test_missing_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """A DataFrame without open_weights column should return empty."""
        df_no_col = sample_df.drop(columns=["open_weights"])
        result = filter_by_open_weights(df_no_col, open_weights_only=True)
        assert len(result) == 0

    def test_result_column_set_preserved(self, sample_df: pd.DataFrame) -> None:
        """Result columns should match input columns."""
        result = filter_by_open_weights(sample_df, open_weights_only=True)
        assert set(result.columns) == set(sample_df.columns)

    def test_real_data_open_weights_true(self, full_df: pd.DataFrame) -> None:
        """All returned models from the real data must have open_weights=True."""
        result = filter_by_open_weights(full_df, open_weights_only=True)
        assert len(result) > 0
        assert all(result["open_weights"] == True)  # noqa: E712

    def test_real_data_open_weights_false_returns_all(self, full_df: pd.DataFrame) -> None:
        """open_weights_only=False should return the same row count as the full df."""
        result = filter_by_open_weights(full_df, open_weights_only=False)
        assert len(result) == len(full_df)

    def test_real_data_open_weights_subset_of_full(self, full_df: pd.DataFrame) -> None:
        """Open-weights-only result should be <= full_df row count."""
        result = filter_by_open_weights(full_df, open_weights_only=True)
        assert len(result) <= len(full_df)


# ---------------------------------------------------------------------------
# sort_by_benchmark
# ---------------------------------------------------------------------------


class TestSortByBenchmark:
    """Tests for sort_by_benchmark."""

    def test_default_sort_descending_mmlu(self, sample_df: pd.DataFrame) -> None:
        """Default sort should be descending (highest first)."""
        result = sort_by_benchmark(sample_df, "mmlu")
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)

    def test_ascending_sort_mmlu(self, sample_df: pd.DataFrame) -> None:
        """ascending=True should produce lowest-first ordering."""
        result = sort_by_benchmark(sample_df, "mmlu", ascending=True)
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores)

    def test_nan_values_placed_last_descending(self, sample_df: pd.DataFrame) -> None:
        """Rows with NaN in the sort column should appear last (descending)."""
        result = sort_by_benchmark(sample_df, "humaneval")
        # model-beta has NaN humaneval → should be last
        last_row = result.iloc[-1]
        assert pd.isna(last_row["benchmark_humaneval"])

    def test_nan_values_placed_last_ascending(self, sample_df: pd.DataFrame) -> None:
        """Rows with NaN should appear last even when sorting ascending."""
        result = sort_by_benchmark(sample_df, "math", ascending=True)
        # model-delta has NaN math
        last_row = result.iloc[-1]
        assert pd.isna(last_row["benchmark_math"])

    def test_index_is_reset(self, sample_df: pd.DataFrame) -> None:
        """After sorting, the index should be 0..n-1."""
        result = sort_by_benchmark(sample_df, "mmlu")
        assert list(result.index) == list(range(len(result)))

    def test_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """The sorted DataFrame should have the same number of rows as the input."""
        result = sort_by_benchmark(sample_df, "gsm8k")
        assert len(result) == len(sample_df)

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        """Mutating the result should not affect the original DataFrame."""
        result = sort_by_benchmark(sample_df, "mmlu")
        result["_extra"] = 0
        assert "_extra" not in sample_df.columns

    def test_raises_value_error_for_unknown_benchmark(self, sample_df: pd.DataFrame) -> None:
        """An unrecognised benchmark name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            sort_by_benchmark(sample_df, "nonexistent")

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError):
            sort_by_benchmark({}, "mmlu")  # type: ignore[arg-type]

    def test_all_benchmark_keys_accepted(self, sample_df: pd.DataFrame) -> None:
        """Every standard benchmark key should be accepted without error."""
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            result = sort_by_benchmark(sample_df, key)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(sample_df)

    def test_column_set_preserved(self, sample_df: pd.DataFrame) -> None:
        """Sorting should not add or remove columns."""
        result = sort_by_benchmark(sample_df, "mmlu")
        assert set(result.columns) == set(sample_df.columns)

    def test_real_data_sort_by_mmlu_descending(self, full_df: pd.DataFrame) -> None:
        """Real data sorted by mmlu descending should have non-increasing scores."""
        result = sort_by_benchmark(full_df, "mmlu")
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)

    def test_real_data_sort_by_humaneval_ascending(self, full_df: pd.DataFrame) -> None:
        """Real data sorted ascending by humaneval should have non-decreasing scores."""
        result = sort_by_benchmark(full_df, "humaneval", ascending=True)
        scores = result["benchmark_humaneval"].dropna().tolist()
        assert scores == sorted(scores)

    def test_real_data_sort_by_math_descending(self, full_df: pd.DataFrame) -> None:
        """Real data sorted by math descending should have non-increasing scores."""
        result = sort_by_benchmark(full_df, "math")
        scores = result["benchmark_math"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# apply_filters (composite)
# ---------------------------------------------------------------------------


class TestApplyFilters:
    """Tests for the composite apply_filters function."""

    def test_no_filters_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """Calling apply_filters with no arguments should be a no-op."""
        result = apply_filters(sample_df)
        assert len(result) == len(sample_df)

    def test_all_none_filters_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """Passing None for each filter dimension should be a no-op."""
        result = apply_filters(
            sample_df,
            task_categories=None,
            size_buckets=None,
            licenses=None,
            families=None,
        )
        assert len(result) == len(sample_df)

    def test_all_empty_filters_returns_all_rows(self, sample_df: pd.DataFrame) -> None:
        """Passing empty lists for each filter dimension should be a no-op."""
        result = apply_filters(
            sample_df,
            task_categories=[],
            size_buckets=[],
            licenses=[],
            families=[],
        )
        assert len(result) == len(sample_df)

    def test_single_task_category_filter(self, sample_df: pd.DataFrame) -> None:
        """Filtering by 'math' should return beta and gamma."""
        result = apply_filters(sample_df, task_categories=["math"])
        assert set(result["model_id"]) == {"model-beta", "model-gamma"}

    def test_single_size_bucket_filter(self, sample_df: pd.DataFrame) -> None:
        """Filtering by ≤7B should return only alpha."""
        result = apply_filters(sample_df, size_buckets=["\u22647B"])
        assert list(result["model_id"]) == ["model-alpha"]

    def test_single_license_filter(self, sample_df: pd.DataFrame) -> None:
        """Filtering by MIT should return only beta."""
        result = apply_filters(sample_df, licenses=["MIT"])
        assert list(result["model_id"]) == ["model-beta"]

    def test_single_family_filter(self, sample_df: pd.DataFrame) -> None:
        """Filtering by Alpha family should return alpha and delta."""
        result = apply_filters(sample_df, families=["Alpha"])
        assert set(result["model_id"]) == {"model-alpha", "model-delta"}

    def test_combined_category_and_bucket_filters(self, sample_df: pd.DataFrame) -> None:
        """coding AND ≤7B or 8-34B should return only alpha (coding, ≤7B)."""
        result = apply_filters(
            sample_df,
            task_categories=["coding"],
            size_buckets=["\u22647B", "8\u201334B"],
        )
        # model-alpha: coding + ≤7B ✓
        # model-beta: math only (no coding) → excluded by category
        # model-gamma: coding + 35B+ → excluded by bucket
        # model-delta: coding + 35B+ → excluded by bucket
        assert set(result["model_id"]) == {"model-alpha"}

    def test_combined_bucket_and_license_filters(self, sample_df: pd.DataFrame) -> None:
        """35B+ AND Apache-2.0 should return only delta."""
        result = apply_filters(
            sample_df,
            size_buckets=["35B+"],
            licenses=["Apache-2.0"],
        )
        # model-gamma is 35B+ but custom/commercial → excluded
        # model-delta is 35B+ and Apache-2.0 ✓
        assert set(result["model_id"]) == {"model-delta"}

    def test_combined_category_bucket_license_filters(self, sample_df: pd.DataFrame) -> None:
        """coding AND ≤7B AND Apache-2.0 should return only alpha."""
        result = apply_filters(
            sample_df,
            task_categories=["coding"],
            size_buckets=["\u22647B"],
            licenses=["Apache-2.0"],
        )
        assert list(result["model_id"]) == ["model-alpha"]

    def test_family_and_bucket_filter(self, sample_df: pd.DataFrame) -> None:
        """Alpha family AND 35B+ bucket should return only delta."""
        result = apply_filters(
            sample_df,
            families=["Alpha"],
            size_buckets=["35B+"],
        )
        assert set(result["model_id"]) == {"model-delta"}

    def test_conflicting_filters_return_empty(self, sample_df: pd.DataFrame) -> None:
        """Mutually exclusive constraints should return an empty DataFrame."""
        # No model is both ≤7B (only alpha) and MIT (only beta)
        result = apply_filters(
            sample_df,
            size_buckets=["\u22647B"],
            licenses=["MIT"],
        )
        assert len(result) == 0

    def test_open_weights_filter_applied(self, sample_df: pd.DataFrame) -> None:
        """open_weights_only=True should exclude model-gamma."""
        result = apply_filters(sample_df, open_weights_only=True)
        assert "model-gamma" not in set(result["model_id"])
        assert len(result) == 3

    def test_open_weights_false_no_effect(self, sample_df: pd.DataFrame) -> None:
        """open_weights_only=False should return all rows."""
        result = apply_filters(sample_df, open_weights_only=False)
        assert len(result) == len(sample_df)

    def test_sort_benchmark_applied_descending(self, sample_df: pd.DataFrame) -> None:
        """sort_benchmark should produce descending order by default."""
        result = apply_filters(sample_df, sort_benchmark="mmlu")
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)

    def test_sort_benchmark_ascending(self, sample_df: pd.DataFrame) -> None:
        """sort_ascending=True should produce ascending order."""
        result = apply_filters(sample_df, sort_benchmark="mmlu", sort_ascending=True)
        scores = result["benchmark_mmlu"].dropna().tolist()
        assert scores == sorted(scores)

    def test_sort_with_filter_combined(self, sample_df: pd.DataFrame) -> None:
        """Filtering then sorting should apply both correctly."""
        result = apply_filters(
            sample_df,
            size_buckets=["35B+"],
            sort_benchmark="humaneval",
        )
        assert len(result) == 2  # gamma and delta
        scores = result["benchmark_humaneval"].dropna().tolist()
        assert scores == sorted(scores, reverse=True)

    def test_index_always_reset(self, sample_df: pd.DataFrame) -> None:
        """The result index should always be 0..n-1."""
        result = apply_filters(sample_df, size_buckets=["\u22647B"])
        assert list(result.index) == list(range(len(result)))

    def test_index_reset_when_no_sort(self, sample_df: pd.DataFrame) -> None:
        """Even without sorting, the index should be reset."""
        result = apply_filters(sample_df, licenses=["MIT"])
        assert list(result.index) == list(range(len(result)))

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        """Mutating the result must not affect the original DataFrame."""
        result = apply_filters(sample_df)
        result["_extra"] = 99
        assert "_extra" not in sample_df.columns

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError):
            apply_filters("not a df")  # type: ignore[arg-type]

    def test_raises_value_error_for_invalid_sort_benchmark(self, sample_df: pd.DataFrame) -> None:
        """An unrecognised sort benchmark should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown benchmark"):
            apply_filters(sample_df, sort_benchmark="invalid_key")

    def test_empty_input_df_returns_empty(self) -> None:
        """An empty input DataFrame should produce an empty result."""
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
            size_buckets=["\u22647B"],
            licenses=["MIT"],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_real_data_no_filters_returns_full(self, full_df: pd.DataFrame) -> None:
        """No filters on real data should return all rows."""
        result = apply_filters(full_df)
        assert len(result) == len(full_df)

    def test_real_data_coding_apache(self, full_df: pd.DataFrame) -> None:
        """coding AND Apache-2.0 should return valid subset."""
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
        """math AND ≤7B should return valid (possibly empty) subset."""
        result = apply_filters(
            full_df,
            task_categories=["math"],
            size_buckets=["\u22647B"],
        )
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert all(result["parameter_size_bucket"] == "\u22647B")
            for cats in result["task_categories"]:
                assert "math" in cats

    def test_real_data_sort_by_humaneval(self, full_df: pd.DataFrame) -> None:
        """Sorting real data by humaneval descending should produce correct order."""
        result = apply_filters(full_df, sort_benchmark="humaneval")
        valid_scores = result["benchmark_humaneval"].dropna().tolist()
        assert valid_scores == sorted(valid_scores, reverse=True)

    def test_real_data_sort_by_gsm8k_ascending(self, full_df: pd.DataFrame) -> None:
        """Sorting real data by gsm8k ascending should produce correct order."""
        result = apply_filters(full_df, sort_benchmark="gsm8k", sort_ascending=True)
        valid_scores = result["benchmark_gsm8k"].dropna().tolist()
        assert valid_scores == sorted(valid_scores)

    def test_real_data_deepseek_family_filter(self, full_df: pd.DataFrame) -> None:
        """Filtering real data by DeepSeek family should return only DeepSeek models."""
        result = apply_filters(full_df, families=["DeepSeek"])
        assert len(result) >= 3
        assert all(result["family"] == "DeepSeek")

    def test_real_data_open_weights_only(self, full_df: pd.DataFrame) -> None:
        """open_weights_only=True on real data should return only open-weight models."""
        result = apply_filters(full_df, open_weights_only=True)
        assert len(result) > 0
        assert all(result["open_weights"] == True)  # noqa: E712

    def test_filter_order_does_not_affect_result(self, sample_df: pd.DataFrame) -> None:
        """The apply_filters result should be equivalent regardless of which filter is named."""
        result_a = apply_filters(
            sample_df,
            task_categories=["coding"],
            licenses=["Apache-2.0"],
        )
        result_b = apply_filters(
            sample_df,
            licenses=["Apache-2.0"],
            task_categories=["coding"],
        )
        assert set(result_a["model_id"]) == set(result_b["model_id"])

    def test_all_filters_combined_narrows_results(self, sample_df: pd.DataFrame) -> None:
        """Applying all filters simultaneously should narrow results correctly."""
        result = apply_filters(
            sample_df,
            task_categories=["coding"],
            size_buckets=["\u22647B"],
            licenses=["Apache-2.0"],
            families=["Alpha"],
            open_weights_only=True,
        )
        # Only model-alpha satisfies all conditions
        assert list(result["model_id"]) == ["model-alpha"]


# ---------------------------------------------------------------------------
# get_model_by_id
# ---------------------------------------------------------------------------


class TestGetModelById:
    """Tests for get_model_by_id."""

    def test_returns_series_for_known_id(self, sample_df: pd.DataFrame) -> None:
        """A known model_id should return a pandas Series."""
        row = get_model_by_id(sample_df, "model-alpha")
        assert row is not None
        assert isinstance(row, pd.Series)

    def test_returned_series_has_correct_model_id(self, sample_df: pd.DataFrame) -> None:
        """The returned Series should have the requested model_id."""
        row = get_model_by_id(sample_df, "model-alpha")
        assert row is not None
        assert row["model_id"] == "model-alpha"

    def test_returned_series_has_correct_values(self, sample_df: pd.DataFrame) -> None:
        """The returned Series should carry correct field values."""
        row = get_model_by_id(sample_df, "model-gamma")
        assert row is not None
        assert row["family"] == "Gamma"
        assert row["license"] == "custom/commercial"
        assert row["benchmark_mmlu"] == pytest.approx(90.0)

    def test_returns_none_for_unknown_id(self, sample_df: pd.DataFrame) -> None:
        """An unknown model_id should return None."""
        row = get_model_by_id(sample_df, "nonexistent-model")
        assert row is None

    def test_returns_none_for_empty_string_id(self, sample_df: pd.DataFrame) -> None:
        """An empty string model_id should return None."""
        row = get_model_by_id(sample_df, "")
        assert row is None

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError):
            get_model_by_id("bad", "model-alpha")  # type: ignore[arg-type]

    def test_missing_model_id_column_returns_none(self, sample_df: pd.DataFrame) -> None:
        """A DataFrame without a model_id column should return None."""
        df_no_col = sample_df.drop(columns=["model_id"])
        result = get_model_by_id(df_no_col, "model-alpha")
        assert result is None

    def test_all_four_sample_models_found(self, sample_df: pd.DataFrame) -> None:
        """Every model in the sample DataFrame should be retrievable."""
        for mid in ("model-alpha", "model-beta", "model-gamma", "model-delta"):
            row = get_model_by_id(sample_df, mid)
            assert row is not None, f"Expected to find model '{mid}'"

    def test_real_data_known_model_deepseek_r1(self, full_df: pd.DataFrame) -> None:
        """deepseek-r1 should be retrievable from the real DataFrame."""
        row = get_model_by_id(full_df, "deepseek-r1")
        assert row is not None
        assert row["display_name"] == "DeepSeek R1"

    def test_real_data_known_model_qwen_72b(self, full_df: pd.DataFrame) -> None:
        """qwen2.5-72b-instruct should be retrievable from the real DataFrame."""
        row = get_model_by_id(full_df, "qwen2.5-72b-instruct")
        assert row is not None
        assert row["family"] == "Qwen"

    def test_real_data_unknown_model_returns_none(self, full_df: pd.DataFrame) -> None:
        """A model not in the data should return None."""
        row = get_model_by_id(full_df, "gpt-5-turbo-ultra")
        assert row is None


# ---------------------------------------------------------------------------
# get_models_by_ids
# ---------------------------------------------------------------------------


class TestGetModelsByIds:
    """Tests for get_models_by_ids."""

    def test_returns_matching_rows(self, sample_df: pd.DataFrame) -> None:
        """Passing two known IDs should return a DataFrame with two rows."""
        result = get_models_by_ids(sample_df, ["model-alpha", "model-gamma"])
        assert set(result["model_id"]) == {"model-alpha", "model-gamma"}

    def test_empty_ids_returns_empty_df(self, sample_df: pd.DataFrame) -> None:
        """An empty IDs list should return an empty DataFrame."""
        result = get_models_by_ids(sample_df, [])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == list(sample_df.columns)

    def test_unknown_ids_return_empty(self, sample_df: pd.DataFrame) -> None:
        """IDs not present in the DataFrame should yield an empty result."""
        result = get_models_by_ids(sample_df, ["does-not-exist"])
        assert len(result) == 0

    def test_partial_match_returns_known_only(self, sample_df: pd.DataFrame) -> None:
        """Known IDs should appear in the result even when mixed with unknown IDs."""
        result = get_models_by_ids(sample_df, ["model-alpha", "does-not-exist"])
        assert set(result["model_id"]) == {"model-alpha"}

    def test_duplicate_ids_not_duplicated_in_result(self, sample_df: pd.DataFrame) -> None:
        """Duplicate IDs in the input should not produce duplicate rows."""
        result = get_models_by_ids(sample_df, ["model-alpha", "model-alpha"])
        assert len(result) == 1

    def test_all_ids_returns_all_matching_rows(self, sample_df: pd.DataFrame) -> None:
        """Passing all four IDs should return all four rows."""
        all_ids = list(sample_df["model_id"])
        result = get_models_by_ids(sample_df, all_ids)
        assert len(result) == len(sample_df)

    def test_returns_copy_not_view(self, sample_df: pd.DataFrame) -> None:
        """Mutating the result must not affect the original DataFrame."""
        result = get_models_by_ids(sample_df, ["model-alpha"])
        result["_extra"] = 1
        assert "_extra" not in sample_df.columns

    def test_raises_type_error_for_non_dataframe(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        with pytest.raises(TypeError):
            get_models_by_ids(None, ["model-alpha"])  # type: ignore[arg-type]

    def test_missing_model_id_column_returns_empty(self, sample_df: pd.DataFrame) -> None:
        """A DataFrame without a model_id column should return empty."""
        df_no_col = sample_df.drop(columns=["model_id"])
        result = get_models_by_ids(df_no_col, ["model-alpha"])
        assert len(result) == 0

    def test_result_columns_match_input(self, sample_df: pd.DataFrame) -> None:
        """Result columns should be identical to the input DataFrame's columns."""
        result = get_models_by_ids(sample_df, ["model-alpha", "model-beta"])
        assert set(result.columns) == set(sample_df.columns)

    def test_single_id_returns_single_row(self, sample_df: pd.DataFrame) -> None:
        """Passing one known ID should return a DataFrame with exactly one row."""
        result = get_models_by_ids(sample_df, ["model-beta"])
        assert len(result) == 1
        assert result.iloc[0]["model_id"] == "model-beta"

    def test_none_in_ids_ignored(self, sample_df: pd.DataFrame) -> None:
        """None entries in the IDs list should be silently ignored."""
        result = get_models_by_ids(sample_df, ["model-alpha", None])  # type: ignore[list-item]
        assert set(result["model_id"]) == {"model-alpha"}

    def test_real_data_select_two_models(self, full_df: pd.DataFrame) -> None:
        """Selecting two known models from real data should return exactly two rows."""
        result = get_models_by_ids(full_df, ["deepseek-r1", "qwen2.5-72b-instruct"])
        assert len(result) == 2
        assert set(result["model_id"]) == {"deepseek-r1", "qwen2.5-72b-instruct"}

    def test_real_data_select_four_models(self, full_df: pd.DataFrame) -> None:
        """Selecting four known models from real data should return exactly four rows."""
        ids = [
            "deepseek-r1",
            "deepseek-v3",
            "llama-3.3-70b-instruct",
            "qwen2.5-72b-instruct",
        ]
        result = get_models_by_ids(full_df, ids)
        assert len(result) == 4
        assert set(result["model_id"]) == set(ids)

    def test_real_data_all_unknown_returns_empty(self, full_df: pd.DataFrame) -> None:
        """All-unknown IDs should return an empty DataFrame."""
        result = get_models_by_ids(full_df, ["fake-1", "fake-2"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Constants integrity
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are consistent with the data store."""

    def test_size_bucket_order_has_exactly_three_entries(self) -> None:
        """SIZE_BUCKET_ORDER must contain exactly three size buckets."""
        assert len(SIZE_BUCKET_ORDER) == 3

    def test_size_bucket_order_contains_expected_labels(self) -> None:
        """SIZE_BUCKET_ORDER must contain the three standard labels."""
        assert "\u22647B" in SIZE_BUCKET_ORDER
        assert "8\u201334B" in SIZE_BUCKET_ORDER
        assert "35B+" in SIZE_BUCKET_ORDER

    def test_size_bucket_values_match_data(self, full_df: pd.DataFrame) -> None:
        """All bucket values in the data should be a subset of SIZE_BUCKET_ORDER."""
        data_buckets = set(full_df["parameter_size_bucket"].dropna().unique())
        assert data_buckets.issubset(frozenset(SIZE_BUCKET_ORDER))

    def test_known_task_categories_has_at_least_three(self) -> None:
        """KNOWN_TASK_CATEGORIES must have at least three entries."""
        assert len(KNOWN_TASK_CATEGORIES) >= 3

    def test_known_task_categories_contains_standard_values(self) -> None:
        """All three standard categories must appear in KNOWN_TASK_CATEGORIES."""
        for cat in ("reasoning", "coding", "math"):
            assert cat in KNOWN_TASK_CATEGORIES

    def test_known_licenses_has_at_least_three(self) -> None:
        """KNOWN_LICENSES must have at least three entries."""
        assert len(KNOWN_LICENSES) >= 3

    def test_known_licenses_contains_standard_values(self) -> None:
        """All three standard licenses must appear in KNOWN_LICENSES."""
        for lic in ("Apache-2.0", "MIT", "custom/commercial"):
            assert lic in KNOWN_LICENSES

    def test_all_data_licenses_in_known_licenses(self, full_df: pd.DataFrame) -> None:
        """Every license value in the real data should appear in KNOWN_LICENSES."""
        data_licenses = set(full_df["license"].dropna().unique())
        # Log any unexpected licenses but do not fail — new licenses may be added
        unexpected = data_licenses - KNOWN_LICENSES
        # All real licenses in the bundled data must be known
        assert not unexpected, f"Unexpected license values in data: {unexpected}"
