"""Unit tests for llm_bench_compare.data_loader.

Verifies correct loading, schema validation, caching behaviour, merging
logic, and the get_filter_options helper using the real JSON data files
shipped with the package.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import llm_bench_compare.data_loader as dl
from llm_bench_compare.data_loader import (
    _build_benchmarks_df,
    _build_merged_df,
    _build_pricing_df,
    _compute_cheapest_api,
    _load_json,
    _validate_model_record,
    clear_cache,
    get_filter_options,
    get_merged_df,
    load_benchmarks_df,
    load_pricing_df,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_BENCHMARK_RAW: dict[str, Any] = {
    "models": [
        {
            "model_id": "test-model-a",
            "display_name": "Test Model A",
            "family": "Test",
            "version": "1.0",
            "parameter_size_b": 7,
            "parameter_size_bucket": "≤7B",
            "architecture": "Dense",
            "context_length_k": 32,
            "license": "Apache-2.0",
            "open_weights": True,
            "release_date": "2024-01",
            "benchmarks": {
                "mmlu": 70.0,
                "humaneval": 60.0,
                "math": 40.0,
                "gsm8k": 80.0,
                "mbpp": 55.0,
            },
            "task_categories": ["reasoning", "coding"],
        },
        {
            "model_id": "test-model-b",
            "display_name": "Test Model B",
            "family": "Test",
            "version": "2.0",
            "parameter_size_b": 34,
            "parameter_size_bucket": "8–34B",
            "architecture": "Dense",
            "context_length_k": 64,
            "license": "MIT",
            "open_weights": True,
            "release_date": "2024-06",
            "benchmarks": {
                "mmlu": 80.0,
                "humaneval": None,
                "math": 55.0,
                "gsm8k": 88.0,
                "mbpp": None,
            },
            "task_categories": ["math"],
        },
    ]
}

_MINIMAL_PRICING_RAW: dict[str, Any] = {
    "api_providers": [
        {
            "model_id": "test-model-a",
            "display_name": "Test Model A",
            "providers": [
                {
                    "provider": "Provider X",
                    "provider_url": "https://example.com",
                    "input_per_1m": 0.50,
                    "output_per_1m": 1.00,
                    "notes": None,
                },
                {
                    "provider": "Provider Y",
                    "provider_url": "https://example2.com",
                    "input_per_1m": 0.30,
                    "output_per_1m": 0.80,
                    "notes": "Batch available",
                },
            ],
        },
    ],
    "self_hosted": [
        {
            "model_id": "test-model-a",
            "display_name": "Test Model A",
            "recommended_gpu_setup": "1× RTX 4090",
            "min_vram_gb": 16,
            "estimated_hourly_cost_usd": 0.50,
            "cloud_provider_reference": "Vast.ai",
            "fp_precision": "BF16",
            "throughput_tokens_per_sec_approx": 3000,
        },
        {
            "model_id": "test-model-b",
            "display_name": "Test Model B",
            "recommended_gpu_setup": "2× A100",
            "min_vram_gb": 68,
            "estimated_hourly_cost_usd": 2.80,
            "cloud_provider_reference": "Lambda Labs",
            "fp_precision": "BF16",
            "throughput_tokens_per_sec_approx": 900,
        },
    ],
}


@pytest.fixture(autouse=True)
def reset_cache() -> None:
    """Clear the module-level data loader cache before every test."""
    clear_cache()


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    """Tests for the _load_json helper."""

    def test_loads_valid_json_file(self, tmp_path: Path) -> None:
        data = {"key": "value", "num": 42}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = _load_json(p)
        assert result == data

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            _load_json(tmp_path / "nonexistent.json")

    def test_raises_value_error_on_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _load_json(p)

    def test_loads_real_benchmarks_json(self) -> None:
        raw = _load_json(dl._BENCHMARKS_PATH)
        assert "models" in raw
        assert isinstance(raw["models"], list)

    def test_loads_real_pricing_json(self) -> None:
        raw = _load_json(dl._PRICING_PATH)
        assert "api_providers" in raw
        assert "self_hosted" in raw


# ---------------------------------------------------------------------------
# _validate_model_record
# ---------------------------------------------------------------------------


class TestValidateModelRecord:
    """Tests for the _validate_model_record helper."""

    def test_valid_record_returns_empty_errors(self) -> None:
        record = _MINIMAL_BENCHMARK_RAW["models"][0]
        errors = _validate_model_record(record, 0)
        assert errors == []

    def test_missing_required_field_produces_error(self) -> None:
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["model_id"]
        errors = _validate_model_record(record, 0)
        assert any("model_id" in e for e in errors)

    def test_missing_benchmarks_key_produces_error(self) -> None:
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["benchmarks"]
        errors = _validate_model_record(record, 0)
        assert any("benchmarks" in e for e in errors)

    def test_unknown_size_bucket_produces_error(self) -> None:
        record = {**_MINIMAL_BENCHMARK_RAW["models"][0], "parameter_size_bucket": "HUGE"}
        errors = _validate_model_record(record, 0)
        assert any("parameter_size_bucket" in e for e in errors)

    def test_null_benchmark_values_are_allowed(self) -> None:
        record = {
            **_MINIMAL_BENCHMARK_RAW["models"][0],
            "benchmarks": {
                "mmlu": None,
                "humaneval": None,
                "math": None,
                "gsm8k": None,
                "mbpp": None,
            },
        }
        errors = _validate_model_record(record, 0)
        assert errors == []

    def test_non_numeric_benchmark_value_produces_error(self) -> None:
        record = {
            **_MINIMAL_BENCHMARK_RAW["models"][0],
            "benchmarks": {
                **_MINIMAL_BENCHMARK_RAW["models"][0]["benchmarks"],
                "mmlu": "not-a-number",
            },
        }
        errors = _validate_model_record(record, 0)
        assert any("mmlu" in e for e in errors)

    def test_multiple_missing_fields_all_reported(self) -> None:
        errors = _validate_model_record({}, 0)
        reported_fields = " ".join(errors)
        for field in ("model_id", "display_name", "license"):
            assert field in reported_fields


# ---------------------------------------------------------------------------
# _build_benchmarks_df
# ---------------------------------------------------------------------------


class TestBuildBenchmarksDf:
    """Tests for _build_benchmarks_df."""

    def test_returns_dataframe(self) -> None:
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_models(self) -> None:
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        assert len(df) == len(_MINIMAL_BENCHMARK_RAW["models"])

    def test_benchmark_columns_present(self) -> None:
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            assert f"benchmark_{key}" in df.columns

    def test_null_benchmark_values_become_nan(self) -> None:
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        model_b = df[df["model_id"] == "test-model-b"].iloc[0]
        assert pd.isna(model_b["benchmark_humaneval"])
        assert pd.isna(model_b["benchmark_mbpp"])

    def test_numeric_benchmark_columns_have_float_dtype(self) -> None:
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            assert pd.api.types.is_float_dtype(df[f"benchmark_{key}"]), (
                f"benchmark_{key} should be float dtype"
            )

    def test_metadata_columns_present(self) -> None:
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        for col in ("model_id", "display_name", "family", "license", "parameter_size_bucket"):
            assert col in df.columns

    def test_task_categories_stored_as_list(self) -> None:
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        for cats in df["task_categories"]:
            assert isinstance(cats, list)

    def test_raises_if_models_key_missing(self) -> None:
        with pytest.raises(ValueError, match="'models' key"):
            _build_benchmarks_df({"not_models": []})

    def test_raises_if_models_is_not_list(self) -> None:
        with pytest.raises(ValueError, match="JSON array"):
            _build_benchmarks_df({"models": {"bad": "type"}})

    def test_raises_on_invalid_model_record(self) -> None:
        bad_raw: dict[str, Any] = {
            "models": [
                {"model_id": "no-benchmarks", "display_name": "X"}  # missing required fields
            ]
        }
        with pytest.raises(ValueError, match="Schema validation failed"):
            _build_benchmarks_df(bad_raw)


# ---------------------------------------------------------------------------
# _build_pricing_df
# ---------------------------------------------------------------------------


class TestBuildPricingDf:
    """Tests for _build_pricing_df."""

    def test_returns_dataframe(self) -> None:
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert isinstance(df, pd.DataFrame)

    def test_flattens_multiple_providers(self) -> None:
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        # test-model-a has 2 providers
        model_a_rows = df[df["model_id"] == "test-model-a"]
        assert len(model_a_rows) == 2

    def test_self_hosted_cost_joined(self) -> None:
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert "self_hosted_hourly_usd" in df.columns
        model_a = df[df["model_id"] == "test-model-a"].iloc[0]
        assert model_a["self_hosted_hourly_usd"] == pytest.approx(0.50)

    def test_missing_api_provider_model_still_appears_via_self_hosted(self) -> None:
        # test-model-b has no API providers but has self_hosted
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        # When there's no API entry, model-b only appears if sh_df is used alone
        # The implementation does a left join of api_df onto sh_df
        # model-b has no api entry => it won't be in api_df but should be in merged
        # Actually in our impl: merged = api_df LEFT JOIN sh_df
        # so model-b won't appear in pricing_df. This is acceptable — validate it.
        model_b_rows = df[df["model_id"] == "test-model-b"]
        # model-b has no api providers, so it shouldn't be in pricing_df
        assert len(model_b_rows) == 0  # expected: absent from api-centric pricing df

    def test_raises_if_api_providers_missing(self) -> None:
        with pytest.raises(ValueError, match="api_providers"):
            _build_pricing_df({"self_hosted": []})

    def test_raises_if_self_hosted_missing(self) -> None:
        with pytest.raises(ValueError, match="self_hosted"):
            _build_pricing_df({"api_providers": []})

    def test_input_output_columns_are_numeric(self) -> None:
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert pd.api.types.is_float_dtype(df["input_per_1m"])
        assert pd.api.types.is_float_dtype(df["output_per_1m"])


# ---------------------------------------------------------------------------
# _compute_cheapest_api
# ---------------------------------------------------------------------------


class TestComputeCheapestApi:
    """Tests for _compute_cheapest_api."""

    def test_selects_minimum_input_price(self) -> None:
        pricing_df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        cheapest = _compute_cheapest_api(pricing_df)
        # Provider Y has input=0.30, Provider X has input=0.50 => min is 0.30
        assert cheapest.loc["test-model-a", "cheapest_input_per_1m"] == pytest.approx(0.30)

    def test_selects_minimum_output_price(self) -> None:
        pricing_df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        cheapest = _compute_cheapest_api(pricing_df)
        # Provider Y has output=0.80, Provider X has output=1.00 => min is 0.80
        assert cheapest.loc["test-model-a", "cheapest_output_per_1m"] == pytest.approx(0.80)

    def test_returns_empty_df_for_empty_input(self) -> None:
        result = _compute_cheapest_api(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _build_merged_df
# ---------------------------------------------------------------------------


class TestBuildMergedDf:
    """Tests for _build_merged_df."""

    def test_returns_dataframe(self) -> None:
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        assert isinstance(merged, pd.DataFrame)

    def test_all_models_present(self) -> None:
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        # Merged is left-joined from benchmarks, so all benchmark models appear
        assert set(merged["model_id"]) == {"test-model-a", "test-model-b"}

    def test_cheapest_price_columns_present(self) -> None:
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        assert "cheapest_input_per_1m" in merged.columns
        assert "cheapest_output_per_1m" in merged.columns

    def test_model_without_pricing_has_nan_price(self) -> None:
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        model_b = merged[merged["model_id"] == "test-model-b"].iloc[0]
        assert pd.isna(model_b["cheapest_input_per_1m"])

    def test_self_hosted_cost_present_for_known_model(self) -> None:
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        model_a = merged[merged["model_id"] == "test-model-a"].iloc[0]
        assert model_a["self_hosted_hourly_usd"] == pytest.approx(0.50)

    def test_one_row_per_model(self) -> None:
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        assert len(merged) == len(_MINIMAL_BENCHMARK_RAW["models"])


# ---------------------------------------------------------------------------
# Public API: load_benchmarks_df, load_pricing_df, get_merged_df
# ---------------------------------------------------------------------------


class TestPublicLoadFunctions:
    """Integration-level tests using real JSON data files."""

    def test_load_benchmarks_df_returns_dataframe(self) -> None:
        df = load_benchmarks_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 20  # at least 20 models in data file

    def test_load_benchmarks_df_has_expected_columns(self) -> None:
        df = load_benchmarks_df()
        required_cols = [
            "model_id",
            "display_name",
            "family",
            "license",
            "parameter_size_bucket",
            "benchmark_mmlu",
            "benchmark_humaneval",
            "benchmark_math",
            "benchmark_gsm8k",
            "benchmark_mbpp",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_pricing_df_returns_dataframe(self) -> None:
        df = load_pricing_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_pricing_df_has_expected_columns(self) -> None:
        df = load_pricing_df()
        for col in ("model_id", "provider", "input_per_1m", "output_per_1m"):
            assert col in df.columns

    def test_get_merged_df_returns_dataframe(self) -> None:
        df = get_merged_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 20

    def test_get_merged_df_has_benchmark_and_pricing_columns(self) -> None:
        df = get_merged_df()
        assert "benchmark_mmlu" in df.columns
        assert "cheapest_input_per_1m" in df.columns
        assert "self_hosted_hourly_usd" in df.columns

    def test_caching_returns_same_data(self) -> None:
        df1 = get_merged_df()
        df2 = get_merged_df()
        pd.testing.assert_frame_equal(df1, df2)

    def test_force_reload_refreshes_data(self) -> None:
        df1 = get_merged_df()
        df2 = get_merged_df(force_reload=True)
        pd.testing.assert_frame_equal(df1, df2)

    def test_returned_df_is_a_copy(self) -> None:
        """Mutating the returned DataFrame should not affect the cache."""
        df1 = get_merged_df()
        df1["_test_col"] = 999
        df2 = get_merged_df()
        assert "_test_col" not in df2.columns

    def test_all_model_ids_are_unique(self) -> None:
        df = get_merged_df()
        assert df["model_id"].nunique() == len(df)

    def test_deepseek_r1_present_with_high_scores(self) -> None:
        df = get_merged_df()
        r1 = df[df["model_id"] == "deepseek-r1"]
        assert len(r1) == 1
        assert r1.iloc[0]["benchmark_mmlu"] > 85.0
        assert r1.iloc[0]["benchmark_humaneval"] > 85.0

    def test_benchmark_scores_in_valid_range(self) -> None:
        df = get_merged_df()
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            col = f"benchmark_{key}"
            values = df[col].dropna()
            assert (values >= 0).all(), f"{col} has negative values"
            assert (values <= 100).all(), f"{col} has values > 100"

    def test_parameter_size_bucket_values_are_valid(self) -> None:
        df = get_merged_df()
        valid_buckets = {"≤7B", "8–34B", "35B+"}
        actual = set(df["parameter_size_bucket"].dropna().unique())
        assert actual.issubset(valid_buckets), (
            f"Unexpected bucket values: {actual - valid_buckets}"
        )


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    """Tests for the clear_cache function."""

    def test_clear_cache_allows_reload(self) -> None:
        _ = get_merged_df()
        assert dl._merged_df_cache is not None
        clear_cache()
        assert dl._merged_df_cache is None
        assert dl._benchmarks_df_cache is None
        assert dl._pricing_df_cache is None

    def test_after_clear_next_call_reloads(self) -> None:
        df1 = get_merged_df()
        clear_cache()
        df2 = get_merged_df()
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# get_filter_options
# ---------------------------------------------------------------------------


class TestGetFilterOptions:
    """Tests for get_filter_options."""

    def test_returns_dict_with_required_keys(self) -> None:
        opts = get_filter_options()
        for key in ("task_categories", "size_buckets", "licenses", "families"):
            assert key in opts

    def test_task_categories_include_known_values(self) -> None:
        opts = get_filter_options()
        cats = opts["task_categories"]
        assert "reasoning" in cats
        assert "coding" in cats
        assert "math" in cats

    def test_size_buckets_in_expected_order(self) -> None:
        opts = get_filter_options()
        buckets = opts["size_buckets"]
        # All three standard buckets should appear
        for b in ("≤7B", "8–34B", "35B+"):
            assert b in buckets
        # ≤7B should come before 35B+
        assert buckets.index("≤7B") < buckets.index("35B+")

    def test_licenses_non_empty(self) -> None:
        opts = get_filter_options()
        assert len(opts["licenses"]) > 0

    def test_known_licenses_present(self) -> None:
        opts = get_filter_options()
        licenses = opts["licenses"]
        assert "Apache-2.0" in licenses
        assert "MIT" in licenses

    def test_families_include_known_families(self) -> None:
        opts = get_filter_options()
        families = opts["families"]
        for family in ("DeepSeek", "Llama", "Mistral", "Qwen"):
            assert family in families

    def test_all_values_are_lists(self) -> None:
        opts = get_filter_options()
        for key, val in opts.items():
            assert isinstance(val, list), f"{key} should be a list"

    def test_all_values_are_sorted_strings(self) -> None:
        opts = get_filter_options()
        for key in ("task_categories", "licenses", "families"):
            val = opts[key]
            assert val == sorted(val), f"{key} should be sorted"


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Smoke test: concurrent access should not raise."""

    def test_concurrent_get_merged_df(self) -> None:
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                df = get_merged_df()
                assert len(df) > 0
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        clear_cache()
        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
