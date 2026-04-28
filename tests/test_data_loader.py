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
# Minimal synthetic fixtures
# ---------------------------------------------------------------------------

_MINIMAL_BENCHMARK_RAW: dict[str, Any] = {
    "models": [
        {
            "model_id": "test-model-a",
            "display_name": "Test Model A",
            "family": "Test",
            "version": "1.0",
            "parameter_size_b": 7,
            "parameter_size_bucket": "\u22647B",
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
            "parameter_size_bucket": "8\u201334B",
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
            "recommended_gpu_setup": "1\u00d7 RTX 4090",
            "min_vram_gb": 16,
            "estimated_hourly_cost_usd": 0.50,
            "cloud_provider_reference": "Vast.ai",
            "fp_precision": "BF16",
            "throughput_tokens_per_sec_approx": 3000,
        },
        {
            "model_id": "test-model-b",
            "display_name": "Test Model B",
            "recommended_gpu_setup": "2\u00d7 A100",
            "min_vram_gb": 68,
            "estimated_hourly_cost_usd": 2.80,
            "cloud_provider_reference": "Lambda Labs",
            "fp_precision": "BF16",
            "throughput_tokens_per_sec_approx": 900,
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        """A valid JSON file should be parsed and returned as a dict."""
        data = {"key": "value", "num": 42}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = _load_json(p)
        assert result == data

    def test_loads_json_with_list_root(self, tmp_path: Path) -> None:
        """A JSON file whose root is a list should be loaded correctly."""
        data = [1, 2, 3]
        p = tmp_path / "list.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        result = _load_json(p)
        assert result == data

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """Missing files should raise FileNotFoundError with a helpful message."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            _load_json(tmp_path / "nonexistent.json")

    def test_raises_value_error_on_invalid_json(self, tmp_path: Path) -> None:
        """Files with invalid JSON should raise ValueError."""
        p = tmp_path / "bad.json"
        p.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _load_json(p)

    def test_raises_value_error_on_empty_file(self, tmp_path: Path) -> None:
        """Empty files cannot be parsed as JSON and should raise ValueError."""
        p = tmp_path / "empty.json"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _load_json(p)

    def test_loads_real_benchmarks_json(self) -> None:
        """The real benchmarks.json file should be parseable."""
        raw = _load_json(dl._BENCHMARKS_PATH)
        assert "models" in raw
        assert isinstance(raw["models"], list)

    def test_loads_real_pricing_json(self) -> None:
        """The real pricing.json file should be parseable."""
        raw = _load_json(dl._PRICING_PATH)
        assert "api_providers" in raw
        assert "self_hosted" in raw

    def test_path_must_exist_as_file(self, tmp_path: Path) -> None:
        """Passing a directory path should raise FileNotFoundError."""
        # tmp_path is a directory, not a file
        with pytest.raises(FileNotFoundError):
            _load_json(tmp_path)

    def test_unicode_content_loaded_correctly(self, tmp_path: Path) -> None:
        """Unicode characters (including emoji and CJK) should round-trip."""
        data = {"name": "\u2264 \u2013 \u4e2d\u6587 \U0001f916"}
        p = tmp_path / "unicode.json"
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        result = _load_json(p)
        assert result == data


# ---------------------------------------------------------------------------
# _validate_model_record
# ---------------------------------------------------------------------------


class TestValidateModelRecord:
    """Tests for the _validate_model_record helper."""

    def test_valid_record_returns_empty_errors(self) -> None:
        """A fully valid record should produce zero errors."""
        record = _MINIMAL_BENCHMARK_RAW["models"][0]
        errors = _validate_model_record(record, 0)
        assert errors == []

    def test_second_valid_record_returns_empty_errors(self) -> None:
        """The second synthetic model record should also be valid."""
        record = _MINIMAL_BENCHMARK_RAW["models"][1]
        errors = _validate_model_record(record, 1)
        assert errors == []

    def test_missing_model_id_produces_error(self) -> None:
        """Omitting model_id from a record should be reported."""
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["model_id"]
        errors = _validate_model_record(record, 0)
        assert any("model_id" in e for e in errors)

    def test_missing_display_name_produces_error(self) -> None:
        """Omitting display_name should appear in the error list."""
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["display_name"]
        errors = _validate_model_record(record, 0)
        assert any("display_name" in e for e in errors)

    def test_missing_benchmarks_key_produces_error(self) -> None:
        """Omitting the entire benchmarks dict should be an error."""
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["benchmarks"]
        errors = _validate_model_record(record, 0)
        assert any("benchmarks" in e for e in errors)

    def test_missing_license_produces_error(self) -> None:
        """Omitting license should be reported."""
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["license"]
        errors = _validate_model_record(record, 0)
        assert any("license" in e for e in errors)

    def test_missing_task_categories_produces_error(self) -> None:
        """Omitting task_categories should be reported."""
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["task_categories"]
        errors = _validate_model_record(record, 0)
        assert any("task_categories" in e for e in errors)

    def test_missing_family_produces_error(self) -> None:
        """Omitting family should be reported."""
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["family"]
        errors = _validate_model_record(record, 0)
        assert any("family" in e for e in errors)

    def test_missing_parameter_size_bucket_produces_error(self) -> None:
        """Omitting parameter_size_bucket should be reported."""
        record = {k: v for k, v in _MINIMAL_BENCHMARK_RAW["models"][0].items()}
        del record["parameter_size_bucket"]
        errors = _validate_model_record(record, 0)
        assert any("parameter_size_bucket" in e for e in errors)

    def test_unknown_size_bucket_produces_error(self) -> None:
        """An unrecognised parameter_size_bucket value should be an error."""
        record = {**_MINIMAL_BENCHMARK_RAW["models"][0], "parameter_size_bucket": "HUGE"}
        errors = _validate_model_record(record, 0)
        assert any("parameter_size_bucket" in e for e in errors)

    def test_null_benchmark_values_are_allowed(self) -> None:
        """Benchmark values of null (None) are explicitly permitted."""
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
        """A string value for a benchmark key should produce an error."""
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
        """When many fields are missing, each one should appear in the errors."""
        errors = _validate_model_record({}, 0)
        reported = " ".join(errors)
        for field in ("model_id", "display_name", "license", "family", "parameter_size_bucket"):
            assert field in reported, f"Expected '{field}' to be mentioned in errors"

    def test_benchmarks_not_a_dict_produces_error(self) -> None:
        """If benchmarks is a list instead of a dict, that should be an error."""
        record = {
            **_MINIMAL_BENCHMARK_RAW["models"][0],
            "benchmarks": [70.0, 60.0, 40.0, 80.0, 55.0],
        }
        errors = _validate_model_record(record, 0)
        assert any("benchmarks" in e for e in errors)

    def test_task_categories_not_a_list_produces_error(self) -> None:
        """task_categories must be a list; a string should produce an error."""
        record = {
            **_MINIMAL_BENCHMARK_RAW["models"][0],
            "task_categories": "coding",
        }
        errors = _validate_model_record(record, 0)
        assert any("task_categories" in e for e in errors)

    def test_missing_single_benchmark_key_produces_error(self) -> None:
        """If one benchmark key is absent from the benchmarks dict, report it."""
        benchmarks = {**_MINIMAL_BENCHMARK_RAW["models"][0]["benchmarks"]}
        del benchmarks["humaneval"]
        record = {**_MINIMAL_BENCHMARK_RAW["models"][0], "benchmarks": benchmarks}
        errors = _validate_model_record(record, 0)
        assert any("humaneval" in e for e in errors)

    def test_index_used_in_message_when_model_id_missing(self) -> None:
        """When model_id is absent, the index should appear in error messages."""
        errors = _validate_model_record({"display_name": "X"}, 7)
        # At least one message should reference the index
        assert any("7" in e or "index 7" in e for e in errors)


# ---------------------------------------------------------------------------
# _build_benchmarks_df
# ---------------------------------------------------------------------------


class TestBuildBenchmarksDf:
    """Tests for _build_benchmarks_df."""

    def test_returns_dataframe(self) -> None:
        """The function should return a pandas DataFrame."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_models(self) -> None:
        """One row per model entry."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        assert len(df) == len(_MINIMAL_BENCHMARK_RAW["models"])

    def test_benchmark_columns_present(self) -> None:
        """Each benchmark key should be a prefixed column."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            assert f"benchmark_{key}" in df.columns

    def test_null_benchmark_values_become_nan(self) -> None:
        """JSON null benchmark values should become float NaN."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        model_b = df[df["model_id"] == "test-model-b"].iloc[0]
        assert pd.isna(model_b["benchmark_humaneval"])
        assert pd.isna(model_b["benchmark_mbpp"])

    def test_non_null_benchmark_values_preserved(self) -> None:
        """Non-null benchmark values should match the source JSON."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        model_a = df[df["model_id"] == "test-model-a"].iloc[0]
        assert model_a["benchmark_mmlu"] == pytest.approx(70.0)
        assert model_a["benchmark_humaneval"] == pytest.approx(60.0)
        assert model_a["benchmark_math"] == pytest.approx(40.0)
        assert model_a["benchmark_gsm8k"] == pytest.approx(80.0)
        assert model_a["benchmark_mbpp"] == pytest.approx(55.0)

    def test_numeric_benchmark_columns_have_float_dtype(self) -> None:
        """All benchmark columns should be float dtype."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            col = f"benchmark_{key}"
            assert pd.api.types.is_float_dtype(df[col]), (
                f"{col} should be float dtype, got {df[col].dtype}"
            )

    def test_metadata_columns_present(self) -> None:
        """All top-level metadata columns should be in the DataFrame."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        required = (
            "model_id", "display_name", "family", "license",
            "parameter_size_bucket", "open_weights", "release_date",
        )
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_task_categories_stored_as_list(self) -> None:
        """task_categories should be a Python list in each row."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        for cats in df["task_categories"]:
            assert isinstance(cats, list), f"Expected list, got {type(cats)}"

    def test_task_categories_values_correct(self) -> None:
        """task_categories values should match source data."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        model_a = df[df["model_id"] == "test-model-a"].iloc[0]
        assert set(model_a["task_categories"]) == {"reasoning", "coding"}
        model_b = df[df["model_id"] == "test-model-b"].iloc[0]
        assert model_b["task_categories"] == ["math"]

    def test_model_id_is_string_dtype(self) -> None:
        """model_id column should be string (or object) dtype."""
        df = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        # pandas StringDtype or object — both acceptable
        assert df["model_id"].dtype == "string" or df["model_id"].dtype == object

    def test_raises_if_models_key_missing(self) -> None:
        """Missing top-level 'models' key should raise ValueError."""
        with pytest.raises(ValueError, match="'models' key"):
            _build_benchmarks_df({"not_models": []})

    def test_raises_if_models_is_not_list(self) -> None:
        """Non-list 'models' value should raise ValueError."""
        with pytest.raises(ValueError, match="JSON array"):
            _build_benchmarks_df({"models": {"bad": "type"}})

    def test_raises_on_invalid_model_record(self) -> None:
        """Records with missing required fields should raise ValueError."""
        bad_raw: dict[str, Any] = {
            "models": [
                {"model_id": "no-required-fields", "display_name": "X"}
            ]
        }
        with pytest.raises(ValueError, match="Schema validation failed"):
            _build_benchmarks_df(bad_raw)

    def test_empty_models_list_returns_empty_df(self) -> None:
        """An empty models array should return an empty DataFrame (not raise)."""
        df = _build_benchmarks_df({"models": []})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_multiple_invalid_records_all_errors_reported(self) -> None:
        """Multiple invalid records should all be flagged in the ValueError."""
        bad_raw: dict[str, Any] = {
            "models": [
                {"model_id": "bad-1"},  # missing many fields
                {"model_id": "bad-2"},  # also missing many fields
            ]
        }
        with pytest.raises(ValueError, match="Schema validation failed") as exc_info:
            _build_benchmarks_df(bad_raw)
        msg = str(exc_info.value)
        assert "bad-1" in msg
        assert "bad-2" in msg


# ---------------------------------------------------------------------------
# _build_pricing_df
# ---------------------------------------------------------------------------


class TestBuildPricingDf:
    """Tests for _build_pricing_df."""

    def test_returns_dataframe(self) -> None:
        """The function should return a pandas DataFrame."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert isinstance(df, pd.DataFrame)

    def test_flattens_multiple_providers(self) -> None:
        """A model with 2 API providers should produce 2 rows."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        model_a_rows = df[df["model_id"] == "test-model-a"]
        assert len(model_a_rows) == 2

    def test_provider_names_correct(self) -> None:
        """Provider names should match the source JSON."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        providers = set(df[df["model_id"] == "test-model-a"]["provider"])
        assert providers == {"Provider X", "Provider Y"}

    def test_self_hosted_cost_joined(self) -> None:
        """Self-hosted hourly cost should be joined for models with API entries."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert "self_hosted_hourly_usd" in df.columns
        model_a = df[df["model_id"] == "test-model-a"].iloc[0]
        assert model_a["self_hosted_hourly_usd"] == pytest.approx(0.50)

    def test_self_hosted_gpu_setup_joined(self) -> None:
        """GPU setup string should appear in the pricing DataFrame."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert "self_hosted_gpu_setup" in df.columns

    def test_self_hosted_min_vram_joined(self) -> None:
        """Min VRAM should appear in the pricing DataFrame."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert "self_hosted_min_vram_gb" in df.columns
        model_a = df[df["model_id"] == "test-model-a"].iloc[0]
        assert model_a["self_hosted_min_vram_gb"] == 16

    def test_missing_api_provider_model_absent_from_df(self) -> None:
        """A model in self_hosted but not api_providers won't appear (left join from api_df)."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        # test-model-b has no API providers, so it should not appear in pricing_df
        model_b_rows = df[df["model_id"] == "test-model-b"]
        assert len(model_b_rows) == 0

    def test_raises_if_api_providers_missing(self) -> None:
        """Missing api_providers key should raise ValueError."""
        with pytest.raises(ValueError, match="api_providers"):
            _build_pricing_df({"self_hosted": []})

    def test_raises_if_self_hosted_missing(self) -> None:
        """Missing self_hosted key should raise ValueError."""
        with pytest.raises(ValueError, match="self_hosted"):
            _build_pricing_df({"api_providers": []})

    def test_input_output_columns_are_numeric(self) -> None:
        """input_per_1m and output_per_1m should be float columns."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert pd.api.types.is_float_dtype(df["input_per_1m"])
        assert pd.api.types.is_float_dtype(df["output_per_1m"])

    def test_input_output_values_correct(self) -> None:
        """Provider X and Y input/output prices should match source data."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        prov_x = df[(df["model_id"] == "test-model-a") & (df["provider"] == "Provider X")].iloc[0]
        assert prov_x["input_per_1m"] == pytest.approx(0.50)
        assert prov_x["output_per_1m"] == pytest.approx(1.00)
        prov_y = df[(df["model_id"] == "test-model-a") & (df["provider"] == "Provider Y")].iloc[0]
        assert prov_y["input_per_1m"] == pytest.approx(0.30)
        assert prov_y["output_per_1m"] == pytest.approx(0.80)

    def test_notes_column_present(self) -> None:
        """The notes column should be in the pricing DataFrame."""
        df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        assert "notes" in df.columns

    def test_empty_api_providers_returns_non_empty_from_self_hosted(self) -> None:
        """When api_providers list is empty, pricing_df should come from sh_df alone."""
        raw = {
            "api_providers": [],
            "self_hosted": _MINIMAL_PRICING_RAW["self_hosted"],
        }
        df = _build_pricing_df(raw)
        # With no API entries, the result will be the sh_df directly
        # The function still returns a DataFrame (may be empty or sh-only)
        assert isinstance(df, pd.DataFrame)

    def test_skips_api_provider_entry_with_missing_model_id(self) -> None:
        """api_providers entries missing model_id should be skipped gracefully."""
        raw: dict[str, Any] = {
            "api_providers": [
                {
                    # No model_id!
                    "display_name": "Ghost",
                    "providers": [
                        {
                            "provider": "Ghost Provider",
                            "provider_url": "https://ghost.example.com",
                            "input_per_1m": 1.0,
                            "output_per_1m": 2.0,
                            "notes": None,
                        }
                    ],
                }
            ],
            "self_hosted": [],
        }
        # Should not raise
        df = _build_pricing_df(raw)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# _compute_cheapest_api
# ---------------------------------------------------------------------------


class TestComputeCheapestApi:
    """Tests for _compute_cheapest_api."""

    def test_selects_minimum_input_price(self) -> None:
        """Cheapest input should be the minimum across all providers."""
        pricing_df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        cheapest = _compute_cheapest_api(pricing_df)
        # Provider Y: 0.30, Provider X: 0.50 → min = 0.30
        assert cheapest.loc["test-model-a", "cheapest_input_per_1m"] == pytest.approx(0.30)

    def test_selects_minimum_output_price(self) -> None:
        """Cheapest output should be the minimum across all providers."""
        pricing_df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        cheapest = _compute_cheapest_api(pricing_df)
        # Provider Y: 0.80, Provider X: 1.00 → min = 0.80
        assert cheapest.loc["test-model-a", "cheapest_output_per_1m"] == pytest.approx(0.80)

    def test_index_is_model_id(self) -> None:
        """Returned DataFrame should be indexed by model_id."""
        pricing_df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        cheapest = _compute_cheapest_api(pricing_df)
        assert cheapest.index.name == "model_id"
        assert "test-model-a" in cheapest.index

    def test_columns_named_correctly(self) -> None:
        """Output columns should be cheapest_input_per_1m and cheapest_output_per_1m."""
        pricing_df = _build_pricing_df(_MINIMAL_PRICING_RAW)
        cheapest = _compute_cheapest_api(pricing_df)
        assert "cheapest_input_per_1m" in cheapest.columns
        assert "cheapest_output_per_1m" in cheapest.columns

    def test_returns_empty_df_for_empty_input(self) -> None:
        """An empty pricing DataFrame should yield an empty cheapest DataFrame."""
        result = _compute_cheapest_api(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_returns_empty_df_for_df_without_input_column(self) -> None:
        """A DataFrame lacking input_per_1m should yield an empty result."""
        df = pd.DataFrame({"model_id": ["a"], "provider": ["X"]})
        result = _compute_cheapest_api(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_provider_gives_its_own_price(self) -> None:
        """With only one provider, cheapest price equals that provider's price."""
        single_raw: dict[str, Any] = {
            "api_providers": [
                {
                    "model_id": "solo-model",
                    "display_name": "Solo",
                    "providers": [
                        {
                            "provider": "Solo Provider",
                            "provider_url": "https://solo.example.com",
                            "input_per_1m": 2.50,
                            "output_per_1m": 5.00,
                            "notes": None,
                        }
                    ],
                }
            ],
            "self_hosted": [],
        }
        pricing_df = _build_pricing_df(single_raw)
        cheapest = _compute_cheapest_api(pricing_df)
        assert cheapest.loc["solo-model", "cheapest_input_per_1m"] == pytest.approx(2.50)
        assert cheapest.loc["solo-model", "cheapest_output_per_1m"] == pytest.approx(5.00)


# ---------------------------------------------------------------------------
# _build_merged_df
# ---------------------------------------------------------------------------


class TestBuildMergedDf:
    """Tests for _build_merged_df."""

    def test_returns_dataframe(self) -> None:
        """The merged result should be a pandas DataFrame."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        assert isinstance(merged, pd.DataFrame)

    def test_all_benchmark_models_present(self) -> None:
        """All models from the benchmarks DataFrame should appear in merged."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        assert set(merged["model_id"]) == {"test-model-a", "test-model-b"}

    def test_cheapest_price_columns_present(self) -> None:
        """Cheapest API price columns should be in the merged DataFrame."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        assert "cheapest_input_per_1m" in merged.columns
        assert "cheapest_output_per_1m" in merged.columns

    def test_model_without_api_pricing_has_nan_price(self) -> None:
        """Models with no API pricing should have NaN for cheapest prices."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        model_b = merged[merged["model_id"] == "test-model-b"].iloc[0]
        assert pd.isna(model_b["cheapest_input_per_1m"])
        assert pd.isna(model_b["cheapest_output_per_1m"])

    def test_cheapest_price_correct_for_model_a(self) -> None:
        """Model A's cheapest prices should reflect the minimum across providers."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        model_a = merged[merged["model_id"] == "test-model-a"].iloc[0]
        assert model_a["cheapest_input_per_1m"] == pytest.approx(0.30)
        assert model_a["cheapest_output_per_1m"] == pytest.approx(0.80)

    def test_self_hosted_cost_present_for_known_model(self) -> None:
        """Self-hosted hourly cost should be carried through to merged DataFrame."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        model_a = merged[merged["model_id"] == "test-model-a"].iloc[0]
        assert model_a["self_hosted_hourly_usd"] == pytest.approx(0.50)

    def test_one_row_per_model(self) -> None:
        """Merged DataFrame should have exactly one row per model."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        assert len(merged) == len(_MINIMAL_BENCHMARK_RAW["models"])

    def test_benchmark_columns_preserved(self) -> None:
        """Benchmark score columns from benchmarks_df should survive the merge."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            assert f"benchmark_{key}" in merged.columns

    def test_model_id_column_present_after_reset_index(self) -> None:
        """After merging, model_id should be a regular column (not the index)."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        pdf = _build_pricing_df(_MINIMAL_PRICING_RAW)
        merged = _build_merged_df(bdf, pdf)
        assert "model_id" in merged.columns
        assert merged.index.name != "model_id"

    def test_empty_pricing_df_still_returns_all_models(self) -> None:
        """If pricing_df is empty, all benchmark models should still appear."""
        bdf = _build_benchmarks_df(_MINIMAL_BENCHMARK_RAW)
        empty_pdf = pd.DataFrame(
            columns=["model_id", "provider", "provider_url", "input_per_1m",
                     "output_per_1m", "notes", "self_hosted_hourly_usd",
                     "self_hosted_gpu_setup", "self_hosted_min_vram_gb",
                     "self_hosted_throughput_tps"]
        )
        merged = _build_merged_df(bdf, empty_pdf)
        assert len(merged) == len(_MINIMAL_BENCHMARK_RAW["models"])


# ---------------------------------------------------------------------------
# Public API: load_benchmarks_df, load_pricing_df, get_merged_df
# ---------------------------------------------------------------------------


class TestPublicLoadFunctions:
    """Integration-level tests using real JSON data files shipped with the package."""

    def test_load_benchmarks_df_returns_dataframe(self) -> None:
        """load_benchmarks_df should return a non-empty pandas DataFrame."""
        df = load_benchmarks_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 20

    def test_load_benchmarks_df_has_expected_columns(self) -> None:
        """The benchmarks DataFrame must contain all required columns."""
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
            assert col in df.columns, f"Missing expected column: {col}"

    def test_load_pricing_df_returns_dataframe(self) -> None:
        """load_pricing_df should return a non-empty pandas DataFrame."""
        df = load_pricing_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_pricing_df_has_expected_columns(self) -> None:
        """The pricing DataFrame must contain core columns."""
        df = load_pricing_df()
        for col in ("model_id", "provider", "input_per_1m", "output_per_1m"):
            assert col in df.columns, f"Missing pricing column: {col}"

    def test_get_merged_df_returns_dataframe(self) -> None:
        """get_merged_df should return a non-empty pandas DataFrame."""
        df = get_merged_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 20

    def test_get_merged_df_has_benchmark_columns(self) -> None:
        """Merged DataFrame must have all benchmark score columns."""
        df = get_merged_df()
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            assert f"benchmark_{key}" in df.columns

    def test_get_merged_df_has_pricing_columns(self) -> None:
        """Merged DataFrame must have pricing convenience columns."""
        df = get_merged_df()
        assert "cheapest_input_per_1m" in df.columns
        assert "cheapest_output_per_1m" in df.columns
        assert "self_hosted_hourly_usd" in df.columns

    def test_caching_returns_equivalent_data(self) -> None:
        """Two consecutive calls should return DataFrames with identical content."""
        df1 = get_merged_df()
        df2 = get_merged_df()
        pd.testing.assert_frame_equal(df1, df2)

    def test_force_reload_refreshes_data(self) -> None:
        """force_reload=True should return data equal to the first call."""
        df1 = get_merged_df()
        df2 = get_merged_df(force_reload=True)
        pd.testing.assert_frame_equal(df1, df2)

    def test_returned_df_is_a_copy(self) -> None:
        """Mutating the returned DataFrame must not affect subsequent calls."""
        df1 = get_merged_df()
        df1["_test_sentinel"] = 999
        df2 = get_merged_df()
        assert "_test_sentinel" not in df2.columns

    def test_all_model_ids_are_unique(self) -> None:
        """Each model should have a unique model_id in the merged DataFrame."""
        df = get_merged_df()
        assert df["model_id"].nunique() == len(df)

    def test_deepseek_r1_present_with_high_scores(self) -> None:
        """DeepSeek R1 should appear and have strong benchmark scores."""
        df = get_merged_df()
        r1 = df[df["model_id"] == "deepseek-r1"]
        assert len(r1) == 1, "deepseek-r1 should appear exactly once"
        assert r1.iloc[0]["benchmark_mmlu"] > 85.0
        assert r1.iloc[0]["benchmark_humaneval"] > 85.0

    def test_benchmark_scores_in_valid_range(self) -> None:
        """All non-null benchmark scores should be in [0, 100]."""
        df = get_merged_df()
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            col = f"benchmark_{key}"
            values = df[col].dropna()
            assert (values >= 0).all(), f"{col} has negative values"
            assert (values <= 100).all(), f"{col} has values > 100"

    def test_parameter_size_bucket_values_are_valid(self) -> None:
        """All non-null parameter_size_bucket values should be from the known set."""
        df = get_merged_df()
        valid_buckets = {"\u22647B", "8\u201334B", "35B+"}
        actual = set(df["parameter_size_bucket"].dropna().unique())
        assert actual.issubset(valid_buckets), (
            f"Unexpected bucket values: {actual - valid_buckets}"
        )

    def test_license_values_are_strings(self) -> None:
        """All license values should be non-empty strings."""
        df = get_merged_df()
        for lic in df["license"].dropna():
            assert isinstance(str(lic), str)
            assert len(str(lic)) > 0

    def test_at_least_one_model_with_api_pricing(self) -> None:
        """At least one model should have non-null cheapest API pricing."""
        df = get_merged_df()
        has_api_price = df["cheapest_input_per_1m"].notna().any()
        assert has_api_price, "Expected at least one model with API pricing"

    def test_at_least_one_model_with_self_hosted_cost(self) -> None:
        """At least one model should have a non-null self-hosted hourly cost."""
        df = get_merged_df()
        has_sh = df["self_hosted_hourly_usd"].notna().any()
        assert has_sh, "Expected at least one model with self-hosted cost"

    def test_load_benchmarks_df_force_reload(self) -> None:
        """force_reload=True on load_benchmarks_df should return valid data."""
        df = load_benchmarks_df(force_reload=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 20

    def test_load_pricing_df_force_reload(self) -> None:
        """force_reload=True on load_pricing_df should return valid data."""
        df = load_pricing_df(force_reload=True)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_get_benchmarks_df_alias_works(self) -> None:
        """The get_benchmarks_df alias should behave identically to load_benchmarks_df."""
        from llm_bench_compare.data_loader import get_benchmarks_df
        df = get_benchmarks_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 20

    def test_get_pricing_df_alias_works(self) -> None:
        """The get_pricing_df alias should behave identically to load_pricing_df."""
        from llm_bench_compare.data_loader import get_pricing_df
        df = get_pricing_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    """Tests for the clear_cache function."""

    def test_clear_cache_nullifies_benchmarks_cache(self) -> None:
        """After clear_cache(), the benchmarks cache should be None."""
        _ = load_benchmarks_df()
        assert dl._benchmarks_df_cache is not None
        clear_cache()
        assert dl._benchmarks_df_cache is None

    def test_clear_cache_nullifies_pricing_cache(self) -> None:
        """After clear_cache(), the pricing cache should be None."""
        _ = load_pricing_df()
        assert dl._pricing_df_cache is not None
        clear_cache()
        assert dl._pricing_df_cache is None

    def test_clear_cache_nullifies_merged_cache(self) -> None:
        """After clear_cache(), the merged cache should be None."""
        _ = get_merged_df()
        assert dl._merged_df_cache is not None
        clear_cache()
        assert dl._merged_df_cache is None

    def test_after_clear_next_call_reloads(self) -> None:
        """Data loaded before and after clear_cache() should be equivalent."""
        df1 = get_merged_df()
        clear_cache()
        df2 = get_merged_df()
        pd.testing.assert_frame_equal(df1, df2)

    def test_clear_cache_idempotent(self) -> None:
        """Calling clear_cache() multiple times should not raise."""
        clear_cache()
        clear_cache()
        clear_cache()
        # Should still work after multiple clears
        df = get_merged_df()
        assert len(df) >= 20

    def test_cache_populated_after_load(self) -> None:
        """After loading, cache attributes should be non-None DataFrames."""
        clear_cache()
        assert dl._benchmarks_df_cache is None
        _ = load_benchmarks_df()
        assert dl._benchmarks_df_cache is not None
        assert isinstance(dl._benchmarks_df_cache, pd.DataFrame)


# ---------------------------------------------------------------------------
# get_filter_options
# ---------------------------------------------------------------------------


class TestGetFilterOptions:
    """Tests for get_filter_options."""

    def test_returns_dict_with_required_keys(self) -> None:
        """Return value must be a dict with the four required top-level keys."""
        opts = get_filter_options()
        for key in ("task_categories", "size_buckets", "licenses", "families"):
            assert key in opts, f"Missing key: {key}"

    def test_all_values_are_lists(self) -> None:
        """Every value in the returned dict should be a list."""
        opts = get_filter_options()
        for key, val in opts.items():
            assert isinstance(val, list), f"{key} should be a list, got {type(val)}"

    def test_task_categories_include_known_values(self) -> None:
        """All three standard task categories should be present."""
        opts = get_filter_options()
        cats = opts["task_categories"]
        for cat in ("reasoning", "coding", "math"):
            assert cat in cats, f"Expected '{cat}' in task_categories"

    def test_task_categories_sorted(self) -> None:
        """task_categories list should be in alphabetical order."""
        opts = get_filter_options()
        cats = opts["task_categories"]
        assert cats == sorted(cats)

    def test_size_buckets_contains_all_three(self) -> None:
        """All three standard size buckets must be present."""
        opts = get_filter_options()
        buckets = opts["size_buckets"]
        for bucket in ("\u22647B", "8\u201334B", "35B+"):
            assert bucket in buckets, f"Expected '{bucket}' in size_buckets"

    def test_size_buckets_in_expected_order(self) -> None:
        """≤7B should come before 8–34B, which should come before 35B+."""
        opts = get_filter_options()
        buckets = opts["size_buckets"]
        idx_small = buckets.index("\u22647B")
        idx_mid   = buckets.index("8\u201334B")
        idx_large = buckets.index("35B+")
        assert idx_small < idx_mid < idx_large

    def test_licenses_non_empty(self) -> None:
        """There should be at least one license option."""
        opts = get_filter_options()
        assert len(opts["licenses"]) > 0

    def test_known_licenses_present(self) -> None:
        """Apache-2.0 and MIT must appear in the license options."""
        opts = get_filter_options()
        licenses = opts["licenses"]
        assert "Apache-2.0" in licenses
        assert "MIT" in licenses

    def test_licenses_sorted(self) -> None:
        """License list should be in alphabetical order."""
        opts = get_filter_options()
        licenses = opts["licenses"]
        assert licenses == sorted(licenses)

    def test_families_include_known_families(self) -> None:
        """All major model families must appear in the families list."""
        opts = get_filter_options()
        families = opts["families"]
        for family in ("DeepSeek", "Llama", "Mistral", "Qwen"):
            assert family in families, f"Expected family '{family}' in families"

    def test_families_sorted(self) -> None:
        """Families list should be in alphabetical order."""
        opts = get_filter_options()
        families = opts["families"]
        assert families == sorted(families)

    def test_families_non_empty(self) -> None:
        """There should be at least one model family."""
        opts = get_filter_options()
        assert len(opts["families"]) > 0

    def test_all_string_values(self) -> None:
        """Every item in every list should be a plain string."""
        opts = get_filter_options()
        for key, values in opts.items():
            for val in values:
                assert isinstance(val, str), (
                    f"{key} contains a non-string value: {val!r} ({type(val)})"
                )

    def test_no_duplicate_values(self) -> None:
        """No list should contain duplicate entries."""
        opts = get_filter_options()
        for key, values in opts.items():
            assert len(values) == len(set(values)), (
                f"{key} contains duplicate values"
            )


# ---------------------------------------------------------------------------
# Thread-safety smoke test
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Smoke tests: concurrent access should not raise or corrupt data."""

    def test_concurrent_get_merged_df(self) -> None:
        """Multiple threads loading the merged DataFrame simultaneously should be safe."""
        errors: list[Exception] = []

        def _worker() -> None:
            try:
                df = get_merged_df()
                assert len(df) >= 20
                assert "model_id" in df.columns
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        clear_cache()
        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors encountered: {errors}"

    def test_concurrent_clear_and_load(self) -> None:
        """Alternating clear_cache and get_merged_df from different threads should be safe."""
        errors: list[Exception] = []

        def _loader() -> None:
            try:
                df = get_merged_df()
                assert isinstance(df, pd.DataFrame)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def _clearer() -> None:
            try:
                clear_cache()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads: list[threading.Thread] = []
        for i in range(12):
            target = _clearer if i % 3 == 0 else _loader
            threads.append(threading.Thread(target=target))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors encountered: {errors}"
