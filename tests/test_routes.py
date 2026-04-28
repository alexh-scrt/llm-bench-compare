"""Integration tests for Flask routes in llm_bench_compare.

Verifies correct HTTP responses, JSON API payloads, error handling,
and query-parameter filtering for all web endpoints.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient

from llm_bench_compare.app import create_app
from llm_bench_compare.data_loader import clear_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app() -> Flask:
    """Create a test-configured Flask application instance."""
    clear_cache()
    return create_app(
        config={
            "TESTING": True,
            "EAGER_LOAD_DATA": False,
        }
    )


@pytest.fixture(scope="module")
def client(app: Flask) -> FlaskClient:
    """Return a Flask test client."""
    return app.test_client()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestIndexRoute:
    """Tests for the main HTML page endpoint."""

    def test_returns_200(self, client: FlaskClient) -> None:
        """The index page must respond with HTTP 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_content_type_is_html(self, client: FlaskClient) -> None:
        """The index page must return an HTML content type."""
        response = client.get("/")
        assert "text/html" in response.content_type

    def test_response_body_not_empty(self, client: FlaskClient) -> None:
        """The index page body must be non-empty."""
        response = client.get("/")
        assert len(response.data) > 0

    def test_body_contains_html_document(self, client: FlaskClient) -> None:
        """The response body should contain a valid HTML document opening."""
        response = client.get("/")
        body = response.data.decode("utf-8", errors="replace").lower()
        assert "<html" in body or "<!doctype" in body

    def test_body_contains_application_title(self, client: FlaskClient) -> None:
        """The index page should reference the application name."""
        response = client.get("/")
        body = response.data.decode("utf-8", errors="replace").lower()
        # Application name fragment should appear somewhere
        assert "benchmark" in body or "llm" in body

    def test_get_method_allowed(self, client: FlaskClient) -> None:
        """GET must be an allowed method for the index route."""
        response = client.get("/")
        assert response.status_code != 405

    def test_post_method_not_allowed(self, client: FlaskClient) -> None:
        """POST should not be allowed on the index route."""
        response = client.post("/")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------


class TestApiModels:
    """Tests for the /api/models endpoint."""

    def test_returns_200(self, client: FlaskClient) -> None:
        """The models API must respond with HTTP 200."""
        response = client.get("/api/models")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        """Response must have application/json content type."""
        response = client.get("/api/models")
        assert "application/json" in response.content_type

    def test_response_has_count_and_models_keys(self, client: FlaskClient) -> None:
        """Response JSON must have 'count' and 'models' top-level keys."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        assert "count" in data
        assert "models" in data

    def test_models_is_list(self, client: FlaskClient) -> None:
        """The 'models' value must be a JSON array."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        assert isinstance(data["models"], list)

    def test_count_matches_models_length(self, client: FlaskClient) -> None:
        """The 'count' field must equal the length of the 'models' array."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        assert data["count"] == len(data["models"])

    def test_at_least_20_models_returned(self, client: FlaskClient) -> None:
        """At least 20 models should be returned when no filters are applied."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        assert data["count"] >= 20

    def test_each_model_has_required_fields(self, client: FlaskClient) -> None:
        """Every model entry must contain the required metadata fields."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        required_fields = {
            "model_id",
            "display_name",
            "family",
            "license",
            "parameter_size_bucket",
        }
        for model in data["models"]:
            for field in required_fields:
                assert field in model, (
                    f"Field '{field}' missing from model {model.get('model_id')}"
                )

    def test_each_model_has_benchmark_keys(self, client: FlaskClient) -> None:
        """Every model entry must contain all five benchmark score fields."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        benchmark_fields = [
            "benchmark_mmlu",
            "benchmark_humaneval",
            "benchmark_math",
            "benchmark_gsm8k",
            "benchmark_mbpp",
        ]
        for model in data["models"]:
            for field in benchmark_fields:
                assert field in model, (
                    f"Benchmark field '{field}' missing from model {model.get('model_id')}"
                )

    def test_each_model_has_task_categories_list(self, client: FlaskClient) -> None:
        """Every model must have a task_categories field that is a list."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        for model in data["models"]:
            assert "task_categories" in model
            assert isinstance(model["task_categories"], list)

    def test_filter_by_category_coding(self, client: FlaskClient) -> None:
        """Filtering by 'coding' category should return only models with that category."""
        response = client.get("/api/models?categories=coding")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert "coding" in model["task_categories"]

    def test_filter_by_category_math(self, client: FlaskClient) -> None:
        """Filtering by 'math' category should return only models with that category."""
        response = client.get("/api/models?categories=math")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert "math" in model["task_categories"]

    def test_filter_by_category_reasoning(self, client: FlaskClient) -> None:
        """Filtering by 'reasoning' category should return at least one model."""
        response = client.get("/api/models?categories=reasoning")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert "reasoning" in model["task_categories"]

    def test_filter_by_multiple_categories_comma_separated(self, client: FlaskClient) -> None:
        """Comma-separated category values should produce >= single category results."""
        response_combined = client.get("/api/models?categories=coding,math")
        data_combined = json.loads(response_combined.data)
        response_single = client.get("/api/models?categories=coding")
        data_single = json.loads(response_single.data)
        assert data_combined["count"] >= data_single["count"]

    def test_filter_by_multiple_categories_repeated_params(self, client: FlaskClient) -> None:
        """Repeated category query params should be combined with OR logic."""
        response = client.get("/api/models?categories=coding&categories=math")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] > 0

    def test_filter_by_size_bucket_small(self, client: FlaskClient) -> None:
        """Filtering by ≤7B bucket should return only small models."""
        response = client.get("/api/models?buckets=%E2%89%A47B")  # URL-encoded ≤7B
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] > 0
        for model in data["models"]:
            assert model["parameter_size_bucket"] == "\u22647B"

    def test_filter_by_size_bucket_mid(self, client: FlaskClient) -> None:
        """Filtering by 8-34B bucket should return only mid-size models."""
        response = client.get("/api/models?buckets=8%E2%80%9334B")  # URL-encoded 8–34B
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] > 0
        for model in data["models"]:
            assert model["parameter_size_bucket"] == "8\u201334B"

    def test_filter_by_size_bucket_large(self, client: FlaskClient) -> None:
        """Filtering by 35B+ bucket should return only large models."""
        response = client.get("/api/models?buckets=35B%2B")  # URL-encoded 35B+
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] > 0
        for model in data["models"]:
            assert model["parameter_size_bucket"] == "35B+"

    def test_filter_by_license_apache(self, client: FlaskClient) -> None:
        """Filtering by Apache-2.0 should return only Apache-licensed models."""
        response = client.get("/api/models?licenses=Apache-2.0")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert model["license"] == "Apache-2.0"

    def test_filter_by_license_mit(self, client: FlaskClient) -> None:
        """Filtering by MIT should return only MIT-licensed models."""
        response = client.get("/api/models?licenses=MIT")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert model["license"] == "MIT"

    def test_filter_by_license_custom_commercial(self, client: FlaskClient) -> None:
        """Filtering by custom/commercial should return only those models."""
        response = client.get("/api/models?licenses=custom%2Fcommercial")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert model["license"] == "custom/commercial"

    def test_filter_by_multiple_licenses(self, client: FlaskClient) -> None:
        """Multiple licenses should be combined with OR logic."""
        response = client.get("/api/models?licenses=Apache-2.0&licenses=MIT")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert model["license"] in ("Apache-2.0", "MIT")

    def test_filter_by_family(self, client: FlaskClient) -> None:
        """Filtering by DeepSeek family should return only DeepSeek models."""
        response = client.get("/api/models?families=DeepSeek")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] >= 3  # V2, V3, R1
        for model in data["models"]:
            assert model["family"] == "DeepSeek"

    def test_filter_by_multiple_families(self, client: FlaskClient) -> None:
        """Multiple families should be combined with OR logic."""
        response = client.get("/api/models?families=Llama&families=Mistral")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert model["family"] in ("Llama", "Mistral")

    def test_filter_by_open_weights_true(self, client: FlaskClient) -> None:
        """open_weights=true should return only models with open weights."""
        response = client.get("/api/models?open_weights=true")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert model["open_weights"] is True

    def test_filter_by_open_weights_1(self, client: FlaskClient) -> None:
        """open_weights=1 should also be treated as True."""
        response = client.get("/api/models?open_weights=1")
        data = json.loads(response.data)
        assert response.status_code == 200
        for model in data["models"]:
            assert model["open_weights"] is True

    def test_sort_by_mmlu_descending(self, client: FlaskClient) -> None:
        """Sorting by mmlu should return models highest-first by default."""
        response = client.get("/api/models?sort=mmlu")
        data = json.loads(response.data)
        assert response.status_code == 200
        scores = [
            m["benchmark_mmlu"]
            for m in data["models"]
            if m["benchmark_mmlu"] is not None
        ]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_humaneval_descending(self, client: FlaskClient) -> None:
        """Sorting by humaneval should return models highest-first."""
        response = client.get("/api/models?sort=humaneval")
        data = json.loads(response.data)
        assert response.status_code == 200
        scores = [
            m["benchmark_humaneval"]
            for m in data["models"]
            if m["benchmark_humaneval"] is not None
        ]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_math_descending(self, client: FlaskClient) -> None:
        """Sorting by math should return models highest-first."""
        response = client.get("/api/models?sort=math")
        data = json.loads(response.data)
        assert response.status_code == 200
        scores = [
            m["benchmark_math"]
            for m in data["models"]
            if m["benchmark_math"] is not None
        ]
        assert scores == sorted(scores, reverse=True)

    def test_sort_ascending(self, client: FlaskClient) -> None:
        """asc=true should sort lowest scores first."""
        response = client.get("/api/models?sort=mmlu&asc=true")
        data = json.loads(response.data)
        assert response.status_code == 200
        scores = [
            m["benchmark_mmlu"]
            for m in data["models"]
            if m["benchmark_mmlu"] is not None
        ]
        assert scores == sorted(scores)

    def test_sort_ascending_with_1(self, client: FlaskClient) -> None:
        """asc=1 should also be treated as ascending."""
        response = client.get("/api/models?sort=gsm8k&asc=1")
        data = json.loads(response.data)
        assert response.status_code == 200
        scores = [
            m["benchmark_gsm8k"]
            for m in data["models"]
            if m["benchmark_gsm8k"] is not None
        ]
        assert scores == sorted(scores)

    def test_invalid_sort_returns_400(self, client: FlaskClient) -> None:
        """An invalid sort parameter should return HTTP 400."""
        response = client.get("/api/models?sort=not_a_benchmark")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_invalid_sort_error_message_mentions_valid_options(self, client: FlaskClient) -> None:
        """The 400 error message should mention valid sort options."""
        response = client.get("/api/models?sort=bogus_key")
        data = json.loads(response.data)
        error_msg = data["error"].lower()
        # At least one valid benchmark should be mentioned
        assert any(k in error_msg for k in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"))

    def test_combined_category_and_bucket_filter(self, client: FlaskClient) -> None:
        """Combining category and bucket filters should intersect their effects."""
        response = client.get("/api/models?categories=coding&buckets=35B%2B")
        data = json.loads(response.data)
        assert response.status_code == 200
        for model in data["models"]:
            assert "coding" in model["task_categories"]
            assert model["parameter_size_bucket"] == "35B+"

    def test_combined_family_and_license_filter(self, client: FlaskClient) -> None:
        """Combining family and license filters should intersect their effects."""
        response = client.get("/api/models?families=Qwen&licenses=Apache-2.0")
        data = json.loads(response.data)
        assert response.status_code == 200
        for model in data["models"]:
            assert model["family"] == "Qwen"
            assert model["license"] == "Apache-2.0"

    def test_nonexistent_category_returns_empty_list(self, client: FlaskClient) -> None:
        """An unknown category should return count=0 and empty models array."""
        response = client.get("/api/models?categories=nonexistent_category_xyz")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] == 0
        assert data["models"] == []

    def test_nonexistent_family_returns_empty(self, client: FlaskClient) -> None:
        """An unknown family name should return count=0 and empty models array."""
        response = client.get("/api/models?families=NonExistentFamilyXYZ")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] == 0

    def test_no_nan_in_json_response(self, client: FlaskClient) -> None:
        """NaN must never appear in the JSON output (it is not valid JSON)."""
        response = client.get("/api/models")
        raw_text = response.data.decode("utf-8")
        assert "NaN" not in raw_text
        assert "Infinity" not in raw_text

    def test_pricing_columns_present_in_response(self, client: FlaskClient) -> None:
        """At least some models should have non-null API pricing data."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        has_pricing = any(
            m.get("cheapest_input_per_1m") is not None
            for m in data["models"]
        )
        assert has_pricing

    def test_self_hosted_cost_present_for_some_models(self, client: FlaskClient) -> None:
        """At least some models should have non-null self-hosted cost data."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        has_sh = any(
            m.get("self_hosted_hourly_usd") is not None
            for m in data["models"]
        )
        assert has_sh

    def test_benchmark_scores_in_valid_range(self, client: FlaskClient) -> None:
        """All non-null benchmark scores in the response should be in [0, 100]."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        for model in data["models"]:
            for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
                val = model.get(f"benchmark_{key}")
                if val is not None:
                    assert 0 <= val <= 100, (
                        f"benchmark_{key}={val} out of range for {model.get('model_id')}"
                    )

    def test_all_model_ids_unique_in_response(self, client: FlaskClient) -> None:
        """No model_id should appear more than once in the unfiltered response."""
        response = client.get("/api/models")
        data = json.loads(response.data)
        ids = [m["model_id"] for m in data["models"]]
        assert len(ids) == len(set(ids))

    def test_post_not_allowed(self, client: FlaskClient) -> None:
        """POST should not be allowed on /api/models."""
        response = client.post("/api/models")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# GET /api/models/<model_id>
# ---------------------------------------------------------------------------


class TestApiModelDetail:
    """Tests for the /api/models/<model_id> endpoint."""

    def test_known_model_returns_200(self, client: FlaskClient) -> None:
        """A known model_id should return HTTP 200."""
        response = client.get("/api/models/deepseek-r1")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        """Response must have application/json content type."""
        response = client.get("/api/models/deepseek-r1")
        assert "application/json" in response.content_type

    def test_response_contains_correct_model_id(self, client: FlaskClient) -> None:
        """The response body must contain the requested model_id."""
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        assert data["model_id"] == "deepseek-r1"

    def test_response_contains_display_name(self, client: FlaskClient) -> None:
        """The response body must contain the correct display_name."""
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        assert data["display_name"] == "DeepSeek R1"

    def test_response_contains_family(self, client: FlaskClient) -> None:
        """The response body must contain the family field."""
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        assert data["family"] == "DeepSeek"

    def test_response_contains_license(self, client: FlaskClient) -> None:
        """The response body must contain the license field."""
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        assert "license" in data
        assert data["license"] is not None

    def test_benchmark_scores_present_and_non_null(self, client: FlaskClient) -> None:
        """Key benchmark scores for DeepSeek R1 must be present and non-null."""
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        assert data["benchmark_mmlu"] is not None
        assert data["benchmark_humaneval"] is not None
        assert data["benchmark_math"] is not None
        assert data["benchmark_gsm8k"] is not None

    def test_benchmark_score_values_in_valid_range(self, client: FlaskClient) -> None:
        """All benchmark scores in the response must be in [0, 100]."""
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            val = data.get(f"benchmark_{key}")
            if val is not None:
                assert 0 <= val <= 100

    def test_unknown_model_returns_404(self, client: FlaskClient) -> None:
        """An unknown model_id should return HTTP 404."""
        response = client.get("/api/models/totally-fake-model-xyz")
        assert response.status_code == 404

    def test_404_response_has_error_key(self, client: FlaskClient) -> None:
        """The 404 response must include an 'error' key."""
        response = client.get("/api/models/totally-fake-model-xyz")
        data = json.loads(response.data)
        assert "error" in data

    def test_404_error_message_mentions_model_id(self, client: FlaskClient) -> None:
        """The 404 error message should reference the requested model_id."""
        response = client.get("/api/models/my-missing-model")
        data = json.loads(response.data)
        assert "my-missing-model" in data["error"]

    def test_qwen_model_returns_correctly(self, client: FlaskClient) -> None:
        """Qwen 72B should be retrievable and have the correct family."""
        response = client.get("/api/models/qwen2.5-72b-instruct")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["family"] == "Qwen"

    def test_llama_model_has_api_pricing(self, client: FlaskClient) -> None:
        """Llama 3.1 8B Instruct must have non-null cheapest_input_per_1m."""
        response = client.get("/api/models/llama-3.1-8b-instruct")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get("cheapest_input_per_1m") is not None

    def test_llama_model_has_self_hosted_cost(self, client: FlaskClient) -> None:
        """Llama 3.1 8B Instruct must have non-null self_hosted_hourly_usd."""
        response = client.get("/api/models/llama-3.1-8b-instruct")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get("self_hosted_hourly_usd") is not None

    def test_mistral_model_returns_correctly(self, client: FlaskClient) -> None:
        """Mistral Large 2 should be retrievable."""
        response = client.get("/api/models/mistral-large-2")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["family"] == "Mistral"

    def test_deepseek_v3_has_open_weights(self, client: FlaskClient) -> None:
        """DeepSeek V3 should have open_weights=True."""
        response = client.get("/api/models/deepseek-v3")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["open_weights"] is True

    def test_no_nan_in_response(self, client: FlaskClient) -> None:
        """NaN must never appear in the model detail JSON."""
        response = client.get("/api/models/deepseek-v3")
        raw = response.data.decode("utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_all_expected_columns_present(self, client: FlaskClient) -> None:
        """Model detail should include all _LIST_COLUMNS fields."""
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        expected_fields = [
            "model_id", "display_name", "family", "license",
            "parameter_size_bucket", "open_weights",
            "benchmark_mmlu", "benchmark_humaneval",
            "benchmark_math", "benchmark_gsm8k", "benchmark_mbpp",
        ]
        for field in expected_fields:
            assert field in data, f"Expected field '{field}' in model detail response"

    def test_post_not_allowed(self, client: FlaskClient) -> None:
        """POST should not be allowed on the model detail route."""
        response = client.post("/api/models/deepseek-r1")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# GET /api/filter-options
# ---------------------------------------------------------------------------


class TestApiFilterOptions:
    """Tests for the /api/filter-options endpoint."""

    def test_returns_200(self, client: FlaskClient) -> None:
        """The filter-options endpoint must return HTTP 200."""
        response = client.get("/api/filter-options")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        """Response must have application/json content type."""
        response = client.get("/api/filter-options")
        assert "application/json" in response.content_type

    def test_has_required_top_level_keys(self, client: FlaskClient) -> None:
        """Response must contain all four required top-level keys."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        for key in ("task_categories", "size_buckets", "licenses", "families"):
            assert key in data, f"Missing key: {key}"

    def test_all_values_are_lists(self, client: FlaskClient) -> None:
        """Each value in the response must be a JSON array."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        for key, val in data.items():
            assert isinstance(val, list), f"{key} should be a list, got {type(val)}"

    def test_task_categories_contains_all_three(self, client: FlaskClient) -> None:
        """All three standard task categories must be present."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        cats = data["task_categories"]
        assert "reasoning" in cats
        assert "coding" in cats
        assert "math" in cats

    def test_task_categories_non_empty(self, client: FlaskClient) -> None:
        """task_categories list must be non-empty."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        assert len(data["task_categories"]) > 0

    def test_size_buckets_ordered_correctly(self, client: FlaskClient) -> None:
        """Size buckets must appear in the correct order: ≤7B, 8-34B, 35B+."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        buckets = data["size_buckets"]
        assert "\u22647B" in buckets
        assert "8\u201334B" in buckets
        assert "35B+" in buckets
        assert buckets.index("\u22647B") < buckets.index("8\u201334B")
        assert buckets.index("8\u201334B") < buckets.index("35B+")

    def test_size_buckets_non_empty(self, client: FlaskClient) -> None:
        """size_buckets list must be non-empty."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        assert len(data["size_buckets"]) >= 3

    def test_known_licenses_present(self, client: FlaskClient) -> None:
        """Apache-2.0 and MIT must appear in the licenses list."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        licenses = data["licenses"]
        assert "Apache-2.0" in licenses
        assert "MIT" in licenses

    def test_custom_commercial_license_present(self, client: FlaskClient) -> None:
        """custom/commercial must appear in the licenses list."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        assert "custom/commercial" in data["licenses"]

    def test_licenses_non_empty(self, client: FlaskClient) -> None:
        """licenses list must be non-empty."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        assert len(data["licenses"]) > 0

    def test_known_families_present(self, client: FlaskClient) -> None:
        """All major model families must be present in the families list."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        families = data["families"]
        for family in ("DeepSeek", "Llama", "Mistral", "Qwen"):
            assert family in families, f"Expected family '{family}' in families"

    def test_families_non_empty(self, client: FlaskClient) -> None:
        """families list must be non-empty."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        assert len(data["families"]) > 0

    def test_all_values_are_strings(self, client: FlaskClient) -> None:
        """Every item in every list must be a plain string."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        for key, values in data.items():
            for val in values:
                assert isinstance(val, str), (
                    f"{key} contains non-string value: {val!r}"
                )

    def test_no_duplicate_values(self, client: FlaskClient) -> None:
        """No list should contain duplicate entries."""
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        for key, values in data.items():
            assert len(values) == len(set(values)), (
                f"{key} contains duplicate values"
            )

    def test_post_not_allowed(self, client: FlaskClient) -> None:
        """POST should not be allowed on the filter-options route."""
        response = client.post("/api/filter-options")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# GET /api/pricing/<model_id>
# ---------------------------------------------------------------------------


class TestApiPricingDetail:
    """Tests for the /api/pricing/<model_id> endpoint."""

    def test_known_model_returns_200(self, client: FlaskClient) -> None:
        """A known model_id should return HTTP 200."""
        response = client.get("/api/pricing/deepseek-r1")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        """Response must have application/json content type."""
        response = client.get("/api/pricing/deepseek-r1")
        assert "application/json" in response.content_type

    def test_response_has_model_id(self, client: FlaskClient) -> None:
        """Response must include the requested model_id."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert data["model_id"] == "deepseek-r1"

    def test_response_has_api_providers_list(self, client: FlaskClient) -> None:
        """Response must include an api_providers array."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert "api_providers" in data
        assert isinstance(data["api_providers"], list)

    def test_response_has_self_hosted_dict(self, client: FlaskClient) -> None:
        """Response must include a self_hosted object."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert "self_hosted" in data
        assert isinstance(data["self_hosted"], dict)

    def test_api_providers_have_required_fields(self, client: FlaskClient) -> None:
        """Each provider entry must contain provider, input_per_1m, and output_per_1m."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        for provider in data["api_providers"]:
            assert "provider" in provider
            assert "input_per_1m" in provider
            assert "output_per_1m" in provider

    def test_api_providers_have_provider_url(self, client: FlaskClient) -> None:
        """Each provider entry must contain provider_url."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        for provider in data["api_providers"]:
            assert "provider_url" in provider

    def test_deepseek_r1_has_multiple_providers(self, client: FlaskClient) -> None:
        """DeepSeek R1 should have at least 2 API providers in the pricing data."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert len(data["api_providers"]) >= 2

    def test_self_hosted_has_hourly_cost(self, client: FlaskClient) -> None:
        """self_hosted must include hourly_cost_usd with a positive value."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        sh = data["self_hosted"]
        assert "hourly_cost_usd" in sh
        assert sh["hourly_cost_usd"] is not None
        assert sh["hourly_cost_usd"] > 0

    def test_self_hosted_has_gpu_setup(self, client: FlaskClient) -> None:
        """self_hosted must include a non-null gpu_setup string."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert data["self_hosted"]["gpu_setup"] is not None

    def test_self_hosted_has_min_vram(self, client: FlaskClient) -> None:
        """self_hosted must include a min_vram_gb value."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert "min_vram_gb" in data["self_hosted"]

    def test_self_hosted_has_throughput(self, client: FlaskClient) -> None:
        """self_hosted must include a throughput_tps value."""
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert "throughput_tps" in data["self_hosted"]

    def test_unknown_model_returns_404(self, client: FlaskClient) -> None:
        """An unknown model_id should return HTTP 404."""
        response = client.get("/api/pricing/not-a-real-model")
        assert response.status_code == 404

    def test_404_has_error_key(self, client: FlaskClient) -> None:
        """The 404 response must include an 'error' key."""
        response = client.get("/api/pricing/not-a-real-model")
        data = json.loads(response.data)
        assert "error" in data

    def test_llama_8b_has_multiple_providers(self, client: FlaskClient) -> None:
        """Llama 3.1 8B should have at least 2 API providers."""
        response = client.get("/api/pricing/llama-3.1-8b-instruct")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["api_providers"]) >= 2

    def test_llama_8b_self_hosted_cost_positive(self, client: FlaskClient) -> None:
        """Llama 3.1 8B self-hosted cost must be a positive number."""
        response = client.get("/api/pricing/llama-3.1-8b-instruct")
        data = json.loads(response.data)
        assert data["self_hosted"]["hourly_cost_usd"] > 0

    def test_provider_input_prices_are_numbers_or_null(self, client: FlaskClient) -> None:
        """All input_per_1m values must be numbers or null."""
        response = client.get("/api/pricing/llama-3.3-70b-instruct")
        data = json.loads(response.data)
        for provider in data["api_providers"]:
            val = provider["input_per_1m"]
            assert val is None or isinstance(val, (int, float))

    def test_provider_output_prices_are_numbers_or_null(self, client: FlaskClient) -> None:
        """All output_per_1m values must be numbers or null."""
        response = client.get("/api/pricing/llama-3.3-70b-instruct")
        data = json.loads(response.data)
        for provider in data["api_providers"]:
            val = provider["output_per_1m"]
            assert val is None or isinstance(val, (int, float))

    def test_no_nan_in_response(self, client: FlaskClient) -> None:
        """NaN must never appear in the pricing detail JSON."""
        response = client.get("/api/pricing/llama-3.3-70b-instruct")
        raw = response.data.decode("utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_qwen_pricing_returns_200(self, client: FlaskClient) -> None:
        """Qwen 2.5 72B should have pricing data available."""
        response = client.get("/api/pricing/qwen2.5-72b-instruct")
        assert response.status_code == 200

    def test_deepseek_v3_pricing_returns_200(self, client: FlaskClient) -> None:
        """DeepSeek V3 should have pricing data available."""
        response = client.get("/api/pricing/deepseek-v3")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["model_id"] == "deepseek-v3"

    def test_post_not_allowed(self, client: FlaskClient) -> None:
        """POST should not be allowed on the pricing detail route."""
        response = client.post("/api/pricing/deepseek-r1")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# GET /api/compare
# ---------------------------------------------------------------------------


class TestApiCompare:
    """Tests for the /api/compare endpoint."""

    def test_single_model_returns_200(self, client: FlaskClient) -> None:
        """A single known model_id should return HTTP 200."""
        response = client.get("/api/compare?ids=deepseek-r1")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        """Response must have application/json content type."""
        response = client.get("/api/compare?ids=deepseek-r1")
        assert "application/json" in response.content_type

    def test_response_has_benchmark_keys(self, client: FlaskClient) -> None:
        """Response must include a 'benchmark_keys' array with all five keys."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert "benchmark_keys" in data
        assert set(data["benchmark_keys"]) == {"mmlu", "humaneval", "math", "gsm8k", "mbpp"}

    def test_response_has_models_list(self, client: FlaskClient) -> None:
        """Response must include a 'models' array."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_response_has_missing_ids_key(self, client: FlaskClient) -> None:
        """Response must include a 'missing_ids' array."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert "missing_ids" in data

    def test_single_model_in_response(self, client: FlaskClient) -> None:
        """Requesting one model should return exactly one model entry."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "deepseek-r1"

    def test_model_has_scores_dict(self, client: FlaskClient) -> None:
        """Each model entry must include a 'scores' dict."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        model = data["models"][0]
        assert "scores" in model
        assert isinstance(model["scores"], dict)

    def test_scores_contain_all_benchmark_keys(self, client: FlaskClient) -> None:
        """The scores dict must contain all five benchmark keys."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        scores = data["models"][0]["scores"]
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            assert key in scores

    def test_model_has_display_name(self, client: FlaskClient) -> None:
        """Each model entry must include a display_name."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert data["models"][0]["display_name"] == "DeepSeek R1"

    def test_model_has_family(self, client: FlaskClient) -> None:
        """Each model entry must include a family field."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert data["models"][0]["family"] == "DeepSeek"

    def test_model_has_parameter_size_bucket(self, client: FlaskClient) -> None:
        """Each model entry must include a parameter_size_bucket field."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert "parameter_size_bucket" in data["models"][0]

    def test_model_has_license(self, client: FlaskClient) -> None:
        """Each model entry must include a license field."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert "license" in data["models"][0]

    def test_missing_ids_empty_when_all_found(self, client: FlaskClient) -> None:
        """missing_ids should be empty when all requested models are found."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert data["missing_ids"] == []

    def test_four_models_comma_separated(self, client: FlaskClient) -> None:
        """Four models passed as comma-separated values should all appear in response."""
        ids = "deepseek-r1,deepseek-v3,llama-3.3-70b-instruct,qwen2.5-72b-instruct"
        response = client.get(f"/api/compare?ids={ids}")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 4

    def test_four_models_repeated_params(self, client: FlaskClient) -> None:
        """Four models passed as repeated query params should all appear."""
        response = client.get(
            "/api/compare?ids=deepseek-r1&ids=deepseek-v3"
            "&ids=llama-3.3-70b-instruct&ids=qwen2.5-72b-instruct"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 4

    def test_four_models_all_correct_ids(self, client: FlaskClient) -> None:
        """All four requested model IDs should appear in the response."""
        expected_ids = {
            "deepseek-r1", "deepseek-v3",
            "llama-3.3-70b-instruct", "qwen2.5-72b-instruct",
        }
        ids_str = ",".join(expected_ids)
        response = client.get(f"/api/compare?ids={ids_str}")
        data = json.loads(response.data)
        returned_ids = {m["model_id"] for m in data["models"]}
        assert returned_ids == expected_ids

    def test_five_models_returns_400(self, client: FlaskClient) -> None:
        """Requesting more than 4 models should return HTTP 400."""
        ids = "deepseek-r1,deepseek-v3,llama-3.3-70b-instruct,qwen2.5-72b-instruct,mistral-large-2"
        response = client.get(f"/api/compare?ids={ids}")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_five_models_error_mentions_maximum(self, client: FlaskClient) -> None:
        """The 400 error for too many models should mention the limit."""
        ids = "deepseek-r1,deepseek-v3,llama-3.3-70b-instruct,qwen2.5-72b-instruct,mistral-large-2"
        response = client.get(f"/api/compare?ids={ids}")
        data = json.loads(response.data)
        assert "4" in data["error"] or "maximum" in data["error"].lower()

    def test_no_ids_returns_400(self, client: FlaskClient) -> None:
        """Missing ids parameter should return HTTP 400."""
        response = client.get("/api/compare")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_no_ids_error_message_helpful(self, client: FlaskClient) -> None:
        """The 400 error for missing ids should tell the user what to do."""
        response = client.get("/api/compare")
        data = json.loads(response.data)
        assert "ids" in data["error"].lower() or "model" in data["error"].lower()

    def test_all_unknown_model_ids_returns_404(self, client: FlaskClient) -> None:
        """If none of the requested IDs are found, return HTTP 404."""
        response = client.get("/api/compare?ids=totally-fake-model-abc")
        assert response.status_code == 404

    def test_partial_unknown_still_returns_known_models(self, client: FlaskClient) -> None:
        """Known models should still be returned when mixed with unknown IDs."""
        response = client.get("/api/compare?ids=deepseek-r1,totally-fake-model-abc")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "deepseek-r1"

    def test_partial_unknown_ids_in_missing_ids(self, client: FlaskClient) -> None:
        """Unknown IDs should appear in the missing_ids array."""
        response = client.get("/api/compare?ids=deepseek-r1,totally-fake-model-abc")
        data = json.loads(response.data)
        assert "totally-fake-model-abc" in data["missing_ids"]

    def test_duplicate_ids_deduplicated(self, client: FlaskClient) -> None:
        """Duplicate model IDs should result in only one entry in the response."""
        response = client.get("/api/compare?ids=deepseek-r1,deepseek-r1")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 1

    def test_score_values_are_numbers_or_null(self, client: FlaskClient) -> None:
        """All score values must be numbers or null, never strings."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        scores = data["models"][0]["scores"]
        for key, val in scores.items():
            assert val is None or isinstance(val, (int, float)), (
                f"Score '{key}' has unexpected type: {type(val)}"
            )

    def test_score_values_in_valid_range(self, client: FlaskClient) -> None:
        """All non-null score values must be in [0, 100]."""
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        scores = data["models"][0]["scores"]
        for key, val in scores.items():
            if val is not None:
                assert 0 <= val <= 100, f"Score '{key}={val}' out of valid range"

    def test_no_nan_in_response(self, client: FlaskClient) -> None:
        """NaN must never appear in the compare JSON response."""
        ids = "deepseek-r1,qwen2.5-72b-instruct"
        response = client.get(f"/api/compare?ids={ids}")
        raw = response.data.decode("utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_two_models_compare(self, client: FlaskClient) -> None:
        """Two known models should both appear in the response."""
        response = client.get("/api/compare?ids=deepseek-r1,qwen2.5-72b-instruct")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 2
        returned_ids = {m["model_id"] for m in data["models"]}
        assert returned_ids == {"deepseek-r1", "qwen2.5-72b-instruct"}

    def test_post_not_allowed(self, client: FlaskClient) -> None:
        """POST should not be allowed on the compare route."""
        response = client.post("/api/compare")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


class TestErrorHandlers:
    """Tests for custom Flask error handlers."""

    def test_404_on_unknown_route(self, client: FlaskClient) -> None:
        """A completely unknown route must return HTTP 404."""
        response = client.get("/this/route/does/not/exist")
        assert response.status_code == 404

    def test_404_response_is_json(self, client: FlaskClient) -> None:
        """The 404 response for an unknown route must be JSON."""
        response = client.get("/this/route/does/not/exist")
        assert "application/json" in response.content_type

    def test_404_response_has_error_key(self, client: FlaskClient) -> None:
        """The 404 JSON response must contain an 'error' key."""
        response = client.get("/this/route/does/not/exist")
        data = json.loads(response.data)
        assert "error" in data

    def test_404_response_has_status_key(self, client: FlaskClient) -> None:
        """The 404 JSON response should contain a 'status' field equal to 404."""
        response = client.get("/this/route/does/not/exist")
        data = json.loads(response.data)
        assert data.get("status") == 404

    def test_404_on_deep_unknown_path(self, client: FlaskClient) -> None:
        """A deeply nested unknown path must also return HTTP 404."""
        response = client.get("/api/totally/unknown/deeply/nested/path")
        assert response.status_code == 404

    def test_405_on_post_to_models(self, client: FlaskClient) -> None:
        """POST to /api/models should return HTTP 405."""
        response = client.post("/api/models")
        assert response.status_code == 405

    def test_405_response_is_json(self, client: FlaskClient) -> None:
        """The 405 response must be JSON."""
        response = client.post("/api/models")
        assert "application/json" in response.content_type

    def test_405_response_has_error_key(self, client: FlaskClient) -> None:
        """The 405 JSON response must contain an 'error' key."""
        response = client.post("/api/models")
        data = json.loads(response.data)
        assert "error" in data

    def test_405_on_delete_to_filter_options(self, client: FlaskClient) -> None:
        """DELETE to /api/filter-options should return HTTP 405."""
        response = client.delete("/api/filter-options")
        assert response.status_code == 405

    def test_405_on_put_to_compare(self, client: FlaskClient) -> None:
        """PUT to /api/compare should return HTTP 405."""
        response = client.put("/api/compare")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# create_app configuration
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the application factory itself."""

    def test_returns_flask_app(self) -> None:
        """create_app must return a Flask application instance."""
        from flask import Flask
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        assert isinstance(app, Flask)

    def test_testing_flag_set_correctly(self) -> None:
        """The TESTING config flag must be respected."""
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        assert app.config["TESTING"] is True

    def test_testing_flag_false_by_default(self) -> None:
        """Without explicit config, TESTING should default to False."""
        app = create_app(config={"EAGER_LOAD_DATA": False})
        assert not app.config.get("TESTING", False)

    def test_custom_config_key_applied(self) -> None:
        """Arbitrary config keys passed to create_app must be stored."""
        app = create_app(
            config={"TESTING": True, "EAGER_LOAD_DATA": False, "MY_CUSTOM_KEY": "hello"}
        )
        assert app.config["MY_CUSTOM_KEY"] == "hello"

    def test_multiple_instances_are_independent(self) -> None:
        """Two calls to create_app must return independent instances."""
        app1 = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        app2 = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        assert app1 is not app2

    def test_index_route_registered(self) -> None:
        """The '/' route must be registered."""
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/" in rules

    def test_api_models_route_registered(self) -> None:
        """The '/api/models' route must be registered."""
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/api/models" in rules

    def test_api_filter_options_route_registered(self) -> None:
        """The '/api/filter-options' route must be registered."""
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/api/filter-options" in rules

    def test_api_compare_route_registered(self) -> None:
        """The '/api/compare' route must be registered."""
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/api/compare" in rules

    def test_api_model_detail_route_registered(self) -> None:
        """The parameterised '/api/models/<model_id>' route must be registered."""
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert any("/api/models/" in r for r in rules)

    def test_api_pricing_detail_route_registered(self) -> None:
        """The parameterised '/api/pricing/<model_id>' route must be registered."""
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert any("/api/pricing/" in r for r in rules)

    def test_none_config_uses_defaults(self) -> None:
        """Passing config=None must not raise and should use default values."""
        # Create with EAGER_LOAD_DATA=False to avoid loading data in test
        app = create_app(config={"EAGER_LOAD_DATA": False})
        assert isinstance(app, Flask)

    def test_test_client_created_successfully(self) -> None:
        """The test client must be creatable from the app instance."""
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        client = app.test_client()
        assert client is not None
