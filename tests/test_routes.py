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
        response = client.get("/")
        assert response.status_code == 200

    def test_content_type_is_html(self, client: FlaskClient) -> None:
        response = client.get("/")
        assert "text/html" in response.content_type

    def test_response_body_not_empty(self, client: FlaskClient) -> None:
        response = client.get("/")
        assert len(response.data) > 0

    def test_body_contains_llm_bench(self, client: FlaskClient) -> None:
        response = client.get("/")
        # The page should mention the application name or benchmark terms
        body = response.data.decode("utf-8", errors="replace").lower()
        # At minimum the HTML skeleton should be present
        assert "<html" in body or "<!doctype" in body


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------


class TestApiModels:
    """Tests for the /api/models endpoint."""

    def test_returns_200(self, client: FlaskClient) -> None:
        response = client.get("/api/models")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        response = client.get("/api/models")
        assert "application/json" in response.content_type

    def test_response_has_count_and_models_keys(self, client: FlaskClient) -> None:
        response = client.get("/api/models")
        data = json.loads(response.data)
        assert "count" in data
        assert "models" in data

    def test_models_is_list(self, client: FlaskClient) -> None:
        response = client.get("/api/models")
        data = json.loads(response.data)
        assert isinstance(data["models"], list)

    def test_count_matches_models_length(self, client: FlaskClient) -> None:
        response = client.get("/api/models")
        data = json.loads(response.data)
        assert data["count"] == len(data["models"])

    def test_at_least_20_models_returned(self, client: FlaskClient) -> None:
        response = client.get("/api/models")
        data = json.loads(response.data)
        assert data["count"] >= 20

    def test_each_model_has_required_fields(self, client: FlaskClient) -> None:
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
                assert field in model, f"Field '{field}' missing from model {model.get('model_id')}"

    def test_each_model_has_benchmark_keys(self, client: FlaskClient) -> None:
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
                assert field in model

    def test_filter_by_category_coding(self, client: FlaskClient) -> None:
        response = client.get("/api/models?categories=coding")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert "coding" in model["task_categories"]

    def test_filter_by_category_math(self, client: FlaskClient) -> None:
        response = client.get("/api/models?categories=math")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert "math" in model["task_categories"]

    def test_filter_by_multiple_categories_comma_separated(self, client: FlaskClient) -> None:
        response = client.get("/api/models?categories=coding,math")
        data_combined = json.loads(response.data)
        response_single = client.get("/api/models?categories=coding")
        data_single = json.loads(response_single.data)
        # Combined should have >= single
        assert data_combined["count"] >= data_single["count"]

    def test_filter_by_multiple_categories_repeated_params(self, client: FlaskClient) -> None:
        response = client.get("/api/models?categories=coding&categories=math")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["count"] > 0

    def test_filter_by_size_bucket_small(self, client: FlaskClient) -> None:
        response = client.get("/api/models?buckets=%E2%89%A47B")  # URL-encoded ≤7B
        assert response.status_code == 200
        data = json.loads(response.data)
        for model in data["models"]:
            assert model["parameter_size_bucket"] == "≤7B"

    def test_filter_by_size_bucket_large(self, client: FlaskClient) -> None:
        response = client.get("/api/models?buckets=35B%2B")  # URL-encoded 35B+
        assert response.status_code == 200
        data = json.loads(response.data)
        for model in data["models"]:
            assert model["parameter_size_bucket"] == "35B+"

    def test_filter_by_license_apache(self, client: FlaskClient) -> None:
        response = client.get("/api/models?licenses=Apache-2.0")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert model["license"] == "Apache-2.0"

    def test_filter_by_license_mit(self, client: FlaskClient) -> None:
        response = client.get("/api/models?licenses=MIT")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] > 0
        for model in data["models"]:
            assert model["license"] == "MIT"

    def test_filter_by_family(self, client: FlaskClient) -> None:
        response = client.get("/api/models?families=DeepSeek")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] >= 3  # V2, V3, R1
        for model in data["models"]:
            assert model["family"] == "DeepSeek"

    def test_filter_by_open_weights(self, client: FlaskClient) -> None:
        response = client.get("/api/models?open_weights=true")
        data = json.loads(response.data)
        assert response.status_code == 200
        for model in data["models"]:
            assert model["open_weights"] is True

    def test_sort_by_mmlu(self, client: FlaskClient) -> None:
        response = client.get("/api/models?sort=mmlu")
        data = json.loads(response.data)
        assert response.status_code == 200
        scores = [
            m["benchmark_mmlu"]
            for m in data["models"]
            if m["benchmark_mmlu"] is not None
        ]
        assert scores == sorted(scores, reverse=True)

    def test_sort_ascending(self, client: FlaskClient) -> None:
        response = client.get("/api/models?sort=mmlu&asc=true")
        data = json.loads(response.data)
        assert response.status_code == 200
        scores = [
            m["benchmark_mmlu"]
            for m in data["models"]
            if m["benchmark_mmlu"] is not None
        ]
        assert scores == sorted(scores)

    def test_invalid_sort_returns_400(self, client: FlaskClient) -> None:
        response = client.get("/api/models?sort=not_a_benchmark")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_combined_category_and_bucket_filter(self, client: FlaskClient) -> None:
        response = client.get("/api/models?categories=coding&buckets=35B%2B")
        data = json.loads(response.data)
        assert response.status_code == 200
        for model in data["models"]:
            assert "coding" in model["task_categories"]
            assert model["parameter_size_bucket"] == "35B+"

    def test_nonexistent_category_returns_empty_list(self, client: FlaskClient) -> None:
        response = client.get("/api/models?categories=nonexistent_category")
        data = json.loads(response.data)
        assert response.status_code == 200
        assert data["count"] == 0
        assert data["models"] == []

    def test_no_nan_in_json_response(self, client: FlaskClient) -> None:
        """NaN should never appear in the JSON output (it is invalid JSON)."""
        response = client.get("/api/models")
        raw_text = response.data.decode("utf-8")
        assert "NaN" not in raw_text
        assert "Infinity" not in raw_text

    def test_pricing_columns_present_in_response(self, client: FlaskClient) -> None:
        response = client.get("/api/models")
        data = json.loads(response.data)
        # At least some models should have pricing info
        has_pricing = any(
            m.get("cheapest_input_per_1m") is not None
            for m in data["models"]
        )
        assert has_pricing

    def test_self_hosted_cost_present_for_some_models(self, client: FlaskClient) -> None:
        response = client.get("/api/models")
        data = json.loads(response.data)
        has_sh = any(
            m.get("self_hosted_hourly_usd") is not None
            for m in data["models"]
        )
        assert has_sh


# ---------------------------------------------------------------------------
# GET /api/models/<model_id>
# ---------------------------------------------------------------------------


class TestApiModelDetail:
    """Tests for the /api/models/<model_id> endpoint."""

    def test_known_model_returns_200(self, client: FlaskClient) -> None:
        response = client.get("/api/models/deepseek-r1")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        response = client.get("/api/models/deepseek-r1")
        assert "application/json" in response.content_type

    def test_response_contains_model_id(self, client: FlaskClient) -> None:
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        assert data["model_id"] == "deepseek-r1"

    def test_response_contains_display_name(self, client: FlaskClient) -> None:
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        assert data["display_name"] == "DeepSeek R1"

    def test_benchmark_scores_present(self, client: FlaskClient) -> None:
        response = client.get("/api/models/deepseek-r1")
        data = json.loads(response.data)
        assert data["benchmark_mmlu"] is not None
        assert data["benchmark_humaneval"] is not None

    def test_unknown_model_returns_404(self, client: FlaskClient) -> None:
        response = client.get("/api/models/totally-fake-model-xyz")
        assert response.status_code == 404

    def test_404_response_has_error_key(self, client: FlaskClient) -> None:
        response = client.get("/api/models/totally-fake-model-xyz")
        data = json.loads(response.data)
        assert "error" in data

    def test_qwen_model_returns_correctly(self, client: FlaskClient) -> None:
        response = client.get("/api/models/qwen2.5-72b-instruct")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["family"] == "Qwen"

    def test_llama_model_has_pricing(self, client: FlaskClient) -> None:
        response = client.get("/api/models/llama-3.1-8b-instruct")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data.get("cheapest_input_per_1m") is not None

    def test_no_nan_in_response(self, client: FlaskClient) -> None:
        response = client.get("/api/models/deepseek-v3")
        raw = response.data.decode("utf-8")
        assert "NaN" not in raw


# ---------------------------------------------------------------------------
# GET /api/filter-options
# ---------------------------------------------------------------------------


class TestApiFilterOptions:
    """Tests for the /api/filter-options endpoint."""

    def test_returns_200(self, client: FlaskClient) -> None:
        response = client.get("/api/filter-options")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        response = client.get("/api/filter-options")
        assert "application/json" in response.content_type

    def test_has_required_top_level_keys(self, client: FlaskClient) -> None:
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        for key in ("task_categories", "size_buckets", "licenses", "families"):
            assert key in data, f"Missing key: {key}"

    def test_task_categories_are_lists(self, client: FlaskClient) -> None:
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        assert isinstance(data["task_categories"], list)

    def test_known_task_categories_present(self, client: FlaskClient) -> None:
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        cats = data["task_categories"]
        assert "reasoning" in cats
        assert "coding" in cats
        assert "math" in cats

    def test_size_buckets_ordered(self, client: FlaskClient) -> None:
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        buckets = data["size_buckets"]
        assert "≤7B" in buckets
        assert "8–34B" in buckets
        assert "35B+" in buckets
        assert buckets.index("≤7B") < buckets.index("35B+")

    def test_known_licenses_present(self, client: FlaskClient) -> None:
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        licenses = data["licenses"]
        assert "Apache-2.0" in licenses
        assert "MIT" in licenses

    def test_known_families_present(self, client: FlaskClient) -> None:
        response = client.get("/api/filter-options")
        data = json.loads(response.data)
        families = data["families"]
        for family in ("DeepSeek", "Llama", "Mistral", "Qwen"):
            assert family in families


# ---------------------------------------------------------------------------
# GET /api/pricing/<model_id>
# ---------------------------------------------------------------------------


class TestApiPricingDetail:
    """Tests for the /api/pricing/<model_id> endpoint."""

    def test_known_model_returns_200(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        assert "application/json" in response.content_type

    def test_response_has_model_id(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert data["model_id"] == "deepseek-r1"

    def test_response_has_api_providers_list(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert "api_providers" in data
        assert isinstance(data["api_providers"], list)

    def test_response_has_self_hosted(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert "self_hosted" in data
        assert isinstance(data["self_hosted"], dict)

    def test_api_providers_have_required_fields(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        for provider in data["api_providers"]:
            assert "provider" in provider
            assert "input_per_1m" in provider
            assert "output_per_1m" in provider

    def test_deepseek_r1_has_multiple_providers(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert len(data["api_providers"]) >= 2

    def test_self_hosted_has_hourly_cost(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        sh = data["self_hosted"]
        assert "hourly_cost_usd" in sh
        assert sh["hourly_cost_usd"] is not None
        assert sh["hourly_cost_usd"] > 0

    def test_self_hosted_has_gpu_setup(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/deepseek-r1")
        data = json.loads(response.data)
        assert data["self_hosted"]["gpu_setup"] is not None

    def test_unknown_model_returns_404(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/not-a-real-model")
        assert response.status_code == 404

    def test_404_has_error_key(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/not-a-real-model")
        data = json.loads(response.data)
        assert "error" in data

    def test_llama_8b_pricing(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/llama-3.1-8b-instruct")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["api_providers"]) >= 2

    def test_no_nan_in_response(self, client: FlaskClient) -> None:
        response = client.get("/api/pricing/llama-3.3-70b-instruct")
        raw = response.data.decode("utf-8")
        assert "NaN" not in raw


# ---------------------------------------------------------------------------
# GET /api/compare
# ---------------------------------------------------------------------------


class TestApiCompare:
    """Tests for the /api/compare endpoint."""

    def test_single_model_returns_200(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        assert "application/json" in response.content_type

    def test_response_has_benchmark_keys(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert "benchmark_keys" in data
        assert set(data["benchmark_keys"]) == {"mmlu", "humaneval", "math", "gsm8k", "mbpp"}

    def test_response_has_models_list(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_single_model_in_response(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "deepseek-r1"

    def test_model_has_scores_dict(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        model = data["models"][0]
        assert "scores" in model
        assert isinstance(model["scores"], dict)

    def test_scores_contain_all_benchmark_keys(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        scores = data["models"][0]["scores"]
        for key in ("mmlu", "humaneval", "math", "gsm8k", "mbpp"):
            assert key in scores

    def test_four_models_compare(self, client: FlaskClient) -> None:
        ids = "deepseek-r1,deepseek-v3,llama-3.3-70b-instruct,qwen2.5-72b-instruct"
        response = client.get(f"/api/compare?ids={ids}")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 4

    def test_four_models_repeated_param(self, client: FlaskClient) -> None:
        response = client.get(
            "/api/compare?ids=deepseek-r1&ids=deepseek-v3"
            "&ids=llama-3.3-70b-instruct&ids=qwen2.5-72b-instruct"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 4

    def test_five_models_returns_400(self, client: FlaskClient) -> None:
        ids = "deepseek-r1,deepseek-v3,llama-3.3-70b-instruct,qwen2.5-72b-instruct,mistral-large-2"
        response = client.get(f"/api/compare?ids={ids}")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_no_ids_returns_400(self, client: FlaskClient) -> None:
        response = client.get("/api/compare")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_unknown_model_returns_404(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=totally-fake-model-abc")
        assert response.status_code == 404

    def test_partial_unknown_still_returns_known(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1,totally-fake-model-abc")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == "deepseek-r1"
        assert "totally-fake-model-abc" in data["missing_ids"]

    def test_duplicate_ids_deduplicated(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1,deepseek-r1")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data["models"]) == 1

    def test_model_has_display_name(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert data["models"][0]["display_name"] == "DeepSeek R1"

    def test_no_nan_in_response(self, client: FlaskClient) -> None:
        ids = "deepseek-r1,qwen2.5-72b-instruct"
        response = client.get(f"/api/compare?ids={ids}")
        raw = response.data.decode("utf-8")
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_response_has_missing_ids_key(self, client: FlaskClient) -> None:
        response = client.get("/api/compare?ids=deepseek-r1")
        data = json.loads(response.data)
        assert "missing_ids" in data
        assert data["missing_ids"] == []


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


class TestErrorHandlers:
    """Tests for custom error handlers."""

    def test_404_on_unknown_route(self, client: FlaskClient) -> None:
        response = client.get("/this/route/does/not/exist")
        assert response.status_code == 404

    def test_404_response_is_json(self, client: FlaskClient) -> None:
        response = client.get("/this/route/does/not/exist")
        assert "application/json" in response.content_type

    def test_404_response_has_error_key(self, client: FlaskClient) -> None:
        response = client.get("/this/route/does/not/exist")
        data = json.loads(response.data)
        assert "error" in data

    def test_405_on_wrong_method(self, client: FlaskClient) -> None:
        response = client.post("/api/models")  # POST not allowed
        assert response.status_code == 405

    def test_405_response_is_json(self, client: FlaskClient) -> None:
        response = client.post("/api/models")
        assert "application/json" in response.content_type


# ---------------------------------------------------------------------------
# create_app configuration
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the application factory itself."""

    def test_returns_flask_app(self) -> None:
        from flask import Flask
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        assert isinstance(app, Flask)

    def test_testing_flag_set(self) -> None:
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        assert app.config["TESTING"] is True

    def test_custom_config_applied(self) -> None:
        app = create_app(
            config={"TESTING": True, "EAGER_LOAD_DATA": False, "MY_CUSTOM_KEY": "hello"}
        )
        assert app.config["MY_CUSTOM_KEY"] == "hello"

    def test_multiple_instances_are_independent(self) -> None:
        app1 = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        app2 = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        assert app1 is not app2

    def test_all_api_routes_registered(self) -> None:
        app = create_app(config={"TESTING": True, "EAGER_LOAD_DATA": False})
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert "/" in rules
        assert "/api/models" in rules
        assert "/api/filter-options" in rules
        assert "/api/compare" in rules
        # Check parameterised routes are present
        parameterised = {rule.rule for rule in app.url_map.iter_rules()}
        assert any("/api/models/" in r for r in parameterised)
        assert any("/api/pricing/" in r for r in parameterised)
