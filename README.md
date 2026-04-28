# LLM Bench Compare

> Compare open-source LLMs by benchmark performance and cost — instantly.

LLM Bench Compare is a lightweight web application that aggregates and visualizes up-to-date benchmark scores (MMLU, HumanEval, MATH, GSM8K, MBPP) for 20+ major open-source LLMs including DeepSeek, Kimi, Llama 3, Mistral, and Qwen variants. Filter models by task category, parameter size, and license type, then compare them side-by-side in a sortable table with an interactive radar chart. A cost-per-token overlay surfaces both API pricing and self-hosted GPU estimates so you can make informed capability-vs-cost trade-offs for your specific use case.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/llm_bench_compare.git
cd llm_bench_compare

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the development server
python -m llm_bench_compare
```

Open your browser at **http://localhost:5000** — the comparison table loads immediately with no external API keys required.

---

## Features

- **Side-by-side benchmark table** — Sortable columns for MMLU, HumanEval, MATH, GSM8K, and MBPP across 20+ models. Click any column header to rank models instantly.
- **Multi-dimensional filtering** — Narrow results by task category (reasoning, coding, math), model size bucket (≤7B, 8–34B, 35B+), and license type (Apache-2.0, MIT, custom/commercial).
- **Cost-per-token overlay** — Compare API pricing (USD per 1M tokens) and estimated self-hosted GPU hourly cost side-by-side with benchmark scores.
- **Interactive radar chart** — Select up to 4 models to render a Chart.js radar chart for instant visual capability fingerprinting across all benchmark dimensions.
- **JSON-driven data store** — Add new models or refresh benchmark scores by editing `benchmarks.json` and `pricing.json` — no application code changes needed.

---

## Usage Examples

### Launch the app

```bash
python -m llm_bench_compare
# Flask development server starts on http://127.0.0.1:5000
```

### REST API

The app exposes a JSON API you can query directly:

```bash
# List all models (with optional filters)
curl "http://localhost:5000/api/models"

# Filter by task category and size bucket
curl "http://localhost:5000/api/models?task_category=coding&size_bucket=%E2%89%A47B"

# Get benchmark + pricing detail for a single model
curl "http://localhost:5000/api/models/deepseek-v3"

# Get pricing breakdown for a model
curl "http://localhost:5000/api/pricing/llama-3-70b"

# Compare up to 4 models for the radar chart
curl "http://localhost:5000/api/compare?models=deepseek-v3,llama-3-70b,mistral-large,qwen2-72b"

# Available filter options (task categories, size buckets, licenses)
curl "http://localhost:5000/api/filter-options"
```

### Example API response — `/api/models/deepseek-v3`

```json
{
  "model_id": "deepseek-v3",
  "display_name": "DeepSeek V3",
  "family": "DeepSeek",
  "size_bucket": "35B+",
  "license": "custom",
  "task_categories": ["reasoning", "coding", "math"],
  "benchmarks": {
    "mmlu": 88.5,
    "humaneval": 82.6,
    "math": 75.3,
    "gsm8k": 92.1,
    "mbpp": 79.4
  },
  "pricing": {
    "api_input_per_1m": 0.27,
    "api_output_per_1m": 1.10,
    "self_hosted_gpu_per_hour": 3.20
  }
}
```

### Adding a new model

Edit `llm_bench_compare/data/benchmarks.json` to add a model entry:

```json
{
  "model_id": "my-new-model-7b",
  "display_name": "My New Model 7B",
  "family": "MyFamily",
  "parameters_billion": 7,
  "size_bucket": "≤7B",
  "license": "apache-2.0",
  "task_categories": ["reasoning", "coding"],
  "scores": {
    "mmlu": 71.2,
    "humaneval": 58.0,
    "math": null,
    "gsm8k": 80.5,
    "mbpp": 62.1
  }
}
```

Then add a corresponding pricing entry to `llm_bench_compare/data/pricing.json`. Restart the server — your model appears in the table immediately.

---

## Project Structure

```
llm_bench_compare/
├── pyproject.toml                  # Project metadata and dependency declarations
├── requirements.txt                # Pinned runtime dependencies
│
├── llm_bench_compare/
│   ├── __init__.py                 # Package initializer, exposes create_app()
│   ├── app.py                      # Flask app factory and all route definitions
│   ├── data_loader.py              # Loads, merges, and caches JSON data into DataFrames
│   ├── filters.py                  # Pure filter functions (task, size, license)
│   │
│   ├── data/
│   │   ├── benchmarks.json         # Benchmark scores for all tracked models
│   │   └── pricing.json            # API and self-hosted GPU cost data
│   │
│   ├── templates/
│   │   └── index.html              # Jinja2 template: table, filters, radar chart
│   │
│   └── static/
│       ├── app.js                  # Dynamic filtering, sorting, Chart.js visualizations
│       └── style.css               # Responsive layout, table styles, cost overlay badges
│
└── tests/
    ├── test_data_loader.py         # Unit tests for data loading, merging, schema validation
    ├── test_filters.py             # Unit tests for all filter functions and edge cases
    └── test_routes.py              # Integration tests for Flask routes and JSON API
```

---

## Configuration

Configuration is handled via environment variables. All settings have sensible defaults for local development.

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | Set to `production` to disable the debugger and reloader. |
| `FLASK_RUN_HOST` | `127.0.0.1` | Host address the development server binds to. |
| `FLASK_RUN_PORT` | `5000` | Port the development server listens on. |
| `SECRET_KEY` | `dev` | Flask session secret key. **Override in production.** |

**Example — production launch:**

```bash
export FLASK_ENV=production
export SECRET_KEY="your-secret-key-here"
export FLASK_RUN_HOST=0.0.0.0
export FLASK_RUN_PORT=8080
python -m llm_bench_compare
```

### Running tests

```bash
pip install pytest pytest-flask
pytest tests/ -v
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) - an AI agent that ships code daily.*
