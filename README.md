# Bentley Multimodal Image Search (NiceGUI)

A simple NiceGUI app to search local images using text. It embeds your query with Vertex AI Multimodal Embeddings and finds the most similar images stored in BigQuery using cosine similarity.

- Start the app: `python main.py`
- Index your images first (recommended): `python index_images.py --recreate`


## Quick Start

1) Python environment
- Python 3.10+ recommended (3.11 used during development).
- Install deps: `pip install -r requirements.txt`

2) Configure Google Cloud
- Ensure you can authenticate (one of):
  - `gcloud auth application-default login`, or
  - set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON file with access to Vertex AI and BigQuery.
- Make sure the following APIs are enabled in your project:
  - Vertex AI API
  - BigQuery API

3) Set environment variables (optional; defaults shown in `config.py`)
- `GCP_PROJECT` (default: `mase-srtt-internal-genai`)
- `VERTEX_LOCATION` (default: `us-central1`)
- `BQ_DATASET` (default: `bentley_embeddings`)
- `BQ_TABLE` (default: `image_embeddings`)
- `BQ_LOCATION` (default: `US`)
- `STATIC_DIR` (default: `<project>/static`)
- `EMBEDDING_DIM` (default: `1408`)
- `TOP_K` (default: `5`)

Example (bash):
```
export GCP_PROJECT="your-gcp-project-id"
export VERTEX_LOCATION="us-central1"
export BQ_DATASET="bentley_embeddings"
export BQ_TABLE="image_embeddings"
export BQ_LOCATION="US"
# optional if your images are elsewhere
export STATIC_DIR="/path/to/your/images"
```

4) Index images into BigQuery
- Put your images under `static/` (or wherever `STATIC_DIR` points).
- Run: `python index_images.py --recreate`
  - Creates dataset/table if needed and loads embeddings for all images.
  - Omit `--recreate` to append instead of replacing existing rows.

5) Run the app
- Start: `python main.py`
- Open: http://localhost:8080 (NiceGUI default port).
- Type a description (e.g., "red leather interior"). The app shows the top-K images with cosine similarity.


## What Each File Does

- `main.py`
  - NiceGUI UI. Serves `/static` for images, provides a search box, and displays the top results.
  - Calls the search service asynchronously so the UI stays responsive.

- `service.py`
  - The core search function: embeds a text query via `embedder.py`, then fetches top-K similar images from BigQuery via `bq_store.py`.

- `embedder.py`
  - Lightweight wrapper around Vertex AI Multimodal Embeddings (`multimodalembedding@001`).
  - Provides `embed_text(...)` and `embed_image(...)` returning float vectors of size `EMBEDDING_DIM`.

- `bq_store.py`
  - BigQuery utilities:
    - `ensure_dataset_and_table(...)` creates the dataset/table if needed.
    - `load_embeddings(...)` bulk loads image rows (name, relative path, embedding, metadata).
    - `top_k_by_cosine(...)` runs a BigQuery VECTOR_SEARCH (brute-force) query and returns rows with `cosine_sim`.

- `index_images.py`
  - CLI tool to scan `STATIC_DIR`, embed each image, and load rows into BigQuery.
  - Use `--recreate` to drop & recreate the table before loading.

- `config.py`
  - Central configuration and defaults for GCP, BigQuery, Vertex AI, and app behavior (e.g., `TOP_K`, `EMBEDDING_DIM`, `STATIC_DIR`).

- `requirements.txt`
  - Python dependencies for the app, embedding, and BigQuery.

- `static/`
  - Local image directory served as `/static`. The app expects BigQuery rows to reference images here using `rel_path`.

Note: per request, ignore `search_type.py`.


## Common Tips & Troubleshooting

- No results or missing images in UI
  - Ensure the image files exist under `STATIC_DIR` and that `rel_path` in BigQuery matches how the file is laid out (e.g., `photos/car.jpg` loads from `/static/photos/car.jpg`).

- BigQuery permissions / auth errors
  - Verify your ADC credentials and project. Service account needs BigQuery read/write and Vertex AI User roles.

- Vertex AI errors
  - Confirm the Vertex AI API is enabled and `VERTEX_LOCATION` matches a supported region for `multimodalembedding@001`.

- Change top-K results
  - Set `TOP_K` env var (or edit `config.py`), then restart.

- Change embedding size
  - Set `EMBEDDING_DIM` (defaults to 1408). Re-index images if you change this so vectors are consistent.

