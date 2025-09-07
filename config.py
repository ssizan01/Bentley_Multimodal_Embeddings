"""
Central configuration for embeddings + app.

Environment variables (optional):
- GCP_PROJECT       (default: 'mase-srtt-internal-genai')
- VERTEX_LOCATION   (default: 'us-central1')
- BQ_DATASET        (default: 'bentley_embeddings')
- BQ_TABLE          (default: 'image_embeddings')
- BQ_LOCATION       (default: 'US')
- STATIC_DIR        (default: '<project>/static')
- EMBEDDING_DIM     (default: '1408')
- TOP_K             (default: '5')
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STATIC = PROJECT_ROOT / "static"

PROJECT_ID: str = os.getenv("GCP_PROJECT", "mase-srtt-internal-genai")
VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")
BQ_DATASET: str = os.getenv("BQ_DATASET", "bentley_embeddings")
BQ_TABLE: str = os.getenv("BQ_TABLE", "image_embeddings")
BQ_LOCATION: str = os.getenv("BQ_LOCATION", "US")

STATIC_DIR: Path = Path(os.getenv("STATIC_DIR", str(DEFAULT_STATIC))).expanduser().resolve()

MME_MODEL_NAME: str = "multimodalembedding@001"
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1408"))
TOP_K: int = int(os.getenv("TOP_K", "5"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
