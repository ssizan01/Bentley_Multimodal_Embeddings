"""
Search service: embed a text query and fetch top-K images from BigQuery.
"""
from __future__ import annotations

from typing import List, Dict, Any

import config
import embedder
import bq_store


def search_images_by_text(query_text: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    k = top_k or config.TOP_K
    qvec = embedder.embed_text(query_text, dimension=config.EMBEDDING_DIM)
    return bq_store.top_k_by_cosine(qvec, k)
