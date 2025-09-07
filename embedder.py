"""
Vertex AI Multimodal Embeddings wrapper (image + text).
"""
from __future__ import annotations

import logging
import warnings
from typing import List

import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

import config

logger = logging.getLogger(__name__)

# Optional: silence the deprecation warning spam from the SDK
warnings.filterwarnings(
    "ignore",
    message="This feature is deprecated",
    category=UserWarning,
    module="vertexai",
)

# Initialize Vertex AI
vertexai.init(project=config.PROJECT_ID, location=config.VERTEX_LOCATION)

# Load model once
_model = MultiModalEmbeddingModel.from_pretrained(config.MME_MODEL_NAME)


def embed_image(path: str, dimension: int | None = None) -> List[float]:
    dim = dimension or config.EMBEDDING_DIM
    img = Image.load_from_file(path)
    emb = _model.get_embeddings(image=img, dimension=dim)
    return list(emb.image_embedding)


def embed_text(text: str, dimension: int | None = None) -> List[float]:
    dim = dimension or config.EMBEDDING_DIM
    try:
        emb = _model.get_embeddings(text=text, dimension=dim)
        vec = getattr(emb, "text_embedding", None)
        if vec is not None:
            return list(vec)
    except TypeError:
        pass
    emb = _model.get_embeddings(contextual_text=text, dimension=dim)
    return list(emb.text_embedding)
