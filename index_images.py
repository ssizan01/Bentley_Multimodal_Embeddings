"""
Index all images under config.STATIC_DIR:
- compute embeddings (Vertex AI Multimodal Embeddings)
- write to BigQuery config.BQ_DATASET.config.BQ_TABLE

Usage:
  python index_images.py            # append
  python index_images.py --recreate # drop & rebuild table first
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterator

import config
import embedder
import bq_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("index_images")


def iter_image_files(root: Path) -> Iterator[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in config.IMAGE_EXTS:
                yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed local images and push to BigQuery")
    parser.add_argument("--recreate", action="store_true", help="Drop & recreate BigQuery table before loading")
    args = parser.parse_args()

    if not config.STATIC_DIR.exists():
        raise SystemExit(f"STATIC_DIR does not exist: {config.STATIC_DIR}")

    bq_store.ensure_dataset_and_table(recreate=args.recreate)

    rows = []
    count = 0
    for path in iter_image_files(config.STATIC_DIR):
        rel_path = path.relative_to(config.STATIC_DIR).as_posix()
        try:
            vec = embedder.embed_image(str(path), dimension=config.EMBEDDING_DIM)
        except Exception as e:
            logger.exception("Failed to embed %s: %s", path, e)
            continue

        row: Dict = {
            "image_name": path.name,
            "rel_path": rel_path,
            "embedding": vec,
            "embedding_dim": config.EMBEDDING_DIM,
            "model_name": config.MME_MODEL_NAME,
        }
        rows.append(row)
        count += 1
        if count % 50 == 0:
            logger.info("Prepared %d images...", count)

    if not rows:
        logger.warning("No images found under %s", config.STATIC_DIR)
        return

    bq_store.load_embeddings(rows, write_truncate=args.recreate)
    logger.info("Done. Indexed %d images into BigQuery.", len(rows))


if __name__ == "__main__":
    main()