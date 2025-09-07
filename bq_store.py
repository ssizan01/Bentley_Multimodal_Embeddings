"""
BigQuery storage and retrieval utilities for embeddings.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Dict, Any

from google.cloud import bigquery
import config

logger = logging.getLogger(__name__)

_CLIENT = bigquery.Client(project=config.PROJECT_ID)


def _fq_dataset() -> str:
    return f"`{config.PROJECT_ID}.{config.BQ_DATASET}`"


def _fq_table() -> str:
    return f"`{config.PROJECT_ID}.{config.BQ_DATASET}.{config.BQ_TABLE}`"


def ensure_dataset_and_table(recreate: bool = False) -> None:
    """Create dataset & table if needed. Optionally recreate table."""
    dataset_sql = (
        f"CREATE SCHEMA IF NOT EXISTS {_fq_dataset()} "
        f"OPTIONS(location='{config.BQ_LOCATION.lower()}')"
    )
    _CLIENT.query(dataset_sql, location=config.BQ_LOCATION).result()
    logger.info("Ensured dataset %s exists (location: %s)", config.BQ_DATASET, config.BQ_LOCATION)

    table_sql = f"""
    CREATE TABLE IF NOT EXISTS {_fq_table()} (
      image_name STRING NOT NULL,
      rel_path   STRING NOT NULL,
      embedding  ARRAY<FLOAT64>,   -- arrays cannot be NOT NULL
      embedding_dim INT64 NOT NULL,
      model_name STRING NOT NULL,
      inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() NOT NULL
    )
    """
    if recreate:
        logger.warning("Recreating table %s.%s", config.BQ_DATASET, config.BQ_TABLE)
        _CLIENT.query(f"DROP TABLE IF EXISTS {_fq_table()}", location=config.BQ_LOCATION).result()

    _CLIENT.query(table_sql, location=config.BQ_LOCATION).result()


def load_embeddings(rows: Iterable[Dict[str, Any]], write_truncate: bool = False) -> None:
    """Bulk load rows into the embeddings table."""
    table_id = f"{config.PROJECT_ID}.{config.BQ_DATASET}.{config.BQ_TABLE}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if write_truncate else bigquery.WriteDisposition.WRITE_APPEND
        )
    )
    job = _CLIENT.load_table_from_json(
        list(rows), table_id, job_config=job_config, location=config.BQ_LOCATION
    )
    job.result()
    logger.info("Loaded %s rows into %s", job.output_rows, table_id)


def top_k_by_cosine(query_vec: List[float], k: int) -> List[Dict[str, Any]]:
    """
    Cosine similarity = 1 - ML.DISTANCE(a, b, 'COSINE').
    """
    sql = f"""
    SELECT
      image_name,
      rel_path,
      1 - ML.DISTANCE(embedding, @qvec, 'COSINE') AS cosine_sim
    FROM {_fq_table()}
    ORDER BY cosine_sim DESC
    LIMIT @k
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("qvec", "FLOAT64", query_vec),
            bigquery.ScalarQueryParameter("k", "INT64", k),
        ]
    )
    job = _CLIENT.query(sql, job_config=job_config, location=config.BQ_LOCATION)
    return [dict(row) for row in job.result()]
