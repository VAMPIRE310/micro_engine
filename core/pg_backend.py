"""
PgBackend — PostgreSQL persistence for experience replay and model checkpoints.

Two tables are created automatically on first connection:

  parquet_logs       — experience rows stored as JSONB (queryable, appendable)
  model_checkpoints  — PyTorch state_dict blobs stored as BYTEA

Connection is lazy and self-healing: a lost connection is transparently
re-established on the next operation.  Every error is logged and swallowed
so the trading engine continues in a degraded-but-functional state.

Railway setup (canvas linking)
-------------------------------
In the Railway dashboard, set the micro-engine service variables using
Reference Variable syntax so Railway draws connecting lines in the canvas:

  REDIS_URL    = ${{redis-tbj0.REDIS_URL}}
  DATABASE_URL = ${{Postgres.DATABASE_URL}}

Using Reference Variables instead of raw strings is what makes Railway
show all services as one connected group rather than "split in 2".
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

import polars as pl

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS parquet_logs (
    id          BIGSERIAL    PRIMARY KEY,
    table_name  TEXT         NOT NULL,
    data        JSONB        NOT NULL,
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_parquet_logs_table_id
    ON parquet_logs (table_name, id DESC);

CREATE TABLE IF NOT EXISTS model_checkpoints (
    id          BIGSERIAL    PRIMARY KEY,
    model_name  TEXT         NOT NULL,
    weights     BYTEA        NOT NULL,
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_model_checkpoints_name_id
    ON model_checkpoints (model_name, id DESC);
"""


class PgBackend:
    """
    Thin psycopg2 wrapper providing:
      - bulk-insert of experience rows as JSONB  (parquet_logs table)
      - save / load of PyTorch model weights as BYTEA  (model_checkpoints table)

    Instantiate with a DATABASE_URL or let it read the env var automatically.
    If DATABASE_URL is empty the backend silently no-ops on every call.
    """

    def __init__(self, database_url: Optional[str] = None):
        self._url  = database_url or os.environ.get("DATABASE_URL", "")
        self._conn = None

    @property
    def available(self) -> bool:
        """True when a DATABASE_URL has been configured."""
        return bool(self._url)

    # ── Internal connection management ───────────────────────────────────────

    def _get_conn(self):
        """Return a live psycopg2 connection, reconnecting transparently."""
        if self._conn is not None:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return self._conn
            except Exception:
                self._conn = None

        if not self._url:
            raise RuntimeError("DATABASE_URL is not configured")

        import psycopg2  # type: ignore

        url = self._url
        # psycopg2 accepts "postgresql://" but not "postgresql+psycopg2://"
        if "+" in url.split("://")[0]:
            url = "postgresql" + url[url.index("://"):]

        self._conn = psycopg2.connect(url)
        self._conn.autocommit = False

        with self._conn.cursor() as cur:
            cur.execute(_DDL)
        self._conn.commit()
        log.info("[PgBackend] Connected to PostgreSQL — schema ready.")
        return self._conn

    def _rollback_and_reset(self) -> None:
        if self._conn is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        self._conn = None

    # ── Experience replay ────────────────────────────────────────────────────

    def log_rows(self, table_name: str, rows: List[Dict[str, Any]]) -> None:
        """Bulk-insert experience rows into parquet_logs as JSONB."""
        if not rows or not self.available:
            return
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO parquet_logs (table_name, data)"
                    " VALUES (%s, %s::jsonb)",
                    [(table_name, json.dumps(r)) for r in rows],
                )
            conn.commit()
        except Exception as exc:
            log.warning("[PgBackend] log_rows('%s') failed: %s", table_name, exc)
            self._rollback_and_reset()

    def read_rows(self, table_name: str,
                  n_recent: Optional[int] = None) -> pl.DataFrame:
        """Return a Polars DataFrame of stored rows, oldest-first."""
        if not self.available:
            return pl.DataFrame()
        try:
            conn  = self._get_conn()
            limit = n_recent or 10_000
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM parquet_logs"
                    " WHERE table_name = %s"
                    " ORDER BY id ASC LIMIT %s",
                    (table_name, limit),
                )
                rows = [r[0] for r in cur.fetchall()]
            if not rows:
                return pl.DataFrame()
            return pl.DataFrame(rows)
        except Exception as exc:
            log.warning("[PgBackend] read_rows('%s') failed: %s", table_name, exc)
            self._rollback_and_reset()
            return pl.DataFrame()

    # ── Model checkpoints ────────────────────────────────────────────────────

    def save_model(self, model_name: str, state_dict: Any) -> None:
        """Serialise a PyTorch state_dict and persist to model_checkpoints."""
        if not self.available:
            return
        try:
            import psycopg2  # type: ignore
            import torch      # type: ignore

            buf = io.BytesIO()
            torch.save(state_dict, buf)
            blob = buf.getvalue()

            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO model_checkpoints (model_name, weights)"
                    " VALUES (%s, %s)",
                    (model_name, psycopg2.Binary(blob)),
                )
            conn.commit()
            log.info("[PgBackend] Saved model '%s' (%d bytes).",
                     model_name, len(blob))
        except Exception as exc:
            log.warning("[PgBackend] save_model('%s') failed: %s",
                        model_name, exc)
            self._rollback_and_reset()

    def load_model(self, model_name: str) -> Optional[Any]:
        """Return the latest state_dict for model_name, or None if absent."""
        if not self.available:
            return None
        try:
            import torch  # type: ignore

            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT weights FROM model_checkpoints"
                    " WHERE model_name = %s"
                    " ORDER BY id DESC LIMIT 1",
                    (model_name,),
                )
                row = cur.fetchone()
            if row is None:
                return None
            state_dict = torch.load(
                io.BytesIO(bytes(row[0])),
                map_location="cpu",
                weights_only=True,
            )
            log.info("[PgBackend] Loaded model '%s' from PostgreSQL.", model_name)
            return state_dict
        except Exception as exc:
            log.warning("[PgBackend] load_model('%s') failed: %s",
                        model_name, exc)
            self._rollback_and_reset()
            return None
