"""
ParquetLogger — Institutional-grade buffered columnar storage.
Thread-safe, snappy-compressed, lock-on-flush pattern.
Write path: pyarrow (minimal overhead).
Read/analytics path: Polars (lazy scan, columnar, 10-100x faster than pandas).

PostgreSQL integration (optional)
----------------------------------
Pass a PgBackend instance and a table_name to mirror every flush to PostgreSQL.
On cold start (no local file), the logger seeds itself from PostgreSQL so that
online learning resumes from where the previous Railway deploy left off.
"""
import os
import time
import threading
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

if TYPE_CHECKING:
    from core.pg_backend import PgBackend

log = logging.getLogger(__name__)


class ParquetLogger:
    """
    Memory-buffered Parquet writer with Polars-powered analytics reads.
    Accumulates rows in RAM, flushes to disk in batches to avoid per-row I/O.
    On flush, appends to existing file via concat (read-modify-write).
    Recovers gracefully from corrupted files by writing a .recovery sidecar.

    When a PgBackend is supplied every flush is mirrored to PostgreSQL, and
    the local file is seeded from PostgreSQL on the first cold start so that
    experience history survives Railway redeployments.
    """

    def __init__(self, filepath: str, flush_size: int = 100,
                 pg_backend: Optional["PgBackend"] = None,
                 table_name: Optional[str] = None):
        self.filepath    = filepath
        self.flush_size  = flush_size
        self._buffer: List[Dict[str, Any]] = []
        self._lock       = threading.Lock()
        self._pg         = pg_backend
        self._table_name = table_name or os.path.splitext(
            os.path.basename(filepath)
        )[0]
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        if self._pg is not None and self._pg.available:
            if not os.path.exists(filepath):
                self._seed_from_pg()

    def _seed_from_pg(self) -> None:
        """Write PostgreSQL rows to the local parquet file on cold start."""
        try:
            df = self._pg.read_rows(self._table_name)
            if len(df) == 0:
                return
            table = df.to_arrow()
            pq.write_table(table, self.filepath, compression="snappy")
            log.info(
                "[ParquetLogger] Cold-start seed: %d rows from PostgreSQL → %s",
                len(df), self.filepath,
            )
        except Exception as exc:
            log.warning("[ParquetLogger] PG seed failed for %s: %s",
                        self.filepath, exc)

    # ── Public API ──────────────────────────────────────────────────────────

    def log(self, data: Dict[str, Any]):
        """Append one row. Flushes automatically when buffer reaches flush_size."""
        with self._lock:
            self._buffer.append(data)
            if len(self._buffer) >= self.flush_size:
                self._flush()

    def flush(self):
        """Force-flush any buffered rows (call on clean shutdown)."""
        with self._lock:
            self._flush()

    def read_polars(self, n_recent: Optional[int] = None) -> pl.DataFrame:
        """
        Fast columnar read via Polars lazy scan.
        Returns a Polars DataFrame — use .to_dicts() if you need plain Python.
        """
        frames = []

        if os.path.exists(self.filepath):
            try:
                df = pl.scan_parquet(self.filepath).collect()
                frames.append(df)
            except Exception as exc:
                log.warning("[ParquetLogger] polars read failed for %s: %s", self.filepath, exc)

        with self._lock:
            pending = list(self._buffer)

        if pending:
            frames.append(pl.DataFrame(pending))

        if not frames:
            return pl.DataFrame()

        combined = pl.concat(frames, how="diagonal_relaxed")
        if n_recent is not None:
            combined = combined.tail(n_recent)
        return combined

    def read(self, n_recent: Optional[int] = None) -> List[Dict[str, Any]]:
        """Convenience wrapper — returns list of dicts."""
        return self.read_polars(n_recent).to_dicts()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _flush(self):
        """Write buffered rows to Parquet. Must be called with self._lock held."""
        if not self._buffer:
            return

        rows       = self._buffer[:]
        self._buffer.clear()

        try:
            new_table = pa.Table.from_pylist(rows)

            if os.path.exists(self.filepath):
                try:
                    existing  = pq.read_table(self.filepath)
                    new_table = pa.concat_tables([existing, new_table])
                except Exception as exc:
                    # Corrupted file — archive it and start fresh
                    recovery = f"{self.filepath}.recovery.{int(time.time())}"
                    os.rename(self.filepath, recovery)
                    log.warning("[ParquetLogger] corrupted %s → archived to %s: %s",
                                self.filepath, recovery, exc)

            pq.write_table(new_table, self.filepath, compression="snappy")

            # Mirror to PostgreSQL for cross-deployment persistence
            if self._pg is not None and self._pg.available:
                self._pg.log_rows(self._table_name, rows)

        except Exception as exc:
            log.error("[ParquetLogger] flush failed for %s: %s", self.filepath, exc)
            # Put rows back so they aren't lost
            self._buffer = rows + self._buffer
