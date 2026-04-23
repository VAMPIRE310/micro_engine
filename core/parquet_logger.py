"""
ParquetLogger — Institutional-grade buffered columnar storage.
Thread-safe, snappy-compressed, lock-on-flush pattern.
Write path: pyarrow (minimal overhead).
Read/analytics path: Polars (lazy scan, columnar, 10-100x faster than pandas).
"""
import os
import time
import threading
import logging
from typing import List, Dict, Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

log = logging.getLogger(__name__)


class ParquetLogger:
    """
    Memory-buffered Parquet writer with Polars-powered analytics reads.
    Accumulates rows in RAM, flushes to disk in batches to avoid per-row I/O.
    On flush, appends to existing file via concat (read-modify-write).
    Recovers gracefully from corrupted files by writing a .recovery sidecar.
    """

    def __init__(self, filepath: str, flush_size: int = 100):
        self.filepath    = filepath
        self.flush_size  = flush_size
        self._buffer: List[Dict[str, Any]] = []
        self._lock       = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

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

        except Exception as exc:
            log.error("[ParquetLogger] flush failed for %s: %s", self.filepath, exc)
            # Put rows back so they aren't lost
            self._buffer = rows + self._buffer
