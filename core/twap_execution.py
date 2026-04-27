"""
NEO SUPREME — TWAP Execution Module
====================================
Time-Weighted Average Price order slicer.

Generates equal-quantity slices spread evenly over a specified duration.
Supports both sync and async execution modes with Redis serialization.

Features:
    - Decimal quantization for precise sizing
    - Equal-quantity slice distribution
    - Redis-schedule serialization
    - Sync and async execution modes
    - Automatic remainder handling (last slice takes residual)

Author: NEO SUPREME
Version: 4.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("twap_execution")


class TWAPSlice:
    """Individual TWAP slice with execution tracking."""

    def __init__(
        self,
        slice_id: int,
        qty: float,
        timestamp: float,
        executed: bool = False,
    ):
        self.slice_id = slice_id
        self.qty = qty
        self.timestamp = timestamp
        self.executed = executed
        self.fill_price: Optional[float] = None
        self.fill_time: Optional[float] = None
        self.order_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "qty": self.qty,
            "timestamp": self.timestamp,
            "executed": self.executed,
            "fill_price": self.fill_price,
            "fill_time": self.fill_time,
            "order_id": self.order_id,
        }


class TWAPExecution:
    """
    Time-Weighted Average Price slicer.
    Generates equal-quantity slices spread evenly over a duration.

    Usage:
        twap = TWAPExecution("BTCUSDT", "buy", 1.0, duration_seconds=60, num_slices=4)
        schedule = twap.to_redis_schedule()
        # Async execution
        results = await twap.execute_async(place_order_fn)
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        duration_seconds: int = 60,
        num_slices: Optional[int] = None,
        price_precision: int = 6,
        qty_precision: int = 6,
        wait_for_fill: bool = True,
    ):
        self.symbol = symbol
        self.side = side.lower()
        self.total_quantity = Decimal(str(total_quantity))
        self.duration_seconds = max(1, duration_seconds)
        self.num_slices = num_slices or max(2, self.duration_seconds // 15)
        self.price_precision = price_precision
        self.qty_precision = qty_precision
        self.wait_for_fill = wait_for_fill

        self._slices: List[TWAPSlice] = []
        self._start_ts: Optional[float] = None
        self._completed_slices = 0
        self._failed_slices = 0
        self._total_filled_qty = 0.0
        self._total_slippage_bps = 0.0

        # Validation
        if self.total_quantity <= 0:
            raise ValueError(f"total_quantity must be positive, got {total_quantity}")
        if self.num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {self.num_slices}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SLICE GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _quantize(self, value: Decimal) -> Decimal:
        """Quantize to specified precision."""
        quant = Decimal(10) ** -self.qty_precision
        return value.quantize(quant, rounding=ROUND_HALF_UP)

    def _generate_slices(self) -> List[TWAPSlice]:
        """Generate equal-quantity slices with proper remainder handling."""
        base_qty = self._quantize(self.total_quantity / Decimal(self.num_slices))
        remaining = self.total_quantity
        interval = self.duration_seconds / self.num_slices
        start_ts = time.time()
        slices: List[TWAPSlice] = []

        for i in range(self.num_slices):
            if remaining <= 0:
                break

            # Last slice takes all remaining quantity
            if i == self.num_slices - 1:
                slice_qty = self._quantize(remaining)
            else:
                slice_qty = self._quantize(min(base_qty, remaining))

            slices.append(TWAPSlice(
                slice_id=i,
                qty=float(slice_qty),
                timestamp=start_ts + i * interval,
            ))
            remaining -= slice_qty

        self._slices = slices
        self._start_ts = start_ts
        return slices

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API — SERIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    def to_redis_schedule(self) -> List[Dict[str, Any]]:
        """
        Return serializable schedule for Redis storage.
        Format: [{"qty": float, "timestamp": float}]
        """
        slices = self._generate_slices()
        return [{"qty": s.qty, "timestamp": s.timestamp} for s in slices]

    def get_next_slice(self) -> Optional[TWAPSlice]:
        """Return the next un-executed slice due for execution."""
        if not self._slices:
            self._generate_slices()
        now = time.time()
        for s in self._slices:
            if not s.executed and now >= s.timestamp:
                return s
        return None

    def mark_executed(self, slice_id: int, fill_price: Optional[float] = None) -> None:
        """Mark a slice as executed with optional fill price."""
        for s in self._slices:
            if s.slice_id == slice_id:
                s.executed = True
                s.fill_price = fill_price
                s.fill_time = time.time()
                self._completed_slices += 1
                if fill_price is not None:
                    self._total_filled_qty += s.qty
                break

    def mark_failed(self, slice_id: int) -> None:
        """Mark a slice as failed."""
        for s in self._slices:
            if s.slice_id == slice_id:
                s.executed = True  # Mark as done (failed)
                self._failed_slices += 1
                break

    # ═══════════════════════════════════════════════════════════════════════════
    # SYNC EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    def execute_sync(
        self,
        place_order_fn: Callable[[str, str, float], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute TWAP slices synchronously.

        Args:
            place_order_fn: callable(symbol, side, qty) -> {"success": bool, "fill_price": float}

        Returns:
            Result dict with fill statistics.
        """
        if not self._slices:
            self._generate_slices()

        logger.info(
            "[TWAP] Starting sync execution: %s %s | qty=%s | slices=%d | duration=%ds",
            self.symbol, self.side, self.total_quantity, len(self._slices),
            self.duration_seconds,
        )

        for s in self._slices:
            # Wait until slice timestamp
            wait_time = s.timestamp - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            try:
                result = place_order_fn(self.symbol, self.side, s.qty)
                if result.get("success"):
                    self.mark_executed(s.slice_id, result.get("fill_price"))
                    logger.debug(
                        "[TWAP] Slice %d executed: qty=%.6f price=%.4f",
                        s.slice_id, s.qty, result.get("fill_price", 0),
                    )
                else:
                    self.mark_failed(s.slice_id)
                    logger.warning("[TWAP] Slice %d failed: %s", s.slice_id, result)
            except Exception as exc:
                self.mark_failed(s.slice_id)
                logger.error("[TWAP] Slice %d exception: %s", s.slice_id, exc)

        return self._build_result()

    # ═══════════════════════════════════════════════════════════════════════════
    # ASYNC EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    async def execute_async(
        self,
        place_order_fn: Callable[[str, str, float], Any],
    ) -> Dict[str, Any]:
        """
        Execute TWAP slices asynchronously.

        Args:
            place_order_fn: async callable(symbol, side, qty) -> {"success": bool, "fill_price": float}

        Returns:
            Result dict with fill statistics.
        """
        if not self._slices:
            self._generate_slices()

        logger.info(
            "[TWAP] Starting async execution: %s %s | qty=%s | slices=%d | duration=%ds",
            self.symbol, self.side, self.total_quantity, len(self._slices),
            self.duration_seconds,
        )

        tasks = []
        for s in self._slices:
            task = self._execute_slice_async(s, place_order_fn)
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)
        return self._build_result()

    async def _execute_slice_async(
        self,
        slice_obj: TWAPSlice,
        place_order_fn: Callable,
    ) -> None:
        """Execute a single slice asynchronously."""
        wait_time = slice_obj.timestamp - time.time()
        if wait_time > 0:
            await asyncio.sleep(wait_time)

        try:
            if asyncio.iscoroutinefunction(place_order_fn):
                result = await place_order_fn(self.symbol, self.side, slice_obj.qty)
            else:
                result = place_order_fn(self.symbol, self.side, slice_obj.qty)

            if isinstance(result, dict) and result.get("success"):
                self.mark_executed(slice_obj.slice_id, result.get("fill_price"))
            else:
                self.mark_failed(slice_obj.slice_id)
        except Exception as exc:
            self.mark_failed(slice_obj.slice_id)
            logger.error("[TWAP] Slice %d exception: %s", slice_obj.slice_id, exc)

    # ═══════════════════════════════════════════════════════════════════════════
    # PROGRESSIVE EXECUTION (Execute one slice at a time)
    # ═══════════════════════════════════════════════════════════════════════════

    async def execute_next_slice(
        self,
        place_order_fn: Callable[[str, str, float], Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Execute only the next due slice. Call repeatedly for progressive execution.
        Returns slice result or None if no slice is due.
        """
        slice_obj = self.get_next_slice()
        if slice_obj is None:
            return None

        try:
            if asyncio.iscoroutinefunction(place_order_fn):
                result = await place_order_fn(self.symbol, self.side, slice_obj.qty)
            else:
                result = place_order_fn(self.symbol, self.side, slice_obj.qty)

            if isinstance(result, dict) and result.get("success"):
                self.mark_executed(slice_obj.slice_id, result.get("fill_price"))
                return {
                    "slice_id": slice_obj.slice_id,
                    "qty": slice_obj.qty,
                    "fill_price": result.get("fill_price"),
                    "success": True,
                }
            else:
                self.mark_failed(slice_obj.slice_id)
                return {"slice_id": slice_obj.slice_id, "success": False, "error": result}
        except Exception as exc:
            self.mark_failed(slice_obj.slice_id)
            return {"slice_id": slice_obj.slice_id, "success": False, "error": str(exc)}

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS & PROPERTIES
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_result(self) -> Dict[str, Any]:
        """Build execution result summary."""
        filled_slices = [s for s in self._slices if s.executed and s.fill_price is not None]
        failed_count = self._failed_slices

        total_filled = sum(s.qty for s in filled_slices)
        total_requested = float(self.total_quantity)
        fill_rate = total_filled / total_requested if total_requested > 0 else 0.0

        # Calculate VWAP of fills
        total_value = sum(s.qty * (s.fill_price or 0) for s in filled_slices)
        vwap = total_value / total_filled if total_filled > 0 else 0.0

        elapsed = time.time() - (self._start_ts or time.time())

        return {
            "symbol": self.symbol,
            "side": self.side,
            "total_quantity": float(self.total_quantity),
            "total_filled": total_filled,
            "fill_rate": fill_rate,
            "vwap": vwap,
            "slices_total": len(self._slices),
            "slices_filled": len(filled_slices),
            "slices_failed": failed_count,
            "execution_time_seconds": round(elapsed, 2),
            "is_complete": self.is_complete,
        }

    @property
    def is_complete(self) -> bool:
        return bool(self._slices) and all(s.executed for s in self._slices)

    @property
    def remaining_qty(self) -> float:
        return sum(s.qty for s in self._slices if not s.executed)

    @property
    def progress_pct(self) -> float:
        if not self._slices:
            return 0.0
        executed = sum(1 for s in self._slices if s.executed)
        return executed / len(self._slices)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "total_quantity": float(self.total_quantity),
            "duration_seconds": self.duration_seconds,
            "num_slices": self.num_slices,
            "slice_interval_s": self.duration_seconds / max(self.num_slices, 1),
            "slices": [s.to_dict() for s in self._slices],
            "is_complete": self.is_complete,
            "progress_pct": self.progress_pct,
            "remaining_qty": self.remaining_qty,
        }
