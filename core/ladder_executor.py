"""
NEO SUPREME — Two-Phase Ladder Executor
========================================
Split execution into a fast market-fill phase and a limit-rung phase.

Phase 1: Market order for configurable pct (default 20%) to fill immediately.
Phase 2: Remaining quantity placed as staggered limit rungs (default 8).

Features:
    - RungState enum: PENDING, SUBMITTED, FILLED, CANCELLED, FAILED
    - Polling fill wait: every 300ms
    - Result dict with fill_rate, vwap, slippage_bps, execution_time_ms
    - Callable injection for exchange-specific order placement
    - Async-first design with sync fallback

Author: NEO SUPREME
Version: 4.0.0
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger("ladder_executor")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class RungState(Enum):
    PENDING = "pending"      # Waiting for execution trigger
    SUBMITTED = "submitted"  # Order sent to exchange
    FILLED = "filled"        # Order fully filled
    PARTIAL = "partial"      # Order partially filled
    CANCELLED = "cancelled"  # Order cancelled (no longer active)
    FAILED = "failed"        # Order failed


@dataclass
class RungOrder:
    """Individual ladder rung with full tracking."""

    rung_id: int
    price: float
    qty: float
    state: RungState = RungState.PENDING
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    fill_price: Optional[float] = None
    submitted_at: float = 0.0
    filled_at: float = 0.0
    attempts: int = 0

    @property
    def remaining_qty(self) -> float:
        return self.qty - self.filled_qty

    @property
    def is_active(self) -> bool:
        return self.state in (RungState.PENDING, RungState.SUBMITTED, RungState.PARTIAL)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rung_id": self.rung_id,
            "price": self.price,
            "qty": self.qty,
            "state": self.state.value,
            "order_id": self.order_id,
            "filled_qty": self.filled_qty,
            "fill_price": self.fill_price,
            "submitted_at": self.submitted_at,
            "filled_at": self.filled_at,
            "attempts": self.attempts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TWO-PHASE LADDER EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class TwoPhaseLadderExecutor:
    """
    Two-phase execution:

    Phase 1 — Market Fill (default 20%):
        Immediate market order for quick-fill. Reduces market exposure time.

    Phase 2 — Staggered Limit Rungs (default 8):
        Remaining quantity split across price rungs to capture spread.
        Rungs are spaced from reference_price toward mid-price.
    """

    def __init__(
        self,
        market_pct: float = 0.20,
        num_rungs: int = 8,
        rung_spacing_bps: float = 10.0,
        fill_poll_interval_ms: float = 300.0,
        max_poll_cycles: int = 100,
        retry_limit: int = 3,
        precision: int = 6,
    ):
        self.market_pct = market_pct
        self.num_rungs = num_rungs
        self.rung_spacing_bps = rung_spacing_bps
        self.fill_poll_interval_ms = fill_poll_interval_ms
        self.max_poll_cycles = max_poll_cycles
        self.retry_limit = retry_limit
        self.precision = precision

        # State
        self._phase1_rungs: List[RungOrder] = []
        self._phase2_rungs: List[RungOrder] = []
        self._result: Dict[str, Any] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # SLICE GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_rungs(
        self,
        total_quantity: float,
        reference_price: float,
        side: str,
    ) -> Tuple[List[RungOrder], List[RungOrder]]:
        """
        Generate Phase 1 (market) and Phase 2 (limit) rungs.

        Phase 1: Single market rung at market_pct of total.
        Phase 2: Staggered limit rungs filling remaining qty from best toward ref.

        Returns:
            Tuple of (phase1_rungs, phase2_rungs).
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side}")
        if total_quantity <= 0 or reference_price <= 0:
            raise ValueError("total_quantity and reference_price must be positive")

        qty = round(total_quantity, self.precision)
        mid = reference_price
        spread_half = mid * (self.rung_spacing_bps / 10_000.0)

        # Determine price direction for rungs
        if side == "buy":
            start_price = mid - spread_half
            end_price = mid
        else:
            start_price = mid + spread_half
            end_price = mid

        # ── Phase 1: Market rung ──
        market_qty = round(qty * self.market_pct, self.precision)
        if market_qty <= 0:
            market_qty = round(max(qty / self.num_rungs, 10 ** -self.precision), self.precision)

        p1 = [RungOrder(
            rung_id=0,
            price=reference_price,
            qty=market_qty,
            state=RungState.PENDING,
        )]

        # ── Phase 2: Limit rungs ──
        remaining = round(qty - market_qty, self.precision)
        if remaining <= 0:
            return p1, []

        per_rung = round(remaining / self.num_rungs, self.precision)
        p2: List[RungOrder] = []

        for i in range(self.num_rungs):
            if remaining <= 0:
                break
            rung_qty = min(per_rung, remaining)
            remaining = round(remaining - rung_qty, self.precision)

            # Linear interpolation from start_price toward end_price
            frac = i / max(self.num_rungs - 1, 1)
            price = start_price + (end_price - start_price) * frac

            p2.append(RungOrder(
                rung_id=i + 1,
                price=round(price, self.precision),
                qty=round(rung_qty, self.precision),
                state=RungState.PENDING,
            ))

        # Last rung takes any remaining dust
        if remaining > 0 and p2:
            p2[-1].qty = round(p2[-1].qty + remaining, self.precision)

        self._phase1_rungs = p1
        self._phase2_rungs = p2

        logger.info(
            "[Ladder] Generated: P1=%d rungs (%.4f qty) | P2=%d rungs (%.4f qty)",
            len(p1), sum(r.qty for r in p1),
            len(p2), sum(r.qty for r in p2),
        )

        return p1, p2

    # ═══════════════════════════════════════════════════════════════════════════
    # SYNC EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    def execute(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        reference_price: float,
        place_order_fn: Callable[[str, str, float, float, bool], Dict[str, Any]],
        cancel_order_fn: Optional[Callable] = None,
        get_fill_status_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous two-phase execution.

        Args:
            symbol: e.g. "BTCUSDT"
            side: "buy" or "sell"
            total_quantity: Total size to execute
            reference_price: Mid/reference price for rung spacing
            place_order_fn: callable(symbol, side, qty, price, is_market) -> order_id_or_dict
            cancel_order_fn: Optional callable(order_id) for cancellation
            get_fill_status_fn: Optional callable(order_id) -> {"filled_qty": float, ...}

        Returns:
            Result dict with fill_rate, vwap, slippage_bps, execution_time_ms.
        """
        self._phase1_rungs, self._phase2_rungs = self.generate_rungs(
            total_quantity, reference_price, side,
        )
        start = time.monotonic()
        side = side.lower()

        total_filled = 0.0
        total_value = 0.0
        total_slippage = 0.0

        # ── Phase 1: Market fill ──
        for rung in self._phase1_rungs:
            try:
                result = place_order_fn(symbol, side, rung.qty, reference_price, True)
                rung.order_id = result.get("order_id") if isinstance(result, dict) else str(result)
                rung.submitted_at = time.time()

                if rung.order_id and get_fill_status_fn:
                    fill = self._wait_for_fill_sync(
                        rung.order_id, get_fill_status_fn, rung.qty,
                    )
                    rung.filled_qty = fill.get("filled_qty", rung.qty)
                    rung.fill_price = fill.get("fill_price", reference_price)
                    rung.state = RungState.FILLED if rung.filled_qty >= rung.qty * 0.99 else RungState.PARTIAL
                    total_filled += rung.filled_qty
                    total_value += rung.filled_qty * (rung.fill_price or reference_price)
                    slippage = abs((rung.fill_price or reference_price) - reference_price) / reference_price * 10_000
                    total_slippage += slippage * rung.filled_qty
                else:
                    # Assume filled at reference price
                    rung.filled_qty = rung.qty
                    rung.fill_price = reference_price
                    rung.state = RungState.FILLED
                    total_filled += rung.qty
                    total_value += rung.qty * reference_price

            except Exception as exc:
                rung.state = RungState.FAILED
                logger.error("[Ladder] Phase 1 rung %d failed: %s", rung.rung_id, exc)

        # ── Phase 2: Limit rungs ──
        for rung in self._phase2_rungs:
            if rung.state != RungState.PENDING:
                continue
            for attempt in range(self.retry_limit):
                try:
                    result = place_order_fn(symbol, side, rung.qty, rung.price, False)
                    rung.order_id = result.get("order_id") if isinstance(result, dict) else str(result)
                    rung.submitted_at = time.time()
                    rung.attempts = attempt + 1
                    rung.state = RungState.SUBMITTED

                    if rung.order_id and get_fill_status_fn:
                        fill = self._wait_for_fill_sync(
                            rung.order_id, get_fill_status_fn, rung.qty,
                        )
                        rung.filled_qty = fill.get("filled_qty", 0.0)
                        rung.fill_price = fill.get("fill_price", rung.price)
                        if rung.filled_qty >= rung.qty * 0.99:
                            rung.state = RungState.FILLED
                            total_filled += rung.filled_qty
                            total_value += rung.filled_qty * (rung.fill_price or rung.price)
                        elif rung.filled_qty > 0:
                            rung.state = RungState.PARTIAL
                            total_filled += rung.filled_qty
                            total_value += rung.filled_qty * (rung.fill_price or rung.price)
                        else:
                            rung.state = RungState.CANCELLED
                            if cancel_order_fn:
                                try:
                                    cancel_order_fn(rung.order_id)
                                except Exception:
                                    pass
                    break
                except Exception as exc:
                    logger.warning(
                        "[Ladder] Phase 2 rung %d attempt %d failed: %s",
                        rung.rung_id, attempt + 1, exc,
                    )
                    if attempt == self.retry_limit - 1:
                        rung.state = RungState.FAILED

        elapsed_ms = (time.monotonic() - start) * 1000.0
        vwap = total_value / total_filled if total_filled > 0 else reference_price
        slippage_bps = (total_slippage / total_filled) if total_filled > 0 else 0.0
        fill_rate = total_filled / total_quantity if total_quantity > 0 else 0.0

        self._result = {
            "success": fill_rate > 0.5,
            "fill_rate": fill_rate,
            "total_filled": total_filled,
            "total_quantity": total_quantity,
            "vwap": vwap,
            "slippage_bps": round(slippage_bps, 2),
            "execution_time_ms": round(elapsed_ms, 2),
            "phase1_filled": sum(r.filled_qty for r in self._phase1_rungs),
            "phase2_filled": sum(r.filled_qty for r in self._phase2_rungs),
            "phase1_rungs": len(self._phase1_rungs),
            "phase2_rungs": len(self._phase2_rungs),
            "phase1_rungs_data": [r.to_dict() for r in self._phase1_rungs],
            "phase2_rungs_data": [r.to_dict() for r in self._phase2_rungs],
            "failed_rungs": sum(
                1 for r in self._phase1_rungs + self._phase2_rungs
                if r.state == RungState.FAILED
            ),
        }

        logger.info(
            "[Ladder] Execution complete: fill_rate=%.2f%% vwap=%.4f "
            "slippage=%.2fbps time=%.0fms",
            fill_rate * 100, vwap, slippage_bps, elapsed_ms,
        )

        return dict(self._result)

    # ═══════════════════════════════════════════════════════════════════════════
    # ASYNC EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    async def execute_async(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        reference_price: float,
        place_order_fn: Callable,
        cancel_order_fn: Optional[Callable] = None,
        get_fill_status_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Async two-phase execution.
        All callable arguments must be async.
        """
        self._phase1_rungs, self._phase2_rungs = self.generate_rungs(
            total_quantity, reference_price, side,
        )
        start = time.monotonic()
        side = side.lower()

        total_filled = 0.0
        total_value = 0.0
        total_slippage = 0.0

        # ── Phase 1: Market fill ──
        for rung in self._phase1_rungs:
            try:
                if asyncio.iscoroutinefunction(place_order_fn):
                    result = await place_order_fn(symbol, side, rung.qty, reference_price, True)
                else:
                    result = place_order_fn(symbol, side, rung.qty, reference_price, True)
                rung.order_id = result.get("order_id") if isinstance(result, dict) else str(result)
                rung.submitted_at = time.time()

                if rung.order_id and get_fill_status_fn:
                    if asyncio.iscoroutinefunction(get_fill_status_fn):
                        fill = await self._wait_for_fill_async(
                            rung.order_id, get_fill_status_fn, rung.qty,
                        )
                    else:
                        fill = self._wait_for_fill_sync(
                            rung.order_id, get_fill_status_fn, rung.qty,
                        )
                    rung.filled_qty = fill.get("filled_qty", rung.qty)
                    rung.fill_price = fill.get("fill_price", reference_price)
                    rung.state = RungState.FILLED if rung.filled_qty >= rung.qty * 0.99 else RungState.PARTIAL
                    total_filled += rung.filled_qty
                    total_value += rung.filled_qty * (rung.fill_price or reference_price)
                else:
                    rung.filled_qty = rung.qty
                    rung.fill_price = reference_price
                    rung.state = RungState.FILLED
                    total_filled += rung.qty
                    total_value += rung.qty * reference_price
            except Exception as exc:
                rung.state = RungState.FAILED
                logger.error("[Ladder] P1 rung %d failed: %s", rung.rung_id, exc)

        # ── Phase 2: Limit rungs ──
        for rung in self._phase2_rungs:
            if rung.state != RungState.PENDING:
                continue
            for attempt in range(self.retry_limit):
                try:
                    if asyncio.iscoroutinefunction(place_order_fn):
                        result = await place_order_fn(symbol, side, rung.qty, rung.price, False)
                    else:
                        result = place_order_fn(symbol, side, rung.qty, rung.price, False)
                    rung.order_id = result.get("order_id") if isinstance(result, dict) else str(result)
                    rung.submitted_at = time.time()
                    rung.attempts = attempt + 1
                    rung.state = RungState.SUBMITTED

                    if rung.order_id and get_fill_status_fn:
                        if asyncio.iscoroutinefunction(get_fill_status_fn):
                            fill = await self._wait_for_fill_async(
                                rung.order_id, get_fill_status_fn, rung.qty,
                            )
                        else:
                            fill = self._wait_for_fill_sync(
                                rung.order_id, get_fill_status_fn, rung.qty,
                            )
                        rung.filled_qty = fill.get("filled_qty", 0.0)
                        rung.fill_price = fill.get("fill_price", rung.price)
                        if rung.filled_qty >= rung.qty * 0.99:
                            rung.state = RungState.FILLED
                            total_filled += rung.filled_qty
                            total_value += rung.filled_qty * (rung.fill_price or rung.price)
                        elif rung.filled_qty > 0:
                            rung.state = RungState.PARTIAL
                            total_filled += rung.filled_qty
                            total_value += rung.filled_qty * (rung.fill_price or rung.price)
                        else:
                            rung.state = RungState.CANCELLED
                    break
                except Exception as exc:
                    if attempt == self.retry_limit - 1:
                        rung.state = RungState.FAILED

        elapsed_ms = (time.monotonic() - start) * 1000.0
        vwap = total_value / total_filled if total_filled > 0 else reference_price
        slippage_bps = (total_slippage / total_filled) if total_filled > 0 else 0.0
        fill_rate = total_filled / total_quantity if total_quantity > 0 else 0.0

        self._result = {
            "success": fill_rate > 0.5,
            "fill_rate": fill_rate,
            "total_filled": total_filled,
            "total_quantity": total_quantity,
            "vwap": vwap,
            "slippage_bps": round(slippage_bps, 2),
            "execution_time_ms": round(elapsed_ms, 2),
            "phase1_filled": sum(r.filled_qty for r in self._phase1_rungs),
            "phase2_filled": sum(r.filled_qty for r in self._phase2_rungs),
            "phase1_rungs": len(self._phase1_rungs),
            "phase2_rungs": len(self._phase2_rungs),
            "phase1_rungs_data": [r.to_dict() for r in self._phase1_rungs],
            "phase2_rungs_data": [r.to_dict() for r in self._phase2_rungs],
            "failed_rungs": sum(
                1 for r in self._phase1_rungs + self._phase2_rungs
                if r.state == RungState.FAILED
            ),
        }

        logger.info(
            "[Ladder] Async execution: fill_rate=%.2f%% vwap=%.4f "
            "slippage=%.2fbps time=%.0fms",
            fill_rate * 100, vwap, slippage_bps, elapsed_ms,
        )

        return dict(self._result)

    # ═══════════════════════════════════════════════════════════════════════════
    # FILL WAIT HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _wait_for_fill_sync(
        self,
        order_id: str,
        get_fill_status_fn: Callable[[str], Dict[str, Any]],
        expected_qty: float,
    ) -> Dict[str, Any]:
        """Poll order status synchronously every 300ms."""
        interval = self.fill_poll_interval_ms / 1000.0
        for _ in range(self.max_poll_cycles):
            try:
                status = get_fill_status_fn(order_id)
                filled = status.get("filled_qty", 0.0)
                if filled >= expected_qty * 0.99:
                    return status
                if status.get("status") in ("FILLED", "CANCELED"):
                    return status
            except Exception:
                pass
            time.sleep(interval)
        return {"filled_qty": 0.0, "fill_price": 0.0, "status": "TIMEOUT"}

    async def _wait_for_fill_async(
        self,
        order_id: str,
        get_fill_status_fn: Callable,
        expected_qty: float,
    ) -> Dict[str, Any]:
        """Poll order status asynchronously every 300ms."""
        interval = self.fill_poll_interval_ms / 1000.0
        for _ in range(self.max_poll_cycles):
            try:
                if asyncio.iscoroutinefunction(get_fill_status_fn):
                    status = await get_fill_status_fn(order_id)
                else:
                    status = get_fill_status_fn(order_id)
                filled = status.get("filled_qty", 0.0)
                if filled >= expected_qty * 0.99:
                    return status
                if status.get("status") in ("FILLED", "CANCELED"):
                    return status
            except Exception:
                pass
            await asyncio.sleep(interval)
        return {"filled_qty": 0.0, "fill_price": 0.0, "status": "TIMEOUT"}

    # ═══════════════════════════════════════════════════════════════════════════
    # PROPERTIES
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def is_complete(self) -> bool:
        return bool(self._result)

    @property
    def phase1_filled(self) -> float:
        return sum(r.filled_qty for r in self._phase1_rungs)

    @property
    def phase2_filled(self) -> float:
        return sum(r.filled_qty for r in self._phase2_rungs)

    @property
    def failed_rungs(self) -> int:
        return sum(
            1 for r in self._phase1_rungs + self._phase2_rungs
            if r.state == RungState.FAILED
        )
