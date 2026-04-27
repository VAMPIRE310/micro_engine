"""
NEO SUPREME — Position Scaling Engine
======================================
Advanced position scaling strategies for dynamic entry/exit management.

Strategies:
    DCAStrategy         — Dollar-cost average at defined drawdown levels
    PyramidingStrategy  — Add to winners at profit levels with optional trailing stop
    ProfitScaleOutStrategy — Scale out portions at profit targets, keep runner

Engine:
    ScalingEngine       — Per-symbol registry, unified interface

Each strategy implements:
    update_price(entry, current, is_long) -> List[ScalingAction]
    calculate_scale_in() -> Recommended add-on quantity
    reset() -> Clear state

Author: NEO SUPREME
Version: 4.0.0
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("position_scaling_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ScalingActionType(Enum):
    SCALE_IN = "scale_in"           # Add to position
    SCALE_OUT = "scale_out"         # Reduce position
    CLOSE = "close"                 # Fully close position
    TAKE_PROFIT = "take_profit"     # Partial take-profit
    STOP_LOSS = "stop_loss"         # Stop triggered
    TRAIL_ADJUST = "trail_adjust"   # Adjust trailing stop
    NO_ACTION = "no_action"         # No action needed


@dataclass
class ScalingAction:
    """A single scaling action with full context."""

    action_type: ScalingActionType
    symbol: str
    side: str
    size: float
    price: float
    reason: str
    remaining_position: float
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action_type.value,
            "symbol": self.symbol,
            "side": self.side,
            "size": round(self.size, 6),
            "price": self.price,
            "reason": self.reason,
            "remaining_position": round(self.remaining_position, 6),
            "total_pnl": round(self.total_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BASE STRATEGY CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BaseScalingStrategy(ABC):
    """Abstract base class for all scaling strategies."""

    def __init__(self, symbol: str, side: str, initial_size: float, entry_price: float):
        self.symbol = symbol
        self.side = side.upper()
        self.initial_size = initial_size
        self.entry_price = entry_price
        self.is_long = self.side in ("LONG", "BUY")
        self.current_size = initial_size
        self.current_price = entry_price
        self.total_pnl = 0.0
        self._actions_history: List[ScalingAction] = []
        self._is_complete = False

    @abstractmethod
    def update_price(self, entry: float, current: float, is_long: bool) -> List[ScalingAction]:
        """Evaluate current price and return list of triggered actions."""
        ...

    @abstractmethod
    def calculate_scale_in(self, current_price: float) -> float:
        """Calculate recommended add-on quantity at current price."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset strategy state for reuse."""
        ...

    def _pnl(self, entry: float, current: float, size: float) -> float:
        if self.is_long:
            return (current - entry) * size
        return (entry - current) * size

    def _pct(self, entry: float, current: float) -> float:
        if entry <= 0:
            return 0.0
        if self.is_long:
            return (current - entry) / entry
        return (entry - current) / entry

    def get_history(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._actions_history]

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def unrealized_pnl(self) -> float:
        return self._pnl(self.entry_price, self.current_price, self.current_size)


# ═══════════════════════════════════════════════════════════════════════════════
# DCA STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class DCAStrategy(BaseScalingStrategy):
    """
    Dollar-Cost Averaging at defined drawdown levels.

    Adds to position at progressively lower (for longs) / higher (for shorts)
    price levels to lower average entry cost.

    Configurable:
        drawdown_pcts: List of drawdown percentages to trigger DCA
        scale_multipliers: Size multiplier for each DCA level
        max_scale_ins: Maximum number of DCA additions
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        initial_size: float,
        entry_price: float,
        drawdown_pcts: Optional[List[float]] = None,
        scale_multipliers: Optional[List[float]] = None,
        max_scale_ins: int = 5,
    ):
        super().__init__(symbol, side, initial_size, entry_price)
        self.drawdown_pcts = drawdown_pcts or [0.01, 0.02, 0.04, 0.08, 0.16]
        self.scale_multipliers = scale_multipliers or [1.0, 1.5, 2.0, 3.0, 5.0]
        self.max_scale_ins = max_scale_ins

        self._scale_in_count = 0
        self._triggered_levels: set = set()
        self._total_invested = initial_size * entry_price
        self._total_qty = initial_size

    def update_price(self, entry: float, current: float, is_long: bool) -> List[ScalingAction]:
        """Check drawdown levels and trigger DCA if conditions met."""
        self.current_price = current
        actions: List[ScalingAction] = []

        if self._is_complete or self._scale_in_count >= self.max_scale_ins:
            return actions

        current_pct = self._pct(entry, current)

        for i, dd_pct in enumerate(self.drawdown_pcts):
            if i in self._triggered_levels:
                continue
            if current_pct >= dd_pct:
                # Trigger DCA at this level
                multiplier = self.scale_multipliers[min(i, len(self.scale_multipliers) - 1)]
                base_size = self.initial_size
                scale_size = base_size * multiplier

                self._triggered_levels.add(i)
                self._scale_in_count += 1
                self.current_size += scale_size

                # Update average entry
                self._total_invested += scale_size * current
                self._total_qty += scale_size
                new_entry = self._total_invested / self._total_qty
                old_entry = self.entry_price
                self.entry_price = new_entry

                action = ScalingAction(
                    action_type=ScalingActionType.SCALE_IN,
                    symbol=self.symbol,
                    side=self.side,
                    size=scale_size,
                    price=current,
                    reason=f"DCA Level {i + 1}: drawdown {current_pct * 100:.1f}% >= {dd_pct * 100:.1f}%",
                    remaining_position=self.current_size,
                    total_pnl=self.total_pnl,
                    unrealized_pnl=self.unrealized_pnl,
                )
                actions.append(action)
                self._actions_history.append(action)

                logger.info(
                    "[DCA] %s %s | Level %d | DCA @ %.4f | "
                    "Size: +%.4f | New avg entry: %.4f (was %.4f) | Total: %.4f",
                    self.symbol, self.side, i + 1, current, scale_size,
                    new_entry, old_entry, self.current_size,
                )

                if self._scale_in_count >= self.max_scale_ins:
                    break

        return actions

    def calculate_scale_in(self, current_price: float) -> float:
        """Return the size for the next DCA level if one is available."""
        if self._is_complete or self._scale_in_count >= self.max_scale_ins:
            return 0.0
        idx = self._scale_in_count
        multiplier = self.scale_multipliers[min(idx, len(self.scale_multipliers) - 1)]
        return self.initial_size * multiplier

    def reset(self) -> None:
        self._scale_in_count = 0
        self._triggered_levels = set()
        self._total_invested = self.initial_size * self.entry_price
        self._total_qty = self.initial_size
        self.current_size = self.initial_size
        self._is_complete = False
        self._actions_history = []


# ═══════════════════════════════════════════════════════════════════════════════
# PYRAMIDING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class PyramidingStrategy(BaseScalingStrategy):
    """
    Add to winning positions at profit thresholds.

    Pyramid into winning trades at defined profit levels with optional
    trailing stop adjustment after each scale-in.

    Configurable:
        profit_pcts: List of profit percentages to trigger add
        scale_factors: Size factor for each pyramid level
        max_pyramids: Maximum number of pyramids
        trailing_stop_pct: Trailing stop distance
        use_trailing_stop: Whether to adjust trailing stop after each add
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        initial_size: float,
        entry_price: float,
        profit_pcts: Optional[List[float]] = None,
        scale_factors: Optional[List[float]] = None,
        max_pyramids: int = 4,
        trailing_stop_pct: float = 0.02,
        use_trailing_stop: bool = True,
    ):
        super().__init__(symbol, side, initial_size, entry_price)
        self.profit_pcts = profit_pcts or [0.01, 0.02, 0.04, 0.08]
        self.scale_factors = scale_factors or [0.5, 0.5, 0.75, 1.0]
        self.max_pyramids = max_pyramids
        self.trailing_stop_pct = trailing_stop_pct
        self.use_trailing_stop = use_trailing_stop

        self._pyramid_count = 0
        self._triggered_levels: set = set()
        self._best_price = entry_price
        self._trailing_stop = (
            entry_price * (1.0 - trailing_stop_pct)
            if self.is_long
            else entry_price * (1.0 + trailing_stop_pct)
        )
        self._total_invested = initial_size * entry_price

    def update_price(self, entry: float, current: float, is_long: bool) -> List[ScalingAction]:
        """Check profit levels and trailing stop, trigger actions as needed."""
        self.current_price = current
        actions: List[ScalingAction] = []

        if self._is_complete:
            return actions

        # Update best price
        if self.is_long and current > self._best_price:
            self._best_price = current
        elif not self.is_long and current < self._best_price:
            self._best_price = current

        # Update trailing stop
        if self.use_trailing_stop:
            if self.is_long:
                new_stop = self._best_price * (1.0 - self.trailing_stop_pct)
                if new_stop > self._trailing_stop:
                    old_stop = self._trailing_stop
                    self._trailing_stop = new_stop
                    actions.append(ScalingAction(
                        action_type=ScalingActionType.TRAIL_ADJUST,
                        symbol=self.symbol, side=self.side,
                        size=0.0, price=current,
                        reason=f"Trailing stop raised: {old_stop:.4f} -> {new_stop:.4f}",
                        remaining_position=self.current_size,
                    ))
            else:
                new_stop = self._best_price * (1.0 + self.trailing_stop_pct)
                if new_stop < self._trailing_stop:
                    old_stop = self._trailing_stop
                    self._trailing_stop = new_stop
                    actions.append(ScalingAction(
                        action_type=ScalingActionType.TRAIL_ADJUST,
                        symbol=self.symbol, side=self.side,
                        size=0.0, price=current,
                        reason=f"Trailing stop lowered: {old_stop:.4f} -> {new_stop:.4f}",
                        remaining_position=self.current_size,
                    ))

        # Check trailing stop trigger
        if self.is_long and current <= self._trailing_stop:
            self._is_complete = True
            action = ScalingAction(
                action_type=ScalingActionType.STOP_LOSS,
                symbol=self.symbol, side=self.side,
                size=self.current_size, price=current,
                reason=f"Trailing stop hit at {self._trailing_stop:.4f}",
                remaining_position=0.0,
                total_pnl=self.total_pnl,
                unrealized_pnl=self.unrealized_pnl,
            )
            actions.append(action)
            self._actions_history.append(action)
            self.current_size = 0.0
            logger.info("[Pyramid] %s %s STOPPED | Price: %.4f | Trail: %.4f",
                        self.symbol, self.side, current, self._trailing_stop)
            return actions
        elif not self.is_long and current >= self._trailing_stop:
            self._is_complete = True
            action = ScalingAction(
                action_type=ScalingActionType.STOP_LOSS,
                symbol=self.symbol, side=self.side,
                size=self.current_size, price=current,
                reason=f"Trailing stop hit at {self._trailing_stop:.4f}",
                remaining_position=0.0,
                total_pnl=self.total_pnl,
                unrealized_pnl=self.unrealized_pnl,
            )
            actions.append(action)
            self._actions_history.append(action)
            self.current_size = 0.0
            logger.info("[Pyramid] %s %s STOPPED | Price: %.4f | Trail: %.4f",
                        self.symbol, self.side, current, self._trailing_stop)
            return actions

        # Check profit pyramid levels
        if self._pyramid_count < self.max_pyramids:
            current_pct = self._pct(entry, current)
            for i, profit_pct in enumerate(self.profit_pcts):
                if i in self._triggered_levels:
                    continue
                if current_pct >= profit_pct:
                    factor = self.scale_factors[min(i, len(self.scale_factors) - 1)]
                    add_size = self.initial_size * factor

                    self._triggered_levels.add(i)
                    self._pyramid_count += 1
                    self.current_size += add_size
                    self._total_invested += add_size * current
                    new_entry = self._total_invested / self.current_size
                    old_entry = self.entry_price
                    self.entry_price = new_entry

                    action = ScalingAction(
                        action_type=ScalingActionType.SCALE_IN,
                        symbol=self.symbol,
                        side=self.side,
                        size=add_size,
                        price=current,
                        reason=(
                            f"Pyramid Level {i + 1}: profit {current_pct * 100:.1f}% "
                            f">= {profit_pct * 100:.1f}%"
                        ),
                        remaining_position=self.current_size,
                        total_pnl=self.total_pnl,
                        unrealized_pnl=self.unrealized_pnl,
                    )
                    actions.append(action)
                    self._actions_history.append(action)

                    logger.info(
                        "[Pyramid] %s %s | Level %d | Add @ %.4f | "
                        "Size: +%.4f | New avg: %.4f (was %.4f) | Total: %.4f",
                        self.symbol, self.side, i + 1, current, add_size,
                        new_entry, old_entry, self.current_size,
                    )

                    if self._pyramid_count >= self.max_pyramids:
                        break

        return actions

    def calculate_scale_in(self, current_price: float) -> float:
        if self._is_complete or self._pyramid_count >= self.max_pyramids:
            return 0.0
        idx = self._pyramid_count
        factor = self.scale_factors[min(idx, len(self.scale_factors) - 1)]
        return self.initial_size * factor

    @property
    def trailing_stop(self) -> float:
        return self._trailing_stop

    def reset(self) -> None:
        self._pyramid_count = 0
        self._triggered_levels = set()
        self._best_price = self.entry_price
        self._trailing_stop = (
            self.entry_price * (1.0 - self.trailing_stop_pct)
            if self.is_long
            else self.entry_price * (1.0 + self.trailing_stop_pct)
        )
        self._total_invested = self.initial_size * self.entry_price
        self.current_size = self.initial_size
        self._is_complete = False
        self._actions_history = []


# ═══════════════════════════════════════════════════════════════════════════════
# PROFIT SCALE-OUT STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class ProfitScaleOutStrategy(BaseScalingStrategy):
    """
    Scale out portions of position at defined profit targets.
    Keeps a "runner" position after initial exits.

    Configurable:
        profit_targets: List of profit % to trigger scale-out
        scale_out_pcts: Portion to sell at each target (fraction of remaining)
        final_runner_pct: Final portion to keep as runner
        final_tp_pct: Take-profit level for final runner
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        initial_size: float,
        entry_price: float,
        profit_targets: Optional[List[float]] = None,
        scale_out_pcts: Optional[List[float]] = None,
        final_runner_pct: float = 0.20,
        final_tp_pct: Optional[float] = None,
    ):
        super().__init__(symbol, side, initial_size, entry_price)
        self.profit_targets = profit_targets or [0.015, 0.03, 0.06, 0.12]
        self.scale_out_pcts = scale_out_pcts or [0.25, 0.25, 0.25, 0.50]
        self.final_runner_pct = final_runner_pct
        self.final_tp_pct = final_tp_pct or self.profit_targets[-1] * 2 if self.profit_targets else 0.20

        self._targets_hit = 0
        self._triggered_levels: set = set()
        self._realized_pnl = 0.0
        self._original_size = initial_size

    def update_price(self, entry: float, current: float, is_long: bool) -> List[ScalingAction]:
        """Check profit targets and trigger scale-outs."""
        self.current_price = current
        actions: List[ScalingAction] = []

        if self._is_complete or self.current_size <= 0:
            return actions

        current_pct = self._pct(entry, current)

        # Check scale-out targets
        for i, tp in enumerate(self.profit_targets):
            if i in self._triggered_levels:
                continue
            if current_pct >= tp:
                self._triggered_levels.add(i)
                self._targets_hit += 1

                # Determine scale-out size
                scale_pct = self.scale_out_pcts[min(i, len(self.scale_out_pcts) - 1)]
                remaining_pct = self.current_size / self._original_size

                if remaining_pct <= self.final_runner_pct:
                    # Already at runner size, skip
                    continue

                max_reduce = self.current_size - (self._original_size * self.final_runner_pct)
                reduce_qty = min(self.current_size * scale_pct, max_reduce)

                if reduce_qty <= 0:
                    continue

                realized = self._pnl(self.entry_price, current, reduce_qty)
                self._realized_pnl += realized
                self.total_pnl += realized
                self.current_size -= reduce_qty

                action = ScalingAction(
                    action_type=ScalingActionType.SCALE_OUT,
                    symbol=self.symbol,
                    side=self.side,
                    size=reduce_qty,
                    price=current,
                    reason=(
                        f"TP Level {i + 1}: profit {current_pct * 100:.1f}% "
                        f">= {tp * 100:.1f}%"
                    ),
                    remaining_position=self.current_size,
                    total_pnl=self.total_pnl,
                    unrealized_pnl=self.unrealized_pnl,
                )
                actions.append(action)
                self._actions_history.append(action)

                logger.info(
                    "[ScaleOut] %s %s | TP%d @ %.4f | Scaled out: %.4f | "
                    "Realized: %+.2f | Remaining: %.4f",
                    self.symbol, self.side, i + 1, current,
                    reduce_qty, realized, self.current_size,
                )

        # Check final take-profit for runner
        if self.final_tp_pct and self.current_size > 0:
            if current_pct >= self.final_tp_pct:
                realized = self._pnl(self.entry_price, current, self.current_size)
                self._realized_pnl += realized
                self.total_pnl += realized

                action = ScalingAction(
                    action_type=ScalingActionType.TAKE_PROFIT,
                    symbol=self.symbol,
                    side=self.side,
                    size=self.current_size,
                    price=current,
                    reason=f"Final TP hit: profit {current_pct * 100:.1f}% >= {self.final_tp_pct * 100:.1f}%",
                    remaining_position=0.0,
                    total_pnl=self.total_pnl,
                    unrealized_pnl=0.0,
                )
                actions.append(action)
                self._actions_history.append(action)
                self.current_size = 0.0
                self._is_complete = True

                logger.info(
                    "[ScaleOut] %s %s | FINAL TP @ %.4f | Size: %.4f | Total PnL: %+.2f",
                    self.symbol, self.side, current, action.size, self.total_pnl,
                )

        return actions

    def calculate_scale_in(self, current_price: float) -> float:
        """No additional scale-in for scale-out strategy."""
        return 0.0

    def reset(self) -> None:
        self._targets_hit = 0
        self._triggered_levels = set()
        self._realized_pnl = 0.0
        self.current_size = self._original_size
        self.total_pnl = 0.0
        self._is_complete = False
        self._actions_history = []

    @property
    def runner_size(self) -> float:
        """Current runner portion size."""
        return self.current_size

    @property
    def scaled_out_size(self) -> float:
        """Total quantity that has been scaled out."""
        return self._original_size - self.current_size

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl


# ═══════════════════════════════════════════════════════════════════════════════
# SCALING ENGINE — UNIFIED REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class ScalingEngine:
    """
    Per-symbol registry of scaling strategies with unified interface.

    Manages multiple strategies across symbols, providing:
        - update_price(symbol, entry, current, is_long) -> List[ScalingAction]
        - register_strategy(symbol, strategy) -> None
        - get_strategy(symbol) -> BaseScalingStrategy
        - get_all_actions() -> Dict[str, List[ScalingAction]]
    """

    def __init__(self):
        self._strategies: Dict[str, BaseScalingStrategy] = {}
        self._default_type: str = "dca"

    # ── Registration ─────────────────────────────────────────────────────────

    def register_strategy(
        self, symbol: str, strategy: BaseScalingStrategy
    ) -> None:
        """Register a scaling strategy for a symbol."""
        self._strategies[symbol] = strategy
        logger.info(
            "[ScalingEngine] Registered %s strategy for %s",
            type(strategy).__name__, symbol,
        )

    def create_dca(
        self,
        symbol: str,
        side: str,
        initial_size: float,
        entry_price: float,
        drawdown_pcts: Optional[List[float]] = None,
        **kwargs,
    ) -> DCAStrategy:
        """Create and register a DCA strategy."""
        strategy = DCAStrategy(
            symbol=symbol, side=side,
            initial_size=initial_size, entry_price=entry_price,
            drawdown_pcts=drawdown_pcts,
            **kwargs,
        )
        self.register_strategy(symbol, strategy)
        return strategy

    def create_pyramid(
        self,
        symbol: str,
        side: str,
        initial_size: float,
        entry_price: float,
        profit_pcts: Optional[List[float]] = None,
        **kwargs,
    ) -> PyramidingStrategy:
        """Create and register a pyramiding strategy."""
        strategy = PyramidingStrategy(
            symbol=symbol, side=side,
            initial_size=initial_size, entry_price=entry_price,
            profit_pcts=profit_pcts,
            **kwargs,
        )
        self.register_strategy(symbol, strategy)
        return strategy

    def create_scale_out(
        self,
        symbol: str,
        side: str,
        initial_size: float,
        entry_price: float,
        profit_targets: Optional[List[float]] = None,
        **kwargs,
    ) -> ProfitScaleOutStrategy:
        """Create and register a profit scale-out strategy."""
        strategy = ProfitScaleOutStrategy(
            symbol=symbol, side=side,
            initial_size=initial_size, entry_price=entry_price,
            profit_targets=profit_targets,
            **kwargs,
        )
        self.register_strategy(symbol, strategy)
        return strategy

    # ── Unified Interface ────────────────────────────────────────────────────

    def update_price(
        self, symbol: str, entry: float, current: float, is_long: bool
    ) -> List[ScalingAction]:
        """
        Update price for a symbol and evaluate all strategies.

        Args:
            symbol: Trading pair
            entry: Entry price
            current: Current market price
            is_long: True for long, False for short

        Returns:
            List of triggered ScalingAction objects.
        """
        strategy = self._strategies.get(symbol)
        if strategy is None:
            return []

        actions = strategy.update_price(entry, current, is_long)

        if actions:
            logger.info(
                "[ScalingEngine] %s generated %d actions at %.4f",
                symbol, len(actions), current,
            )

        return actions

    def calculate_scale_in(self, symbol: str, current_price: float) -> float:
        """Get recommended scale-in quantity for a symbol."""
        strategy = self._strategies.get(symbol)
        if strategy is None:
            return 0.0
        return strategy.calculate_scale_in(current_price)

    def get_strategy(self, symbol: str) -> Optional[BaseScalingStrategy]:
        return self._strategies.get(symbol)

    def remove_strategy(self, symbol: str) -> bool:
        """Remove a strategy for a symbol. Returns True if removed."""
        if symbol in self._strategies:
            del self._strategies[symbol]
            logger.info("[ScalingEngine] Removed strategy for %s", symbol)
            return True
        return False

    # ── Bulk Operations ──────────────────────────────────────────────────────

    def update_all_prices(
        self, prices: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[ScalingAction]]:
        """
        Batch update prices for multiple symbols.

        Args:
            prices: Dict of {symbol: {"entry": float, "current": float, "is_long": bool}}

        Returns:
            Dict of {symbol: [ScalingAction, ...]}
        """
        all_actions: Dict[str, List[ScalingAction]] = {}
        for symbol, data in prices.items():
            if symbol not in self._strategies:
                continue
            actions = self.update_price(
                symbol,
                data["entry"],
                data["current"],
                data.get("is_long", True),
            )
            if actions:
                all_actions[symbol] = actions
        return all_actions

    def get_all_actions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get action history for all strategies."""
        return {
            sym: strat.get_history()
            for sym, strat in self._strategies.items()
        }

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all registered strategies."""
        return {
            sym: {
                "type": type(strat).__name__,
                "side": strat.side,
                "entry_price": strat.entry_price,
                "current_size": strat.current_size,
                "unrealized_pnl": strat.unrealized_pnl,
                "total_pnl": strat.total_pnl,
                "is_complete": strat.is_complete,
            }
            for sym, strat in self._strategies.items()
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset strategies for a symbol or all strategies."""
        if symbol:
            strat = self._strategies.get(symbol)
            if strat:
                strat.reset()
        else:
            for strat in self._strategies.values():
                strat.reset()

    def get_symbols(self) -> List[str]:
        return list(self._strategies.keys())

    @property
    def strategy_count(self) -> int:
        return len(self._strategies)
