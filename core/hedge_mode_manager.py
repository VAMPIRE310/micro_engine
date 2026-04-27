"""
NEO SUPREME — Hedge Mode Manager (Dual Long/Short Hedge)
=========================================================
Manages dual long/short hedge positions for a single symbol.
Tracks long and short legs independently with full PnL accounting.

Components:
    HedgePosition    — Dataclass tracking both legs with PnL
    HedgeManager     — Core hedge logic with open/close/update
    WalletManager    — Multi-wallet equity and drawdown tracker

Thread-safe throughout with RLock.

Author: NEO SUPREME
Version: 4.0.0
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hedge_mode_manager")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class HedgeState(Enum):
    NO_POSITION = "no_position"
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    FULL_HEDGE = "full_hedge"      # Equal long + short
    NET_LONG = "net_long"          # Long > Short
    NET_SHORT = "net_short"        # Short > Long


@dataclass
class HedgePosition:
    """
    Tracks both legs of a hedge position independently.
    Provides combined PnL and state classification.
    """

    symbol: str
    # Long leg
    long_size: float = 0.0
    long_entry: float = 0.0
    long_pnl: float = 0.0
    long_unrealized: float = 0.0
    long_stop: float = 0.0
    long_take_profit: float = 0.0
    # Short leg
    short_size: float = 0.0
    short_entry: float = 0.0
    short_pnl: float = 0.0
    short_unrealized: float = 0.0
    short_stop: float = 0.0
    short_take_profit: float = 0.0
    # Meta
    total_fees: float = 0.0
    open_timestamp: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    def update_pnl(self, current_price: float) -> None:
        """Recalculate PnL for both legs at current price."""
        if self.long_size > 0 and self.long_entry > 0:
            self.long_unrealized = (current_price - self.long_entry) * self.long_size
            self.long_pnl = self.long_unrealized
        if self.short_size > 0 and self.short_entry > 0:
            self.short_unrealized = (self.short_entry - current_price) * self.short_size
            self.short_pnl = self.short_unrealized
        self.last_update = time.time()

    @property
    def combined_pnl(self) -> float:
        return self.long_pnl + self.short_pnl

    @property
    def combined_unrealized(self) -> float:
        return self.long_unrealized + self.short_unrealized

    @property
    def state(self) -> HedgeState:
        has_long = self.long_size > 1e-9
        has_short = self.short_size > 1e-9
        if not has_long and not has_short:
            return HedgeState.NO_POSITION
        if has_long and not has_short:
            return HedgeState.LONG_ONLY
        if not has_long and has_short:
            return HedgeState.SHORT_ONLY
        if abs(self.long_size - self.short_size) < 1e-9:
            return HedgeState.FULL_HEDGE
        return HedgeState.NET_LONG if self.long_size > self.short_size else HedgeState.NET_SHORT

    @property
    def net_size(self) -> float:
        return self.long_size - self.short_size

    @property
    def gross_size(self) -> float:
        return self.long_size + self.short_size

    @property
    def hedge_ratio(self) -> float:
        if self.long_size == 0:
            return 0.0
        return self.short_size / self.long_size

    @property
    def net_direction(self) -> str:
        net = self.net_size
        if net > 1e-9:
            return "LONG"
        elif net < -1e-9:
            return "SHORT"
        return "FLAT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "state": self.state.value,
            "long_size": self.long_size,
            "long_entry": self.long_entry,
            "long_pnl": self.long_pnl,
            "long_unrealized": self.long_unrealized,
            "short_size": self.short_size,
            "short_entry": self.short_entry,
            "short_pnl": self.short_pnl,
            "short_unrealized": self.short_unrealized,
            "combined_pnl": self.combined_pnl,
            "combined_unrealized": self.combined_unrealized,
            "net_size": self.net_size,
            "gross_size": self.gross_size,
            "hedge_ratio": self.hedge_ratio,
            "net_direction": self.net_direction,
            "total_fees": self.total_fees,
            "duration_seconds": time.time() - self.open_timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HEDGE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class HedgeManager:
    """
    Dual-leg position manager for a single symbol.
    Tracks long/short independently and enforces hedge logic.

    Thread-safe with RLock.
    """

    def __init__(
        self,
        symbol: str,
        max_hedge_ratio: float = 1.0,
        min_profit_to_unwind: float = 0.0,
        fee_rate_per_trade: float = 0.00055,  # Taker fee
    ):
        self.symbol = symbol
        self.max_hedge_ratio = max_hedge_ratio
        self.min_profit_to_unwind = min_profit_to_unwind
        self.fee_rate = fee_rate_per_trade
        self._lock = threading.RLock()
        self.position = HedgePosition(symbol=symbol)
        self._trade_history: List[Dict[str, Any]] = []
        self._hedge_count = 0

    # ── Open Legs ────────────────────────────────────────────────────────────

    def open_long(
        self,
        size: float,
        entry_price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> bool:
        """Open or add to the long leg with average-in pricing."""
        with self._lock:
            if size <= 0 or entry_price <= 0:
                logger.warning("[HedgeMgr] %s invalid long params: size=%.4f price=%.4f",
                               self.symbol, size, entry_price)
                return False

            if self.position.long_size == 0:
                self.position.long_size = size
                self.position.long_entry = entry_price
            else:
                total = self.position.long_size + size
                self.position.long_entry = (
                    self.position.long_entry * self.position.long_size
                    + entry_price * size
                ) / total
                self.position.long_size = total

            self.position.long_stop = stop_loss if stop_loss > 0 else self.position.long_stop
            self.position.long_take_profit = (
                take_profit if take_profit > 0 else self.position.long_take_profit
            )
            self.position.total_fees += size * entry_price * self.fee_rate

            self._log_trade("open_long", size, entry_price)
            logger.info(
                "[HedgeMgr] %s LONG | size=%.4f @ %.4f | total_long=%.4f avg_entry=%.4f",
                self.symbol, size, entry_price, self.position.long_size,
                self.position.long_entry,
            )
            return True

    def open_short(
        self,
        size: float,
        entry_price: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> bool:
        """Open or add to the short leg with average-in pricing."""
        with self._lock:
            if size <= 0 or entry_price <= 0:
                logger.warning("[HedgeMgr] %s invalid short params: size=%.4f price=%.4f",
                               self.symbol, size, entry_price)
                return False

            ratio = (self.position.short_size + size) / max(self.position.long_size, 1e-9)
            if ratio > self.max_hedge_ratio:
                logger.warning(
                    "[HedgeMgr] %s hedge ratio %.2f > max %.2f — rejecting",
                    self.symbol, ratio, self.max_hedge_ratio,
                )
                return False

            if self.position.short_size == 0:
                self.position.short_size = size
                self.position.short_entry = entry_price
            else:
                total = self.position.short_size + size
                self.position.short_entry = (
                    self.position.short_entry * self.position.short_size
                    + entry_price * size
                ) / total
                self.position.short_size = total

            self.position.short_stop = stop_loss if stop_loss > 0 else self.position.short_stop
            self.position.short_take_profit = (
                take_profit if take_profit > 0 else self.position.short_take_profit
            )
            self.position.total_fees += size * entry_price * self.fee_rate
            self._hedge_count += 1

            self._log_trade("open_short", size, entry_price)
            logger.info(
                "[HedgeMgr] %s SHORT | size=%.4f @ %.4f | total_short=%.4f avg_entry=%.4f",
                self.symbol, size, entry_price, self.position.short_size,
                self.position.short_entry,
            )
            return True

    # ── Close Legs ───────────────────────────────────────────────────────────

    def close_long(
        self,
        size: Optional[float] = None,
        current_price: float = 0.0,
    ) -> Tuple[bool, float]:
        """
        Close all or part of the long leg.
        Returns (success, realized_pnl).
        """
        with self._lock:
            if self.position.long_size <= 0:
                return False, 0.0

            close_size = min(size or self.position.long_size, self.position.long_size)
            pnl = 0.0
            if current_price > 0:
                pnl = (current_price - self.position.long_entry) * close_size

            self.position.long_size -= close_size
            self.position.total_fees += close_size * current_price * self.fee_rate

            if self.position.long_size <= 1e-9:
                self.position.long_size = 0.0
                self.position.long_entry = 0.0

            self._log_trade("close_long", close_size, current_price, pnl=pnl)
            logger.info(
                "[HedgeMgr] %s CLOSE LONG | size=%.4f @ %.4f | pnl=%+.2f | remaining=%.4f",
                self.symbol, close_size, current_price, pnl, self.position.long_size,
            )
            return True, pnl

    def close_short(
        self,
        size: Optional[float] = None,
        current_price: float = 0.0,
    ) -> Tuple[bool, float]:
        """
        Close all or part of the short leg.
        Returns (success, realized_pnl).
        """
        with self._lock:
            if self.position.short_size <= 0:
                return False, 0.0

            close_size = min(size or self.position.short_size, self.position.short_size)
            pnl = 0.0
            if current_price > 0:
                pnl = (self.position.short_entry - current_price) * close_size

            self.position.short_size -= close_size
            self.position.total_fees += close_size * current_price * self.fee_rate

            if self.position.short_size <= 1e-9:
                self.position.short_size = 0.0
                self.position.short_entry = 0.0

            self._log_trade("close_short", close_size, current_price, pnl=pnl)
            logger.info(
                "[HedgeMgr] %s CLOSE SHORT | size=%.4f @ %.4f | pnl=%+.2f | remaining=%.4f",
                self.symbol, close_size, current_price, pnl, self.position.short_size,
            )
            return True, pnl

    # ── Partial Close by Percentage ──────────────────────────────────────────

    def close_long_pct(self, pct: float, current_price: float) -> Tuple[bool, float]:
        """Close a percentage (0-1) of the long leg."""
        with self._lock:
            if self.position.long_size <= 0 or pct <= 0:
                return False, 0.0
            size = self.position.long_size * min(pct, 1.0)
            return self.close_long(size, current_price)

    def close_short_pct(self, pct: float, current_price: float) -> Tuple[bool, float]:
        """Close a percentage (0-1) of the short leg."""
        with self._lock:
            if self.position.short_size <= 0 or pct <= 0:
                return False, 0.0
            size = self.position.short_size * min(pct, 1.0)
            return self.close_short(size, current_price)

    # ── Unwind Decision ──────────────────────────────────────────────────────

    def should_unwind_hedge(self, current_price: float) -> bool:
        """
        Returns True if combined PnL > min_profit_to_unwind and hedge leg exists.
        Called periodically to determine if the hedge should be closed.
        """
        self.update_prices(current_price)
        return (
            self.position.short_size > 0
            and self.position.combined_pnl > self.min_profit_to_unwind
        )

    def should_unwind_either_leg(self, current_price: float) -> Optional[str]:
        """
        Determine if either leg should be unwound based on profit/loss.
        Returns 'long', 'short', or None.
        """
        self.update_prices(current_price)

        # Unwind hedge (short) if it's profitable and covers long losses
        if self.position.short_size > 0 and self.position.short_pnl > 0:
            if self.position.combined_pnl > self.min_profit_to_unwind:
                return "short"

        # Unwind long if hedge is deep underwater
        if self.position.long_size > 0 and self.position.long_pnl < 0:
            long_dd = abs(self.position.long_pnl)
            short_profit = max(self.position.short_pnl, 0)
            if short_profit > long_dd * 0.5:
                return "long"

        return None

    # ── Price Updates ────────────────────────────────────────────────────────

    def update_prices(self, current_price: float) -> None:
        """Update PnL for both legs at current price."""
        with self._lock:
            self.position.update_pnl(current_price)

    # ── Stop / Take-Profit Checks ────────────────────────────────────────────

    def check_stops(self, current_price: float) -> List[Dict[str, Any]]:
        """Check if any stops or take-profits are hit."""
        triggered = []
        with self._lock:
            # Long stop
            if (self.position.long_size > 0 and self.position.long_stop > 0
                    and current_price <= self.position.long_stop):
                triggered.append({
                    "leg": "long", "type": "stop", "price": current_price,
                    "stop_price": self.position.long_stop,
                })
            # Long take-profit
            if (self.position.long_size > 0 and self.position.long_take_profit > 0
                    and current_price >= self.position.long_take_profit):
                triggered.append({
                    "leg": "long", "type": "take_profit", "price": current_price,
                    "tp_price": self.position.long_take_profit,
                })
            # Short stop
            if (self.position.short_size > 0 and self.position.short_stop > 0
                    and current_price >= self.position.short_stop):
                triggered.append({
                    "leg": "short", "type": "stop", "price": current_price,
                    "stop_price": self.position.short_stop,
                })
            # Short take-profit
            if (self.position.short_size > 0 and self.position.short_take_profit > 0
                    and current_price <= self.position.short_take_profit):
                triggered.append({
                    "leg": "short", "type": "take_profit", "price": current_price,
                    "tp_price": self.position.short_take_profit,
                })
        return triggered

    # ── State Accessors ──────────────────────────────────────────────────────

    def get_combined_pnl(self, current_price: float = 0.0) -> float:
        if current_price > 0:
            self.update_prices(current_price)
        return self.position.combined_pnl

    def get_state(self) -> HedgeState:
        return self.position.state

    def get_status(self) -> Dict[str, Any]:
        return self.position.to_dict()

    def is_hedged(self) -> bool:
        return self.position.state in (HedgeState.FULL_HEDGE, HedgeState.NET_SHORT)

    # ── Trade History ────────────────────────────────────────────────────────

    def _log_trade(self, action: str, size: float, price: float, pnl: float = 0.0) -> None:
        self._trade_history.append({
            "action": action,
            "size": size,
            "price": price,
            "pnl": pnl,
            "timestamp": time.time(),
        })

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._trade_history[-limit:]

    def reset(self) -> None:
        """Reset all position state."""
        with self._lock:
            self.position = HedgePosition(symbol=self.symbol)
            self._trade_history = []
            self._hedge_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# WALLET MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class WalletManager:
    """
    Multi-wallet tracker. Tracks balances, equity, drawdown per wallet.
    Thread-safe with RLock.
    """

    @dataclass
    class Wallet:
        name: str
        balance: float
        equity: float = 0.0
        unrealized_pnl: float = 0.0
        realized_pnl: float = 0.0
        peak_equity: float = 0.0
        current_drawdown: float = 0.0
        max_drawdown: float = 0.0
        trade_count: int = 0
        win_count: int = 0
        loss_count: int = 0

        def update_equity(self, unrealized: float = 0.0) -> None:
            self.unrealized_pnl = unrealized
            self.equity = self.balance + unrealized
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity
            if self.peak_equity > 0:
                self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        def record_trade(self, pnl: float) -> None:
            self.trade_count += 1
            if pnl > 0:
                self.win_count += 1
            elif pnl < 0:
                self.loss_count += 1

        def to_dict(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "balance": round(self.balance, 4),
                "equity": round(self.equity, 4),
                "unrealized_pnl": round(self.unrealized_pnl, 4),
                "realized_pnl": round(self.realized_pnl, 4),
                "peak_equity": round(self.peak_equity, 4),
                "current_drawdown": round(self.current_drawdown, 6),
                "max_drawdown": round(self.max_drawdown, 6),
                "trade_count": self.trade_count,
                "win_count": self.win_count,
                "loss_count": self.loss_count,
                "win_rate": (self.win_count / self.trade_count if self.trade_count > 0 else 0),
            }

    def __init__(self):
        self._wallets: Dict[str, WalletManager.Wallet] = {}
        self._lock = threading.RLock()

    def add_wallet(self, name: str, initial_balance: float) -> None:
        with self._lock:
            w = self.Wallet(name=name, balance=initial_balance, peak_equity=initial_balance)
            w.update_equity()
            self._wallets[name] = w
            logger.info("[WalletMgr] Added wallet '%s' balance=%.2f", name, initial_balance)

    def update_balance(self, name: str, new_balance: float, unrealized: float = 0.0) -> None:
        with self._lock:
            if name not in self._wallets:
                self.add_wallet(name, new_balance)
            self._wallets[name].balance = new_balance
            self._wallets[name].update_equity(unrealized)

    def record_realized_pnl(self, name: str, pnl: float) -> None:
        with self._lock:
            if name in self._wallets:
                self._wallets[name].balance += pnl
                self._wallets[name].realized_pnl += pnl
                self._wallets[name].record_trade(pnl)

    def get_total_balance(self) -> float:
        with self._lock:
            return sum(w.balance for w in self._wallets.values())

    def get_total_equity(self) -> float:
        with self._lock:
            return sum(w.equity for w in self._wallets.values())

    def get_total_drawdown(self) -> float:
        with self._lock:
            total_peak = sum(w.peak_equity for w in self._wallets.values())
            total_eq = sum(w.equity for w in self._wallets.values())
            if total_peak > 0:
                return (total_peak - total_eq) / total_peak
            return 0.0

    def get_wallet(self, name: str) -> Optional[Wallet]:
        with self._lock:
            return self._wallets.get(name)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                name: w.to_dict()
                for name, w in self._wallets.items()
            }

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate summary across all wallets."""
        with self._lock:
            stats = self.get_stats()
            total_balance = sum(w["balance"] for w in stats.values())
            total_equity = sum(w["equity"] for w in stats.values())
            total_realized = sum(w["realized_pnl"] for w in stats.values())
            total_trades = sum(w["trade_count"] for w in stats.values())
            total_wins = sum(w["win_count"] for w in stats.values())
            max_dd = max((w["max_drawdown"] for w in stats.values()), default=0.0)
            return {
                "total_balance": round(total_balance, 4),
                "total_equity": round(total_equity, 4),
                "total_realized_pnl": round(total_realized, 4),
                "total_trades": total_trades,
                "total_wins": total_wins,
                "win_rate": total_wins / total_trades if total_trades > 0 else 0,
                "max_drawdown": round(max_dd, 6),
                "wallet_count": len(self._wallets),
                "wallets": stats,
            }
