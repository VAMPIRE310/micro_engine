"""
NEO SUPREME — Correlation Hedge Engine
=======================================
Cross-symbol hedging based on rolling correlation analysis.

Components:
    CorrelationMatrix  — Rolling EWM correlation tracker across symbols
    HedgingEngine      — Main engine generating hedge recommendations
    HedgeRatioCalculator — Multiple hedge ratio computation methods

Hedge Types:
    DELTA_NEUTRAL  — Offset portfolio delta exposure
    BETA_NEUTRAL   — Hedge using market beta
    CORRELATION    — Direct correlation-pair hedge
    PAIRS_TRADE    — Statistical arbitrage pair
    CROSS_ASSET    — Cross-asset class hedge
    SECTOR         — Sector-based hedge

Features:
    - Rolling EWM correlation with configurable decay
    - Minimum variance portfolio calculation
    - Staleness detection (>5 min triggers warning)
    - Hedge ratio clamped to [0.1, 2.0]
    - Priority-ranked recommendations

Author: NEO SUPREME
Version: 4.0.0
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("correlation_hedge_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class HedgeType(Enum):
    DELTA_NEUTRAL = "delta_neutral"
    BETA_NEUTRAL = "beta_neutral"
    CORRELATION = "correlation"
    PAIRS_TRADE = "pairs_trade"
    CROSS_ASSET = "cross_asset"
    SECTOR = "sector"


class HedgePriority(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class HedgeRecommendation:
    """A single hedge recommendation with full metadata."""

    primary_symbol: str
    hedge_symbol: str
    hedge_type: HedgeType
    hedge_ratio: float
    primary_size: float
    hedge_size: float
    correlation: float
    confidence: float
    expected_cost: float
    priority: HedgePriority
    rationale: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_symbol": self.primary_symbol,
            "hedge_symbol": self.hedge_symbol,
            "hedge_type": self.hedge_type.value,
            "hedge_ratio": round(self.hedge_ratio, 6),
            "primary_size": round(self.primary_size, 6),
            "hedge_size": round(self.hedge_size, 6),
            "correlation": round(self.correlation, 4),
            "confidence": round(self.confidence, 4),
            "expected_cost": round(self.expected_cost, 4),
            "priority": self.priority.name,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
        }


@dataclass
class Position:
    """Tracked position for hedge calculations."""

    symbol: str
    size: float
    entry_price: float
    current_price: float
    direction: str           # "long" or "short"
    delta: float = 1.0
    beta: float = 1.0
    sector: str = ""

    @property
    def notional_value(self) -> float:
        return abs(self.size) * self.current_price

    @property
    def market_value(self) -> float:
        sign = 1.0 if self.direction.lower() in ("long", "buy") else -1.0
        return sign * self.notional_value

    def update_price(self, price: float) -> None:
        self.current_price = price


@dataclass
class CorrelationPair:
    """Correlation metrics between two symbols."""

    symbol1: str
    symbol2: str
    correlation: float
    beta: float
    r_squared: float
    lookback_period: int = 100

    def is_strong(self, threshold: float = 0.7) -> bool:
        return abs(self.correlation) >= threshold

    def is_perfect(self) -> bool:
        return abs(self.correlation) >= 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

class CorrelationMatrix:
    """
    Rolling Exponentially-Weighted Moving (EWM) correlation tracker.
    Maintains price history per symbol and computes correlations on demand.
    """

    def __init__(self, decay_factor: float = 0.94, max_history: int = 500):
        self.decay_factor = decay_factor
        self.max_history = max_history
        self._prices: Dict[str, deque] = {}
        self._returns: Dict[str, deque] = {}
        self._last_update: Dict[str, float] = {}

    def add_price(self, symbol: str, price: float) -> None:
        """Add a price point for a symbol."""
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.max_history)
            self._returns[symbol] = deque(maxlen=self.max_history)

        # Calculate return if we have a previous price
        if self._prices[symbol]:
            prev = self._prices[symbol][-1]
            if prev > 0:
                ret = math.log(price / prev)
                self._returns[symbol].append(ret)

        self._prices[symbol].append(price)
        self._last_update[symbol] = time.time()

    def add_prices(self, prices: Dict[str, float]) -> None:
        """Batch add prices for multiple symbols."""
        for sym, px in prices.items():
            self.add_price(sym, px)

    def get_price_history(self, symbol: str) -> Optional[np.ndarray]:
        """Get price history array for a symbol."""
        if symbol not in self._prices:
            return None
        return np.array(self._prices[symbol])

    def get_return_history(self, symbol: str) -> Optional[np.ndarray]:
        """Get return history array for a symbol."""
        if symbol not in self._returns or len(self._returns[symbol]) < 2:
            return None
        return np.array(self._returns[symbol])

    def calculate_correlation(
        self, sym1: str, sym2: str
    ) -> Optional[CorrelationPair]:
        """
        Calculate EWM correlation between two symbols.
        Returns CorrelationPair or None if insufficient data.
        """
        r1 = self.get_return_history(sym1)
        r2 = self.get_return_history(sym2)
        if r1 is None or r2 is None:
            return None

        n = min(len(r1), len(r2))
        if n < 10:
            return None

        r1, r2 = r1[-n:], r2[-n:]

        # Exponential weights (more recent = higher weight)
        weights = np.array([self.decay_factor ** (len(r1) - 1 - i) for i in range(len(r1))])
        weights_sum = weights.sum()
        if weights_sum <= 0:
            return None
        weights /= weights_sum

        # Weighted means
        m1 = np.dot(weights, r1)
        m2 = np.dot(weights, r2)

        # Weighted covariance and variances
        cov = np.dot(weights, (r1 - m1) * (r2 - m2))
        var1 = np.dot(weights, (r1 - m1) ** 2)
        var2 = np.dot(weights, (r2 - m2) ** 2)

        if var1 < 1e-12 or var2 < 1e-12:
            return None

        corr = cov / math.sqrt(var1 * var2)
        beta = cov / var1 if var1 > 1e-12 else 1.0
        r_sq = corr ** 2

        return CorrelationPair(
            sym1, sym2,
            float(np.clip(corr, -1.0, 1.0)),
            float(beta),
            float(r_sq),
            lookback_period=n,
        )

    def get_all_correlations(self, symbol: str) -> List[CorrelationPair]:
        """Get all correlation pairs for a symbol against all tracked symbols."""
        results = []
        for sym2 in self.get_symbols():
            if sym2 == symbol:
                continue
            pair = self.calculate_correlation(symbol, sym2)
            if pair:
                results.append(pair)
        return sorted(results, key=lambda p: abs(p.correlation), reverse=True)

    def get_symbols(self) -> List[str]:
        return list(self._prices.keys())

    def get_stale_symbols(self, max_age_seconds: float = 300.0) -> List[str]:
        """Return symbols with stale data (>5 min old)."""
        now = time.time()
        return [
            s for s, ts in self._last_update.items()
            if now - ts > max_age_seconds
        ]

    def to_matrix(self, symbols: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """Return full correlation matrix as a numpy array."""
        syms = symbols or self.get_symbols()
        n = len(syms)
        if n < 2:
            return None
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                pair = self.calculate_correlation(syms[i], syms[j])
                if pair:
                    matrix[i, j] = matrix[j, i] = pair.correlation
                else:
                    matrix[i, j] = matrix[j, i] = 0.0
        return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# HEDGE RATIO CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class HedgeRatioCalculator:
    """Multiple hedge ratio computation methods."""

    @staticmethod
    def ols_regression(returns1: np.ndarray, returns2: np.ndarray) -> float:
        """OLS hedge ratio from returns."""
        if len(returns1) < 5 or len(returns2) < 5:
            return 1.0
        min_len = min(len(returns1), len(returns2))
        r1, r2 = returns1[-min_len:], returns2[-min_len:]
        cov = np.cov(r1, r2)
        if cov[1, 1] < 1e-12:
            return 1.0
        return float(cov[0, 1] / cov[1, 1])

    @staticmethod
    def volatility_weighted(vol1: float, vol2: float) -> float:
        """Volatility-weighted hedge ratio."""
        if vol2 < 1e-12:
            return 1.0
        return float(vol1 / vol2)

    @staticmethod
    def beta_adjusted(
        beta: float, size1: float, price1: float, price2: float
    ) -> float:
        """Beta-adjusted hedge ratio."""
        if price2 < 1e-12:
            return 1.0
        return float(beta * (size1 * price1) / price2)

    @staticmethod
    def delta_neutral(delta1: float, delta2: float) -> float:
        """Delta-neutral hedge ratio."""
        if abs(delta2) < 1e-12:
            return 1.0
        return float(abs(delta1 / delta2))

    @staticmethod
    def minimum_variance(
        cov_matrix: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate minimum variance portfolio weights.
        Uses pseudo-inverse for numerical stability.
        """
        n = cov_matrix.shape[0]
        if n < 2:
            return np.ones(1)
        try:
            inv_cov = np.linalg.pinv(cov_matrix)
            ones = np.ones(n)
            raw = inv_cov @ ones
            if raw.sum() == 0:
                return np.ones(n) / n
            return raw / raw.sum()
        except Exception:
            return np.ones(n) / n


# ═══════════════════════════════════════════════════════════════════════════════
# HEDGING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HedgingEngine:
    """
    Main hedging engine. Tracks positions and generates hedge recommendations
    using correlation analysis.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        min_history: int = 50,
        max_hedge_cost_pct: float = 0.005,
        refresh_interval: float = 300.0,
    ):
        self.correlation_threshold = correlation_threshold
        self.min_history = min_history
        self.max_hedge_cost_pct = max_hedge_cost_pct
        self.refresh_interval = refresh_interval

        self.positions: Dict[str, Position] = {}
        self.correlation_matrix = CorrelationMatrix()
        self.ratio_calc = HedgeRatioCalculator()
        self._last_matrix_refresh: float = 0.0

    # ── Position Management ──────────────────────────────────────────────────

    def add_position(self, position: Position) -> None:
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)

    def update_position_price(self, symbol: str, price: float) -> None:
        if symbol in self.positions:
            self.positions[symbol].update_price(price)

    def update_prices(self, prices: Dict[str, float]) -> None:
        self.correlation_matrix.add_prices(prices)
        for sym, px in prices.items():
            if sym in self.positions:
                self.positions[sym].update_price(px)
        self._last_matrix_refresh = time.time()

    # ── Hedge Recommendations ────────────────────────────────────────────────

    def generate_hedge_recommendations(self) -> List[HedgeRecommendation]:
        """
        Generate prioritized hedge recommendations for all positions.
        Returns sorted list by priority (highest first).
        """
        recs: List[HedgeRecommendation] = []
        if not self.positions:
            return recs

        # Portfolio delta exposure
        port_delta = self._calculate_portfolio_delta()
        if abs(port_delta) < 1e-6:
            return recs

        for sym, pos in list(self.positions.items()):
            # Try correlation hedge first
            rec = self._find_correlation_hedge(pos)
            if rec:
                recs.append(rec)
            else:
                # Fallback to beta-neutral
                rec = self._find_beta_neutral_hedge(pos)
                if rec:
                    recs.append(rec)

        # Sort by priority descending
        recs.sort(key=lambda r: r.priority.value, reverse=True)
        return recs

    def _calculate_portfolio_delta(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            sign = 1.0 if pos.direction.lower() in ("long", "buy") else -1.0
            total += sign * pos.delta * pos.notional_value
        return total

    def _find_correlation_hedge(
        self, pos: Position
    ) -> Optional[HedgeRecommendation]:
        """Find best correlation-based hedge for a position."""
        best_pair: Optional[CorrelationPair] = None
        best_corr = 0.0

        all_pairs = self.correlation_matrix.get_all_correlations(pos.symbol)
        for pair in all_pairs:
            if pair.is_strong(self.correlation_threshold) and abs(pair.correlation) > best_corr:
                best_corr = abs(pair.correlation)
                best_pair = pair

        if not best_pair:
            return None

        hedge_sym = (
            best_pair.symbol2 if best_pair.symbol1 == pos.symbol else best_pair.symbol1
        )
        ratio = abs(best_pair.beta) if abs(best_pair.beta) > 1e-6 else 1.0
        hedge_sz = pos.size * ratio
        cost_pct = 0.001 * 2  # Taker both sides

        priority = (
            HedgePriority.CRITICAL if abs(best_pair.correlation) > 0.9
            else HedgePriority.HIGH if abs(best_pair.correlation) > 0.8
            else HedgePriority.MEDIUM
        )

        return HedgeRecommendation(
            primary_symbol=pos.symbol,
            hedge_symbol=hedge_sym,
            hedge_type=HedgeType.CORRELATION,
            hedge_ratio=ratio,
            primary_size=pos.size,
            hedge_size=hedge_sz,
            correlation=best_pair.correlation,
            confidence=best_pair.r_squared,
            expected_cost=cost_pct,
            priority=priority,
            rationale=(
                f"Correlation hedge via {hedge_sym} "
                f"(rho={best_pair.correlation:.2f}, beta={best_pair.beta:.2f})"
            ),
        )

    def _find_beta_neutral_hedge(
        self, pos: Position
    ) -> Optional[HedgeRecommendation]:
        """Find beta-neutral hedge using BTC as market proxy."""
        if abs(pos.beta) < 1e-6:
            return None

        market_sym = "BTCUSDT" if "BTCUSDT" in self.correlation_matrix.get_symbols() else None
        if not market_sym or market_sym == pos.symbol:
            return None

        pair = self.correlation_matrix.calculate_correlation(pos.symbol, market_sym)
        if not pair:
            return None

        ratio = abs(pair.beta) if abs(pair.beta) > 1e-6 else 1.0
        hedge_sz = pos.size * ratio

        return HedgeRecommendation(
            primary_symbol=pos.symbol,
            hedge_symbol=market_sym,
            hedge_type=HedgeType.BETA_NEUTRAL,
            hedge_ratio=ratio,
            primary_size=pos.size,
            hedge_size=hedge_sz,
            correlation=pair.correlation,
            confidence=pair.r_squared,
            expected_cost=0.002,
            priority=HedgePriority.LOW,
            rationale=f"Beta-neutral hedge via {market_sym} (beta={pair.beta:.2f})",
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def get_hedge_ratio(self, symbol: str) -> float:
        """
        Get correlation-based hedge ratio for a symbol.
        Returns abs(beta) of best-correlated pair, clamped to [0.1, 2.0].
        Falls back to 1.0 if no strong pair found.
        Logs warning if matrix data is stale (>5 min).
        """
        now = time.time()
        if (self._last_matrix_refresh > 0
                and (now - self._last_matrix_refresh) > self.refresh_interval):
            logger.warning(
                "[HedgeEngine] Correlation matrix stale (%.0fs since last refresh)",
                now - self._last_matrix_refresh,
            )

        stale = self.correlation_matrix.get_stale_symbols(self.refresh_interval)
        if stale:
            logger.debug("[HedgeEngine] Stale symbols: %s", stale)

        best_beta = 1.0
        best_corr = 0.0

        for sym2 in self.correlation_matrix.get_symbols():
            if sym2 == symbol:
                continue
            pair = self.correlation_matrix.calculate_correlation(symbol, sym2)
            if pair and pair.is_strong(self.correlation_threshold):
                if abs(pair.correlation) > best_corr:
                    best_corr = abs(pair.correlation)
                    best_beta = abs(pair.beta) if abs(pair.beta) > 1e-6 else 1.0

        clamped = max(0.1, min(best_beta, 2.0))
        logger.debug(
            "[HedgeEngine] get_hedge_ratio(%s) = %.4f (corr=%.3f)",
            symbol, clamped, best_corr,
        )
        return clamped

    def get_correlation(self, sym1: str, sym2: str) -> Optional[float]:
        """Get raw correlation between two symbols."""
        pair = self.correlation_matrix.calculate_correlation(sym1, sym2)
        return pair.correlation if pair else None

    def get_minimum_variance_weights(
        self, symbols: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Calculate minimum variance portfolio weights for given symbols.
        Returns dict of symbol -> weight, or None if insufficient data.
        """
        matrix = self.correlation_matrix.to_matrix(symbols)
        if matrix is None:
            return None
        weights = self.ratio_calc.minimum_variance(matrix)
        return {sym: float(w) for sym, w in zip(symbols, weights)}

    def mark_matrix_refreshed(self) -> None:
        """Call after feeding new price data to reset staleness."""
        self._last_matrix_refresh = time.time()

    def get_status(self) -> Dict[str, Any]:
        return {
            "positions_tracked": len(self.positions),
            "symbols_in_matrix": len(self.correlation_matrix.get_symbols()),
            "correlation_threshold": self.correlation_threshold,
            "last_refresh": self._last_matrix_refresh,
            "stale": (time.time() - self._last_matrix_refresh > self.refresh_interval)
            if self._last_matrix_refresh > 0 else True,
        }
