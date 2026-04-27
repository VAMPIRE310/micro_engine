"""
NEO SUPREME — Order Flow Indicators Module
===========================================
Complete order flow analysis framework using Polars for vectorized operations.

Indicators:
- OrderBookImbalanceIndicator: Multi-depth imbalance (5, 10, 20 levels)
- CVDIndicator: Cumulative Volume Delta with divergence detection
- TradeFlowImbalanceIndicator: Buy/sell ratio over lookback
- VolumeProfileVisibleRangeIndicator: POC, Value Area
- BidAskSpreadIndicator: Spread analysis with liquidity quality
- LargeTradeTrackerIndicator: Whale order detection
- OrderFlowCompositeIndicator: Weighted composite of all signals

All indicators return IndicatorResult with:
    name, value, signal (BULLISH/BEARISH/NEUTRAL), confidence, interpretation, metadata

All vectorized operations use Polars DataFrames (not pandas).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

logger = logging.getLogger("neo_supreme.orderflow_indicators")


# ── Enums ──────────────────────────────────────────────────────────────────────

class SignalDirection(Enum):
    """Directional signal classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class IndicatorType(Enum):
    """Indicator category classification."""
    ORDERFLOW = "orderflow"
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    COMPOSITE = "composite"


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class IndicatorResult:
    """
    Standard result container for all indicator calculations.

    Attributes
    ----------
    name : str
        Indicator name
    value : any
        Primary indicator value (numeric, dict, or series)
    signal : SignalDirection
        BULLISH, BEARISH, or NEUTRAL
    confidence : float
        Confidence level [0, 1]
    interpretation : str
        Human-readable interpretation
    metadata : dict
        Additional computed values
    """
    name: str
    value: Any
    signal: SignalDirection
    confidence: float = 0.0
    interpretation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Base Indicator Class ──────────────────────────────────────────────────────

class BaseIndicator:
    """
    Base class for all indicators.

    Subclasses must implement ``calculate()`` and optionally
    ``calculate_vectorized()`` for DataFrame augmentation.
    """

    def __init__(self, name: str, indicator_type: IndicatorType):
        self.name = name
        self.indicator_type = indicator_type

    def calculate(self, data: pl.DataFrame) -> IndicatorResult:
        """Calculate the indicator from a Polars DataFrame."""
        raise NotImplementedError

    def calculate_vectorized(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add indicator columns to a DataFrame (vectorized)."""
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# Order Book Imbalance Indicator
# ═══════════════════════════════════════════════════════════════════════════════

class OrderBookImbalanceIndicator(BaseIndicator):
    """
    Multi-depth order book imbalance indicator.

    Computes bid/ask imbalance at multiple depth levels (5, 10, 20 by default).
    Values near +1 indicate strong bid dominance (buying pressure).
    Values near -1 indicate strong ask dominance (selling pressure).
    """

    def __init__(self, depth_levels: Optional[List[int]] = None, threshold: float = 0.2):
        super().__init__("OrderBookImbalance", IndicatorType.ORDERFLOW)
        self.depth_levels = depth_levels or [5, 10, 20]
        self.threshold = threshold

    def calculate(self, data: pl.DataFrame) -> IndicatorResult:
        """Calculate order book imbalance from bid/ask data."""
        results = {}

        for depth in self.depth_levels:
            bid_col = f"bid_volume_{depth}"
            ask_col = f"ask_volume_{depth}"

            if bid_col not in data.columns or ask_col not in data.columns:
                continue

            bid_vol = data[bid_col].tail(1).to_list()[0]
            ask_vol = data[ask_col].tail(1).to_list()[0]

            if bid_vol + ask_vol > 0:
                imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
            else:
                imbalance = 0.0

            results[f"depth_{depth}"] = {
                "imbalance": imbalance,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
            }

        if not results:
            return IndicatorResult(
                name=self.name,
                value={"error": "order book data not available"},
                signal=SignalDirection.NEUTRAL,
                confidence=0.0,
                interpretation="Order book data not available",
            )

        deepest = list(results.keys())[-1]
        imbalance = results[deepest]["imbalance"]

        if imbalance > self.threshold:
            signal = SignalDirection.BULLISH
            interpretation = f"Strong bid imbalance at {deepest}: {imbalance:.2%}"
        elif imbalance < -self.threshold:
            signal = SignalDirection.BEARISH
            interpretation = f"Strong ask imbalance at {deepest}: {imbalance:.2%}"
        else:
            signal = SignalDirection.NEUTRAL
            interpretation = f"Balanced order book at {deepest}: {imbalance:.2%}"

        return IndicatorResult(
            name=self.name,
            value=results,
            signal=signal,
            confidence=min(abs(imbalance) / 0.5, 1.0),
            interpretation=interpretation,
            metadata={
                "bid_dominant": imbalance > self.threshold,
                "ask_dominant": imbalance < -self.threshold,
                "deepest_imbalance": imbalance,
            },
        )

    def calculate_vectorized(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add order book imbalance columns to DataFrame."""
        for depth in self.depth_levels:
            bid_col = f"bid_volume_{depth}"
            ask_col = f"ask_volume_{depth}"

            if bid_col in data.columns and ask_col in data.columns:
                bid = pl.col(bid_col)
                ask = pl.col(ask_col)
                imbalance = (bid - ask) / (bid + ask)
                data = data.with_columns([
                    imbalance.alias(f"ob_imbalance_{depth}")
                ])
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# CVD Indicator
# ═══════════════════════════════════════════════════════════════════════════════

class CVDIndicator(BaseIndicator):
    """
    Cumulative Volume Delta indicator with divergence detection.

    Tracks net buying vs selling pressure over time.
    Detects bullish/bearish divergences between CVD and price.
    """

    def __init__(self, period: int = 14):
        super().__init__(f"CVD_{period}", IndicatorType.ORDERFLOW)
        self.period = period

    def calculate(self, data: pl.DataFrame) -> IndicatorResult:
        """Calculate CVD with divergence detection."""
        if "buy_volume" not in data.columns or "sell_volume" not in data.columns:
            return IndicatorResult(
                name=self.name,
                value={"error": "buy_volume or sell_volume not found"},
                signal=SignalDirection.NEUTRAL,
                confidence=0.0,
                interpretation="Trade flow data not available",
            )

        buy_vol = data["buy_volume"]
        sell_vol = data["sell_volume"]
        close = data["close"]

        # Volume delta and cumulative
        delta = buy_vol - sell_vol
        cvd = delta.cum_sum()

        current_cvd = cvd.tail(1).to_list()[0]
        prev_cvd = (
            cvd.shift(self.period).tail(1).to_list()[0]
            if len(cvd) > self.period
            else current_cvd
        )
        cvd_change = current_cvd - prev_cvd

        # Price trend
        current_price = close.tail(1).to_list()[0]
        prev_price = (
            close.shift(self.period).tail(1).to_list()[0]
            if len(close) > self.period
            else current_price
        )
        price_change = current_price - prev_price

        # Divergence detection
        bullish_divergence = cvd_change > 0 and price_change < 0
        bearish_divergence = cvd_change < 0 and price_change > 0

        if cvd_change > 0:
            signal = SignalDirection.BULLISH
            interpretation = "CVD rising - buying pressure"
        else:
            signal = SignalDirection.BEARISH
            interpretation = "CVD falling - selling pressure"

        if bullish_divergence:
            interpretation += " - BULLISH DIVERGENCE"
        elif bearish_divergence:
            interpretation += " - BEARISH DIVERGENCE"

        return IndicatorResult(
            name=self.name,
            value={
                "cvd": current_cvd,
                "cvd_change": cvd_change,
                "buy_volume": buy_vol.tail(1).to_list()[0],
                "sell_volume": sell_vol.tail(1).to_list()[0],
            },
            signal=signal,
            confidence=min(abs(cvd_change) / abs(current_cvd) * 10, 1.0)
            if current_cvd != 0
            else 0.5,
            interpretation=interpretation,
            metadata={
                "bullish_divergence": bullish_divergence,
                "bearish_divergence": bearish_divergence,
                "buying_pressure": cvd_change > 0,
            },
        )

    def calculate_vectorized(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add CVD and CVD SMA columns to DataFrame."""
        if "buy_volume" in data.columns and "sell_volume" in data.columns:
            buy = pl.col("buy_volume")
            sell = pl.col("sell_volume")
            delta = buy - sell
            cvd = delta.cum_sum()
            cvd_sma = cvd.rolling_mean(window_size=self.period)

            return data.with_columns([
                cvd.alias("cvd"),
                cvd_sma.alias(f"cvd_sma_{self.period}"),
            ])
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Flow Imbalance Indicator
# ═══════════════════════════════════════════════════════════════════════════════

class TradeFlowImbalanceIndicator(BaseIndicator):
    """
    Trade flow (market order) imbalance indicator.

    Computes buy/sell ratio over a lookback window.
    Values > 0.2 indicate buying dominance.
    Values < -0.2 indicate selling dominance.
    """

    def __init__(self, lookback: int = 20):
        super().__init__(f"TradeFlow_{lookback}", IndicatorType.ORDERFLOW)
        self.lookback = lookback

    def calculate(self, data: pl.DataFrame) -> IndicatorResult:
        """Calculate trade flow imbalance over lookback window."""
        if "buy_volume" not in data.columns or "sell_volume" not in data.columns:
            return IndicatorResult(
                name=self.name,
                value={"error": "buy_volume or sell_volume not found"},
                signal=SignalDirection.NEUTRAL,
                confidence=0.0,
                interpretation="Trade flow data not available",
            )

        buy_vol = data["buy_volume"].tail(self.lookback).sum()
        sell_vol = data["sell_volume"].tail(self.lookback).sum()
        total_vol = buy_vol + sell_vol

        if total_vol > 0:
            buy_ratio = buy_vol / total_vol
            sell_ratio = sell_vol / total_vol
            imbalance = buy_ratio - sell_ratio
        else:
            buy_ratio = 0.5
            sell_ratio = 0.5
            imbalance = 0.0

        if imbalance > 0.2:
            signal = SignalDirection.BULLISH
            interpretation = f"Strong buying flow: {buy_ratio:.1%} buy vs {sell_ratio:.1%} sell"
        elif imbalance < -0.2:
            signal = SignalDirection.BEARISH
            interpretation = f"Strong selling flow: {sell_ratio:.1%} sell vs {buy_ratio:.1%} buy"
        else:
            signal = SignalDirection.NEUTRAL
            interpretation = f"Balanced trade flow: {buy_ratio:.1%} buy vs {sell_ratio:.1%} sell"

        return IndicatorResult(
            name=self.name,
            value={
                "buy_ratio": buy_ratio,
                "sell_ratio": sell_ratio,
                "imbalance": imbalance,
                "total_volume": total_vol,
            },
            signal=signal,
            confidence=min(abs(imbalance) / 0.4, 1.0),
            interpretation=interpretation,
            metadata={
                "buy_dominant": imbalance > 0.2,
                "sell_dominant": imbalance < -0.2,
            },
        )

    def calculate_vectorized(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add trade flow imbalance columns to DataFrame."""
        if "buy_volume" in data.columns and "sell_volume" in data.columns:
            buy = pl.col("buy_volume")
            sell = pl.col("sell_volume")

            buy_sum = buy.rolling_sum(window_size=self.lookback)
            sell_sum = sell.rolling_sum(window_size=self.lookback)
            total = buy_sum + sell_sum

            buy_ratio = buy_sum / total
            imbalance = (buy_sum - sell_sum) / total

            return data.with_columns([
                buy_ratio.alias("trade_buy_ratio"),
                imbalance.alias("trade_flow_imbalance"),
            ])
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# Volume Profile Visible Range Indicator
# ═══════════════════════════════════════════════════════════════════════════════

class VolumeProfileVisibleRangeIndicator(BaseIndicator):
    """
    Volume Profile Visible Range (VPVR) indicator.

    Computes Point of Control (POC) and Value Area from price/volume histogram.
    Signal depends on current price position relative to POC.
    """

    def __init__(self, bins: int = 24):
        super().__init__(f"VPVR_{bins}", IndicatorType.ORDERFLOW)
        self.bins = bins

    def calculate(self, data: pl.DataFrame) -> IndicatorResult:
        """Calculate volume profile with POC and Value Area."""
        close = data["close"]
        volume = data["volume"] if "volume" in data.columns else pl.Series([0] * len(close))

        price_min = close.min()
        price_max = close.max()
        current_price = close.tail(1).to_list()[0]

        if price_min == price_max:
            return IndicatorResult(
                name=self.name,
                value={"error": "insufficient price variation"},
                signal=SignalDirection.NEUTRAL,
                confidence=0.0,
                interpretation="Insufficient price variation for VPVR",
            )

        bin_size = (price_max - price_min) / self.bins

        volumes_list = volume.to_list()
        prices_list = close.to_list()

        bin_volumes = {}
        for i in range(self.bins):
            bin_low = price_min + i * bin_size
            bin_high = bin_low + bin_size
            bin_volumes[f"bin_{i}"] = sum(
                v for p, v in zip(prices_list, volumes_list)
                if bin_low <= p < bin_high
            )

        # Point of Control
        poc_bin = max(bin_volumes.items(), key=lambda x: x[1])
        poc_price = price_min + int(poc_bin[0].split("_")[1]) * bin_size + bin_size / 2

        # Value Area (70% of volume)
        total_volume = sum(bin_volumes.values())
        sorted_bins = sorted(bin_volumes.items(), key=lambda x: x[1], reverse=True)

        va_volume = 0.0
        va_bins = []
        for bin_name, vol in sorted_bins:
            va_volume += vol
            va_bins.append(int(bin_name.split("_")[1]))
            if va_volume >= total_volume * 0.7:
                break

        va_low = price_min + min(va_bins) * bin_size
        va_high = price_min + max(va_bins) * bin_size + bin_size

        # Signal
        if current_price > poc_price * 1.02:
            signal = SignalDirection.BULLISH
            interpretation = f"Price above POC ({poc_price:.4f}) - bullish"
        elif current_price < poc_price * 0.98:
            signal = SignalDirection.BEARISH
            interpretation = f"Price below POC ({poc_price:.4f}) - bearish"
        else:
            signal = SignalDirection.NEUTRAL
            interpretation = f"Price near POC ({poc_price:.4f})"

        return IndicatorResult(
            name=self.name,
            value={
                "poc": poc_price,
                "value_area_low": va_low,
                "value_area_high": va_high,
                "bin_volumes": bin_volumes,
            },
            signal=signal,
            confidence=min(abs(current_price - poc_price) / poc_price * 50, 1.0),
            interpretation=interpretation,
            metadata={
                "above_poc": current_price > poc_price,
                "in_value_area": va_low <= current_price <= va_high,
            },
        )

    def calculate_vectorized(self, data: pl.DataFrame) -> pl.DataFrame:
        """VPVR requires histogram computation — simplified passthrough."""
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# Bid-Ask Spread Indicator
# ═══════════════════════════════════════════════════════════════════════════════

class BidAskSpreadIndicator(BaseIndicator):
    """
    Bid-ask spread analysis indicator.

    Measures spread in basis points and classifies liquidity quality.
    Direction-agnostic — provides liquidity context rather than directional signal.
    """

    def __init__(self):
        super().__init__("BidAskSpread", IndicatorType.ORDERFLOW)

    def calculate(self, data: pl.DataFrame) -> IndicatorResult:
        """Calculate spread analysis from bid/ask data."""
        if "best_bid" not in data.columns or "best_ask" not in data.columns:
            return IndicatorResult(
                name=self.name,
                value={"error": "best_bid or best_ask not found"},
                signal=SignalDirection.NEUTRAL,
                confidence=0.0,
                interpretation="Spread data not available",
            )

        bid = data["best_bid"].tail(1).to_list()[0]
        ask = data["best_ask"].tail(1).to_list()[0]
        mid = (bid + ask) / 2.0

        spread = ask - bid
        spread_bps = (spread / mid) * 10000.0 if mid > 0 else 0.0

        # Liquidity quality classification
        if spread_bps < 1.0:
            liquidity = "excellent"
        elif spread_bps < 5.0:
            liquidity = "good"
        elif spread_bps < 10.0:
            liquidity = "moderate"
        else:
            liquidity = "poor"

        return IndicatorResult(
            name=self.name,
            value={
                "spread": spread,
                "spread_bps": spread_bps,
                "best_bid": bid,
                "best_ask": ask,
                "mid_price": mid,
                "liquidity": liquidity,
            },
            signal=SignalDirection.NEUTRAL,
            confidence=0.5,
            interpretation=f"Spread: {spread_bps:.1f} bps - {liquidity} liquidity",
            metadata={
                "tight_spread": spread_bps < 5.0,
                "wide_spread": spread_bps > 10.0,
                "liquidity_quality": liquidity,
            },
        )

    def calculate_vectorized(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add spread columns to DataFrame."""
        if "best_bid" in data.columns and "best_ask" in data.columns:
            bid = pl.col("best_bid")
            ask = pl.col("best_ask")

            spread = ask - bid
            mid = (bid + ask) / 2.0
            spread_bps = (spread / mid) * 10000.0

            return data.with_columns([
                spread.alias("bid_ask_spread"),
                spread_bps.alias("spread_bps"),
                mid.alias("mid_price"),
            ])
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# Large Trade Tracker Indicator
# ═══════════════════════════════════════════════════════════════════════════════

class LargeTradeTrackerIndicator(BaseIndicator):
    """
    Large trade (whale order) tracking indicator.

    Detects trades exceeding a volume threshold multiplier vs recent average.
    Classifies direction from accompanying price movement.
    """

    def __init__(self, threshold_multiplier: float = 3.0):
        super().__init__(
            f"LargeTrades_{threshold_multiplier}x", IndicatorType.ORDERFLOW
        )
        self.threshold_multiplier = threshold_multiplier

    def calculate(self, data: pl.DataFrame) -> IndicatorResult:
        """Detect and classify large trades."""
        if "volume" not in data.columns:
            return IndicatorResult(
                name=self.name,
                value={"error": "volume not found"},
                signal=SignalDirection.NEUTRAL,
                confidence=0.0,
                interpretation="Volume data not available",
            )

        volume = data["volume"]
        close = data["close"]

        avg_volume = volume.tail(50).mean()
        threshold = avg_volume * self.threshold_multiplier

        recent_volumes = volume.tail(20).to_list()
        recent_prices = close.tail(20).to_list()

        large_buys = []
        large_sells = []

        for i in range(1, len(recent_volumes)):
            if recent_volumes[i] > threshold:
                price_change = (
                    (recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
                    if recent_prices[i - 1] != 0
                    else 0.0
                )

                if price_change > 0:
                    large_buys.append({
                        "volume": recent_volumes[i],
                        "price": recent_prices[i],
                        "price_change_pct": price_change * 100.0,
                    })
                else:
                    large_sells.append({
                        "volume": recent_volumes[i],
                        "price": recent_prices[i],
                        "price_change_pct": price_change * 100.0,
                    })

        if len(large_buys) > len(large_sells):
            signal = SignalDirection.BULLISH
            interpretation = f"{len(large_buys)} large buys vs {len(large_sells)} large sells"
        elif len(large_sells) > len(large_buys):
            signal = SignalDirection.BEARISH
            interpretation = f"{len(large_sells)} large sells vs {len(large_buys)} large buys"
        else:
            signal = SignalDirection.NEUTRAL
            interpretation = "Balanced large trades"

        return IndicatorResult(
            name=self.name,
            value={
                "large_buys": len(large_buys),
                "large_sells": len(large_sells),
                "threshold": threshold,
                "buy_details": large_buys[:3],
                "sell_details": large_sells[:3],
            },
            signal=signal,
            confidence=min(abs(len(large_buys) - len(large_sells)) / 5.0, 1.0),
            interpretation=interpretation,
            metadata={
                "buy_pressure": len(large_buys) > len(large_sells),
                "sell_pressure": len(large_sells) > len(large_buys),
            },
        )

    def calculate_vectorized(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add large trade detection column to DataFrame."""
        if "volume" in data.columns:
            volume = pl.col("volume")
            avg_volume = volume.rolling_mean(window_size=50)
            threshold = avg_volume * self.threshold_multiplier
            large_trade = volume > threshold

            return data.with_columns([
                large_trade.alias("large_trade"),
            ])
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# Order Flow Composite Indicator
# ═══════════════════════════════════════════════════════════════════════════════

class OrderFlowCompositeIndicator(BaseIndicator):
    """
    Weighted composite of all orderflow signals.

    Combines CVD, order book imbalance, and large trade signals into a
    single composite score [-1, +1] with confidence weighting.

    Weights:
        - CVD component: 40%
        - Order book imbalance: 40%
        - Large trades: 20%
    """

    def __init__(self):
        super().__init__("OrderFlowComposite", IndicatorType.COMPOSITE)

    def calculate(self, data: pl.DataFrame) -> IndicatorResult:
        """Calculate weighted composite of all orderflow signals."""
        signals = []
        weights = []

        # CVD component (40%)
        if "buy_volume" in data.columns and "sell_volume" in data.columns:
            buy_vol = data["buy_volume"].tail(20).sum()
            sell_vol = data["sell_volume"].tail(20).sum()
            if buy_vol + sell_vol > 0:
                cvd_signal = (buy_vol - sell_vol) / (buy_vol + sell_vol)
                signals.append(cvd_signal)
                weights.append(0.4)

        # Order book imbalance component (40%)
        ob_imbalance = 0.0
        for depth in [10, 20]:
            bid_col = f"bid_volume_{depth}"
            ask_col = f"ask_volume_{depth}"
            if bid_col in data.columns and ask_col in data.columns:
                bid = data[bid_col].tail(1).to_list()[0]
                ask = data[ask_col].tail(1).to_list()[0]
                if bid + ask > 0:
                    ob_imbalance = (bid - ask) / (bid + ask)

        if ob_imbalance != 0:
            signals.append(ob_imbalance)
            weights.append(0.4)

        # Large trades component (20%)
        if "volume" in data.columns:
            volume = data["volume"]
            avg_vol = volume.tail(50).mean()
            recent_large = sum(
                1 for v in volume.tail(10).to_list() if v > avg_vol * 3
            )
            if recent_large > 0:
                close = data["close"]
                recent_prices = close.tail(2).to_list()
                if len(recent_prices) >= 2 and recent_prices[0] != 0:
                    price_change_pct = (recent_prices[-1] / recent_prices[0] - 1) * 100.0
                    large_signal = 1.0 if price_change_pct > 0 else -1.0
                    signals.append(large_signal * min(recent_large / 5.0, 1.0))
                    weights.append(0.2)

        # Calculate composite
        if signals and weights:
            total_weight = sum(weights)
            composite = sum(s * w for s, w in zip(signals, weights)) / total_weight

            if composite > 0.3:
                signal = SignalDirection.BULLISH
            elif composite < -0.3:
                signal = SignalDirection.BEARISH
            else:
                signal = SignalDirection.NEUTRAL

            interpretation = f"Order flow composite: {composite:.3f}"
        else:
            signal = SignalDirection.NEUTRAL
            composite = 0.0
            interpretation = "Insufficient order flow data"

        return IndicatorResult(
            name=self.name,
            value={
                "composite": composite,
                "components": len(signals),
            },
            signal=signal,
            confidence=min(abs(composite), 1.0),
            interpretation=interpretation,
            metadata={
                "cvd_component": signals[0] if len(signals) > 0 else 0,
                "ob_component": signals[1] if len(signals) > 1 else 0,
                "large_trade_component": signals[2] if len(signals) > 2 else 0,
            },
        )

    def calculate_vectorized(self, data: pl.DataFrame) -> pl.DataFrame:
        """Composite requires scalar computation — passthrough."""
        return data


# ═══════════════════════════════════════════════════════════════════════════════
# Indicator Registry
# ═══════════════════════════════════════════════════════════════════════════════

class IndicatorRegistry:
    """
    Central registry for all indicator instances.
    Enables discovery and batch computation.
    """

    _indicators: Dict[str, BaseIndicator] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, indicator: BaseIndicator) -> None:
        """Register an indicator instance."""
        with cls._lock:
            cls._indicators[indicator.name] = indicator
        logger.debug("Registered indicator: %s", indicator.name)

    @classmethod
    def get(cls, name: str) -> Optional[BaseIndicator]:
        """Get a registered indicator by name."""
        return cls._indicators.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered indicator names."""
        return list(cls._indicators.keys())

    @classmethod
    def by_type(cls, indicator_type: IndicatorType) -> List[BaseIndicator]:
        """Get all indicators of a specific type."""
        return [
            ind for ind in cls._indicators.values()
            if ind.indicator_type == indicator_type
        ]

    @classmethod
    def compute_all(cls, data: pl.DataFrame) -> Dict[str, IndicatorResult]:
        """
        Compute all registered indicators against a DataFrame.

        Returns
        -------
        dict
            Mapping of indicator name -> IndicatorResult
        """
        results = {}
        for name, indicator in cls._indicators.items():
            try:
                results[name] = indicator.calculate(data)
            except Exception as exc:
                logger.warning("Indicator %s computation failed: %s", name, exc)
                results[name] = IndicatorResult(
                    name=name,
                    value={"error": str(exc)},
                    signal=SignalDirection.NEUTRAL,
                    confidence=0.0,
                    interpretation=f"Computation error: {exc}",
                )
        return results

    @classmethod
    def clear(cls) -> None:
        """Clear all registered indicators."""
        with cls._lock:
            cls._indicators.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════════

def register_all_orderflow_indicators() -> int:
    """
    Register all order flow indicators with the global registry.

    Returns
    -------
    int
        Number of indicators registered
    """
    indicators = [
        OrderBookImbalanceIndicator(),
        OrderBookImbalanceIndicator([5, 10, 20, 50], 0.15),
        CVDIndicator(),
        CVDIndicator(20),
        TradeFlowImbalanceIndicator(),
        TradeFlowImbalanceIndicator(50),
        VolumeProfileVisibleRangeIndicator(),
        BidAskSpreadIndicator(),
        LargeTradeTrackerIndicator(),
        LargeTradeTrackerIndicator(5.0),
        OrderFlowCompositeIndicator(),
    ]

    for indicator in indicators:
        IndicatorRegistry.register(indicator)

    logger.info("Registered %d order flow indicators", len(indicators))
    return len(indicators)


# Auto-register on import
ORDERFLOW_INDICATOR_COUNT = register_all_orderflow_indicators()

__all__ = [
    "BaseIndicator",
    "IndicatorResult",
    "IndicatorRegistry",
    "SignalDirection",
    "IndicatorType",
    "OrderBookImbalanceIndicator",
    "CVDIndicator",
    "TradeFlowImbalanceIndicator",
    "VolumeProfileVisibleRangeIndicator",
    "BidAskSpreadIndicator",
    "LargeTradeTrackerIndicator",
    "OrderFlowCompositeIndicator",
    "register_all_orderflow_indicators",
]
