"""
NEO SUPREME — Whale Tracker Module
===================================
Institutional-grade large player (whale) activity detection and tracking.

Features:
- Large order detection with configurable USD-value thresholds
- Accumulation / Distribution zone recognition
- Smart Money Flow Index (SMFI) with divergence detection
- Order flow imbalance analysis
- Iceberg order detection (series of similar-sized orders at same price)
- Multi-symbol tracking across the entire trading universe

Module-level convenience functions:
- ensure_tracker(symbol) -> WhaleTracker
- get_tracker(symbol) -> Optional[WhaleTracker]
- get_whale_snapshot(symbol) -> dict
- process_trade(symbol, ...) -> Optional[WhaleTransaction]

Publishes:
  market:whale_alert:{symbol}  — JSON alert on significant whale activity
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("neo_supreme.whale_tracker")


class WhaleActivityType(Enum):
    """Types of whale activities detected by the tracker."""
    LARGE_BUY = "large_buy"
    LARGE_SELL = "large_sell"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    ICEBERG_ORDER = "iceberg_order"
    SWEEP_ORDER = "sweep_order"
    SMART_MONEY_INFLOW = "smart_money_inflow"
    SMART_MONEY_OUTFLOW = "smart_money_outflow"


class OrderClassification(Enum):
    """Classification of order size by USD value."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    WHALE = "whale"
    MEGA_WHALE = "mega_whale"


@dataclass
class WhaleTransaction:
    """Represents a detected large / whale transaction."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    value: float
    side: str
    activity_type: WhaleActivityType
    order_classification: OrderClassification
    impact_score: float
    related_orders: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AccumulationZone:
    """Represents a price zone where whales are accumulating or distributing."""
    start_time: datetime
    end_time: Optional[datetime]
    price_low: float
    price_high: float
    total_volume: float
    net_position: float
    confidence: float
    transactions: List[WhaleTransaction] = field(default_factory=list)


@dataclass
class SmartMoneyFlow:
    """Smart Money Flow Index data point with divergence detection."""
    timestamp: datetime
    smfi: float
    raw_money_flow: float
    typical_price: float
    volume: float
    trend_direction: int
    bullish_divergence: bool = False
    bearish_divergence: bool = False
    divergence_signal: Optional[str] = None


@dataclass
class OrderFlowImbalance:
    """Order flow imbalance metrics from order book analysis."""
    timestamp: datetime
    bid_imbalance: float
    ask_imbalance: float
    net_imbalance: float
    large_bid_volume: float
    large_ask_volume: float
    total_large_volume: float
    pressure_direction: str


@dataclass
class WhaleAlert:
    """Alert for significant whale activity."""
    timestamp: datetime
    symbol: str
    alert_type: WhaleActivityType
    severity: str
    message: str
    transactions: List[WhaleTransaction]
    recommended_action: str
    confidence: float


class WhaleTracker:
    """
    Track and analyze large player (whale) activity.

    Detects large orders, accumulation/distribution patterns, smart money flow,
    order flow imbalances, and iceberg orders.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., "BTCUSDT")
    large_order_threshold : float
        Minimum USD value for a large order (default $100,000)
    whale_order_threshold : float
        Minimum USD value for a whale order (default $500,000)
    mega_whale_threshold : float
        Minimum USD value for a mega whale order (default $2,000,000)
    accumulation_lookback : int
        Periods for accumulation analysis
    smfi_period : int
        Period for Smart Money Flow Index calculation
    imbalance_threshold : float
        Threshold for significant imbalance detection
    lookback_periods : int
        Number of periods to keep in history
    """

    def __init__(
        self,
        symbol: str,
        large_order_threshold: float = 100000.0,
        whale_order_threshold: float = 500000.0,
        mega_whale_threshold: float = 2000000.0,
        accumulation_lookback: int = 100,
        smfi_period: int = 14,
        imbalance_threshold: float = 0.6,
        lookback_periods: int = 1000,
    ):
        self.symbol = symbol
        self.large_order_threshold = large_order_threshold
        self.whale_order_threshold = whale_order_threshold
        self.mega_whale_threshold = mega_whale_threshold
        self.accumulation_lookback = accumulation_lookback
        self.smfi_period = smfi_period
        self.imbalance_threshold = imbalance_threshold

        # Data storage
        self.transactions: Deque[dict] = deque(maxlen=lookback_periods)
        self.whale_transactions: Deque[WhaleTransaction] = deque(maxlen=lookback_periods)
        self.price_history: Deque[float] = deque(maxlen=lookback_periods)
        self.volume_history: Deque[float] = deque(maxlen=lookback_periods)
        self.timestamp_history: Deque[datetime] = deque(maxlen=lookback_periods)

        # Analysis data
        self.accumulation_zones: List[AccumulationZone] = []
        self.smfi_history: Deque[SmartMoneyFlow] = deque(maxlen=lookback_periods)
        self.imbalance_history: Deque[OrderFlowImbalance] = deque(maxlen=lookback_periods)
        self.alerts: Deque[WhaleAlert] = deque(maxlen=100)

        # Current state
        self.current_zone: Optional[AccumulationZone] = None
        self.cumulative_whale_volume: float = 0.0
        self.cumulative_whale_value: float = 0.0

        # Statistics
        self.stats = {
            "total_large_orders": 0,
            "total_whale_orders": 0,
            "total_mega_whale_orders": 0,
            "buy_pressure": 0.0,
            "sell_pressure": 0.0,
            "net_position": 0.0,
        }

        # Thread safety
        self._lock = threading.RLock()

        # Callbacks
        self._alert_callbacks: List[Callable[[WhaleAlert], None]] = []

        logger.info("WhaleTracker initialized for %s", symbol)

    def process_trade(
        self,
        timestamp: datetime,
        price: float,
        volume: float,
        side: str,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        order_id: Optional[str] = None,
    ) -> Optional[WhaleTransaction]:
        """
        Process a trade and detect whale activity.

        Parameters
        ----------
        timestamp : datetime
            Trade timestamp
        price : float
            Trade price
        volume : float
            Trade volume (in base asset)
        side : str
            Trade side ('buy' or 'sell')
        bid, ask : float, optional
            Current best bid/ask prices
        order_id : str, optional
            Optional order identifier

        Returns
        -------
        WhaleTransaction or None
            WhaleTransaction if whale activity detected, None otherwise
        """
        with self._lock:
            value = price * volume

            # Store basic data
            self.price_history.append(price)
            self.volume_history.append(volume)
            self.timestamp_history.append(timestamp)

            # Classify order
            order_class = self._classify_order(value)

            # Check if whale activity
            if order_class in (OrderClassification.LARGE, OrderClassification.WHALE, OrderClassification.MEGA_WHALE):
                activity_type = self._classify_activity(price, volume, side, bid, ask)
                impact_score = self._calculate_impact_score(value, order_class, volume)

                transaction = WhaleTransaction(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    price=price,
                    volume=volume,
                    value=value,
                    side=side.lower(),
                    activity_type=activity_type,
                    order_classification=order_class,
                    impact_score=impact_score,
                    related_orders=[order_id] if order_id else [],
                    metadata={
                        "bid": bid,
                        "ask": ask,
                        "spread": ask - bid if bid and ask else 0,
                    },
                )

                self.whale_transactions.append(transaction)
                self._update_statistics(transaction)
                self._detect_iceberg_order(transaction)
                self._update_accumulation_analysis(transaction)
                self._update_smfi(timestamp, price, volume, side)
                self._check_alerts(transaction)

                return transaction

            return None

    def process_order_book_update(
        self,
        timestamp: datetime,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        last_price: float,
    ) -> Optional[OrderFlowImbalance]:
        """
        Process order book update for imbalance analysis.

        Parameters
        ----------
        timestamp : datetime
            Update timestamp
        bids : list of (price, size)
            Bid orders
        asks : list of (price, size)
            Ask orders
        last_price : float
            Last traded price

        Returns
        -------
        OrderFlowImbalance or None
        """
        with self._lock:
            large_bids = [
                (p, s) for p, s in bids
                if p * s >= self.large_order_threshold
            ]
            large_asks = [
                (p, s) for p, s in asks
                if p * s >= self.large_order_threshold
            ]

            large_bid_volume = sum(s for _, s in large_bids)
            large_ask_volume = sum(s for _, s in large_asks)
            total_large_volume = large_bid_volume + large_ask_volume

            if total_large_volume == 0:
                return None

            bid_imbalance = large_bid_volume / total_large_volume
            ask_imbalance = large_ask_volume / total_large_volume
            net_imbalance = bid_imbalance - ask_imbalance

            if abs(net_imbalance) < 0.1:
                pressure_direction = "neutral"
            elif net_imbalance > 0:
                pressure_direction = "buying"
            else:
                pressure_direction = "selling"

            imbalance = OrderFlowImbalance(
                timestamp=timestamp,
                bid_imbalance=bid_imbalance,
                ask_imbalance=ask_imbalance,
                net_imbalance=net_imbalance,
                large_bid_volume=large_bid_volume,
                large_ask_volume=large_ask_volume,
                total_large_volume=total_large_volume,
                pressure_direction=pressure_direction,
            )

            self.imbalance_history.append(imbalance)

            if abs(net_imbalance) >= self.imbalance_threshold:
                self._create_imbalance_alert(imbalance)

            return imbalance

    def _classify_order(self, value: float) -> OrderClassification:
        """Classify order by USD value."""
        if value >= self.mega_whale_threshold:
            return OrderClassification.MEGA_WHALE
        elif value >= self.whale_order_threshold:
            return OrderClassification.WHALE
        elif value >= self.large_order_threshold:
            return OrderClassification.LARGE
        elif value >= self.large_order_threshold / 10:
            return OrderClassification.MEDIUM
        else:
            return OrderClassification.SMALL

    def _classify_activity(
        self,
        price: float,
        volume: float,
        side: str,
        bid: Optional[float],
        ask: Optional[float],
    ) -> WhaleActivityType:
        """Classify whale activity type based on context."""
        side_lower = side.lower()

        # Check for sweep orders (hitting multiple levels)
        if bid and ask:
            spread = ask - bid
            spread_pct = spread / price if price > 0 else 0
            if side_lower in ("buy", "bid") and price >= ask:
                if spread_pct > 0.001:
                    return WhaleActivityType.SWEEP_ORDER

        # Classify as accumulation/distribution based on context
        if len(self.price_history) >= 20:
            recent_prices = list(self.price_history)[-20:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

            if side_lower in ("buy", "bid"):
                if price_trend < -0.02:
                    return WhaleActivityType.ACCUMULATION
                else:
                    return WhaleActivityType.LARGE_BUY
            else:
                if price_trend > 0.02:
                    return WhaleActivityType.DISTRIBUTION
                else:
                    return WhaleActivityType.LARGE_SELL

        return (
            WhaleActivityType.LARGE_BUY
            if side_lower in ("buy", "bid")
            else WhaleActivityType.LARGE_SELL
        )

    def _calculate_impact_score(
        self, value: float, order_class: OrderClassification, volume: float
    ) -> float:
        """Calculate market impact score [0, 1]."""
        class_scores = {
            OrderClassification.LARGE: 0.3,
            OrderClassification.WHALE: 0.6,
            OrderClassification.MEGA_WHALE: 0.9,
        }
        base_score = class_scores.get(order_class, 0.1)

        if len(self.volume_history) > 0:
            avg_volume = np.mean(list(self.volume_history)[-50:])
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            volume_factor = min(0.2, volume_ratio / 10)
        else:
            volume_factor = 0.0

        return min(1.0, base_score + volume_factor)

    def _detect_iceberg_order(self, transaction: WhaleTransaction):
        """
        Detect potential iceberg orders.

        Looks for a series of similar-sized orders at the same price level
        within a 60-second window. If 3+ orders match with coefficient of
        variation < 0.2, flags them as iceberg.
        """
        recent = [
            t for t in self.whale_transactions
            if (transaction.timestamp - t.timestamp).total_seconds() < 60
            and abs(t.price - transaction.price) / transaction.price < 0.001
        ]

        if len(recent) >= 3:
            sizes = [t.volume for t in recent]
            size_std = np.std(sizes)
            size_mean = np.mean(sizes)

            if size_mean > 0 and size_std / size_mean < 0.2:
                for t in recent:
                    t.activity_type = WhaleActivityType.ICEBERG_ORDER
                    t.metadata["iceberg_detected"] = True

    def _update_statistics(self, transaction: WhaleTransaction):
        """Update whale statistics counters."""
        self.stats["total_large_orders"] += 1

        if transaction.order_classification == OrderClassification.WHALE:
            self.stats["total_whale_orders"] += 1
        elif transaction.order_classification == OrderClassification.MEGA_WHALE:
            self.stats["total_mega_whale_orders"] += 1

        self.cumulative_whale_volume += transaction.volume
        self.cumulative_whale_value += transaction.value

        if transaction.side == "buy":
            self.stats["buy_pressure"] += transaction.value
            self.stats["net_position"] += transaction.volume
        else:
            self.stats["sell_pressure"] += transaction.value
            self.stats["net_position"] -= transaction.volume

    def _update_accumulation_analysis(self, transaction: WhaleTransaction):
        """Update accumulation / distribution zone analysis."""
        if self.current_zone:
            self.current_zone.end_time = transaction.timestamp
            self.current_zone.price_high = max(
                self.current_zone.price_high, transaction.price
            )
            self.current_zone.price_low = min(
                self.current_zone.price_low, transaction.price
            )
            self.current_zone.total_volume += transaction.volume
            self.current_zone.transactions.append(transaction)

            if transaction.side == "buy":
                self.current_zone.net_position += transaction.volume
            else:
                self.current_zone.net_position -= transaction.volume

            # Check if zone should end (max 1 hour)
            zone_duration = (
                transaction.timestamp - self.current_zone.start_time
            ).total_seconds()
            if zone_duration > 3600:
                self._finalize_current_zone()
        else:
            self.current_zone = AccumulationZone(
                start_time=transaction.timestamp,
                end_time=transaction.timestamp,
                price_low=transaction.price,
                price_high=transaction.price,
                total_volume=transaction.volume,
                net_position=(
                    transaction.volume
                    if transaction.side == "buy"
                    else -transaction.volume
                ),
                confidence=0.5,
                transactions=[transaction],
            )

    def _finalize_current_zone(self):
        """Finalize the current accumulation / distribution zone."""
        if self.current_zone:
            if len(self.current_zone.transactions) >= 5:
                buy_ratio = (
                    sum(1 for t in self.current_zone.transactions if t.side == "buy")
                    / len(self.current_zone.transactions)
                )
                self.current_zone.confidence = max(buy_ratio, 1 - buy_ratio)
                self.accumulation_zones.append(self.current_zone)

                if (
                    self.current_zone.confidence > 0.7
                    and self.current_zone.total_volume > self.whale_order_threshold * 10
                ):
                    zone_type = (
                        "ACCUMULATION"
                        if self.current_zone.net_position > 0
                        else "DISTRIBUTION"
                    )
                    self._create_zone_alert(self.current_zone, zone_type)

            self.current_zone = None

    def _update_smfi(
        self, timestamp: datetime, price: float, volume: float, side: str
    ):
        """
        Update Smart Money Flow Index (SMFI).

        SMFI measures "smart money" activity: buying on weakness,
        selling on strength. Includes divergence detection.
        """
        if len(self.price_history) < 2:
            return

        prev_price = list(self.price_history)[-2]
        typical_price = (price + prev_price) / 2.0
        raw_money_flow = typical_price * volume
        price_change = price - prev_price

        # Smart money buys on weakness, sells on strength
        side_lower = side.lower()
        if price_change < 0 and side_lower in ("buy", "bid"):
            trend_direction = 1
        elif price_change > 0 and side_lower in ("sell", "ask"):
            trend_direction = -1
        else:
            trend_direction = 0

        # Calculate SMFI
        if len(self.smfi_history) == 0:
            smfi = 50.0
        else:
            prev_smfi = self.smfi_history[-1].smfi
            if trend_direction != 0:
                recent_vols = list(self.volume_history)[-self.smfi_period:]
                avg_vol = np.mean(recent_vols) if recent_vols else volume
                flow_ratio = raw_money_flow / (avg_vol * typical_price) if avg_vol * typical_price > 0 else 0
                smfi_change = trend_direction * min(5.0, flow_ratio * 2)
                smfi = prev_smfi + smfi_change
            else:
                smfi = prev_smfi * 0.95 + 50.0 * 0.05  # Decay towards neutral

        smfi = max(0.0, min(100.0, smfi))

        # Detect divergence
        bullish_divergence = False
        bearish_divergence = False
        divergence = None
        if len(self.smfi_history) >= 10:
            recent_smfi = [s.smfi for s in list(self.smfi_history)[-10:]]
            recent_prices = list(self.price_history)[-10:]

            smfi_trend = recent_smfi[-1] - recent_smfi[0]
            price_trend = recent_prices[-1] - recent_prices[0]

            if smfi_trend > 5 and price_trend < 0:
                bullish_divergence = True
                divergence = "bullish"
            elif smfi_trend < -5 and price_trend > 0:
                bearish_divergence = True
                divergence = "bearish"

        smfi_data = SmartMoneyFlow(
            timestamp=timestamp,
            smfi=smfi,
            raw_money_flow=raw_money_flow,
            typical_price=typical_price,
            volume=volume,
            trend_direction=trend_direction,
            bullish_divergence=bullish_divergence,
            bearish_divergence=bearish_divergence,
            divergence_signal=divergence,
        )

        self.smfi_history.append(smfi_data)

        # Alert on extreme SMFI values
        if smfi > 80 or smfi < 20:
            self._create_smfi_alert(smfi_data)

    def _check_alerts(self, transaction: WhaleTransaction):
        """Check if a transaction should trigger an alert."""
        if transaction.order_classification == OrderClassification.MEGA_WHALE:
            alert = WhaleAlert(
                timestamp=transaction.timestamp,
                symbol=self.symbol,
                alert_type=transaction.activity_type,
                severity="critical",
                message=f"MEGA WHALE {transaction.side.upper()} detected: ${transaction.value:,.0f}",
                transactions=[transaction],
                recommended_action="MONITOR_CLOSELY;FOLLOW_WHALE",
                confidence=0.95,
            )
            self._notify_alert(alert)
        elif transaction.order_classification == OrderClassification.WHALE:
            if transaction.activity_type == WhaleActivityType.ACCUMULATION:
                alert = WhaleAlert(
                    timestamp=transaction.timestamp,
                    symbol=self.symbol,
                    alert_type=WhaleActivityType.ACCUMULATION,
                    severity="high",
                    message=f"Whale accumulation detected: ${transaction.value:,.0f}",
                    transactions=[transaction],
                    recommended_action="CONSIDER_ACCUMULATING",
                    confidence=0.8,
                )
                self._notify_alert(alert)
            elif transaction.activity_type == WhaleActivityType.DISTRIBUTION:
                alert = WhaleAlert(
                    timestamp=transaction.timestamp,
                    symbol=self.symbol,
                    alert_type=WhaleActivityType.DISTRIBUTION,
                    severity="high",
                    message=f"Whale distribution detected: ${transaction.value:,.0f}",
                    transactions=[transaction],
                    recommended_action="CONSIDER_REDUCING_EXPOSURE",
                    confidence=0.8,
                )
                self._notify_alert(alert)

    def _create_imbalance_alert(self, imbalance: OrderFlowImbalance):
        """Create alert for significant order flow imbalance."""
        direction = "BUYING" if imbalance.net_imbalance > 0 else "SELLING"
        alert = WhaleAlert(
            timestamp=imbalance.timestamp,
            symbol=self.symbol,
            alert_type=(
                WhaleActivityType.SMART_MONEY_INFLOW
                if imbalance.net_imbalance > 0
                else WhaleActivityType.SMART_MONEY_OUTFLOW
            ),
            severity="medium" if abs(imbalance.net_imbalance) < 0.8 else "high",
            message=f"Significant {direction} pressure: {abs(imbalance.net_imbalance)*100:.1f}% imbalance",
            transactions=[],
            recommended_action=(
                "FOLLOW_IMBALANCE"
                if abs(imbalance.net_imbalance) > 0.7
                else "MONITOR"
            ),
            confidence=abs(imbalance.net_imbalance),
        )
        self._notify_alert(alert)

    def _create_zone_alert(self, zone: AccumulationZone, zone_type: str):
        """Create alert for accumulation / distribution zone completion."""
        alert = WhaleAlert(
            timestamp=zone.end_time or zone.start_time,
            symbol=self.symbol,
            alert_type=(
                WhaleActivityType.ACCUMULATION
                if zone_type == "ACCUMULATION"
                else WhaleActivityType.DISTRIBUTION
            ),
            severity="high",
            message=f"{zone_type} zone completed: {len(zone.transactions)} transactions, ${zone.total_volume * zone.price_low:,.0f} volume",
            transactions=zone.transactions,
            recommended_action=(
                "ACCUMULATE_WITH_WHALES"
                if zone_type == "ACCUMULATION"
                else "REDUCE_EXPOSURE"
            ),
            confidence=zone.confidence,
        )
        self._notify_alert(alert)

    def _create_smfi_alert(self, smfi: SmartMoneyFlow):
        """Create alert for extreme SMFI value."""
        if smfi.smfi > 80:
            message = f"SMFI extremely high: {smfi.smfi:.1f} (potential distribution)"
            alert_type = WhaleActivityType.SMART_MONEY_OUTFLOW
        else:
            message = f"SMFI extremely low: {smfi.smfi:.1f} (potential accumulation)"
            alert_type = WhaleActivityType.SMART_MONEY_INFLOW

        alert = WhaleAlert(
            timestamp=smfi.timestamp,
            symbol=self.symbol,
            alert_type=alert_type,
            severity="medium",
            message=message,
            transactions=[],
            recommended_action="WATCH_FOR_REVERSAL",
            confidence=abs(smfi.smfi - 50) / 50,
        )
        self._notify_alert(alert)

    def register_alert_callback(self, callback: Callable[[WhaleAlert], None]):
        """Register a callback for whale alerts."""
        self._alert_callbacks.append(callback)

    def _notify_alert(self, alert: WhaleAlert):
        """Notify all registered alert callbacks."""
        self.alerts.append(alert)
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Alert callback error: %s", e)

    def get_current_smfi(self) -> Optional[SmartMoneyFlow]:
        """Get current Smart Money Flow Index data."""
        with self._lock:
            return self.smfi_history[-1] if self.smfi_history else None

    def get_current_imbalance(self) -> Optional[OrderFlowImbalance]:
        """Get current order flow imbalance."""
        with self._lock:
            return self.imbalance_history[-1] if self.imbalance_history else None

    def get_accumulation_zones(self, active_only: bool = False) -> List[AccumulationZone]:
        """Get accumulation / distribution zones."""
        with self._lock:
            zones = list(self.accumulation_zones)
            if active_only and self.current_zone:
                zones.append(self.current_zone)
            return zones

    def get_whale_transactions(
        self,
        n: int = 50,
        min_classification: OrderClassification = OrderClassification.LARGE,
    ) -> List[WhaleTransaction]:
        """Get recent whale transactions filtered by minimum classification."""
        with self._lock:
            class_order = [
                OrderClassification.SMALL,
                OrderClassification.MEDIUM,
                OrderClassification.LARGE,
                OrderClassification.WHALE,
                OrderClassification.MEGA_WHALE,
            ]
            min_idx = class_order.index(min_classification)
            return [
                t for t in list(self.whale_transactions)[-n:]
                if class_order.index(t.order_classification) >= min_idx
            ]

    def get_net_whale_position(self, lookback_periods: Optional[int] = None) -> float:
        """Get net whale position over a lookback period."""
        with self._lock:
            transactions = list(self.whale_transactions)
            if lookback_periods:
                transactions = transactions[-lookback_periods:]
            net_position = 0.0
            for t in transactions:
                if t.side == "buy":
                    net_position += t.volume
                else:
                    net_position -= t.volume
            return net_position

    def get_summary(self) -> Dict:
        """Get whale tracking summary as a dictionary."""
        with self._lock:
            smfi = self.smfi_history[-1] if self.smfi_history else None
            imbalance = self.imbalance_history[-1] if self.imbalance_history else None
            return {
                "symbol": self.symbol,
                "total_large_orders": self.stats["total_large_orders"],
                "total_whale_orders": self.stats["total_whale_orders"],
                "total_mega_whale_orders": self.stats["total_mega_whale_orders"],
                "cumulative_whale_volume": self.cumulative_whale_volume,
                "cumulative_whale_value": self.cumulative_whale_value,
                "buy_pressure": self.stats["buy_pressure"],
                "sell_pressure": self.stats["sell_pressure"],
                "net_position": self.stats["net_position"],
                "current_smfi": smfi.smfi if smfi else None,
                "smfi_bullish_divergence": smfi.bullish_divergence if smfi else False,
                "smfi_bearish_divergence": smfi.bearish_divergence if smfi else False,
                "current_imbalance": imbalance.net_imbalance if imbalance else None,
                "accumulation_zones": len(self.accumulation_zones),
                "active_zone": self.current_zone is not None,
            }


class MultiSymbolWhaleTracker:
    """
    Track whale activity across multiple symbols.

    Parameters
    ----------
    symbols : list of str
        Trading symbols to track
    large_order_threshold : float
        Minimum USD value for large order detection
    """

    def __init__(
        self,
        symbols: List[str],
        large_order_threshold: float = 100000.0,
        **kwargs,
    ):
        self.symbols = symbols
        self.trackers: Dict[str, WhaleTracker] = {
            symbol: WhaleTracker(symbol, large_order_threshold, **kwargs)
            for symbol in symbols
        }
        logger.info("MultiSymbolWhaleTracker initialized for %d symbols", len(symbols))

    def process_trade(self, symbol: str, **kwargs) -> Optional[WhaleTransaction]:
        """Process a trade for a specific symbol."""
        if symbol not in self.trackers:
            return None
        return self.trackers[symbol].process_trade(**kwargs)

    def process_order_book_update(
        self, symbol: str, **kwargs
    ) -> Optional[OrderFlowImbalance]:
        """Process an order book update for a specific symbol."""
        if symbol not in self.trackers:
            return None
        return self.trackers[symbol].process_order_book_update(**kwargs)

    def get_tracker(self, symbol: str) -> Optional[WhaleTracker]:
        """Get the WhaleTracker for a symbol."""
        return self.trackers.get(symbol)

    def get_all_smfi(self) -> Dict[str, Optional[float]]:
        """Get SMFI for all symbols."""
        return {
            symbol: tracker.get_current_smfi().smfi if tracker.get_current_smfi() else None
            for symbol, tracker in self.trackers.items()
        }

    def get_all_imbalances(self) -> Dict[str, Optional[float]]:
        """Get order flow imbalances for all symbols."""
        return {
            symbol: tracker.get_current_imbalance().net_imbalance
            if tracker.get_current_imbalance()
            else None
            for symbol, tracker in self.trackers.items()
        }

    def get_whale_activity_ranking(self) -> List[Tuple[str, float]]:
        """Rank symbols by whale activity level."""
        activity_scores = []
        for symbol, tracker in self.trackers.items():
            summary = tracker.get_summary()
            score = (
                summary["total_whale_orders"] * 10
                + summary["total_mega_whale_orders"] * 50
                + summary["cumulative_whale_value"] / 1000000
                + abs(summary["net_position"]) / 1000
            )
            activity_scores.append((symbol, score))
        return sorted(activity_scores, key=lambda x: x[1], reverse=True)

    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summary for all tracked symbols."""
        return {
            symbol: tracker.get_summary()
            for symbol, tracker in self.trackers.items()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level shared registry for live integration with execution ensemble
# ═══════════════════════════════════════════════════════════════════════════════

_MODULE_TRACKERS: Dict[str, WhaleTracker] = {}
_module_registry_lock = threading.Lock()


def ensure_tracker(symbol: str, **kwargs) -> WhaleTracker:
    """
    Return (or create) a module-level WhaleTracker for *symbol*.

    Thread-safe. Additional kwargs are passed to WhaleTracker constructor
    on first creation.
    """
    if symbol not in _MODULE_TRACKERS:
        with _module_registry_lock:
            if symbol not in _MODULE_TRACKERS:
                _MODULE_TRACKERS[symbol] = WhaleTracker(symbol=symbol, **kwargs)
    return _MODULE_TRACKERS[symbol]


def get_tracker(symbol: str) -> Optional[WhaleTracker]:
    """Return an existing WhaleTracker for *symbol* or None."""
    return _MODULE_TRACKERS.get(symbol)


def get_whale_snapshot(symbol: str) -> Dict:
    """
    Lightweight snapshot of whale state for consensus modules.

    Returns dict with:
        - cvd: cumulative whale volume
        - order_flow_imbalance: current imbalance [-1, 1]
        - smfi: current SMFI value [0, 100]
        - smfi_bullish_divergence: bool
        - smfi_bearish_divergence: bool
        - accumulation_zone_active: bool
        - net_whale_position: net position in base units

    Safe to call even if no tracker exists yet.
    """
    tracker = _MODULE_TRACKERS.get(symbol)
    if tracker is None:
        return {
            "cvd": 0.0,
            "order_flow_imbalance": 0.0,
            "smfi": None,
            "smfi_bullish_divergence": False,
            "smfi_bearish_divergence": False,
            "accumulation_zone_active": False,
            "net_whale_position": 0.0,
        }
    smfi = tracker.get_current_smfi()
    imbalance = tracker.get_current_imbalance()
    summary = tracker.get_summary()
    return {
        "cvd": tracker.cumulative_whale_volume,
        "order_flow_imbalance": imbalance.net_imbalance if imbalance else 0.0,
        "smfi": smfi.smfi if smfi else None,
        "smfi_bullish_divergence": smfi.bullish_divergence if smfi else False,
        "smfi_bearish_divergence": smfi.bearish_divergence if smfi else False,
        "accumulation_zone_active": summary.get("active_zone", False),
        "net_whale_position": summary.get("net_position", 0.0),
    }


def process_trade(symbol: str, **kwargs) -> Optional[WhaleTransaction]:
    """Process a trade through the module-level tracker for *symbol*."""
    tracker = ensure_tracker(symbol)
    return tracker.process_trade(**kwargs)


def process_order_book_update(
    symbol: str, **kwargs
) -> Optional[OrderFlowImbalance]:
    """Process an order-book update through the module-level tracker."""
    tracker = ensure_tracker(symbol)
    return tracker.process_order_book_update(**kwargs)


def get_all_tracked_symbols() -> List[str]:
    """Return list of all symbols with active whale trackers."""
    return list(_MODULE_TRACKERS.keys())


__all__ = [
    "WhaleTracker",
    "MultiSymbolWhaleTracker",
    "WhaleTransaction",
    "WhaleAlert",
    "WhaleActivityType",
    "OrderClassification",
    "AccumulationZone",
    "SmartMoneyFlow",
    "OrderFlowImbalance",
    "ensure_tracker",
    "get_tracker",
    "get_whale_snapshot",
    "process_trade",
    "process_order_book_update",
    "get_all_tracked_symbols",
]
