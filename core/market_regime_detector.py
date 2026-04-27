"""
NEO SUPREME 2026 — Market Regime Detector
Detects market regime in real-time: trending, ranging, volatile, accumulation,
distribution. Used for regime-conditioned agent behaviour.
"""
import numpy as np
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    PUMP = "pump"
    CRASH = "crash"
    UNKNOWN = "unknown"


@dataclass
class RegimeDetector:
    """
    Real-time market regime detection using price action, volume profile,
    and order flow analysis. Stateless — operates on a window of klines.
    """
    lookback: int = 50
    trend_threshold: float = 0.3     # ADX threshold for trend
    range_threshold: float = 0.15    # Max ADX for ranging
    vol_percentile: float = 75.0     # Volatility percentile threshold

    def detect(self, klines: List[list]) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Detect market regime from recent klines.
        klines: list of [ts, open, high, low, close, volume]
        Returns: (regime, confidence_scores)
        """
        if len(klines) < self.lookback:
            return MarketRegime.UNKNOWN, {}

        closes = np.array([k[4] for k in klines[-self.lookback:]], dtype=np.float64)
        highs = np.array([k[2] for k in klines[-self.lookback:]], dtype=np.float64)
        lows = np.array([k[3] for k in klines[-self.lookback:]], dtype=np.float64)
        volumes = np.array([k[5] for k in klines[-self.lookback:]], dtype=np.float64)

        # Trend detection: ADX proxy using directional movement
        adx = self._calculate_adx(highs, lows, closes)

        # Volatility
        returns = np.diff(np.log(closes))
        current_vol = np.std(returns[-20:]) * np.sqrt(365 * 24 * 12)
        hist_vol = np.std(returns) * np.sqrt(365 * 24 * 12)
        vol_ratio = current_vol / max(hist_vol, 1e-8)

        # Price momentum
        sma20 = np.mean(closes[-20:])
        sma50 = np.mean(closes[-self.lookback:]) if len(closes) >= self.lookback else sma20
        price_vs_sma20 = (closes[-1] - sma20) / sma20 if sma20 > 0 else 0
        price_vs_sma50 = (closes[-1] - sma50) / sma50 if sma50 > 0 else 0

        # Volume profile
        vol_sma = np.mean(volumes[-20:])
        vol_surge = volumes[-1] / max(vol_sma, 1e-8)

        scores = {
            "adx": float(adx),
            "volatility_annual": float(current_vol),
            "vol_ratio": float(vol_ratio),
            "price_vs_sma20": float(price_vs_sma20),
            "price_vs_sma50": float(price_vs_sma50),
            "volume_surge": float(vol_surge),
        }

        # Regime classification
        if adx > self.trend_threshold:
            if price_vs_sma20 > 0.01 and price_vs_sma50 > 0:
                regime = MarketRegime.TRENDING_UP
            elif price_vs_sma20 < -0.01 and price_vs_sma50 < 0:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING
        elif adx < self.range_threshold:
            if vol_ratio > 1.5:
                regime = MarketRegime.HIGH_VOLATILITY
            elif vol_ratio < 0.5:
                regime = MarketRegime.LOW_VOLATILITY
            else:
                regime = MarketRegime.RANGING
        else:
            regime = MarketRegime.RANGING

        # Override: pump/crash detection
        if vol_surge > 3.0 and price_vs_sma20 > 0.03:
            regime = MarketRegime.PUMP
        elif vol_surge > 3.0 and price_vs_sma20 < -0.03:
            regime = MarketRegime.CRASH

        # Override: accumulation/distribution
        if vol_surge > 2.0 and abs(price_vs_sma20) < 0.005:
            if closes[-1] > np.mean(closes[-10:]):
                regime = MarketRegime.ACCUMULATION
            else:
                regime = MarketRegime.DISTRIBUTION

        return regime, scores

    @staticmethod
    def _calculate_adx(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)."""
        if len(highs) < period + 1:
            return 0.0
        up_moves = highs[1:] - highs[:-1]
        down_moves = lows[:-1] - lows[1:]
        plus_dm = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0)
        minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0)
        tr = np.maximum(highs[1:] - lows[1:],
                       np.maximum(np.abs(highs[1:] - closes[:-1]),
                                 np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-period:])
        if atr == 0:
            return 0.0
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr
        dx = 100 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-8)
        return dx / 100.0  # Normalise to 0-1
