"""
Feature Engine V2 — Numba @njit 160-Dimensional State Tensor
=============================================================
Exodia Stack Foundation.

All heavy math compiled to machine-code via Numba @njit (bypasses Python GIL).
Non-JIT wrapper handles dict unpacking, Redis publish, and running-stats.

Layout
------
  [  0 -  31]  Price features      — returns, momentum, SMA/EMA distances, volume, trend
  [ 32 -  95]  Technical features  — RSI, MACD, Bollinger, ATR, Stochastic, VPIN, Regime,
                                     Williams %R, MFI, CCI, ADX
  [ 96 - 143]  Orderbook features  — spread, OFI, imbalance, cumulative depth, price levels
  [144 - 159]  Position features   — size, P&L, distance, leverage, margin, side

Publishes completed tensor to Redis: ``market:state_tensor:{symbol}``
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
from numba import njit, prange

logger = logging.getLogger("feature_engine_v2")


# ═══════════════════════════════════════════════════════════════════════════════
# JIT-compiled helper functions (module-level pure functions — no Python objects)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True, fastmath=True)
def _calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Log returns — each element independent -> prange safe."""
    n = len(prices)
    if n < 2:
        return np.zeros(1, dtype=np.float32)
    returns = np.empty(n - 1, dtype=np.float32)
    for i in prange(n - 1):
        if prices[i] > 0.0:
            returns[i] = np.log(prices[i + 1] / prices[i])
        else:
            returns[i] = 0.0
    return returns


@njit(cache=True, fastmath=True)
def _calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average — sequential dependency on cumsum."""
    n = len(data)
    result = np.empty(n, dtype=np.float32)
    cumsum = 0.0
    for i in range(n):
        cumsum += data[i]
        if i >= period:
            cumsum -= data[i - period]
        if i >= period - 1:
            result[i] = cumsum / period
        else:
            result[i] = cumsum / (i + 1)
    return result


@njit(cache=True, fastmath=True)
def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average — sequential (each bar depends on previous)."""
    n = len(data)
    result = np.empty(n, dtype=np.float32)
    alpha = np.float32(2.0 / (period + 1))
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1]
    return result


@njit(cache=True, fastmath=True)
def _calculate_rsi(returns: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder-smoothed RSI from log-return series."""
    n = len(returns)
    result = np.empty(n, dtype=np.float32)
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(n):
        gain = max(returns[i], 0.0)
        loss = abs(min(returns[i], 0.0))
        if i < period:
            avg_gain += gain / period
            avg_loss += loss / period
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss < 1e-12:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    return result


@njit(cache=True, fastmath=True)
def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   period: int = 14) -> np.ndarray:
    """Average True Range (EMA-smoothed)."""
    n = len(close)
    tr = np.empty(n, dtype=np.float32)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, max(tr2, tr3))
    return _calculate_ema(tr, period)


@njit(cache=True, fastmath=True)
def _calculate_bollinger(prices: np.ndarray, period: int = 20,
                         num_std: float = 2.0):
    """Bollinger Bands -> (upper, middle, lower)."""
    sma = _calculate_sma(prices, period)
    n = len(prices)
    std = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - period + 1)
        window = prices[start:i + 1]
        std[i] = np.std(window)
    upper = sma + std * num_std
    lower = sma - std * num_std
    return upper, sma, lower


@njit(cache=True, fastmath=True)
def _robust_normalize(value, median, mad):
    """Robust normalisation: (x − median) / (MAD + ε)."""
    return (value - median) / (mad + 1e-8)


@njit(cache=True, fastmath=True)
def _calculate_vpin(buy_vols: np.ndarray, sell_vols: np.ndarray,
                    window: int = 50) -> np.ndarray:
    """
    Volume-Synchronised Probability of Informed Trading.

    Rolling |buy − sell| / (buy + sell) over *window* buckets.
    Values > 0.75 indicate toxic flow (informed traders dominating).
    """
    n = len(buy_vols)
    result = np.empty(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        abs_imb = 0.0
        total_vol = 0.0
        for j in range(start, i + 1):
            abs_imb += abs(buy_vols[j] - sell_vols[j])
            total_vol += buy_vols[j] + sell_vols[j]
        if total_vol > 0.0:
            result[i] = abs_imb / total_vol
        else:
            result[i] = 0.0
    return result


@njit(cache=True, fastmath=True)
def _calculate_ofi(bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:
    """
    Order Flow Imbalance at top-of-book.

    Returns [-1, +1]:  positive = bid-heavy (whale buying).
    OFI > 0.6  ->  whale sweep signal (high conviction).
    """
    if len(bid_volumes) == 0 or len(ask_volumes) == 0:
        return 0.0
    total = bid_volumes[0] + ask_volumes[0]
    if total > 0.0:
        return (bid_volumes[0] - ask_volumes[0]) / total
    return 0.0


@njit(cache=True, fastmath=True)
def _calculate_regime_features(returns: np.ndarray):
    """
    Regime triad: (annualised_vol, autocorrelation_lag1, skewness).

    volatility  : σ × √252  — current regime energy
    autocorr    : lag-1 serial correlation  — trend persistence
    skew        : return skewness  — tail directionality
    """
    n = len(returns)

    # Annualised volatility
    volatility = np.float32(np.std(returns) * np.sqrt(252.0))

    # Lag-1 autocorrelation (manual — np.corrcoef has issues in older Numba)
    autocorr = np.float32(0.0)
    if n > 2:
        r1 = returns[:-1]
        r2 = returns[1:]
        m1 = np.mean(r1)
        m2 = np.mean(r2)
        s1 = np.std(r1)
        s2 = np.std(r2)
        if s1 > 1e-10 and s2 > 1e-10:
            autocorr = np.float32(np.mean((r1 - m1) * (r2 - m2)) / (s1 * s2))

    # Skewness
    skew = np.float32(0.0)
    if n > 2:
        m = np.mean(returns)
        s = np.std(returns)
        if s > 1e-10:
            skew = np.float32(np.mean((returns - m) ** 3) / (s ** 3))

    return volatility, autocorr, skew


@njit(cache=True, fastmath=True)
def _calculate_mfi(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                   volumes: np.ndarray, period: int = 14) -> float:
    """Money Flow Index (oscillator 0-100)."""
    n = len(closes)
    if n < period + 1:
        return 50.0
    pos_flow = 0.0
    neg_flow = 0.0
    for i in range(n - period, n):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        mf = tp * volumes[i]
        if i > 0:
            prev_tp = (highs[i - 1] + lows[i - 1] + closes[i - 1]) / 3.0
            if tp > prev_tp:
                pos_flow += mf
            else:
                neg_flow += mf
        else:
            pos_flow += mf
    if neg_flow < 1e-12:
        return 100.0
    return 100.0 - (100.0 / (1.0 + pos_flow / neg_flow))


@njit(cache=True, fastmath=True)
def _calculate_cci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                   period: int = 20) -> float:
    """Commodity Channel Index."""
    n = len(closes)
    if n < period:
        return 0.0
    tp = (highs[-period:] + lows[-period:] + closes[-period:]) / 3.0
    tp_mean = np.mean(tp)
    tp_mad = np.mean(np.abs(tp - tp_mean))
    if tp_mad < 1e-12:
        return 0.0
    return (tp[-1] - tp_mean) / (0.015 * tp_mad)


@njit(cache=True, fastmath=True)
def _calculate_adx_components(highs: np.ndarray, lows: np.ndarray,
                              closes: np.ndarray, period: int = 14):
    """
    Simplified ADX computation -> (dx, di_direction).

    dx           ∈ [0, 1]   — trend strength (higher = stronger trend)
    di_direction ∈ [-1, 1]  — positive = uptrend, negative = downtrend
    """
    n = len(closes)
    if n < period + 1:
        return np.float32(0.0), np.float32(0.0)
    plus_dm = 0.0
    minus_dm = 0.0
    tr_sum = 0.0
    for i in range(n - period, n):
        if i < 1:
            continue
        tr_val = max(highs[i] - lows[i],
                     max(abs(highs[i] - closes[i - 1]),
                         abs(lows[i] - closes[i - 1])))
        tr_sum += tr_val
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0.0:
            plus_dm += up
        if down > up and down > 0.0:
            minus_dm += down
    if tr_sum < 1e-12:
        return np.float32(0.0), np.float32(0.0)
    plus_di = plus_dm / tr_sum
    minus_di = minus_dm / tr_sum
    di_sum = plus_di + minus_di
    if di_sum < 1e-12:
        return np.float32(0.0), np.float32(0.0)
    dx = np.float32(abs(plus_di - minus_di) / di_sum)
    di_dir = np.float32((plus_di - minus_di) / di_sum)
    return dx, di_dir


@njit(cache=True, fastmath=True)
def calculate_advanced_state_tensor(
    buy_vols: np.ndarray,
    sell_vols: np.ndarray,
    prices: np.ndarray,
    bid_vols: np.ndarray,
    ask_vols: np.ndarray,
) -> np.ndarray:
    """
    Fast-path Numba JIT kernel — core signal features from raw stream inputs.

    Fills slots [0-14] of the 160-dim tensor; remaining slots stay zero and are
    populated by FeatureEngineV2.create_state_vector (which has full OHLCV +
    orderbook + position context).  Call both and overlay for the complete vector.

    Slot layout
    -----------
    [0]  VPIN          — rolling toxic flow (50-bucket window)
    [1]  OFI           — top-of-book order flow imbalance  [-1, +1]
    [2]  volatility    — annualised return σ  (regime energy)
    [3]  autocorr      — lag-1 serial correlation  (trend persistence)
    [4]  skew          — return skewness  (tail directionality)
    [5]  ema9_dist     — (price − EMA9)  / price
    [6]  ema21_dist    — (price − EMA21) / price
    [7]  mom5          — 5-bar log momentum
    [8]  mom10         — 10-bar log momentum
    [9]  mom20         — 20-bar log momentum
    [10] buy_sell_imb  — latest bucket (buy − sell) / total
    [11] bid_ask_imb   — top-of-book (bid − ask) / total
    [12] price_velocity  — last log return
    [13] price_accel     — Δvelocity (second derivative of price)
    [14] vpin_delta      — VPIN[-1] − VPIN[-5]  (toxicity acceleration)
    """
    out = np.zeros(160, dtype=np.float32)
    n = len(prices)
    if n < 2:
        return out

    # Log returns
    returns = np.empty(n - 1, dtype=np.float32)
    for i in range(n - 1):
        if prices[i] > 0.0:
            returns[i] = np.float32(np.log(prices[i + 1] / prices[i]))
        else:
            returns[i] = np.float32(0.0)

    # [0] VPIN — rolling toxic flow
    vpin_arr = _calculate_vpin(buy_vols, sell_vols, 50)
    out[0] = vpin_arr[-1]

    # [1] OFI — top-of-book whale pressure
    out[1] = np.float32(_calculate_ofi(bid_vols, ask_vols))

    # [2-4] Regime triad
    vol, autocorr, skew = _calculate_regime_features(returns)
    out[2] = vol
    out[3] = autocorr
    out[4] = skew

    # [5-6] EMA momentum distances
    p = prices[-1]
    if p > 0.0:
        ema9  = _calculate_ema(prices, 9)[-1]
        ema21 = _calculate_ema(prices, 21)[-1]
        out[5] = np.float32((p - ema9)  / p)
        out[6] = np.float32((p - ema21) / p)

    # [7-9] Log momentum over multiple windows
    if n > 5  and prices[-5]  > 0.0:
        out[7] = np.float32(np.log(prices[-1] / prices[-5]))
    if n > 10 and prices[-10] > 0.0:
        out[8] = np.float32(np.log(prices[-1] / prices[-10]))
    if n > 20 and prices[-20] > 0.0:
        out[9] = np.float32(np.log(prices[-1] / prices[-20]))

    # [10] Buy/sell bucket imbalance
    bv  = buy_vols[-1]
    sv  = sell_vols[-1]
    tot = bv + sv
    if tot > 0.0:
        out[10] = np.float32((bv - sv) / tot)

    # [11] Bid/ask depth imbalance
    if len(bid_vols) > 0 and len(ask_vols) > 0:
        bid0   = bid_vols[-1]
        ask0   = ask_vols[-1]
        tot_ob = bid0 + ask0
        if tot_ob > 0.0:
            out[11] = np.float32((bid0 - ask0) / tot_ob)

    # [12] Price velocity (last log return)
    out[12] = returns[-1]

    # [13] Price acceleration (Δvelocity)
    if len(returns) > 1:
        out[13] = np.float32(returns[-1] - returns[-2])

    # [14] VPIN delta (toxicity acceleration)
    if len(vpin_arr) >= 5:
        out[14] = np.float32(vpin_arr[-1] - vpin_arr[-5])

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureEngineV2 — stateful wrapper (manages normalisation + Redis publish)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngineV2:
    """
    Institutional-grade 160-dimensional state tensor engine.

    JIT-compiled indicator functions execute at C++ speed.
    The class wrapper handles Python-dict unpacking, running-stats, and Redis publish.

    Consumers: AutonomousEnsembleExecutor, InstitutionalEnsemble, DQN trainer, SAC trainer.
    """

    STATE_DIM = 160

    # Feature-group boundaries (inclusive)
    PRICE_START, PRICE_END = 0, 31
    TECH_START, TECH_END = 32, 95
    OB_START, OB_END = 96, 143
    POS_START, POS_END = 144, 159

    def __init__(self, redis_client=None):
        self.state_dim: int = self.STATE_DIM

        # Running statistics for robust normalisation
        self.price_median: float = 0.0
        self.price_mad: float = 1.0
        self.return_median: float = 0.0
        self.return_mad: float = 0.01

        # Output buffer — reused to avoid allocation churn
        self.output: np.ndarray = np.zeros(self.state_dim, dtype=np.float32)

        # Optional Redis handle for tensor publishing
        self._redis = redis_client

        # Per-symbol publish throttle — max one pubsub message per 100ms per symbol
        self._last_publish: dict = {}

        self._warmed_up: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_state_vector(
        self,
        ohlcv: Dict[str, np.ndarray],
        orderbook: Optional[Dict[str, np.ndarray]] = None,
        position: Optional[Dict] = None,
        buy_volumes: Optional[np.ndarray] = None,
        sell_volumes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the complete 160-dim state vector.

        Parameters
        ----------
        ohlcv : dict  – 'open', 'high', 'low', 'close', 'volume' (np.ndarray each)
        orderbook : optional dict – 'bids', 'asks', 'bid_volumes', 'ask_volumes'
        position : optional dict – 'size', 'entry_price', 'pnl', 'leverage',
                   'margin_used', 'account_equity', 'side'
        buy_volumes : optional np.ndarray – per-bucket buy volumes (for VPIN)
        sell_volumes : optional np.ndarray – per-bucket sell volumes (for VPIN)

        Returns
        -------
        np.ndarray  shape (160,)  dtype float32
        """
        self.output.fill(0.0)

        closes = np.asarray(ohlcv.get("close", np.array([], dtype=np.float32)), dtype=np.float32)
        opens = np.asarray(ohlcv.get("open", closes), dtype=np.float32)
        highs = np.asarray(ohlcv.get("high", closes), dtype=np.float32)
        lows = np.asarray(ohlcv.get("low", closes), dtype=np.float32)
        volumes = np.asarray(ohlcv.get("volume", np.zeros_like(closes)), dtype=np.float32)

        n = len(closes)
        if n < 20:
            return self.output.copy()

        current_price = float(closes[-1])

        # Update running normalisation stats
        self._update_stats(closes)

        # ═══════ Price Features [0–31] ═══════
        self._extract_price_features(closes, volumes)

        # ═══════ Technical Features [32–95] ═══════
        self._extract_technical_features(opens, highs, lows, closes, volumes,
                                         buy_volumes, sell_volumes)

        # ═══════ Orderbook Features [96–143] ═══════
        if orderbook is not None:
            self._extract_orderbook_features(orderbook, current_price)

        # ═══════ Position Features [144–159] ═══════
        if position is not None:
            self._extract_position_features(position, current_price)

        return self.output.copy()

    def compute_and_publish(
        self,
        symbol: str,
        ohlcv: Dict[str, np.ndarray],
        orderbook: Optional[Dict[str, np.ndarray]] = None,
        position: Optional[Dict] = None,
        buy_volumes: Optional[np.ndarray] = None,
        sell_volumes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the tensor **and** publish to Redis key ``market:state_tensor:{symbol}``.
        Also publishes:
        - ``market:cvd:{symbol}``    — normalised CVD trend via volume_tracker_ext
        - ``market:stress:{symbol}`` — composite stress index [0, 1]

        Returns the tensor for immediate local consumption.
        """
        tensor = self.create_state_vector(ohlcv, orderbook, position,
                                          buy_volumes, sell_volumes)
        if self._redis is not None:
            try:
                key = f"market:state_tensor:{symbol}"
                payload = json.dumps({
                    "tensor": tensor.tolist(),
                    "ts": time.time(),
                    "symbol": symbol,
                    "dim": self.state_dim,
                })
                self._redis.set(key, payload)
                # Throttle pubsub publish to 100ms per symbol — prevents subscriber buffer overflow
                _now = time.time()
                if _now - self._last_publish.get(symbol, 0.0) >= 0.1:
                    self._redis.publish("feature:tensors", payload)
                    self._last_publish[symbol] = _now
            except Exception as exc:
                logger.warning("Redis publish failed for %s: %s", symbol, exc)

            # ─── CVD publisher (piggybacks on buy/sell volumes) ───────────
            if buy_volumes is not None and sell_volumes is not None and len(buy_volumes) > 0:
                try:
                    from volume_tracker_ext import update_volume, init_redis
                    # Ensure the tracker module has our Redis client
                    init_redis(self._redis)
                    # Feed latest bucket — update() auto-publishes market:cvd:{symbol}
                    update_volume(symbol, float(buy_volumes[-1]), float(sell_volumes[-1]))
                except Exception as exc:
                    logger.debug("CVD piggyback failed for %s: %s", symbol, exc)

            # ─── Stress index publisher ───────────────────────────────────
            try:
                self._publish_stress_index(symbol, tensor, ohlcv)
            except Exception as exc:
                logger.debug("Stress publish failed for %s: %s", symbol, exc)

        return tensor

    # ── Market Stress Index ──────────────────────────────────────────────
    def _publish_stress_index(self, symbol: str, tensor: np.ndarray, ohlcv: Dict[str, np.ndarray]):
        """
        Composite stress index [0.0, 1.0] published to ``market:stress:{symbol}``.

        Components (equal-weighted):
        1. ATR spike — current ATR / rolling-mean ATR   (>2× = max stress)
        2. Volume spike — current vol / rolling-mean vol (>3× = max stress)
        3. Spread toxicity — VPIN proxy from tensor      (>0.7 = max stress)
        4. Orderbook thinning — bid-ask imbalance extreme (|imb|>0.8 = max stress)

        Consumed by:
        - ORGAN 5 (~L2735): blocks entry if >0.80, halves if >0.65
        - autonomous_ensemble_executor pre-flight (~L135)
        """
        if self._redis is None:
            return

        closes = ohlcv.get("close")
        volumes = ohlcv.get("volume")
        if closes is None or len(closes) < 20:
            return

        components = []

        # 1. ATR spike vs 100-bar baseline
        try:
            highs = ohlcv.get("high", closes)
            lows  = ohlcv.get("low", closes)
            # Simple ATR from last 14 bars
            n = min(14, len(closes) - 1)
            if n > 0:
                trs = np.empty(n, dtype=np.float64)
                for i in range(n):
                    j = len(closes) - n + i
                    trs[i] = max(
                        highs[j] - lows[j],
                        abs(highs[j] - closes[j - 1]),
                        abs(lows[j]  - closes[j - 1]),
                    )
                current_atr = np.mean(trs)
                # Baseline: average over longer window
                baseline_n = min(100, len(closes) - 1)
                if baseline_n > 14:
                    bl_trs = np.empty(baseline_n, dtype=np.float64)
                    for i in range(baseline_n):
                        j = len(closes) - baseline_n + i
                        bl_trs[i] = max(
                            highs[j] - lows[j],
                            abs(highs[j] - closes[j - 1]),
                            abs(lows[j]  - closes[j - 1]),
                        )
                    baseline_atr = np.mean(bl_trs)
                else:
                    baseline_atr = current_atr
                ratio = current_atr / (baseline_atr + 1e-12)
                # Normalise: ratio 1.0->0.0 stress, ratio ≥2.0->1.0 stress
                atr_stress = float(np.clip((ratio - 1.0), 0.0, 1.0))
                components.append(atr_stress)
        except Exception:
            pass

        # 2. Volume spike vs rolling mean
        if volumes is not None and len(volumes) >= 20:
            try:
                recent_vol = float(volumes[-1])
                mean_vol = float(np.mean(volumes[-100:]))
                if mean_vol > 0:
                    vol_ratio = recent_vol / mean_vol
                    # ratio 1.0->0.0, ≥3.0->1.0
                    vol_stress = float(np.clip((vol_ratio - 1.0) / 2.0, 0.0, 1.0))
                    components.append(vol_stress)
            except Exception:
                pass

        # 3. VPIN from tensor (slot varies, use calculated value from output)
        # VPIN is typically at index ~88 (after ADX pair at 86-87)
        try:
            vpin_idx = 49  # VPIN slot in 160-dim layout (technical block, after DI pair at 47-48)
            if len(tensor) > vpin_idx:
                vpin_val = float(tensor[vpin_idx])
                # VPIN >0.5 is concerning, >0.75 is extreme
                vpin_stress = float(np.clip((vpin_val - 0.3) / 0.5, 0.0, 1.0))
                components.append(vpin_stress)
        except Exception:
            pass

        # 4. Orderbook imbalance from tensor
        try:
            ofi_idx = 98  # OFI slot in orderbook section (orderbook block [96-143])
            if len(tensor) > ofi_idx:
                ofi_val = abs(float(tensor[ofi_idx]))
                # |OFI|>0.5 is one-sided, >0.8 is extreme
                ob_stress = float(np.clip((ofi_val - 0.3) / 0.5, 0.0, 1.0))
                components.append(ob_stress)
        except Exception:
            pass

        if not components:
            return

        # Composite: equal-weighted average
        stress = float(np.mean(components))
        stress = float(np.clip(stress, 0.0, 1.0))

        self._redis.set(f"market:stress:{symbol}", str(round(stress, 4)))

    def warmup(self):
        """
        Force JIT compilation of every @njit helper so the first real call
        is not penalised by compilation latency.

        This is CPU-bound and can take 5–30 s on a cold container.  Prefer
        calling ``warmup_async()`` from an async context so the event loop
        stays responsive during startup.
        """
        if self._warmed_up:
            return
        logger.info("FeatureEngineV2 — JIT warmup starting …")
        t0 = time.time()

        dp = np.random.rand(100).astype(np.float32) + 1.0
        dv = np.random.rand(100).astype(np.float32) + 0.1
        dh = dp * 1.01
        dl = dp * 0.99

        _calculate_returns(dp)
        _calculate_sma(dp, 14)
        _calculate_ema(dp, 14)
        rets = _calculate_returns(dp)
        _calculate_rsi(rets, 14)
        _calculate_atr(dh, dl, dp, 14)
        _calculate_bollinger(dp, 20, 2.0)
        _robust_normalize(np.float32(0.5), 0.0, 1.0)
        _calculate_vpin(dv, dv, 50)
        _calculate_ofi(dv[:5], dv[:5])
        _calculate_regime_features(rets)
        _calculate_mfi(dh, dl, dp, dv, 14)
        _calculate_cci(dh, dl, dp, 20)
        _calculate_adx_components(dh, dl, dp, 14)

        self._warmed_up = True
        logger.info("FeatureEngineV2 — JIT warmup complete (%.2fs)", time.time() - t0)

    async def warmup_async(self, executor=None) -> None:
        """
        Run ``warmup()`` in a thread-pool executor so the asyncio event loop
        stays free during the CPU-intensive Numba JIT compilation phase.

        ``executor`` is passed directly to ``loop.run_in_executor``; pass
        ``None`` to use the default executor (usually a ThreadPoolExecutor
        whose size is ``min(32, cpu_count + 4)``).
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.warmup)

    # ------------------------------------------------------------------
    # Internal extraction methods
    # ------------------------------------------------------------------

    def _update_stats(self, prices: np.ndarray):
        """Update running median / MAD for robust normalisation."""
        self.price_median = float(np.median(prices))
        self.price_mad = float(np.median(np.abs(prices - self.price_median))) + 1e-8
        if len(prices) > 1:
            rets = _calculate_returns(prices)
            self.return_median = float(np.median(rets))
            self.return_mad = float(np.median(np.abs(rets - self.return_median))) + 1e-8

    # ──────────────────────────── PRICE [0–31] ────────────────────────────

    def _extract_price_features(self, closes: np.ndarray, volumes: np.ndarray):
        """Price / return / momentum / volume features -> slots [0, 31]."""
        n = len(closes)
        returns = _calculate_returns(closes) if n > 1 else np.zeros(1, dtype=np.float32)
        nr = len(returns)

        idx = 0

        # ── Current return (robust-normalised) ──
        self.output[idx] = _robust_normalize(returns[-1], self.return_median, self.return_mad)
        idx += 1

        # ── Return statistics over multiple horizons ──
        for period in (5, 10, 20):
            if nr >= period:
                self.output[idx] = _robust_normalize(
                    np.mean(returns[-period:]), self.return_median, self.return_mad
                )
                idx += 1
                self.output[idx] = float(np.std(returns[-period:])) * np.sqrt(252.0)
                idx += 1
            else:
                idx += 2

        # ── SMA distance from price (%) ──
        for period in (5, 10, 20, 50):
            if n >= period:
                sma = _calculate_sma(closes, period)
                self.output[idx] = (closes[-1] - sma[-1]) / closes[-1] * 100.0
            idx += 1

        # ── Price momentum (% change over N bars) ──
        for period in (3, 5, 10, 20):
            if n > period and closes[-(period + 1)] > 0:
                self.output[idx] = (closes[-1] - closes[-(period + 1)]) / closes[-(period + 1)] * 100.0
            idx += 1

        # ── Volume ratio vs 20-bar SMA ──
        if len(volumes) >= 20:
            vol_sma = float(np.mean(volumes[-20:]))
            if vol_sma > 0:
                self.output[idx] = volumes[-1] / vol_sma
        idx += 1

        # ── Trend structure (higher-highs / lower-lows, last 10 bars) ──
        if n >= 10:
            hh = bool(np.all(closes[-5:] > closes[-10:-5]))
            ll = bool(np.all(closes[-5:] < closes[-10:-5]))
            self.output[idx] = 1.0 if hh else (-1.0 if ll else 0.0)
        idx += 1

        # ── EMA distance from price (%) ──
        for period in (12, 26, 50):
            if n >= period:
                ema = _calculate_ema(closes, period)
                self.output[idx] = (closes[-1] - ema[-1]) / closes[-1] * 100.0
            idx += 1

        # ── 5-bar price range (normalised) ──
        if n >= 5:
            rh = float(np.max(closes[-5:]))
            rl = float(np.min(closes[-5:]))
            if closes[-1] > 0:
                self.output[idx] = (rh - rl) / closes[-1] * 100.0
        idx += 1

        # ── Position within 20-bar range [0, 1] ──
        if n >= 20:
            h20 = float(np.max(closes[-20:]))
            l20 = float(np.min(closes[-20:]))
            rng = h20 - l20
            if rng > 0:
                self.output[idx] = (closes[-1] - l20) / rng
        idx += 1

        # Slots up to 31 remain zero-filled (reserved for future price features)

    # ──────────────────────────── TECHNICAL [32–95] ────────────────────────

    def _extract_technical_features(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        buy_volumes: Optional[np.ndarray],
        sell_volumes: Optional[np.ndarray],
    ):
        """Technical indicators + VPIN + Regime -> slots [32, 95]."""
        n = len(closes)
        returns = _calculate_returns(closes) if n > 1 else np.zeros(1, dtype=np.float32)
        nr = len(returns)

        idx = 32

        # ── RSI(14) centred ──
        if nr >= 14:
            rsi = _calculate_rsi(returns, 14)
            self.output[idx] = rsi[-1] / 100.0 - 0.5
        idx += 1

        # ── RSI(7) ──
        if nr >= 7:
            rsi7 = _calculate_rsi(returns, 7)
            self.output[idx] = rsi7[-1] / 100.0 - 0.5
        idx += 1

        # ── RSI(21) ──
        if nr >= 21:
            rsi21 = _calculate_rsi(returns, 21)
            self.output[idx] = rsi21[-1] / 100.0 - 0.5
        idx += 1

        # ── MACD(12,26,9) ──
        if n >= 26:
            ema12 = _calculate_ema(closes, 12)
            ema26 = _calculate_ema(closes, 26)
            macd = ema12 - ema26
            signal = _calculate_ema(macd, 9)
            histogram = macd - signal
            self.output[idx] = macd[-1] / closes[-1]
            idx += 1
            self.output[idx] = histogram[-1] / closes[-1]
            idx += 1
            self.output[idx] = signal[-1] / closes[-1]
            idx += 1
        else:
            idx += 3

        # ── Bollinger Bands(20,2) ──
        if n >= 20:
            upper, middle, lower = _calculate_bollinger(closes, 20, 2.0)
            bw = upper[-1] - middle[-1]
            self.output[idx] = (closes[-1] - middle[-1]) / (bw + 1e-8)
            idx += 1
            self.output[idx] = (upper[-1] - lower[-1]) / middle[-1]
            idx += 1
        else:
            idx += 2

        # ── ATR(14) normalised by price ──
        if n >= 14:
            atr14 = _calculate_atr(highs, lows, closes, 14)
            self.output[idx] = atr14[-1] / closes[-1]
        idx += 1

        # ── ATR(7) ──
        if n >= 7:
            atr7 = _calculate_atr(highs, lows, closes, 7)
            self.output[idx] = atr7[-1] / closes[-1]
        idx += 1

        # ── ATR(21) ──
        if n >= 21:
            atr21 = _calculate_atr(highs, lows, closes, 21)
            self.output[idx] = atr21[-1] / closes[-1]
        idx += 1

        # ── Stochastic %K(14) centred ──
        if n >= 14:
            lo14 = float(np.min(lows[-14:]))
            hi14 = float(np.max(highs[-14:]))
            rng = hi14 - lo14
            if rng > 0:
                self.output[idx] = (closes[-1] - lo14) / rng - 0.5
        idx += 1

        # ── Williams %R(14) centred ──
        if n >= 14:
            hi14 = float(np.max(highs[-14:]))
            lo14 = float(np.min(lows[-14:]))
            rng = hi14 - lo14
            if rng > 0:
                self.output[idx] = (hi14 - closes[-1]) / rng - 0.5
        idx += 1

        # ── MFI(14) centred ──
        if n >= 15:
            mfi = _calculate_mfi(highs, lows, closes, volumes, 14)
            self.output[idx] = mfi / 100.0 - 0.5
        idx += 1

        # ── CCI(20) normalised ──
        if n >= 20:
            cci = _calculate_cci(highs, lows, closes, 20)
            self.output[idx] = cci / 200.0
        idx += 1

        # ── ADX / DI direction ──
        if n >= 15:
            dx, di_dir = _calculate_adx_components(highs, lows, closes, 14)
            self.output[idx] = dx
            idx += 1
            self.output[idx] = di_dir
            idx += 1
        else:
            idx += 2

        # ═══════ VPIN (Volume-Sync Probability of Informed Trading) ═══════
        if buy_volumes is not None and sell_volumes is not None and len(buy_volumes) > 0:
            vpin = _calculate_vpin(
                np.asarray(buy_volumes, dtype=np.float32),
                np.asarray(sell_volumes, dtype=np.float32),
                min(50, len(buy_volumes)),
            )
            self.output[idx] = vpin[-1]
            idx += 1
            # VPIN delta (recent shift in toxicity)
            if len(vpin) >= 5:
                self.output[idx] = vpin[-1] - vpin[-5]
            idx += 1
        else:
            idx += 2  # reserve

        # ═══════ Regime Features ═══════
        if nr >= 20:
            window = returns[-60:] if nr >= 60 else returns
            vol, autocorr, skew = _calculate_regime_features(window)
            self.output[idx] = vol
            idx += 1
            self.output[idx] = autocorr
            idx += 1
            self.output[idx] = skew
            idx += 1

            # Short-term / long-term volatility ratio (regime shift detector)
            if nr >= 60:
                short_vol = float(np.std(returns[-20:])) * np.sqrt(252.0)
                long_vol = float(np.std(returns[-60:])) * np.sqrt(252.0)
                self.output[idx] = short_vol / (long_vol + 1e-8)
            idx += 1
        else:
            idx += 4  # reserve

        # ── Return kurtosis (tail risk) ──
        if nr >= 20:
            m = float(np.mean(returns[-20:]))
            s = float(np.std(returns[-20:]))
            if s > 1e-10:
                self.output[idx] = float(np.mean((returns[-20:] - m) ** 4)) / (s ** 4) - 3.0
        idx += 1

        # Slots up to 95 remain zero-filled (reserved for future technical features)

    # ──────────────────────────── ORDERBOOK [96–143] ──────────────────────

    def _extract_orderbook_features(self, orderbook: Dict[str, np.ndarray],
                                    mid_price: float):
        """Orderbook microstructure -> slots [96, 143]."""
        bids = np.asarray(orderbook.get("bids", np.array([], dtype=np.float32)), dtype=np.float32)
        asks = np.asarray(orderbook.get("asks", np.array([], dtype=np.float32)), dtype=np.float32)
        bid_vols = np.asarray(orderbook.get("bid_volumes", np.array([], dtype=np.float32)), dtype=np.float32)
        ask_vols = np.asarray(orderbook.get("ask_volumes", np.array([], dtype=np.float32)), dtype=np.float32)

        if len(bids) == 0 or len(asks) == 0 or mid_price <= 0:
            return

        idx = 96

        # ── Spread (bps) ──
        spread = asks[0] - bids[0]
        self.output[idx] = spread / mid_price * 10000.0
        idx += 1

        # ── Mid-price distance (bps) ──
        mid = (bids[0] + asks[0]) / 2.0
        self.output[idx] = (mid - mid_price) / mid_price * 10000.0
        idx += 1

        # ── Aggregate OFI (top-of-book) ──
        if len(bid_vols) > 0 and len(ask_vols) > 0:
            self.output[idx] = _calculate_ofi(bid_vols, ask_vols)
        idx += 1

        # ── Per-level orderbook imbalance (5 levels) ──
        for i in range(5):
            if i < len(bid_vols) and i < len(ask_vols):
                total = bid_vols[i] + ask_vols[i]
                if total > 0:
                    self.output[idx] = (bid_vols[i] - ask_vols[i]) / total
            idx += 1

        # ── Cumulative depth imbalance (5 levels) ──
        max_depth = min(10, len(bid_vols), len(ask_vols))
        if max_depth > 0:
            cum_bid = np.cumsum(bid_vols[:max_depth])
            cum_ask = np.cumsum(ask_vols[:max_depth])
            for i in range(min(5, max_depth)):
                total = cum_bid[i] + cum_ask[i]
                if total > 0:
                    self.output[idx] = (cum_bid[i] - cum_ask[i]) / total
                idx += 1
        else:
            idx += 5

        # ── Bid price levels — distance from mid (bps), top 10 ──
        for i in range(10):
            if i < len(bids):
                self.output[idx] = (bids[i] - mid_price) / mid_price * 10000.0
            idx += 1

        # ── Ask price levels — distance from mid (bps), top 10 ──
        for i in range(10):
            if i < len(asks):
                self.output[idx] = (asks[i] - mid_price) / mid_price * 10000.0
            idx += 1

        # ── Total depth ratio (bid vs ask over 10 levels) ──
        if max_depth > 0:
            total_bid = float(np.sum(bid_vols[:max_depth]))
            total_ask = float(np.sum(ask_vols[:max_depth]))
            total_depth = total_bid + total_ask
            if total_depth > 0:
                self.output[idx] = (total_bid - total_ask) / total_depth
        idx += 1

        # ── Weighted mid price (volume-weighted) distance ──
        if len(bid_vols) > 0 and len(ask_vols) > 0:
            if bid_vols[0] + ask_vols[0] > 0:
                wmid = (bids[0] * ask_vols[0] + asks[0] * bid_vols[0]) / (bid_vols[0] + ask_vols[0])
                self.output[idx] = (wmid - mid_price) / mid_price * 10000.0
        idx += 1

        # Slots up to 143 remain zero-filled (reserved)

    # ──────────────────────────── POSITION [144–159] ──────────────────────

    def _extract_position_features(self, position: Dict, current_price: float):
        """Position state -> slots [144, 159]."""
        idx = 144

        size = float(position.get("size", 0))
        entry = float(position.get("entry_price", 0))
        pnl = float(position.get("pnl", 0))
        leverage = float(position.get("leverage", 1))
        margin_used = float(position.get("margin_used", 0))
        equity = float(position.get("account_equity", 1))
        side = str(position.get("side", "none")).lower()
        duration_s = float(position.get("duration_seconds", 0))

        # Position size normalised by account capacity
        if equity > 0 and current_price > 0:
            self.output[idx] = size / (equity / current_price)
        idx += 1

        # P&L normalised by equity
        if equity > 0:
            self.output[idx] = pnl / equity
        idx += 1

        # Distance from entry (%)
        if entry > 0:
            self.output[idx] = (current_price - entry) / entry * 100.0
        idx += 1

        # Leverage normalised (max 125x on Bybit -> /125)
        self.output[idx] = leverage / 125.0
        idx += 1

        # Margin utilisation
        if equity > 0:
            self.output[idx] = margin_used / equity
        idx += 1

        # Side encoding: −1 short, 0 flat, +1 long
        side_enc = {"long": 1.0, "buy": 1.0, "short": -1.0, "sell": -1.0}
        self.output[idx] = side_enc.get(side, 0.0)
        idx += 1

        # Position duration (log-seconds normalised)
        if duration_s > 0:
            self.output[idx] = np.log1p(duration_s) / 10.0  # ~10 ≈ ln(22026) ≈ 6h
        idx += 1

        # Unrealised P&L / ATR proxy (risk-normalised)
        atr_pct = float(position.get("atr_pct", 0))
        if atr_pct > 0 and entry > 0:
            price_move_pct = abs(current_price - entry) / entry
            self.output[idx] = price_move_pct / atr_pct
        idx += 1

        # Slots up to 159 remain zero-filled (reserved)


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singleton accessor
# ═══════════════════════════════════════════════════════════════════════════════

_engine_instance: Optional[FeatureEngineV2] = None


def get_feature_engine(redis_client=None) -> FeatureEngineV2:
    """Get or create the global FeatureEngineV2 instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FeatureEngineV2(redis_client=redis_client)
        _engine_instance.warmup()
    elif redis_client is not None and _engine_instance._redis is None:
        _engine_instance._redis = redis_client
    return _engine_instance


if __name__ == "__main__":
    import redis as _redis_mod
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    log = logging.getLogger("NEO.FeatureEngine")

    r = _redis_mod.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    engine = get_feature_engine(r)

    KLINE_HISTORY_KEY = "market:kline_history:{}:1"
    TICKER_KEY        = "market:ticker:{}"
    MIN_BARS          = 20
    SWEEP_INTERVAL    = 10.0  # periodic sweep for symbols not yet seen on ticks

    def _build_ohlcv(symbol: str):
        raw = r.lrange(KLINE_HISTORY_KEY.format(symbol), -200, -1)
        if not raw or len(raw) < MIN_BARS:
            return None
        bars = []
        for b in raw:
            try:
                bars.append(json.loads(b))
            except Exception:
                pass
        if len(bars) < MIN_BARS:
            return None
        return {
            "open":   np.array([b["open"]   for b in bars], dtype=np.float64),
            "high":   np.array([b["high"]   for b in bars], dtype=np.float64),
            "low":    np.array([b["low"]    for b in bars], dtype=np.float64),
            "close":  np.array([b["close"]  for b in bars], dtype=np.float64),
            "volume": np.array([b["volume"] for b in bars], dtype=np.float64),
        }

    def _build_orderbook(symbol: str):
        ticker_raw = r.get(TICKER_KEY.format(symbol))
        if not ticker_raw:
            return None
        try:
            t   = json.loads(ticker_raw)
            bid = float(t.get("bid", t.get("last_price", 0)))
            ask = float(t.get("ask", bid * 1.0001))
            if bid <= 0:
                return None
            return {
                "bids": np.array([[bid, 1.0]], dtype=np.float64),
                "asks": np.array([[ask, 1.0]], dtype=np.float64),
            }
        except Exception:
            return None

    def _process(symbol: str):
        ohlcv = _build_ohlcv(symbol)
        if ohlcv is None:
            return
        ob = _build_orderbook(symbol)
        try:
            engine.compute_and_publish(symbol=symbol, ohlcv=ohlcv, orderbook=ob)
        except Exception as exc:
            log.error("Feature compute failed for %s: %s", symbol, exc)

    import queue as _queue_mod
    import threading as _fe_threading
    import websocket as _ws_client

    _RUST_WS_ALL = "ws://127.0.0.1:8080/ws/all"
    _tick_q: _queue_mod.SimpleQueue = _queue_mod.SimpleQueue()

    def _ws_feeder():
        """Background thread — connects to Axum fast-lane and enqueues tick symbols."""
        while True:
            try:
                ws = _ws_client.create_connection(_RUST_WS_ALL, timeout=5)
                ws.settimeout(1.0)
                log.info("FeatureEngine WS connected to %s", _RUST_WS_ALL)
                while True:
                    try:
                        raw = ws.recv()
                    except _ws_client.WebSocketTimeoutException:
                        continue
                    try:
                        d = json.loads(raw)
                        # Trigger tensor computation on:
                        #  - "trade"  messages from the publicTrade.* livestream
                        #              (every individual exchange fill — true micro-structure)
                        #  - "tick"   messages from the tickers.* snapshot stream
                        #              (best bid/ask + 24 h stats; lower frequency)
                        if d.get("type") in ("tick", "trade"):
                            sym = d.get("symbol", "")
                            if sym:
                                _tick_q.put_nowait(sym)
                    except Exception:
                        pass
            except Exception as exc:
                log.warning("FeatureEngine WS disconnected: %s — reconnecting in 2s", exc)
            time.sleep(2)

    _fe_ws_thread = _fe_threading.Thread(target=_ws_feeder, daemon=True, name="fe-ws-feed")
    _fe_ws_thread.start()
    log.info("🚀 FeatureEngineV2 started — Axum fast-lane %s", _RUST_WS_ALL)

    last_sweep = 0.0
    try:
        while True:
            now = time.time()

            # Drain all queued tick symbols — batch process to never fall behind the firehose
            while True:
                try:
                    _process(_tick_q.get_nowait())
                except _queue_mod.Empty:
                    break

            if now - last_sweep >= SWEEP_INTERVAL:
                for k in r.keys("market:kline_history:*:1"):
                    parts = k.split(":")
                    if len(parts) >= 3:
                        _process(parts[2])
                last_sweep = now

            time.sleep(0.001)
    except KeyboardInterrupt:
        log.info("FeatureEngineV2 stopped.")
