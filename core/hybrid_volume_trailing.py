"""
Hybrid Volume-Anchored Trailing Stop — NEO SUPREME
===================================================
Combines standard Price-Based Trailing (ATR/Pct) with an Anchored Micro-VWAP.
- If price crashes through the volume-weighted shelf, it exits.
- If volume surges, it dynamically widens the price stop to avoid wick-outs.
- Thread-safe and designed for direct import into any execution script (Sniper/DQN).
"""

import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math

class TrailingDirection(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class HybridStopConfig:
    direction: TrailingDirection
    entry_price: float
    base_trail_pct: float = 0.01       # Baseline trailing distance (1.0%)
    vwap_trail_pct: float = 0.003       # Tight VWAP breach threshold (0.3%)
    dynamic_expansion: bool = True      # Widen stop during volume surges?

class HybridVolumeTrailingStop:
    """
    Dual-Threat Trailing Stop: 
    Monitors both the Peak Price and the Peak Volume-Weighted Average Price.
    Triggers if EITHER the price retraces too far, or the volume support breaks.
    """
    def __init__(self, symbol: str, config: HybridStopConfig):
        self.symbol = symbol
        self.config = config
        
        # ── Price State ──
        self.highest_price = config.entry_price
        self.lowest_price = config.entry_price
        
        # ── Volume (Micro-VWAP) State ──
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.highest_vwap = 0.0
        self.lowest_vwap = float('inf')
        self.peak_volume_rate = 0.0
        self.symbol_vol_profile = {}  # populated from Rust tensor or Redis: {'atr_pct': 0.012, 'avg_vol': 1e6, 'is_high_beta': True, ...}
        
        # ── Execution State ──
        self.is_triggered = False
        self.trigger_reason = ""
        self.current_stop_price = 0.0

        # ── Breakout Follow-Through State ──
        self._breakout_active: bool = False
        self._breakout_level: float = 0.0
        self._breakout_vwap_anchor: float = 0.0
        self._breakout_trail_saved: float = 0.0
        self._pullback_signal_fired: bool = False
        self._near_extreme_tightened: bool = False
        self.pullback_entry_signal: bool = False  # consumed + reset by caller after read

        # ── Volume EMA (self-calibrating surge detector) ──
        self._vol_ema: float = 0.0
        self._vol_ema_alpha: float = 0.05   # ~20-tick window

        self._lock = threading.RLock()

    def _get_adaptive_trail_pct(self, current_price: float, tick_volume: float, rust_metrics: dict = None) -> float:
        """Helper to calculate the dynamic trail percentage based on volume and ATR."""
        import math
        base = self.config.base_trail_pct  # e.g. 0.008 for BTC, 0.003 for PEPE

        if rust_metrics:
            vol_ratio = tick_volume / (rust_metrics.get("avg_vol_20", 100000.0) + 1)
            near_sr = rust_metrics.get("near_sr_strength", 0.0)  

            if rust_metrics.get("vol_indicates_breakout", False) and vol_ratio > 1.8:
                base = min(base * 2.2, 0.06)  # widen to follow through
            elif near_sr > 0.7:
                base = max(0.0025, base * 0.65)  # tighten near S/R

            # Volume surge widens to avoid wick-outs
            if self.config.dynamic_expansion and vol_ratio > 2.0:
                base *= (1.0 + math.log1p(vol_ratio - 1.0) * 0.6)

        # Symbol-aware floor/ceiling clamps
        symbol_upper = self.symbol.upper()
        if "BTC" in symbol_upper:
            base = max(base, 0.006)   # min ~0.6% for BTC
        elif any(x in symbol_upper for x in ["PEPE", "DOGE", "SHIB"]):
            base = min(base, 0.004)   # tighter max for memes

        return max(0.0025, min(base, 0.08))  # absolute hard clamps

    def ingest_tick(self, current_price: float, tick_volume: float, rust_metrics: dict = None) -> bool:
        """Main tick processor. Returns True if the trailing stop is triggered."""
        with self._lock:
            if self.is_triggered:
                return True

            # 1. Capture pre-update extremes for breakout detection
            old_highest = self.highest_price
            old_lowest  = self.lowest_price

            # 2. Update Price Extremes
            if current_price > self.highest_price:
                self.highest_price = current_price
            if current_price < self.lowest_price:
                self.lowest_price = current_price

            # 3. Update Anchored Micro-VWAP + Volume EMA
            self.cum_pv += current_price * tick_volume
            self.cum_vol += tick_volume
            current_vwap = self.vwap
            
            self._vol_ema = (
                self._vol_ema_alpha * tick_volume
                + (1.0 - self._vol_ema_alpha) * self._vol_ema
                if self._vol_ema > 0.0 else tick_volume
            )

            if tick_volume > self.peak_volume_rate:
                self.peak_volume_rate = tick_volume

            if self.config.direction == TrailingDirection.LONG:
                if current_vwap > self.highest_vwap:
                    self.highest_vwap = current_vwap
            else:
                if current_vwap < self.lowest_vwap:
                    self.lowest_vwap = current_vwap

            # 4. Self-detecting S/R breakout & pullback
            _is_vol_surge = self._vol_ema > 0 and tick_volume > self._vol_ema * 2.2
            
            if self.config.direction == TrailingDirection.LONG:
                _new_extreme  = current_price > old_highest
                _near_extreme = current_price >= self.highest_price * 0.998
            else:
                _new_extreme  = current_price < old_lowest
                _near_extreme = current_price <= self.lowest_price * 1.002

            if _new_extreme:
                self._near_extreme_tightened = False  # reset on every new extreme

            if _new_extreme and _is_vol_surge and not self._breakout_active:
                # New extreme confirmed by institutional volume → follow the breakout
                self._enter_breakout_follow_internal(current_price, current_vwap)
            elif (_near_extreme and not _new_extreme
                  and not _is_vol_surge
                  and not self._near_extreme_tightened
                  and not self._breakout_active):
                # Drifting near the extreme without volume conviction → reversal risk → tighten once
                self.config.base_trail_pct = max(0.003, self.config.base_trail_pct * 0.7)
                self._near_extreme_tightened = True

            # Pullback-capture: after a breakout, price returns to VWAP anchor with receding volume
            if self._breakout_active and not self._pullback_signal_fired:
                anchor = self._breakout_vwap_anchor
                if anchor > 0 and abs(current_price - anchor) / anchor <= 0.003:
                    if self._vol_ema > 0 and tick_volume < self._vol_ema * 0.6:
                        self._pullback_signal_fired = True
                        self.pullback_entry_signal  = True

            # 5. Calculate Dynamic Stop Distances using the cleanly separated helper
            dynamic_trail_pct = self._get_adaptive_trail_pct(current_price, tick_volume, rust_metrics)

            # 6. Check Triggers (Dual-Threat: Price & VWAP)
            if self.config.direction == TrailingDirection.LONG:
                price_stop = self.highest_price * (1.0 - dynamic_trail_pct)
                vwap_stop  = self.highest_vwap  * (1.0 - self.config.vwap_trail_pct)
                self.current_stop_price = max(price_stop, vwap_stop)
                
                if current_price <= self.current_stop_price:
                    self.is_triggered   = True
                    self.trigger_reason = "VWAP_BREACH" if self.current_stop_price == vwap_stop else "PRICE_RETRACE"
                    return True
            else:
                price_stop = self.lowest_price * (1.0 + dynamic_trail_pct)
                vwap_stop  = self.lowest_vwap  * (1.0 + self.config.vwap_trail_pct)
                self.current_stop_price = min(price_stop, vwap_stop)
                
                if current_price >= self.current_stop_price:
                    self.is_triggered   = True
                    self.trigger_reason = "VWAP_BREACH" if self.current_stop_price == vwap_stop else "PRICE_RETRACE"
                    return True

            return False

    def _enter_breakout_follow_internal(self, current_price: float, current_vwap: float) -> None:
        """
        Internal: widen the trail and anchor the VWAP shelf when a volume-confirmed
        breakout is detected.  Resets peak tracking from the breakout point so the
        trailing stop follows cleanly from the new high/low.
        """
        self._breakout_active      = True
        self._breakout_level       = current_price
        self._breakout_vwap_anchor = current_vwap
        self._breakout_trail_saved = self.config.base_trail_pct
        # Widen trail up to 2.5× for follow-through headroom, cap at 4%
        self.config.base_trail_pct = min(self._breakout_trail_saved * 2.5, 0.04)
        if self.config.direction == TrailingDirection.LONG:
            self.highest_price = current_price
            self.highest_vwap  = current_vwap
        else:
            self.lowest_price = current_price
            self.lowest_vwap  = current_vwap
        self._pullback_signal_fired = False
        self.pullback_entry_signal  = False

    @property
    def vwap(self) -> float:
        """Returns the current Anchored Micro-VWAP."""
        with self._lock:
            return (self.cum_pv / self.cum_vol) if self.cum_vol > 0 else self.config.entry_price

    def force_tighten_leash(self, factor: float = 0.5) -> None:
        """
        Shrink trailing distance when price approaches an HTF S/R zone.
        Called by T1 when market_memory signals proximity to a macro resistance/support level.
        factor < 1.0 tightens the stop, > 1.0 widens. Clamps to 0.003 minimum (0.3%).
        """
        with self._lock:
            self.config.base_trail_pct = max(0.003, self.config.base_trail_pct * factor)

    def get_state(self) -> Dict[str, Any]:
        """Returns current tracking metrics for UI/Telemetry."""
        with self._lock:
            return {
                "symbol": self.symbol,
                "direction": self.config.direction.value,
                "current_vwap": self.vwap,
                "highest_vwap": self.highest_vwap,
                "lowest_vwap": self.lowest_vwap,
                "highest_price": self.highest_price,
                "lowest_price": self.lowest_price,
                "current_stop_price": self.current_stop_price,
                "is_triggered": self.is_triggered,
                "trigger_reason": self.trigger_reason
            }