"""
NEO SUPREME 2026 — Ensemble Helper Utilities
Bridging functions for SAC continuous output, divergence detection, etc.
"""
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("ensemble_helpers")


class ParanoiaConfig:
    """
    Configuration for ensemble paranoia level.
    More paranoid = higher thresholds, more conservative.
    """
    def __init__(self, level: str = "normal"):
        self.level = level
        configs = {
            "aggressive": {
                "min_confidence": 0.30,
                "min_consensus": 0.25,
                "max_drawdown": 0.15,
            },
            "normal": {
                "min_confidence": 0.45,
                "min_consensus": 0.35,
                "max_drawdown": 0.10,
            },
            "conservative": {
                "min_confidence": 0.60,
                "min_consensus": 0.50,
                "max_drawdown": 0.05,
            },
        }
        self.config = configs.get(level, configs["normal"])

    @property
    def min_confidence(self) -> float:
        return self.config["min_confidence"]

    @property
    def min_consensus(self) -> float:
        return self.config["min_consensus"]

    @property
    def max_drawdown(self) -> float:
        return self.config["max_drawdown"]


class ParanoiaEnsemble:
    """
    Paranoia-adjusted ensemble wrapper.
    Dynamically adjusts thresholds based on recent performance.
    """
    def __init__(self, config: ParanoiaConfig = None):
        self.config = config or ParanoiaConfig()
        self.recent_errors = 0
        self.recent_trades = 0

    def adjust(self, win_rate: float):
        """Adjust paranoia based on recent win rate."""
        if win_rate < 0.4:
            self.config = ParanoiaConfig("conservative")
            logger.info("[Paranoia] Switching to CONSERVATIVE (low win rate)")
        elif win_rate > 0.65:
            self.config = ParanoiaConfig("aggressive")
            logger.info("[Paranoia] Switching to AGGRESSIVE (high win rate)")
        else:
            self.config = ParanoiaConfig("normal")


def extract_sac_directional_scalar(sac_actor, state_tensor, regime: int = 0) -> float:
    """
    Bridge SAC continuous action to a discrete directional scalar.
    Returns a value in [-1, 1] where positive = bullish, negative = bearish.
    """
    try:
        import torch
        with torch.no_grad():
            if hasattr(sac_actor, "sample"):
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                regime_t = torch.tensor([regime], dtype=torch.long, device=state_tensor.device)
                action, _ = sac_actor.sample(state_tensor, regime_t, deterministic=True)
                # Use first dimension as directional signal
                scalar = float(action[0, 0].cpu().item())
                return np.clip(scalar, -1.0, 1.0)
            else:
                # Fallback: forward pass
                mean, _ = sac_actor(state_tensor)
                scalar = float(mean[0, 0].cpu().item())
                return np.clip(scalar, -1.0, 1.0)
    except Exception as e:
        logger.debug(f"[SAC Bridge] Error: {e}")
        return 0.0


def detect_smart_money_divergence(price_history: list, smfi_history: list,
                                  lookback: int = 10) -> Optional[str]:
    """
    Detect bullish or bearish divergence between price and Smart Money Flow Index.

    Returns:
        'bullish_divergence', 'bearish_divergence', or None
    """
    if len(price_history) < lookback or len(smfi_history) < lookback:
        return None

    recent_prices = price_history[-lookback:]
    recent_smfi = smfi_history[-lookback:]

    price_trend = recent_prices[-1] - recent_prices[0]
    smfi_trend = recent_smfi[-1] - recent_smfi[0]

    # Bullish: price making lower lows but SMFI making higher lows
    if price_trend < 0 and smfi_trend > 5:
        return "bullish_divergence"

    # Bearish: price making higher highs but SMFI making lower highs
    if price_trend > 0 and smfi_trend < -5:
        return "bearish_divergence"

    return None

