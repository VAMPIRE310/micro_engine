"""
NEO SUPREME 2026 — Institutional Ensemble Decision Engine
Hierarchical Pipeline: Aggregates specialized AI signals into a single Rust-compatible decision payload.
"""
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("ensemble")

class InstitutionalEnsemble:
    """
    Hierarchical pipeline that processes AI decisions.
    Instead of democratic voting, it uses strict role delegation:
    - Strike = Direction & Base Confidence
    - Negotiator = Sizing 
    - Whale Tracker = Confidence Multiplier
    - Rust (Downstream) = Risk Management & Execution
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.65)
        self.decision_history = []

    def decide(self, strike_decision: Dict[str, Any], negotiator_sizing: Dict[str, Any],
               whale_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synthesize specialized AI outputs into the final Rust command.
        
        Args:
            strike_decision: Dict containing 'action' (LONG/SHORT/HOLD) and 'confidence'.
            negotiator_sizing: Dict containing 'margin_scale' (0.0 - 1.0).
            whale_data: Dict containing 'order_flow_imbalance'.
            
        Returns:
            A dictionary perfectly formatted for the Rust portfolio_state.rs listener.
        """
        action_str = strike_decision.get("action", "HOLD").upper()
        base_confidence = strike_decision.get("confidence", 0.0)
        
        # 1. Base filter: If Strike says HOLD or confidence is garbage, abort early.
        if action_str == "HOLD" or base_confidence < 0.4:
            return self._format_rust_payload("NEUTRAL", 0, 0.0, base_confidence, "Low confidence or HOLD")

        # 2. Apply Smart Money (Whale) Momentum Modifiers
        whale_imbalance = whale_data.get("order_flow_imbalance", 0.0) if whale_data else 0.0
        final_confidence = base_confidence
        
        if action_str == "LONG" and whale_imbalance > 0.2:
            final_confidence += 0.1  # Smart money supports the pump
            logger.info("🐋 Whale flow aligns with LONG. Confidence boosted.")
        elif action_str == "SHORT" and whale_imbalance < -0.2:
            final_confidence += 0.1  # Smart money supports the dump
            logger.info("🐋 Whale flow aligns with SHORT. Confidence boosted.")
        elif (action_str == "LONG" and whale_imbalance < -0.3) or (action_str == "SHORT" and whale_imbalance > 0.3):
            final_confidence -= 0.2  # Fighting smart money is dangerous
            logger.warning("🚨 Whale flow contradicts AI direction. Confidence slashed.")

        # Clamp confidence
        final_confidence = max(0.0, min(1.0, final_confidence))

        # 3. Final Confidence Check against strict threshold
        if final_confidence < self.min_confidence:
            return self._format_rust_payload("NEUTRAL", 0, 0.0, final_confidence, "Vetoed by Whale Flow / Threshold")

        # 4. Extract sizing from the Negotiator
        sizing_fraction = negotiator_sizing.get("margin_scale", 0.01)

        # 5. Determine Action Code for Rust
        # Action Codes in Rust: 1/3 = Entry/Wave, 2 = Close, 4 = Hedge, 5/6 = Wave Management
        # Since this engine runs the entry pipeline, we output code 1 (Core Entry). 
        # (If a position already exists, Rust's portfolio_state will automatically convert it to a wave/hedge)
        action_code = 1 

        payload = self._format_rust_payload(
            bias=action_str,
            action_code=action_code,
            sizing=sizing_fraction,
            confidence=final_confidence,
            reason="ensemble_approved"
        )
        
        self.decision_history.append(payload)
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]

        return payload

    def _format_rust_payload(self, bias: str, action_code: int, sizing: float, 
                             confidence: float, reason: str) -> Dict[str, Any]:
        """Formats the exact JSON structure expected by Rust's brain:decision channel."""
        return {
            "directional_bias": bias,
            "action": action_code,
            "position_size": round(sizing, 4),
            "confidence_score": round(confidence, 4),
            "reason": reason,
            "timestamp_ms": int(time.time() * 1000)
        }