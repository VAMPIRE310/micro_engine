"""
NEO SUPREME 2026 — Exodia Execution Orchestrator
Listens for Rust tensors, runs sub-millisecond AI inference,
applies Campaign Reward Shaping, and routes Wave/Hedge decisions
to Rust for dynamic S/R trailing.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import redis
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.agents.strike_agent import StrikeAgent
from core.agents.position_manager_agent import PositionManagerAgent
from core.agents.negotiator_agent import NegotiatorAgent
from core.agents.forecaster_agent import ForecasterAgent
from core.inference.execution_ensemble import InstitutionalEnsemble

# Custom engines
try:
    from core.Advanced_orders.position_scaling_engine import PositionScalingEngine
    from core.Advanced_orders.correlation_hedge_engine import CorrelationHedgeEngine
except ImportError:
    PositionScalingEngine = CorrelationHedgeEngine = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("exodia_orchestrator")


class ExodiaOrchestrator:
    def __init__(self, redis_url: str):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ensemble = InstitutionalEnsemble()

        logger.info("Initializing Exodia Agents & Custom Engines...")
        self.strike = StrikeAgent(redis_client=self.redis)
        self.negotiator = NegotiatorAgent(
            device="cuda" if os.environ.get("RAILWAY_ENVIRONMENT") is None else "cpu"
        )
        self.position_manager = PositionManagerAgent(redis_client=self.redis)
        self.forecaster = ForecasterAgent(redis_client=self.redis)

        self.position_scaler = PositionScalingEngine() if PositionScalingEngine else None
        self.correlation_hedge = CorrelationHedgeEngine() if CorrelationHedgeEngine else None

        # State for pending trailing (minimal — Rust owns the hot path)
        self.pending_trailing_entries = {}
        self._last_wave_add_ts = {}
        self.WAVE_ADD_COOLDOWN_SECONDS = 15
        self.WAVE_SIZE_FACTOR = 0.5

        # Load fine-tuned models
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            "merged_models"
        )
        for agent, filename in [
            (self.strike, "strike_mlp.pt"),
            (self.negotiator, "negotiator_sac.pt"),
            (self.position_manager, "lifecycle_lstm.pt"),
            (self.forecaster, "forecaster_tcn.pt")
        ]:
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                try:
                    if hasattr(agent, 'load'):
                        agent.load(path)
                    else:
                        agent.load_model(path)
                    logger.info(f"Loaded model: {filename}")
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")

    def get_portfolio_state(self, symbol: str) -> dict:
        """Fetches Rust-generated portfolio snapshot with full error resilience."""
        snap = self.redis.get("neo:portfolio_snapshot")
        if not snap:
            return {}
        try:
            data = json.loads(snap)
            return data.get("symbols", {}).get(symbol, {})
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"Failed to decode Rust portfolio snapshot for {symbol}: {e}")
            return {}
        except Exception as e:
            logger.exception(f"Unexpected error reading portfolio state for {symbol}")
            return {}

    def calculate_campaign_reward(self, port_state: dict, is_terminal: bool = False) -> float:
        """Enhanced reward with tunable weights via Redis."""
        w_base = float(self.redis.hget("reward_weights", "base") or 1.25)
        w_sr = float(self.redis.hget("reward_weights", "sr_bonus") or 15.0)
        w_trail = float(self.redis.hget("reward_weights", "trail_bonus") or 8.0)
        w_fee = float(self.redis.hget("reward_weights", "fee_mult") or 2.5)
        w_dd = float(self.redis.hget("reward_weights", "dd_penalty") or 12.0)

        combined_pnl = port_state.get("net_combined_pnl", 0.0)
        has_active_hedge = bool(port_state.get("active_hedges_count", 0))
        max_dd_pct = abs(port_state.get("max_drawdown_pct", 0.0))
        fees = port_state.get("cumulative_fees", 0.0)

        entry_ts = port_state.get("core_entry_ts", time.time())
        if entry_ts > 1e11:  # ms to seconds
            entry_ts /= 1000.0
        time_held_hours = max(0.01, (time.time() - entry_ts) / 3600.0)

        base = (time_held_hours ** 0.65) * w_base

        if combined_pnl > 0:
            reward = base * 3.2 if not has_active_hedge else base * 1.85
            if has_active_hedge or port_state.get("wave_adds", 0) > 0:
                reward *= 0.65
        else:
            reward = -abs(base) * 1.45

        reward -= (max_dd_pct * 100) ** 2.0 * 0.8
        reward -= fees * w_fee

        if port_state.get("hit_sr_target", False):
            reward += w_sr if not has_active_hedge else w_sr * 0.55

        if port_state.get("used_vol_trailing_breakout", False) or port_state.get("is_trailing_breakout", False):
            reward += w_trail

        if is_terminal and combined_pnl > 0.0:
            reward += combined_pnl * 0.75

        return float(reward)

    def _get_dynamic_trail_dist(self, symbol: str, current_price: float, port_state: dict) -> float:
        """Dynamic trail using ATR + volatility regime from Rust."""
        if current_price <= 0:
            return current_price * 0.0035

        atr_value = port_state.get("atr_value") or port_state.get("atr_14", current_price * 0.012)
        vol_regime = port_state.get("vol_regime", "normal")
        near_sr = port_state.get("near_sr_strength", 0.0)

        multiplier = {
            "breakout": 2.4,
            "high": 1.9,
            "low": 1.1,
            "normal": 1.6
        }.get(vol_regime, 1.6)

        dynamic_dist = atr_value * multiplier

        if near_sr > 0.7:
            dynamic_dist *= 0.65

        min_dist = current_price * (0.006 if "BTC" in symbol.upper() else 0.0035)
        return max(dynamic_dist, min_dist)

    def _should_allow_close(self, symbol: str, action_idx: int, port_state: dict, current_price: float) -> bool:
        """Core Guard: Block closing when combined PnL is negative (unless hedge salvage)."""
        combined_pnl = port_state.get("net_combined_pnl", 0.0)
        core_pnl = port_state.get("core_pnl", 0.0)
        has_hedge = port_state.get("active_hedges_count", 0) > 0

        if combined_pnl < 0 and action_idx in (2, 5, 6):
            if action_idx in (5, 6) and has_hedge:
                estimated_fee = (port_state.get("core_size", 0.0) * current_price) * 0.001
                if core_pnl - estimated_fee > 0:
                    logger.info(f"🛡️ [GUARD BYPASS] {symbol} closing losing hedge while core green")
                    return True
            logger.info(f"🛡️ [COMBINED PNL GUARD] {symbol} combined PnL {combined_pnl:.2f} < 0 → blocking action {action_idx}")
            return False
        return True

    def _dispatch_to_rust(self, symbol: str, action_idx: int, size: float, bias: str, kwargs: dict = None):
        payload = {
            "directional_bias": bias.upper(),
            "action": action_idx,
            "position_size": round(size, 4),
            "confidence_score": 1.0,
            "reason": f"advanced_routing_{action_idx}",
            "timestamp_ms": int(time.time() * 1000)
        }
        if kwargs:
            payload.update(kwargs)
        self.redis.xadd("brain:decisions:stream", {"symbol": symbol, "payload": json.dumps(payload)})

    def _handle_advanced_routing(self, symbol: str, action_idx: int, port_state: dict, current_price: float):
        core_size = port_state.get("core_size", 0.0)
        core_side = port_state.get("core_side", "BUY").capitalize()
        near_sr = port_state.get("near_sr", False)
        dynamic_trail_dist = self._get_dynamic_trail_dist(symbol, current_price, port_state)

        if action_idx == 3:  # WAVE ADD
            now = time.time()
            if now - self._last_wave_add_ts.get(symbol, 0.0) < self.WAVE_ADD_COOLDOWN_SECONDS:
                return

            wave_qty = core_size * self.WAVE_SIZE_FACTOR
            if self.position_scaler:
                try:
                    scaled = self.position_scaler.calculate_scale_in(
                        symbol=symbol,
                        current_size=core_size,
                        current_price=current_price,
                        entry_price=port_state.get("core_entry_price", 0.0),
                        unrealized_pnl=port_state.get("core_pnl", 0.0)
                    )
                    if scaled and scaled > 0:
                        wave_qty = scaled
                except Exception as e:
                    logger.debug(f"Scale error: {e}")

            if near_sr:
                wave_key = f"{symbol}_wave_add"
                self.pending_trailing_entries[wave_key] = {
                    "symbol": symbol,
                    "side": core_side,
                    "qty": wave_qty,
                    "action_idx": 3,
                    "extremum": current_price,
                    "trail_dist": dynamic_trail_dist,
                    "dynamic": True
                }
                self._last_wave_add_ts[symbol] = now
                logger.info(f"🌊 [WAVE-ADD] {symbol} near S/R — queued dynamic trail {dynamic_trail_dist:.4f}")
            else:
                self._dispatch_to_rust(symbol, 3, wave_qty, core_side)
                self._last_wave_add_ts[symbol] = now

        elif action_idx == 4:  # HEDGE
            hedge_qty = core_size * self.WAVE_SIZE_FACTOR
            if self.correlation_hedge:
                try:
                    ratio = self.correlation_hedge.get_hedge_ratio(symbol)
                    if ratio and 0.1 <= ratio <= 2.0:
                        hedge_qty = core_size * ratio
                except Exception:
                    pass

            hedge_side = "Sell" if core_side.lower() in ("buy", "long") else "Buy"

            if near_sr:
                hedge_key = f"{symbol}_hedge"
                if hedge_key not in self.pending_trailing_entries:
                    self.pending_trailing_entries[hedge_key] = {
                        "symbol": symbol,
                        "side": hedge_side,
                        "qty": hedge_qty,
                        "action_idx": 4,
                        "extremum": current_price,
                        "trail_dist": dynamic_trail_dist * 0.9,
                        "dynamic": True
                    }
                    logger.info(f"🛡️ [HEDGE] {symbol} near S/R — queued dynamic trail")
            else:
                self._dispatch_to_rust(symbol, 4, hedge_qty, hedge_side, kwargs={"check_delta_neutral": True})

        elif action_idx in [5, 6]:
            self._dispatch_to_rust(symbol, action_idx, 0.0, core_side, kwargs={"apply_safety_sl_pct": 0.02})

    def _check_trailing_entries(self, symbol: str, current_price: float):
        """Light Python fallback — main trailing logic moved to Rust."""
        for key, entry in list(self.pending_trailing_entries.items()):
            if entry.get("symbol") != symbol:
                continue
            # ... keep your existing logic as safety net if needed ...
            pass

    def process_tensor(self, symbol: str, state_vector: list, current_price: float):
        if self.redis.exists(f"pump_dump:active:{symbol}"):
            return

        self._check_trailing_entries(symbol, current_price)

        state_np = np.array(state_vector, dtype=np.float32)
        market_data = {"symbol": symbol, "price": current_price}

        self.forecaster.predict(state_np, market_data)
        port_state = self.get_portfolio_state(symbol)

        has_core = port_state.get("has_core", False)

        if has_core:
            pm_decision = self.position_manager.predict(state_np, market_data)
            action_idx = pm_decision.get("action_idx", 0)

            custom_reward = self.calculate_campaign_reward(port_state)
            is_done = port_state.get("is_campaign_done", False)

            self.position_manager.store(
                prev_state=state_np,
                action_idx=action_idx,
                reward=custom_reward,
                state=state_np,
                done=is_done
            )

            if action_idx in [2, 5, 6]:
                if not self._should_allow_close(symbol, action_idx, port_state, current_price):
                    action_idx = 0

            if action_idx == 2:
                self._dispatch_to_rust(symbol, 2, 0.0, port_state.get("core_side", "NEUTRAL"))
            elif action_idx in [3, 4, 5, 6]:
                self._handle_advanced_routing(symbol, action_idx, port_state, current_price)

        else:
            # Entry logic (unchanged)
            strike_dec = self.strike.predict(state_np, market_data)
            sizing_dec = self.negotiator.predict(state_np, market_data)
            self.strike.store(state=state_np, action_idx=strike_dec.get("action_idx", 0),
                              reward=0.0, next_state=state_np, done=False)

            whale_raw = self.redis.get(f"whale_flow:{symbol}")
            whale_data = json.loads(whale_raw) if whale_raw else {}

            final_decision = self.ensemble.decide(strike_dec, sizing_dec, whale_data)

            if final_decision.get("action") == 1:
                self.redis.xadd("brain:decisions:stream", {"symbol": symbol, "payload": json.dumps(final_decision)})
                logger.info(f"[{symbol}] Entry Payload Sent: {final_decision.get('directional_bias')}")

    def run(self):
        logger.info("🔥 Exodia Orchestrator Online. Listening to Redis Streams on mtf:stream")
        last_id = "$"

        while True:
            try:
                streams = self.redis.xread({"mtf:stream": last_id}, count=10, block=0)
                for stream_name, messages in streams:
                    for message_id, message_data in messages:
                        last_id = message_id
                        try:
                            raw_payload = message_data.get("payload") or message_data.get(b"payload")
                            if raw_payload:
                                data = json.loads(raw_payload)
                                symbol = data.get("symbol")
                                tensor = data.get("tensor", [])
                                price = data.get("close", 0.0)
                                if symbol and len(tensor) == 160:
                                    self.process_tensor(symbol, tensor, price)
                        except json.JSONDecodeError:
                            logger.error("Malformed tensor JSON from Stream")
                        except Exception as e:
                            logger.error(f"Tensor decoding error: {e}")
            except Exception as e:
                logger.error(f"Stream read error: {e}")
                time.sleep(0.1)


if __name__ == "__main__":
    redis_url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379")
    orchestrator = ExodiaOrchestrator(redis_url)
    orchestrator.run()