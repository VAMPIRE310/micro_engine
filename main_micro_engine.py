"""
main_micro_engine.py — Railway-ready AI position manager
=========================================================

Connection hierarchy
--------------------
Market data (prices/ticks):
  1. PRIMARY   — Rust WebSocket engine  ws://{RUST_WS_URL}/ws/{symbol}
  2. FALLBACK  — Bybit public WebSocket wss://stream.bybit.com/v5/public/linear

Account data (positions / executions):
  - Bybit PRIVATE WebSocket wss://stream.bybit.com/v5/private
    authenticated with RSA key + API key (same credentials as rsa_auth.py)
  - REST fallback via BybitV5Client.get_positions() used for initial sync only

Behaviour
---------
* The AI ONLY manages positions that the user opens manually.
  It does NOT open new trades itself.
* When a position is detected (size > 0), the agent activates.
* Physical close-in-loss blocker:
    - A single leg can NEVER be closed while it is in unrealised loss.
    - A hedged pair (2 legs) can only be resolved when
      combined_pnl = core_unrealised + hedge_unrealised + session_realized > PROFIT_BUFFER
* Session realised PnL memory:
    - When leg(s) close, their realised PnL is added to session_realized_pnl[symbol].
    - This persists for the life of the process so the PnL baseline never resets
      between individual leg closures.

Environment variables
---------------------
  BYBIT_API_KEY             — Bybit API key (required)
  BYBIT_RSA_PRIVATE_KEY     — full PEM content (preferred on Railway)
  BYBIT_RSA_PRIVATE_KEY_PATH — path to .pem file (local alt)
  BYBIT_DEMO                — "true" for paper trading (default false)
  SYMBOL                    — trading pair (default BTCUSDT)
  RUST_WS_URL               — Rust engine base URL (default ws://localhost:8080)
  HEDGE_TRIGGER_LOSS_USDT   — loss threshold to open hedge leg (default -15.0)
  PROFIT_BUFFER_USDT        — min net profit to allow close (default 0.5)
  POLL_INTERVAL_SEC         — REST fallback poll interval (default 3)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from security.rsa_auth import APIConfig, BybitV5Client
from core.agents.micro_agent import MicroAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("MicroEngine")

# ---------------------------------------------------------------------------
# Constants / env-vars
# ---------------------------------------------------------------------------
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
RUST_WS_URL = os.environ.get("RUST_WS_URL", "ws://localhost:8080")
HEDGE_TRIGGER_LOSS = float(os.environ.get("HEDGE_TRIGGER_LOSS_USDT", "-15.0"))
PROFIT_BUFFER = float(os.environ.get("PROFIT_BUFFER_USDT", "0.5"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SEC", "3"))
FEE_RATE = 0.0006   # Bybit taker fee 0.06 % — conservative

BYBIT_PUBLIC_WS = "wss://stream.bybit.com/v5/public/linear"
BYBIT_PRIVATE_WS_LIVE = "wss://stream.bybit.com/v5/private"
BYBIT_PRIVATE_WS_DEMO = "wss://stream-demo.bybit.com/v5/private"

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "lifecycle_lstm.pt")


# ===========================================================================
# RSA WebSocket auth helper
# ===========================================================================

def _ws_auth_payload(config: APIConfig) -> dict:
    """
    Build the Bybit private-WS authentication message.

    Bybit V5 private-WS auth format:
        sign_str = "GET/realtime{expires_ms}"
        expires  = now_ms + 10_000   (10 s grace window)
        signature = RSA-SHA256-sign(sign_str) → base64   OR   HMAC-SHA256-hex
    """
    expires = int((time.time() + 10) * 1000)
    sign_str = f"GET/realtime{expires}"

    if config.api_secret:
        # HMAC path (fallback when user sets api_secret instead of RSA key)
        import hashlib, hmac as _hmac
        sig = _hmac.new(
            config.api_secret.encode(),
            sign_str.encode(),
            hashlib.sha256,
        ).hexdigest()
    else:
        # RSA path (production)
        from Crypto.Hash import SHA256
        from Crypto.Signature import pkcs1_15
        from Crypto.PublicKey import RSA

        key_data: Optional[bytes] = None
        if config.private_key_content:
            raw = config.private_key_content.replace("\\n", "\n")
            key_data = raw.encode()
        elif config.private_key_path and os.path.exists(config.private_key_path):
            with open(config.private_key_path, "rb") as fh:
                key_data = fh.read()

        if key_data is None:
            raise RuntimeError("No RSA private key available for WebSocket auth.")

        private_key = RSA.import_key(key_data)
        h = SHA256.new(sign_str.encode())
        sig = base64.b64encode(pkcs1_15.new(private_key).sign(h)).decode()

    return {"op": "auth", "args": [config.api_key, expires, sig]}


# ===========================================================================
# Market data WebSocket (Rust engine primary / Bybit public fallback)
# ===========================================================================

class MarketDataStream:
    """
    Connects to the Rust engine WS for a single symbol.
    Falls back to Bybit public WS if Rust is unreachable.

    Delivers ticks to self.last_tick = {'price': float, 'volume': float}.
    """

    def __init__(self, symbol: str, rust_base_url: str):
        self.symbol = symbol
        self.rust_url = f"{rust_base_url}/ws/{symbol}"
        self.bybit_public_url = BYBIT_PUBLIC_WS
        self.last_tick: Dict = {"price": 0.0, "volume": 0.0}
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            connected = await self._try_rust()
            if not connected:
                log.warning("[MarketData] Rust WS unavailable — falling back to Bybit public WS")
                await self._run_bybit_public()

    async def _try_rust(self) -> bool:
        """Attempt one connection to the Rust WS. Returns True on clean connect."""
        try:
            import websockets
            async with websockets.connect(
                self.rust_url,
                ping_interval=15,
                ping_timeout=10,
            ) as ws:
                log.info("[MarketData] Connected to Rust engine: %s", self.rust_url)
                # Rust engine streams all data automatically — no subscribe needed.
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        self._ingest(msg)
                    except Exception:
                        pass
        except Exception as exc:
            log.debug("[MarketData] Rust WS connect failed: %s", exc)
            return False
        return True

    async def _run_bybit_public(self) -> None:
        """Subscribe to tickers.{symbol} on Bybit public WS (fallback)."""
        try:
            import websockets
            async with websockets.connect(
                self.bybit_public_url,
                ping_interval=15,
                ping_timeout=10,
            ) as ws:
                sub = {"op": "subscribe", "args": [f"tickers.{self.symbol}"]}
                await ws.send(json.dumps(sub))
                log.info("[MarketData] Subscribed to Bybit public WS: tickers.%s", self.symbol)

                ping_task = asyncio.create_task(self._heartbeat(ws))
                try:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            if msg.get("topic", "").startswith("tickers."):
                                d = msg.get("data", {})
                                price = float(d.get("lastPrice", 0) or 0)
                                vol = float(d.get("volume24h", 0) or 0)
                                if price > 0:
                                    self.last_tick = {"price": price, "volume": vol}
                        except Exception:
                            pass
                finally:
                    ping_task.cancel()
        except Exception as exc:
            log.error("[MarketData] Bybit public WS error: %s — retrying in 5s", exc)
            await asyncio.sleep(5)

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(20)
            try:
                await ws.send(json.dumps({"op": "ping"}))
            except Exception:
                break

    def _ingest(self, msg: dict) -> None:
        """Parse messages from the Rust engine format."""
        msg_type = msg.get("type", "")
        if msg_type == "tick":
            price = float(msg.get("price", 0) or 0)
            vol = float(msg.get("size", 0) or 0)
            if price > 0:
                self.last_tick = {"price": price, "volume": vol}
        elif msg_type == "trade":
            price = float(msg.get("price", 0) or 0)
            vol = float(msg.get("size", 0) or 0)
            if price > 0:
                self.last_tick = {"price": price, "volume": vol}

    def stop(self) -> None:
        self._running = False


# ===========================================================================
# Bybit private WebSocket (position + execution stream)
# ===========================================================================

class PrivateStream:
    """
    Maintains a single authenticated WebSocket to Bybit private stream.
    Delivers position and execution updates via callback.

    Callbacks:
        on_position(data: list)   — list of position dicts from Bybit
        on_execution(data: list)  — list of execution dicts from Bybit
    """

    def __init__(self, config: APIConfig):
        self._config = config
        self._url = BYBIT_PRIVATE_WS_DEMO if config.demo else BYBIT_PRIVATE_WS_LIVE
        self._running = False
        self.on_position = None     # set by engine
        self.on_execution = None

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        while self._running:
            try:
                import websockets
                async with websockets.connect(
                    self._url,
                    ping_interval=None,   # we send our own heartbeat
                ) as ws:
                    # Authenticate
                    auth_msg = _ws_auth_payload(self._config)
                    await ws.send(json.dumps(auth_msg))

                    # Wait for auth confirmation
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    resp = json.loads(raw)
                    if not resp.get("success"):
                        log.error("[PrivateWS] Auth failed: %s", resp)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue

                    log.info("[PrivateWS] Authenticated. Subscribing to position + execution.")

                    # Subscribe
                    sub = {
                        "op": "subscribe",
                        "args": ["position", "execution", "order"],
                    }
                    await ws.send(json.dumps(sub))

                    backoff = 1.0   # reset on successful connect

                    heartbeat = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for raw in ws:
                            self._dispatch(raw)
                    finally:
                        heartbeat.cancel()

            except Exception as exc:
                log.error("[PrivateWS] Disconnected: %s — reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(20)
            try:
                await ws.send(json.dumps({"op": "ping"}))
            except Exception:
                break

    def _dispatch(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return

        topic = msg.get("topic", "")
        data = msg.get("data", [])

        if topic == "position" and self.on_position and data:
            self.on_position(data)
        elif topic == "execution" and self.on_execution and data:
            self.on_execution(data)

    def stop(self) -> None:
        self._running = False


# ===========================================================================
# Main execution engine
# ===========================================================================

class MicroExecutionEngine:
    """
    AI position manager.

    Rules
    -----
    * Never opens trades independently — only manages what the user opens.
    * Single leg in loss  → BLOCKED from closing. Opens hedge if beyond HEDGE_TRIGGER.
    * Hedged (2 legs):
        - Only resolves when combined_pnl > PROFIT_BUFFER (after fees).
        - combined_pnl = core_unrealised + hedge_unrealised + session_realized[symbol]
    * When a leg closes, its realised PnL is added to session_realized[symbol] so the
      PnL baseline carries over across leg-by-leg closures.
    """

    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol

        # Bybit REST client (initial position sync + order placement)
        api_config = APIConfig()
        self.client = BybitV5Client(config=api_config)
        self._api_config = api_config

        # AI agent
        self.agent = MicroAgent(device="cpu")
        self.agent.load_weights(WEIGHTS_PATH)

        # Market data stream (Rust primary / Bybit public fallback)
        self.market = MarketDataStream(symbol, RUST_WS_URL)

        # Private stream (positions + executions)
        self.private_ws = PrivateStream(api_config)
        self.private_ws.on_position = self._on_position_update
        self.private_ws.on_execution = self._on_execution_update

        # ── State ────────────────────────────────────────────────────────────
        # Positions currently under management, keyed by positionIdx (0 or 1)
        self._positions: Dict[int, dict] = {}

        # Cumulative realised PnL memory — NEVER resets during this process
        # lifetime.  Keyed by symbol so multi-symbol support is trivial later.
        self._session_realized: Dict[str, float] = defaultdict(float)

        # Agent active flag (set when ≥1 position detected)
        self._agent_active = False

        log.info("MicroExecutionEngine armed for %s.", symbol)

    # ── Private WS callbacks ─────────────────────────────────────────────────

    def _on_position_update(self, data: list) -> None:
        """
        Called on every 'position' topic push from Bybit private WS.
        Detects new positions (user opened) and closed ones.
        """
        for pos in data:
            if pos.get("symbol") != self.symbol:
                continue

            idx = int(pos.get("positionIdx", 0))
            size = float(pos.get("size", 0) or 0)

            if size > 0:
                prev = self._positions.get(idx)
                self._positions[idx] = pos
                if prev is None:
                    log.info(
                        "[PositionDetect] New position idx=%d side=%s size=%s entry=%.4f",
                        idx,
                        pos.get("side"),
                        pos.get("size"),
                        float(pos.get("avgPrice", 0) or 0),
                    )
            else:
                # size == 0 → position closed
                if idx in self._positions:
                    closed = self._positions.pop(idx)
                    realised = float(closed.get("cumRealisedPnl", 0) or 0)
                    self._session_realized[self.symbol] += realised
                    log.info(
                        "[PositionClosed] idx=%d realised=%.4f session_total=%.4f",
                        idx,
                        realised,
                        self._session_realized[self.symbol],
                    )
                    self.agent.reset_sequence()

            # Update active flag
            self._agent_active = len(self._positions) > 0

    def _on_execution_update(self, data: list) -> None:
        """
        Track filled executions to update cumulative realised PnL in real-time.
        Bybit sends executions BEFORE the position update on reduce/close orders,
        so we capture realised PnL here for completeness.
        """
        for ex in data:
            if ex.get("symbol") != self.symbol:
                continue
            exec_type = ex.get("execType", "")
            if exec_type in ("Trade",):
                closed_size = float(ex.get("closedSize", 0) or 0)
                if closed_size > 0:
                    realised = float(ex.get("closedPnl", 0) or 0)
                    if realised != 0:
                        # Accumulate; will be reconciled on position-close callback
                        self._session_realized[self.symbol] += realised
                        log.debug(
                            "[Execution] closedSize=%.4f pnl=%.4f session_total=%.4f",
                            closed_size,
                            realised,
                            self._session_realized[self.symbol],
                        )

    # ── Order helpers ────────────────────────────────────────────────────────

    def _estimate_fees(self, pos: dict) -> float:
        size = float(pos.get("size", 0) or 0)
        entry = float(pos.get("avgPrice", 0) or 0)
        return size * entry * FEE_RATE

    def _execute_order(
        self, side: str, qty: float, reduce_only: bool = False, position_idx: int = 0
    ) -> bool:
        try:
            res = self.client.place_order(
                symbol=self.symbol,
                side=side,
                order_type="Market",
                qty=str(qty),
                category="linear",
                time_in_force="IOC",
                reduce_only=reduce_only,
                position_idx=position_idx,
            )
            log.info("ORDER EXECUTED: %s %.4f reduce_only=%s | id=%s", side, qty, reduce_only, res)
            return True
        except Exception as exc:
            log.error("Order execution failed: %s", exc)
            return False

    # ── Core management logic ────────────────────────────────────────────────

    def _combined_pnl(self) -> float:
        """
        Total P&L including all open legs + realised memory.
        """
        unrealised = sum(float(p.get("unrealisedPnl", 0) or 0) for p in self._positions.values())
        return unrealised + self._session_realized[self.symbol]

    def _manage_hedged(self) -> None:
        """Resolve a 2-leg hedge when combined PnL is positive enough."""
        positions = list(self._positions.values())
        if len(positions) < 2:
            return

        fees = sum(self._estimate_fees(p) for p in positions)
        combined = self._combined_pnl() - fees

        if combined > PROFIT_BUFFER:
            log.info(
                "[RESOLVE] Combined PnL=%.4f USDT (after fees=%.4f). Unwinding both legs.",
                combined,
                fees,
            )
            for pos in positions:
                close_side = "Sell" if pos["side"] == "Buy" else "Buy"
                idx = int(pos.get("positionIdx", 0))
                self._execute_order(close_side, float(pos["size"]), reduce_only=True, position_idx=idx)

            # Self-reflection reward (use a real-data state if possible)
            if self._positions:
                any_pos = next(iter(self._positions.values()))
                tick = self.market.last_tick
                state = self._build_state_tensor(any_pos, tick)
            else:
                state = torch.zeros(160)
            self.agent.self_reflect(state, "EXIT", combined)
        else:
            log.debug("[HEDGED] Combined PnL=%.4f — waiting.", combined)

    def _build_state_tensor(self, pos: dict, market_data: dict) -> torch.Tensor:
        """
        Build a 160-dim feature vector from available position + market data.

        This is a minimal stand-in until the full feature engine is wired up.
        The vector is deterministic (no random) so the LSTM hidden state evolves
        consistently across ticks.  Swap this method's body for a real feature
        call when ready.
        """
        price = float(market_data.get("price", 0) or 0)
        volume = float(market_data.get("volume", 0) or 0)
        entry = float(pos.get("avgPrice", 0) or 0)
        size = float(pos.get("size", 0) or 0)
        upnl = float(pos.get("unrealisedPnl", 0) or 0)
        session = self._session_realized[self.symbol]

        # Normalised scalars packed into a 160-dim vector.
        # First 6 dims = real data; rest = 0 until feature engine is integrated.
        raw = [
            price / max(entry, 1e-9) - 1.0,          # price-vs-entry ratio
            upnl / max(size * entry, 1e-9),            # pnl as fraction of notional
            session / max(size * entry, 1e-9),         # session pnl fraction
            volume / 1e6,                              # rough volume normalisation
            float(size),
            1.0 if pos.get("side") == "Buy" else -1.0,
        ] + [0.0] * 154

        return torch.tensor(raw, dtype=torch.float32)

    def _manage_single(self, pos: dict) -> None:
        """Manage a single open position."""
        pnl = float(pos.get("unrealisedPnl", 0) or 0)
        side = pos.get("side", "Buy")
        size = float(pos.get("size", 0) or 0)
        idx = int(pos.get("positionIdx", 0))
        fees = self._estimate_fees(pos)
        session_pnl = self._session_realized[self.symbol]

        # ── Physical blocker: cannot close in loss ────────────────────────
        if pnl < 0:
            if pnl <= HEDGE_TRIGGER_LOSS:
                log.warning(
                    "[DEFENSE] PnL=%.4f <= trigger=%.4f — opening delta hedge.",
                    pnl,
                    HEDGE_TRIGGER_LOSS,
                )
                hedge_side = "Sell" if side == "Buy" else "Buy"
                self._execute_order(hedge_side, size)
            else:
                log.debug("[LOCKED] PnL=%.4f — close blocked (in loss).", pnl)
            return

        # ── Profitable: ask AI whether to close ──────────────────────────
        # Only allow close if current PnL + session memory covers fees
        net_pnl = pnl + session_pnl - fees * 2
        if net_pnl <= PROFIT_BUFFER:
            log.debug("[WAIT] net_pnl=%.4f below buffer=%.4f.", net_pnl, PROFIT_BUFFER)
            return

        tick = self.market.last_tick
        market_data = {"price": tick["price"], "volume": tick["volume"]}

        # Guard: if we have no live price yet, do not let the agent act —
        # running on a zero-price tick would produce meaningless decisions.
        if market_data["price"] <= 0:
            log.debug("[WAIT] No live price available yet — skipping AI decision.")
            return

        # Build a minimal deterministic feature vector from what we DO have.
        # Replace this block with a proper feature-engine call once integrated.
        state_tensor = self._build_state_tensor(pos, market_data)

        action = self.agent.predict(state_tensor, market_data)

        if action == "EXIT":
            log.info("[PROFIT] AI signals EXIT. PnL=%.4f net=%.4f", pnl, net_pnl)
            close_side = "Sell" if side == "Buy" else "Buy"
            ok = self._execute_order(close_side, size, reduce_only=True, position_idx=idx)
            if ok:
                self.agent.self_reflect(state_tensor, "EXIT", net_pnl)
        elif action == "HEDGE":
            log.info("[WAVE-HEDGE] AI signals HEDGE. PnL=%.4f", pnl)
            hedge_side = "Sell" if side == "Buy" else "Buy"
            self._execute_order(hedge_side, size)
        else:
            log.debug("[HOLD] AI=%s  PnL=%.4f  session=%.4f", action, pnl, session_pnl)

    async def _management_loop(self) -> None:
        """
        Main management cycle — runs every POLL_INTERVAL seconds.

        The private WS provides near-real-time position updates, so this loop
        only acts on already-known state; it does NOT poll REST every iteration.
        We do a single REST sync on startup then rely on WS updates.
        """
        # Initial REST sync to pick up any positions already open at startup
        await self._initial_rest_sync()

        while True:
            try:
                if self._agent_active:
                    n = len(self._positions)
                    if n >= 2:
                        self._manage_hedged()
                    elif n == 1:
                        pos = next(iter(self._positions.values()))
                        self._manage_single(pos)
            except Exception as exc:
                log.error("[Loop] Unhandled error: %s", exc)

            await asyncio.sleep(POLL_INTERVAL)

    async def _initial_rest_sync(self) -> None:
        """Fetch current positions via REST on startup."""
        try:
            positions = self.client.get_positions(category="linear", symbol=self.symbol)
            for sym, pos in positions.items():
                if pos.size > 0:
                    raw = {
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "size": str(pos.size),
                        "avgPrice": str(pos.entry_price),
                        "unrealisedPnl": str(pos.unrealized_pnl),
                        "cumRealisedPnl": str(pos.realized_pnl),
                        "positionIdx": str(pos.position_idx),
                    }
                    idx = pos.position_idx
                    self._positions[idx] = raw
                    log.info("[Startup] Found existing position idx=%d side=%s size=%s", idx, pos.side, pos.size)
            self._agent_active = len(self._positions) > 0
        except Exception as exc:
            log.error("[Startup] REST sync failed: %s — relying on WS updates.", exc)

    # ── Entry point ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        log.info("Starting MicroExecutionEngine — symbol=%s", self.symbol)
        log.info("Rust WS: %s  |  Demo: %s", RUST_WS_URL, self._api_config.demo)

        await asyncio.gather(
            self.market.run(),
            self.private_ws.run(),
            self._management_loop(),
        )


# ===========================================================================
# Entrypoint
# ===========================================================================

if __name__ == "__main__":
    engine = MicroExecutionEngine(symbol=SYMBOL)
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        log.info("Engine stopped.")
