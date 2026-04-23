"""
main_micro_engine.py — Railway-ready AI position manager
=========================================================

Connection hierarchy
--------------------
Market data (prices/ticks):
  1. PRIMARY   — Rust WebSocket engine  ws://{RUST_WS_URL}/ws/{symbol}
  2. FALLBACK  — Bybit public WebSocket wss://stream.bybit.com/v5/public/linear
  Market data stream is started ONLY when a live position is detected and
  stopped when all positions for that symbol close. Ticks are throttled to
  one update per 100 ms inside _ingest().

Account / order data (fully live — NO polling):
  - Bybit PRIVATE WebSocket wss://stream.bybit.com/v5/private
    Topics: position, execution, order, wallet
    Authenticated with RSA key + API key.
  - REST is used ONCE at startup for initial position sync only.

Trade execution (fully live — NO REST for orders):
  - Bybit TRADE WebSocket wss://stream.bybit.com/v5/trade
    RSA-signed per-order requests.  REST fallback only if WS is not yet
    connected.

Management loop:
  - Driven entirely by the asyncio tick_event that fires on every accepted
    tick (after 100 ms throttle).  No sleep-based polling.

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
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
import torch

from security.rsa_auth import APIConfig, BybitV5Client
from core.agents.micro_agent import MicroAgent
from core.feature_engine_v2 import FeatureEngineV2
from core.hybrid_volume_trailing import HybridVolumeTrailingStop, HybridStopConfig, TrailingDirection
from core.token_profiler import TokenProfiler

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
FEE_RATE = 0.0006   # Bybit taker fee 0.06 % — conservative

BYBIT_PUBLIC_WS       = "wss://stream.bybit.com/v5/public/linear"
BYBIT_PRIVATE_WS_LIVE = "wss://stream.bybit.com/v5/private"
BYBIT_PRIVATE_WS_DEMO = "wss://stream-demo.bybit.com/v5/private"
BYBIT_TRADE_WS_LIVE   = "wss://stream.bybit.com/v5/trade"
BYBIT_TRADE_WS_DEMO   = "wss://stream-demo.bybit.com/v5/trade"

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

    Ticks are throttled to one accepted update per 100 ms.
    tick_event (asyncio.Event) is set on every accepted tick so the
    management loop can wake up immediately instead of polling.

    Lifecycle: start dynamically via asyncio.create_task(stream.run())
    when a position opens; call stop() when the last position closes.
    """

    def __init__(self, symbol: str, rust_base_url: str):
        self.symbol = symbol
        self.rust_url = f"{rust_base_url}/ws/{symbol}"
        self.bybit_public_url = BYBIT_PUBLIC_WS
        self.last_tick: Dict = {"price": 0.0, "volume": 0.0}
        self._running = False
        self._last_emit_ts: float = 0.0        # monotonic clock; throttle anchor
        self.tick_event: asyncio.Event = asyncio.Event()

    async def run(self) -> None:
        if self._running:
            return   # guard against duplicate asyncio.create_task() calls
        self._running = True
        log.info("[MarketData] Stream starting for %s", self.symbol)
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
        """Parse messages from the Rust engine format, throttled to 100 ms."""
        now = time.monotonic()
        if now - self._last_emit_ts < 0.100:    # 100 ms throttle
            return
        msg_type = msg.get("type", "")
        price = 0.0
        vol = 0.0
        if msg_type in ("tick", "trade"):
            price = float(msg.get("price", 0) or 0)
            vol = float(msg.get("size", 0) or 0)
        if price > 0:
            self.last_tick = {"price": price, "volume": vol}
            self._last_emit_ts = now
            self.tick_event.set()               # wake management loop

    def stop(self) -> None:
        log.info("[MarketData] Stream stopping for %s", self.symbol)
        self._running = False


# ===========================================================================
# Bybit private WebSocket (position + execution stream)
# ===========================================================================

class PrivateStream:
    """
    Maintains a single authenticated WebSocket to Bybit private stream.
    Subscribes to: position, execution, order, wallet (all live — no polling).

    Callbacks (set by engine):
        on_position(data: list)   — position updates
        on_execution(data: list)  — fill/execution updates
        on_order(data: list)      — order status updates
        on_wallet(data: list)     — wallet / equity updates
        on_reconnect()            — async coroutine called after every successful
                                    auth, covering both startup and reconnects after
                                    drops.  Used by the engine to resync position
                                    state via REST when the live stream comes back.
    """

    def __init__(self, config: APIConfig):
        self._config = config
        self._url = BYBIT_PRIVATE_WS_DEMO if config.demo else BYBIT_PRIVATE_WS_LIVE
        self._running = False
        self.on_position  = None
        self.on_execution = None
        self.on_order     = None
        self.on_wallet    = None
        self.on_reconnect = None   # async coroutine

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

                    log.info("[PrivateWS] Authenticated. Subscribing to position/execution/order/wallet.")

                    # Subscribe to all private topics — fully live, no polling
                    sub = {
                        "op": "subscribe",
                        "args": ["position", "execution", "order", "wallet"],
                    }
                    await ws.send(json.dumps(sub))

                    backoff = 1.0   # reset on successful connect

                    # Fire reconnect callback so the engine can resync position
                    # state via REST (covers startup and reconnects after drops).
                    if self.on_reconnect:
                        asyncio.create_task(self.on_reconnect())

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
        data  = msg.get("data", [])

        if topic == "position" and self.on_position and data:
            self.on_position(data)
        elif topic == "execution" and self.on_execution and data:
            self.on_execution(data)
        elif topic == "order" and self.on_order and data:
            self.on_order(data)
        elif topic == "wallet" and self.on_wallet and data:
            self.on_wallet(data)

    def stop(self) -> None:
        self._running = False


# ===========================================================================
# Bybit trade WebSocket — RSA-signed order placement (no REST for orders)
# ===========================================================================

class TradeStream:
    """
    Async WebSocket to Bybit's Trade channel.

    Authenticated with RSA on connect. Each order request is signed with the
    same RSA key and matched back to a caller via reqId/asyncio.Future.

    place_order() returns the Bybit response dict on success, or None if the
    WS is not yet connected — callers should fall back to REST in that case.
    """

    def __init__(self, config: APIConfig):
        self._config  = config
        self._url     = BYBIT_TRADE_WS_DEMO if config.demo else BYBIT_TRADE_WS_LIVE
        self._ws      = None
        self._running = False
        self._pending: Dict[str, asyncio.Future] = {}
        self._req_seq = 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        while self._running:
            try:
                import websockets
                async with websockets.connect(
                    self._url, ping_interval=None
                ) as ws:
                    auth_msg = _ws_auth_payload(self._config)
                    await ws.send(json.dumps(auth_msg))

                    raw  = await asyncio.wait_for(ws.recv(), timeout=10)
                    resp = json.loads(raw)
                    if not resp.get("success"):
                        log.error("[TradeWS] Auth failed: %s", resp)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 60)
                        continue

                    log.info("[TradeWS] Authenticated — ready for live order execution.")
                    self._ws = ws
                    backoff  = 1.0

                    hb = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for raw in ws:
                            self._dispatch(raw)
                    finally:
                        hb.cancel()
            except Exception as exc:
                log.error("[TradeWS] Disconnected: %s — reconnecting in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
            finally:
                self._ws = None
                # Fail any pending futures so callers don't hang forever
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(ConnectionError("TradeWS disconnected"))
                self._pending.clear()

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
        req_id = msg.get("reqId", "")
        if req_id and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                fut.set_result(msg)

    # ── Order placement ──────────────────────────────────────────────────────

    def _next_req_id(self) -> str:
        self._req_seq += 1
        return f"me_{self._req_seq}_{int(time.time() * 1000)}"

    def _sign_order(self, timestamp_ms: str, payload_json: str) -> str:
        """RSA-SHA256 sign for WS trade requests (same key as auth)."""
        sign_str = (
            f"{timestamp_ms}{self._config.api_key}"
            f"{self._config.recv_window}{payload_json}"
        )
        if self._config.api_secret:
            import hashlib, hmac as _hmac
            return _hmac.new(
                self._config.api_secret.encode(),
                sign_str.encode(),
                hashlib.sha256,
            ).hexdigest()
        from Crypto.Hash import SHA256
        from Crypto.Signature import pkcs1_15
        from Crypto.PublicKey import RSA

        key_data: Optional[bytes] = None
        if self._config.private_key_content:
            key_data = self._config.private_key_content.replace("\\n", "\n").encode()
        elif self._config.private_key_path and os.path.exists(self._config.private_key_path):
            with open(self._config.private_key_path, "rb") as fh:
                key_data = fh.read()
        if key_data is None:
            raise RuntimeError("No RSA key available for WS trade signing.")
        private_key = RSA.import_key(key_data)
        h = SHA256.new(sign_str.encode())
        return base64.b64encode(pkcs1_15.new(private_key).sign(h)).decode()

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        reduce_only: bool = False,
        position_idx: int = 0,
        timeout: float = 5.0,
    ) -> Optional[dict]:
        """
        Place a market order via the WS trade channel.
        Returns the Bybit response dict, or None on failure/timeout.
        """
        if self._ws is None:
            return None   # not connected — caller falls back to REST

        req_id      = self._next_req_id()
        ts_ms       = str(int(time.time() * 1000))
        order_args  = {
            "category":    "linear",
            "symbol":      symbol,
            "side":        side,
            "orderType":   "Market",
            "qty":         qty,
            "timeInForce": "IOC",
            "reduceOnly":  reduce_only,
            "positionIdx": position_idx,
        }
        payload_json = json.dumps(order_args, separators=(",", ":"))
        signature    = self._sign_order(ts_ms, payload_json)

        body = {
            "reqId": req_id,
            "header": {
                "X-BAPI-TIMESTAMP":    ts_ms,
                "X-BAPI-RECV-WINDOW":  str(self._config.recv_window),
                "X-BAPI-SIGN":         signature,
            },
            "op":   "order.create",
            "args": [order_args],
        }

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        try:
            await self._ws.send(json.dumps(body))
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            log.error("[TradeWS] Order %s timed out after %.1fs", req_id, timeout)
            return None
        except Exception as exc:
            self._pending.pop(req_id, None)
            log.error("[TradeWS] Order send error: %s", exc)
            return None

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

        # Bybit REST client — used ONCE at startup for position sync only
        api_config = APIConfig()
        self.client = BybitV5Client(config=api_config)
        self._api_config = api_config

        # AI agent
        self.agent = MicroAgent(device="cpu")
        self.agent.load_weights(WEIGHTS_PATH)

        # Market data stream (Rust primary / Bybit public fallback).
        # NOT started here — started dynamically when a live position is detected.
        self.market = MarketDataStream(symbol, RUST_WS_URL)

        # Private stream (positions + executions + orders + wallet — all live)
        self.private_ws = PrivateStream(api_config)
        self.private_ws.on_position  = self._on_position_update
        self.private_ws.on_execution = self._on_execution_update
        self.private_ws.on_order     = self._on_order_update
        self.private_ws.on_wallet    = self._on_wallet_update
        self.private_ws.on_reconnect = self._rest_sync_positions

        # Trade WS — RSA-signed order placement, no REST for orders
        self.trade_ws = TradeStream(api_config)

        # ── State ────────────────────────────────────────────────────────────
        self._positions: Dict[int, dict] = {}
        self._session_realized: Dict[str, float] = defaultdict(float)
        self._agent_active = False

        # ── Feature engine ───────────────────────────────────────────────────
        self._feature_engine = FeatureEngineV2(redis_client=None)
        self._price_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

        # ── Trailing stops per position idx ──────────────────────────────────
        self._trailing_stops: Dict[int, HybridVolumeTrailingStop] = {}

        log.info("MicroExecutionEngine armed for %s.", symbol)

    # ── Private WS callbacks ─────────────────────────────────────────────────

    def _on_position_update(self, data: list) -> None:
        """
        Called on every 'position' topic push from Bybit private WS.
        Detects new positions (user opened) and closed ones.
        Dynamically starts/stops the market data stream so we only
        subscribe to symbols with an active live trade.
        """
        for pos in data:
            if pos.get("symbol") != self.symbol:
                continue

            idx  = int(pos.get("positionIdx", 0))
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
                    # Arm a trailing stop anchored at entry price
                    entry = float(pos.get("avgPrice", 0) or 0)
                    if entry > 0:
                        direction = (TrailingDirection.LONG if pos.get("side") == "Buy"
                                     else TrailingDirection.SHORT)
                        self._trailing_stops[idx] = HybridVolumeTrailingStop(
                            symbol=self.symbol,
                            config=HybridStopConfig(direction=direction, entry_price=entry),
                        )
                    # Subscribe to market data only for this live position
                    if not self.market._running:
                        asyncio.create_task(self.market.run())
            else:
                # size == 0 → position closed
                if idx in self._positions:
                    closed = self._positions.pop(idx)
                    realised = float(closed.get("cumRealisedPnl", 0) or 0)
                    self._session_realized[self.symbol] += realised
                    self._trailing_stops.pop(idx, None)
                    log.info(
                        "[PositionClosed] idx=%d realised=%.4f session_total=%.4f",
                        idx,
                        realised,
                        self._session_realized[self.symbol],
                    )
                    self.agent.reset_sequence()

            # Update active flag
            self._agent_active = len(self._positions) > 0

        # Stop market data when there are no open positions
        if not self._agent_active and self.market._running:
            self.market.stop()

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
                        self._session_realized[self.symbol] += realised
                        log.debug(
                            "[Execution] closedSize=%.4f pnl=%.4f session_total=%.4f",
                            closed_size,
                            realised,
                            self._session_realized[self.symbol],
                        )

    def _on_order_update(self, data: list) -> None:
        """Live order status feed from private WS (no polling needed)."""
        for order in data:
            if order.get("symbol") != self.symbol:
                continue
            log.info(
                "[OrderWS] id=%s status=%s side=%s qty=%s",
                order.get("orderId"),
                order.get("orderStatus"),
                order.get("side"),
                order.get("qty"),
            )

    def _on_wallet_update(self, data: list) -> None:
        """Live wallet / equity feed from private WS."""
        for wallet in data:
            for coin_info in wallet.get("coin", []):
                if coin_info.get("coin") == "USDT":
                    log.info(
                        "[WalletWS] USDT equity=%s available=%s",
                        coin_info.get("equity", "?"),
                        coin_info.get("availableToWithdraw", "?"),
                    )
                    # Keep account_equity up-to-date so the feature engine can use it
                    self._usdt_equity = float(coin_info.get("equity", 0) or 0)
                    break

    # ── Order helpers ────────────────────────────────────────────────────────

    def _estimate_fees(self, pos: dict) -> float:
        size  = float(pos.get("size", 0) or 0)
        entry = float(pos.get("avgPrice", 0) or 0)
        return size * entry * FEE_RATE

    async def _execute_order(
        self, side: str, qty: float, reduce_only: bool = False, position_idx: int = 0
    ) -> bool:
        """
        Place a market order in hedge-mode (positionIdx 1=Long, 2=Short).

        position_idx conventions
        ------------------------
        1  — Long position  (Buy to open, Sell+reduce_only to close)
        2  — Short position (Sell to open, Buy+reduce_only to close)

        All close/reduce orders MUST pass reduce_only=True — callers are
        responsible for this; it is asserted here as a safety net.
        """
        assert not (reduce_only is False and position_idx == 0), (
            "_execute_order: position_idx must be 1 or 2 in hedge mode"
        )
        qty_str = str(qty)
        # ── Primary: WS trade ────────────────────────────────────────────────
        ws_res = await self.trade_ws.place_order(
            symbol=self.symbol,
            side=side,
            qty=qty_str,
            reduce_only=reduce_only,
            position_idx=position_idx,
        )
        if ws_res is not None:
            ret_code = ws_res.get("retCode", -1)
            if ret_code == 0:
                log.info(
                    "[ORDER via WS] %s %.4f reduce_only=%s",
                    side, qty, reduce_only,
                )
                return True
            log.warning(
                "[ORDER via WS] retCode=%d msg=%s — falling back to REST",
                ret_code, ws_res.get("retMsg", ""),
            )

        # ── Fallback: REST ───────────────────────────────────────────────────
        try:
            res = self.client.place_order(
                symbol=self.symbol,
                side=side,
                order_type="Market",
                qty=qty_str,
                category="linear",
                time_in_force="IOC",
                reduce_only=reduce_only,
                position_idx=position_idx,
            )
            log.info("[ORDER via REST] %s %.4f reduce_only=%s | %s", side, qty, reduce_only, res)
            return True
        except Exception as exc:
            log.error("[ORDER FAILED] %s", exc)
            return False

    # ── Core management logic ────────────────────────────────────────────────

    def _combined_pnl(self) -> float:
        unrealised = sum(float(p.get("unrealisedPnl", 0) or 0) for p in self._positions.values())
        return unrealised + self._session_realized[self.symbol]

    async def _manage_hedged(self) -> None:
        """Resolve a 2-leg hedge when combined PnL is positive enough."""
        positions = list(self._positions.values())
        if len(positions) < 2:
            return

        fees     = sum(self._estimate_fees(p) for p in positions)
        combined = self._combined_pnl() - fees

        if combined > PROFIT_BUFFER:
            log.info(
                "[RESOLVE] Combined PnL=%.4f USDT (after fees=%.4f). Unwinding both legs.",
                combined, fees,
            )
            for pos in positions:
                close_side = "Sell" if pos["side"] == "Buy" else "Buy"
                idx = int(pos.get("positionIdx", 0))
                await self._execute_order(close_side, float(pos["size"]), reduce_only=True, position_idx=idx)

            if self._positions:
                any_pos = next(iter(self._positions.values()))
                tick  = self.market.last_tick
                state = self._build_state_tensor(any_pos, tick)
            else:
                state = torch.zeros(160)
            self.agent.self_reflect(state, "EXIT", combined)
        else:
            log.debug("[HEDGED] Combined PnL=%.4f — waiting.", combined)

    def _build_state_tensor(self, pos: dict, market_data: dict) -> torch.Tensor:
        """
        Build the 160-dim feature tensor via FeatureEngineV2.
        Falls back to zeros during warm-up (< 20 price bars).
        """
        price = float(market_data.get("price", 0) or 0)
        if price <= 0:
            return torch.zeros(160)

        buf = self._price_buffers[self.symbol]
        if len(buf) < 20:
            return torch.zeros(160)

        closes = np.array(list(buf), dtype=np.float32)
        ohlcv = {
            "close":  closes,
            "open":   closes,
            "high":   closes,
            "low":    closes,
            "volume": np.ones_like(closes),
        }

        equity = getattr(self, "_usdt_equity", 1.0)
        position_ctx = {
            "size":             float(pos.get("size", 0) or 0),
            "entry_price":      float(pos.get("avgPrice", 0) or 0),
            "pnl":              float(pos.get("unrealisedPnl", 0) or 0),
            "leverage":         float(pos.get("leverage", 1) or 1),
            "margin_used":      float(pos.get("positionBalance", 0) or 0),
            "account_equity":   equity,
            "side":             str(pos.get("side", "none")),
            "duration_seconds": 0.0,
        }

        raw = self._feature_engine.create_state_vector(ohlcv, position=position_ctx)
        raw = TokenProfiler.inject_profile(self.symbol, raw)
        return torch.tensor(raw, dtype=torch.float32)

    async def _manage_single(self, pos: dict) -> None:
        """Manage a single open position."""
        pnl         = float(pos.get("unrealisedPnl", 0) or 0)
        side        = pos.get("side", "Buy")
        size        = float(pos.get("size", 0) or 0)
        idx         = int(pos.get("positionIdx", 0))
        fees        = self._estimate_fees(pos)
        session_pnl = self._session_realized[self.symbol]
        tick        = self.market.last_tick

        # ── Physical blocker: cannot close in loss ────────────────────────
        if pnl < 0:
            if pnl <= HEDGE_TRIGGER_LOSS:
                log.warning(
                    "[DEFENSE] PnL=%.4f <= trigger=%.4f — opening delta hedge.",
                    pnl, HEDGE_TRIGGER_LOSS,
                )
                hedge_side = "Sell" if side == "Buy" else "Buy"
                # Buy hedge → Long slot (idx 1); Sell hedge → Short slot (idx 2)
                hedge_idx  = 1 if hedge_side == "Buy" else 2
                await self._execute_order(hedge_side, size, position_idx=hedge_idx)
            else:
                log.debug("[LOCKED] PnL=%.4f — close blocked (in loss).", pnl)
            return

        # ── Trailing stop (only active when position is in profit) ────────
        if tick["price"] > 0 and idx in self._trailing_stops:
            ts        = self._trailing_stops[idx]
            triggered = ts.ingest_tick(tick["price"], tick["volume"])
            if triggered and pnl > 0:
                log.info(
                    "[TRAIL] Trailing stop fired (%s). PnL=%.4f — locking in profit.",
                    ts.trigger_reason, pnl,
                )
                close_side   = "Sell" if side == "Buy" else "Buy"
                state_tensor = self._build_state_tensor(pos, tick)
                ok = await self._execute_order(close_side, size, reduce_only=True, position_idx=idx)
                if ok:
                    self.agent.self_reflect(state_tensor, "EXIT", pnl + session_pnl - fees * 2)
                return

        # ── Profitable: ask AI whether to close ──────────────────────────
        net_pnl = pnl + session_pnl - fees * 2
        if net_pnl <= PROFIT_BUFFER:
            log.debug("[WAIT] net_pnl=%.4f below buffer=%.4f.", net_pnl, PROFIT_BUFFER)
            return

        if tick["price"] <= 0:
            log.debug("[WAIT] No live price available yet — skipping AI decision.")
            return

        state_tensor = self._build_state_tensor(pos, tick)
        action       = self.agent.predict(state_tensor)

        if action == "EXIT":
            log.info("[PROFIT] AI signals EXIT. PnL=%.4f net=%.4f", pnl, net_pnl)
            close_side = "Sell" if side == "Buy" else "Buy"
            ok = await self._execute_order(close_side, size, reduce_only=True, position_idx=idx)
            if ok:
                self.agent.self_reflect(state_tensor, "EXIT", net_pnl)
        elif action == "HEDGE":
            log.info("[WAVE-HEDGE] AI signals HEDGE. PnL=%.4f", pnl)
            hedge_side = "Sell" if side == "Buy" else "Buy"
            hedge_idx  = 1 if hedge_side == "Buy" else 2
            await self._execute_order(hedge_side, size, position_idx=hedge_idx)
        else:
            log.debug("[HOLD] AI=%s  PnL=%.4f  session=%.4f", action, pnl, session_pnl)

    async def _management_loop(self) -> None:
        """
        Event-driven management loop.

        Waits on market.tick_event (set by the 100 ms throttled _ingest callback)
        instead of sleeping a fixed interval.  Falls back to a 1 s timeout to
        handle the rare case where a position update arrives while the market
        stream is not yet connected.  No polling.

        Position state is populated entirely by the private WS callbacks
        (_on_position_update) and by _rest_sync_positions which fires on every
        private WS (re)connect — no manual startup call needed here.
        """
        while True:
            try:
                try:
                    await asyncio.wait_for(self.market.tick_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    self.market.tick_event.clear()

                price = self.market.last_tick["price"]
                if price > 0:
                    self._price_buffers[self.symbol].append(price)

                if self._agent_active:
                    n = len(self._positions)
                    if n >= 2:
                        await self._manage_hedged()
                    elif n == 1:
                        pos = next(iter(self._positions.values()))
                        await self._manage_single(pos)
            except Exception as exc:
                log.error("[Loop] Unhandled error: %s", exc)

    async def _rest_sync_positions(self) -> None:
        """
        REST position resync — called by PrivateStream on every successful
        (re)connect so the engine never has stale state after a WS drop.

        Fetches both hedge legs (Long positionIdx=1, Short positionIdx=2) for
        the tracked symbol and merges them into self._positions.  Any leg with
        size=0 is treated as closed and removed.  Does NOT reset session_realized
        so cumulative PnL memory survives reconnects.
        """
        try:
            log.info("[Sync] REST position resync triggered (WS reconnect).")
            positions = self.client.get_positions(category="linear", symbol=self.symbol)
            seen_idxs = set()
            for pos in positions.values():
                idx = pos.position_idx
                seen_idxs.add(idx)
                if pos.size > 0:
                    self._positions[idx] = {
                        "symbol":        pos.symbol,
                        "side":          pos.side,
                        "size":          str(pos.size),
                        "avgPrice":      str(pos.entry_price),
                        "unrealisedPnl": str(pos.unrealized_pnl),
                        "cumRealisedPnl": str(pos.realized_pnl),
                        "positionIdx":   str(pos.position_idx),
                        "leverage":      str(pos.leverage),
                        "positionBalance": str(pos.position_balance),
                    }
                    log.info(
                        "[Sync] Position idx=%d side=%s size=%s entry=%.4f",
                        idx, pos.side, pos.size, pos.entry_price,
                    )
                    # Arm trailing stop if not already armed
                    if idx not in self._trailing_stops and pos.entry_price > 0:
                        direction = (TrailingDirection.LONG if pos.side == "Buy"
                                     else TrailingDirection.SHORT)
                        self._trailing_stops[idx] = HybridVolumeTrailingStop(
                            symbol=self.symbol,
                            config=HybridStopConfig(direction=direction, entry_price=pos.entry_price),
                        )
                else:
                    # Flat leg — remove if we had it tracked
                    self._positions.pop(idx, None)
                    self._trailing_stops.pop(idx, None)

            self._agent_active = len(self._positions) > 0

            # Start market data stream if we have live positions and it's not running
            if self._agent_active and not self.market._running:
                asyncio.create_task(self.market.run())
            elif not self._agent_active and self.market._running:
                self.market.stop()

            log.info("[Sync] Done — %d active position(s).", len(self._positions))
        except Exception as exc:
            log.error("[Sync] REST resync failed: %s — live WS will self-correct.", exc)

    # ── Entry point ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        log.info("Starting MicroExecutionEngine — symbol=%s", self.symbol)
        log.info("Rust WS: %s  |  Demo: %s", RUST_WS_URL, self._api_config.demo)
        log.info("All data paths: LIVE WebSocket only (no polling).")

        # Market stream is NOT included here — it starts dynamically in
        # _on_position_update / _rest_sync_positions when a live position exists.
        await asyncio.gather(
            self.private_ws.run(),
            self.trade_ws.run(),
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
