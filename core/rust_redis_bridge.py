"""
rust_redis_bridge.py — Rust Pub/Sub → Persistent List Bridge
=============================================================
Subscribes to the ephemeral Redis pub/sub channels that the Rust Bybit ingester
publishes into and materialises them into the persistent rolling lists that the
Exodia Feature Engine reads via LRANGE.

Channel → List mapping
----------------------
  klines:{SYMBOL}:{TF}   →  market:kline_history:{SYMBOL}:{TF}
  trades:{SYMBOL}        →  market:live_trades:{SYMBOL}

Timeframe values published by Rust: "1", "5", "15", "60", "120", "240", "D"
These match the suffixes already expected by the feature engine, so no remapping
is required.

Run as a standalone daemon:
  python -m core.rust_redis_bridge
  # or directly:
  python core/rust_redis_bridge.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("rust_redis_bridge")

# How many bars / trades to keep per list key (prevents unbounded memory growth).
# Override via environment variables if your deployment has different memory constraints.
MAX_KLINE_HISTORY: int = int(os.environ.get("BRIDGE_MAX_KLINE_HISTORY", "500"))
MAX_TRADE_HISTORY: int = int(os.environ.get("BRIDGE_MAX_TRADE_HISTORY", "1000"))


class RustRedisBridge:
    """
    Listens to Rust's ephemeral pub/sub streams and maintains the persistent
    rolling lists consumed by the Exodia Feature Engine.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self.redis_url = redis_url
        self._r: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        self._r = await aioredis.from_url(self.redis_url, decode_responses=True)
        self._pubsub = self._r.pubsub(ignore_subscribe_messages=True)
        # Subscribe to all kline and trade channels published by the Rust ingester.
        await self._pubsub.psubscribe("klines:*", "trades:*")
        log.info(
            "🌉 Rust-to-Exodia Bridge connected to %s — listening on klines:* and trades:*",
            self.redis_url,
        )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle(self, message: dict) -> None:
        """Route an incoming pub/sub message to the correct persistent list."""
        try:
            channel: str = message["channel"]
            raw_data: str = message["data"]

            parts = channel.split(":")
            if len(parts) < 2:
                return

            stream_type = parts[0]  # "klines" or "trades"

            if stream_type == "klines":
                # Channel format: klines:{SYMBOL}:{TF}
                if len(parts) < 3:
                    return
                symbol = parts[1]
                tf = parts[2]  # "1", "5", "15", "60", "120", "240", "D"
                list_key = f"market:kline_history:{symbol}:{tf}"
                max_len = MAX_KLINE_HISTORY

            elif stream_type == "trades":
                # Channel format: trades:{SYMBOL}
                symbol = parts[1]
                list_key = f"market:live_trades:{symbol}"
                max_len = MAX_TRADE_HISTORY

            else:
                return

            # Validate that the payload is well-formed JSON before storing it.
            try:
                json.loads(raw_data)
            except (json.JSONDecodeError, ValueError):
                log.debug("Skipping non-JSON payload on channel %s", channel)
                return

            # Atomically append and trim inside a pipeline to keep the list bounded.
            assert self._r is not None
            async with self._r.pipeline(transaction=True) as pipe:
                pipe.rpush(list_key, raw_data)
                pipe.ltrim(list_key, -max_len, -1)
                await pipe.execute()

            log.debug("Bridged %s → %s", channel, list_key)

        except Exception as exc:  # noqa: BLE001
            log.error("Bridge handle error: %s", exc)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Connect and process messages forever, reconnecting on any error."""
        while True:
            try:
                await self._connect()
                assert self._pubsub is not None
                while True:
                    msg = await self._pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )
                    if msg:
                        await self._handle(msg)
                    # Yield to the event loop between messages so we never starve
                    # other coroutines even under high-volume feeds.
                    await asyncio.sleep(0.0)
            except Exception as exc:  # noqa: BLE001
                log.error("Bridge loop error: %s — reconnecting in 2s", exc)
                if self._pubsub is not None:
                    try:
                        await self._pubsub.close()
                    except Exception:
                        pass
                    self._pubsub = None
                if self._r is not None:
                    try:
                        await self._r.aclose()
                    except Exception:
                        pass
                    self._r = None
                await asyncio.sleep(2.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    bridge = RustRedisBridge(redis_url=redis_url)
    try:
        asyncio.run(bridge.run_forever())
    except KeyboardInterrupt:
        log.info("Bridge safely shut down.")


if __name__ == "__main__":
    main()
