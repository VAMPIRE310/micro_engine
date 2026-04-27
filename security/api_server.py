"""
NEO SUPREME 2026 — HEADLESS JSON REST API
===========================================
NO HTML, NO CSS, NO JINJA. Pure JSON API with CORS.
Consumed by Next.js (Vercel) and Flutter (Mobile).

Endpoints:
  GET  /api/v1/health           → System health
  GET  /api/v1/agents/state     → All agent states with AI personality
  GET  /api/v1/agents/{name}    → Single agent detail
  GET  /api/v1/portfolio        → Portfolio + positions
  GET  /api/v1/market/overview  → Market data overview
  GET  /api/v1/trades/recent    → Recent trades
  GET  /api/v1/metrics          → System metrics + Redis + DB + WS status
  POST /api/v1/control/start    → Start AI
  POST /api/v1/control/pause    → Pause AI
  POST /api/v1/control/stop     → Stop AI
  GET  /api/v1/ingester/status  → Rust ingester shard status
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

logger = logging.getLogger("api")

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)

_orchestrator: Any = None


def register_orchestrator(orch):
    """Called by main.py to link the running orchestrator."""
    global _orchestrator
    _orchestrator = orch
    logger.info("[API] Orchestrator registered")


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
@app.route("/api/v1/health", methods=["GET"])
def health():
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2026.0.1",
        "services": {},
    }

    if _orchestrator:
        status = _orchestrator.get_status()
        health_status["services"]["orchestrator"] = status.get("status", "unknown")
        health_status["services"]["database"] = status.get("database", "unknown")
        health_status["services"]["ai_started"] = status.get("ai_started", False)
        health_status["services"]["weights_loaded"] = status.get("weights_loaded", False)
    else:
        health_status["services"]["orchestrator"] = "not_registered"

    # Check Redis
    try:
        import redis
        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=2)
        r.ping()
        health_status["services"]["redis"] = "connected"
    except Exception:
        health_status["services"]["redis"] = "disconnected"

    # Check PostgreSQL
    try:
        from database.connection import get_db
        db = get_db()
        result = db.health_check()
        health_status["services"]["postgresql"] = result.get("status", "unknown")
    except Exception:
        health_status["services"]["postgresql"] = "disconnected"

    all_healthy = all(v in ("healthy", "connected", "not_registered")
                      for v in health_status["services"].values())
    health_status["status"] = "healthy" if all_healthy else "degraded"

    return jsonify(health_status), 200 if all_healthy else 503


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTS — AI PERSONALITY EXPOSED
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/agents/state", methods=["GET"])
def agents_state():
    """All agents with mathematical confidence translated to human-readable personality."""
    if not _orchestrator:
        return jsonify({"error": "Orchestrator not ready"}), 503

    agents_data = []
    for name, agent in (_orchestrator.agents or {}).items():
        try:
            stats = agent.get_stats() if hasattr(agent, "get_stats") else {}
            personality = getattr(agent, "_last_personality", {}) or {}

            agents_data.append({
                "id": name,
                "model": _agent_model_name(name),
                "status": "active" if stats.get("inference_count", 0) > 0 else "idle",
                "personality": {
                    "confidence_score": round(personality.get("confidence_score", 0.0), 4),
                    "state": personality.get("state", "OBSERVING"),
                    "color": personality.get("color", "#00FFFF"),
                    "reflection": personality.get("reflection", "Scanning market..."),
                    "entropy": round(personality.get("entropy", 0.0), 4),
                    "last_inference_ms": round(stats.get("avg_latency_ms", 0), 2),
                    "inference_count": stats.get("inference_count", 0),
                    "weight_hash": stats.get("weight_hash", "")[:8] + "...",
                },
            })
        except Exception as e:
            logger.debug(f"[API] Agent state error: {e}")

    return jsonify({
        "timestamp": time.time(),
        "agents": agents_data,
        "ai_started": _orchestrator._ai_started if _orchestrator else False,
        "weights_loaded": _orchestrator._weights_loaded if _orchestrator else False,
    })


@app.route("/api/v1/agents/<name>", methods=["GET"])
def agent_detail(name: str):
    if not _orchestrator or not _orchestrator.agents:
        return jsonify({"error": "Not ready"}), 503

    agent = _orchestrator.agents.get(name)
    if not agent:
        return jsonify({"error": f"Agent '{name}' not found"}), 404

    try:
        stats = agent.get_stats() if hasattr(agent, "get_stats") else {}
        personality = getattr(agent, "_last_personality", {}) or {}

        return jsonify({
            "id": name,
            "model": _agent_model_name(name),
            "status": "active",
            "personality": personality,
            "stats": stats,
            "device": getattr(agent, "device", "cpu"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _agent_model_name(name: str) -> str:
    mapping = {
        "sniper": "CNN 1D",
        "forecaster": "TCN",
        "strike": "QR-DQN",
        "negotiator": "SAC",
        "position_manager": "QR-DQN+LSTM",
    }
    return mapping.get(name, name)


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/portfolio", methods=["GET"])
def portfolio():
    if not _orchestrator:
        return jsonify({"error": "Not ready"}), 503

    positions = []
    for sym, pos in (_orchestrator.positions or {}).items():
        positions.append({
            "symbol": sym,
            "side": pos.get("side", "Long"),
            "size": pos.get("size", 0),
            "entry_price": pos.get("entry_price", 0),
            "mark_price": pos.get("mark_price", 0),
            "unrealized_pnl": pos.get("unrealized_pnl", 0),
            "unrealized_pnl_pct": pos.get("unrealized_pnl_pct", 0),
            "leverage": pos.get("leverage", 1),
        })

    # Risk/CoreSat data now comes from Rust via Redis — return empty if not available
    risk = {}
    core_sat = {}
    if hasattr(_orchestrator, "risk_engine"):
        try:
            risk = _orchestrator.risk_engine.get_stats()
        except Exception:
            pass
    if hasattr(_orchestrator, "core_satellite"):
        try:
            core_sat = _orchestrator.core_satellite.get_stats()
        except Exception:
            pass

    return jsonify({
        "timestamp": time.time(),
        "mode": _orchestrator.mode if _orchestrator else "unknown",
        "active_positions": len(positions),
        "positions": positions,
        "core_satellite": core_sat,
        "risk": risk,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/market/overview", methods=["GET"])
def market_overview():
    if not _orchestrator:
        return jsonify({"error": "Not ready"}), 503

    # Get Redis-cached market data
    market_data = []
    for sym in list((_orchestrator.market_data or {}).keys())[:20]:
        data = _orchestrator.market_data.get(sym, {})
        market_data.append({
            "symbol": sym,
            "price": data.get("price", 0),
            "bid": data.get("bid", 0),
            "ask": data.get("ask", 0),
            "volume_24h": data.get("volume_24h", 0),
            "change_24h": data.get("price_change_24h", 0),
        })

    # Whale metricsmandatory, 3th inference gate

    whale = {}
    if hasattr(_orchestrator, "whale_tracker"):
        try:
            for sym in list(_orchestrator.market_data.keys())[:5]:
                whale[sym] = _orchestrator.whale_tracker.get_all_metrics(sym)
        except Exception:
            pass

    return jsonify({
        "timestamp": time.time(),
        "symbols_tracked": len(_orchestrator.market_data) if _orchestrator else 0,
        "market_data": market_data,
        "whale_metrics": whale,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# TRADES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/trades/recent", methods=["GET"])
def recent_trades():
    limit = request.args.get("limit", 50, type=int)

    try:
        from database.connection import get_db
        db = get_db()
        rows = db.fetchall("""
            SELECT symbol, side, price, qty as size, realized_pnl as pnl, status, created_at
            FROM trades
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))

        trades = []
        for r in rows or []:
            trades.append({
                "symbol": r.get("symbol", ""),
                "side": r.get("side", ""),
                "price": float(r.get("price", 0) or 0),
                "size": float(r.get("size", 0) or 0),
                "pnl": float(r.get("pnl", 0) or 0),
                "status": r.get("status", ""),
                "time": r.get("created_at", "").isoformat() if hasattr(r.get("created_at"), "isoformat") else str(r.get("created_at", "")),
            })

        return jsonify({"trades": trades, "count": len(trades)})
    except Exception as e:
        return jsonify({"trades": [], "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/metrics", methods=["GET"])
def metrics():
    if not _orchestrator:
        return jsonify({"error": "Not ready"}), 503

    status = _orchestrator.get_status()

    # Get Redis metrics
    redis_status = "disconnected"
    redis_messages = 0
    try:
        import redis
        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=2)
        r.ping()
        redis_status = "connected"
        redis_info = r.info("stats")
        redis_messages = redis_info.get("total_commands_processed", 0)
    except Exception:
        pass

    return jsonify({
        "timestamp": time.time(),
        "orchestrator": status,
        "redis": {
            "status": redis_status,
            "total_commands": redis_messages,
        },
        "system": {
            "uptime_seconds": status.get("uptime", 0),
            "tick_count": status.get("tick_count", 0),
            "tier2_cycles": status.get("tier2_cycles", 0),
            "active_positions": status.get("active_positions", 0),
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# INGESTER STATUS (Rust shards)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/ingester/status", methods=["GET"])
def ingester_status():
    """Poll Rust ingester status via Redis or fallback."""
    try:
        import redis
        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=2)

        # Try to get last metrics published by Rust
        metrics_raw = r.get("system:metrics")
        if metrics_raw:
            rust_metrics = json.loads(metrics_raw) if isinstance(metrics_raw, bytes) else json.loads(metrics_raw)
        else:
            rust_metrics = {}

        return jsonify({
            "source": "rust_ingester",
            "status": "running" if rust_metrics else "unknown",
            "rust_metrics": rust_metrics,
            "channels": [
                "ticks:*", "trades:*", "orderbook:*",
                "candles:*:*", "mtf:*", "anomalies:*",
            ],
        })
    except Exception as e:
        return jsonify({
            "source": "rust_ingester",
            "status": "unavailable",
            "error": str(e),
            "channels": ["ticks:*", "trades:*", "orderbook:*", "candles:*:*", "mtf:*"],
        }), 503


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/v1/control/start", methods=["POST"])
def control_start():
    if _orchestrator:
        _orchestrator._paused = False
        _orchestrator.running = True
    return jsonify({"status": "started", "timestamp": time.time()})


@app.route("/api/v1/control/pause", methods=["POST"])
def control_pause():
    if _orchestrator:
        _orchestrator._paused = True
    return jsonify({"status": "paused", "timestamp": time.time()})


@app.route("/api/v1/control/stop", methods=["POST"])
def control_stop():
    if _orchestrator:
        _orchestrator.stop()
    return jsonify({"status": "stopped", "timestamp": time.time()})


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER START
# ═══════════════════════════════════════════════════════════════════════════════

def create_app(orch=None) -> Flask:
    if orch:
        register_orchestrator(orch)
    return app


def start_api_server(host: str = "0.0.0.0", port: int = 8080, orch=None):
    if orch:
        register_orchestrator(orch)
    app.run(host=host, port=port, threaded=True, debug=False)

