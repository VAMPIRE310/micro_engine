"""
NEO SUPREME 2026 — Health Check API
FastAPI/Flask-compatible health endpoint for Railway probes.
Checks: PostgreSQL, Redis, AI agents.
"""
import json
import time
import logging
from typing import Dict, Any

logger = logging.getLogger("health")


class HealthEndpoint:
    """
    Health check handler. Mount at /health for Railway.
    Returns 200 only if critical systems are operational.
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._start_time = time.time()
        self._db = None
        self._redis = None

    def set_db(self, db):
        self._db = db

    def set_redis(self, redis_client):
        self._redis = redis_client

    def check(self) -> Dict[str, Any]:
        """Run full health check."""
        checks = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - self._start_time,
            "version": "2026.0.1",
        }

        # PostgreSQL check
        if self._db:
            try:
                result = self._db.health_check()
                checks["database"] = result
                if result.get("status") != "healthy":
                    checks["status"] = "degraded"
            except Exception as e:
                checks["database"] = {"status": "unhealthy", "error": str(e)}
                checks["status"] = "degraded"
        else:
            checks["database"] = {"status": "not_configured"}

        # Redis check
        if self._redis:
            try:
                ping = self._redis.ping()
                checks["redis"] = {"status": "healthy" if ping else "unhealthy"}
                if not ping:
                    checks["status"] = "degraded"
            except Exception as e:
                checks["redis"] = {"status": "unhealthy", "error": str(e)}
                checks["status"] = "degraded"
        else:
            checks["redis"] = {"status": "not_configured"}

        # Agent checks
        if self.orchestrator:
            agent_status = {}
            for name, agent in self.orchestrator.agents.items():
                try:
                    stats = agent.get_stats() if hasattr(agent, "get_stats") else {}
                    agent_status[name] = {
                        "status": "healthy",
                        "inferences": stats.get("total_inferences", 0),
                        "error_rate": stats.get("error_rate", 0),
                    }
                except Exception as e:
                    agent_status[name] = {"status": "unhealthy", "error": str(e)}
                    checks["status"] = "degraded"
            checks["agents"] = agent_status

            # Weight sync status
            checks["weights_loaded"] = getattr(self.orchestrator, '_weights_loaded', False)
            checks["ai_started"] = getattr(self.orchestrator, '_ai_started', False)

        # Critical failure = database down
        if checks.get("database", {}).get("status") == "unhealthy":
            checks["status"] = "unhealthy"

        return checks

    def to_http_response(self) -> tuple:
        """Return (status_code, body_dict) for HTTP response."""
        result = self.check()
        code = 200 if result["status"] == "healthy" else (503 if result["status"] == "unhealthy" else 200)
        return code, result


# ─── Flask-compatible route handlers ───

def create_flask_app(orchestrator=None):
    """Create minimal Flask app with health endpoint."""
    try:
        from flask import Flask, jsonify
    except ImportError:
        logger.warning("[Health] Flask not installed")
        return None

    app = Flask("neo_supreme_health")
    health = HealthEndpoint(orchestrator)

    @app.route("/health")
    def health_route():
        code, body = health.to_http_response()
        return jsonify(body), code

    @app.route("/ready")
    def ready_route():
        """Readiness probe."""
        if orchestrator and orchestrator.running:
            return jsonify({"ready": True}), 200
        return jsonify({"ready": False}), 503

    @app.route("/metrics")
    def metrics_route():
        """Prometheus-style metrics."""
        if not orchestrator:
            return jsonify({}), 200
        status = orchestrator.get_status()
        metrics_text = f"""# NEO SUPREME 2026 Metrics
neo_supreme_tick_count {status.get('tick_count', 0)}
neo_supreme_uptime_seconds {status.get('uptime', 0)}
neo_supreme_active_positions {status.get('active_positions', 0)}
neo_supreme_ai_started {1 if status.get('ai_started') else 0}
neo_supreme_weights_loaded {1 if status.get('weights_loaded') else 0}
"""
        return metrics_text, 200, {"Content-Type": "text/plain"}

    return app


def start_health_server(orchestrator=None, host: str = "0.0.0.0", port: int = 8080):
    """Start health check server in background thread."""
    app = create_flask_app(orchestrator)
    if not app:
        return None

    def run():
        app.run(host=host, port=port, threaded=True, debug=False)

    import threading
    t = threading.Thread(target=run, daemon=True, name="HealthServer")
    t.start()
    logger.info(f"[Health] Server started on {host}:{port}")
    return t
