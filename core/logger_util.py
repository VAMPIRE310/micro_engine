"""
NEO SUPREME 2026 — Logging Infrastructure
Redis-backed log handler with sanitization and structured formatting.
"""
import logging
import redis
import json
import re
from typing import Optional


class SanitizedFormatter(logging.Formatter):
    """Formatter that strips API keys and secrets from log output."""

    SECRET_PATTERNS = [
        (re.compile(r'(api[_-]?key[=:\s]+)[\w-]+', re.I), r'\g<1>***'),
        (re.compile(r'(secret[=:\s]+)[\w-]+', re.I), r'\g<1>***'),
        (re.compile(r'(signature[=:\s]+)[\w-]+', re.I), r'\g<1>***'),
        (re.compile(r'(private[_-]?key[=:\s]+)[\w\s+=/\n-]+', re.I | re.S), r'\g<1>***'),
        (re.compile(r'(bearer\s+)[\w-]+\.[\w-]+\.[\w-]+', re.I), r'\g<1>***'),
    ]

    def format(self, record):
        msg = super().format(record)
        for pattern, repl in self.SECRET_PATTERNS:
            msg = pattern.sub(repl, msg)
        return msg


class RedisLogHandler(logging.Handler):
    """
    Logging handler that publishes to Redis pub/sub.
    UI subscribes to this channel for real-time log display.
    """

    def __init__(self, redis_client=None, channel: str = "neosupreme:logs", level=logging.INFO):
        super().__init__(level)
        self.channel = channel
        self._redis = redis_client
        self._local_buffer = []
        self._max_buffer = 1000
        self.setFormatter(SanitizedFormatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
        ))

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "thread": record.thread,
            }
            if self._redis:
                self._redis.publish(self.channel, json.dumps(log_entry))
        except Exception:
            pass


def setup_root_logger(redis_client=None, log_level=logging.INFO):
    """Configure root logger with console + Redis handlers."""
    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear existing handlers
    root.handlers = []

    # Console handler with sanitized formatter
    console = logging.StreamHandler()
    console.setFormatter(SanitizedFormatter(
        '%(asctime)s [%(name)-20s] %(levelname)-8s %(message)s'
    ))
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler("logs/neo_supreme.log", mode='a')
    file_handler.setFormatter(SanitizedFormatter(
        '%(asctime)s [%(name)-20s] %(levelname)-8s %(message)s'
    ))
    root.addHandler(file_handler)

    # Redis handler if available
    if redis_client:
        root.addHandler(RedisLogHandler(redis_client))

    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return root
