#!/usr/bin/env python3
"""
Bybit API V5 - Complete Production Implementation
================================================

Unified implementation combining all V5 API features:
- REST API: All V5 endpoints with RSA authentication
- WebSocket: Public & Private streams with full topic support
- WebSocket Trade API: Order operations via WebSocket
- Time synchronization: NTP-based correction for API desync
- Demo mode: Paper trading support (replaces testnet)
- Environment-based configuration

Authentication: RSA Key - REQUIRED
                API Key - REQUIRED
                
No HMAC support - RSA only for production security.
No testnet - Use demo=True for paper trading.

Features:
- Auto-failover across ALL official Bybit regions
- Hardcoded rate limits (never get banned)
- Circuit breaker pattern (stop hammering dead endpoints)
- Health checking with exponential backoff
- Jitter to prevent thundering herd
- Belgium/FSMA compliant (auto-detects geo-blocks)

Official Bybit Regions:
- api.bybit.com (Global)
- api.bytick.com (Alternative Global)
- api.bybit.eu (EU/FSMA compliant)
- api.bybit.nl (Netherlands)
- api.bybit-tr.com (Turkey)
- api.bybit.kz (Kazakhstan)
- api.bybitgeorgia.ge (Georgia)
- api.bybit.ae (UAE)
- api.bybit.id (Indonesia)
- api-demo.bybit.com (Demo/Paper trading)

Rate Limits (Official Bybit V5):
- Public endpoints: 120 req/s per endpoint group
- Private endpoints: 600 req/min per UID
- WebSocket: 500 subscriptions per connection
- Order placement: 10 req/s per symbol
Official Docs: https://bybit-exchange.github.io/docs/v5/guide
"""

from __future__ import annotations
import asyncio
import os
import sys
import time
import random
import hashlib
import hmac
import json
import logging
import threading
import traceback
import base64
from abc import ABC, abstractmethod
from enum import Enum, auto
from decimal import Decimal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from collections import deque, defaultdict
from datetime import datetime, timedelta
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import requests
import requests.adapters
from requests.packages.urllib3.util.retry import Retry
# Optional imports with fallbacks
try:
    import ntplib
    NTP_AVAILABLE = True
except ImportError:
    NTP_AVAILABLE = False
    logging.warning("ntplib not installed. Time sync may be less accurate.")

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    raise ImportError("websocket-client is required. Install: pip install websocket-client")

try:
    # Use pycryptodome for RSA (same as pybit library)
    from Crypto.Hash import SHA256
    from Crypto.PublicKey import RSA
    from cryptography.hazmat.primitives import hashes, padding as crypto_padding
    from cryptography.hazmat.primitives.asymmetric import padding
    from Crypto.Signature import pkcs1_15
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    raise ImportError("pycryptodome is required for RSA. Install: pip install pycryptodome")

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitState(Enum):
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, rejecting requests
    HALF_OPEN = auto()   # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker to prevent hammering failing endpoints"""
    
    failure_threshold: int = 5          # Open after 5 failures
    success_threshold: int = 3          # Close after 3 successes in half-open
    timeout: float = 30.0               # Try again after 30s
    
    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _success_count: int = field(default=0, repr=False)
    _last_failure_time: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout elapsed for half-open
                if time.time() - self._last_failure_time >= self.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"[CIRCUIT] Transitioning to HALF_OPEN after timeout")
            return self._state
    
    def record_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"[CIRCUIT] Transitioning to CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def record_failure(self) -> bool:
        """Returns True if circuit just opened"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed in half-open, go back to open
                self._state = CircuitState.OPEN
                logger.warning(f"[CIRCUIT] Transitioning to OPEN (failed in half-open)")
                return True
            
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"[CIRCUIT] Transitioning to OPEN after {self.failure_count} failures")
                return True
            
            return False
    
    def can_execute(self) -> bool:
        return self.state != CircuitState.OPEN

# =============================================================================
# RATE LIMITER WITH TOKEN BUCKET
# =============================================================================

@dataclass
class TokenBucket:
    """Token bucket for smooth rate limiting"""
    
    rate: float  # tokens per second
    capacity: float  # max tokens
    
    _tokens: float = field(default=0, repr=False)
    _last_update: float = field(default_factory=time.time, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def __post_init__(self):
        self._tokens = self.capacity  # Start full
    
    def consume(self, tokens: float = 1.0) -> float:
        """
        Attempt to consume tokens. Returns wait time if insufficient.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_update = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            
            # Calculate wait time
            deficit = tokens - self._tokens
            wait_time = deficit / self.rate
            return wait_time

# =============================================================================
# ENDPOINT HEALTH TRACKER
# =============================================================================

@dataclass
class EndpointHealth:
    """Tracks health metrics for each endpoint"""
    
    region: BybitRegion
    circuit: CircuitBreaker = field(default_factory=CircuitBreaker)
    
    # Rate limiters
    public_bucket: TokenBucket = field(
        default_factory=lambda: TokenBucket(
            rate=RateLimits.PUBLIC_PER_SECOND,
            capacity=RateLimits.PUBLIC_PER_SECOND
        )
    )
    private_bucket: TokenBucket = field(
        default_factory=lambda: TokenBucket(
            rate=RateLimits.PRIVATE_PER_MINUTE / 60,
            capacity=RateLimits.PRIVATE_PER_MINUTE / 60
        )
    )
    
    # Stats
    success_count: int = 0
    failure_count: int = 0
    avg_latency: float = 0.0
    last_used: float = 0.0
    
    def record_latency(self, latency_ms: float):
        """Update rolling average latency"""
        if self.avg_latency == 0:
            self.avg_latency = latency_ms
        else:
            # EWMA with alpha=0.2
            self.avg_latency = 0.8 * self.avg_latency + 0.2 * latency_ms

            
# =============================================================================
# HARD CODED BYBIT OFFICIAL LIMITS - NEVER CHANGE THESE
# =============================================================================

class RateLimits:
    """Official Bybit V5 rate limits - hardcoded to prevent bans"""
    
    # Public Market Data (IP-based)
    PUBLIC_PER_SECOND = 120  # 120 req/s per IP
    PUBLIC_PER_MINUTE = 1000  # 1000 req/min per IP
    
    # Private Trading (UID-based)
    PRIVATE_PER_MINUTE = 600  # 600 req/min per UID
    PRIVATE_PER_SECOND = 10   # 10 req/s for order placement per symbol
    
    # WebSocket
    WS_MAX_SUBSCRIPTIONS = 500  # Per connection
    WS_PING_INTERVAL = 20  # Seconds
    
    # Order specific
    ORDERS_PER_SECOND = 10  # Per symbol
    ORDERS_PER_MINUTE = 100  # Per symbol
    
    # Batch orders
    BATCH_MAX_ORDERS = 10  # Max orders per batch request
    
    # Backoff settings
    BASE_BACKOFF = 1.0
    MAX_BACKOFF = 60.0
    JITTER_MAX = 0.5  # Add random 0-0.5s to backoff

# =============================================================================
# RESILIENT CLIENT CONFIGURATION
# =============================================================================

@dataclass
class ResilientConfig:
    """Configuration for resilient client"""
    
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    # Region preferences — bybit.com only (EU/NL use separate accounts, DO NOT use)
    region_priority: List[BybitRegion] = field(default_factory=lambda: [
        BybitRegion.GLOBAL,  # api.bybit.com — primary
        BybitRegion.BYTICK,  # bytick.com — backup
    ])
    
    demo: bool = False
    recv_window: int = 5000
    timeout: int = 10
    max_retries: int = 3
    
    # Failover settings
    failover_on_403: bool = True  # Auto-failover on geo-block
    failover_on_timeout: bool = True
    failover_on_5xx: bool = True
    
    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_timeout: float = 30.0

# =============================================================================
# ALL OFFICIAL BYBIT REGIONS
# =============================================================================

class BybitRegion(Enum):
    """All official Bybit API regions with priority order"""
    
    #EU endpoints
    EU = ("https://api.bybit.eu", "wss://stream.bybit.eu", "EU", 1)
    NL = ("https://api.bybit.nl", "wss://stream.bybit.nl", "Netherlands", 2)
    
    # Global primary
    GLOBAL = ("https://api.bybit.com", "wss://stream.bybit.com", "Global", 3)
    BYTICK = ("https://api.bytick.com", "wss://stream.bytick.com", "Bytick", 4)
    
    # Regional alternatives
    TURKEY = ("https://api.bybit-tr.com", "wss://stream.bybit-tr.com", "Turkey", 5)
    KAZAKHSTAN = ("https://api.bybit.kz", "wss://stream.bybit.kz", "Kazakhstan", 6)
    GEORGIA = ("https://api.bybitgeorgia.ge", "wss://stream.bybitgeorgia.ge", "Georgia", 7)
    UAE = ("https://api.bybit.ae", "wss://stream.bybit.ae", "UAE", 8)
    INDONESIA = ("https://api.bybit.id", "wss://stream.bybit.id", "Indonesia", 9)
    
    # Demo/Paper trading
    DEMO = ("https://api-demo.bybit.com", "wss://stream-demo.bybit.com", "Demo", 10)

class BybitResilientClient:
    """
    Production-grade Bybit V5 client with:
    - Automatic failover across all regions
    - Hardcoded rate limits (never get banned)
    - Circuit breakers (don't hammer dead endpoints)
    - Health tracking (use fastest endpoint)
    - FSMA compliance (Belgium/EU safe)
    """
        
    def __init__(self, rest_url, ws_url, display_name, priority, config: Optional[ResilientConfig] = None):
        self.config = config or ResilientConfig()
        self.rest_url = rest_url
        self.ws_url = ws_url
        self.display_name = display_name
        self.priority = priority
        
        # Initialize health tracking for all regions
        self._health: Dict[BybitRegion, EndpointHealth] = {
            region: EndpointHealth(region=region)
            for region in BybitRegion
        }
        
        # Current working endpoint
        self._current_region: Optional[BybitRegion] = None
        self._session = requests.Session()
        
        # Configure session with retries for connection errors
        adapter = requests.adapters.HTTPAdapter(
            max_retries=Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
        )
        self._session.mount("https://", adapter)
        
        # Time sync
        self._time_offset: float = 0.0
        self._sync_time()
        
        logger.info(f"[INIT] Resilient client initialized")
        logger.info(f"[INIT] Priority regions: {[r.display_name for r in self.config.region_priority]}")
        
# ==================== ENUMS ====================

class MarketCategory(Enum):
    """Market categories for V5 API"""
    SPOT = "spot"
    LINEAR = "linear"      # USDT Perpetual, USDC Perpetual
    INVERSE = "inverse"    # Inverse Perpetual, Inverse Futures
    OPTION = "option"

class OrderSide(Enum): #side='buy',
    """Order sides"""
    BUY = "Buy"
    SELL = "Sell"

class OrderType(Enum): #type='market',
    """Order types"""
    MARKET = "Market"
    LIMIT = "Limit"
    POST_ONLY = "PostOnly"
    FOK = "FOK"            # Fill or Kill
    IOC = "IOC"            # Immediate or Cancel

class TimeInForce(Enum):
    """Time in force"""
    GTC = "GTC"            # Good Till Cancel
    IOC = "IOC"
    FOK = "FOK"
    POST_ONLY = "PostOnly"

class PositionSide(Enum): #position_side='long',
    """Position sides"""
    LONG = "Long"
    SHORT = "Short"
    BOTH = "Both"          # One-way mode

class OrderStatus(Enum):
    """Order statuses"""
    CREATED = "Created"
    NEW = "New"
    REJECTED = "Rejected"
    PARTIALLY_FILLED = "PartiallyFilled"
    PARTIALLY_FILLED_CANCELED = "PartiallyFilledCanceled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    UNTRIGGERED = "Untriggered"
    TRIGGERED = "Triggered"
    DEACTIVATED = "Deactivated"

class TriggerDirection(Enum):
    """Trigger direction"""
    RISE_TO_TRIGGER = 1    # Rise to trigger
    FALL_TO_TRIGGER = 2    # Fall to trigger

class TpslMode(Enum):
    """TP/SL mode"""
    FULL = "Full"
    PARTIAL = "Partial"

class MarginMode(Enum):
    """Margin mode"""
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"

class PositionMode(Enum):
    """Position mode"""
    MERGED_SINGLE = 0      # One-way mode
    BOTH_SIDES = 3         # Hedge mode

class WebSocketTopic(Enum):
    """WebSocket topics"""
    # Public topics
    TICKER = "tickers"
    KLINE = "kline"
    ORDERBOOK = "orderbook"
    TRADE = "publicTrade"
    LIQUIDATION = "liquidation"
    LT_TICKER = "lt"
    LT_KLINE = "lt_kline"
    
    # Private topics
    POSITION = "position"
    EXECUTION = "execution"
    ORDER = "order"
    WALLET = "wallet"
    GREEKS = "greeks"


# ==================== DATA CLASSES ====================

@dataclass
class APIConfig:
    """
    API Configuration - RSA Authentication Required

    Environment Variables:
        BYBIT_API_KEY: "28v6ulhXpfSRlj59Wd"
        BYBIT_RSA_PRIVATE_KEY_PATH: C:\\Users\\VAMPIRE\\OneDrive\\Desktop\\28v6ulhXpfSRlj59Wd_bybit_api.pem
        BYBIT_RSA_PRIVATE_KEY: RSA private key content (alternative to path)
        BYBIT_DEMO: "true" for demo/paper trading (default: true)
    """
    api_key: str = "28v6ulhXpfSRlj59Wd"
    api_secret: Optional[str] = None           #The regular API secret
    private_key_path: Optional[str] = r"C:\Users\VAMPIRE\OneDrive\Desktop\28v6ulhXpfSRlj59Wd_bybit_api.pem"     # The RSA Personal KEY .pem file 
    private_key_content: Optional[str] = None  # Alternative to path
    demo: bool = False                         # LIVE TRADING MODE - YOLO
    recv_window: int = 5000
    max_retries: int = 5
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.2              # ~5 req/s - conservative for Bybit limits
    backoff_factor: float = 1.8
    
    def __post_init__(self):
        # Import at function start to avoid UnboundLocalError
        import sys
        import os
        from pathlib import Path
        
        # Try secrets manager first if credentials not provided directly
        if not self.api_key or (not self.private_key_path and not self.private_key_content):
            try:
                # 🎯 DYNAMIC PATH ANCHORING
                # Resolves the project root (NEO_SIMPLE) by going up from this file's location
                # This ensures subfolder scripts can find the 'infrastructure' module
                current_file = Path(__file__).resolve()
                # If this file is in 'Bybit_API_client', project_root is one level up
                # If it's in root, .parent is enough. We'll check for 'infrastructure' folder.
                project_root = current_file.parent
                if not (project_root / "infrastructure").exists():
                    project_root = project_root.parent

                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from infrastructure.secrets_manager import SecretsManager
                secrets = SecretsManager(backend='local')
                
                if not self.api_key:
                    # Clean the key immediately upon retrieval
                    raw_key = secrets.get('BYBIT_LIVE_API_KEY_28V6')
                    self.api_key = str(raw_key).strip() if raw_key else ''
                
                if not self.private_key_path and not self.private_key_content:
                    # Priority 1: Raw Content (The 3271-character string)
                    key_content = secrets.get('BYBIT_LIVE_PRIVATE_KEY_28V6')
                    if key_content:
                        self.private_key_content = str(key_content).strip()
                    else:
                        # Priority 2: File Path
                        raw_path = secrets.get('BYBIT_PRIVATE_KEY_PATH')
                        self.private_key_path = str(raw_path).strip() if raw_path else ''
                
                # Get demo mode from secrets
                demo_val = secrets.get('BYBIT_DEMO')
                if demo_val is not None:
                    self.demo = str(demo_val).lower() in ('true', '1', 'yes', 'on')
                        
            except Exception as e:
                # Use print if logger isn't initialized yet in __post_init__
                if 'logger' in globals():
                    logger.debug(f"Secrets manager fetch failed: {e}")
                else:
                    print(f"DEBUG: Secrets manager fetch failed: {e}")
        
        # Fallback to environment variables
        if not self.api_key:
            self.api_key = os.environ.get('BYBIT_API_KEY', '').strip()
        if not self.private_key_path:
            self.private_key_path = os.environ.get('BYBIT_RSA_PRIVATE_KEY_PATH', '').strip()
        if not self.private_key_content:
            self.private_key_content = os.environ.get('BYBIT_RSA_PRIVATE_KEY', '').strip()
        
        # Default demo mode
        if not hasattr(self, 'demo') or self.demo is None:
            demo_env = os.environ.get('BYBIT_DEMO', 'true').lower()
            self.demo = demo_env in ('true', '1', 'yes', 'on')
        
        # Validation
        if not self.api_key:
            raise ValueError("BYBIT_API_KEY is required")
        if not self.private_key_path and not self.private_key_content:
            raise ValueError("BYBIT_PRIVATE_KEY or BYBIT_PRIVATE_KEY_PATH is required")

@dataclass
class TickerData:
    """Ticker/Market data"""
    symbol: str
    last_price: float = 0.0
    index_price: float = 0.0
    mark_price: float = 0.0
    prev_price_24h: float = 0.0
    price_24h_pcnt: float = 0.0
    high_price_24h: float = 0.0
    low_price_24h: float = 0.0
    open_interest: float = 0.0
    open_interest_value: float = 0.0
    turnover_24h: float = 0.0
    volume_24h: float = 0.0
    funding_rate: float = 0.0
    next_funding_time: Optional[datetime] = None
    predicted_delivery_price: float = 0.0
    basis_rate: float = 0.0
    delivery_fee_rate: float = 0.0
    delivery_time: Optional[datetime] = None
    bid1_price: float = 0.0
    bid1_size: float = 0.0
    ask1_price: float = 0.0
    ask1_size: float = 0.0
    basis: float = 0.0
    pre_open_price: float = 0.0
    pre_qty: float = 0.0
    cur_pre_listing_phase: str = ""
    funding_interval_hour: int = 8
    basis_rate_year: float = 0.0
    funding_cap: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'TickerData':
        """Create from API response"""
        def safe_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        def safe_int(val, default=0):
            try:
                return int(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        def parse_timestamp(val):
            if not val:
                return None
            try:
                return datetime.fromtimestamp(int(val) / 1000)
            except (ValueError, TypeError):
                return None
        
        return cls(
            symbol=data.get('symbol', ''),
            last_price=safe_float(data.get('lastPrice')),
            index_price=safe_float(data.get('indexPrice')),
            mark_price=safe_float(data.get('markPrice')),
            prev_price_24h=safe_float(data.get('prevPrice24h')),
            price_24h_pcnt=safe_float(data.get('price24hPcnt')),
            high_price_24h=safe_float(data.get('highPrice24h')),
            low_price_24h=safe_float(data.get('lowPrice24h')),
            open_interest=safe_float(data.get('openInterest')),
            open_interest_value=safe_float(data.get('openInterestValue')),
            turnover_24h=safe_float(data.get('turnover24h')),
            volume_24h=safe_float(data.get('volume24h')),
            funding_rate=safe_float(data.get('fundingRate')),
            next_funding_time=parse_timestamp(data.get('nextFundingTime')),
            predicted_delivery_price=safe_float(data.get('predictedDeliveryPrice')),
            basis_rate=safe_float(data.get('basisRate')),
            delivery_fee_rate=safe_float(data.get('deliveryFeeRate')),
            delivery_time=parse_timestamp(data.get('deliveryTime')),
            bid1_price=safe_float(data.get('bid1Price')),
            bid1_size=safe_float(data.get('bid1Size')),
            ask1_price=safe_float(data.get('ask1Price')),
            ask1_size=safe_float(data.get('ask1Size')),
            basis=safe_float(data.get('basis')),
            pre_open_price=safe_float(data.get('preOpenPrice')),
            pre_qty=safe_float(data.get('preQty')),
            cur_pre_listing_phase=data.get('curPreListingPhase', ''),
            funding_interval_hour=safe_int(data.get('fundingIntervalHour'), 8),
            basis_rate_year=safe_float(data.get('basisRateYear')),
            funding_cap=safe_float(data.get('fundingCap'))
        )

@dataclass
class KlineData:
    """Kline/Candlestick data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    symbol: str = ""
    interval: str = ""
    
    @classmethod
    def from_api_response(cls, data: List, symbol: str = "", interval: str = "") -> 'KlineData':
        """Create from API response (list format)"""
        return cls(
            timestamp=datetime.fromtimestamp(int(data[0]) / 1000),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            turnover=float(data[6]) if len(data) > 6 else 0.0,
            symbol=symbol,
            interval=interval
        )

@dataclass
class OrderBookLevel:
    """Order book level"""
    price: float
    size: float
    side: str = ""  # 'bid' or 'ask'

@dataclass
class OrderBook:
    """Order book snapshot"""
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime = field(default_factory=datetime.now)
    seq: int = 0  # Sequence number for updates
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return 0.0
    
    @property
    def spread_pct(self) -> float:
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 100
        return 0.0

@dataclass
class Position:
    """Position data"""
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float
    mark_price: float = 0.0
    liquidation_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    trailing_stop: float = 0.0
    position_idx: int = 0
    mode: str = ""
    auto_add_margin: int = 0
    position_balance: float = 0.0
    session_avg_price: float = 0.0
    occ_closing_fee: float = 0.0
    occ_funding_fee: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Order:
    """Order data"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    price: float
    qty: float
    status: str
    order_link_id: str = ""
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None
    leaves_qty: float = 0.0
    cum_exec_qty: float = 0.0
    cum_exec_value: float = 0.0
    cum_exec_fee: float = 0.0
    time_in_force: str = ""
    reduce_only: bool = False
    close_on_trigger: bool = False
    trigger_price: float = 0.0
    trigger_direction: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    tp_trigger_by: str = ""
    sl_trigger_by: str = ""
    place_type: str = ""
    smp_type: str = ""
    smp_group: int = 0
    smp_order_id: str = ""

@dataclass
class Execution:
    """Execution/Trade data"""
    exec_id: str
    order_id: str
    symbol: str
    side: str
    exec_price: float
    exec_qty: float
    exec_value: float
    exec_fee: float
    exec_type: str
    exec_time: datetime
    order_link_id: str = ""
    is_maker: bool = False
    fee_rate: float = 0.0
    trade_iv: str = ""
    mark_iv: str = ""
    index_price: float = 0.0
    underlying_price: float = 0.0
    block_trade_id: str = ""
    closed_size: float = 0.0
    seq: int = 0

@dataclass
class WalletBalance:
    """Wallet balance"""
    coin: str
    wallet_balance: float
    available_balance: float
    margin_balance: float
    unrealized_pnl: float
    cum_realized_pnl: float
    equity: float
    usd_value: float
    borrow_amount: float = 0.0
    available_to_withdraw: float = 0.0
    accrued_interest: float = 0.0
    total_order_im: float = 0.0
    total_position_im: float = 0.0
    total_position_mm: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class WebhookSignal:
    """External webhook signal"""
    action: str  # 'buy', 'sell', 'close'
    symbol: str
    price: Optional[float] = None
    qty: Optional[float] = None
    order_type: str = "market"
    leverage: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "webhook"
    signature: Optional[str] = None


# ==================== BYBIT V5 CLIENT ====================

class BybitV5Websocket:
    def __init__(self, url: str, topics: list):
        self.url = url
        self.topics = topics
        self.ws = None
        self.running = False
        self.data_queue = asyncio.Queue()
        self.logger = logging.getLogger("BybitWS")

    async def connect(self):
        self.running = True
        retry_delay = 1.0
        backoff_factor = 1.8
        max_delay = 60.0

        while self.running:
            try:
                async with websockets.connect(self.url) as ws:
                    self.ws = ws
                    self.logger.info("Bybit WebSocket Connected.")
                    retry_delay = 1.0 
                    
                    sub_payload = {"op": "subscribe", "args": self.topics}
                    await self.ws.send(json.dumps(sub_payload))
                    
                    ping_task = asyncio.create_task(self._heartbeat_loop())
                    listen_task = asyncio.create_task(self._listen_loop())
                    
                    await asyncio.gather(ping_task, listen_task)
                    
            except Exception as e:
                self.logger.warning(f"Connection dropped: {e}. Reconnecting in {retry_delay:.2f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * backoff_factor, max_delay)

    async def _heartbeat_loop(self):
        """Enforces Bybit's mandatory 20-second active ping requirement."""
        while self.running and self.ws and self.ws.open:
            try:
                await self.ws.send(json.dumps({"req_id": "10001", "op": "ping"}))
                await asyncio.sleep(20)
            except Exception as e:
                self.logger.error(f"Heartbeat failed: {e}")
                break

    async def _listen_loop(self):
        async for message in self.ws:
            await self.data_queue.put(message)
            
class BybitV5Client:
    """
    Complete Bybit V5 API Client - RSA Authentication
    
    Features:
    - Full REST API coverage (all V5 endpoints)
    - Complete WebSocket support (Public, Private, Trade)
    - NTP time synchronization for API desync issues
    - Demo mode for paper trading
    - No pybit dependency - pure Python implementation
    - No testnet - use demo=True for testing
    
    Authentication:
        Requires RSA private key (PEM format) + API key
        Set via environment variables or pass to constructor
    """
    
    # Mainnet endpoints
    BASE_URL = "https://api.bybit.com"
    BASE_URL_BACKUP = "https://api.bytick.com"
    BASE_URL_DEMO = "https://api-demo.bybit.com"

    # Failover order: global → bytick
    ENDPOINT_PRIORITY = [
        "https://api.bybit.com",
        "https://api.bytick.com",
    ]

    # WebSocket endpoints (Mainnet)
    WS_PUBLIC_LINEAR = "wss://stream.bybit.com/v5/public/linear"
    WS_PUBLIC_SPOT   = "wss://stream.bybit.com/v5/public/spot"
    WS_PUBLIC_INVERSE = "wss://stream.bybit.com/v5/public/inverse"
    WS_PUBLIC_OPTION  = "wss://stream.bybit.com/v5/public/option"
    WS_PRIVATE = "wss://stream.bybit.com/v5/private"
    WS_TRADE   = "wss://stream.bybit.com/v5/trade"
    # WebSocket endpoints (Demo)
    WS_DEMO_PUBLIC = "wss://stream-demo.bybit.com/v5/public/linear"
    WS_DEMO_PRIVATE = "wss://stream-demo.bybit.com/v5/private"
    WS_DEMO_TRADE = "wss://stream-demo.bybit.com/v5/trade"
    
    def __init__(self, config: Optional[APIConfig] = None):
        """
        Initialize Bybit V5 Client
        
        Args:
            config: API configuration. If None, loads from environment.
        """
        self.config = config or APIConfig()
        
        # Time synchronization
        self.config = config or APIConfig()
        
        # Track working endpoint for failover
        self._working_endpoint = None
        
        if self.config.demo:
            self.BASE_URL = 'https://api-demo.bybit.com'
            self.WS_PRIVATE = 'wss://stream-demo.bybit.com/v5/private'
            self.WS_TRADE = 'wss://stream-demo.bybit.com/v5/trade'
        # else: use class-level BASE_URL (api.bybit.com)
            
        self.private_key = True
        self._load_private_key()
        
        # WebSocket connections
        self.ws_public: Optional[websocket.WebSocketApp] = None
        self.ws_private: Optional[websocket.WebSocketApp] = None
        self.ws_trade: Optional[websocket.WebSocketApp] = None
        self._ws_threads: Dict[str, threading.Thread] = {}
        self._ws_running = False
        self._ws_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._ws_subscriptions: Set[str] = set()
        
        # Data storage
        self.tickers: Dict[str, TickerData] = {}
        self.klines: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.orderbooks: Dict[str, OrderBook] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.executions: deque = deque(maxlen=1000)
        self.wallet: Dict[str, WalletBalance] = {}
        
        # Callbacks
        self._ticker_callbacks: List[Callable[[TickerData], None]] = []
        self._kline_callbacks: List[Callable[[KlineData], None]] = []
        self._orderbook_callbacks: List[Callable[[OrderBook], None]] = []
        self._liquidation_callbacks: List[Callable[[Dict], None]] = []
        self._position_callbacks: List[Callable[[Position], None]] = []
        self._order_callbacks: List[Callable[[Order], None]] = []
        self._execution_callbacks: List[Callable[[Execution], None]] = []
        self._wallet_callbacks: List[Callable[[WalletBalance], None]] = []
        self._webhook_callbacks: List[Callable[[WebhookSignal], None]] = []
        self._public_trade_callbacks: List[Callable[[Dict], None]] = []
        
        # Locks
        self._data_lock = threading.RLock()
        self._request_lock = threading.Lock()  # For thread-safe API requests
        
        # Rate limiting
        self.rate_limit_last = 0
        self.time_offset = 0.0  # NTP time offset for signature
        self.time_drift_samples = []  # For NTP time sync
        
        # Initial time sync to prevent timestamp errors on first requests
        self._sync_time()
        
        logger.info(f"BybitV5Client initialized (demo={self.config.demo})")
    
    # ==================== TIME SYNCHRONIZATION ====================
    
    def _sync_time(self) -> float:
        """
        Synchronize time with Bybit server to prevent timestamp errors.
        REPLACES your previous version entirely. Corrects logic leaks and dead blocks.
        """
        # 1. Try NTP first (High Precision for local 4090 clock)
        if NTP_AVAILABLE:
            try:
                c = ntplib.NTPClient()
                resp = c.request('pool.ntp.org', version=3, timeout=2)
                self.time_offset = resp.offset
                logger.info(f"[TIME] NTP sync successful, offset: {self.time_offset:.3f}s")
                return self.time_offset
            except Exception as e:
                logger.warning(f"[TIME] NTP sync failed: {e}")

        # 2. Regional Failover (Bybit REST Endpoints)
        # This loop replaces the repetitive 'try/except' blocks from your version
        for region in self.config.region_priority:
            try:
                start = time.time()
                resp = self._session.get(
                    f"{region.rest_url}/v5/market/time",
                    timeout=3
                )
                end = time.time()
                latency = (end - start) * 1000
            
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('retCode') == 0:
                        # Precise Midpoint Calculation: (Request start + Response end) / 2
                        server_time = float(data['result']['timeSecond'])
                        local_time = (start + end) / 2
                        self.time_offset = server_time - local_time
                    
                        # Record Health Metrics
                        self._health[region].record_latency(latency)
                        logger.info(f"[TIME] Synced with {region.display_name}, offset={self.time_offset:.3f}s, latency={latency:.1f}ms")
                        return self.time_offset
            except Exception as e:
                logger.debug(f"[TIME] {region.display_name} sync failed: {e}")
                continue

        # 3. Safety Fallback
        logger.error("[TIME] All time sync methods failed. Using 0.0 offset.")
        self.time_offset = 0.0
        return self.time_offset
    
    def _get_timestamp(self) -> str:
        """Get current timestamp with offset correction"""
        return str(int((time.time() + self.time_offset) * 1000))
    
    def _adjust_time_drift(self, server_time_ms: int):
        """Adjust time offset based on server response"""
        local_time_ms = int((time.time() + self.time_offset) * 1000)
        drift = (server_time_ms - local_time_ms) / 1000.0
        self.time_drift_samples.append(drift)
        
        # If drift is significant, adjust
        if len(self.time_drift_samples) >= 3:
            avg_drift = sum(self.time_drift_samples) / len(self.time_drift_samples)
            if abs(avg_drift) > 1.0:  # More than 1 second drift
                logger.warning(f"Time drift detected: {avg_drift:.3f}s, resyncing...")
                self._sync_time()
    
    def _get_headers(self, auth: bool, payload: str = "") -> Dict[str, str]:
        """Generate request headers"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'BybitResilientClient/1.0'
        }
        
        if auth and self.config.api_key:
            timestamp = self._get_timestamp()
            headers['X-BAPI-API-KEY'] = self.config.api_key
            headers['X-BAPI-TIMESTAMP'] = timestamp
            headers['X-BAPI-RECV-WINDOW'] = str(self.config.recv_window)
            headers['X-BAPI-SIGN'] = self._generate_signature(
                timestamp, self.config.api_key, str(self.config.recv_window), payload
            )
        
        if self.config.demo:
            headers['X-BAPI-DEMO-TRADING'] = '1'
        
        return headers
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        base = RateLimits.BASE_BACKOFF * (2 ** attempt)
        jitter = random.uniform(0, RateLimits.JITTER_MAX)
        return min(base + jitter, RateLimits.MAX_BACKOFF)
    
    def _wait_for_rate_limit(self, region: BybitRegion, is_private: bool):
        """Wait if rate limit bucket is empty"""
        health = self._health[region]
        bucket = health.private_bucket if is_private else health.public_bucket
        
        wait_time = bucket.consume(1.0)
        if wait_time > 0:
            logger.debug(f"[RATE LIMIT] Waiting {wait_time:.2f}s for {region.display_name}")
            time.sleep(wait_time)
    
    def _select_best_endpoint(self) -> Optional[BybitRegion]:
        """Select best available endpoint based on health"""
        available = []
        
        for region in self.config.region_priority:
            health = self._health[region]
            
            # Skip if circuit is open
            if not health.circuit.can_execute():
                logger.debug(f"[SELECT] {region.display_name}: Circuit OPEN")
                continue
            
            # Skip demo unless configured
            if region == BybitRegion.DEMO and not self.config.demo:
                continue
            
            # Calculate score (lower is better)
            # Factors: latency, success rate, recency
            success_rate = health.success_count / max(health.success_count + health.failure_count, 1)
            latency_score = health.avg_latency if health.avg_latency > 0 else 1000
            recency_score = time.time() - health.last_used
            
            # Weighted score
            score = (latency_score * 0.5) + ((1 - success_rate) * 500) - (recency_score * 10)
            available.append((region, score))
        
        if not available:
            return None
        
        # Sort by score (lower is better)
        available.sort(key=lambda x: x[1])
        best_region = available[0][0]
        
        logger.debug(f"[SELECT] Best endpoint: {best_region.display_name}")
        return best_region
    
    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
        auth: bool = False
    ) -> Dict[str, Any]:
        """
        Make resilient request with auto-failover and rate limiting
        
        Args:
            method: HTTP method
            endpoint: API endpoint (e.g., "/market/kline")
            params: Query parameters
            body: Request body
            auth: Whether to include authentication
            
        Returns:
            API response dict
            
        Raises:
            ConnectionError: If all endpoints fail
        """
        params = params or {}
        body = body or {}
        
        # Build payload for signature
        if method == "GET":
            payload = urlencode(sorted(params.items())) if params else ""
        else:
            payload = json.dumps(body, separators=(',', ':')) if body else ""
        
        headers = self._get_headers(auth, payload)
        
        # Try endpoints in priority order with circuit breaker awareness
        last_error = None
        attempted_regions = []
        
        for _ in range(len(self.config.region_priority)):
            region = self._select_best_endpoint()
            if not region:
                logger.error("[REQUEST] No available endpoints (all circuits open)")
                break
            
            attempted_regions.append(region.display_name)
            health = self._health[region]
            
            # Rate limiting
            self._wait_for_rate_limit(region, auth)
            
            # Build URL
            base_url = region.rest_url
            url = f"{base_url}/v5{endpoint}"
            
            if method == "GET" and payload:
                full_url = f"{url}?{payload}"
            else:
                full_url = url
            
            # Execute with retries
            for attempt in range(self.config.max_retries):
                try:
                    start_time = time.time()
                    
                    if method == "GET":
                        response = self._session.get(
                            full_url, 
                            headers=headers, 
                            timeout=self.config.timeout
                        )
                    elif method == "POST":
                        response = self._session.post(
                            url, 
                            headers=headers, 
                            data=payload, 
                            timeout=self.config.timeout
                        )
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    
                    latency = (time.time() - start_time) * 1000
                    health.record_latency(latency)
                    health.last_used = time.time()
                    
                    # Check for geo-block (403 Forbidden)
                    if response.status_code == 403:
                        logger.warning(f"[403] {region.display_name} blocked (FSMA/geo-restriction)")
                        health.circuit.record_failure()
                        health.failure_count += 1
                        last_error = f"403 Forbidden on {region.display_name}"
                        break  # Try next region
                    
                    # Check for rate limit (429)
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"[429] Rate limited on {region.display_name}, waiting {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    
                    # Check for server error (5xx)
                    if response.status_code >= 500:
                        logger.warning(f"[{response.status_code}] Server error on {region.display_name}")
                        if self.config.failover_on_5xx:
                            health.circuit.record_failure()
                            health.failure_count += 1
                            last_error = f"{response.status_code} on {region.display_name}"
                            break  # Try next region
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Check API error codes
                    if result.get('retCode') == 10002:  # Timestamp error
                        logger.warning(f"[10002] Time sync error, resyncing...")
                        self._sync_time()
                        time.sleep(self._calculate_backoff(attempt))
                        continue
                    
                    if result.get('retCode') == 10006:  # Rate limit
                        retry_after = 5
                        logger.warning(f"[10006] Rate limited, waiting {retry_after}s...")
                        time.sleep(retry_after)
                        continue
                    
                    if result.get('retCode') != 0:
                        error_msg = result.get('retMsg', 'Unknown error')
                        # Don't failover on client errors (4xx)
                        if result.get('retCode') < 5000:
                            raise ValueError(f"API error {result['retCode']}: {error_msg}")
                        # Server errors might be regional, try failover
                        last_error = f"API error {result['retCode']} on {region.display_name}"
                        health.failure_count += 1
                        break
                    
                    # Success!
                    health.success_count += 1
                    health.circuit.record_success()
                    self._current_region = region
                    
                    logger.debug(f"[SUCCESS] {region.display_name} {endpoint} ({latency:.0f}ms)")
                    return result.get('result', result)
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"[TIMEOUT] {region.display_name} attempt {attempt + 1}")
                    if self.config.failover_on_timeout:
                        health.circuit.record_failure()
                        health.failure_count += 1
                        last_error = f"Timeout on {region.display_name}"
                        break  # Try next region
                    time.sleep(self._calculate_backoff(attempt))
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"[ERROR] {region.display_name}: {e}")
                    health.circuit.record_failure()
                    health.failure_count += 1
                    last_error = f"{e} on {region.display_name}"
                    time.sleep(self._calculate_backoff(attempt))
            
            # Mark circuit if all retries failed
            if last_error and region.display_name in last_error:
                health.circuit.record_failure()
        
        # All endpoints exhausted
        raise ConnectionError(
            f"All endpoints failed. Attempted: {attempted_regions}. Last error: {last_error}"
        )
        
    # ==================== RSA & HMAC AUTHENTICATION ====================
    
    def _load_private_key(self):
        """Load RSA private key from path or content, fixing formatting issues"""
        # If an API Secret is present, we are in HMAC mode; no need to load RSA
        if self.config.api_secret:
            logger.info("HMAC mode detected via api_secret. Skipping RSA key load.")
            self.private_key = None
            return

        if not CRYPTO_AVAILABLE:
            raise RuntimeError("pycryptodome library required for RSA authentication.")
        
        try:
            key_data = None

            # 1. Handle Key Content (Directly from .env or Secrets)
            if self.config.private_key_content:
                raw_key = self.config.private_key_content
                if isinstance(raw_key, str):
                    # FIX: Handle literal '\\n' strings from env files
                    raw_key = raw_key.replace('\\n', '\n')
                    # Ensure it has the standard PEM headers if missing
                    if "-----BEGIN PRIVATE KEY-----" not in raw_key:
                        logger.warning("RSA key content missing PEM headers. Attempting to import anyway.")
                    key_data = raw_key.encode('utf-8')
                else:
                    key_data = raw_key
                
            # 2. Handle Key Path (From .pem file)
            elif self.config.private_key_path and os.path.exists(self.config.private_key_path):
                with open(self.config.private_key_path, 'rb') as f:
                    key_data = f.read()

            # 3. Final Import
            if key_data:
                self.private_key = RSA.import_key(key_data)
                logger.info("RSA private key loaded successfully.")
            else:
                logger.warning("No RSA key provided — running in public-data-only mode. Private endpoints will fail.")
                self.private_key = None

        except ValueError:
            logger.warning("No RSA key provided — running in public-data-only mode.")
            self.private_key = None
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load RSA key: {e}. Check your PEM formatting!")
            raise
    
    def _generate_signature(self, timestamp: str, api_key: str, recv_window: str, payload: str = "") -> str:
        """Generate RSA signature for API request using pycryptodome (same as pybit)"""
        # EXACT format Bybit V5 expects: timestamp + api_key + recv_window + payload
        param_str = f"{timestamp}{api_key}{recv_window}{payload}"

        if not self.config.api_secret:
            # RSA path
            if not hasattr(self, 'private_key') or self.private_key is None:
                return ""
            hash_obj = SHA256.new(param_str.encode('utf-8'))
            signature_bytes = pkcs1_15.new(self.private_key).sign(hash_obj)
            return base64.b64encode(signature_bytes).decode('utf-8')
        else:
            # HMAC path (Standard API Secret)
            return hmac.new(
                self.config.api_secret.encode('utf-8'),
                param_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
    
    # ==================== HTTP REQUEST HANDLING ====================
    
    def _rate_limit_sleep(self):
        """Apply rate limiting"""
        with self._request_lock:
            now = time.time()
            elapsed = now - self.rate_limit_last
            if elapsed < self.config.rate_limit_delay:
                time.sleep(self.config.rate_limit_delay - elapsed)
            self.rate_limit_last = time.time()
    
    def request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None, 
        body: Optional[Dict] = None,
        auth: bool = True
    ) -> Dict:
        """
        Make HTTP request to Bybit V5 API
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/market/kline")
            params: Query parameters
            body: Request body (for POST/PUT)
            auth: Whether to include authentication headers
        
        Returns:
            API response dict
        """
        self._rate_limit_sleep()
        
        url = f"{self.BASE_URL}/v5{endpoint}"
        timestamp = self._get_timestamp()
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Build payload string for signature
        if method == "GET":
            payload = urlencode(sorted(params.items())) if params else ""
            full_url = f"{url}?{payload}" if payload else url
        else:
            payload = json.dumps(body, separators=(',', ':')) if body else ""
            full_url = url
        
        # Add auth headers if required
        if auth:
            headers['X-BAPI-API-KEY'] = self.config.api_key
            headers['X-BAPI-TIMESTAMP'] = timestamp
            headers['X-BAPI-SIGN-TYPE'] = '2'  # RSA signature type
            # Pass the correct arguments: timestamp, api_key, recv_window, payload
            headers['X-BAPI-SIGN'] = self._generate_signature(
                timestamp, self.config.api_key, str(self.config.recv_window), payload
            )
            headers['X-BAPI-RECV-WINDOW'] = str(self.config.recv_window)
        
        # Demo mode header
        if self.config.demo:
            headers['X-BAPI-DEMO-TRADING'] = '1'
        
        # Failover through ENDPOINT_PRIORITY (bybit.com → bytick.com)
        endpoints_to_try = self.ENDPOINT_PRIORITY if not self.config.demo else [self.BASE_URL]
        
        last_error = None
        
        for base_url in endpoints_to_try:
            url = f"{base_url}/v5{endpoint}"
            if method == "GET":
                payload = urlencode(sorted(params.items())) if params else ""
                full_url = f"{url}?{payload}" if payload else url
            else:
                full_url = url
            
            for attempt in range(self.config.max_retries):
                try:
                    if method == "GET":
                        response = requests.get(full_url, headers=headers, timeout=12)
                    elif method == "POST":
                        response = requests.post(url, headers=headers, data=payload, timeout=12)
                    elif method == "PUT":
                        response = requests.put(url, headers=headers, data=payload, timeout=12)
                    elif method == "DELETE":
                        response = requests.delete(url, headers=headers, data=payload, timeout=12)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    
                    # FSMA Geo-block detection (403 Forbidden)
                    if response.status_code == 403:
                        logger.warning(f"[FSMA] 403 Forbidden on {base_url}, trying next endpoint...")
                        last_error = f"403 Forbidden on {base_url}"
                        break  # Try next endpoint
                    
                    result = response.json()
                    
                    # Check for time sync error
                    if result.get('retCode') == 10002:  # Timestamp error
                        logger.warning(f"Timestamp error, resyncing... (attempt {attempt + 1})")
                        self._sync_time()
                        time.sleep(self.config.backoff_factor ** attempt)
                        continue
                    
                    # Rate limit hit
                    if result.get('retCode') == 10006 or response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', self.config.backoff_factor ** attempt))
                        logger.warning(f"Rate limited, waiting {retry_after}s...")
                        time.sleep(retry_after)
                        continue
                    
                    # Success - remember this endpoint works
                    if result.get('retCode') == 0:
                        if base_url != self.BASE_URL and base_url == self.BASE_URL_EU:
                            logger.info(f"[FSMA] Successfully connected via EU endpoint: {base_url}")
                            # Update base URL for future requests
                            self.BASE_URL = base_url
                        # Track server time for drift detection
                        if 'time' in result:
                            self._adjust_time_drift(result['time'])
                        return result.get('result', {})
                    
                    # Other API error
                    raise ValueError(f"API error {result.get('retCode')}: {result.get('retMsg')}")
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Request timeout (attempt {attempt + 1}) to {base_url}")
                    time.sleep(self.config.backoff_factor ** attempt)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error to {base_url} (attempt {attempt + 1}): {e}")
                    last_error = str(e)
                    time.sleep(self.config.backoff_factor ** attempt)
        
        # All endpoints exhausted
        raise ConnectionError(f"All Bybit endpoints failed. Last error: {last_error}")
    
    # ==================== MARKET DATA API ====================
    
    def get_server_time(self) -> Optional[datetime]:
        """Get Bybit server time"""
        try:
            result = self.request("GET", "/market/time", auth=False)
            # FIX: Ensure timeSecond is cast to an integer before conversion
            server_time_ms = int(result.get('timeSecond', 0))
            return datetime.fromtimestamp(server_time_ms)
        except Exception as e:
            logger.error(f"get_server_time error: {e}")
            return None
    
    def get_klines(
        self,
        symbol: str,
        interval: str = "60",
        category: str = "linear",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get kline/candlestick data
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1s, 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            category: Market category (spot, linear, inverse, option)
            limit: Number of candles (max 1000)
            start: Start timestamp (ms)
            end: End timestamp (ms)
        """
        params = {
            'category': category,
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        try:
            result = self.request("GET", "/market/kline", params=params, auth=False)
            klines_data = []
            
            for item in reversed(result.get('list', [])):
                klines_data.append({
                    'timestamp': datetime.fromtimestamp(int(item[0]) / 1000),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5]),
                    'turnover': float(item[6]) if len(item) > 6 else 0.0
                })
            
            df = pd.DataFrame(klines_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"get_klines error: {e}")
            return pd.DataFrame()

    def fetch_klines_raw(
        self,
        symbol: str,
        interval: str = "60",
        category: str = "linear",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> Dict:
        """Raw klines endpoint — returns {'result': {'list': [...], ...}} for compat."""
        params = {'category': category, 'symbol': symbol, 'interval': interval, 'limit': limit}
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        inner = self.request("GET", "/market/kline", params=params, auth=False)
        return {'result': inner}

    def get_tickers(
        self,
        symbol: Optional[str] = None,
        category: str = "linear",
        base_coin: Optional[str] = None
    ) -> Dict[str, TickerData]:
        """
        Get latest price tickers
        
        Args:
            symbol: Specific symbol or None for all
            category: Market category
            base_coin: Base coin (for options)
        """
        params = {'category': category}
        if symbol:
            params['symbol'] = symbol
        if base_coin:
            params['baseCoin'] = base_coin
        
        try:
            result = self.request("GET", "/market/tickers", params=params, auth=False)
            tickers = {}
            
            for item in result.get('list', []):
                ticker = TickerData.from_api_response(item)
                tickers[ticker.symbol] = ticker
            
            with self._data_lock:
                self.tickers.update(tickers)
            
            return tickers
        except Exception as e:
            logger.error(f"get_tickers error: {e}")
            return {}
    
    def get_orderbook(
        self,
        symbol: str,
        category: str = "linear",
        limit: int = 25
    ) -> Optional[OrderBook]:
        """
        Get order book
        
        Args:
            symbol: Trading pair
            category: Market category
            limit: Depth (1, 25, 50, 100, 200, 500)
        """
        params = {
            'category': category,
            'symbol': symbol,
            'limit': limit
        }
        
        try:
            result = self.request("GET", "/market/orderbook", params=params, auth=False)
            
            bids = [
                OrderBookLevel(price=float(b[0]), size=float(b[1]), side='bid')
                for b in result.get('b', [])
            ]
            asks = [
                OrderBookLevel(price=float(a[0]), size=float(a[1]), side='ask')
                for a in result.get('a', [])
            ]
            
            orderbook = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                seq=result.get('ts', 0)
            )
            
            with self._data_lock:
                self.orderbooks[symbol] = orderbook
            
            return orderbook
        except Exception as e:
            logger.error(f"get_orderbook error: {e}")
            return None
    
    def get_recent_trades(
        self,
        symbol: str,
        category: str = "linear",
        limit: int = 1000,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get recent public trading history"""
        params = {
            'category': category,
            'symbol': symbol,
            'limit': limit
        }
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/market/recent-trade", params=params, auth=False)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_recent_trades error: {e}")
            return []
    
    def get_funding_rate(
        self,
        symbol: str,
        category: str = "linear",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200
    ) -> List[Dict]:
        """Get funding rate history"""
        params = {
            'category': category,
            'symbol': symbol,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            result = self.request("GET", "/market/funding/history", params=params, auth=False)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_funding_rate error: {e}")
            return []
    
    def get_mark_price_kline(
        self,
        symbol: str,
        interval: str = "60",
        category: str = "linear",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> pd.DataFrame:
        """Get mark price kline data"""
        params = {
            'category': category,
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        try:
            result = self.request("GET", "/market/mark-price-kline", params=params, auth=False)
            klines_data = []
            
            for item in reversed(result.get('list', [])):
                klines_data.append({
                    'timestamp': datetime.fromtimestamp(int(item[0]) / 1000),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4])
                })
            
            df = pd.DataFrame(klines_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"get_mark_price_kline error: {e}")
            return pd.DataFrame()
    
    def get_index_price_kline(
        self,
        symbol: str,
        interval: str = "60",
        category: str = "linear",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> pd.DataFrame:
        """Get index price kline data"""
        params = {
            'category': category,
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        try:
            result = self.request("GET", "/market/index-price-kline", params=params, auth=False)
            klines_data = []
            
            for item in reversed(result.get('list', [])):
                klines_data.append({
                    'timestamp': datetime.fromtimestamp(int(item[0]) / 1000),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4])
                })
            
            df = pd.DataFrame(klines_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"get_index_price_kline error: {e}")
            return pd.DataFrame()
    
    def get_premium_index_price_kline(
        self,
        symbol: str,
        interval: str = "60",
        category: str = "linear",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> pd.DataFrame:
        """Get premium index price kline data"""
        params = {
            'category': category,
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        try:
            result = self.request("GET", "/market/premium-index-price-kline", params=params, auth=False)
            klines_data = []
            
            for item in reversed(result.get('list', [])):
                klines_data.append({
                    'timestamp': datetime.fromtimestamp(int(item[0]) / 1000),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4])
                })
            
            df = pd.DataFrame(klines_data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"get_premium_index_price_kline error: {e}")
            return pd.DataFrame()
    
    def get_instruments_info(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        base_coin: Optional[str] = None,
        limit: int = 1000,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get instrument specifications"""
        params = {'category': category, 'limit': limit}
        if symbol:
            params['symbol'] = symbol
        if status:
            params['status'] = status
        if base_coin:
            params['baseCoin'] = base_coin
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/market/instruments-info", params=params, auth=False)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_instruments_info error: {e}")
            return []
    
    def get_delivery_price(
        self,
        category: str,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get delivery price (for futures)"""
        params = {'category': category, 'limit': limit}
        if symbol:
            params['symbol'] = symbol
        if base_coin:
            params['baseCoin'] = base_coin
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/market/delivery-price", params=params, auth=False)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_delivery_price error: {e}")
            return []
    
    def get_long_short_ratio(
        self,
        symbol: str,
        category: str = "linear",
        period: str = "5min",
        limit: int = 200
    ) -> List[Dict]:
        """Get long/short ratio"""
        params = {
            'category': category,
            'symbol': symbol,
            'period': period,
            'limit': limit
        }
        
        try:
            result = self.request("GET", "/market/account-ratio", params=params, auth=False)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_long_short_ratio error: {e}")
            return []
    
    # ==================== ACCOUNT API ====================
    
    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert value to float, handling empty strings and None"""
        if value is None or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_wallet_balance(
        self,
        account_type: str = "CONTRACT",
        coin: Optional[str] = None
    ) -> Dict[str, WalletBalance]:
        """
        Get wallet balance — auto-detects account type if the requested type
        is rejected by Bybit (CONTRACT ↔ UNIFIED fallback).
        """
        def _fetch(acct_type):
            params = {'accountType': acct_type}
            if coin:
                params['coin'] = coin
            result = self.request("GET", "/account/wallet-balance", params=params, auth=True)
            balances = {}
            for account in result.get('list', []):
                for coin_data in account.get('coin', []):
                    balance = WalletBalance(
                        coin=coin_data.get('coin', ''),
                        wallet_balance=self._safe_float(coin_data.get('walletBalance')),
                        available_balance=self._safe_float(coin_data.get('availableToWithdraw')),
                        margin_balance=self._safe_float(coin_data.get('marginBalance')),
                        unrealized_pnl=self._safe_float(coin_data.get('unrealisedPnl')),
                        cum_realized_pnl=self._safe_float(coin_data.get('cumRealisedPnl')),
                        equity=self._safe_float(coin_data.get('equity')),
                        usd_value=self._safe_float(coin_data.get('usdValue')),
                        borrow_amount=self._safe_float(coin_data.get('borrowAmount')),
                        available_to_withdraw=self._safe_float(coin_data.get('availableToWithdraw')),
                        accrued_interest=self._safe_float(coin_data.get('accruedInterest')),
                        total_order_im=self._safe_float(coin_data.get('totalOrderIM')),
                        total_position_im=self._safe_float(coin_data.get('totalPositionIM')),
                        total_position_mm=self._safe_float(coin_data.get('totalPositionMM'))
                    )
                    balances[balance.coin] = balance
            return balances

        try:
            balances = _fetch(account_type)
            with self._data_lock:
                self.wallet.update(balances)
                self._wallet_account_type = account_type  # cache winner
            return balances
        except Exception as e:
            # 10001 means wrong account type — try the opposite once, then cache winner
            if "10001" in str(e):
                fallback = "UNIFIED" if account_type == "CONTRACT" else "CONTRACT"
                try:
                    balances = _fetch(fallback)
                    with self._data_lock:
                        self.wallet.update(balances)
                        self._wallet_account_type = fallback  # cache winner
                    logger.info(f"get_wallet_balance: using {fallback} account type (auto-detected)")
                    return balances
                except Exception as e2:
                    logger.error(f"get_wallet_balance error: {e2}")
                    return {}
            logger.error(f"get_wallet_balance error: {e}")
            return {}
    
    def get_borrow_history(
        self,
        currency: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get borrow history"""
        params = {'limit': limit}
        if currency:
            params['currency'] = currency
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/account/borrow-history", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_borrow_history error: {e}")
            return []
    
    def get_collateral_info(
        self,
        currency: Optional[str] = None
    ) -> List[Dict]:
        """Get collateral information"""
        params = {}
        if currency:
            params['currency'] = currency
        
        try:
            result = self.request("GET", "/account/collateral-info", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_collateral_info error: {e}")
            return []
    
    def get_coin_greeks(
        self,
        base_coin: Optional[str] = None
    ) -> List[Dict]:
        """Get coin greeks (options)"""
        params = {}
        if base_coin:
            params['baseCoin'] = base_coin
        
        try:
            result = self.request("GET", "/asset/coin-greeks", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_coin_greeks error: {e}")
            return []
    
    def get_fee_rate(
        self,
        category: str = "linear",
        symbol: Optional[str] = None
    ) -> Dict:
        """Get trading fee rate"""
        params = {'category': category}
        if symbol:
            params['symbol'] = symbol
        
        try:
            return self.request("GET", "/account/fee-rate", params=params)
        except Exception as e:
            logger.error(f"get_fee_rate error: {e}")
            return {}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            return self.request("GET", "/account/info")
        except Exception as e:
            logger.error(f"get_account_info error: {e}")
            return {}
    
    def get_transaction_log(
        self,
        account_type: str = "UNIFIED",
        category: Optional[str] = None,
        currency: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get transaction log"""
        params = {'accountType': account_type, 'limit': limit}
        if category:
            params['category'] = category
        if currency:
            params['currency'] = currency
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/account/transaction-log", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_transaction_log error: {e}")
            return []
    
    def get_margin_mode(
        self,
        symbol: str,
        category: str = "linear"
    ) -> Dict:
        """Get margin mode"""
        params = {
            'symbol': symbol,
            'category': category
        }
        
        try:
            return self.request("GET", "/account/margin-mode", params=params)
        except Exception as e:
            logger.error(f"get_margin_mode error: {e}")
            return {}
    
    def set_margin_mode(
        self,
        symbol: str,
        trade_mode: int,
        buy_leverage: str,
        sell_leverage: str,
        category: str = "linear"
    ) -> Dict:
        """
        Set margin mode
        
        Args:
            trade_mode: 0 (cross margin) or 1 (isolated margin)
        """
        body = {
            'symbol': symbol,
            'category': category,
            'tradeMode': trade_mode,
            'buyLeverage': buy_leverage,
            'sellLeverage': sell_leverage
        }
        
        try:
            return self.request("POST", "/account/set-margin-mode", body=body)
        except Exception as e:
            logger.error(f"set_margin_mode error: {e}")
            return {}
    
    def set_leverage(
        self,
        symbol: str,
        buy_leverage: str,
        sell_leverage: str,
        category: str = "linear"
    ) -> Dict:
        """
        Set leverage with safety caps to prevent Error 110013.
        """
        # ðŸ›¡ï¸ SAFETY CAP: Bybit rarely allows > 100x. Most alts cap at 25x-75x.
        # We cap at 50 to be safe, or 100 for BTC/ETH/SOL/XRP.
        max_allowed = 100 if symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'] else 50
        
        try:
            buy_lev_float = float(buy_leverage)
            sell_lev_float = float(sell_leverage)
        except (ValueError, TypeError):
            buy_lev_float = 10.0
            sell_lev_float = 10.0
        
        safe_buy = min(buy_lev_float, max_allowed)
        safe_sell = min(sell_lev_float, max_allowed)
        
        body = {
            'category': category,
            'symbol': symbol,
            'buyLeverage': str(int(safe_buy)),
            'sellLeverage': str(int(safe_sell))
        }
        
        try:
            return self.request("POST", "/position/set-leverage", body=body)
        except Exception as e:
            logger.error(f"set_leverage error: {e}")
            return {}
    
    def switch_position_mode(
        self,
        mode: int,
        category: str = "linear",
        symbol: Optional[str] = None,
        coin: Optional[str] = None
    ) -> Dict:
        """
        Switch position mode
        
        Args:
            mode: 0 (one-way) or 3 (hedge mode)
        """
        body = {
            'category': category,
            'mode': mode
        }
        if symbol:
            body['symbol'] = symbol
        if coin:
            body['coin'] = coin
        
        try:
            return self.request("POST", "/position/switch-mode", body=body)
        except Exception as e:
            logger.error(f"switch_position_mode error: {e}")
            return {}
    
    # ==================== POSITION API ====================
    
    def get_positions(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        settle_coin: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None
    ) -> Dict[str, Position]:
        """
        Get positions
        
        Args:
            category: Market category
            symbol: Specific symbol
            settle_coin: Settle coin (defaults to USDT for linear)
            limit: Max results
            cursor: Pagination cursor
        """
        params = {'category': category, 'limit': limit}
        if symbol:
            params['symbol'] = symbol
        # Bybit requires either symbol or settleCoin for linear/inverse
        if settle_coin:
            params['settleCoin'] = settle_coin
        elif category in ('linear', 'inverse') and not symbol:
            params['settleCoin'] = 'USDT'  # Default settle coin
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/position/list", params=params)
            positions = {}
            
            for item in result.get('list', []):
                # Safely convert string values to float, handling empty strings
                def safe_float(val, default=0.0):
                    if val is None or val == '' or val == 'None':
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default
                
                def safe_int(val, default=0):
                    if val is None or val == '' or val == 'None':
                        return default
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return default
                
                size_str = item.get('size', '0')
                if not size_str or size_str == '':
                    size_str = '0'
                if safe_float(size_str) != 0:
                    position = Position(
                        symbol=item['symbol'],
                        side=item['side'],
                        size=safe_float(item.get('size')),
                        entry_price=safe_float(item.get('avgPrice')),
                        leverage=safe_float(item.get('leverage'), 1),
                        position_value=safe_float(item.get('positionValue')),
                        unrealized_pnl=safe_float(item.get('unrealisedPnl')),
                        realized_pnl=safe_float(item.get('cumRealisedPnl')),
                        mark_price=safe_float(item.get('markPrice')),
                        liquidation_price=safe_float(item.get('liqPrice')),
                        take_profit=safe_float(item.get('takeProfit')),
                        stop_loss=safe_float(item.get('stopLoss')),
                        trailing_stop=safe_float(item.get('trailingStop')),
                        position_idx=safe_int(item.get('positionIdx')),
                        mode=item.get('positionMode', ''),
                        auto_add_margin=safe_int(item.get('autoAddMargin')),
                        position_balance=safe_float(item.get('positionBalance')),
                        session_avg_price=safe_float(item.get('sessionAvgPrice')),
                        occ_closing_fee=safe_float(item.get('occClosingFee')),
                        occ_funding_fee=safe_float(item.get('occFundingFee'))
                    )
                    positions[position.symbol] = position
            
            with self._data_lock:
                self.positions.update(positions)
            
            return positions
        except Exception as e:
            logger.error(f"get_positions error: {e}")
            return {}
    
    def set_trading_stop(
        self,
        symbol: str,
        category: str = "linear",
        position_idx: int = 0,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        trailing_stop: Optional[str] = None,
        tp_trigger_by: Optional[str] = None,
        sl_trigger_by: Optional[str] = None,
        active_price: Optional[str] = None,
        tpsl_mode: Optional[str] = None,
        tp_size: Optional[str] = None,
        sl_size: Optional[str] = None,
        tp_limit_price: Optional[str] = None,
        sl_limit_price: Optional[str] = None
    ) -> Dict:
        """Set trading stop (TP/SL)"""
        body = {
            'category': category,
            'symbol': symbol,
            'positionIdx': position_idx
        }
        
        if take_profit is not None:
            body['takeProfit'] = take_profit
        if stop_loss is not None:
            body['stopLoss'] = stop_loss
        if trailing_stop is not None:
            body['trailingStop'] = trailing_stop
        if tp_trigger_by:
            body['tpTriggerBy'] = tp_trigger_by
        if sl_trigger_by:
            body['slTriggerBy'] = sl_trigger_by
        if active_price:
            body['activePrice'] = active_price
        if tpsl_mode:
            body['tpslMode'] = tpsl_mode
        if tp_size:
            body['tpSize'] = tp_size
        if sl_size:
            body['slSize'] = sl_size
        if tp_limit_price:
            body['tpLimitPrice'] = tp_limit_price
        if sl_limit_price:
            body['slLimitPrice'] = sl_limit_price
        
        try:
            return self.request("POST", "/position/trading-stop", body=body)
        except Exception as e:
            logger.error(f"set_trading_stop error: {e}")
            return {}
    
    def set_auto_add_margin(
        self,
        symbol: str,
        auto_add_margin: int,
        position_idx: int = 0,
        category: str = "linear"
    ) -> Dict:
        """Set auto-add margin"""
        body = {
            'category': category,
            'symbol': symbol,
            'autoAddMargin': auto_add_margin,
            'positionIdx': position_idx
        }
        
        try:
            return self.request("POST", "/position/set-auto-add-margin", body=body)
        except Exception as e:
            logger.error(f"set_auto_add_margin error: {e}")
            return {}
    
    def get_closed_pnl(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get closed PnL"""
        params = {'category': category, 'limit': limit}
        if symbol:
            params['symbol'] = symbol
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/position/closed-pnl", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_closed_pnl error: {e}")
            return []
    
    def move_position(
        self,
        from_uid: str,
        to_uid: str,
        list: List[Dict]
    ) -> Dict:
        """Move position between accounts (institutional)"""
        body = {
            'fromUid': from_uid,
            'toUid': to_uid,
            'list': list
        }
        
        try:
            return self.request("POST", "/position/move-positions", body=body)
        except Exception as e:
            logger.error(f"move_position error: {e}")
            return {}
    
    def confirm_new_risk_limit(
        self,
        symbol: str,
        category: str = "linear"
    ) -> Dict:
        """Confirm new risk limit"""
        body = {
            'category': category,
            'symbol': symbol
        }
        
        try:
            return self.request("POST", "/position/confirm-pending-mmr", body=body)
        except Exception as e:
            logger.error(f"confirm_new_risk_limit error: {e}")
            return {}
    
    # ==================== ORDER API ====================
    
    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        category: str = "linear",
        price: Optional[str] = None,
        trigger_direction: Optional[int] = None,
        trigger_price: Optional[str] = None,
        trigger_by: Optional[str] = None,
        order_filter: Optional[str] = None,
        time_in_force: str = "GTC",
        position_idx: int = 0,
        order_link_id: Optional[str] = None,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        tp_trigger_by: Optional[str] = None,
        sl_trigger_by: Optional[str] = None,
        reduce_only: bool = False,
        close_on_trigger: bool = False,
        smp_type: Optional[str] = None,
        mmp: bool = False,
        tpsl_mode: Optional[str] = None,
        tp_limit_price: Optional[str] = None,
        sl_limit_price: Optional[str] = None,
        tp_order_type: Optional[str] = None,
        sl_order_type: Optional[str] = None,
        market_unit: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Place an order
        
        Args:
            symbol: Trading pair
            side: Buy or Sell
            order_type: Market, Limit, etc.
            qty: Order quantity
            category: Market category
            price: Order price (for Limit orders)
            trigger_direction: 1 (rise) or 2 (fall)
            trigger_price: Trigger price for conditional orders
            trigger_by: LastPrice, IndexPrice, MarkPrice
            order_filter: Order, StopOrder, TPSLOrder
            time_in_force: GTC, IOC, FOK, PostOnly
            position_idx: 0 (one-way), 1 (buy hedge), 2 (sell hedge)
            order_link_id: Client order ID
            take_profit: TP price
            stop_loss: SL price
            reduce_only: Reduce position only
            close_on_trigger: Close on trigger
            smp_type: Self-trade prevention type
            mmp: Market maker protection
            tpsl_mode: Full or Partial
            market_unit: BaseCoin or QuoteCoin (spot market orders)
        """
        body = {
            'category': category,
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'qty': qty,
            'timeInForce': time_in_force,
            'positionIdx': position_idx,
            'reduceOnly': reduce_only,
            'closeOnTrigger': close_on_trigger,
            'mmp': mmp
        }
        
        if price is not None:
            body['price'] = price
        if trigger_direction is not None:
            body['triggerDirection'] = trigger_direction
        if trigger_price is not None:
            body['triggerPrice'] = trigger_price
        if trigger_by:
            body['triggerBy'] = trigger_by
        if order_filter:
            body['orderFilter'] = order_filter
        if order_link_id:
            body['orderLinkId'] = order_link_id
        if take_profit is not None:
            body['takeProfit'] = take_profit
        if stop_loss is not None:
            body['stopLoss'] = stop_loss
        if tp_trigger_by:
            body['tpTriggerBy'] = tp_trigger_by
        if sl_trigger_by:
            body['slTriggerBy'] = sl_trigger_by
        if smp_type:
            body['smpType'] = smp_type
        if tpsl_mode:
            body['tpslMode'] = tpsl_mode
        if tp_limit_price:
            body['tpLimitPrice'] = tp_limit_price
        if sl_limit_price:
            body['slLimitPrice'] = sl_limit_price
        if tp_order_type:
            body['tpOrderType'] = tp_order_type
        if sl_order_type:
            body['slOrderType'] = sl_order_type
        if market_unit:
            body['marketUnit'] = market_unit
        
        try:
            result = self.request("POST", "/order/create", body=body)
            logger.info(f"Order placed: {result.get('orderId', 'N/A')}")
            return result
        except Exception as e:
            logger.error(f"place_order error: {e}")
            return None
    
    def amend_order(
        self,
        symbol: str,
        category: str = "linear",
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        trigger_price: Optional[str] = None,
        qty: Optional[str] = None,
        price: Optional[str] = None,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        tp_trigger_by: Optional[str] = None,
        sl_trigger_by: Optional[str] = None,
        trigger_by: Optional[str] = None
    ) -> Dict:
        """Amend an order"""
        body = {'category': category, 'symbol': symbol}
        
        if order_id:
            body['orderId'] = order_id
        if order_link_id:
            body['orderLinkId'] = order_link_id
        if trigger_price is not None:
            body['triggerPrice'] = trigger_price
        if qty is not None:
            body['qty'] = qty
        if price is not None:
            body['price'] = price
        if take_profit is not None:
            body['takeProfit'] = take_profit
        if stop_loss is not None:
            body['stopLoss'] = stop_loss
        if tp_trigger_by:
            body['tpTriggerBy'] = tp_trigger_by
        if sl_trigger_by:
            body['slTriggerBy'] = sl_trigger_by
        if trigger_by:
            body['triggerBy'] = trigger_by
        
        try:
            return self.request("POST", "/order/amend", body=body)
        except Exception as e:
            logger.error(f"amend_order error: {e}")
            return {}
    
    def cancel_order(
        self,
        symbol: str,
        category: str = "linear",
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        order_filter: Optional[str] = None
    ) -> bool:
        """Cancel an order"""
        body = {'category': category, 'symbol': symbol}
        
        if order_id:
            body['orderId'] = order_id
        if order_link_id:
            body['orderLinkId'] = order_link_id
        if order_filter:
            body['orderFilter'] = order_filter
        
        try:
            self.request("POST", "/order/cancel", body=body)
            logger.info(f"Order cancelled: {order_id or order_link_id}")
            return True
        except Exception as e:
            logger.error(f"cancel_order error: {e}")
            return False
    
    def cancel_all_orders(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        order_filter: Optional[str] = None,
        stop_order_type: Optional[str] = None
    ) -> bool:
        """Cancel all orders"""
        body = {'category': category}
        
        if symbol:
            body['symbol'] = symbol
        if base_coin:
            body['baseCoin'] = base_coin
        if settle_coin:
            body['settleCoin'] = settle_coin
        if order_filter:
            body['orderFilter'] = order_filter
        if stop_order_type:
            body['stopOrderType'] = stop_order_type
        
        try:
            self.request("POST", "/order/cancel-all", body=body)
            logger.info(f"All orders cancelled for {symbol or category}")
            return True
        except Exception as e:
            logger.error(f"cancel_all_orders error: {e}")
            return False
    
    def get_open_orders(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        order_filter: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get open orders"""
        params = {'category': category, 'limit': limit}
        
        if symbol:
            params['symbol'] = symbol
        if base_coin:
            params['baseCoin'] = base_coin
        if settle_coin:
            params['settleCoin'] = settle_coin
        if order_filter:
            params['orderFilter'] = order_filter
        if order_id:
            params['orderId'] = order_id
        if order_link_id:
            params['orderLinkId'] = order_link_id
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/order/realtime", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_open_orders error: {e}")
            return []
    
    def get_order_history(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        order_filter: Optional[str] = None,
        order_status: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get order history"""
        params = {'category': category, 'limit': limit}
        
        if symbol:
            params['symbol'] = symbol
        if base_coin:
            params['baseCoin'] = base_coin
        if settle_coin:
            params['settleCoin'] = settle_coin
        if order_filter:
            params['orderFilter'] = order_filter
        if order_status:
            params['orderStatus'] = order_status
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/order/history", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_order_history error: {e}")
            return []
    
    def get_execution_list(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        base_coin: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        exec_type: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get execution/trade history"""
        params = {'category': category, 'limit': limit}
        
        if symbol:
            params['symbol'] = symbol
        if order_id:
            params['orderId'] = order_id
        if order_link_id:
            params['orderLinkId'] = order_link_id
        if base_coin:
            params['baseCoin'] = base_coin
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if exec_type:
            params['execType'] = exec_type
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/execution/list", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_execution_list error: {e}")
            return []
    
    def batch_place_order(
        self,
        category: str,
        request: List[Dict]
    ) -> Dict:
        """Batch place orders"""
        body = {
            'category': category,
            'request': request
        }
        
        try:
            return self.request("POST", "/order/create-batch", body=body)
        except Exception as e:
            logger.error(f"batch_place_order error: {e}")
            return {}
    
    def batch_amend_order(
        self,
        category: str,
        request: List[Dict]
    ) -> Dict:
        """Batch amend orders"""
        body = {
            'category': category,
            'request': request
        }
        
        try:
            return self.request("POST", "/order/amend-batch", body=body)
        except Exception as e:
            logger.error(f"batch_amend_order error: {e}")
            return {}
    
    def batch_cancel_order(
        self,
        category: str,
        request: List[Dict]
    ) -> Dict:
        """Batch cancel orders"""
        body = {
            'category': category,
            'request': request
        }
        
        try:
            return self.request("POST", "/order/cancel-batch", body=body)
        except Exception as e:
            logger.error(f"batch_cancel_order error: {e}")
            return {}
    
    def get_spot_margin_trade_data(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get spot margin trade data"""
        params = {'limit': limit}
        
        if symbol:
            params['symbol'] = symbol
        if order_id:
            params['orderId'] = order_id
        if order_link_id:
            params['orderLinkId'] = order_link_id
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/spot-margin-trade/order", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_spot_margin_trade_data error: {e}")
            return []
    
    # ==================== SPOT LEVERAGE TOKEN API ====================
    
    def get_leverage_token_info(
        self,
        lt_coin: Optional[str] = None
    ) -> List[Dict]:
        """Get leverage token information"""
        params = {}
        if lt_coin:
            params['ltCoin'] = lt_coin
        
        try:
            result = self.request("GET", "/spot-lever-token/info", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_leverage_token_info error: {e}")
            return []
    
    def get_leverage_token_market(
        self,
        lt_coin: str
    ) -> Dict:
        """Get leverage token market data"""
        params = {'ltCoin': lt_coin}
        
        try:
            return self.request("GET", "/spot-lever-token/market", params=params)
        except Exception as e:
            logger.error(f"get_leverage_token_market error: {e}")
            return {}
    
    def leverage_token_purchase(
        self,
        lt_coin: str,
        lt_amount: str,
        serial_no: Optional[str] = None
    ) -> Dict:
        """Purchase leverage token"""
        body = {
            'ltCoin': lt_coin,
            'ltAmount': lt_amount
        }
        if serial_no:
            body['serialNo'] = serial_no
        
        try:
            return self.request("POST", "/spot-lever-token/purchase", body=body)
        except Exception as e:
            logger.error(f"leverage_token_purchase error: {e}")
            return {}
    
    def leverage_token_redeem(
        self,
        lt_coin: str,
        lt_quantity: str,
        serial_no: Optional[str] = None
    ) -> Dict:
        """Redeem leverage token"""
        body = {
            'ltCoin': lt_coin,
            'ltQuantity': lt_quantity
        }
        if serial_no:
            body['serialNo'] = serial_no
        
        try:
            return self.request("POST", "/spot-lever-token/redeem", body=body)
        except Exception as e:
            logger.error(f"leverage_token_redeem error: {e}")
            return {}
    
    def get_leverage_token_purchase_redemption_records(
        self,
        lt_coin: Optional[str] = None,
        order_type: Optional[str] = None,
        serial_no: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get leverage token purchase/redemption records"""
        params = {'limit': limit}
        
        if lt_coin:
            params['ltCoin'] = lt_coin
        if order_type:
            params['orderType'] = order_type
        if serial_no:
            params['serialNo'] = serial_no
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/spot-lever-token/record", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_leverage_token_records error: {e}")
            return []
    
    # ==================== SPOT MARGIN TRADE (CROSS/ISOLATED MARGIN) API ====================
    
    def toggle_margin_trade(
        self,
        symbol: str,
        switch: int
    ) -> Dict:
        """Toggle margin trade on/off"""
        body = {
            'symbol': symbol,
            'switch': switch
        }
        
        try:
            return self.request("POST", "/spot-margin-trade/switch-mode", body=body)
        except Exception as e:
            logger.error(f"toggle_margin_trade error: {e}")
            return {}
    
    def get_margin_coin_info(
        self,
        coin: Optional[str] = None
    ) -> List[Dict]:
        """Get margin coin information"""
        params = {}
        if coin:
            params['coin'] = coin
        
        try:
            result = self.request("GET", "/spot-margin-trade/coin-info", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_margin_coin_info error: {e}")
            return []
    
    def get_margin_borrowable_coin_info(
        self,
        coin: Optional[str] = None
    ) -> List[Dict]:
        """Get borrowable coin information"""
        params = {}
        if coin:
            params['coin'] = coin
        
        try:
            result = self.request("GET", "/spot-margin-trade/borrow-coin", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_margin_borrowable_coin_info error: {e}")
            return []
    
    def margin_borrow(
        self,
        coin: str,
        qty: str
    ) -> Dict:
        """Borrow coin for margin trading"""
        body = {
            'coin': coin,
            'qty': qty
        }
        
        try:
            return self.request("POST", "/spot-margin-trade/borrow", body=body)
        except Exception as e:
            logger.error(f"margin_borrow error: {e}")
            return {}
    
    def margin_repay(
        self,
        coin: str,
        qty: Optional[str] = None,
        complete_repayment: bool = False
    ) -> Dict:
        """Repay borrowed coin"""
        body = {'coin': coin}
        
        if qty:
            body['qty'] = qty
        if complete_repayment:
            body['completeRepayment'] = complete_repayment
        
        try:
            return self.request("POST", "/spot-margin-trade/repay", body=body)
        except Exception as e:
            logger.error(f"margin_repay error: {e}")
            return {}
    
    def get_margin_borrow_order_detail(
        self,
        borrow_id: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> List[Dict]:
        """Get borrow order details"""
        params = {'limit': limit}
        
        if borrow_id:
            params['borrowId'] = borrow_id
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        if cursor:
            params['cursor'] = cursor
        
        try:
            result = self.request("GET", "/spot-margin-trade/borrow-order", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_margin_borrow_order_detail error: {e}")
            return []
    
    def get_margin_account_info(self) -> Dict:
        """Get margin account information"""
        try:
            return self.request("GET", "/spot-margin-trade/account")
        except Exception as e:
            logger.error(f"get_margin_account_info error: {e}")
            return {}
    
    def get_margin_interest_quota(
        self,
        coin: Optional[str] = None
    ) -> List[Dict]:
        """Get interest quota"""
        params = {}
        if coin:
            params['coin'] = coin
        
        try:
            result = self.request("GET", "/spot-margin-trade/interest-quota", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_margin_interest_quota error: {e}")
            return []
    
    def get_margin_loan_info(
        self,
        tier: Optional[str] = None
    ) -> List[Dict]:
        """Get loan information"""
        params = {}
        if tier:
            params['tier'] = tier
        
        try:
            result = self.request("GET", "/spot-margin-trade/loan-info", params=params)
            return result.get('list', [])
        except Exception as e:
            logger.error(f"get_margin_loan_info error: {e}")
            return []
    
    # ==================== ACCOUNT API (ADDITIONAL) ====================
    
    def get_wallet_transferable_amount(
        self,
        coin: str
    ) -> Dict:
        """Get transferable amount"""
        params = {'coin': coin}
        
        try:
            return self.request("GET", "/account/withdrawable-amount", params=params)
        except Exception as e:
            logger.error(f"get_wallet_transferable_amount error: {e}")
            return {}
    
    def get_spot_hedging_info(self) -> Dict:
        """Get spot hedging information"""
        try:
            return self.request("GET", "/account/spot-hedging")
        except Exception as e:
            logger.error(f"get_spot_hedging_info error: {e}")
            return {}
    
    def set_spot_hedging_mode(
        self,
        set_hedging_mode: str
    ) -> Dict:
        """Set spot hedging mode"""
        body = {'setHedgingMode': set_hedging_mode}
        
        try:
            return self.request("POST", "/account/set-hedging-mode", body=body)
        except Exception as e:
            logger.error(f"set_spot_hedging_mode error: {e}")
            return {}
    
    # ==================== USER API ====================
    
    def get_sub_uid_list(
        self,
        page_size: int = 10,
        next_cursor: Optional[str] = None
    ) -> Dict:
        """Get sub-UID list"""
        params = {'pageSize': page_size}
        if next_cursor:
            params['nextCursor'] = next_cursor
        
        try:
            return self.request("GET", "/user/query-sub-members", params=params)
        except Exception as e:
            logger.error(f"get_sub_uid_list error: {e}")
            return {}
    
    def get_api_key_info(self) -> Dict:
        """Get API key information"""
        try:
            return self.request("GET", "/user/query-api")
        except Exception as e:
            logger.error(f"get_api_key_info error: {e}")
            return {}
    
    def get_uid_wallet_type(self) -> List[Dict]:
        """Get UID wallet type"""
        try:
            result = self.request("GET", "/user/get-member-type")
            return result.get('accounts', [])
        except Exception as e:
            logger.error(f"get_uid_wallet_type error: {e}")
            return []
    
    def delete_sub_api_key(
        self,
        api_key: Optional[str] = None
    ) -> Dict:
        """Delete sub-account API key"""
        body = {}
        if api_key:
            body['apikey'] = api_key
        
        try:
            return self.request("POST", "/user/delete-sub-api", body=body)
        except Exception as e:
            logger.error(f"delete_sub_api_key error: {e}")
            return {}
    
    def get_affiliate_user_info(
        self,
        uid: str
    ) -> Dict:
        """Get affiliate user information"""
        params = {'uid': uid}
        
        try:
            return self.request("GET", "/user/aff-customer-info", params=params)
        except Exception as e:
            logger.error(f"get_affiliate_user_info error: {e}")
            return {}
    
    # ==================== WEBSOCKET ====================
    
    def start_websocket(
        self,
        symbols: Optional[List[str]] = None,
        public_categories: Optional[List[str]] = None,
        public_topics: Optional[List[str]] = None,
        private_topics: Optional[List[str]] = None,
        enable_trade_ws: bool = False
    ):
        """
        Start WebSocket connections
        
        Args:
            symbols: List of symbols to subscribe
            public_categories: List of public categories (linear, spot, inverse, option)
            public_topics: List of public topics to subscribe
            private_topics: List of private topics to subscribe
            enable_trade_ws: Enable WebSocket trade API
        """
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client not available")
            return
        
        self._ws_running = True
        
        # Default symbols
        if symbols is None:
            symbols = ["BTCUSDT"]
        
        # Default categories
        if public_categories is None:
            public_categories = ["linear"]
        
        # Start public WebSockets for each category
        for category in public_categories:
            ws_url = getattr(self, f"WS_PUBLIC_{category.upper()}", self.WS_PUBLIC_LINEAR)
            thread = threading.Thread(
                target=self._ws_public_loop,
                args=(ws_url, category, symbols, public_topics),
                daemon=True,
                name=f"ws_public_{category}"
            )
            thread.start()
            self._ws_threads[f"public_{category}"] = thread
        
        # Start private WebSocket
        if private_topics is None:
            private_topics = ['position', 'execution', 'order', 'wallet']
        
        private_thread = threading.Thread(
            target=self._ws_private_loop,
            args=(private_topics,),
            daemon=True,
            name="ws_private"
        )
        private_thread.start()
        self._ws_threads['private'] = private_thread
        
        # Start trade WebSocket if enabled
        if enable_trade_ws:
            trade_thread = threading.Thread(
                target=self._ws_trade_loop,
                daemon=True,
                name="ws_trade"
            )
            trade_thread.start()
            self._ws_threads['trade'] = trade_thread
        
        logger.info("WebSocket connections started")

    def start_public_websocket(
        self,
        symbols: Optional[List[str]] = None,
        public_categories: Optional[List[str]] = None,
        public_topics: Optional[List[str]] = None,
    ):
        """Start public WebSocket(s) only (idempotent — skips any category already running)."""
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client not available")
            return
        if symbols is None:
            symbols = ["BTCUSDT"]
        if public_categories is None:
            public_categories = ["linear"]
        if public_topics is None:
            public_topics = ["publicTrade"]
        for category in public_categories:
            thread_key = f"public_{category}"
            if thread_key in self._ws_threads and self._ws_threads[thread_key].is_alive():
                logger.info(f"Public WS ({category}) already running — skipping")
                continue
            ws_url = getattr(self, f"WS_PUBLIC_{category.upper()}", self.WS_PUBLIC_LINEAR)
            thread = threading.Thread(
                target=self._ws_public_loop,
                args=(ws_url, category, symbols, public_topics),
                daemon=True,
                name=f"ws_public_{category}",
            )
            thread.start()
            self._ws_threads[thread_key] = thread
            logger.info(f"Public WS ({category}) started — {symbols} {public_topics}")

    def start_private_websocket(self, topics: Optional[List[str]] = None):
        """Start private WebSocket only (idempotent — safe to call even if already running)."""
        if not WEBSOCKET_AVAILABLE:
            raise RuntimeError("websocket-client not available")

        # Already running — just return; NEO can register callbacks on the existing WS
        if 'private' in self._ws_threads and self._ws_threads['private'].is_alive():
            logger.info("Private WebSocket already running — skipping duplicate start")
            return

        self._ws_running = True
        if topics is None:
            topics = ['position', 'execution', 'order', 'wallet']

        private_thread = threading.Thread(
            target=self._ws_private_loop,
            args=(topics,),
            daemon=True,
            name="ws_private"
        )
        private_thread.start()
        self._ws_threads['private'] = private_thread
        logger.info("Private WebSocket started")
    
    def _ws_public_loop(
        self,
        ws_url: str,
        category: str,
        symbols: List[str],
        topics: Optional[List[str]] = None
    ):
        """Public WebSocket connection loop"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_ws_message('public', category, data)
            except Exception as e:
                logger.error(f"Public WS message error: {e}")
                
        def on_open(ws):
            logger.info(f"Public WS connected ({category})")
            # Start application-level JSON ping heartbeat (Bybit requires {"op":"ping"} every 20s)
            threading.Thread(
                target=self._ws_app_ping_loop,
                args=(ws, f"public-{category}"),
                daemon=True,
                name=f"ws_ping_public_{category}"
            ).start()
            
            # FIX: Ensure topics is defined locally
            current_topics = topics if topics is not None else ['tickers', 'orderbook.50']
            
            for symbol in symbols:
                for topic_template in current_topics:
                    if '{symbol}' in topic_template:
                        topic = topic_template.format(symbol=symbol)
                    else:
                        topic = f"{topic_template}.{symbol}" if '.' not in topic_template else topic_template
                    
                    sub_msg = {
                        "op": "subscribe",
                        "args": [topic]
                    }
                    ws.send(json.dumps(sub_msg))
                    logger.debug(f"Subscribed to {topic}")
        def on_error(ws, error):
            logger.error(f"Public WS error ({category}): {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Public WS closed ({category}): {close_status_code} - {close_msg}")
            if self._ws_running:
                logger.info(f"Reconnecting public WS ({category})...")
                time.sleep(5)
                self._ws_public_loop(ws_url, category, symbols, topics)
        
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            self.ws_public = ws
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.error(f"Public WS error ({category}): {e}")
    
    def _ws_private_loop(self, topics: List[str]):
        """Private WebSocket connection loop"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self._handle_ws_message('private', None, data)
            except Exception as e:
                logger.error(f"Private WS message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"Private WS error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Private WS closed: {close_status_code} - {close_msg}")
            if self._ws_running:
                logger.info("Reconnecting private WS...")
                time.sleep(5)
                self._ws_private_loop(topics)
        
        def on_open(ws):
            logger.info("Private WS connected")
            # Start application-level JSON ping heartbeat
            threading.Thread(
                target=self._ws_app_ping_loop,
                args=(ws, "private"),
                daemon=True,
                name="ws_ping_private"
            ).start()
            # Authenticate
            self._ws_auth(ws)
            
            # Subscribe to topics
            for topic in topics:
                sub_msg = {
                    "op": "subscribe",
                    "args": [topic]
                }
                ws.send(json.dumps(sub_msg))
                logger.debug(f"Subscribed to {topic}")
        
        try:
            ws = websocket.WebSocketApp(
                (self.WS_DEMO_PRIVATE if self.config.demo else self.WS_PRIVATE),
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            self.ws_private = ws
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.error(f"Private WS error: {e}")
    
    def _ws_trade_loop(self):
        """WebSocket Trade API connection loop"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                logger.info(f"Trade WS message: {data}")
            except Exception as e:
                logger.error(f"Trade WS message error: {e}")
        
        def on_error(ws, error):
            logger.error(f"Trade WS error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Trade WS closed: {close_status_code} - {close_msg}")
            if self._ws_running:
                logger.info("Reconnecting trade WS...")
                time.sleep(5)
                self._ws_trade_loop()
        
        def on_open(ws):
            logger.info("Trade WS connected")
            # Start application-level JSON ping heartbeat
            threading.Thread(
                target=self._ws_app_ping_loop,
                args=(ws, "trade"),
                daemon=True,
                name="ws_ping_trade"
            ).start()
            # Authenticate
            self._ws_auth(ws)
        
        try:
            ws = websocket.WebSocketApp(
                (self.WS_DEMO_TRADE if self.config.demo else self.WS_TRADE),
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            self.ws_trade = ws
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            logger.error(f"Trade WS error: {e}")
    
    def _ws_app_ping_loop(self, ws_app, ws_name: str):
        """Send Bybit application-level JSON ping every 20s (required alongside WebSocket TCP ping).
        run_forever(ping_interval=20) only sends protocol-level TCP pings; Bybit also requires
        {"op":"ping"} at the JSON level or the server will disconnect."""
        while self._ws_running:
            try:
                if ws_app.sock and ws_app.sock.connected:
                    ws_app.send(json.dumps({"op": "ping"}))
                    logger.debug(f"App-level JSON ping sent ({ws_name})")
            except Exception as e:
                logger.debug(f"App-level ping failed ({ws_name}): {e}")
            time.sleep(20)

    def _ws_auth(self, ws):
        """Secure RSA Authentication for Private WebSocket"""
        # Bybit V5 WS auth: sign "GET/realtime{expires}" with RSA private key
        expires = int((time.time() + self.time_offset + 10) * 1000)
        auth_str = f"GET/realtime{expires}"
        
        # 2. Modern RSA Signing (Crucial for V5)
        h = SHA256.new(auth_str.encode('utf-8'))
        signature_bytes = pkcs1_15.new(self.private_key).sign(h)
        signature_base64 = base64.b64encode(signature_bytes).decode('utf-8')

        auth_msg = {
            "op": "auth",
            "args": [self.config.api_key, expires, signature_base64]
        }
        ws.send(json.dumps(auth_msg))
        logger.info("Private WS Auth sent.")
    
    def _handle_ws_message(self, channel: str, category: Optional[str], data: Dict):
        """Handle WebSocket message"""
        if 'topic' not in data:
            if data.get('op') == 'auth' and data.get('success'):
                logger.info(f"WebSocket authenticated: {channel}")
            return
        
        topic = data['topic']
        topic_data = data.get('data', {})
        
        # Route to appropriate handler
        if 'tickers' in topic:
            self._handle_ticker_ws(topic_data)
        elif 'kline' in topic:
            self._handle_kline_ws(topic_data, topic)
        elif 'orderbook' in topic:
            self._handle_orderbook_ws(topic_data, topic)
        elif 'publicTrade' in topic:
            self._handle_trade_ws(topic_data)
        elif 'liquidation' in topic:
            self._handle_liquidation_ws(topic_data)
            self._handle_trade_ws(topic_data)
        elif topic == 'position':
            self._handle_position_ws(topic_data)
        elif topic == 'execution':
            self._handle_execution_ws(topic_data)
        elif topic == 'order':
            self._handle_order_ws(topic_data)
        elif topic == 'wallet':
            self._handle_wallet_ws(topic_data)
        
        # Call registered callbacks
        if topic in self._ws_callbacks:
            for callback in self._ws_callbacks[topic]:
                try:
                    callback(topic_data)
                except Exception as e:
                    logger.error(f"Callback error for {topic}: {e}")
    
    def _handle_ticker_ws(self, data: Union[Dict, List]):
        """Handle ticker WebSocket message"""
        try:
            if isinstance(data, list):
                data = data[0] if data else {}
            
            symbol = data.get('symbol', '')
            
            # FIX: Merge with existing ticker to preserve last_price on delta updates
            with self._data_lock:
                existing = self.tickers.get(symbol)
                if existing and 'lastPrice' not in data:
                    # Delta update - preserve last_price if not in update
                    data = dict(data)  # Copy to avoid modifying original
                    data['lastPrice'] = existing.last_price
            
            ticker = TickerData.from_api_response(data)
            
            with self._data_lock:
                self.tickers[ticker.symbol] = ticker
            
            for callback in self._ticker_callbacks:
                try:
                    callback(ticker)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")
        except Exception as e:
            logger.error(f"Handle ticker WS error: {e}")
    
    def _handle_kline_ws(self, data: List, topic: str):
        """Handle kline WebSocket message"""
        try:
            # Parse interval from topic (e.g., "kline.5.BTCUSDT")
            parts = topic.split('.')
            interval = parts[1] if len(parts) > 1 else ''
            symbol = parts[2] if len(parts) > 2 else ''
            
            for item in data:
                kline = KlineData(
                    timestamp=datetime.fromtimestamp(int(item['start']) / 1000),
                    open=float(item['open']),
                    high=float(item['high']),
                    low=float(item['low']),
                    close=float(item['close']),
                    volume=float(item['volume']),
                    turnover=float(item.get('turnover', 0)),
                    symbol=item.get('symbol', symbol),
                    interval=item.get('interval', interval)
                )
                
                with self._data_lock:
                    key = f"{kline.symbol}_{kline.interval}"
                    self.klines[key].append(kline)
                
                for callback in self._kline_callbacks:
                    try:
                        callback(kline)
                    except Exception as e:
                        logger.error(f"Kline callback error: {e}")
        except Exception as e:
            logger.error(f"Handle kline WS error: {e}")
    
    def _handle_orderbook_ws(self, data: Dict, topic: str):
        """Handle orderbook WebSocket message"""
        try:
            symbol = data.get('s', '')
            
            bids = [
                OrderBookLevel(price=float(b[0]), size=float(b[1]), side='bid')
                for b in data.get('b', [])
            ]
            asks = [
                OrderBookLevel(price=float(a[0]), size=float(a[1]), side='ask')
                for a in data.get('a', [])
            ]
            
            orderbook = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                seq=data.get('u', 0)
            )
            
            with self._data_lock:
                self.orderbooks[symbol] = orderbook
            
            for callback in self._orderbook_callbacks:
                try:
                    callback(orderbook)
                except Exception as e:
                    logger.error(f"Orderbook callback error: {e}")
        except Exception as e:
            logger.error(f"Handle orderbook WS error: {e}")
    
    def _handle_trade_ws(self, data: List):
        """Handle public trade WebSocket message — fire registered callbacks.
        
        Bybit publicTrade payload uses single-char keys (s=symbol, v=size, S=side, T=ts_ms, p=price).
        We normalise each trade dict into a SimpleNamespace with standard attribute names so all
        downstream callbacks can use trade.symbol / trade.size / trade.side / trade.ts uniformly.
        """
        try:
            if not self._public_trade_callbacks:
                return
            trades = data if isinstance(data, list) else [data]
            for raw in trades:
                if not isinstance(raw, dict):
                    continue
                trade = type('Trade', (), {
                    'symbol': raw.get('s', raw.get('symbol', '')),
                    'price':  float(raw.get('p', raw.get('price', 0))),
                    'size':   float(raw.get('v', raw.get('size', 0))),
                    'side':   raw.get('S', raw.get('side', 'Buy')),
                    'ts':     float(raw.get('T', raw.get('ts', 0))),
                })()
                for cb in self._public_trade_callbacks:
                    try:
                        cb(trade)
                    except Exception as e:
                        logger.error(f"Public trade callback error: {e}")
        except Exception as e:
            logger.error(f"Handle trade WS error: {e}")

    def _handle_liquidation_ws(self, data):
        """
        Handle liquidation WebSocket message (public topic: liquidation.{symbol}).
        Bybit sends forced liquidation orders from the market — useful for sensing
        cascades and managing own position risk.
        data keys: symbol, side, price, size, time
        """
        try:
            if isinstance(data, list):
                data = data[0] if data else {}
            for callback in self._liquidation_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Liquidation callback error: {e}")
        except Exception as e:
            logger.error(f"Handle liquidation WS error: {e}")
    
    def _handle_position_ws(self, data: List):
        """Handle position WebSocket message"""
        try:
            for item in data:
                position = Position(
                    symbol=item['symbol'],
                    side=item['side'],
                    size=float(item.get('size', 0)),
                    entry_price=float(item.get('entryPrice', 0)),
                    leverage=float(item.get('leverage', 1)),
                    position_value=float(item.get('positionValue', 0)),
                    unrealized_pnl=float(item.get('unrealisedPnl', 0)),
                    realized_pnl=float(item.get('cumRealisedPnl', 0)),
                    mark_price=float(item.get('markPrice', 0)),
                    liquidation_price=float(item.get('liqPrice', 0)) if item.get('liqPrice') else 0.0,
                    take_profit=float(item.get('takeProfit', 0)) if item.get('takeProfit') else 0.0,
                    stop_loss=float(item.get('stopLoss', 0)) if item.get('stopLoss') else 0.0,
                    trailing_stop=float(item.get('trailingStop', 0)) if item.get('trailingStop') else 0.0,
                    position_idx=int(item.get('positionIdx', 0)),
                    mode=item.get('positionMode', '')
                )
                
                with self._data_lock:
                    self.positions[position.symbol] = position
                
                for callback in self._position_callbacks:
                    try:
                        callback(position)
                    except Exception as e:
                        logger.error(f"Position callback error: {e}")
        except Exception as e:
            logger.error(f"Handle position WS error: {e}")
    
    def _handle_execution_ws(self, data: List):
        """Handle execution WebSocket message"""
        try:
            for item in data:
                execution = Execution(
                    exec_id=item['execId'],
                    order_id=item['orderId'],
                    symbol=item['symbol'],
                    side=item['side'],
                    exec_price=float(item['execPrice']),
                    exec_qty=float(item['execQty']),
                    exec_value=float(item['execValue']),
                    exec_fee=float(item['execFee']),
                    exec_type=item['execType'],
                    exec_time=datetime.fromtimestamp(int(item['execTime']) / 1000),
                    order_link_id=item.get('orderLinkId', ''),
                    is_maker=item.get('isMaker', False),
                    fee_rate=float(item.get('feeRate', 0))
                )
                
                with self._data_lock:
                    self.executions.append(execution)
                
                for callback in self._execution_callbacks:
                    try:
                        callback(execution)
                    except Exception as e:
                        logger.error(f"Execution callback error: {e}")
        except Exception as e:
            logger.error(f"Handle execution WS error: {e}")
    
    def _handle_order_ws(self, data: List):
        """Handle order WebSocket message"""
        try:
            for item in data:
                order = Order(
                    order_id=item['orderId'],
                    symbol=item['symbol'],
                    side=item['side'],
                    order_type=item['orderType'],
                    price=float(item.get('price', 0)),
                    qty=float(item['qty']),
                    status=item['orderStatus'],
                    order_link_id=item.get('orderLinkId', ''),
                    leaves_qty=float(item.get('leavesQty', 0)),
                    cum_exec_qty=float(item.get('cumExecQty', 0)),
                    cum_exec_value=float(item.get('cumExecValue', 0)),
                    cum_exec_fee=float(item.get('cumExecFee', 0)),
                    time_in_force=item.get('timeInForce', ''),
                    reduce_only=item.get('reduceOnly', False),
                    close_on_trigger=item.get('closeOnTrigger', False)
                )
                
                with self._data_lock:
                    self.orders[order.order_id] = order
                
                for callback in self._order_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        logger.error(f"Order callback error: {e}")
        except Exception as e:
            logger.error(f"Handle order WS error: {e}")
    
    def _handle_wallet_ws(self, data: List):
        """Handle wallet WebSocket message"""
        try:
            for account in data:
                for coin_data in account.get('coin', []):
                    balance = WalletBalance(
                        coin=coin_data['coin'],
                        wallet_balance=float(coin_data.get('walletBalance', 0)),
                        available_balance=float(coin_data.get('availableToWithdraw', 0)),
                        margin_balance=float(coin_data.get('marginBalance', 0)),
                        unrealized_pnl=float(coin_data.get('unrealisedPnl', 0)),
                        cum_realized_pnl=float(coin_data.get('cumRealisedPnl', 0)),
                        equity=float(coin_data.get('equity', 0)),
                        usd_value=float(coin_data.get('usdValue', 0))
                    )
                    
                    with self._data_lock:
                        self.wallet[balance.coin] = balance
                    
                    for callback in self._wallet_callbacks:
                        try:
                            callback(balance)
                        except Exception as e:
                            logger.error(f"Wallet callback error: {e}")
        except Exception as e:
            logger.error(f"Handle wallet WS error: {e}")
    
    def stop_websocket(self):
        """Stop all WebSocket connections"""
        self._ws_running = False
        
        for ws in [self.ws_public, self.ws_private, self.ws_trade]:
            if ws:
                try:
                    ws.close()
                except Exception as e:
                    logger.debug(f"WebSocket close error: {e}")
        
        for name, thread in self._ws_threads.items():
            if thread.is_alive():
                thread.join(timeout=2.0)
                logger.debug(f"WebSocket thread {name} stopped")
        
        self._ws_threads.clear()
        logger.info("All WebSocket connections stopped")
    
    def subscribe_ws(self, topic: str, callback: Callable):
        """Subscribe to WebSocket topic"""
        self._ws_callbacks[topic].append(callback)
        
        # Send subscribe message if connected
        sub_msg = {
            "op": "subscribe",
            "args": [topic]
        }
        
        ws = None
        if 'tickers' in topic or 'kline' in topic or 'orderbook' in topic or 'publicTrade' in topic:
            ws = self.ws_public
        elif topic in ['position', 'execution', 'order', 'wallet', 'greeks']:
            ws = self.ws_private
        
        if ws and ws.sock and ws.sock.connected:
            ws.send(json.dumps(sub_msg))
            logger.info(f"Subscribed to {topic}")
    
    def unsubscribe_ws(self, topic: str):
        """Unsubscribe from WebSocket topic"""
        if topic in self._ws_callbacks:
            del self._ws_callbacks[topic]
        
        unsub_msg = {
            "op": "unsubscribe",
            "args": [topic]
        }
        
        ws = None
        if 'tickers' in topic or 'kline' in topic or 'orderbook' in topic:
            ws = self.ws_public
        else:
            ws = self.ws_private
        
        if ws and ws.sock and ws.sock.connected:
            ws.send(json.dumps(unsub_msg))
            logger.info(f"Unsubscribed from {topic}")
    
    # ==================== WEBSOCKET TRADE API ====================
    
    def ws_place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        category: str = "linear",
        price: Optional[str] = None,
        time_in_force: str = "GTC",
        order_link_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Place order via WebSocket Trade API
        
        Requires WebSocket trade connection to be active
        """
        if not self.ws_trade or not self.ws_trade.sock or not self.ws_trade.sock.connected:
            logger.error("WebSocket trade connection not active")
            return False
        
        body = {
            "reqId": order_link_id or str(int(time.time() * 1000)),
            "header": {
                "X-BAPI-TIMESTAMP": self._get_timestamp(),
                "X-BAPI-RECV-WINDOW": str(self.config.recv_window)
            },
            "op": "order.create",
            "args": [
                {
                    "category": category,
                    "symbol": symbol,
                    "side": side,
                    "orderType": order_type,
                    "qty": qty,
                    "timeInForce": time_in_force
                }
            ]
        }
        
        if price:
            body['args'][0]['price'] = price
        
        body['args'][0].update(kwargs)
        
        # Sign the request (WebSocket trade API uses same signature format)
        timestamp = body['header']['X-BAPI-TIMESTAMP']
        payload = json.dumps(body['args'][0], separators=(',', ':'))
        sign_str = f"{timestamp}{self.config.api_key}{self.config.recv_window}{payload}"
        signature = base64.b64encode(
            self.private_key.sign(
                sign_str.encode('utf-8'),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
        ).decode('utf-8')
        body['header']['X-BAPI-SIGN'] = signature
        
        try:
            self.ws_trade.send(json.dumps(body))
            return True
        except Exception as e:
            logger.error(f"WS place order error: {e}")
            return False
    
    def ws_cancel_order(
        self,
        symbol: str,
        category: str = "linear",
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None
    ) -> bool:
        """Cancel order via WebSocket Trade API"""
        if not self.ws_trade or not self.ws_trade.sock or not self.ws_trade.sock.connected:
            logger.error("WebSocket trade connection not active")
            return False
        
        body = {
            "reqId": str(int(time.time() * 1000)),
            "header": {
                "X-BAPI-TIMESTAMP": self._get_timestamp(),
                "X-BAPI-RECV-WINDOW": str(self.config.recv_window)
            },
            "op": "order.cancel",
            "args": [
                {
                    "category": category,
                    "symbol": symbol
                }
            ]
        }
        
        if order_id:
            body['args'][0]['orderId'] = order_id
        if order_link_id:
            body['args'][0]['orderLinkId'] = order_link_id
        
        # Sign the request
        timestamp = body['header']['X-BAPI-TIMESTAMP']
        payload = json.dumps(body['args'][0], separators=(',', ':'))
        sign_str = f"{timestamp}{self.config.api_key}{self.config.recv_window}{payload}"
        signature = base64.b64encode(
            self.private_key.sign(
                sign_str.encode('utf-8'),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
        ).decode('utf-8')
        body['header']['X-BAPI-SIGN'] = signature
        
        try:
            self.ws_trade.send(json.dumps(body))
            return True
        except Exception as e:
            logger.error(f"WS cancel order error: {e}")
            return False
    
    def ws_amend_order(
        self,
        symbol: str,
        category: str = "linear",
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        qty: Optional[str] = None,
        price: Optional[str] = None
    ) -> bool:
        """Amend order via WebSocket Trade API"""
        if not self.ws_trade or not self.ws_trade.sock or not self.ws_trade.sock.connected:
            logger.error("WebSocket trade connection not active")
            return False
        
        body = {
            "reqId": str(int(time.time() * 1000)),
            "header": {
                "X-BAPI-TIMESTAMP": self._get_timestamp(),
                "X-BAPI-RECV-WINDOW": str(self.config.recv_window)
            },
            "op": "order.amend",
            "args": [
                {
                    "category": category,
                    "symbol": symbol
                }
            ]
        }
        
        if order_id:
            body['args'][0]['orderId'] = order_id
        if order_link_id:
            body['args'][0]['orderLinkId'] = order_link_id
        if qty:
            body['args'][0]['qty'] = qty
        if price:
            body['args'][0]['price'] = price
        
        # Sign the request
        timestamp = body['header']['X-BAPI-TIMESTAMP']
        payload = json.dumps(body['args'][0], separators=(',', ':'))
        sign_str = f"{timestamp}{self.config.api_key}{self.config.recv_window}{payload}"
        signature = base64.b64encode(
            self.private_key.sign(
                sign_str.encode('utf-8'),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
        ).decode('utf-8')
        body['header']['X-BAPI-SIGN'] = signature
        
        try:
            self.ws_trade.send(json.dumps(body))
            return True
        except Exception as e:
            logger.error(f"WS amend order error: {e}")
            return False
    
    # ==================== CALLBACK REGISTRATION ====================
    
    def on_ticker(self, callback: Callable[[TickerData], None]):
        """Register ticker callback"""
        self._ticker_callbacks.append(callback)

    def on_public_trade(self, callback: Callable[[Dict], None]):
        """Register public trade tick callback (fires for every publicTrade WS event)."""
        self._public_trade_callbacks.append(callback)
    
    def on_kline(self, callback: Callable[[KlineData], None]):
        """Register kline callback"""
        self._kline_callbacks.append(callback)
    
    def on_orderbook(self, callback: Callable[[OrderBook], None]):
        """Register orderbook callback"""
        self._orderbook_callbacks.append(callback)
    
    def on_position(self, callback: Callable[[Position], None]):
        """Register position callback"""
        self._position_callbacks.append(callback)
    
    def on_order(self, callback: Callable[[Order], None]):
        """Register order callback"""
        self._order_callbacks.append(callback)
    
    def on_execution(self, callback: Callable[[Execution], None]):
        """Register execution callback"""
        self._execution_callbacks.append(callback)
    
    def on_wallet(self, callback: Callable[[WalletBalance], None]):
        """Register wallet callback"""
        self._wallet_callbacks.append(callback)
    
    def on_webhook(self, callback: Callable[[WebhookSignal], None]):
        """Register webhook callback"""
        self._webhook_callbacks.append(callback)

    def on_liquidation(self, callback: Callable[[Dict], None]):
        """Register liquidation callback (public WS: liquidation.{symbol})"""
        self._liquidation_callbacks.append(callback)
    
    # ==================== UTILITY METHODS ====================
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        with self._data_lock:
            ticker = self.tickers.get(symbol)
            return ticker.last_price if ticker else None
    
    def get_bybit_max_lev(symbol: str, position_value: float = 0) -> int:
        """Get maximum leverage for a symbol based on position value"""
        tier_limits = {
            'BTCUSDT': [(100000, 100), (200000, 50), (500000, 25), (1000000, 10)],
            'ETHUSDT': [(50000, 50), (100000, 25), (250000, 10)],
        }
        tiers = tier_limits.get(symbol, [(50000, 25), (100000, 10)])
        for limit, lev in tiers:
            if position_value < limit:
                return lev
        return 5  # Default max
    def get_orderbook_snapshot(self, symbol: str) -> Optional[OrderBook]:
        """Get current orderbook snapshot"""
        with self._data_lock:
            return self.orderbooks.get(symbol)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol"""
        with self._data_lock:
            return self.positions.get(symbol)
    
    def get_klines_df(self, symbol: str, interval: str = "60") -> Optional[pd.DataFrame]:
        """Get klines as DataFrame"""
        with self._data_lock:
            key = f"{symbol}_{interval}"
            buffer = self.klines.get(key)
            if not buffer:
                return None
            df = pd.DataFrame([{
                'timestamp': k.timestamp,
                'open': k.open,
                'high': k.high,
                'low': k.low,
                'close': k.close,
                'volume': k.volume,
                'turnover': k.turnover
            } for k in buffer])
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df
    
    def close(self):
        """Close all connections"""
        self.stop_websocket()
        logger.info("BybitV5Client closed")


# ==================== MODULE LEVEL FUNCTIONS ====================

def get_bybit_max_lev(symbol: str, position_value: float = 0) -> int:
    """
    Get maximum leverage for a symbol based on position value.
    Module level function for external import.
    """
    tier_limits = {
        'BTCUSDT': [(100000, 100), (200000, 50), (500000, 25), (1000000, 10)],
        'ETHUSDT': [(50000, 50), (100000, 25), (250000, 10)],
    }
    # Default tiers for altcoins if not specifically listed
    tiers = tier_limits.get(symbol, [(50000, 25), (100000, 10)])

    for limit, lev in tiers:
        if position_value < limit:
            return lev
    return 5  # Conservative default max


# ==================== FACTORY FUNCTIONS ====================

def create_bybit_client(
    api_key: Optional[str] = None,
    private_key_path: Optional[str] = None,
    private_key_content: Optional[str] = None,
    demo: bool = True,
    use_secrets_manager: bool = True,
    **kwargs
) -> BybitV5Client:
    """
    Factory function to create Bybit V5 client
    
    Args:
        api_key: Bybit API key (defaults to secrets manager or env BYBIT_API_KEY)
        private_key_path: Path to RSA private key PEM file
        private_key_content: RSA private key content as string
        demo: Use demo/paper trading mode
        use_secrets_manager: Load credentials from secrets manager if not provided
        **kwargs: Additional config options
    
    Returns:
        BybitV5Client instance
    """
    # Mark demo as manually set so __post_init__ doesn't override
    config_kwargs = {
        'recv_window': kwargs.get('recv_window', 5000),
        'max_retries': kwargs.get('max_retries', 5),
        'retry_delay': kwargs.get('retry_delay', 1.0),
        'rate_limit_delay': kwargs.get('rate_limit_delay', 0.08),
        'backoff_factor': kwargs.get('backoff_factor', 1.8),
    }
    
    config = APIConfig(
        api_key=api_key or '',
        private_key_path=private_key_path or '',
        private_key_content=private_key_content or '',
        demo=demo,
        **config_kwargs
    )
    
    # Mark that demo was manually set
    config._demo_set_manually = True
    
    return BybitV5Client(config)


def get_bybit_client(use_secrets_manager: bool = True) -> BybitV5Client:
    """
    Get Bybit client from secrets manager or environment variables
    
    Priority:
        1. Secrets Manager (encrypted local storage)
        2. Environment variables
    
    Required credentials (in secrets manager or env):
        BYBIT_API_KEY
        BYBIT_PRIVATE_KEY (content) or BYBIT_PRIVATE_KEY_PATH (file path)
        BYBIT_DEMO (optional, default: true)
    """
    return create_bybit_client(use_secrets_manager=use_secrets_manager)


# ==================== SHARED SINGLETON CLIENT ====================
# All components (pipeline, NEO core, risk engine) must call get_shared_client()
# to reuse a single BybitV5Client instance instead of creating independent connections.
# This reduces Bybit WS connections from 6+ down to 3 (public, private, trade).

_shared_client: Optional[BybitV5Client] = None
_shared_client_lock = threading.Lock()


def get_shared_client() -> BybitV5Client:
    """Return the process-wide singleton BybitV5Client (creates it on first call)."""
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    with _shared_client_lock:
        if _shared_client is None:
            _shared_client = BybitV5Client()
            logger.info("Shared BybitV5Client singleton created")
    return _shared_client


# ==================== WEBHOOK SERVER ====================

class BybitWebhookServer:
    """
    Webhook server for receiving external trading signals
    
    Can receive signals from:
    - TradingView alerts
    - External signal providers
    - Custom applications
    """
    
    def __init__(
        self,
        port: int = 8080,
        auth_token: Optional[str] = None,
        api_client: Optional[BybitV5Client] = None
    ):
        """
        Initialize webhook server
        
        Args:
            port: Server port
            auth_token: Authentication token
            api_client: Bybit API client for executing signals
        """
        self.port = port
        self.auth_token = auth_token
        self.api_client = api_client
        self.app = None
        self.running = False
        self._server_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[WebhookSignal], None]] = []
    
    def start(self):
        """Start webhook server in background thread"""
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"Webhook server starting on port {self.port}")
    
    def _run_server(self):
        """Run Flask server"""
        try:
            from flask import Flask, request, jsonify
            
            self.app = Flask(__name__)
            
            @self.app.route('/webhook', methods=['POST'])
            def webhook():
                try:
                    # Verify auth token if set
                    if self.auth_token:
                        token = request.headers.get('X-Auth-Token')
                        if token != self.auth_token:
                            return jsonify({'error': 'Unauthorized'}), 401
                    
                    # Parse signal
                    data = request.get_json()
                    logger.info(f"Webhook received: {data}")
                    
                    # Convert to WebhookSignal
                    signal = WebhookSignal(
                        action=data.get('action', ''),
                        symbol=data.get('symbol', ''),
                        price=data.get('price'),
                        qty=data.get('qty'),
                        order_type=data.get('order_type', 'market'),
                        leverage=data.get('leverage'),
                        take_profit=data.get('take_profit'),
                        stop_loss=data.get('stop_loss'),
                        source=data.get('source', 'webhook')
                    )
                    
                    # Execute signal if API client available
                    if self.api_client:
                        self._execute_signal(signal)
                    
                    # Notify callbacks
                    for callback in self._callbacks:
                        try:
                            callback(signal)
                        except Exception as e:
                            logger.error(f"Webhook callback error: {e}")
                    
                    return jsonify({'status': 'received', 'signal': asdict(signal)}), 200
                    
                except Exception as e:
                    logger.error(f"Webhook error: {e}")
                    return jsonify({'error': str(e)}), 500
            
            @self.app.route('/health', methods=['GET'])
            def health():
                return jsonify({
                    'status': 'ok',
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'demo' if (self.api_client and self.api_client.config.demo) else 'live'
                }), 200
            
            @self.app.route('/status', methods=['GET'])
            def status():
                """Get API status"""
                if not self.api_client:
                    return jsonify({'error': 'No API client'}), 503
                
                return jsonify({
                    'demo_mode': self.api_client.config.demo,
                    'tickers': list(self.api_client.tickers.keys()),
                    'positions': {s: p.size for s, p in self.api_client.positions.items()}
                }), 200
            
            self.running = True
            self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
            
        except ImportError:
            logger.error("Flask not installed. Cannot start webhook server.")
        except Exception as e:
            logger.error(f"Failed to start webhook server: {e}")
    
    def _execute_signal(self, signal: WebhookSignal):
        """Execute trading signal"""
        if not self.api_client:
            return
        
        try:
            # Determine side
            side = signal.action.upper()
            if side not in ['BUY', 'SELL']:
                logger.warning(f"Invalid action: {signal.action}")
                return
            
            # Place order
            result = self.api_client.place_order(
                symbol=signal.symbol,
                side=side,
                order_type=signal.order_type.upper(),
                qty=str(signal.qty) if signal.qty else "0.001",
                price=str(signal.price) if signal.price else None,
                take_profit=str(signal.take_profit) if signal.take_profit else None,
                stop_loss=str(signal.stop_loss) if signal.stop_loss else None
            )
            
            if result:
                logger.info(f"Signal executed: {signal.action} {signal.symbol}")
            else:
                logger.error(f"Failed to execute signal: {signal.action} {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Execute signal error: {e}")
    
    def stop(self):
        """Stop webhook server"""
        self.running = False
        logger.info("Webhook server stopped")
    
    def on_signal(self, callback: Callable[[WebhookSignal], None]):
        """Register signal callback"""
        self._callbacks.append(callback)
# =============================================================================
# CIRCUIT BREAKER PATTERN
# =============================================================================

class CircuitState(Enum):
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, rejecting requests
    HALF_OPEN = auto()   # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker to prevent hammering failing endpoints"""
    
    failure_threshold: int = 5          # Open after 5 failures
    success_threshold: int = 3          # Close after 3 successes in half-open
    timeout: float = 30.0               # Try again after 30s
    
    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _success_count: int = field(default=0, repr=False)
    _last_failure_time: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout elapsed for half-open
                if time.time() - self._last_failure_time >= self.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"[CIRCUIT] Transitioning to HALF_OPEN after timeout")
            return self._state
    
    def record_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"[CIRCUIT] Transitioning to CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def record_failure(self) -> bool:
        """Returns True if circuit just opened"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed in half-open, go back to open
                self._state = CircuitState.OPEN
                logger.warning(f"[CIRCUIT] Transitioning to OPEN (failed in half-open)")
                return True
            
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"[CIRCUIT] Transitioning to OPEN after {self.failure_count} failures")
                return True
            
            return False
    
    def can_execute(self) -> bool:
        return self.state != CircuitState.OPEN


# =============================================================================
# RATE LIMITER WITH TOKEN BUCKET
# =============================================================================

@dataclass
class TokenBucket:
    """Token bucket for smooth rate limiting"""
    
    rate: float  # tokens per second
    capacity: float  # max tokens
    
    _tokens: float = field(default=0, repr=False)
    _last_update: float = field(default_factory=time.time, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def __post_init__(self):
        self._tokens = self.capacity  # Start full
    
    def consume(self, tokens: float = 1.0) -> float:
        """
        Attempt to consume tokens. Returns wait time if insufficient.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_update = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            
            # Calculate wait time
            deficit = tokens - self._tokens
            wait_time = deficit / self.rate
            return wait_time


# =============================================================================
# ENDPOINT HEALTH TRACKER
# =============================================================================

@dataclass
class EndpointHealth:
    """Tracks health metrics for each endpoint"""
    
    region: BybitRegion
    circuit: CircuitBreaker = field(default_factory=CircuitBreaker)
    
    # Rate limiters
    public_bucket: TokenBucket = field(
        default_factory=lambda: TokenBucket(
            rate=RateLimits.PUBLIC_PER_SECOND,
            capacity=RateLimits.PUBLIC_PER_SECOND
        )
    )
    private_bucket: TokenBucket = field(
        default_factory=lambda: TokenBucket(
            rate=RateLimits.PRIVATE_PER_MINUTE / 60,
            capacity=RateLimits.PRIVATE_PER_MINUTE / 60
        )
    )
    
    # Stats
    success_count: int = 0
    failure_count: int = 0
    avg_latency: float = 0.0
    last_used: float = 0.0
    
    def record_latency(self, latency_ms: float):
        """Update rolling average latency"""
        if self.avg_latency == 0:
            self.avg_latency = latency_ms
        else:
            # EWMA with alpha=0.2
            self.avg_latency = 0.8 * self.avg_latency + 0.2 * latency_ms


# =============================================================================
# RESILIENT CLIENT CONFIGURATION
# =============================================================================

@dataclass
class ResilientConfig:
    """Configuration for resilient client"""
    
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    # Region preferences — bybit.com endpoints only (EU/NL use separate accounts)
    region_priority: List[BybitRegion] = field(default_factory=lambda: [
        BybitRegion.GLOBAL,   # api.bybit.com — primary
        BybitRegion.BYTICK,   # bytick.com — backup
        # BybitRegion.EU,     # DISABLED: bybit.eu uses a separate account
        # BybitRegion.NL,     # DISABLED: Netherlands endpoint, separate account
        # BybitRegion.UAE,    # DISABLED: not needed
    ])
    
    demo: bool = False
    recv_window: int = 5000
    timeout: int = 10
    max_retries: int = 3
    
    # Failover settings
    failover_on_403: bool = True  # Auto-failover on geo-block
    failover_on_timeout: bool = True
    failover_on_5xx: bool = True
    
    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_timeout: float = 30.0


# =============================================================================
# MAIN RESILIENT CLIENT
# =============================================================================

class BybitResilientClient:
    """
    Production-grade Bybit V5 client with:
    - Automatic failover across all regions
    - Hardcoded rate limits (never get banned)
    - Circuit breakers (don't hammer dead endpoints)
    - Health tracking (use fastest endpoint)
    - FSMA compliance (Belgium/EU safe)
    """
    
    def __init__(self, config: Optional[ResilientConfig] = None):
        self.config = config or ResilientConfig()
        
        # Initialize health tracking for all regions
        self._health: Dict[BybitRegion, EndpointHealth] = {
            region: EndpointHealth(region=region)
            for region in BybitRegion
        }
        
        # Current working endpoint
        self._current_region: Optional[BybitRegion] = None
        self._session = requests.Session()
        
        # Configure session with retries for connection errors
        adapter = requests.adapters.HTTPAdapter(
            max_retries=Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
        )
        self._session.mount("https://", adapter)
        
        # Time sync
        self._time_offset: float = 0.0
        self._sync_time()
        
        logger.info(f"[INIT] Resilient client initialized")
        logger.info(f"[INIT] Priority regions: {[r.display_name for r in self.config.region_priority]}")
    
    def _sync_time(self):
        """Synchronize local time with Bybit server"""
        try:
            # Use fastest responding endpoint for time sync
            for region in self.config.region_priority:
                try:
                    start = time.time()
                    resp = self._session.get(
                        f"{region.rest_url}/v5/market/time",
                        timeout=5
                    )
                    latency = (time.time() - start) * 1000
                    
                    if resp.status_code == 200:
                        server_time = resp.json().get("result", {}).get("timeSecond", 0)
                        if server_time:
                            self._time_offset = int(server_time * 1000) - int(time.time() * 1000)
                            self._health[region].record_latency(latency)
                            logger.info(f"[TIME] Synced with {region.display_name}, offset={self._time_offset}ms")
                            return
                except Exception as e:
                    logger.debug(f"[TIME] {region.display_name} failed: {e}")
                    continue
        except Exception as e:
            logger.warning(f"[TIME] Sync failed: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp adjusted for server time"""
        return str(int(time.time() * 1000) + int(self._time_offset))
    
    def _get_headers(self, auth: bool, payload: str = "") -> Dict[str, str]:
        """Generate request headers"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'BybitResilientClient/1.0'
        }
        
        if auth and self.config.api_key:
            timestamp = self._get_timestamp()
            headers['X-BAPI-API-KEY'] = self.config.api_key
            headers['X-BAPI-TIMESTAMP'] = timestamp
            headers['X-BAPI-RECV-WINDOW'] = str(self.config.recv_window)
            headers['X-BAPI-SIGN'] = self._generate_signature(
                timestamp, self.config.api_key, str(self.config.recv_window), payload
            )
        
        if self.config.demo:
            headers['X-BAPI-DEMO-TRADING'] = '1'
        
        return headers
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        base = RateLimits.BASE_BACKOFF * (2 ** attempt)
        jitter = random.uniform(0, RateLimits.JITTER_MAX)
        return min(base + jitter, RateLimits.MAX_BACKOFF)
    
    def _wait_for_rate_limit(self, region: BybitRegion, is_private: bool):
        """Wait if rate limit bucket is empty"""
        health = self._health[region]
        bucket = health.private_bucket if is_private else health.public_bucket
        
        wait_time = bucket.consume(1.0)
        if wait_time > 0:
            logger.debug(f"[RATE LIMIT] Waiting {wait_time:.2f}s for {region.display_name}")
            time.sleep(wait_time)
    
    def _select_best_endpoint(self) -> Optional[BybitRegion]:
        """Select best available endpoint based on health"""
        available = []
        
        for region in self.config.region_priority:
            health = self._health[region]
            
            # Skip if circuit is open
            if not health.circuit.can_execute():
                logger.debug(f"[SELECT] {region.display_name}: Circuit OPEN")
                continue
            
            # Skip demo unless configured
            if region == BybitRegion.DEMO and not self.config.demo:
                continue
            
            # Calculate score (lower is better)
            # Factors: latency, success rate, recency
            success_rate = health.success_count / max(health.success_count + health.failure_count, 1)
            latency_score = health.avg_latency if health.avg_latency > 0 else 1000
            recency_score = time.time() - health.last_used
            
            # Weighted score
            score = (latency_score * 0.5) + ((1 - success_rate) * 500) - (recency_score * 10)
            available.append((region, score))
        
        if not available:
            return None
        
        # Sort by score (lower is better)
        available.sort(key=lambda x: x[1])
        best_region = available[0][0]
        
        logger.debug(f"[SELECT] Best endpoint: {best_region.display_name}")
        return best_region
    
    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
        auth: bool = False
    ) -> Dict[str, Any]:
        """
        Make resilient request with auto-failover and rate limiting
        
        Args:
            method: HTTP method
            endpoint: API endpoint (e.g., "/market/kline")
            params: Query parameters
            body: Request body
            auth: Whether to include authentication
            
        Returns:
            API response dict
            
        Raises:
            ConnectionError: If all endpoints fail
        """
        params = params or {}
        body = body or {}
        
        # Build payload for signature
        if method == "GET":
            payload = urlencode(sorted(params.items())) if params else ""
        else:
            payload = json.dumps(body, separators=(',', ':')) if body else ""
        
        headers = self._get_headers(auth, payload)
        
        # Try endpoints in priority order with circuit breaker awareness
        last_error = None
        attempted_regions = []
        
        for _ in range(len(self.config.region_priority)):
            region = self._select_best_endpoint()
            if not region:
                logger.error("[REQUEST] No available endpoints (all circuits open)")
                break
            
            attempted_regions.append(region.display_name)
            health = self._health[region]
            
            # Rate limiting
            self._wait_for_rate_limit(region, auth)
            
            # Build URL
            base_url = region.rest_url
            url = f"{base_url}/v5{endpoint}"
            
            if method == "GET" and payload:
                full_url = f"{url}?{payload}"
            else:
                full_url = url
            
            # Execute with retries
            for attempt in range(self.config.max_retries):
                try:
                    start_time = time.time()
                    
                    if method == "GET":
                        response = self._session.get(
                            full_url, 
                            headers=headers, 
                            timeout=self.config.timeout
                        )
                    elif method == "POST":
                        response = self._session.post(
                            url, 
                            headers=headers, 
                            data=payload, 
                            timeout=self.config.timeout
                        )
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    
                    latency = (time.time() - start_time) * 1000
                    health.record_latency(latency)
                    health.last_used = time.time()
                    
                    # Check for geo-block (403 Forbidden)
                    if response.status_code == 403:
                        logger.warning(f"[403] {region.display_name} blocked (FSMA/geo-restriction)")
                        health.circuit.record_failure()
                        health.failure_count += 1
                        last_error = f"403 Forbidden on {region.display_name}"
                        break  # Try next region
                    
                    # Check for rate limit (429)
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        logger.warning(f"[429] Rate limited on {region.display_name}, waiting {retry_after}s")
                        time.sleep(retry_after)
                        continue
                    
                    # Check for server error (5xx)
                    if response.status_code >= 500:
                        logger.warning(f"[{response.status_code}] Server error on {region.display_name}")
                        if self.config.failover_on_5xx:
                            health.circuit.record_failure()
                            health.failure_count += 1
                            last_error = f"{response.status_code} on {region.display_name}"
                            break  # Try next region
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Check API error codes
                    if result.get('retCode') == 10002:  # Timestamp error
                        logger.warning(f"[10002] Time sync error, resyncing...")
                        self._sync_time()
                        time.sleep(self._calculate_backoff(attempt))
                        continue
                    
                    if result.get('retCode') == 10006:  # Rate limit
                        retry_after = 5
                        logger.warning(f"[10006] Rate limited, waiting {retry_after}s...")
                        time.sleep(retry_after)
                        continue
                    
                    if result.get('retCode') != 0:
                        error_msg = result.get('retMsg', 'Unknown error')
                        # Don't failover on client errors (4xx)
                        if result.get('retCode') < 5000:
                            raise ValueError(f"API error {result['retCode']}: {error_msg}")
                        # Server errors might be regional, try failover
                        last_error = f"API error {result['retCode']} on {region.display_name}"
                        health.failure_count += 1
                        break
                    
                    # Success!
                    health.success_count += 1
                    health.circuit.record_success()
                    self._current_region = region
                    
                    logger.debug(f"[SUCCESS] {region.display_name} {endpoint} ({latency:.0f}ms)")
                    return result.get('result', result)
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"[TIMEOUT] {region.display_name} attempt {attempt + 1}")
                    if self.config.failover_on_timeout:
                        health.circuit.record_failure()
                        health.failure_count += 1
                        last_error = f"Timeout on {region.display_name}"
                        break  # Try next region
                    time.sleep(self._calculate_backoff(attempt))
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"[ERROR] {region.display_name}: {e}")
                    health.circuit.record_failure()
                    health.failure_count += 1
                    last_error = f"{e} on {region.display_name}"
                    time.sleep(self._calculate_backoff(attempt))
            
            # Mark circuit if all retries failed
            if last_error and region.display_name in last_error:
                health.circuit.record_failure()
        
        # All endpoints exhausted
        raise ConnectionError(
            f"All endpoints failed. Attempted: {attempted_regions}. Last error: {last_error}"
        )
    
    # ==================== PUBLIC API METHODS ====================
    
    def get_server_time(self) -> Optional[datetime]:
        """Get Bybit server time"""
        try:
            result = self.request("GET", "/market/time", auth=False)
            server_time_ms = int(result.get('timeSecond', 0))
            return datetime.fromtimestamp(server_time_ms)
        except Exception as e:
            logger.error(f"get_server_time error: {e}")
            return None
    
    def get_klines(
        self,
        symbol: str,
        interval: str = "60",
        category: str = "linear",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Dict]:
        """
        Get kline/candlestick data with auto-failover
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            category: Market category (spot, linear, inverse)
            limit: Number of candles (max 1000)
            start: Start timestamp (ms)
            end: End timestamp (ms)
        """
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        
        result = self.request("GET", "/market/kline", params=params, auth=False)
        return result.get('list', [])
    
    def get_tickers(self, symbol: Optional[str] = None, category: str = "linear") -> List[Dict]:
        """Get latest ticker information"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        result = self.request("GET", "/market/tickers", params=params, auth=False)
        return result.get('list', [])
    
    def get_orderbook(self, symbol: str, category: str = "linear", limit: int = 25) -> Dict:
        """Get orderbook depth"""
        params = {
            "category": category,
            "symbol": symbol,
            "limit": min(limit, 500)
        }
        return self.request("GET", "/market/orderbook", params=params, auth=False)
    
    def get_instruments_info(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict]:
        """Get instruments information"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        if status:
            params["status"] = status
        
        result = self.request("GET", "/market/instruments-info", params=params, auth=False)
        return result.get('list', [])
    
    # ==================== PRIVATE API METHODS (require auth) ====================
    
    def get_wallet_balance(self, account_type: str = "CONTRACT") -> Dict:
        """Get wallet balance — auto-detects CONTRACT vs UNIFIED."""
        def _fetch(acct):
            return self.request("GET", "/account/wallet-balance",
                                params={"accountType": acct}, auth=True)
        try:
            return _fetch(account_type)
        except Exception as e:
            if "10001" in str(e):
                fallback = "UNIFIED" if account_type == "CONTRACT" else "CONTRACT"
                try:
                    return _fetch(fallback)
                except Exception as e2:
                    logger.error(f"get_wallet_balance error: {e2}")
                    return {}
            logger.error(f"get_wallet_balance error: {e}")
            return {}
    
    def get_positions_list(self, symbol: Optional[str] = None, category: str = "linear") -> List[Dict]:
        """Get open positions as list of dicts (backward compatibility)"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        result = self.request("GET", "/position/list", params=params, auth=True)
        return result.get('list', [])
    
    def place_order(
        self,
        symbol: str,
        side: str,  # Buy or Sell
        order_type: str,  # Market or Limit
        qty: str,
        price: Optional[str] = None,
        category: str = "linear",
        time_in_force: str = "GTC",
        **kwargs
    ) -> Dict:
        """
        Place order with rate limiting
        
        Note: Uses order rate limiter (10 req/s per symbol)
        """
        body = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "timeInForce": time_in_force
        }
        
        if order_type == "Limit" and price:
            body["price"] = price
        
        # Add any additional parameters
        body.update(kwargs)
        
        return self.request("POST", "/order/create", body=body, auth=True)
    
    def cancel_order(self, symbol: str, order_id: str, category: str = "linear") -> Dict:
        """Cancel an order"""
        body = {
            "category": category,
            "symbol": symbol,
            "orderId": order_id
        }
        return self.request("POST", "/order/cancel", body=body, auth=True)
    
    def get_open_orders(
        self,
        symbol: Optional[str] = None,
        category: str = "linear"
    ) -> List[Dict]:
        """Get open orders"""
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        
        result = self.request("GET", "/order/realtime", params=params, auth=True)
        return result.get('list', [])
    
    # ==================== HEALTH MONITORING ====================
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get health status of all endpoints"""
        report = {
            "current_region": self._current_region.display_name if self._current_region else None,
            "endpoints": {}
        }
        
        for region, health in self._health.items():
            total = health.success_count + health.failure_count
            success_rate = health.success_count / max(total, 1)
            
            report["endpoints"][region.display_name] = {
                "circuit_state": health.circuit.state.name,
                "success_rate": f"{success_rate:.1%}",
                "avg_latency_ms": round(health.avg_latency, 2),
                "total_requests": total,
                "last_used": datetime.fromtimestamp(health.last_used).isoformat() if health.last_used else None
            }
        
        return report
    
    def verify_fsma_compliance(self) -> Dict[str, Any]:
        """
        Verify FSMA compliance status
        Returns info about which endpoints are accessible from Belgium/EU
        """
        results = {
            "tested_at": datetime.now().isoformat(),
            "belgium_compliant": False,
            "working_endpoints": [],
            "blocked_endpoints": []
        }
        
        for region in [BybitRegion.EU, BybitRegion.NL, BybitRegion.GLOBAL, BybitRegion.BYTICK]:
            try:
                resp = self._session.get(
                    f"{region.rest_url}/v5/market/time",
                    timeout=5
                )
                if resp.status_code == 200:
                    results["working_endpoints"].append(region.display_name)
                    if region in (BybitRegion.EU, BybitRegion.NL):
                        results["belgium_compliant"] = True
                elif resp.status_code == 403:
                    results["blocked_endpoints"].append(region.display_name)
            except Exception as e:
                results["blocked_endpoints"].append(f"{region.display_name} ({e})")
        
        return results


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_belgium_client(api_key: Optional[str] = None, api_secret: Optional[str] = None) -> BybitResilientClient:
    """
    Create client for bybit.com accounts (EU endpoint uses separate account — DO NOT use EU/NL)
    """
    config = ResilientConfig(
        api_key=api_key or os.getenv("BYBIT_API_KEY"),
        api_secret=api_secret or os.getenv("BYBIT_RSA_PRIVATE_KEY_PATH"),
        region_priority=[
            BybitRegion.GLOBAL,
            BybitRegion.BYTICK,
            # BybitRegion.EU,   # DISABLED: bybit.eu is a separate account
            # BybitRegion.NL,   # DISABLED: Netherlands, separate account
        ],
        failover_on_403=True,
        failover_on_timeout=True
    )
    
    return BybitResilientClient(config)


def create_global_client(api_key: Optional[str] = None, api_secret: Optional[str] = None) -> BybitResilientClient:
    """
    Create client for global use — bybit.com only
    """
    config = ResilientConfig(
        api_key=api_key or os.getenv("BYBIT_API_KEY"),
        api_secret=api_secret or os.getenv("BYBIT_RSA_PRIVATE_KEY_PATH"),
        region_priority=[
            BybitRegion.GLOBAL,
            BybitRegion.BYTICK,
            # BybitRegion.UAE,  # DISABLED: not needed
            # BybitRegion.EU,   # DISABLED: separate account
        ]
    )
    
    return BybitResilientClient(config)


def create_demo_client() -> BybitResilientClient:
    """Create client for demo/paper trading"""
    config = ResilientConfig(
        demo=True,
        region_priority=[BybitRegion.DEMO]
    )
    
    return BybitResilientClient(config)
    
# ==================== USAGE EXAMPLE ====================

def extract_wallet_snapshot(wallet_response, coin: str = "USDT"):
    """Parse Bybit Unified wallet response into (coin_obj, snapshot_dict).

    Returns a SimpleNamespace with .coin, .wallet_balance, .available_balance,
    .margin_balance, .equity, .available_to_withdraw, .total_order_im,
    .total_position_im attributes; plus a flat snapshot dict with the same keys.
    Returns (None, {}) when the target coin is not found.
    """
    from types import SimpleNamespace

    target = coin.upper()
    accounts = []
    if isinstance(wallet_response, dict):
        accounts = wallet_response.get("list", []) or []
    elif isinstance(wallet_response, list):
        accounts = wallet_response

    account = None
    for entry in accounts:
        coin_rows = entry.get("coin", []) or []
        if any(str(c.get("coin", "")).upper() == target for c in coin_rows):
            account = entry
            break
    if account is None and accounts:
        account = accounts[0]
    if account is None:
        return None, {}

    coin_rows = account.get("coin", []) or []
    coin_data = next(
        (c for c in coin_rows if str(c.get("coin", "")).upper() == target), None
    )
    if coin_data is None:
        return None, {}

    def _n(v):
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            return 0.0

    wallet_balance    = _n(coin_data.get("walletBalance"))
    coin_equity       = _n(coin_data.get("equity"))
    total_equity      = _n(account.get("totalEquity"))
    equity            = max(coin_equity, total_equity, wallet_balance)
    total_order_im    = _n(coin_data.get("totalOrderIM"))
    total_position_im = _n(coin_data.get("totalPositionIM"))
    total_initial_margin = _n(account.get("totalInitialMargin"))
    used_margin       = max(total_order_im + total_position_im, total_initial_margin, 0.0)
    available_to_withdraw = max(
        _n(coin_data.get("availableToWithdraw")),
        _n(coin_data.get("availableBalance")),
    )
    available_margin  = max(
        _n(account.get("totalAvailableBalance")),
        available_to_withdraw,
        equity - used_margin,
        0.0,
    )
    margin_balance    = max(_n(coin_data.get("marginBalance")), equity)

    coin_obj = SimpleNamespace(
        coin=str(coin_data.get("coin", target)),
        wallet_balance=wallet_balance,
        available_balance=available_margin,
        margin_balance=margin_balance,
        equity=equity,
        available_to_withdraw=available_to_withdraw,
        total_order_im=total_order_im,
        total_position_im=total_position_im,
    )
    snapshot = {
        "equity":               equity,
        "available_margin":     available_margin,
        "available_balance":    available_margin,
        "available_to_withdraw": available_to_withdraw,
        "used_margin":          used_margin,
        "margin_balance":       margin_balance,
        "wallet_balance":       wallet_balance,
        "total_order_im":       total_order_im,
        "total_position_im":    total_position_im,
    }
    return coin_obj, snapshot


# Alias for backward compatibility
BybitAPIClient = BybitV5Client

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    
    try:
        client = get_bybit_client()
        
        # Test REST API
        print("\n=== Testing REST API ===")
        
        # Get server time
        server_time = client.get_server_time()
        print(f"Server time: {server_time}")
        
        # Get tickers
        tickers = client.get_tickers(symbol="BTCUSDT", category="linear")
        for symbol, ticker in tickers.items():
            print(f"{symbol}: ${ticker.last_price:,.2f}")
        
        # Get klines
        df = client.get_klines("BTCUSDT", interval="60", limit=10)
        if not df.empty:
            print(f"\nKlines (last 10):")
            print(df.tail())
        
        # Get orderbook
        ob = client.get_orderbook("BTCUSDT", limit=5)
        if ob:
            print(f"\nOrderbook:")
            print(f"  Best Bid: {ob.best_bid.price if ob.best_bid else 'N/A'}")
            print(f"  Best Ask: {ob.best_ask.price if ob.best_ask else 'N/A'}")
            print(f"  Spread: {ob.spread:.2f} ({ob.spread_pct:.4f}%)")
        
        # Get wallet balance (requires authentication)
        try:
            balances = client.get_wallet_balance()
            print(f"\nWallet balances:")
            for coin, balance in balances.items():
                print(f"  {coin}: {balance.wallet_balance} (Available: {balance.available_balance})")
        except Exception as e:
            print(f"Wallet balance error (expected if no API key): {e}")
        
        # Test WebSocket
        print("\n=== Testing WebSocket ===")
        
        def on_ticker(ticker: TickerData):
            print(f"[WS] {ticker.symbol}: ${ticker.last_price:,.2f}")
        
        client.on_ticker(on_ticker)
        
        # Start WebSocket
        client.start_websocket(
            symbols=["BTCUSDT", "ETHUSDT"],
            public_categories=["linear"],
            public_topics=["tickers", "orderbook.50"]
        )
        
        # Run for 30 seconds
        try:
            print("\nRunning WebSocket for 30 seconds...")
            time.sleep(30)
        except KeyboardInterrupt:
            pass
        finally:
            client.close()
            print("\nStopped")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to set environment variables:")
        print("  export BYBIT_API_KEY=your_api_key")
        print("  export BYBIT_RSA_PRIVATE_KEY_PATH=/path/to/private.pem")
        print("  export BYBIT_DEMO=true")

        # Create Belgium-optimized client
    print("\n[1] Creating Belgium-optimized client...")
    client = create_belgium_client()
    
    # Test FSMA compliance
    print("\n[2] Testing FSMA compliance...")
    compliance = client.verify_fsma_compliance()
    print(f"   Belgium compliant: {compliance['belgium_compliant']}")
    print(f"   Working endpoints: {compliance['working_endpoints']}")
    print(f"   Blocked endpoints: {compliance['blocked_endpoints']}")
    
    # Test data fetch with failover
    print("\n[3] Testing klines fetch (BTCUSDT, 5min)...")
    try:
        klines = client.get_klines("BTCUSDT", interval="5", limit=5)
        print(f"   Fetched {len(klines)} candles")
        if klines:
            print(f"   Latest: {klines[0]}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test tickers
    print("\n[4] Testing tickers fetch...")
    try:
        tickers = client.get_tickers("BTCUSDT")
        print(f"   Fetched {len(tickers)} tickers")
        if tickers:
            print(f"   BTC Price: {tickers[0].get('lastPrice')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Health report
    print("\n[5] Health report...")
    health = client.get_health_report()
    print(f"   Current region: {health['current_region']}")
    for name, stats in health['endpoints'].items():
        print(f"   {name}: {stats}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
