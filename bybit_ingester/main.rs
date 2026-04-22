use std::sync::Arc;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    routing::get,
    Router,
};
use chrono::Utc;
use dashmap::DashMap;
use flume::{bounded, Sender, Receiver};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio::time::interval;
use tracing::{info, warn, error};
use futures_util::future::join_all;

mod websocket;
mod orderbook;
mod metrics;

use websocket::{BybitWebSocket, fetch_all_symbols};
use orderbook::OrderbookManager;
use metrics::Metrics;

/// Market data message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MarketMessage {
    #[serde(rename = "tick")]
    Tick {
        symbol: String,
        price: f64,
        size: f64,
        side: String,
        timestamp: i64,
    },
    #[serde(rename = "orderbook")]
    Orderbook {
        symbol: String,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        timestamp: i64,
        update_type: String,
    },
    #[serde(rename = "trade")]
    Trade {
        symbol: String,
        price: f64,
        size: f64,
        side: String,
        timestamp: i64,
    },
    #[serde(rename = "kline")]
    Kline {
        symbol: String,
        timeframe: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        timestamp: i64,
    },
}

/// Configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub redis_url: String,
    pub max_symbols: usize,
    pub symbol_offset: usize,  // Skip first N symbols — used for WS sharding
    pub message_buffer_size: usize,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            redis_url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
            max_symbols: std::env::var("MAX_SYMBOLS")
                .unwrap_or_else(|_| "0".to_string())  // 0 = stream all
                .parse()
                .unwrap_or(0),
            symbol_offset: std::env::var("SYMBOL_OFFSET")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
            message_buffer_size: std::env::var("MESSAGE_BUFFER_SIZE")
                .unwrap_or_else(|_| "10000".to_string())
                .parse()
                .unwrap_or(10000),
        })
    }
}

/// Shared state between components
pub struct SharedState {
    /// Orderbook manager for all symbols
    pub orderbooks: OrderbookManager,
    /// Recent trades for volume calculation
    pub recent_trades: DashMap<String, VecDeque<TradeInfo>>,
    /// Metrics
    pub metrics: Metrics,
    /// Last time each symbol's orderbook was published to Redis (throttle: 100ms)
    pub last_ob_publish: DashMap<String, Instant>,
}

#[derive(Debug, Clone)]
pub struct TradeInfo {
    pub price: f64,
    pub size: f64,
    pub side: String,
    pub timestamp: i64,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            orderbooks: OrderbookManager::new(),
            recent_trades: DashMap::new(),
            metrics: Metrics::new(),
            last_ob_publish: DashMap::new(),
        }
    }
}

/// Redis publisher for market data
pub struct RedisPublisher {
    client: redis::aio::MultiplexedConnection,
    batch: Vec<(String, String)>,
    batch_size: usize,
    last_flush: Instant,
}

impl RedisPublisher {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = redis::Client::open(redis_url)?;
        let conn = client.get_multiplexed_async_connection().await?;
        
        Ok(Self {
            client: conn,
            batch: Vec::with_capacity(1000),
            batch_size: 100,
            last_flush: Instant::now(),
        })
    }
    
    pub async fn publish(&mut self, channel: &str, message: &str) -> Result<()> {
        self.batch.push((channel.to_string(), message.to_string()));
        
        // Flush if batch is full or 10ms passed
        if self.batch.len() >= self.batch_size 
            || self.last_flush.elapsed() > Duration::from_millis(10) {
            self.flush().await?;
        }
        
        Ok(())
    }
    
    pub async fn flush(&mut self) -> Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }
        
        let pipe = redis::pipe();
        let mut pipe = pipe;
        
        for (channel, message) in &self.batch {
            pipe.cmd("PUBLISH").arg(channel).arg(message);
        }
        
        pipe.query_async::<_, ()>(&mut self.client).await?;
        
        self.batch.clear();
        self.last_flush = Instant::now();
        
        Ok(())
    }
}

/// Main ingester
pub struct Ingester {
    config: Config,
    state: Arc<SharedState>,
    message_tx: Sender<MarketMessage>,
    message_rx: Receiver<MarketMessage>,
    /// Broadcast sender — every MarketMessage is fanned out to all Axum WS clients
    ws_tx: broadcast::Sender<MarketMessage>,
}

impl Ingester {
    pub fn new(config: Config) -> Self {
        let (tx, rx) = bounded(config.message_buffer_size);
        // 10 000-slot ring buffer — lagging WS clients are dropped, not stalled
        let (ws_tx, _) = broadcast::channel(10_000);
        
        Self {
            config,
            state: Arc::new(SharedState::new()),
            message_tx: tx,
            message_rx: rx,
            ws_tx,
        }
    }
    
    pub async fn run(&self) -> Result<()> {
        info!("Starting Bybit Ingester (SHARDED ARCHITECTURE)");
        info!("Redis URL: {}", self.config.redis_url);
        info!("Max symbols: {} (0 = stream all)", self.config.max_symbols);

        // 1. Start the central message processor — listens to ALL shards via the shared channel
        let rx = self.message_rx.clone();
        let state = self.state.clone();
        let redis_url = self.config.redis_url.clone();
        let ws_tx_proc = self.ws_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = run_processor(rx, state, redis_url, ws_tx_proc).await {
                error!("Processor error: {}", e);
            }
        });

        // 1b. Spawn the Axum fast-lane WebSocket server
        //     /ws/all       — every MarketMessage, no filter (feature engine, pump/dump)
        //     /ws/:symbol   — filtered by symbol (sniper, per-position monitors)
        let ws_tx_axum = self.ws_tx.clone();
        let app = Router::new()
            .route("/ws/all", get(ws_handler_all))   // static path wins over dynamic
            .route("/ws/:symbol", get(ws_handler))
            .with_state(ws_tx_axum);

        tokio::spawn(async move {
            info!("🚀 Fast-Lane WS Server on ws://0.0.0.0:8080 — direct Rust→Python");
            let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await
                .expect("Failed to bind fast-lane WS port 8080");
            axum::serve(listener, app).await
                .expect("Axum WS server crashed");
        });

        // 2. Fetch all eligible symbols once — avoids N shards hammering the Bybit REST API
        let all_symbols = fetch_all_symbols(self.config.max_symbols).await?;
        let total = all_symbols.len();

        // 3. SHARDING LOGIC: 50 symbols per WebSocket connection.
        //    3 topics per symbol × 3 symbols = 9 args per subscribe message ≤ Bybit's 10-arg limit.
        //    With ~400 symbols this yields ~8 shards — each with its own tokio task + OS TCP buffer.
        //    A memecoin flash-pump floods only its shard; BTC/ETH shard stays at microsecond latency.
        const SHARD_SIZE: usize = 50;
        let chunks: Vec<Vec<String>> = all_symbols
            .chunks(SHARD_SIZE)
            .map(|c| c.to_vec())
            .collect();

        info!("Divided {} symbols into {} shards of ≤{} symbols each", total, chunks.len(), SHARD_SIZE);

        // 4. Spawn a dedicated WebSocket task per shard — all feed into the same message_tx channel
        let mut shard_handles = Vec::with_capacity(chunks.len());

        for (shard_id, symbol_chunk) in chunks.iter().enumerate() {
            let tx = self.message_tx.clone();
            let chunk = symbol_chunk.clone();

            let handle = tokio::spawn(async move {
                let ws = BybitWebSocket::new_with_symbols(chunk, tx);
                let mut reconnect_count: u64 = 0;
                let mut last_disconnect = Instant::now();

                loop {
                    match ws.connect_and_stream(shard_id).await {
                        Ok(_) => {
                            reconnect_count += 1;
                            let gap_ms = last_disconnect.elapsed().as_millis();
                            warn!("[Shard {}] Disconnected cleanly, reconnect #{} (gap={}ms)", shard_id, reconnect_count, gap_ms);
                            last_disconnect = Instant::now();
                        }
                        Err(e) => {
                            reconnect_count += 1;
                            let gap_ms = last_disconnect.elapsed().as_millis();
                            error!("[Shard {}] Error: {} — reconnect #{} (gap={}ms), retry in 5s", shard_id, e, reconnect_count, gap_ms);
                            last_disconnect = Instant::now();
                            tokio::time::sleep(Duration::from_secs(5)).await;
                        }
                    }
                }
            });

            shard_handles.push(handle);

            // Stagger each shard connection by 500ms.
            // Opening 8 WebSockets in 2ms looks like a DDoS to Bybit's Cloudflare edge.
            // 500ms gaps are invisible to trading latency but completely safe for rate limiting.
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Keep the main task alive — all shard tasks run forever
        join_all(shard_handles).await;
        Ok(())
    }
    
}

async fn run_processor(
    rx: Receiver<MarketMessage>,
    state: Arc<SharedState>,
    redis_url: String,
    ws_tx: broadcast::Sender<MarketMessage>,
) -> Result<()> {
    let mut redis = RedisPublisher::new(&redis_url).await?;
    let mut flush_interval = interval(Duration::from_millis(10));
    
    info!("Message processor started");
    
    loop {
        tokio::select! {
            Ok(msg) = rx.recv_async() => {
                // Fast lane: broadcast raw message to all Axum WS subscribers
                // before any Redis I/O — sub-microsecond fan-out
                let _ = ws_tx.send(msg.clone());
                process_message(&state, &msg, &mut redis).await?;
            }
            _ = flush_interval.tick() => {
                redis.flush().await?;
            }
        }
    }
}

async fn process_message(
    state: &SharedState,
    msg: &MarketMessage,
    redis: &mut RedisPublisher,
) -> Result<()> {
    let json = serde_json::to_string(msg)?;
    
    match msg {
        MarketMessage::Tick { symbol, .. } => {
            // Publish to tick channel
            redis.publish(&format!("ticks:{}", symbol), &json).await?;
            
            // Update metrics
            state.metrics.increment_tick(symbol);
        }
        MarketMessage::Orderbook { symbol, bids, asks, timestamp, update_type } => {
            // Always update local orderbook instantly — WS fast lane already has the raw tick
            let is_snapshot = update_type == "snapshot";
            state.orderbooks.update(symbol, bids.clone(), asks.clone(), *timestamp, is_snapshot);
            
            // Throttle Redis to 100ms per symbol — eliminates the pubsub flood
            // DashMap lock is released before any .await (no cross-await lock held)
            let should_publish = {
                let now = Instant::now();
                let mut entry = state.last_ob_publish
                    .entry(symbol.clone())
                    .or_insert_with(|| now - Duration::from_secs(1));
                if now.duration_since(*entry).as_millis() >= 100 {
                    *entry = now;
                    true
                } else {
                    false
                }
            };

            state.metrics.increment_orderbook();

            if should_publish {
                redis.publish(&format!("orderbooks:{}", symbol), &json).await?;
                check_volume_anomaly(state, symbol, redis).await?;
            }
        }
        MarketMessage::Trade { symbol, price, size, side, timestamp } => {
            // Store trade
            let trade = TradeInfo {
                price: *price,
                size: *size,
                side: side.clone(),
                timestamp: *timestamp,
            };
            
            let mut trades = state.recent_trades
                .entry(symbol.clone())
                .or_insert_with(VecDeque::new);
            
            trades.push_back(trade);
            
            // Keep only last 1000 trades
            if trades.len() > 1000 {
                trades.pop_front();
            }
            
            state.metrics.increment_trade();
            redis.publish(&format!("trades:{}", symbol), &json).await?;
        }
        MarketMessage::Kline { symbol, timeframe, .. } => {
            redis.publish(&format!("klines:{}:{}", symbol, timeframe), &json).await?;
        }
    }
    
    Ok(())
}

async fn check_volume_anomaly(
    state: &SharedState,
    symbol: &str,
    redis: &mut RedisPublisher,
) -> Result<()> {
    // Clone only the numeric data we need to avoid holding the map lock across await
    let (volumes, recent_volume) = match state.recent_trades.get(symbol) {
        Some(trades) if trades.len() >= 100 => {
            let vols: Vec<f64> = trades.iter().map(|t| t.size).collect();
            let recent: f64 = trades.iter().rev().take(10).map(|t| t.size).sum::<f64>();
            (vols, recent)
        }
        _ => return Ok(()),
    };
    
    // Calculate volume metrics
    let mean_vol = volumes.iter().sum::<f64>() / volumes.len() as f64;
    let variance = volumes.iter()
        .map(|v| (v - mean_vol).powi(2))
        .sum::<f64>() / volumes.len() as f64;
    let std_vol = variance.sqrt();
    
    let z_score: f64 = (recent_volume - mean_vol) / (std_vol + 1e-10);
    
    // Detect anomaly
    if z_score.abs() > 3.0 {
        let direction = if z_score > 0.0 { "pump" } else { "dump" };
        
        let anomaly = serde_json::json!({
            "type": "volume_anomaly",
            "symbol": symbol,
            "z_score": z_score,
            "direction": direction,
            "recent_volume": recent_volume,
            "mean_volume": mean_vol,
            "timestamp": Utc::now().timestamp_millis(),
        });
        
        redis.publish("anomalies:volume", &anomaly.to_string()).await?;
    }
    
    Ok(())
}

/// Upgrades HTTP → WebSocket; Python bots connect to ws://127.0.0.1:8080/ws/BTCUSDT
async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(target_symbol): Path<String>,
    State(tx): State<broadcast::Sender<MarketMessage>>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_socket(socket, target_symbol, tx))
}

/// Streams raw MarketMessages to a connected Python client — zero Redis hop
async fn handle_socket(
    mut socket: WebSocket,
    target_symbol: String,
    tx: broadcast::Sender<MarketMessage>,
) {
    let mut rx = tx.subscribe();

    while let Ok(msg) = rx.recv().await {
        let msg_symbol = match &msg {
            MarketMessage::Tick     { symbol, .. } => symbol,
            MarketMessage::Trade    { symbol, .. } => symbol,
            MarketMessage::Orderbook{ symbol, .. } => symbol,
            MarketMessage::Kline    { symbol, .. } => symbol,
        };

        if msg_symbol == &target_symbol {
            if let Ok(json) = serde_json::to_string(&msg) {
                if socket.send(WsMessage::Text(json)).await.is_err() {
                    break; // client disconnected
                }
            }
        }
    }
}

/// Upgrades HTTP → WebSocket; multi-symbol consumers connect to ws://127.0.0.1:8080/ws/all
/// Receives every MarketMessage without symbol filtering — used by feature engine, pump/dump detector
async fn ws_handler_all(
    ws: WebSocketUpgrade,
    State(tx): State<broadcast::Sender<MarketMessage>>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_socket_all(socket, tx))
}

/// Streams all MarketMessages to the client with no symbol filter
async fn handle_socket_all(mut socket: WebSocket, tx: broadcast::Sender<MarketMessage>) {
    let mut rx = tx.subscribe();
    while let Ok(msg) = rx.recv().await {
        if let Ok(json) = serde_json::to_string(&msg) {
            if socket.send(WsMessage::Text(json)).await.is_err() {
                break;
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("bybit_ingester=debug")
        .init();
    
    // Load config
    let config = Config::from_env()?;
    
    // Create and run ingester
    let ingester = Ingester::new(config);
    ingester.run().await?;
    
    Ok(())
}
