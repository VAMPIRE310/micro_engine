use std::time::Duration;

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{info, warn, error, trace};

use crate::MarketMessage;
use flume::Sender;

/// Bybit WebSocket client for a single shard — holds a pre-assigned slice of symbols.
/// Reconnect logic is handled by the caller (Ingester::run shard task).
pub struct BybitWebSocket {
    symbols: Vec<String>,
    message_tx: Sender<MarketMessage>,
}

/// WebSocket response
#[derive(Debug, Deserialize)]
struct WsResponse {
    topic: Option<String>,
    #[serde(rename = "type")]
    msg_type: Option<String>,
    data: Option<serde_json::Value>,
}

/// Symbol info from REST API
#[derive(Debug, Deserialize)]
struct SymbolInfo {
    symbol: String,
    #[serde(rename = "quoteCoin")]
    quote_coin: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct InstrumentsResponse {
    result: InstrumentsResult,
}

#[derive(Debug, Deserialize)]
struct InstrumentsResult {
    list: Vec<SymbolInfo>,
}

impl BybitWebSocket {
    /// Create a shard with a pre-assigned symbol slice.
    /// Call `fetch_all_symbols()` in the Ingester, chunk the result, pass each chunk here.
    pub fn new_with_symbols(symbols: Vec<String>, message_tx: Sender<MarketMessage>) -> Self {
        Self { symbols, message_tx }
    }

    /// Connect and stream until disconnected. `shard_id` is for log prefixing only.
    pub async fn connect_and_stream(&self, shard_id: usize) -> Result<()> {
        let url = "wss://stream.bybit.com/v5/public/linear";
        
        info!("[Shard {}] Connecting ({} symbols)", shard_id, self.symbols.len());
        
        let (ws_stream, _) = connect_async(url).await?;
        let (mut write, mut read) = ws_stream.split();
        
        info!("[Shard {}] WebSocket connected", shard_id);
        
        self.subscribe_to_symbols(&mut write, &self.symbols, shard_id).await?;
        
        // ==========================================================
        // 🛡️ DECOUPLED HEARTBEAT TASK
        // Takes ownership of 'write' so it is immune to read blocks
        // ==========================================================
        let ping_task = tokio::spawn(async move {
            let mut heartbeat = interval(Duration::from_secs(15)); // 15s is safer than 20s
            loop {
                heartbeat.tick().await;
                if let Err(e) = write.send(Message::Text(r#"{"req_id":"heartbeat","op":"ping"}"#.to_string())).await {
                    error!("[Shard {}] Heartbeat send failed: {}", shard_id, e);
                    break;
                }
                trace!("[Shard {}] Heartbeat ping sent", shard_id);
            }
        });
        
        // ==========================================================
        // 🚀 DEDICATED READ LOOP
        // ==========================================================
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    // Quick-drop JSON pongs to save CPU before parsing
                    if text.contains(r#""ret_msg":"pong""#) || text.contains(r#""op":"pong""#) {
                        continue;
                    }
                    if let Err(e) = self.handle_message(&text).await {
                        // Log but don't crash the stream on a single bad JSON payload
                        trace!("Failed to parse message: {}", e);
                    }
                }
                Ok(Message::Close(_)) => {
                    warn!("[Shard {}] WebSocket closed by server", shard_id);
                    break;
                }
                Err(e) => {
                    error!("[Shard {}] WebSocket error: {}", shard_id, e);
                    break;
                }
                // Native Ping/Pong frames are automatically handled by tokio-tungstenite under the hood
                _ => {}
            }
        }
        
        ping_task.abort();
        Ok(())
    }
    
    async fn subscribe_to_symbols(
        &self,
        write: &mut (impl SinkExt<Message> + Unpin),
        symbols: &[String],
        shard_id: usize,
    ) -> Result<()> {
        info!("[Shard {}] Preparing multi-timeframe subscriptions...", shard_id);

        // Bybit V5 Kline intervals: 1m, 5m, 15m, 1h(60), 2h(120), 4h(240), 1Day(D)
        // Note: 1-second klines do not exist in Bybit WS. You must build 1s candles
        // locally using the 'publicTrade' stream data.
        let intervals = ["1", "5", "15", "60", "120", "240", "D"];

        let mut all_args = Vec::new();

        // Generate every topic string for every symbol on this shard
        for symbol in symbols {
            all_args.push(format!("tickers.{}", symbol));
            all_args.push(format!("publicTrade.{}", symbol));
            all_args.push(format!("orderbook.50.{}", symbol));

            for interval in intervals {
                all_args.push(format!("kline.{}.{}", interval, symbol));
            }
        }

        // CRITICAL: Bybit strictly enforces a maximum of 10 args per subscribe request.
        // We chunk the entire flattened list of topics by 10.
        for chunk in all_args.chunks(10) {
            let msg = json!({
                "op": "subscribe",
                "args": chunk
            });

            write.send(Message::Text(msg.to_string())).await
                .map_err(|_| anyhow::anyhow!("WebSocket send failed"))?;

            // 50ms delay per 10 topics to prevent buffer overflow and rate limiting
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        info!("[Shard {}] Subscribed to {} individual data streams", shard_id, all_args.len());
        Ok(())
    }
    
    async fn handle_message(&self, text: &str) -> Result<()> {
        let response: WsResponse = serde_json::from_str(text)?;
        
        // Capture update type before partial move (snapshot vs delta for orderbook)
        let msg_type = response.msg_type.clone().unwrap_or_else(|| "snapshot".to_string());
        
        let topic = match response.topic {
            Some(t) => t,
            None => return Ok(()),  // Pong or other control message
        };
        
        let data = match response.data {
            Some(d) => d,
            None => return Ok(()),
        };
        
        // Parse based on topic
        if topic.starts_with("tickers.") {
            self.parse_ticker(&topic, &data).await?;
        } else if topic.starts_with("orderbook.") {
            self.parse_orderbook(&topic, &data, &msg_type).await?;
        } else if topic.starts_with("publicTrade.") {
            self.parse_trade(&topic, &data).await?;
        } else if topic.starts_with("kline.") {
            self.parse_kline(&topic, &data).await?;
        }
        
        Ok(())
    }
    
    async fn parse_ticker(&self, topic: &str, data: &serde_json::Value) -> Result<()> {
        let symbol = topic.split('.').nth(1).unwrap_or("").to_string();
        
        let price: f64 = data["lastPrice"].as_str().unwrap_or("0").parse().unwrap_or(0.0);
        if price <= 0.0 {
            trace!("Invalid ticker price for {}: {}", symbol, price);
            return Ok(());
        }
        
        let msg = MarketMessage::Tick {
            symbol,
            price,
            size: data["volume24h"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
            side: "unknown".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        };
        
        self.message_tx.send_async(msg).await?;
        Ok(())
    }
    
    async fn parse_orderbook(&self, topic: &str, data: &serde_json::Value, msg_type: &str) -> Result<()> {
        let symbol = topic.split('.').nth(2).unwrap_or("").to_string();
        
        let bids: Vec<(f64, f64)> = data["b"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| {
                let price: f64 = v[0].as_str()?.parse().ok()?;
                let size: f64 = v[1].as_str()?.parse().ok()?;
                // Validate: price must be positive (size=0 valid for delta DELETE)
                if price > 0.0 { Some((price, size)) } else { None }
            })
            .collect();
        
        let asks: Vec<(f64, f64)> = data["a"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| {
                let price: f64 = v[0].as_str()?.parse().ok()?;
                let size: f64 = v[1].as_str()?.parse().ok()?;
                if price > 0.0 { Some((price, size)) } else { None }
            })
            .collect();
        
        let msg = MarketMessage::Orderbook {
            symbol,
            bids,
            asks,
            timestamp: chrono::Utc::now().timestamp_millis(),
            update_type: msg_type.to_string(),
        };
        
        self.message_tx.send_async(msg).await?;
        Ok(())
    }
    
    async fn parse_trade(&self, topic: &str, data: &serde_json::Value) -> Result<()> {
        let symbol = topic.split('.').nth(1).unwrap_or("").to_string();
        
        // Data is an array of trades
        if let Some(trades) = data.as_array() {
            for trade in trades {
                let price: f64 = trade["p"].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                let size: f64 = trade["v"].as_str().unwrap_or("0").parse().unwrap_or(0.0);
                // Skip invalid trade data
                if price <= 0.0 || size < 0.0 {
                    warn!("Invalid trade for {}: price={}, size={}", symbol, price, size);
                    continue;
                }
                
                let msg = MarketMessage::Trade {
                    symbol: symbol.clone(),
                    price,
                    size,
                    side: trade["S"].as_str().unwrap_or("Buy").to_string(),
                    timestamp: trade["T"].as_i64().unwrap_or(0),
                };
                
                self.message_tx.send_async(msg).await?;
            }
        }
        
        Ok(())
    }
    
    async fn parse_kline(&self, topic: &str, data: &serde_json::Value) -> Result<()> {
        // topic format: kline.{timeframe}.{symbol}
        let parts: Vec<&str> = topic.split('.').collect();
        let timeframe = parts.get(1).unwrap_or(&"1").to_string();
        let symbol = parts.get(2).unwrap_or(&"").to_string();
        
        if let Some(klines) = data.as_array() {
            for kline in klines {
                let msg = MarketMessage::Kline {
                    symbol: symbol.clone(),
                    timeframe: timeframe.clone(),
                    open: kline["open"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    high: kline["high"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    low: kline["low"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    close: kline["close"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    volume: kline["volume"].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                    timestamp: kline["start"].as_i64().unwrap_or(0),
                };
                
                self.message_tx.send_async(msg).await?;
            }
        }
        
        Ok(())
    }
}

// =============================================================================
// Module-level symbol fetch — called ONCE by the Ingester before sharding.
// Placing it here avoids each shard hitting the REST API independently.
// =============================================================================

/// Fetch all eligible USDT linear perpetuals from Bybit REST, priority-ordered.
/// `max_symbols = 0` means "stream all". Priority tokens guaranteed to be in shard 0.
pub async fn fetch_all_symbols(max_symbols: usize) -> Result<Vec<String>> {
    let url = "https://api.bybit.com/v5/market/instruments-info?category=linear";

    let client = reqwest::Client::new();
    let response = client.get(url).send().await?;
    let data: InstrumentsResponse = response.json().await?;

    let all_symbols: Vec<String> = data.result.list
        .into_iter()
        .filter(|s| s.quote_coin == "USDT" && s.status == "Trading")
        .map(|s| s.symbol)
        .collect();

    // Priority tokens always appear at the front so they land in shard 0 (first spawned).
    // This guarantees BTC/ETH get the freshest, least-congested connection.
    const PRIORITY: &[&str] = &[
        "XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "DOGEUSDT", "AVAXUSDT", "ADAUSDT", "LINKUSDT", "LTCUSDT",
        "DOTUSDT", "MATICUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT",
        "TRXUSDT", "XLMUSDT", "PEPEUSDT", "SHIBUSDT", "ARBUSDT",
    ];

    let (mut prio, rest): (Vec<_>, Vec<_>) = all_symbols.into_iter()
        .partition(|s| PRIORITY.contains(&s.as_str()));
    prio.sort_by_key(|s| PRIORITY.iter().position(|&p| p == s.as_str()).unwrap_or(usize::MAX));
    let mut symbols = prio;
    symbols.extend(rest);

    if max_symbols > 0 && symbols.len() > max_symbols {
        symbols.truncate(max_symbols);
    }

    info!("Fetched {} USDT linear perpetuals from Bybit (max_symbols={})", symbols.len(), max_symbols);
    Ok(symbols)
}