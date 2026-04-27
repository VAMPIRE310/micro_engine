use futures_util::{SinkExt, StreamExt};
use reqwest::Url;
use serde_json::json;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use tracing::{error, info, warn};

/// Command enum to securely pass instructions to the active WebSocket loop
#[derive(Debug)]
pub enum WsCommand {
    Subscribe(String),
    Unsubscribe(String),
}

pub struct DynamicExchangeClient {
    url: String,
    command_tx: mpsc::Sender<WsCommand>,
}

impl DynamicExchangeClient {
    /// Initializes the client and spawns the background connection loop
    pub fn new(exchange_ws_url: &str) -> (Self, mpsc::Receiver<String>) {
        let (command_tx, command_rx) = mpsc::channel::<WsCommand>(100);
        let (data_tx, data_rx) = mpsc::channel::<String>(5000);

        let client = Self {
            url: exchange_ws_url.to_string(),
            command_tx,
        };

        tokio::spawn(Self::run_connection_loop(client.url.clone(), command_rx, data_tx));

        (client, data_rx)
    }

    pub async fn subscribe_symbol(&self, symbol: &str) {
        if let Err(e) = self.command_tx.send(WsCommand::Subscribe(symbol.to_string())).await {
            error!("Failed to send subscribe command for {}: {}", symbol, e);
        }
    }

    pub async fn unsubscribe_symbol(&self, symbol: &str) {
        if let Err(e) = self.command_tx.send(WsCommand::Unsubscribe(symbol.to_string())).await {
            error!("Failed to send unsubscribe command for {}: {}", symbol, e);
        }
    }

    async fn run_connection_loop(
        url: String,
        mut command_rx: mpsc::Receiver<WsCommand>,
        data_tx: mpsc::Sender<String>,
    ) {
        loop {
            info!("🔌 Connecting to Exchange WebSocket: {}", url);
            let parsed_url = Url::parse(&url).expect("Invalid WebSocket URL");

            match connect_async(parsed_url).await {
                Ok((ws_stream, _response)) => {
                    info!("✅ WebSocket Connected successfully.");
                    let (mut write, mut read) = ws_stream.split();

                    loop {
                        tokio::select! {
                            // Listen for dynamic commands from Redis via main.rs
                            Some(cmd) = command_rx.recv() => {
                                let payload = match cmd {
                                    WsCommand::Subscribe(sym) => {
                                        info!("📡 Multiplexing new subscription for: {}", sym);
                                        json!({
                                            "op": "subscribe",
                                            "args": [format!("publicTrade.{}", sym), format!("orderbook.50.{}", sym)]
                                        })
                                    },
                                    WsCommand::Unsubscribe(sym) => {
                                        warn!("🔇 Dropping subscription for: {}", sym);
                                        json!({
                                            "op": "unsubscribe",
                                            "args": [format!("publicTrade.{}", sym), format!("orderbook.50.{}", sym)]
                                        })
                                    }
                                };

                                if let Err(e) = write.send(Message::Text(payload.to_string())).await {
                                    error!("Failed to inject WS command: {}. Reconnecting...", e);
                                    break; 
                                }
                            }

                            // Forward incoming exchange ticks
                            msg = read.next() => {
                                match msg {
                                    Some(Ok(Message::Text(text))) => {
                                        if let Err(_) = data_tx.send(text).await {
                                            error!("Internal data channel full or closed.");
                                        }
                                    }
                                    Some(Ok(Message::Ping(ping))) => {
                                        let _ = write.send(Message::Pong(ping)).await;
                                    }
                                    Some(Ok(Message::Close(_))) | None => {
                                        warn!("⚠️ Exchange closed WebSocket. Reconnecting...");
                                        break;
                                    }
                                    Some(Err(e)) => {
                                        error!("❌ WS read error: {}. Reconnecting...", e);
                                        break;
                                    }
                                    _ => {} 
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("❌ Failed to connect to {}: {}. Retrying in 3s...", url, e);
                }
            }
            sleep(Duration::from_secs(3)).await;
        }
    }
}