// ============================================================================
// NEO SUPREME 2026 - Institutional Trailing Stop & Breakout Manager
// ============================================================================
// This module handles sub-millisecond trailing stops, volume regime classification,
// and stateful S/R breakout monitoring completely independent of the Python GIL.
// ============================================================================

use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::time::Duration;
use redis::AsyncCommands;
use chrono::Utc;
use tracing::{info, debug};
use crate::orderbook::OrderbookManager; // for microprice / imbalance confirmation

// ============================================================================
// 1. VOLUME REGIME CLASSIFICATION
// ============================================================================

/// Represents the current market volatility and volume state.
/// Used to dynamically expand or tighten trailing stop distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeRegime {
    Low,        // Quiet market, tighten trails
    Normal,     // Standard ATR multiplier
    High,       // Widen to avoid wicks
    Breakout,   // Strong volume + ATR expansion → follow-through
    Squeeze,    // Low vol + building pressure
}

/// Classifies the current tick into a VolumeRegime based on momentum and ATR.
pub fn classify_volume_regime(
    atr_pct: f64,
    vol_ratio: f64,
    momentum: f64
) -> VolumeRegime {
    if vol_ratio > 2.8 && atr_pct > 0.018 && momentum.abs() > 0.009 {
        VolumeRegime::Breakout
    } else if vol_ratio > 1.85 || atr_pct > 0.023 {
        VolumeRegime::High
    } else if vol_ratio < 0.55 && atr_pct < 0.007 {
        VolumeRegime::Low
    } else if vol_ratio < 0.85 && atr_pct < 0.011 {
        VolumeRegime::Squeeze
    } else {
        VolumeRegime::Normal
    }
}

// ============================================================================
// 2. STATE STRUCTURES
// ============================================================================

/// Represents a single active, silent, or standby trailing stop order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingEntry {
    pub id: String,
    pub symbol: String,
    pub side: String,
    pub qty: f64,
    pub action_idx: i32,
    pub extremum: f64,
    pub trail_dist: f64,
    pub near_sr_level: f64,
    pub status: TrailingStatus,
    pub created_at_ms: i64,
    pub is_dynamic: bool,
}

/// The lifecycle state of a trailing entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrailingStatus {
    Silent,     // Monitoring S/R but not yet actively ratcheting
    Active,     // Standard trailing stop engaged
    Standby,    // Broke out of S/R, now protecting flipped level
    Triggered,  // Executed
    Cancelled,  // Closed by Python
}

// ============================================================================
// 3. TRAILING QUEUE MANAGER
// ============================================================================

/// Lock-free manager that handles hundreds of trailing stops simultaneously.
#[derive(Clone)]
pub struct TrailingQueueManager {
    entries: Arc<DashMap<String, TrailingEntry>>,
    redis: Arc<redis::Client>,
    active_symbols: Arc<DashMap<String, bool>>,
    orderbooks: Arc<OrderbookManager>, // for volume confirmation
}

impl TrailingQueueManager {
    
    /// Initializes a new lock-free trailing queue manager.
    pub fn new(
        redis_client: redis::Client, 
        orderbooks: Arc<OrderbookManager>
    ) -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            redis: Arc::new(redis_client),
            active_symbols: Arc::new(DashMap::new()),
            orderbooks,
        }
    }

    /// Queues a new wave or hedge intent from Python, assigning dynamic distances.
    pub async fn queue_intent(
        &self,
        symbol: &str,
        side: &str,
        qty: f64,
        action_idx: i32,
        current_price: f64,
        atr_value: f64,
        vol_regime: VolumeRegime,
        near_sr_level: f64,
    ) {
        // Step 1: Assign multiplier based on volume regime
        let regime_mult = match vol_regime {
            VolumeRegime::Breakout => 2.45,
            VolumeRegime::High => 1.95,
            VolumeRegime::Low => 1.05,
            VolumeRegime::Squeeze => 1.35,
            _ => 1.65,
        };

        // Step 2: Calculate raw trail distance
        let mut trail_dist = atr_value * regime_mult;

        // Step 3: Tighten distance if hovering near S/R
        if near_sr_level > 0.0 {
            trail_dist *= 0.62;
        }

        // Step 4: Apply strict symbol-agnostic floor
        trail_dist = trail_dist.max(current_price * 0.0038);

        // Step 5: Format unique ticket ID
        let action_str = if action_idx == 3 { "wave" } else { "hedge" };
        let timestamp = Utc::now().timestamp_millis();
        let id = format!("{}_{}_{}", symbol, action_str, timestamp);

        // Step 6: Determine initial state (Silent if near S/R, otherwise Active)
        let initial_status = if near_sr_level > 0.0 { 
            TrailingStatus::Silent 
        } else { 
            TrailingStatus::Active 
        };

        // Step 7: Construct Entry
        let entry = TrailingEntry {
            id: id.clone(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            qty,
            action_idx,
            extremum: current_price,
            trail_dist,
            near_sr_level,
            status: initial_status,
            created_at_ms: timestamp,
            is_dynamic: true,
        };

        // Step 8: Insert into memory and mark symbol as active
        self.entries.insert(id.clone(), entry.clone());
        self.active_symbols.insert(symbol.to_string(), true);

        // Step 9: Persist to Redis
        if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
            let entry_json = serde_json::to_string(&entry).unwrap_or_default();
            let _: Result<(), _> = conn.hset(format!("trailing:{}", symbol), &id, entry_json).await;
        }

        info!(
            "Trailing queued {} | {} | action={} | regime={:?} | dist={:.5}", 
            symbol, side, action_idx, vol_regime, trail_dist
        );
    }

    /// The hot-path tick processor. Evaluates state for all active entries.
    pub async fn process_tick(
        &self, 
        symbol: &str, 
        price: f64, 
        vol_confirmed: bool, 
        current_atr: f64
    ) {
        let mut to_remove = Vec::new();

        for entry_ref in self.entries.iter() {
            // Filter by symbol
            if entry_ref.symbol != symbol { 
                continue; 
            }
            
            let mut entry = entry_ref.value().clone();

            match entry.status {
                
                // --- SILENT STATE: Waiting to reach S/R ---
                TrailingStatus::Silent => {
                    let reached_buy = entry.side == "Buy" && price >= entry.near_sr_level * 0.999;
                    let reached_sell = entry.side == "Sell" && price <= entry.near_sr_level * 1.001;
                    
                    if reached_buy || reached_sell {
                        entry.status = TrailingStatus::Active;
                        entry.trail_dist = (current_atr * 1.7).max(price * 0.0035);
                        debug!("{} trailing ACTIVATED from Silent at S/R {:.4}", symbol, entry.near_sr_level);
                    }
                }
                
                // --- ACTIVE/STANDBY STATE: Ratcheting and Breakout checking ---
                TrailingStatus::Active | TrailingStatus::Standby => {
                    // Orderbook confirmation: spread, liquidity, and imbalance
                    let ob_confirmed = if let Some(ob) = self.orderbooks.get(symbol) {
                        let spread_ok = ob.spread_pct().map(|s| s < 50.0).unwrap_or(true);
                        let liquidity_ok = self.orderbooks.can_absorb(symbol, entry.qty, &entry.side, 25.0);
                        let imbalance_ok = ob.imbalance(5).map(|i| {
                            (entry.side == "Buy" && i > 0.15) || (entry.side == "Sell" && i < -0.15)
                        }).unwrap_or(true);
                        spread_ok && liquidity_ok && imbalance_ok
                    } else {
                        true // no orderbook data yet, allow through
                    };
                    
                    if entry.side == "Buy" {
                        entry.extremum = entry.extremum.min(price);
                        
                        // Check execution trigger
                        if price >= entry.extremum + entry.trail_dist && vol_confirmed && ob_confirmed {
                            self.trigger_execution(&entry).await;
                            entry.status = TrailingStatus::Triggered;
                            to_remove.push(entry.id.clone());
                        }
                    } else {
                        entry.extremum = entry.extremum.max(price);
                        
                        // Check execution trigger
                        if price <= entry.extremum - entry.trail_dist && vol_confirmed && ob_confirmed {
                            self.trigger_execution(&entry).await;
                            entry.status = TrailingStatus::Triggered;
                            to_remove.push(entry.id.clone());
                        }
                    }

                    // Breakout → Standby on role reversal (New Support/Resistance)
                    if entry.status == TrailingStatus::Active && entry.near_sr_level > 0.0 {
                        let broke_out_buy = entry.side == "Buy" && price > entry.near_sr_level * 1.003;
                        let broke_out_sell = entry.side == "Sell" && price < entry.near_sr_level * 0.997;
                        
                        if broke_out_buy || broke_out_sell {
                            entry.status = TrailingStatus::Standby;
                            entry.trail_dist = (price - entry.near_sr_level).abs() * 0.55;
                            debug!("{} BREAKOUT → Standby on flipped level {:.4}", symbol, price);
                        }
                    }
                }
                _ => {}
            }

            // Write back to DashMap if state changed
            if let Some(mut e) = self.entries.get_mut(&entry.id) {
                *e = entry;
            }
        }

        // Garbage collection
        for id in to_remove {
            self.entries.remove(&id);
        }
    }

    /// Formats the payload and commands the execution engine via Redis Stream.
    async fn trigger_execution(&self, entry: &TrailingEntry) {
        let payload = serde_json::json!({
            "symbol": entry.symbol,
            "action": entry.action_idx,
            "directional_bias": entry.side.to_uppercase(),
            "position_size": entry.qty,
            "confidence_score": 1.0,
            "reason": "rust_trailing_trigger",
            "trail_triggered": true,
            "timestamp_ms": Utc::now().timestamp_millis()
        });

        if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
            let _: Result<(), _> = conn.xadd::<_, _, _, _, ()>(
                "brain:decisions:stream",
                "*",
                &[("symbol", entry.symbol.clone()), ("payload", payload.to_string())]
            ).await;
        }
        
        info!("TRAILING TRIGGERED → {} action={}", entry.symbol, entry.action_idx);
    }

    /// Garbage collects the entries when Python explicitly closes the core/hedge.
    pub async fn cancel_for_symbol(&self, symbol: &str, is_full_close: bool) {
        let mut to_remove = Vec::new();
        
        // Find all active entries for the symbol
        for entry in self.entries.iter() {
            if entry.symbol == symbol {
                to_remove.push(entry.id.clone());
            }
        }
        
        // Remove from memory and Redis
        for id in to_remove {
            if let Some((_, mut e)) = self.entries.remove(&id) {
                e.status = TrailingStatus::Cancelled;
                
                if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
                    let _: Result<(), _> = conn.hdel(format!("trailing:{}", symbol), &id).await;
                }
            }
        }
        
        if is_full_close {
            self.active_symbols.remove(symbol);
        }
    }

    /// Background maintenance loop — cleanup stale entries and sync to Redis
    pub async fn run_maintenance(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            let now_ms = Utc::now().timestamp_millis();
            let mut to_remove = Vec::new();
            for entry in self.entries.iter() {
                // Remove entries older than 1 hour
                if now_ms - entry.created_at_ms > 3_600_000 {
                    to_remove.push(entry.id.clone());
                }
            }
            for id in to_remove {
                if let Some((_, entry)) = self.entries.remove(&id) {
                    if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
                        let _: Result<(), _> = conn.hdel(format!("trailing:{}", entry.symbol), &id).await;
                    }
                }
            }
        }
    }
}
