use dashmap::DashMap;
use parking_lot::RwLock;

/// Orderbook for a single symbol
#[derive(Debug, Clone)]
pub struct Orderbook {
    pub symbol: String,
    pub bids: Vec<(f64, f64)>,  // (price, size)
    pub asks: Vec<(f64, f64)>,
    pub timestamp: i64,
    pub last_update: i64,
}

impl Orderbook {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: Vec::with_capacity(50),
            asks: Vec::with_capacity(50),
            timestamp: 0,
            last_update: 0,
        }
    }
    
    /// Get best bid
    pub fn best_bid(&self) -> Option<(f64, f64)> {
        self.bids.first().copied()
    }
    
    /// Get best ask
    pub fn best_ask(&self) -> Option<(f64, f64)> {
        self.asks.first().copied()
    }
    
    /// Get mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }
    
    /// Get spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some(ask - bid),
            _ => None,
        }
    }
    
    /// Get spread as percentage
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) => Some((spread / mid) * 10000.0),  // bps
            _ => None,
        }
    }
    
    /// Calculate orderbook imbalance
    pub fn imbalance(&self, depth: usize) -> Option<f64> {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|(_, s)| s).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|(_, s)| s).sum();
        let total = bid_volume + ask_volume;
        
        if total > 0.0 {
            Some((bid_volume - ask_volume) / total)
        } else {
            None
        }
    }
    
    /// Estimate slippage for a given quantity
    pub fn estimate_slippage(&self, qty: f64, side: &str) -> Option<f64> {
        let levels = if side == "Buy" {
            &self.asks
        } else {
            &self.bids
        };
        
        let best_price = levels.first()?.0;
        let mut remaining = qty;
        let mut total_cost = 0.0;
        
        for (price, size) in levels.iter() {
            let fill_qty = remaining.min(*size);
            total_cost += fill_qty * price;
            remaining -= fill_qty;
            
            if remaining <= 0.0 {
                break;
            }
        }
        
        if remaining > 0.0 {
            return None;  // Can't fill entire order
        }
        
        let avg_price = total_cost / qty;
        let slippage = ((avg_price - best_price) / best_price).abs() * 10000.0;  // bps
        
        Some(slippage)
    }
    
    /// Get market depth at price level
    pub fn depth_at_price(&self, price: f64) -> f64 {
        let bid_depth: f64 = self.bids
            .iter()
            .filter(|(p, _)| *p >= price)
            .map(|(_, s)| s)
            .sum();
        
        let ask_depth: f64 = self.asks
            .iter()
            .filter(|(p, _)| *p <= price)
            .map(|(_, s)| s)
            .sum();
        
        bid_depth + ask_depth
    }
    
    /// Update orderbook
    pub fn update(&mut self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>, timestamp: i64) {
        // Sort bids descending (highest first)
        let mut bids = bids;
        bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        // Sort asks ascending (lowest first)
        let mut asks = asks;
        asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        self.bids = bids;
        self.asks = asks;
        self.timestamp = timestamp;
        self.last_update = chrono::Utc::now().timestamp_millis();
    }
    
    /// Apply delta (incremental) orderbook update — merge changed levels only
    /// Bybit delta semantics: size=0 → DELETE level, size>0 → INSERT/UPDATE level
    pub fn apply_delta(&mut self, bid_updates: Vec<(f64, f64)>, ask_updates: Vec<(f64, f64)>, timestamp: i64) -> bool {
        // Reject stale deltas (timestamp must advance or equal)
        if timestamp > 0 && self.timestamp > 0 && timestamp < self.timestamp {
            return false;
        }
        
        // Merge bid updates
        for (price, size) in &bid_updates {
            if *size <= 0.0 {
                // DELETE: remove level at this price
                self.bids.retain(|(p, _)| (*p - *price).abs() > 1e-10);
            } else {
                // UPSERT: update existing or insert new level
                if let Some(existing) = self.bids.iter_mut().find(|(p, _)| (*p - *price).abs() < 1e-10) {
                    existing.1 = *size;
                } else {
                    self.bids.push((*price, *size));
                }
            }
        }
        
        // Merge ask updates
        for (price, size) in &ask_updates {
            if *size <= 0.0 {
                self.asks.retain(|(p, _)| (*p - *price).abs() > 1e-10);
            } else {
                if let Some(existing) = self.asks.iter_mut().find(|(p, _)| (*p - *price).abs() < 1e-10) {
                    existing.1 = *size;
                } else {
                    self.asks.push((*price, *size));
                }
            }
        }
        
        // Re-sort: bids descending (best bid first), asks ascending (best ask first)
        self.bids.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        self.asks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        self.timestamp = timestamp;
        self.last_update = chrono::Utc::now().timestamp_millis();
        true
    }
}

/// Manager for all orderbooks
pub struct OrderbookManager {
    orderbooks: DashMap<String, RwLock<Orderbook>>,
}

impl OrderbookManager {
    pub fn new() -> Self {
        Self {
            orderbooks: DashMap::new(),
        }
    }
    
    /// Update or create orderbook for symbol (snapshot = full replace, delta = merge levels)
    pub fn update(&self, symbol: &str, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>, timestamp: i64, is_snapshot: bool) {
        let entry = self.orderbooks
            .entry(symbol.to_string())
            .or_insert_with(|| RwLock::new(Orderbook::new(symbol.to_string())));
        
        let mut ob = entry.write();
        if is_snapshot {
            ob.update(bids, asks, timestamp);
        } else {
            ob.apply_delta(bids, asks, timestamp);
        }
    }
    
    /// Get orderbook for symbol
    pub fn get(&self, symbol: &str) -> Option<Orderbook> {
        self.orderbooks
            .get(symbol)
            .map(|entry| entry.read().clone())
    }
    
    /// Get best bid/ask for symbol
    pub fn get_best_bid_ask(&self, symbol: &str) -> Option<(f64, f64)> {
        let ob = self.get(symbol)?;
        match (ob.best_bid(), ob.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => Some((bid, ask)),
            _ => None,
        }
    }
    
    /// Get mid price for symbol
    pub fn get_mid_price(&self, symbol: &str) -> Option<f64> {
        self.get(symbol)?.mid_price()
    }
    
    /// Check if can absorb order without excessive slippage
    pub fn can_absorb(&self, symbol: &str, qty: f64, side: &str, max_slippage_bps: f64) -> bool {
        match self.get(symbol) {
            Some(ob) => match ob.estimate_slippage(qty, side) {
                Some(slippage) => slippage <= max_slippage_bps,
                None => false,
            },
            None => false,
        }
    }
    
    /// Get all symbols
    pub fn get_symbols(&self) -> Vec<String> {
        self.orderbooks.iter().map(|entry| entry.key().clone()).collect()
    }
    
    /// Get count
    pub fn len(&self) -> usize {
        self.orderbooks.len()
    }
}

impl Default for OrderbookManager {
    fn default() -> Self {
        Self::new()
    }
}
