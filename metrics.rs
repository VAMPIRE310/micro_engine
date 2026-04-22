use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use dashmap::DashMap;

/// Metrics tracking
pub struct Metrics {
    /// Total ticks received
    pub ticks_total: AtomicU64,
    /// Ticks per symbol
    pub ticks_by_symbol: DashMap<String, AtomicU64>,
    /// Total orderbook updates
    pub orderbooks_total: AtomicU64,
    /// Total trades
    pub trades_total: AtomicU64,
    /// Messages per second
    pub messages_per_second: AtomicU64,
    /// Last calculation time
    last_calculation: std::sync::Mutex<Instant>,
    /// Message count since last calculation
    message_count: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            ticks_total: AtomicU64::new(0),
            ticks_by_symbol: DashMap::new(),
            orderbooks_total: AtomicU64::new(0),
            trades_total: AtomicU64::new(0),
            messages_per_second: AtomicU64::new(0),
            last_calculation: std::sync::Mutex::new(Instant::now()),
            message_count: AtomicU64::new(0),
        }
    }
    
    /// Increment tick counter
    pub fn increment_tick(&self, symbol: &str) {
        self.ticks_total.fetch_add(1, Ordering::Relaxed);
        
        let entry = self.ticks_by_symbol
            .entry(symbol.to_string())
            .or_insert_with(|| AtomicU64::new(0));
        entry.fetch_add(1, Ordering::Relaxed);
        
        self.increment_message_count();
    }
    
    /// Increment orderbook counter
    pub fn increment_orderbook(&self) {
        self.orderbooks_total.fetch_add(1, Ordering::Relaxed);
        self.increment_message_count();
    }
    
    /// Increment trade counter
    pub fn increment_trade(&self) {
        self.trades_total.fetch_add(1, Ordering::Relaxed);
        self.increment_message_count();
    }
    
    fn increment_message_count(&self) {
        let count = self.message_count.fetch_add(1, Ordering::Relaxed);
        
        // Calculate MPS every second
        let mut last = self.last_calculation.lock().unwrap();
        if last.elapsed().as_secs() >= 1 {
            self.messages_per_second.store(count, Ordering::Relaxed);
            self.message_count.store(0, Ordering::Relaxed);
            *last = Instant::now();
            
            // Log metrics
            tracing::info!(
                "Metrics - Ticks: {}, OBs: {}, Trades: {}, MPS: {}",
                self.ticks_total.load(Ordering::Relaxed),
                self.orderbooks_total.load(Ordering::Relaxed),
                self.trades_total.load(Ordering::Relaxed),
                count
            );
        }
    }
    
    /// Get current stats
    pub fn get_stats(&self) -> MetricsStats {
        MetricsStats {
            ticks_total: self.ticks_total.load(Ordering::Relaxed),
            orderbooks_total: self.orderbooks_total.load(Ordering::Relaxed),
            trades_total: self.trades_total.load(Ordering::Relaxed),
            messages_per_second: self.messages_per_second.load(Ordering::Relaxed),
            symbols_tracked: self.ticks_by_symbol.len(),
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct MetricsStats {
    pub ticks_total: u64,
    pub orderbooks_total: u64,
    pub trades_total: u64,
    pub messages_per_second: u64,
    pub symbols_tracked: usize,
}
