use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Supported aggregation timeframes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    Tick,       // Raw — no aggregation
    M1,         // 1 minute
    M5,         // 5 minutes
    M15,        // 15 minutes
    H1,         // 1 hour
    H2,         // 2 hours
    H4,         // 4 hours
    D1,         // 1 day
}

impl Timeframe {
    /// All timeframes for multi-timeframe pipeline
    pub fn all() -> Vec<Timeframe> {
        vec![
            Timeframe::M1,
            Timeframe::M5,
            Timeframe::M15,
            Timeframe::H1,
            Timeframe::H2,
            Timeframe::H4,
            Timeframe::D1,
        ]
    }

    /// Duration in seconds for this timeframe
    pub fn duration_secs(&self) -> u64 {
        match self {
            Timeframe::Tick => 0,
            Timeframe::M1 => 60,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::H1 => 3600,
            Timeframe::H2 => 7200,
            Timeframe::H4 => 14400,
            Timeframe::D1 => 86400,
        }
    }

    /// String label for Redis channel naming
    pub fn label(&self) -> &'static str {
        match self {
            Timeframe::Tick => "tick",
            Timeframe::M1 => "1",
            Timeframe::M5 => "5",
            Timeframe::M15 => "15",
            Timeframe::H1 => "60",
            Timeframe::H2 => "120",
            Timeframe::H4 => "240",
            Timeframe::D1 => "D",
        }
    }
}

/// Support/Resistance level
/// Volume profile bucket (price level volume)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfile {
    pub price_level: f64,
    pub volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SRLevel {
    pub price: f64,
    pub strength: f64,
    pub level_type: String,   // "support" | "resistance"
    pub touches: u32,
    pub timeframe: String,
    pub is_strong: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub symbol: String,
    pub timeframe: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub turnover: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub tick_count: u32,
    pub start_time: i64,
    pub end_time: i64,
    pub vwap: f64,
    pub volume_profile: Vec<VolumeProfile>,
    pub sr_levels: Vec<SRLevel>,
    pub strong_sr: Option<SRLevel>,      // strongest level on this TF
    pub role_reversal: bool,             // detected role flip
}

/// In-progress candle being built from live ticks
#[derive(Debug, Clone)]
struct BuildingCandle {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    turnover: f64,
    buy_volume: f64,
    sell_volume: f64,
    tick_count: u32,
    start_time: i64,
    prices: Vec<f64>,
    volumes: Vec<f64>,
    price_volumes: HashMap<u64, (f64, f64, f64)>, // rounded_price -> (volume, buy_vol, sell_vol)
    touches_high: u32,
    touches_low: u32,
}

impl BuildingCandle {
    fn new(first_price: f64, first_size: f64, side: &str, ts: i64) -> Self {
        let (buy_vol, sell_vol) = if side == "Buy" {
            (first_size, 0.0)
        } else {
            (0.0, first_size)
        };
        let mut pv = HashMap::new();
        let key = price_key(first_price);
        pv.insert(key, (first_size, buy_vol, sell_vol));
        Self {
            open: first_price,
            high: first_price,
            low: first_price,
            close: first_price,
            volume: first_size,
            turnover: first_price * first_size,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            tick_count: 1,
            start_time: ts,
            prices: vec![first_price],
            volumes: vec![first_size],
            price_volumes: pv,
            touches_high: 1,
            touches_low: 1,
        }
    }

    fn add_tick(&mut self, price: f64, size: f64, side: &str) {
        self.close = price;
        if price > self.high {
            self.high = price;
            self.touches_high = 1;
        } else if (price - self.high).abs() < 1e-6 {
            self.touches_high += 1;
        }
        if price < self.low {
            self.low = price;
            self.touches_low = 1;
        } else if (price - self.low).abs() < 1e-6 {
            self.touches_low += 1;
        }

        self.volume += size;
        self.turnover += price * size;
        self.tick_count += 1;
        self.prices.push(price);
        self.volumes.push(size);

        // Accumulate volume profile
        let key = price_key(price);
        let entry = self.price_volumes.entry(key).or_insert((0.0, 0.0, 0.0));
        entry.0 += size;
        if side == "Buy" {
            self.buy_volume += size;
            entry.1 += size;
        } else {
            self.sell_volume += size;
            entry.2 += size;
        }
    }

    fn finalize(&self, end_time: i64, symbol: &str, tf_label: &str) -> Candle {
        let vwap = if self.volume > 0.0 {
            self.prices.iter().zip(self.volumes.iter())
                .map(|(p, v)| p * v)
                .sum::<f64>() / self.volume
        } else {
            self.close
        };

        // Volume profile (top 20 levels)
        let mut vp: Vec<VolumeProfile> = self.price_volumes.iter()
            .map(|(key, (vol, buy, sell))| {
                let price = unround_price(*key);
                VolumeProfile {
                    price_level: price,
                    volume: *vol,
                    buy_volume: *buy,
                    sell_volume: *sell,
                }
            })
            .collect();
        vp.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap());
        vp.truncate(20);

        // ── Enhanced Multi-Timeframe S/R Detection ──
        let mut sr = Vec::new();
        let mut strongest: Option<SRLevel> = None;
        let mut reversal_detected = false;

        // Resistance (high)
        if self.touches_high >= 2 {
            let strength = (self.touches_high as f64 / self.tick_count.max(1) as f64).min(1.0);
            let level = SRLevel {
                price: self.high,
                strength,
                level_type: "resistance".to_string(),
                touches: self.touches_high,
                timeframe: tf_label.to_string(),
                is_strong: strength > 0.68,
            };
            sr.push(level.clone());
            if strongest.is_none() || level.strength > strongest.as_ref().unwrap().strength {
                strongest = Some(level);
            }
        }

        // Support (low)
        if self.touches_low >= 2 {
            let strength = (self.touches_low as f64 / self.tick_count.max(1) as f64).min(1.0);
            let level = SRLevel {
                price: self.low,
                strength,
                level_type: "support".to_string(),
                touches: self.touches_low,
                timeframe: tf_label.to_string(),
                is_strong: strength > 0.68,
            };
            sr.push(level.clone());
            if strongest.is_none() || level.strength > strongest.as_ref().unwrap().strength {
                strongest = Some(level);
            }
        }

        if let Some(strong) = &strongest {
            reversal_detected = strong.strength > 0.75 && strong.touches >= 4;
        }

        Candle {
            symbol: symbol.to_string(),
            timeframe: tf_label.to_string(),
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            turnover: self.turnover,
            buy_volume: self.buy_volume,
            sell_volume: self.sell_volume,
            tick_count: self.tick_count,
            start_time: self.start_time,
            end_time,
            vwap,
            volume_profile: vp,
            sr_levels: sr,
            strong_sr: strongest,
            role_reversal: reversal_detected,
        }
    }
}

/// Round price to nearest 0.1% for volume profile bucketing
fn price_key(price: f64) -> u64 {
    let pct = price * 0.001;
    (price / pct).round() as u64
}

fn unround_price(key: u64) -> f64 {
    key as f64 * 0.001
}

/// Per-symbol aggregation state — holds one BuildingCandle per timeframe
#[derive(Debug)]
struct SymbolAggState {
    /// Current building candles: timeframe -> candle
    building: HashMap<Timeframe, BuildingCandle>,
    /// Last trade price for gap-fill on new candle
    last_price: f64,
    /// Last timestamp seen
    last_ts: i64,
    /// Recent prices for ATR calculation (keep last N closes)
    recent_closes: Vec<f64>,
}

impl SymbolAggState {
    fn new() -> Self {
        Self {
            building: HashMap::new(),
            last_price: 0.0,
            last_ts: 0,
            recent_closes: Vec::with_capacity(50),
        }
    }

    /// Calculate ATR as percentage of current price
    fn atr_pct(&self) -> Option<f64> {
        if self.recent_closes.len() < 2 {
            return None;
        }
        let period = self.recent_closes.len().min(14);
        let closes = &self.recent_closes[self.recent_closes.len().saturating_sub(period)..];
        if closes.len() < 2 {
            return None;
        }
        let mut tr_sum = 0.0;
        for i in 1..closes.len() {
            let prev = closes[i - 1];
            let curr = closes[i];
            let tr = (curr - prev).abs();
            tr_sum += tr;
        }
        let atr = tr_sum / (closes.len() - 1) as f64;
        let current = closes.last().copied()?;
        if current > 0.0 {
            Some(atr / current)
        } else {
            None
        }
    }
}

/// Multi-timeframe aggregator — runs entirely in Rust memory.
/// Bins raw ticks into candles, emits completed candles to caller.
pub struct TimeframeAggregator {
    /// symbol -> aggregation state
    states: DashMap<String, Mutex<SymbolAggState>>,
    /// Timeframes to aggregate (usually all except Tick)
    timeframes: Vec<Timeframe>,
}

impl TimeframeAggregator {
    pub fn new() -> Self {
        Self {
            states: DashMap::new(),
            timeframes: Timeframe::all(),
        }
    }

    /// Process a single trade tick. Returns Vec of completed candles that just closed.
    pub fn process_trade(
        &self,
        symbol: &str,
        price: f64,
        size: f64,
        side: &str,
        timestamp_ms: i64,
    ) -> Vec<Candle> {
        if price <= 0.0 || size < 0.0 {
            return vec![];
        }

        let entry = self.states
            .entry(symbol.to_string())
            .or_insert_with(|| Mutex::new(SymbolAggState::new()));

        let mut state = entry.lock();
        let mut completed = Vec::new();

        for tf in &self.timeframes {
            let candle_start = align_timestamp(timestamp_ms, tf.duration_secs());
            let prev_start = if state.last_ts > 0 {
                align_timestamp(state.last_ts, tf.duration_secs())
            } else {
                candle_start
            };

            if candle_start > prev_start && state.last_ts > 0 {
                if let Some(prev_candle) = state.building.remove(tf) {
                    let finished = prev_candle.finalize(state.last_ts, symbol, tf.label());
                    state.recent_closes.push(finished.close);
                    if state.recent_closes.len() > 50 {
                        state.recent_closes.remove(0);
                    }
                    completed.push(finished);
                }
            }

            match state.building.get_mut(tf) {
                Some(candle) => {
                    candle.add_tick(price, size, side);
                }
                None => {
                    let open_price = if state.last_price > 0.0 && candle_start > prev_start {
                        state.last_price
                    } else {
                        price
                    };
                    let mut new_candle = BuildingCandle::new(open_price, size, side, candle_start);
                    if open_price != price {
                        new_candle.add_tick(price, size, side);
                    }
                    state.building.insert(*tf, new_candle);
                }
            }
        }

        state.last_price = price;
        state.last_ts = timestamp_ms;

        completed
    }

    /// Process a ticker update (no size info, just price update for OHLC)
    pub fn process_ticker(
        &self,
        symbol: &str,
        price: f64,
        timestamp_ms: i64,
    ) -> Vec<Candle> {
        self.process_trade(symbol, price, 0.0, "unknown", timestamp_ms)
    }

    /// Force-flush all building candles
    pub fn flush_all(&self, timestamp_ms: i64) -> Vec<Candle> {
        let mut all_completed = Vec::new();

        for entry in self.states.iter() {
            let mut state = entry.lock();
            for (tf, candle) in state.building.drain() {
                let finished = candle.finalize(timestamp_ms, entry.key().as_str(), tf.label());
                all_completed.push(finished);
            }
        }

        all_completed
    }

    /// Get ATR percentage for a symbol
    pub fn get_atr_pct(&self, symbol: &str) -> Option<f64> {
        self.states.get(symbol)?.lock().atr_pct()
    }

    /// Get number of tracked symbols
    pub fn symbol_count(&self) -> usize {
        self.states.len()
    }

    /// Get stats per symbol
    pub fn get_symbol_stats(&self, symbol: &str) -> Option<SymbolAggStats> {
        self.states.get(symbol).map(|entry| {
            let state = entry.lock();
            SymbolAggStats {
                timeframes_active: state.building.len(),
                last_price: state.last_price,
                last_ts: state.last_ts,
            }
        })
    }
}

/// Align a timestamp (ms) to the start of its timeframe window
fn align_timestamp(ts_ms: i64, window_secs: u64) -> i64 {
    let secs = ts_ms / 1000;
    let aligned_secs = (secs / (window_secs as i64)) * (window_secs as i64);
    aligned_secs * 1000
}

#[derive(Debug)]
pub struct SymbolAggStats {
    pub timeframes_active: usize,
    pub last_price: f64,
    pub last_ts: i64,
}

/// Batch aggregator — collects completed candles and emits them on interval
pub struct BatchedAggregator {
    aggregator: Arc<TimeframeAggregator>,
    pending: Mutex<Vec<Candle>>,
    publish_interval_ms: u64,
}

impl BatchedAggregator {
    pub fn new(aggregator: Arc<TimeframeAggregator>, publish_interval_ms: u64) -> Self {
        Self {
            aggregator,
            pending: Mutex::new(Vec::new()),
            publish_interval_ms,
        }
    }

    pub fn process_trade(
        &self,
        symbol: &str,
        price: f64,
        size: f64,
        side: &str,
        timestamp_ms: i64,
    ) {
        let completed = self.aggregator.process_trade(symbol, price, size, side, timestamp_ms);
        if !completed.is_empty() {
            let mut pending = self.pending.lock();
            pending.extend(completed);
        }
    }

    pub fn process_ticker(&self, symbol: &str, price: f64, timestamp_ms: i64) {
        let completed = self.aggregator.process_ticker(symbol, price, timestamp_ms);
        if !completed.is_empty() {
            let mut pending = self.pending.lock();
            pending.extend(completed);
        }
    }

    pub fn drain_pending(&self) -> Vec<Candle> {
        let mut pending = self.pending.lock();
        std::mem::take(&mut *pending)
    }

    pub fn flush_all(&self, timestamp_ms: i64) -> Vec<Candle> {
        let mut all = self.aggregator.flush_all(timestamp_ms);
        {
            let mut pending = self.pending.lock();
            all.extend(std::mem::take(&mut *pending));
        }
        all
    }

    /// Get ATR percentage for a symbol (delegates to inner aggregator)
    pub fn get_atr_pct(&self, symbol: &str) -> Option<f64> {
        self.aggregator.get_atr_pct(symbol)
    }

    pub fn get_interval(&self) -> Duration {
        Duration::from_millis(self.publish_interval_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_building() {
        let agg = TimeframeAggregator::new();
        let now = 1704067200000i64;

        let completed = agg.process_trade("BTCUSDT", 42000.0, 1.5, "Buy", now);
        assert!(completed.is_empty());

        let completed = agg.process_trade("BTCUSDT", 42100.0, 2.0, "Sell", now + 61000);
        assert_eq!(completed.len(), 6);

        let m1 = &completed[0];
        assert_eq!(m1.open, 42000.0);
        assert_eq!(m1.high, 42100.0);
        assert_eq!(m1.low, 42000.0);
        assert_eq!(m1.close, 42100.0);
        assert_eq!(m1.volume, 1.5);
        assert!(!m1.volume_profile.is_empty());
        assert_eq!(m1.symbol, "BTCUSDT");
    }

    #[test]
    fn test_timestamp_alignment() {
        let ts = 1704067234000i64;
        assert_eq!(align_timestamp(ts, 60), 1704067200000);
        assert_eq!(align_timestamp(ts, 300), 1704067200000);
    }
}

