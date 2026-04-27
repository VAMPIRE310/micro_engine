"""
NegotiatorAgent — LSTM Regime-Conditioned SAC for trade sizing.
Standalone module for NEO SUPREME — state_dim=160, action_dim=3.
Outputs: margin_scale, tp_scale, sl_scale — each rescaled to [0, 1].
Architecture ported from regime_sac_agent.py (action_dim patched to 3).
"""
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(ROOT_DIR, "merged_models", "negotiator_sac.pt")

import logging
import random
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from sqlalchemy import text

from core.utils.parquet_logger import ParquetLogger

try:
    from database.connection import get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

log = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING_UP      = "TRENDING_UP"
    TRENDING_DOWN    = "TRENDING_DOWN"
    RANGING          = "RANGING"
    HIGH_VOLATILITY  = "HIGH_VOLATILITY"
    LOW_VOLATILITY   = "LOW_VOLATILITY"
    CRASH            = "CRASH"
    PUMP             = "PUMP"
    ACCUMULATION     = "ACCUMULATION"
    DISTRIBUTION     = "DISTRIBUTION"

REGIME_TO_INDEX: Dict[MarketRegime, int] = {
    MarketRegime.TRENDING_UP:     0,
    MarketRegime.TRENDING_DOWN:   1,
    MarketRegime.RANGING:         2,
    MarketRegime.HIGH_VOLATILITY: 3,
    MarketRegime.LOW_VOLATILITY:  4,
    MarketRegime.CRASH:           5,
    MarketRegime.PUMP:            6,
    MarketRegime.ACCUMULATION:    7,
    MarketRegime.DISTRIBUTION:    8,
}

class RegimeConditionedBuffer:
    def __init__(self, capacity_per_regime: int = 100_000):
        self.capacity = capacity_per_regime
        self._buffers: Dict[MarketRegime, deque] = {
            r: deque(maxlen=capacity_per_regime) for r in MarketRegime
        }
        self._current_regime = MarketRegime.RANGING

    def push(self, experience: dict):
        regime = experience.get("regime", self._current_regime)
        self._buffers[regime].append(experience)

    def set_regime(self, regime: MarketRegime):
        self._current_regime = regime

class RegimeEmbedding(nn.Module):
    def __init__(self, num_regimes: int = 9, embedding_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_regimes, embedding_dim)

    def forward(self, regime_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(regime_indices)

class LSTMCascadeActor(nn.Module):
    def __init__(self, state_dim=160, action_dim=3, hidden_dim=256,
                 num_regimes=9, regime_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.regime_emb = nn.Embedding(num_regimes, regime_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + regime_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_head = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, state_seq, regime_idx):
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(1)
        _, (h_n, _) = self.lstm(state_seq)
        lstm_out = self.layer_norm(h_n[-1])
        r_emb = self.regime_emb(regime_idx)
        x = torch.cat([lstm_out, r_emb], dim=-1)
        x = self.fusion(x)
        mean    = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), min=-20, max=2)
        return mean, log_std

    def sample(self, state_seq, regime_idx, deterministic=False):
        mean, log_std = self.forward(state_seq, regime_idx)
        if deterministic:
            action   = torch.tanh(mean)
            log_prob = torch.zeros_like(mean).sum(dim=-1, keepdim=True)
            return action, log_prob
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t    = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class SACActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, hidden_dim)
        self.mean    = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x       = F.relu(self.fc1(x))
        x       = F.relu(self.fc2(x))
        mean    = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def sample(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(x)
        std      = log_std.exp()
        normal   = Normal(mean, std)
        x_t      = normal.rsample()
        action   = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob  = log_prob.sum(1, keepdim=True)
        return action, log_prob

class NegotiatorAgent:
    AGENT_ROLE  = "negotiator"
    ACTION_DIM  = 3 
    STATE_DIM   = 160

    def __init__(self,
                 state_dim:   int   = 160,
                 action_dim:  int   = 3,
                 num_regimes: int   = 9,
                 device: str = "cuda"):

        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.num_regimes = num_regimes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Network (Inference only)
        self.actor = LSTMCascadeActor(state_dim, action_dim, num_regimes=num_regimes).to(self.device)

        self.replay_buffer  = RegimeConditionedBuffer(capacity_per_regime=100_000)
        self.current_regime = MarketRegime.RANGING

        _log_path = os.path.join(ROOT_DIR, "data", "negotiator_audit.parquet")
        os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
        self.parquet_log = ParquetLogger(_log_path, flush_size=50)

    def _build_network(self) -> nn.Module:
        return LSTMCascadeActor(self.state_dim, self.action_dim, num_regimes=self.num_regimes)

    def set_regime(self, regime: MarketRegime):
        self.current_regime = regime
        self.replay_buffer.set_regime(regime)

    def select_action(self, state: np.ndarray,
                      regime: Optional[MarketRegime] = None,
                      deterministic: bool = False) -> np.ndarray:
        regime     = regime or self.current_regime
        regime_idx = REGIME_TO_INDEX.get(regime, 2)
        with torch.no_grad():
            state_t  = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            regime_t = torch.tensor([regime_idx], dtype=torch.long, device=self.device)
            action, _ = self.actor.sample(state_t, regime_t, deterministic)
            return action.cpu().numpy()[0]

    def predict(self, tensor_state: torch.Tensor,
                market_data: Optional[Dict[str, Any]] = None,
                regime: Optional[MarketRegime] = None,
                deterministic: bool = True) -> Dict[str, Any]:
        if isinstance(tensor_state, np.ndarray):
            state_np = tensor_state
        else:
            state_np = tensor_state.cpu().numpy()

        if state_np.ndim == 2:
            state_np = state_np[0]

        raw = self.select_action(state_np, regime=regime, deterministic=deterministic)

        margin_scale = float((raw[0] + 1) / 2)
        tp_scale     = float((raw[1] + 1) / 2)
        sl_scale     = float((raw[2] + 1) / 2)

        return {
            "margin_scale": margin_scale,
            "tp_scale":     tp_scale,
            "sl_scale":     sl_scale,
            "raw_action":   raw.tolist(),
        }

    def _persist_to_db(self, s: np.ndarray, a: float, r: float, s2: np.ndarray, done: bool, regime: int):
        if not DB_AVAILABLE:
            return
        try:
            db_gen = get_db()
            session = next(db_gen)
            sql = text("""
                INSERT INTO agent_experiences 
                (agent_name, regime, state_blob, action, reward, next_state_blob, done, ts) 
                VALUES (:name, :reg, :s, :a, :r, :s2, :d, :ts)
            """)
            session.execute(sql, {
                "name": self.AGENT_ROLE,
                "reg": regime,
                "s": s.tobytes(),
                "a": a,
                "r": r,
                "s2": s2.tobytes(),
                "d": int(done),
                "ts": time.time()
            })
            session.commit()
        except Exception as e:
            log.debug(f"[{self.AGENT_ROLE}] DB Persist skipped: {e}")

    def store_transition(self, state, action, reward, next_state, done,
                         regime: Optional[MarketRegime] = None):
        self.replay_buffer.push({
            "state":      state,
            "action":     action,
            "reward":     reward,
            "next_state": next_state,
            "done":       done,
            "regime":     regime or self.current_regime,
        })
        
        regime_val = REGIME_TO_INDEX.get(regime or self.current_regime, 2)
        
        self.parquet_log.log({
            "ts": time.time(),
            "regime": regime_val,
            "reward": float(reward)
        })

        threading.Thread(
            target=self._persist_to_db,
            args=(np.asarray(state, dtype=np.float32), float(action[0]), float(reward), np.asarray(next_state, dtype=np.float32), bool(done), regime_val),
            daemon=True
        ).start()

    def save(self, path: Optional[str] = None):
        path = path or WEIGHTS_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"actor": self.actor.state_dict()}, path)
        log.info("[NegotiatorAgent] checkpoint saved → %s", path)

    def load(self, path: Optional[str] = None):
        path = path or WEIGHTS_PATH
        if not os.path.exists(path):
            log.warning("[NegotiatorAgent] checkpoint not found: %s", path)
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "actor" not in ckpt:
            self.actor.load_state_dict(ckpt, strict=False)
            return
        self.actor.load_state_dict(ckpt["actor"], strict=False)
        log.info("[NegotiatorAgent] checkpoint loaded ← %s", path)

    def evaluate(self, n_recent: int = 500) -> Dict[str, Any]:
        import polars as pl
        df = self.parquet_log.read_polars(n_recent)
        if df.is_empty():
            return {"status": "no_data"}
        n = len(df)
        log.info(f"[Negotiator] EVAL | n={n}")
        return {"n": n}