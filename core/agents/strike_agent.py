import os
import time
import random
import numpy as np
import threading
from collections import deque
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(ROOT_DIR, "merged_models", "strike_mlp.pt")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from core.agents.base_agent import BaseAgent

class StrikeNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, num_quantiles: int):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_actions * num_quantiles)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x.view(-1, self.num_actions, self.num_quantiles)

class StrikeAgent(BaseAgent):
    """
    QR-DQN (MLP) Strike Trigger.
    Finds the 'Edge' with distributional classification.
    """
    
    AGENT_ROLE = "strike_trigger"
    NUM_ACTIONS = 3 # 0: Hold, 1: Long, 2: Short
    NUM_QUANTILES = 51

    def __init__(self, config: Dict = None, model_path: str = None, redis_client: Any = None):
        super().__init__(config, model_path, redis_client)
        from core.utils.parquet_logger import ParquetLogger
        _log_path = os.path.join(ROOT_DIR, "data", "strike_audit.parquet")
        self.parquet_log = ParquetLogger(_log_path, flush_size=50)

    def _build_network(self) -> nn.Module:
        return StrikeNetwork(self.FEATURE_DIM, self.NUM_ACTIONS, self.NUM_QUANTILES)

    def predict(self, tensor_state: torch.Tensor, 
                market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Ensure batch dimension
        if tensor_state.dim() == 1:
            tensor_state = tensor_state.unsqueeze(0)
            
        quantiles = self.model(tensor_state) # [Batch, Actions, Quantiles]
        q_values = quantiles.mean(dim=2) # [Batch, Actions]
        
        action_idx = torch.argmax(q_values, dim=1).item()
        confidence = torch.softmax(q_values, dim=1)[0, action_idx].item()

        # Check Mood
        self._evaluate_mood(win_rate=0.5, current_drawdown=0.0, regime=0)
        if confidence < self.confidence_threshold:
            action_idx = 0 # Default to HOLD if underconfident
        
        action_map = {0: "HOLD", 1: "LONG", 2: "SHORT"}
        
        return {
            "action": action_map[action_idx],
            "action_idx": action_idx,
            "confidence": confidence,
            "q_values": q_values.cpu().numpy().tolist()
        }

    def store(self, state: Any, action_idx: int, reward: float, next_state: Any, done: bool):
        """Store transition to Parquet and PostgreSQL."""
        def _cpu_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return np.asarray(x, dtype=np.float32)
            
        s_arr = _cpu_numpy(state)
        ns_arr = _cpu_numpy(next_state)

        # Log locally to Parquet
        self.parquet_log.log({
            "ts": time.time(), "action": int(action_idx),
            "reward": float(reward), "done": int(done),
        })

        # Push to PostgreSQL via BaseConsciousAgent method in a background thread
        threading.Thread(
            target=self._persist_to_db,
            args=(s_arr, float(action_idx), float(reward), ns_arr, bool(done), 0),
            daemon=True
        ).start()

    def evaluate(self, n_recent: int = 500) -> Dict[str, Any]:
        """Polars-based self-evaluation from parquet audit log."""
        import polars as pl
        df = self.parquet_log.read_polars(n_recent)
        if df.is_empty():
            return {"status": "no_data"}
        
        n = len(df)
        total_rew = float(df["reward"].sum()) if "reward" in df.columns else 0.0
        win_rate  = float((df["reward"] > 0).sum()) / max(n, 1) if "reward" in df.columns else 0.0

        summary = f"[Strike] EVAL | n={n} reward={total_rew:.2f} win_rate={win_rate:.1%}"
        self.think(summary, category="reflection", tone="neutral", confidence=0.85)
        return {"n": n, "total_reward": total_rew, "win_rate": win_rate}