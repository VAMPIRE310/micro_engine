import os
import random
import time
import threading
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(ROOT_DIR, "merged_models", "lifecycle_lstm.pt")
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Deque
from collections import deque
from core.agents.base_agent import BaseAgent
from core.Advanced_orders.hybrid_volume_trailing import HybridVolumeTrailingStop, HybridStopConfig, TrailingDirection


class LifecycleNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int, num_quantiles: int):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.out = nn.Linear(128, num_actions * num_quantiles)

    def forward(self, x):
        # x shape: [Batch, Sequence, Features]
        _, (h_n, _) = self.lstm(x)
        x = F.relu(self.fc1(h_n[-1]))
        x = self.out(x)
        return x.view(-1, self.num_actions, self.num_quantiles)

class PositionManagerAgent(BaseAgent):
    """
    QR-DQN LSTM Position Manager.
    The 'Survivalist' analyzing price/volume velocity.
    Supports HTF S/R navigation and intermediate hedging.
    """
    
    AGENT_ROLE = "position_manager"
    HIDDEN_DIM = 256
    NUM_ACTIONS = 7 # 0: Hold, 1: Trail, 2: Short/Close, 3: Wave Add, 4: Hedge, 5: Close Wave, 6: Merge Wave
    NUM_QUANTILES = 51
    WINDOW_SIZE = 60 # Increased for better sequence awareness

    _REPLAY_CAP = 2_000
    _BATCH_SIZE = 32
    _GAMMA      = 0.99
    _KAPPA      = 1.0
    _TAU        = 0.005

    def __init__(self, config: Dict = None, model_path: str = None, redis_client: Any = None):
        super().__init__(config, model_path, redis_client)
        self.history: Deque[torch.Tensor] = deque(maxlen=self.WINDOW_SIZE)
        self.is_active = False
        self.core_side = "LONG"
        self.trailing_stop: Optional[HybridVolumeTrailingStop] = None

        # Replay buffer, snapshot, target network, optimizer
        self._replay: deque = deque(maxlen=self._REPLAY_CAP)
        self._prev_seq_snapshot: Optional[List[torch.Tensor]] = None
        self.target_net = self._build_network().to(self.device)
        # torch.compile wraps keys under _orig_mod.* — strip prefix for target net
        _sd = {k.replace("_orig_mod.", ""): v for k, v in self.model.state_dict().items()}
        self.target_net.load_state_dict(_sd, strict=False)
        for p in self.target_net.parameters():
            p.requires_grad = False
        _fused = self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5, weight_decay=1e-4, fused=_fused
        )
        from core.utils.parquet_logger import ParquetLogger
        _log_path = os.path.join(ROOT_DIR, "data", "position_manager_audit.parquet")
        self.parquet_log = ParquetLogger(_log_path, flush_size=50)

    def _build_network(self) -> nn.Module:
        return LifecycleNetwork(self.FEATURE_DIM, self.HIDDEN_DIM, self.NUM_ACTIONS, self.NUM_QUANTILES)

    def update_history(self, tensor_state: torch.Tensor):
        self.history.append(tensor_state)

    def predict(self, tensor_state: torch.Tensor, 
                market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.update_history(tensor_state)
        
        if not self.is_active or len(self.history) < self.WINDOW_SIZE:
            return {"action": "HOLD", "confidence": 1.0, "reason": "Inactive or warming up"}

        # 1. Volume-Based Dynamic Trailing (Survival Rule)
        if market_data and self.trailing_stop:
            current_price = market_data.get('price', 0.0)
            tick_volume = market_data.get('volume', 0.0)
            
            if self.trailing_stop.ingest_tick(current_price, tick_volume):
                self.think(f"S/R BREAKOUT/REVERSAL: {self.trailing_stop.trigger_reason}.", category="action")
                return {"action": "EXIT", "confidence": 1.0, "reason": f"Trailing: {self.trailing_stop.trigger_reason}"}

        # 2. QR-DQN LSTM Hierarchical Inference
        with torch.no_grad():
            sequence = torch.stack(list(self.history)).unsqueeze(0)
            quantiles = self.model(sequence)
            q_values = quantiles.mean(dim=2)
            
            action_idx = torch.argmax(q_values, dim=1).item()
            confidence = torch.softmax(q_values, dim=1)[0, action_idx].item()
        
        action_map = {
            0: "HOLD", 1: "TRAIL", 2: "CLOSE", 
            3: "WAVE_ADD", 4: "HEDGE", 5: "CLOSE_WAVE", 6: "MERGE_WAVE"
        }
        
        return {
            "action": action_map[action_idx],
            "action_idx": action_idx,
            "confidence": confidence,
            "is_active": self.is_active,
            "core_side": self.core_side
        }

    def activate(self, symbol: str, side: str, entry_price: float):
        self.is_active = True
        self.core_side = side
        self._prev_seq_snapshot = None
        direction = TrailingDirection.LONG if side == "LONG" else TrailingDirection.SHORT
        config = HybridStopConfig(direction=direction, entry_price=entry_price)
        self.trailing_stop = HybridVolumeTrailingStop(symbol, config)
        self.think(f"CORE POSITION LOCKED ({side}). Monitoring HTF S/R target.", category="action")

    def deactivate(self):
        self.is_active = False
        self.trailing_stop = None
        self._prev_seq_snapshot = None
        self.history.clear()
        self.think("Position closed. Entering hibernation.", category="action")

    def store(self, prev_state: Any, action_idx: int, reward: float, state: Any, done: bool):
        """
        Snapshot-based store. The flat prev_state/state args from the EC are ignored —
        we use the LSTM history deque snapshots instead (correct temporal context).
        """
        if self._prev_seq_snapshot is None or len(self._prev_seq_snapshot) < self.WINDOW_SIZE:
            # First call — just snapshot and wait for next step
            self._prev_seq_snapshot = list(self.history)
            return
        prev_seq = list(self._prev_seq_snapshot)
        curr_seq = list(self.history)
        self._replay.append((prev_seq, int(action_idx), float(reward), curr_seq, bool(done)))
        self._prev_seq_snapshot = curr_seq
        self.parquet_log.log({
            "ts": time.time(), "action": int(action_idx),
            "reward": float(reward), "done": int(done),
            "replay_len": len(self._replay),
        })

    def optimize(self, current_regime: int = 0) -> Dict[str, Any]:
        """
        Double QR-DQN Bellman update + target network soft-update.
        Called every N steps by the execution engine's continuous learning loop.
        """
        if len(self._replay) < self._BATCH_SIZE:
            return {}

        batch = random.sample(list(self._replay), self._BATCH_SIZE)
        prev_seqs, actions, rewards, curr_seqs, dones = zip(*batch)

        prev_t = torch.stack([torch.stack(s) for s in prev_seqs]).to(self.device)   # [B, 60, 160]
        curr_t = torch.stack([torch.stack(s) for s in curr_seqs]).to(self.device)   # [B, 60, 160]
        act_t  = torch.tensor(actions, dtype=torch.long,    device=self.device)
        rew_t  = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        don_t  = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        self.model.train()

        # Current Q-quantiles for taken actions
        quantiles = self.model(prev_t)                               # [B, A, Q]
        idx_exp   = act_t.view(-1, 1, 1).expand(-1, 1, self.NUM_QUANTILES)
        taken_q   = quantiles.gather(1, idx_exp).squeeze(1)         # [B, Q]

        # Double DQN: online selects best next action, target evaluates it
        with torch.no_grad():
            online_next = self.model(curr_t).mean(dim=2)             # [B, A]
            best_action = online_next.argmax(dim=1)                  # [B]
            tgt_q_all   = self.target_net(curr_t)                    # [B, A, Q]
            idx_tgt     = best_action.view(-1, 1, 1).expand(-1, 1, self.NUM_QUANTILES)
            tgt_q       = tgt_q_all.gather(1, idx_tgt).squeeze(1)   # [B, Q]
            bellman     = rew_t.unsqueeze(1) + self._GAMMA * (1 - don_t).unsqueeze(1) * tgt_q

        # QR-DQN Huber quantile regression loss
        tau   = torch.linspace(0.5 / self.NUM_QUANTILES,
                               1 - 0.5 / self.NUM_QUANTILES,
                               self.NUM_QUANTILES, device=self.device)
        diff  = bellman.unsqueeze(2) - taken_q.unsqueeze(1)          # [B, Q, Q]
        huber = torch.where(diff.abs() <= self._KAPPA,
                            0.5 * diff.pow(2) / self._KAPPA,
                            diff.abs() - 0.5 * self._KAPPA)
        loss  = (torch.abs(tau.unsqueeze(1) - (diff < 0).float()) * huber).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.eval()

        # Soft-update target network
        for p, tp in zip(self.model.parameters(), self.target_net.parameters()):
            tp.data.copy_(self._TAU * p.data + (1 - self._TAU) * tp.data)

        loss_val = float(loss.item())
        self.parquet_log.log({
            "ts": time.time(), "event": "optimize",
            "loss": loss_val, "regime": int(current_regime),
            "replay_len": len(self._replay),
        })
        return {"loss": loss_val}


    def evaluate(self, n_recent: int = 500) -> Dict[str, Any]:
        """Polars-based self-evaluation from parquet audit log."""
        import polars as pl

        df = self.parquet_log.read_polars(n_recent)
        if df.is_empty():
            return {"status": "no_data"}

        stores = df.filter(~pl.col("event").is_not_null()) if "event" in df.columns else df
        opts   = df.filter(pl.col("event") == "optimize") if "event" in df.columns else pl.DataFrame()

        n_stores  = len(stores)
        total_rew = float(stores["reward"].sum()) if "reward" in stores.columns else 0.0
        win_rate  = float((stores["reward"] > 0).sum()) / max(n_stores, 1) if "reward" in stores.columns else 0.0
        avg_loss  = float(opts["loss"].mean()) if (not opts.is_empty() and "loss" in opts.columns) else float("nan")

        summary = (
            f"[PositionManager] EVAL | n={n_stores} reward={total_rew:.2f} "
            f"win_rate={win_rate:.1%} avg_loss={avg_loss:.4f}"
        )
        self.think(summary, category="reflection", tone="neutral", confidence=0.85)
        return {"n": n_stores, "total_reward": total_rew, "win_rate": win_rate, "avg_loss": avg_loss}


