"""
MicroAgent — headless LSTM position manager agent.
Writes experience transitions via PyArrow → Parquet.
Learns online via Polars lazy evaluation + QR-DQN backward pass.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from typing import Dict, Any, Optional

log = logging.getLogger("MicroAgent")


class _LifecycleNetwork(nn.Module):
    """QR-DQN LSTM matching the locally-trained lifecycle_lstm.pt architecture."""

    def __init__(
        self,
        input_dim: int = 160,
        hidden_dim: int = 256,
        num_actions: int = 4,
        num_quantiles: int = 32,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.out = nn.Linear(128, num_actions * num_quantiles)

    def forward(self, x: torch.Tensor, hidden=None):
        out, hidden = self.lstm(x, hidden)
        x = F.relu(self.fc1(out[:, -1, :]))
        x = self.out(x)
        return x.view(-1, self.num_actions, self.num_quantiles), hidden


class MicroAgent:
    """
    Headless LSTM agent for Railway deployment.

    Actions:
        0 = HOLD
        1 = HEDGE
        2 = EXIT
        3 = WAVE_ADD
    """

    _ACTION_MAP = {0: "HOLD", 1: "HEDGE", 2: "EXIT", 3: "WAVE_ADD"}
    _ACTION_IDX = {"HOLD": 0, "HEDGE": 1, "EXIT": 2, "WAVE_ADD": 3}

    def __init__(self, device: str = "cpu", data_dir: str = "core/data"):
        self.device = device
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self._parquet_path = os.path.join(self.data_dir, "micro_experience.parquet")

        self.model = _LifecycleNetwork().to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # LSTM hidden state — persists across ticks within one open position,
        # reset to None when a trade closes (see reset_sequence).
        self._hidden: Optional[tuple] = None

        # PyArrow write buffer — flushed to Parquet every _BUFFER_LIMIT steps.
        self._write_buffer: list = []
        self._BUFFER_LIMIT = 10

        log.info("MicroAgent initialised on %s. Parquet backend: %s", self.device, self._parquet_path)

    # ── Weights ─────────────────────────────────────────────────────────────

    def load_weights(self, path: str) -> None:
        """Load locally-trained lifecycle_lstm.pt weights."""
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            log.info("Master LSTM weights loaded from %s", path)
        else:
            log.warning("No weights at %s — running with random init.", path)

    # ── Inference ────────────────────────────────────────────────────────────

    def reset_sequence(self) -> None:
        """Clear LSTM hidden state.  Call this every time a trade closes."""
        self._hidden = None

    def predict(self, state_tensor: torch.Tensor, market_data: Dict[str, Any]) -> str:
        """
        Run one inference step.

        Args:
            state_tensor: 1-D feature vector of shape [160].
            market_data:  dict with at least {'price': float, 'volume': float}.

        Returns:
            Action string: 'HOLD' | 'HEDGE' | 'EXIT' | 'WAVE_ADD'
        """
        self.model.eval()
        # LSTM expects [Batch=1, Seq=1, Features=160]
        x = state_tensor.view(1, 1, -1).to(self.device)

        with torch.no_grad():
            quantiles, self._hidden = self.model(x, self._hidden)
            q_values = quantiles.mean(dim=2).squeeze(0)  # [4]
            action_idx = int(torch.argmax(q_values).item())

        return self._ACTION_MAP.get(action_idx, "HOLD")

    # ── Online learning ──────────────────────────────────────────────────────

    def self_reflect(self, state: torch.Tensor, action: str, reward: float) -> None:
        """
        Buffer experience → flush to Parquet → trigger Polars learning pass.

        PyArrow keeps RAM overhead minimal for Railway's constrained containers.
        """
        if action not in self._ACTION_IDX:
            return

        self._write_buffer.append(
            {
                "state": state.cpu().numpy().tolist(),
                "action": self._ACTION_IDX[action],
                "reward": float(reward),
            }
        )

        if len(self._write_buffer) >= self._BUFFER_LIMIT:
            self._flush_buffer()
            self._learn_from_parquet()

    def _flush_buffer(self) -> None:
        table = pa.Table.from_pylist(self._write_buffer)
        if os.path.exists(self._parquet_path):
            existing = pq.read_table(self._parquet_path)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, self._parquet_path)
        n = len(self._write_buffer)
        self._write_buffer.clear()
        log.debug("Flushed %d experiences to Parquet.", n)

    def _learn_from_parquet(self) -> None:
        """Polars lazy tail-read → QR-DQN backward pass."""
        if not os.path.exists(self._parquet_path):
            return
        try:
            df = pl.scan_parquet(self._parquet_path).tail(256).collect()
            if len(df) < 32:
                return

            self.model.train()

            state_t = torch.tensor(
                df["state"].to_list(), dtype=torch.float32
            ).unsqueeze(1).to(self.device)          # [B, 1, 160]
            action_t = torch.tensor(
                df["action"].to_numpy(), dtype=torch.long
            ).to(self.device)
            reward_t = torch.tensor(
                df["reward"].to_numpy(), dtype=torch.float32
            ).to(self.device)

            quantiles, _ = self.model(state_t)       # [B, 4, 32]
            q_values = quantiles.mean(dim=2)          # [B, 4]
            current_q = q_values.gather(
                1, action_t.unsqueeze(1)
            ).squeeze(1)                              # [B]

            loss = F.smooth_l1_loss(current_q, reward_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()

            log.debug("Online learning pass complete. Loss: %.4f", loss.item())
        except Exception as exc:
            log.error("Polars learning loop failed: %s", exc)
