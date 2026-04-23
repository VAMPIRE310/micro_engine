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
import polars as pl
from typing import Optional

from core.parquet_logger import ParquetLogger

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

    def __init__(self, device: str = "auto", data_dir: str = "core/data"):
        # Resolve "auto" → CUDA if available, else CPU
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        parquet_path = os.path.join(self.data_dir, "micro_experience.parquet")

        self.model = _LifecycleNetwork().to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # LSTM hidden state — persists across ticks within one open position,
        # reset to None when a trade closes (see reset_sequence).
        self._hidden: Optional[tuple] = None

        # ParquetLogger: auto-flushes every 10 rows, snappy-compressed, corruption-safe.
        self._pq_logger = ParquetLogger(filepath=parquet_path, flush_size=10)

        log.info("MicroAgent initialised on %s. Parquet backend: %s", self.device, parquet_path)

    # ── Weights ─────────────────────────────────────────────────────────────

    def load_weights(self, path: str) -> None:
        """Load locally-trained lifecycle_lstm.pt weights."""
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device, weights_only=True)
            model_state = self.model.state_dict()
            compatible = {
                k: v for k, v in state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            skipped = [k for k in state if k not in compatible]
            if skipped:
                log.warning(
                    "Skipped %d checkpoint key(s) with incompatible shapes: %s",
                    len(skipped), skipped,
                )
            self.model.load_state_dict(compatible, strict=False)
            log.info("Master LSTM weights loaded from %s", path)
        else:
            log.warning("No weights at %s — running with random init.", path)

    # ── Inference ────────────────────────────────────────────────────────────

    def reset_sequence(self) -> None:
        """Clear LSTM hidden state.  Call this every time a trade closes."""
        self._hidden = None

    def predict(self, state_tensor: torch.Tensor) -> str:
        """
        Run one inference step.

        Args:
            state_tensor: 1-D feature vector of shape [160].

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
        Log experience via ParquetLogger (auto-flush + snappy compression).
        Triggers online learning pass when the logger flushes.
        """
        if action not in self._ACTION_IDX:
            return

        self._pq_logger.log(
            {
                "state": state.cpu().numpy().tolist(),
                "action": self._ACTION_IDX[action],
                "reward": float(reward),
            }
        )
        # Learn from what's on disk after every flush cycle
        self._learn_from_parquet()

    def shutdown(self) -> None:
        """Force-flush pending experiences on clean shutdown."""
        self._pq_logger.flush()

    def _learn_from_parquet(self) -> None:
        """Polars lazy tail-read → QR-DQN backward pass."""
        path = self._pq_logger.filepath
        if not os.path.exists(path):
            return
        try:
            df = pl.scan_parquet(path).tail(256).collect()
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
