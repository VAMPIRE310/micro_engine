"""
MicroAgent — headless LSTM position manager agent.
Writes experience transitions via PyArrow → Parquet.
Learns online via Polars lazy evaluation + QR-DQN backward pass.
Model weights are persisted to PostgreSQL after every learning pass so
training survives Railway redeployments.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING

from core.parquet_logger import ParquetLogger

if TYPE_CHECKING:
    from core.pg_backend import PgBackend

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

    def __init__(self, device: str = "auto", data_dir: str = "core/data",
                 pg_backend: Optional["PgBackend"] = None):
        # Resolve "auto" → CUDA if available, else CPU
        if device == "auto" or device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.data_dir = data_dir
        self._pg      = pg_backend
        os.makedirs(self.data_dir, exist_ok=True)
        parquet_path = os.path.join(self.data_dir, "micro_experience.parquet")

        self.model = _LifecycleNetwork().to(self.device)
        self.model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # LSTM hidden state — persists across ticks within one open position,
        # reset to None when a trade closes (see reset_sequence).
        self._hidden: Optional[tuple] = None

        # ParquetLogger: auto-flushes every 10 rows, snappy-compressed, corruption-safe.
        # Wired to PgBackend so every flush is mirrored to PostgreSQL and the
        # local file is seeded from PostgreSQL on a cold Railway start.
        self._pq_logger = ParquetLogger(
            filepath=parquet_path,
            flush_size=10,
            pg_backend=pg_backend,
            table_name="micro_experience",
        )

        log.info("MicroAgent initialised on %s. Parquet backend: %s", self.device, parquet_path)

    # ── Weights ─────────────────────────────────────────────────────────────

    def load_weights(self, path: str) -> None:
        """
        Load model weights, preferring the latest PostgreSQL checkpoint so that
        online training persists across Railway redeployments.  Falls back to
        the local .pt file, then random initialisation.
        """
        # ── 1. PostgreSQL checkpoint (most recent trained weights) ───────────
        if self._pg is not None and self._pg.available:
            pg_state = self._pg.load_model("lifecycle_lstm")
            if pg_state is not None:
                model_state = self.model.state_dict()
                compatible  = {
                    k: v for k, v in pg_state.items()
                    if k in model_state and v.shape == model_state[k].shape
                }
                skipped = [k for k in pg_state if k not in compatible]
                if skipped:
                    log.warning(
                        "Skipped %d PG checkpoint key(s) with incompatible shapes: %s",
                        len(skipped), skipped,
                    )
                self.model.load_state_dict(compatible, strict=False)
                log.info("Model weights loaded from PostgreSQL checkpoint.")
                return

        # ── 2. Local .pt file ────────────────────────────────────────────────
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
        """
        QR-DQN backward pass over the most recent 256 experience rows.

        Reads via ParquetLogger.read_polars() which merges:
          1. local parquet file (fast, already written)
          2. in-memory buffer   (rows not yet flushed)
          3. PostgreSQL         (historical rows from previous Railway deploys,
                                 via cold-start seed — no extra query at runtime)

        After a successful pass the updated weights are saved to PostgreSQL
        so training survives the next redeployment.
        """
        df = self._pq_logger.read_polars(256)
        if len(df) < 32:
            return
        try:
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

            log.debug("Online learning pass complete. Loss: %.4f  rows: %d",
                      loss.item(), len(df))

            # Persist updated weights to PostgreSQL so the next Railway deploy
            # starts from the trained state rather than the static .pt file.
            if self._pg is not None and self._pg.available:
                self._pg.save_model("lifecycle_lstm", self.model.state_dict())

        except Exception as exc:
            log.error("Polars learning loop failed: %s", exc)
            self.model.eval()
