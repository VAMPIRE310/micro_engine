import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Deque
from collections import deque
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(ROOT_DIR, "merged_models", "forecaster_tcn.pt")
from core.agents.base_agent import BaseAgent
from core.utils.parquet_logger import ParquetLogger
import threading
import time
import json
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNRegressor(nn.Module):
    def __init__(self, input_dim, num_channels, output_dim=1):
        super().__init__()
        if isinstance(num_channels, int):
            num_channels = [num_channels] * 4
        self.tcn = TemporalConvNet(input_dim, num_channels)
        self.fc = nn.Linear(num_channels[-1], output_dim)
        self.residual_buffer = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.fc(x)
        return x + self.residual_buffer

class ForecasterAgent(BaseAgent):
    """
    TCN Predictor. Predicts future price deltas.
    Inference only.
    """
    AGENT_ROLE = "forecaster"
    WINDOW_SIZE = 60
    EVALUATION_DELAY_SECONDS = 600  

    def __init__(self, config: Dict = None, model_path: str = None, redis_client: Any = None):
        super().__init__(config, model_path, redis_client)
        self.history = deque(maxlen=self.WINDOW_SIZE)
        self.pending_evaluations = deque()
        self.experience_buffer: deque = deque(maxlen=10_000)
        self.batch_size: int = 64

        _db_path = os.path.join(ROOT_DIR, "data", "forecaster_audit.parquet")
        os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
        self.parquet_log = ParquetLogger(_db_path, flush_size=50)
        
        self.learning_thread = threading.Thread(target=self._hindsight_evaluation_loop, daemon=True)
        self.learning_thread.start()

    def _build_network(self) -> torch.nn.Module:
        return TCNRegressor(self.FEATURE_DIM, num_channels=[128, 128, 64], output_dim=5)

    def adapt_online(self, error: torch.Tensor, alpha: float = 0.01):
        with torch.no_grad():
            if hasattr(self.model, 'residual_buffer'):
                self.model.residual_buffer -= alpha * error
                self.think(f"ONLINE ADAPTATION: Corrected residual bias by {error.abs().mean().item():.6f}", category="reflection")

    def predict(self, tensor_state: torch.Tensor, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.history.append(tensor_state)
        
        if len(self.history) < self.WINDOW_SIZE or market_data is None:
            return {"delta_t10": 0.0, "ready": False}

        sequence = torch.stack(list(self.history)).transpose(0, 1).unsqueeze(0)
        out = self.model(sequence) 
        
        pred_delta = float(out[0, 0].item())
        tp_offset = float(out[0, 3].item()) if out.shape[1] > 3 else 0.0
        sl_offset = float(out[0, 4].item()) if out.shape[1] > 4 else 0.0
        symbol = market_data.get("symbol", "UNKNOWN")
        current_price = market_data.get("price", 0.0)

        tp_price = current_price + tp_offset if current_price > 0 else 0.0
        sl_price = current_price - abs(sl_offset) if current_price > 0 else 0.0

        if self.redis:
            self.redis.hset(f"forecast:live:{symbol}", mapping={
                "delta_t10": pred_delta,
                "projected_price": current_price + pred_delta,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "timestamp": time.time()
            })

        self.pending_evaluations.append({
            "symbol":          symbol,
            "target_time":     time.time() + self.EVALUATION_DELAY_SECONDS,
            "predicted_delta": pred_delta,
            "baseline_price":  current_price,
            "sequence_cpu":    sequence.cpu(),
        })

        return {
            "delta_t10": pred_delta,
            "projected_price": current_price + pred_delta,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "ready": True
        }

    def _hindsight_evaluation_loop(self):
        while True:
            try:
                now = time.time()
                while self.pending_evaluations and self.pending_evaluations[0]["target_time"] <= now:
                    eval_data = self.pending_evaluations.popleft()
                    symbol = eval_data["symbol"]
                    
                    ticker_raw = self.redis.get(f"market:ticker:{symbol}") if self.redis else None
                    if ticker_raw:
                        actual_price = float(json.loads(ticker_raw).get("last_price", eval_data["baseline_price"]))
                        actual_delta = actual_price - eval_data["baseline_price"]
                        error_val    = eval_data["predicted_delta"] - actual_delta

                        error_tensor = torch.tensor([error_val], dtype=torch.float32).to(self.device)
                        self.adapt_online(error_tensor)

                        if eval_data.get("sequence_cpu") is not None:
                            self.experience_buffer.append({
                                "sequence":     eval_data["sequence_cpu"],
                                "actual_delta": float(actual_delta),
                            })
                            # Push to DB
                            s_arr = np.asarray(eval_data["sequence_cpu"].numpy()[-1][-1], dtype=np.float32)
                            threading.Thread(
                                target=self._persist_to_db,
                                args=(s_arr, float(eval_data["predicted_delta"]), float(actual_delta), s_arr, True, 0),
                                daemon=True
                            ).start()

                        try:
                            self.parquet_log.log({
                                "symbol":          symbol,
                                "ts":              now,
                                "baseline_price":  eval_data["baseline_price"],
                                "predicted_delta": eval_data["predicted_delta"],
                                "actual_delta":    actual_delta,
                                "error":           error_val,
                                "buffer_size":     len(self.experience_buffer),
                            })
                        except Exception as db_err:
                            self.think(f"Parquet log failed: {db_err}", category="error")

                time.sleep(5)
            except Exception as e:
                self.think(f"Hindsight loop error: {e}", category="error")
                time.sleep(5)

    def evaluate(self, n_recent: int = 500) -> Dict[str, Any]:
        import polars as pl

        df = self.parquet_log.read_polars(n_recent)
        if df.is_empty():
            return {"status": "no_data"}

        n      = len(df)
        mae    = float(df["error"].abs().mean()) if "error" in df.columns else float("nan")
        std    = float(df["error"].std() or 0.0) if "error" in df.columns else float("nan")
        if "predicted_delta" in df.columns and "actual_delta" in df.columns:
            same_sign = ((df["predicted_delta"] * df["actual_delta"]) > 0).sum()
            dir_acc   = float(same_sign) / max(n, 1)
        else:
            dir_acc   = float("nan")

        summary = (
            f"[Forecaster] EVAL | n={n} MAE={mae:.4f} σ={std:.4f} dir_acc={dir_acc:.1%}"
        )
        self.think(summary, category="reflection", tone="neutral", confidence=0.85)
        return {"n": n, "mae": mae, "std": std, "directional_accuracy": dir_acc}