import os
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

    def __init__(self, config: Dict = None, model_path: str = None, redis_client: Any = None):
        super().__init__(config, model_path, redis_client)
        self.history: Deque[torch.Tensor] = deque(maxlen=self.WINDOW_SIZE)
        self.is_active = False
        self.core_side = "LONG"
        self.trailing_stop: Optional[HybridVolumeTrailingStop] = None

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
        direction = TrailingDirection.LONG if side == "LONG" else TrailingDirection.SHORT
        config = HybridStopConfig(direction=direction, entry_price=entry_price)
        self.trailing_stop = HybridVolumeTrailingStop(symbol, config)
        self.think(f"CORE POSITION LOCKED ({side}). Monitoring HTF S/R target.", category="action")

    def deactivate(self):
        self.is_active = False
        self.trailing_stop = None
        self.history.clear()
        self.think("Position closed. Entering hibernation.", category="action")


