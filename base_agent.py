import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(ROOT_DIR, "merged_models", "strike_mlp.pt")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.hardware.gpu_setup import arm_tensor_cores
# Arm Tensor Cores once at import time — idempotent, safe across multiple agent imports
_CUDA_DEVICE = arm_tensor_cores()
import logging
import json
from abc import abstractmethod
from typing import Dict, Any, Optional, Union
from datetime import datetime
from core.agents.base_conscious_agent import BaseConsciousAgent, ConsciousThought

logger = logging.getLogger("conscious_agent")

class BaseAgent(BaseConsciousAgent):
    """
    Enhanced Base Agent for Neo Supreme Hydra architecture.
    Tailored for the 160-dim feature vector and surgical HFT execution.
    """
    
    FEATURE_DIM = 160
    
    def __init__(self, config: Dict = None, model_path: Optional[str] = None, redis_client: Any = None):
        super().__init__(config)
        # Prefer CUDA when available, fall back to CPU gracefully
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # BF16 on CUDA: same dynamic range as FP32 (no overflow during online backprop),
        # native Tensor Core support on RTX 30+ (Ampere) and RTX 40+ (Ada Lovelace).
        # Falls back to FP16 on older CUDA hardware that lacks BF16 Tensor Cores.
        # CPU uses FP32 (BF16/FP16 are unsupported for most CPU ops).
        if self.device.type == "cuda":
            # BF16 needs compute capability >= 8.0 (Ampere/Ada: RTX 30xx/40xx).
            # torch.cuda.is_bf16_supported() is available in PyTorch >= 1.10 but
            # falls back gracefully: use capability check for maximum compatibility.
            try:
                major, _ = torch.cuda.get_device_capability()
                self._compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
            except Exception:
                self._compute_dtype = torch.float16
        else:
            self._compute_dtype = torch.float32
        self.model = self._build_network().to(self.device).to(self._compute_dtype)
        self.redis = redis_client
        self.paused = False
        
        if model_path:
            self.load_model(model_path)

        # torch.compile — bypasses Python interpreter via CUDA graphs on first forward pass.
        # mode='reduce-overhead' is optimal for fixed-shape HFT inference (T1/T2 tick loops).
        if self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as _ce:
                # torch.compile requires PyTorch >= 2.0; skip gracefully on older builds
                logger.warning(f"[{self.__class__.__name__}] torch.compile unavailable: {_ce}")
            
        precision = str(self._compute_dtype).split(".")[-1]
        self.think(f"{self.device.type.upper()} ACCELERATED: {self.AGENT_ROLE} model loaded on {self.device} ({precision})", category="birth")

    def analyze(self, state_vector: Union[np.ndarray, torch.Tensor],
                market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Sub-millisecond GPU Inference.
        """
        self._check_pause_status()
        if self.paused:
            return {"status": "paused", "confidence": 0.0}

        tensor_state = self._preprocess(state_vector)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self.predict(tensor_state, market_data)
            else:
                output = self.predict(tensor_state, market_data)
            
        return self.process_with_consciousness(output, state_vector)

    def _preprocess(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        # non_blocking=True: PCIe transfer doesn't stall the Python thread
        return state.to(device=self.device, dtype=self._compute_dtype, non_blocking=True)

    def load_model(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.to(self._compute_dtype)
            logger.info(f"[{self.agent_id}] Model loaded from {path}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to load model: {str(e)}")

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        logger.info(f"[{self.agent_id}] Model saved to {path}")

    async def execute_trade(self, signal: Dict, balance: float, exchange: Any) -> Any:
        """
        Default implementation for BaseConsciousAgent. 
        Most agents in Hydra will delegate execution to the Negotiator/Sniper.
        """
        return {"status": "delegated"}

