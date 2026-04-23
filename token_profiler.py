8import os

import logging
from typing import Dict, List, Any

logger = logging.getLogger("token_profiler")

class TokenProfiler:
    """
    Categorizes tokens by blockchain and tier to generalize learning
    from 20 tokens to 400+.
    """
    
    # Blockchain Encodings (Ordinal for simple feature injection)
    CHAINS = {
        "BITCOIN": 1,
        "ETHEREUM": 2,
        "SOLANA": 3,
        "XRP": 4,
        "CARDANO": 5,
        "AVALANCHE": 6,
        "POLKADOT": 7,
        "LAYER2": 8,
        "SUI": 9,
        "OTHER": 10
    }
    
    # Tier Encodings
    TIERS = {
        "ULTRA": 1,   # BTC, ETH
        "HIGH": 2,    # Top Alts
        "MEDIUM": 3,  # Mid Alts
        "MEME": 4     # High Volatility
    }

    SYMBOL_MAP = {
        'BTCUSDT': {"chain": "BITCOIN", "tier": "ULTRA"},
        'ETHUSDT': {"chain": "ETHEREUM", "tier": "ULTRA"},
        'SOLUSDT': {"chain": "SOLANA", "tier": "HIGH"},
        'XRPUSDT': {"chain": "XRP", "tier": "HIGH"},
        'ADAUSDT': {"chain": "CARDANO", "tier": "HIGH"},
        'AVAXUSDT': {"chain": "AVALANCHE", "tier": "MEDIUM"},
        'DOTUSDT': {"chain": "POLKADOT", "tier": "MEDIUM"},
        'NEARUSDT': {"chain": "NEAR", "tier": "MEDIUM"}, # Generic mapping
        'LINKUSDT': {"chain": "ETHEREUM", "tier": "MEDIUM"},
        'LTCUSDT': {"chain": "BITCOIN", "tier": "MEDIUM"},
        'BCHUSDT': {"chain": "BITCOIN", "tier": "MEDIUM"},
        'POLUSDT': {"chain": "LAYER2", "tier": "HIGH"},
        'ARBUSDT': {"chain": "LAYER2", "tier": "MEDIUM"},
        'OPUSDT': {"chain": "LAYER2", "tier": "MEDIUM"},
        'SUIUSDT': {"chain": "SUI", "tier": "MEDIUM"},
        'DOGEUSDT': {"chain": "BITCOIN", "tier": "MEME"},
        '1000PEPEUSDT': {"chain": "ETHEREUM", "tier": "MEME"},
        'WIFUSDT': {"chain": "SOLANA", "tier": "MEME"},
        '1000BONKUSDT': {"chain": "SOLANA", "tier": "MEME"},
        'SHIB1000USDT': {"chain": "ETHEREUM", "tier": "MEME"},
    }

    @classmethod
    def get_token_features(cls, symbol: str) -> List[float]:
        """Returns 2-dim feature [chain_id, tier_id] for the given symbol."""
        profile = cls.SYMBOL_MAP.get(symbol, {"chain": "OTHER", "tier": "MEDIUM"})
        
        # Fallback logic for 400+ tokens
        if symbol.endswith("USDT") and symbol not in cls.SYMBOL_MAP:
            # Simple heuristic: Meme detection
            if any(x in symbol for x in ["PEPE", "DOGE", "SHIB", "FLOKI", "BONK", "WIF"]):
                profile = {"chain": "OTHER", "tier": "MEME"}
            # L2 detection
            elif any(x in symbol for x in ["ARB", "OP", "POL", "MATIC", "ZK"]):
                profile = {"chain": "LAYER2", "tier": "MEDIUM"}
        
        return [
            float(cls.CHAINS.get(profile["chain"], 10)),
            float(cls.TIERS.get(profile["tier"], 3))
        ]

    @classmethod
    def inject_profile(cls, symbol: str, tensor: Any) -> Any:
        """Injects token identity into reserved slots [24, 25]."""
        features = cls.get_token_features(symbol)
        tensor[24] = features[0]
        tensor[25] = features[1]
        return tensor

