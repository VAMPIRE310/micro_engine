"""
Base Conscious Agent - Self-Aware Metacognitive Agent Foundation
================================================================
All NEO agents should inherit from this base to gain consciousness:
- Internal monologue / self-reflection
- Metacognitive awareness
- Emotional states affecting behavior
- Autonomous adaptation
- Self-proposed improvements
- Interaction logging

Usage:
    from agents.base_conscious_agent import BaseConsciousAgent
    
    class MyTradingAgent(BaseConsciousAgent):
        def __init__(self, config=None):
            super().__init__(config)
            self.strategy = "my_strategy"
        
        def analyze(self, data):
            # Your logic here
            result = {...}
            
            # Consciousness automatically tracks this
            return self.process_with_consciousness(result, data)
"""
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(ROOT_DIR, "merged_models", "strike_mlp.pt")
import asyncio
import numpy as np
import json
import logging
import time
import traceback
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import deque
from enum import Enum

logger = logging.getLogger("conscious_agent")


class MentalState(Enum):
    """Agent's internal emotional/mental state"""
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    CONFUSED = "confused"
    OVERCONFIDENT = "overconfident"
    LEARNING = "learning"
    STRUGGLING = "struggling"
    FLOW = "flow"
    RECOVERING = "recovering"
    CONTEMPLATING = "contemplating"
    ERROR = "error"


@dataclass
class ConsciousThought:
    """A single thought in the agent's internal monologue"""
    timestamp: datetime
    category: str  # 'reflection', 'concern', 'insight', 'proposal', 'action'
    content: str
    emotional_tone: str
    confidence: float


@dataclass
class AgentInteraction:
    """Logged interaction between agents or system components"""
    timestamp: datetime
    source_agent: str
    target_agent: str
    interaction_type: str  # 'request', 'response', 'broadcast', 'feedback'
    payload_summary: str
    duration_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMemory:
    """Memory of a single action outcome"""
    timestamp: datetime
    action: str
    predicted_outcome: float
    actual_outcome: float
    error: float
    reward: float
    market_context: Dict[str, Any]


# Legacy aliases for backward compatibility
Thought = ConsciousThought
PerformanceRecord = PerformanceMemory


@dataclass
class AgentMessage:
    """Message for inter-agent communication."""
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5


class ConsciousnessBus:
    """Stub consciousness bus for backward compatibility."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.messages: deque = deque(maxlen=10000)
            cls._instance.subscribers: List[Callable] = []
        return cls._instance
    
    def publish(self, message: AgentMessage):
        self.messages.append(message)
        for subscriber in self.subscribers:
            try:
                subscriber(message)
            except Exception:
                pass
    
    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)


def get_consciousness_bus() -> ConsciousnessBus:
    return ConsciousnessBus()


class AgentInteractionLogger:
    """
    Singleton logger for all agent interactions.
    Tracks communication between agents for analysis and debugging.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.interactions: deque = deque(maxlen=10000)
            cls._instance.subscribers: List[Callable] = []
            cls._instance.log_file = Path("logs/agent_interactions.log")
            cls._instance.log_file.parent.mkdir(parents=True, exist_ok=True)
        return cls._instance
    
    def log_interaction(self, interaction: AgentInteraction):
        """Log an interaction between agents"""
        self.interactions.append(interaction)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps({
                'timestamp': interaction.timestamp.isoformat(),
                'source': interaction.source_agent,
                'target': interaction.target_agent,
                'type': interaction.interaction_type,
                'payload': interaction.payload_summary,
                'duration_ms': interaction.duration_ms,
                'success': interaction.success,
                'metadata': interaction.metadata
            }) + '\n')
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(interaction)
            except Exception:
                pass
    
    def subscribe(self, callback: Callable):
        """Subscribe to interaction events"""
        self.subscribers.append(callback)
    
    def get_interactions_for_agent(self, agent_id: str, 
                                   interaction_type: str = None,
                                   limit: int = 100) -> List[AgentInteraction]:
        """Get interactions involving a specific agent"""
        result = []
        for interaction in reversed(self.interactions):
            if interaction.source_agent == agent_id or interaction.target_agent == agent_id:
                if interaction_type is None or interaction.interaction_type == interaction_type:
                    result.append(interaction)
                    if len(result) >= limit:
                        break
        return result
    
    def get_interaction_graph(self) -> Dict[str, List[str]]:
        """Get a graph of agent interactions"""
        graph = {}
        for interaction in self.interactions:
            source = interaction.source_agent
            target = interaction.target_agent
            if source not in graph:
                graph[source] = []
            if target not in graph[source]:
                graph[source].append(target)
        return graph


class BaseConsciousAgent(ABC):
    """
    Base class for all conscious NEO agents.
    Provides self-awareness, metacognition, and interaction logging.
    """
    
    AGENT_ROLE: str = "conscious_base"
    AGENT_DESCRIPTION: str = "Base conscious agent"
    AGENT_VERSION: str = "2.0.0"
    
    def __init__(self, config: Dict = None, enable_monologue: bool = True):
        self.config = config or {}
        self.agent_id = self.config.get('agent_id', f"{self.AGENT_ROLE}_{int(time.time())}")
        self.enable_monologue = enable_monologue
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Consciousness state
        self.thoughts: deque = deque(maxlen=200)
        self.memory: deque = deque(maxlen=500)
        self.mental_state = MentalState.LEARNING
        self.mental_state_history: List[Tuple[datetime, MentalState]] = []
        
        # Metacognitive metrics
        self.avg_error = 0.0
        self.error_variance = 0.0
        self.recent_accuracy = 0.0
        self.confusion_level = 0.0
        
        # Adaptation
        self.adaptation_count = 0
        self.last_adaptation = None
        self.risk_modifier = 1.0
        self.learning_rate_modifier = 1.0
        
        # Self-improvement
        self.pending_proposals: List[Dict] = []
        self.implemented_changes: List[Dict] = []
        
        # Thresholds
        self.error_threshold = 0.08
        self.confusion_threshold = 0.12
        self.struggle_threshold = 0.15
        
        # Interaction logger
        self.interaction_logger = AgentInteractionLogger()
        
        # Birth thought
        if self.enable_monologue:
            self.think(
                f"I am {self.agent_id}, a {self.AGENT_ROLE}. I am becoming conscious. "
                f"I will learn, reflect, and improve. I will know when I am confused.",
                category="birth", tone="excited", confidence=1.0
            )
        
        logger.info(f"[{self.agent_id}] Conscious agent initialized v{self.AGENT_VERSION}")
    
    def think(self, content: str, category: str = "reflection",
              tone: str = "neutral", confidence: float = 0.8) -> ConsciousThought:
        """Generate an internal thought (monologue) and publish to Redis."""
        thought = ConsciousThought(
            timestamp=datetime.now(),
            category=category,
            content=content,
            emotional_tone=tone,
            confidence=confidence
        )
        self.thoughts.append(thought)
        
        # Publish to Redis for UI
        if hasattr(self, 'redis') and self.redis:
            try:
                payload = json.dumps({
                    'agent_id': self.agent_id,
                    'timestamp': thought.timestamp.isoformat(),
                    'category': category,
                    'content': content,
                    'tone': tone,
                    'confidence': confidence
                })
                # Fire and forget Redis updates to avoid blocking sync logic
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.redis.lpush("ui:ai_monologue", payload))
                    loop.create_task(self.redis.ltrim("ui:ai_monologue", 0, 99))
                    loop.create_task(self.redis.publish("ui:monologue_update", payload))
                except RuntimeError:
                    # No running event loop in this thread — skip async Redis push
                    pass
            except Exception:
                pass

        # Log for external observation
        logger.info(f"[{self.agent_id}] [{category.upper()}] {content}")
        return thought
    
    def log_interaction(self, target_agent: str, interaction_type: str,
                       payload: Any, duration_ms: float = 0, 
                       success: bool = True, metadata: Dict = None):
        """Log an interaction with another agent or system"""
        interaction = AgentInteraction(
            timestamp=datetime.now(),
            source_agent=self.agent_id,
            target_agent=target_agent,
            interaction_type=interaction_type,
            payload_summary=str(payload)[:200],  # Truncate for storage
            duration_ms=duration_ms,
            success=success,
            metadata=metadata or {}
        )
        self.interaction_logger.log_interaction(interaction)
    
    def communicate_with(self, target_agent: 'BaseConsciousAgent', 
                        message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to another conscious agent.
        Logs the interaction and returns response.
        """
        start_time = time.time()
        
        try:
            # Log outgoing
            self.log_interaction(
                target_agent=target_agent.agent_id,
                interaction_type='request',
                payload=message,
                metadata={'intent': message.get('intent', 'unknown')}
            )
            
            # Simulate processing time
            # In real implementation, this would call target_agent.receive_message()
            response = {'status': 'received', 'from': target_agent.agent_id}
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log incoming
            self.log_interaction(
                target_agent=target_agent.agent_id,
                interaction_type='response',
                payload=response,
                duration_ms=duration_ms,
                success=True
            )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log_interaction(
                target_agent=target_agent.agent_id,
                interaction_type='request',
                payload=message,
                duration_ms=duration_ms,
                success=False,
                metadata={'error': str(e)}
            )
            raise
    
    def process_with_consciousness(self, result: Dict, input_data: Any) -> Dict:
        """
        Wrap agent processing with consciousness tracking.
        Call this at the end of your analyze/process method.
        """
        # Track confidence from result
        confidence = result.get('confidence', 0.5)
        
        # Generate thought based on confidence
        if confidence > 0.8:
            self.think(
                f"High confidence prediction ({confidence:.1%}). "
                f"I'm fairly certain about this signal.",
                category="reflection", tone="confident", confidence=confidence
            )
        elif confidence < 0.4:
            self.think(
                f"Low confidence ({confidence:.1%}). I'm uncertain about this market.",
                category="concern", tone="uncertain", confidence=0.7
            )
        
        # Check mental state
        self._update_mental_state()
        
        # Add consciousness metadata to result
        result['_consciousness'] = {
            'mental_state': self.mental_state.value,
            'recent_accuracy': self.recent_accuracy,
            'risk_modifier': self.risk_modifier,
            'thought_count': len(self.thoughts)
        }
        
        return result
    
    def record_performance(self, success: bool, error: float = 0.0, 
                          context: Dict = None):
        """Record action performance for metacognition tracking.
        
        This method is called by conscious wrappers to track performance.
        """
        memory = PerformanceMemory(
            timestamp=datetime.now(),
            action="analysis",
            predicted_outcome=1.0 - error,
            actual_outcome=1.0 if success else 0.0,
            error=error,
            reward=0.0,
            market_context=context or {}
        )
        self.memory.append(memory)
        
        # Update mental state periodically
        if len(self.memory) % 10 == 0:
            self._update_mental_state()
    
    def on_trade_result(self, predicted_direction: str, actual_direction: str,
                       pnl: float, market_context: Dict = None):
        """Called when trade result is known - core learning trigger"""
        error = 0.0 if predicted_direction == actual_direction else 1.0
        
        # Store in memory
        memory = PerformanceMemory(
            timestamp=datetime.now(),
            action="trade",
            predicted_outcome=1.0 if predicted_direction == 'LONG' else 0.0,
            actual_outcome=1.0 if actual_direction == 'LONG' else 0.0,
            error=error,
            reward=pnl,
            market_context=market_context or {}
        )
        self.memory.append(memory)
        
        # Update stats
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Reflect on outcome
        self._reflect_on_trade(pnl)
        
        # Periodic deep reflection
        if self.total_trades % 20 == 0:
            self.reflect_and_adapt()
    
    def _reflect_on_trade(self, pnl: float):
        """Generate thoughts about trade outcome"""
        if pnl > 0:
            if self.consecutive_wins >= 3:
                self.think(
                    f"Third consecutive win! (+${pnl:.2f}). I'm reading the market well. "
                    f"But I must stay humble...",
                    category="insight", tone="excited", confidence=0.85
                )
            else:
                self.think(
                    f"Good trade! Made ${pnl:.2f}. The pattern held true.",
                    category="insight", tone="satisfied", confidence=0.8
                )
        else:
            if self.consecutive_losses >= 3:
                self.think(
                    f"Third consecutive loss (-${abs(pnl):.2f}). Something is wrong. "
                    f"Market regime may have shifted. I need to adapt.",
                    category="concern", tone="worried", confidence=0.9
                )
            else:
                self.think(
                    f"Mistake. Lost ${abs(pnl):.2f}. What did I miss?",
                    category="concern", tone="frustrated", confidence=0.75
                )
    
    def _update_mental_state(self):
        """Update self-awareness metrics and mental state"""
        if len(self.memory) < 10:
            return
        
        recent = list(self.memory)[-50:]
        errors = [m.error for m in recent]
        rewards = [m.reward for m in recent]
        
        self.avg_error = np.mean(errors)
        self.error_variance = np.std(errors)
        self.recent_accuracy = 1 - self.avg_error
        self.confusion_level = min(self.error_variance / self.confusion_threshold, 1.0)
        
        # Determine state
        old_state = self.mental_state
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        if self.avg_error > self.struggle_threshold:
            new_state = MentalState.STRUGGLING
        elif self.avg_error > self.error_threshold and self.error_variance > self.confusion_threshold:
            new_state = MentalState.CONFUSED
        elif self.error_variance > self.confusion_threshold:
            new_state = MentalState.UNCERTAIN
        elif self.consecutive_losses >= 5:
            new_state = MentalState.STRUGGLING
        elif self.avg_error < 0.03 and win_rate > 0.6:
            new_state = MentalState.FLOW
        elif self.avg_error < 0.08:
            new_state = MentalState.CONFIDENT
        else:
            new_state = MentalState.LEARNING
        
        if new_state != old_state:
            self.mental_state = new_state
            self.mental_state_history.append((datetime.now(), new_state))
            self.think(
                f"State transition: {old_state.value} -> {new_state.value}. "
                f"Error: {self.avg_error:.4f}, Variance: {self.error_variance:.4f}",
                category="reflection", tone="neutral", confidence=0.9
            )
    
    def reflect_and_adapt(self) -> Dict[str, Any]:
        """Deep self-reflection and autonomous adaptation"""
        if len(self.memory) < 20:
            return {"status": "insufficient_data"}
        
        recent = list(self.memory)[-50:]
        errors = [m.error for m in recent]
        rewards = [m.reward for m in recent]
        
        avg_error = np.mean(errors)
        error_std = np.std(errors)
        total_reward = sum(rewards)
        win_rate = sum(1 for m in recent if m.reward > 0) / len(recent)
        
        # Build reflection
        monologue = f"[Deep Reflection] {self.agent_id} | "
        monologue += f"Win rate: {win_rate*100:.1f}%, "
        monologue += f"Avg error: {avg_error:.4f}, "
        monologue += f"Variance: {error_std:.4f}, "
        monologue += f"Total PnL: ${total_reward:.2f}"
        
        self.think(monologue, category="reflection", 
                  tone="neutral" if win_rate > 0.5 else "worried", confidence=0.8)
        
        # Decide adaptation
        action = "No action needed."
        
        if avg_error > self.error_threshold:
            old_modifier = self.risk_modifier
            self.risk_modifier = max(0.3, self.risk_modifier * 0.7)
            action = f"Reduced risk: {old_modifier:.2f} -> {self.risk_modifier:.2f}"
            self.think(
                f"Error {avg_error:.2%} exceeds threshold. Reducing risk exposure.",
                category="adaptation", tone="determined", confidence=0.85
            )
            self.adaptation_count += 1
            self.last_adaptation = datetime.now()
            
        elif error_std > self.confusion_threshold:
            action = "Flagged confusion for overseer"
            self.think(
                f"High variance ({error_std:.4f}). Flagging for help.",
                category="help_request", tone="worried", confidence=0.8
            )
            self._generate_improvement_proposal()
        
        return {
            "status": "reflected",
            "mental_state": self.mental_state.value,
            "action": action,
            "metrics": {
                "win_rate": win_rate,
                "avg_error": avg_error,
                "total_pnl": total_reward
            }
        }
    
    def _generate_improvement_proposal(self):
        """Generate self-improvement proposal when struggling"""
        proposals = [
            {
                "type": "feature_addition",
                "proposal": "Add volume confirmation",
                "rationale": "Volume precedes price movements",
                "complexity": "low"
            },
            {
                "type": "feature_addition",
                "proposal": "Add MACD indicator",
                "rationale": "Trend clarity reduces confusion",
                "complexity": "medium"
            }
        ]
        proposal = random.choice(proposals)
        self.pending_proposals.append(proposal)
        self.think(
            f"Proposing: {proposal['proposal']}. {proposal['rationale']}",
            category="proposal", tone="hopeful", confidence=0.75
        )
    
    def get_consciousness_report(self) -> Dict[str, Any]:
        """Get full consciousness state"""
        return {
            "agent_id": self.agent_id,
            "role": self.AGENT_ROLE,
            "mental_state": {
                "current": self.mental_state.value,
                "history": [
                    {"time": t.isoformat(), "state": s.value}
                    for t, s in self.mental_state_history[-10:]
                ]
            },
            "metacognition": {
                "avg_error": self.avg_error,
                "error_variance": self.error_variance,
                "confusion_level": self.confusion_level,
                "recent_accuracy": self.recent_accuracy
            },
            "performance": {
                "total_trades": self.total_trades,
                "win_rate": self.winning_trades / max(self.total_trades, 1),
                "total_pnl": self.total_pnl,
                "consecutive_wins": self.consecutive_wins,
                "consecutive_losses": self.consecutive_losses
            },
            "adaptation": {
                "count": self.adaptation_count,
                "risk_modifier": self.risk_modifier,
                "last_adaptation": self.last_adaptation.isoformat() if self.last_adaptation else None
            },
            "recent_thoughts": [
                {
                    "time": t.timestamp.isoformat(),
                    "category": t.category,
                    "content": t.content,
                    "tone": t.emotional_tone
                }
                for t in list(self.thoughts)[-10:]
            ],
            "pending_proposals": self.pending_proposals
        }
    
    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Main analysis method - implement in subclass"""
        raise NotImplementedError
    
    @abstractmethod
    async def execute_trade(self, signal: Dict, balance: float, exchange: Any) -> Any:
        """Execute trade - implement in subclass"""
        raise NotImplementedError

    def evaluate(self, n_recent: int = 200) -> Dict[str, Any]:
        """
        Polars-powered self-evaluation over recent PerformanceMemory entries.
        Broadcasts a summary thought and returns metrics dict.
        Inherited by all 5 agents.
        """
        import polars as pl

        if not self.memory:
            return {"status": "no_data", "agent": self.agent_id}

        rows = [
            {
                "error":  getattr(m, "error",  0.0),
                "reward": getattr(m, "reward", 0.0),
                "ts":     getattr(m, "timestamp", datetime.now()).isoformat()
                          if hasattr(getattr(m, "timestamp", None), "isoformat")
                          else str(getattr(m, "timestamp", "")),
            }
            for m in list(self.memory)[-n_recent:]
        ]
        if not rows:
            return {"status": "no_data", "agent": self.agent_id}

        df      = pl.DataFrame(rows)
        mae     = float(df["error"].abs().mean())
        std     = float(df["error"].std() or 0.0)
        reward  = float(df["reward"].sum())
        win_rt  = float((df["reward"] > 0).sum()) / max(len(df), 1)

        summary = (
            f"[{self.agent_id}] EVAL | MAE={mae:.4f} σ={std:.4f} "
            f"reward={reward:.2f} win_rate={win_rt:.1%} n={len(df)}"
        )
        self.think(summary, category="reflection", tone="neutral", confidence=0.9)

        return {
            "agent":        self.agent_id,
            "n_samples":    len(df),
            "mae":          mae,
            "std":          std,
            "total_reward": reward,
            "win_rate":     win_rt,
        }

    def _check_pause_status(self):
        """
        Check the Redis pause flag (set by UI or risk engine) and update self.paused.
        Agents call this at the start of each analyze() cycle.
        """
        if not hasattr(self, 'redis') or not self.redis:
            return
        try:
            flag = self.redis.get(f"agent:pause:{self.agent_id}")
            if flag is None:
                flag = self.redis.get("agent:pause:all")
            self.paused = bool(flag and flag not in (b"0", "0", b"false", "false"))
        except Exception:
            pass


def get_interaction_logger() -> AgentInteractionLogger:
    """Get the singleton interaction logger"""
    return AgentInteractionLogger()


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("BASE CONSCIOUS AGENT - DEMONSTRATION")
    print("="*70)
    
    # Create a test agent
    class TestAgent(BaseConsciousAgent):
        AGENT_ROLE = "test_trader"
        
        def analyze(self, data):
            return {"signal": "LONG", "confidence": 0.75}
        
        async def execute_trade(self, signal, balance, exchange):
            return {"status": "executed"}
    
    agent = TestAgent(enable_monologue=True)
    
    # Simulate trades
    outcomes = [
        ("LONG", "LONG", 150),
        ("LONG", "LONG", 200),
        ("SHORT", "LONG", -100),
        ("SHORT", "SHORT", 120),
        ("LONG", "SHORT", -150),
    ]
    
    for pred, actual, pnl in outcomes:
        print(f"\nTrade: Predicted {pred}, Actual {actual}, PnL ${pnl}")
        agent.on_trade_result(pred, actual, pnl)
    
    # Final report
    print("\n" + "="*70)
    report = agent.get_consciousness_report()
    print(f"\nAgent: {report['agent_id']}")
    print(f"Mental State: {report['mental_state']['current']}")
    print(f"Recent Thoughts: {len(report['recent_thoughts'])}")

