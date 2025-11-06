"""
Meta-Learning System

This system learns how to learn - it tracks learning strategies, their effectiveness,
and adaptively improves the learning process itself.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import json
import numpy as np


@dataclass
class LearningStrategy:
    """Represents a learning strategy"""
    strategy_id: str
    name: str
    description: str
    strategy_type: str  # memorization, association, inference, pattern_recognition, etc.
    effectiveness_scores: List[float] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    contexts_used: List[str] = field(default_factory=list)
    best_contexts: List[str] = field(default_factory=list)
    avg_effectiveness: float = 0.5

    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'effectiveness_scores': self.effectiveness_scores[-20:],  # Keep recent
            'usage_count': self.usage_count,
            'success_count': self.success_count,
            'contexts_used': self.contexts_used[-20:],
            'best_contexts': self.best_contexts,
            'avg_effectiveness': self.avg_effectiveness
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningStrategy':
        return cls(**data)


@dataclass
class LearningExperience:
    """Represents a learning experience and its outcome"""
    experience_id: str
    context: str
    strategy_used: str
    information_learned: str
    success: bool
    retention_score: float  # How well was it retained
    application_score: float  # How well could it be applied
    speed_score: float  # How quickly was it learned
    overall_effectiveness: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'experience_id': self.experience_id,
            'context': self.context,
            'strategy_used': self.strategy_used,
            'information_learned': self.information_learned,
            'success': self.success,
            'retention_score': self.retention_score,
            'application_score': self.application_score,
            'speed_score': self.speed_score,
            'overall_effectiveness': self.overall_effectiveness,
            'timestamp': self.timestamp
        }


@dataclass
class LearningGoal:
    """Represents a learning goal"""
    goal_id: str
    description: str
    target_knowledge: str
    priority: float
    progress: float = 0.0
    strategies_tried: List[str] = field(default_factory=list)
    achieved: bool = False
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'goal_id': self.goal_id,
            'description': self.description,
            'target_knowledge': self.target_knowledge,
            'priority': self.priority,
            'progress': self.progress,
            'strategies_tried': self.strategies_tried,
            'achieved': self.achieved,
            'created_at': self.created_at
        }


class MetaLearningSystem:
    """
    System that learns about learning itself.
    Tracks what learning strategies work, adapts approach, and optimizes learning.
    """

    def __init__(self):
        self.learning_strategies: Dict[str, LearningStrategy] = {}
        self.learning_experiences: List[LearningExperience] = []
        self.learning_goals: Dict[str, LearningGoal] = {}

        # Track what works in what contexts
        self.context_strategy_effectiveness: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Learning insights
        self.insights: List[Dict] = []

        # Initialize with basic learning strategies
        self._initialize_strategies()

        # Current learning state
        self.current_learning_mode = "balanced"  # balanced, fast, deep, exploratory
        self.learning_rate_adjustment = 1.0
        self.strategy_id_counter = 0
        self.experience_id_counter = 0

    def _initialize_strategies(self):
        """Initialize with basic learning strategies"""
        basic_strategies = [
            LearningStrategy(
                strategy_id="strategy_0",
                name="Direct Memorization",
                description="Store information directly with minimal processing",
                strategy_type="memorization"
            ),
            LearningStrategy(
                strategy_id="strategy_1",
                name="Association Building",
                description="Connect new information to existing knowledge",
                strategy_type="association"
            ),
            LearningStrategy(
                strategy_id="strategy_2",
                name="Pattern Recognition",
                description="Identify patterns in information for easier recall",
                strategy_type="pattern"
            ),
            LearningStrategy(
                strategy_id="strategy_3",
                name="Causal Understanding",
                description="Learn by understanding cause-effect relationships",
                strategy_type="causal"
            ),
            LearningStrategy(
                strategy_id="strategy_4",
                name="Analogical Learning",
                description="Learn by comparing to similar situations",
                strategy_type="analogical"
            ),
            LearningStrategy(
                strategy_id="strategy_5",
                name="Inference-Based Learning",
                description="Learn by deriving implications and inferences",
                strategy_type="inference"
            ),
            LearningStrategy(
                strategy_id="strategy_6",
                name="Repetition with Variation",
                description="Encounter information multiple times in different forms",
                strategy_type="reinforcement"
            ),
            LearningStrategy(
                strategy_id="strategy_7",
                name="Contextual Embedding",
                description="Learn information deeply embedded in context",
                strategy_type="contextual"
            ),
            LearningStrategy(
                strategy_id="strategy_8",
                name="Multi-Modal Learning",
                description="Combine different types of information (semantic, emotional, temporal)",
                strategy_type="multimodal"
            ),
            LearningStrategy(
                strategy_id="strategy_9",
                name="Elaborative Rehearsal",
                description="Actively think about and elaborate on information",
                strategy_type="elaborative"
            )
        ]

        self.strategy_id_counter = len(basic_strategies)

        for strategy in basic_strategies:
            self.learning_strategies[strategy.strategy_id] = strategy

    def select_learning_strategy(
        self,
        information: str,
        context: str,
        learning_goal: Optional[str] = None
    ) -> LearningStrategy:
        """
        Meta-learning: Select the best learning strategy for this information and context.
        """

        # Score each strategy for this context
        strategy_scores = {}

        for strategy_id, strategy in self.learning_strategies.items():
            score = self._score_strategy_for_context(strategy, context, learning_goal)
            strategy_scores[strategy_id] = score

        # Add exploration bonus (try less-used strategies occasionally)
        for strategy_id, strategy in self.learning_strategies.items():
            if strategy.usage_count < 5:
                exploration_bonus = 0.2 * (1.0 - strategy.usage_count / 5.0)
                strategy_scores[strategy_id] += exploration_bonus

        # Select best strategy
        best_strategy_id = max(strategy_scores, key=strategy_scores.get)
        selected_strategy = self.learning_strategies[best_strategy_id]

        # Update usage
        selected_strategy.usage_count += 1
        selected_strategy.contexts_used.append(context)

        return selected_strategy

    def _score_strategy_for_context(
        self,
        strategy: LearningStrategy,
        context: str,
        learning_goal: Optional[str]
    ) -> float:
        """Score how effective this strategy is likely to be in this context"""

        # Base score: average effectiveness
        score = strategy.avg_effectiveness

        # Context-specific effectiveness
        if context in self.context_strategy_effectiveness:
            if strategy.strategy_id in self.context_strategy_effectiveness[context]:
                context_scores = self.context_strategy_effectiveness[context][strategy.strategy_id]
                if context_scores:
                    context_avg = sum(context_scores) / len(context_scores)
                    # Weight context-specific more heavily
                    score = 0.3 * score + 0.7 * context_avg

        # Check if this strategy is good for similar contexts
        for known_context in strategy.best_contexts:
            if self._contexts_similar(context, known_context):
                score += 0.1

        # Adjust based on learning mode
        if self.current_learning_mode == "fast":
            # Prefer simpler, faster strategies
            if strategy.strategy_type in ['memorization', 'pattern']:
                score += 0.15
        elif self.current_learning_mode == "deep":
            # Prefer deeper understanding strategies
            if strategy.strategy_type in ['causal', 'inference', 'elaborative']:
                score += 0.15
        elif self.current_learning_mode == "exploratory":
            # Prefer diverse, multi-faceted strategies
            if strategy.strategy_type in ['analogical', 'multimodal', 'contextual']:
                score += 0.15

        return score

    def _contexts_similar(self, context1: str, context2: str) -> bool:
        """Check if two contexts are similar"""
        # Simple similarity check
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        similarity = overlap / max(len(words1), len(words2))

        return similarity > 0.3

    def record_learning_experience(
        self,
        context: str,
        strategy: LearningStrategy,
        information: str,
        success: bool,
        retention_score: float,
        application_score: float = 0.5,
        speed_score: float = 0.5
    ):
        """
        Record a learning experience and update meta-learning knowledge.
        """

        # Calculate overall effectiveness
        overall = (retention_score * 0.4 + application_score * 0.3 + speed_score * 0.3)

        # Create experience
        experience = LearningExperience(
            experience_id=f"exp_{self.experience_id_counter}",
            context=context,
            strategy_used=strategy.strategy_id,
            information_learned=information,
            success=success,
            retention_score=retention_score,
            application_score=application_score,
            speed_score=speed_score,
            overall_effectiveness=overall
        )

        self.experience_id_counter += 1
        self.learning_experiences.append(experience)

        # Update strategy effectiveness
        strategy.effectiveness_scores.append(overall)
        if success:
            strategy.success_count += 1

        # Update average effectiveness
        if strategy.effectiveness_scores:
            # Use recent scores more heavily
            recent_scores = strategy.effectiveness_scores[-10:]
            strategy.avg_effectiveness = sum(recent_scores) / len(recent_scores)

        # Update context-strategy mapping
        self.context_strategy_effectiveness[context][strategy.strategy_id].append(overall)

        # Update best contexts for this strategy
        if overall > 0.7 and context not in strategy.best_contexts:
            strategy.best_contexts.append(context)

        # Generate insights periodically
        if len(self.learning_experiences) % 20 == 0:
            self._generate_insights()

        # Adapt learning approach
        self._adapt_learning_approach()

    def _generate_insights(self):
        """Generate insights about learning patterns"""

        # Insight 1: Which strategies work best overall
        best_strategies = sorted(
            self.learning_strategies.values(),
            key=lambda s: s.avg_effectiveness,
            reverse=True
        )[:3]

        if best_strategies:
            insight = {
                'type': 'best_strategies',
                'timestamp': datetime.now().timestamp(),
                'content': f"Most effective learning strategies: {', '.join(s.name for s in best_strategies)}",
                'strategies': [s.strategy_id for s in best_strategies]
            }
            self.insights.append(insight)

        # Insight 2: Context-specific patterns
        for context, strategies in self.context_strategy_effectiveness.items():
            if len(strategies) >= 2:
                best_for_context = max(
                    strategies.items(),
                    key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0
                )

                if best_for_context[1] and (sum(best_for_context[1]) / len(best_for_context[1])) > 0.7:
                    strategy = self.learning_strategies[best_for_context[0]]
                    insight = {
                        'type': 'context_specific',
                        'timestamp': datetime.now().timestamp(),
                        'content': f"For '{context}' context, '{strategy.name}' works best",
                        'context': context,
                        'strategy': strategy.strategy_id
                    }
                    self.insights.append(insight)

        # Insight 3: Learning efficiency trends
        if len(self.learning_experiences) >= 10:
            recent_effectiveness = [
                exp.overall_effectiveness
                for exp in self.learning_experiences[-10:]
            ]
            older_effectiveness = [
                exp.overall_effectiveness
                for exp in self.learning_experiences[-20:-10]
            ] if len(self.learning_experiences) >= 20 else []

            if older_effectiveness:
                recent_avg = sum(recent_effectiveness) / len(recent_effectiveness)
                older_avg = sum(older_effectiveness) / len(older_effectiveness)

                if recent_avg > older_avg + 0.1:
                    insight = {
                        'type': 'improvement',
                        'timestamp': datetime.now().timestamp(),
                        'content': f"Learning effectiveness improving: {older_avg:.2f} → {recent_avg:.2f}",
                        'improvement': recent_avg - older_avg
                    }
                    self.insights.append(insight)
                elif recent_avg < older_avg - 0.1:
                    insight = {
                        'type': 'decline',
                        'timestamp': datetime.now().timestamp(),
                        'content': f"Learning effectiveness declining: {older_avg:.2f} → {recent_avg:.2f}",
                        'decline': older_avg - recent_avg
                    }
                    self.insights.append(insight)

        # Keep only recent insights
        if len(self.insights) > 50:
            self.insights = self.insights[-50:]

    def _adapt_learning_approach(self):
        """Adaptively adjust learning approach based on experience"""

        if len(self.learning_experiences) < 10:
            return

        # Analyze recent effectiveness
        recent_experiences = self.learning_experiences[-10:]
        avg_recent_effectiveness = sum(
            exp.overall_effectiveness for exp in recent_experiences
        ) / len(recent_experiences)

        # If learning is going well, maintain current mode
        if avg_recent_effectiveness > 0.75:
            # Doing well, no major changes needed
            pass

        # If learning struggling, consider changing mode
        elif avg_recent_effectiveness < 0.5:
            # Try a different learning mode
            modes = ["balanced", "fast", "deep", "exploratory"]
            current_idx = modes.index(self.current_learning_mode)
            self.current_learning_mode = modes[(current_idx + 1) % len(modes)]

            # Add insight
            self.insights.append({
                'type': 'mode_change',
                'timestamp': datetime.now().timestamp(),
                'content': f"Switching to '{self.current_learning_mode}' learning mode",
                'reason': 'low_effectiveness'
            })

        # Adjust learning rate
        if avg_recent_effectiveness > 0.8:
            # Learning well, can try to learn more/faster
            self.learning_rate_adjustment = min(1.5, self.learning_rate_adjustment * 1.1)
        elif avg_recent_effectiveness < 0.4:
            # Struggling, slow down
            self.learning_rate_adjustment = max(0.5, self.learning_rate_adjustment * 0.9)

    def get_learning_recommendations(self, context: str) -> Dict:
        """Get recommendations for learning in a specific context"""

        # Find best strategies for this context
        recommended_strategies = []

        for strategy in self.learning_strategies.values():
            score = self._score_strategy_for_context(strategy, context, None)
            if score > 0.6:
                recommended_strategies.append({
                    'strategy': strategy.name,
                    'description': strategy.description,
                    'score': score,
                    'effectiveness': strategy.avg_effectiveness
                })

        recommended_strategies.sort(key=lambda x: x['score'], reverse=True)

        # Find relevant insights
        relevant_insights = [
            insight for insight in self.insights[-10:]
            if insight.get('context', '') == context or insight['type'] in ['best_strategies', 'improvement']
        ]

        return {
            'context': context,
            'recommended_strategies': recommended_strategies[:3],
            'current_learning_mode': self.current_learning_mode,
            'learning_rate_adjustment': self.learning_rate_adjustment,
            'relevant_insights': relevant_insights
        }

    def create_learning_goal(self, description: str, target_knowledge: str, priority: float = 0.5) -> str:
        """Create a new learning goal"""
        goal_id = f"goal_{len(self.learning_goals)}"

        goal = LearningGoal(
            goal_id=goal_id,
            description=description,
            target_knowledge=target_knowledge,
            priority=priority
        )

        self.learning_goals[goal_id] = goal
        return goal_id

    def update_learning_goal(self, goal_id: str, progress: float, strategy_used: Optional[str] = None):
        """Update progress on a learning goal"""
        if goal_id not in self.learning_goals:
            return

        goal = self.learning_goals[goal_id]
        goal.progress = progress

        if strategy_used and strategy_used not in goal.strategies_tried:
            goal.strategies_tried.append(strategy_used)

        if progress >= 1.0:
            goal.achieved = True

    def get_learning_statistics(self) -> Dict:
        """Get comprehensive statistics about learning"""

        if not self.learning_experiences:
            return {
                'total_experiences': 0,
                'message': 'No learning experiences yet'
            }

        total_experiences = len(self.learning_experiences)
        successful_experiences = sum(1 for exp in self.learning_experiences if exp.success)

        recent_experiences = self.learning_experiences[-20:]
        avg_recent_effectiveness = sum(
            exp.overall_effectiveness for exp in recent_experiences
        ) / len(recent_experiences)

        # Strategy performance
        strategy_performance = []
        for strategy in self.learning_strategies.values():
            if strategy.usage_count > 0:
                strategy_performance.append({
                    'name': strategy.name,
                    'usage_count': strategy.usage_count,
                    'success_rate': strategy.success_count / strategy.usage_count,
                    'avg_effectiveness': strategy.avg_effectiveness
                })

        strategy_performance.sort(key=lambda x: x['avg_effectiveness'], reverse=True)

        return {
            'total_experiences': total_experiences,
            'successful_experiences': successful_experiences,
            'success_rate': successful_experiences / total_experiences,
            'avg_recent_effectiveness': avg_recent_effectiveness,
            'current_learning_mode': self.current_learning_mode,
            'learning_rate_adjustment': self.learning_rate_adjustment,
            'strategy_performance': strategy_performance[:5],
            'total_strategies': len(self.learning_strategies),
            'total_insights': len(self.insights),
            'recent_insights': self.insights[-5:],
            'learning_goals': {
                'total': len(self.learning_goals),
                'achieved': sum(1 for g in self.learning_goals.values() if g.achieved),
                'in_progress': sum(1 for g in self.learning_goals.values() if not g.achieved)
            }
        }

    def save_state(self, filepath: str):
        """Save meta-learning state"""
        state = {
            'learning_strategies': {
                k: v.to_dict() for k, v in self.learning_strategies.items()
            },
            'learning_experiences': [exp.to_dict() for exp in self.learning_experiences[-100:]],
            'learning_goals': {
                k: v.to_dict() for k, v in self.learning_goals.items()
            },
            'context_strategy_effectiveness': {
                k: {k2: v2 for k2, v2 in v.items()}
                for k, v in self.context_strategy_effectiveness.items()
            },
            'insights': self.insights[-50:],
            'current_learning_mode': self.current_learning_mode,
            'learning_rate_adjustment': self.learning_rate_adjustment,
            'counters': {
                'strategy': self.strategy_id_counter,
                'experience': self.experience_id_counter
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load meta-learning state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Load strategies
            self.learning_strategies = {
                k: LearningStrategy.from_dict(v)
                for k, v in state.get('learning_strategies', {}).items()
            }

            # Load experiences
            self.learning_experiences = [
                LearningExperience(**exp)
                for exp in state.get('learning_experiences', [])
            ]

            # Load goals
            self.learning_goals = {
                k: LearningGoal(**v)
                for k, v in state.get('learning_goals', {}).items()
            }

            # Load context effectiveness
            self.context_strategy_effectiveness = defaultdict(
                lambda: defaultdict(list),
                {k: defaultdict(list, v) for k, v in state.get('context_strategy_effectiveness', {}).items()}
            )

            # Load insights
            self.insights = state.get('insights', [])

            # Load settings
            self.current_learning_mode = state.get('current_learning_mode', 'balanced')
            self.learning_rate_adjustment = state.get('learning_rate_adjustment', 1.0)

            # Load counters
            counters = state.get('counters', {})
            self.strategy_id_counter = counters.get('strategy', len(self.learning_strategies))
            self.experience_id_counter = counters.get('experience', len(self.learning_experiences))

        except FileNotFoundError:
            pass
