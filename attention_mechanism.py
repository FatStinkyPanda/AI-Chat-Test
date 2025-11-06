"""
Advanced Attention Mechanism

This system intelligently selects and weights the most relevant information
for processing, mimicking human selective attention.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import json


@dataclass
class AttentionScore:
    """Represents attention score for an item"""
    item_id: str
    score: float
    relevance_score: float
    recency_score: float
    importance_score: float
    emotional_score: float
    contextual_score: float
    final_weight: float

    def to_dict(self) -> Dict:
        return {
            'item_id': self.item_id,
            'score': self.score,
            'relevance_score': self.relevance_score,
            'recency_score': self.recency_score,
            'importance_score': self.importance_score,
            'emotional_score': self.emotional_score,
            'contextual_score': self.contextual_score,
            'final_weight': self.final_weight
        }


@dataclass
class AttentionContext:
    """Represents current attention context"""
    focus_items: List[str]  # What we're currently focused on
    focus_strength: float  # How strongly focused (0-1)
    attention_breadth: float  # How broad is attention (0-1, higher = broader)
    distractibility: float  # How easily distracted (0-1)
    novelty_seeking: float  # Preference for novel info (0-1)

    def to_dict(self) -> Dict:
        return {
            'focus_items': self.focus_items,
            'focus_strength': self.focus_strength,
            'attention_breadth': self.attention_breadth,
            'distractibility': self.distractibility,
            'novelty_seeking': self.novelty_seeking
        }


class AttentionMechanism:
    """
    Advanced attention mechanism for selective information processing.
    Uses multi-headed attention with various attention types:
    - Semantic attention (relevance to current context)
    - Temporal attention (recency)
    - Importance attention (significance)
    - Emotional attention (emotional salience)
    - Novelty attention (new vs. familiar)
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.attention_history: List[Dict] = []
        self.item_attention_stats: Dict[str, Dict] = {}  # Track what gets attention

        # Attention parameters (learnable weights)
        self.attention_weights = {
            'semantic': 0.35,      # Relevance to current context
            'temporal': 0.15,      # Recency
            'importance': 0.20,    # Significance/impact
            'emotional': 0.15,     # Emotional salience
            'novelty': 0.10,       # Newness/unexpectedness
            'contextual': 0.05     # Fit with current goals
        }

        # Current attention context
        self.context = AttentionContext(
            focus_items=[],
            focus_strength=0.7,
            attention_breadth=0.5,
            distractibility=0.3,
            novelty_seeking=0.4
        )

    def compute_attention(
        self,
        query_embedding: np.ndarray,
        items: List[Dict],
        item_embeddings: List[np.ndarray],
        top_k: Optional[int] = None,
        attention_mode: str = "balanced"
    ) -> List[AttentionScore]:
        """
        Compute attention scores for items given a query.

        Args:
            query_embedding: Embedding representing what we're looking for
            items: List of items with metadata (id, content, timestamp, importance, etc.)
            item_embeddings: Embeddings for each item
            top_k: Return only top k items
            attention_mode: "focused", "balanced", "exploratory"

        Returns:
            List of AttentionScore objects sorted by attention score
        """

        if len(items) != len(item_embeddings):
            raise ValueError("Items and embeddings must have same length")

        # Adjust attention context based on mode
        self._set_attention_mode(attention_mode)

        attention_scores = []

        for item, embedding in zip(items, item_embeddings):
            # Compute various attention components
            semantic_score = self._compute_semantic_attention(query_embedding, embedding)
            temporal_score = self._compute_temporal_attention(item.get('timestamp', 0))
            importance_score = self._compute_importance_attention(item)
            emotional_score = self._compute_emotional_attention(item, query_embedding)
            novelty_score = self._compute_novelty_attention(item)
            contextual_score = self._compute_contextual_attention(item)

            # Weighted combination
            final_score = (
                semantic_score * self.attention_weights['semantic'] +
                temporal_score * self.attention_weights['temporal'] +
                importance_score * self.attention_weights['importance'] +
                emotional_score * self.attention_weights['emotional'] +
                novelty_score * self.attention_weights['novelty'] +
                contextual_score * self.attention_weights['contextual']
            )

            # Apply attention context modulation
            final_score = self._modulate_with_context(final_score, semantic_score, novelty_score)

            # Create attention score object
            att_score = AttentionScore(
                item_id=item.get('id', 'unknown'),
                score=final_score,
                relevance_score=semantic_score,
                recency_score=temporal_score,
                importance_score=importance_score,
                emotional_score=emotional_score,
                contextual_score=contextual_score,
                final_weight=0.0  # Will be set after softmax
            )

            attention_scores.append(att_score)

        # Apply softmax to get attention weights
        attention_scores = self._apply_attention_weights(attention_scores)

        # Sort by score
        attention_scores.sort(key=lambda x: x.score, reverse=True)

        # Track attention
        self._record_attention(attention_scores[:top_k] if top_k else attention_scores)

        # Return top k if specified
        if top_k:
            return attention_scores[:top_k]

        return attention_scores

    def _compute_semantic_attention(
        self,
        query_embedding: np.ndarray,
        item_embedding: np.ndarray
    ) -> float:
        """Compute semantic similarity attention"""
        # Cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        item_norm = np.linalg.norm(item_embedding)

        if query_norm == 0 or item_norm == 0:
            return 0.0

        similarity = np.dot(query_embedding, item_embedding) / (query_norm * item_norm)

        # Convert from [-1, 1] to [0, 1]
        normalized = (similarity + 1) / 2

        return float(normalized)

    def _compute_temporal_attention(self, timestamp: float) -> float:
        """Compute attention based on recency"""
        # Ensure timestamp is numeric
        if isinstance(timestamp, str):
            try:
                timestamp = float(timestamp)
            except:
                return 0.5  # Default if can't parse

        if timestamp is None or timestamp == 0:
            return 0.5  # Default for missing timestamps

        current_time = datetime.now().timestamp()
        time_diff = current_time - timestamp

        # Handle negative time diff (future timestamps - shouldn't happen but be safe)
        if time_diff < 0:
            return 1.0

        # Exponential decay (half-life of 1 day = 86400 seconds)
        half_life = 86400
        decay_rate = np.log(2) / half_life

        recency_score = np.exp(-decay_rate * time_diff)

        return float(recency_score)

    def _compute_importance_attention(self, item: Dict) -> float:
        """Compute attention based on item importance"""
        # Multiple factors determine importance
        importance = 0.5  # Base importance

        # Explicit importance score
        if 'importance' in item:
            importance = item['importance']

        # Activation level (from spreading activation)
        if 'activation_level' in item:
            importance = max(importance, item['activation_level'])

        # Number of connections (well-connected = important)
        if 'connection_count' in item:
            conn_score = min(1.0, item['connection_count'] / 10.0)
            importance = max(importance, conn_score)

        # User-marked as important
        if item.get('marked_important', False):
            importance = max(importance, 0.9)

        return importance

    def _compute_emotional_attention(self, item: Dict, query_embedding: np.ndarray) -> float:
        """Compute attention based on emotional salience"""
        # Emotionally charged content gets more attention
        emotional_score = 0.5

        # Check emotional valence
        if 'emotional_valence' in item:
            valence = item['emotional_valence']
            # Both strong positive and negative emotions attract attention
            emotional_score = abs(valence)

        # Check emotional intensity
        if 'emotional_intensity' in item:
            intensity = item['emotional_intensity']
            emotional_score = max(emotional_score, intensity)

        # Emotion keywords
        emotion_keywords = ['love', 'hate', 'fear', 'joy', 'sad', 'angry', 'excited', 'worried']
        content = item.get('content', '').lower()
        emotion_count = sum(1 for keyword in emotion_keywords if keyword in content)

        if emotion_count > 0:
            emotional_score = max(emotional_score, min(1.0, 0.5 + emotion_count * 0.15))

        return emotional_score

    def _compute_novelty_attention(self, item: Dict) -> float:
        """Compute attention based on novelty"""
        novelty_score = 0.5

        item_id = item.get('id', 'unknown')

        # Check if we've attended to this before
        if item_id in self.item_attention_stats:
            stats = self.item_attention_stats[item_id]
            attention_count = stats.get('attention_count', 0)

            # Novel items get higher scores
            novelty_score = max(0.1, 1.0 - (attention_count / 10.0))
        else:
            # Completely new item
            novelty_score = 1.0

        # Newly created items
        if 'timestamp' in item:
            age_hours = (datetime.now().timestamp() - item['timestamp']) / 3600
            if age_hours < 1:
                novelty_score = max(novelty_score, 0.9)
            elif age_hours < 24:
                novelty_score = max(novelty_score, 0.7)

        return novelty_score

    def _compute_contextual_attention(self, item: Dict) -> float:
        """Compute attention based on fit with current context"""
        contextual_score = 0.5

        item_id = item.get('id', 'unknown')

        # Check if item relates to current focus
        if self.context.focus_items:
            if item_id in self.context.focus_items:
                contextual_score = 1.0
            elif any(focus in item.get('content', '') for focus in self.context.focus_items):
                contextual_score = 0.8

        # Check item metadata for context markers
        if 'context_tags' in item and self.context.focus_items:
            tags = set(item['context_tags'])
            focus_set = set(self.context.focus_items)
            overlap = len(tags & focus_set)

            if overlap > 0:
                contextual_score = max(contextual_score, min(1.0, 0.6 + overlap * 0.2))

        return contextual_score

    def _modulate_with_context(
        self,
        base_score: float,
        semantic_score: float,
        novelty_score: float
    ) -> float:
        """Modulate attention score based on current attention context"""

        modulated = base_score

        # Focus strength affects attention narrowing
        if self.context.focus_strength > 0.7:
            # Strong focus = boost highly relevant, suppress less relevant
            if semantic_score > 0.7:
                modulated *= 1.2
            elif semantic_score < 0.4:
                modulated *= 0.6

        # Attention breadth affects how many items get attention
        if self.context.attention_breadth > 0.7:
            # Broad attention = more uniform scores
            modulated = 0.4 * modulated + 0.6 * 0.5
        elif self.context.attention_breadth < 0.3:
            # Narrow attention = more extreme scores
            if modulated > 0.6:
                modulated = min(1.0, modulated * 1.3)
            else:
                modulated *= 0.7

        # Novelty seeking affects attention to new items
        if self.context.novelty_seeking > 0.7:
            # High novelty seeking = boost novel items
            modulated = 0.7 * modulated + 0.3 * novelty_score
        elif self.context.novelty_seeking < 0.3:
            # Low novelty seeking = boost familiar items
            modulated = 0.8 * modulated + 0.2 * (1 - novelty_score)

        return modulated

    def _set_attention_mode(self, mode: str):
        """Set attention context based on mode"""
        if mode == "focused":
            self.context.focus_strength = 0.9
            self.context.attention_breadth = 0.3
            self.context.distractibility = 0.1
            self.context.novelty_seeking = 0.3
        elif mode == "exploratory":
            self.context.focus_strength = 0.4
            self.context.attention_breadth = 0.8
            self.context.distractibility = 0.6
            self.context.novelty_seeking = 0.8
        else:  # balanced
            self.context.focus_strength = 0.7
            self.context.attention_breadth = 0.5
            self.context.distractibility = 0.3
            self.context.novelty_seeking = 0.5

    def _apply_attention_weights(self, attention_scores: List[AttentionScore]) -> List[AttentionScore]:
        """Apply softmax to convert scores to weights that sum to 1"""
        if not attention_scores:
            return attention_scores

        # Softmax with temperature
        temperature = 0.5  # Lower = more focused, higher = more uniform
        scores = np.array([a.score for a in attention_scores])

        # Numerical stability
        scores = scores / temperature
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / np.sum(exp_scores)

        # Assign weights
        for i, att_score in enumerate(attention_scores):
            att_score.final_weight = float(weights[i])

        return attention_scores

    def _record_attention(self, attention_scores: List[AttentionScore]):
        """Record what received attention for learning"""
        timestamp = datetime.now().timestamp()

        for att_score in attention_scores:
            item_id = att_score.item_id

            if item_id not in self.item_attention_stats:
                self.item_attention_stats[item_id] = {
                    'attention_count': 0,
                    'total_attention_weight': 0.0,
                    'last_attended': 0.0,
                    'avg_attention_score': 0.0
                }

            stats = self.item_attention_stats[item_id]
            stats['attention_count'] += 1
            stats['total_attention_weight'] += att_score.final_weight
            stats['last_attended'] = timestamp
            stats['avg_attention_score'] = (
                stats['total_attention_weight'] / stats['attention_count']
            )

        # Keep attention history (last 100 events)
        self.attention_history.append({
            'timestamp': timestamp,
            'top_items': [a.item_id for a in attention_scores[:5]],
            'context': self.context.to_dict()
        })

        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]

    def update_focus(self, new_focus_items: List[str], focus_strength: float = 0.7):
        """Update attention focus"""
        self.context.focus_items = new_focus_items
        self.context.focus_strength = focus_strength

    def adjust_attention_weights(self, component: str, new_weight: float):
        """Adjust attention component weights (learning)"""
        if component in self.attention_weights:
            self.attention_weights[component] = max(0.0, min(1.0, new_weight))

            # Renormalize weights to sum to 1
            total = sum(self.attention_weights.values())
            if total > 0:
                self.attention_weights = {
                    k: v / total for k, v in self.attention_weights.items()
                }

    def get_attention_summary(self, item_id: str) -> Optional[Dict]:
        """Get attention statistics for an item"""
        return self.item_attention_stats.get(item_id)

    def get_most_attended_items(self, top_k: int = 10) -> List[Tuple[str, Dict]]:
        """Get items that have received the most attention"""
        sorted_items = sorted(
            self.item_attention_stats.items(),
            key=lambda x: x[1]['attention_count'],
            reverse=True
        )
        return sorted_items[:top_k]

    def get_recently_attended_items(self, top_k: int = 10) -> List[Tuple[str, Dict]]:
        """Get items that were recently attended to"""
        sorted_items = sorted(
            self.item_attention_stats.items(),
            key=lambda x: x[1]['last_attended'],
            reverse=True
        )
        return sorted_items[:top_k]

    def compute_multi_head_attention(
        self,
        query_embedding: np.ndarray,
        items: List[Dict],
        item_embeddings: List[np.ndarray],
        num_heads: int = 4,
        top_k: Optional[int] = None
    ) -> List[AttentionScore]:
        """
        Compute multi-head attention (different attention modes in parallel).
        """
        all_scores = []

        # Different attention heads with different parameters
        modes = ["focused", "balanced", "exploratory"]

        for head_idx in range(num_heads):
            mode = modes[head_idx % len(modes)]

            # Temporarily set mode
            old_context = self.context
            head_scores = self.compute_attention(
                query_embedding,
                items,
                item_embeddings,
                top_k=None,
                attention_mode=mode
            )
            self.context = old_context

            all_scores.append(head_scores)

        # Aggregate scores from all heads
        item_score_map = {}

        for head_scores in all_scores:
            for att_score in head_scores:
                item_id = att_score.item_id
                if item_id not in item_score_map:
                    item_score_map[item_id] = []
                item_score_map[item_id].append(att_score.score)

        # Average scores across heads
        final_scores = []
        for item_id, scores in item_score_map.items():
            avg_score = sum(scores) / len(scores)

            # Find original attention score for metadata
            original = next(s for s in all_scores[0] if s.item_id == item_id)

            final_att = AttentionScore(
                item_id=item_id,
                score=avg_score,
                relevance_score=original.relevance_score,
                recency_score=original.recency_score,
                importance_score=original.importance_score,
                emotional_score=original.emotional_score,
                contextual_score=original.contextual_score,
                final_weight=0.0
            )
            final_scores.append(final_att)

        # Apply weights and sort
        final_scores = self._apply_attention_weights(final_scores)
        final_scores.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            return final_scores[:top_k]

        return final_scores

    def save_state(self, filepath: str):
        """Save attention mechanism state"""
        state = {
            'attention_weights': self.attention_weights,
            'context': self.context.to_dict(),
            'item_attention_stats': self.item_attention_stats,
            'attention_history': self.attention_history[-50:]  # Keep recent history
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load attention mechanism state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.attention_weights = state.get('attention_weights', self.attention_weights)

            context_data = state.get('context', {})
            self.context = AttentionContext(**context_data)

            self.item_attention_stats = state.get('item_attention_stats', {})
            self.attention_history = state.get('attention_history', [])

        except FileNotFoundError:
            pass

    def get_statistics(self) -> Dict:
        """Get attention statistics"""
        return {
            'total_items_attended': len(self.item_attention_stats),
            'attention_events': len(self.attention_history),
            'current_focus': self.context.focus_items,
            'attention_weights': self.attention_weights,
            'most_attended_items': self.get_most_attended_items(5),
            'avg_attention_per_item': (
                sum(s['attention_count'] for s in self.item_attention_stats.values()) /
                len(self.item_attention_stats)
                if self.item_attention_stats else 0
            )
        }
