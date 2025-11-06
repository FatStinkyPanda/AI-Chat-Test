"""
Cross-Session Pattern Aggregation and Transfer Learning System

This system aggregates patterns across conversations and sessions,
enabling transfer of learning from one context to another.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import json
import numpy as np


@dataclass
class AbstractPattern:
    """Represents an abstract pattern that can transfer across contexts"""
    pattern_id: str
    pattern_type: str  # behavioral, conceptual, structural, causal, etc.
    abstract_representation: str
    concrete_instances: List[Dict] = field(default_factory=list)
    contexts_seen: Set[str] = field(default_factory=set)
    transferability_score: float = 0.5
    confidence: float = 0.5
    successful_transfers: int = 0
    failed_transfers: int = 0
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'abstract_representation': self.abstract_representation,
            'concrete_instances': self.concrete_instances[-20:],
            'contexts_seen': list(self.contexts_seen),
            'transferability_score': self.transferability_score,
            'confidence': self.confidence,
            'successful_transfers': self.successful_transfers,
            'failed_transfers': self.failed_transfers,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AbstractPattern':
        data['contexts_seen'] = set(data.get('contexts_seen', []))
        return cls(**data)


@dataclass
class TransferAttempt:
    """Represents an attempt to transfer learning from one context to another"""
    transfer_id: str
    source_context: str
    target_context: str
    pattern_transferred: str
    success: bool
    similarity_score: float
    adaptation_required: str  # How much adaptation was needed
    outcome: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'transfer_id': self.transfer_id,
            'source_context': self.source_context,
            'target_context': self.target_context,
            'pattern_transferred': self.pattern_transferred,
            'success': self.success,
            'similarity_score': self.similarity_score,
            'adaptation_required': self.adaptation_required,
            'outcome': self.outcome,
            'timestamp': self.timestamp
        }


@dataclass
class SessionSummary:
    """Summary of learning from a conversation session"""
    session_id: str
    topics_discussed: List[str]
    patterns_learned: List[str]
    key_insights: List[str]
    entities_encountered: List[str]
    relationships_formed: List[str]
    duration: float
    turn_count: int
    learning_quality: float  # How much was learned
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'topics_discussed': self.topics_discussed,
            'patterns_learned': self.patterns_learned,
            'key_insights': self.key_insights,
            'entities_encountered': self.entities_encountered,
            'relationships_formed': self.relationships_formed,
            'duration': self.duration,
            'turn_count': self.turn_count,
            'learning_quality': self.learning_quality,
            'timestamp': self.timestamp
        }


class TransferLearningSystem:
    """
    System for aggregating patterns across sessions and enabling transfer learning.
    """

    def __init__(self):
        self.abstract_patterns: Dict[str, AbstractPattern] = {}
        self.transfer_attempts: List[TransferAttempt] = []
        self.session_summaries: List[SessionSummary] = []

        # Cross-session aggregation
        self.global_topic_knowledge: Dict[str, Dict] = defaultdict(lambda: {
            'total_discussions': 0,
            'contexts': set(),
            'related_topics': defaultdict(int),
            'typical_patterns': []
        })

        self.context_similarities: Dict[Tuple[str, str], float] = {}
        self.successful_transfer_paths: List[Tuple[str, str]] = []  # (source, target) pairs

        # Counters
        self.pattern_id_counter = 0
        self.transfer_id_counter = 0
        self.session_id_counter = 0

    def aggregate_session_patterns(
        self,
        topics: List[str],
        patterns: List[Dict],
        insights: List[str],
        entities: List[str],
        relationships: List[str],
        duration: float,
        turn_count: int
    ) -> SessionSummary:
        """
        Aggregate patterns from a conversation session.
        """

        # Calculate learning quality
        learning_quality = self._calculate_learning_quality(
            patterns, insights, turn_count
        )

        # Create session summary
        session_id = f"session_{self.session_id_counter}"
        self.session_id_counter += 1

        summary = SessionSummary(
            session_id=session_id,
            topics_discussed=topics,
            patterns_learned=[p.get('pattern_id', 'unknown') for p in patterns],
            key_insights=insights,
            entities_encountered=entities,
            relationships_formed=relationships,
            duration=duration,
            turn_count=turn_count,
            learning_quality=learning_quality
        )

        self.session_summaries.append(summary)

        # Update global topic knowledge
        for topic in topics:
            self.global_topic_knowledge[topic]['total_discussions'] += 1
            self.global_topic_knowledge[topic]['contexts'].add(session_id)

            # Track topic co-occurrences
            for other_topic in topics:
                if other_topic != topic:
                    self.global_topic_knowledge[topic]['related_topics'][other_topic] += 1

        # Abstract patterns from this session
        for pattern_data in patterns:
            self._abstract_pattern(pattern_data, session_id, topics)

        return summary

    def _calculate_learning_quality(
        self,
        patterns: List[Dict],
        insights: List[str],
        turn_count: int
    ) -> float:
        """Calculate how much quality learning happened"""

        quality = 0.5  # Base

        # More patterns learned = higher quality
        quality += min(0.3, len(patterns) * 0.05)

        # Insights indicate deep understanding
        quality += min(0.2, len(insights) * 0.04)

        # Longer conversations generally have more learning opportunities
        if turn_count > 10:
            quality += 0.1

        return min(1.0, quality)

    def _abstract_pattern(
        self,
        pattern_data: Dict,
        session_id: str,
        contexts: List[str]
    ):
        """
        Abstract a pattern to make it transferable.
        """

        # Extract key features
        pattern_type = pattern_data.get('type', 'general')
        concrete_description = pattern_data.get('description', '')

        # Create abstract representation (remove context-specific details)
        abstract_repr = self._create_abstract_representation(
            concrete_description, pattern_type
        )

        # Check if we already have this abstract pattern
        existing_pattern_id = self._find_similar_abstract_pattern(abstract_repr)

        if existing_pattern_id:
            # Update existing pattern
            pattern = self.abstract_patterns[existing_pattern_id]
            pattern.concrete_instances.append({
                'session': session_id,
                'description': concrete_description,
                'context': contexts
            })
            pattern.contexts_seen.update(contexts)
            pattern.confidence = min(1.0, pattern.confidence + 0.1)

        else:
            # Create new abstract pattern
            pattern_id = f"pattern_{self.pattern_id_counter}"
            self.pattern_id_counter += 1

            pattern = AbstractPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                abstract_representation=abstract_repr,
                concrete_instances=[{
                    'session': session_id,
                    'description': concrete_description,
                    'context': contexts
                }],
                contexts_seen=set(contexts)
            )

            self.abstract_patterns[pattern_id] = pattern

    def _create_abstract_representation(
        self,
        concrete_description: str,
        pattern_type: str
    ) -> str:
        """
        Create abstract representation of a pattern.
        Removes specific details, keeps structure.
        """

        # Simple abstraction (could be enhanced)
        abstract = concrete_description

        # Remove specific names (capitals)
        words = abstract.split()
        abstracted_words = []

        for word in words:
            if word[0].isupper() and word not in ['I', 'A', 'The']:
                abstracted_words.append('[ENTITY]')
            elif word.isdigit():
                abstracted_words.append('[NUMBER]')
            else:
                abstracted_words.append(word.lower())

        abstract = ' '.join(abstracted_words)

        # Add pattern type prefix for grouping
        abstract = f"[{pattern_type}] {abstract}"

        return abstract

    def _find_similar_abstract_pattern(self, abstract_repr: str) -> Optional[str]:
        """Find if we already have a similar abstract pattern"""

        for pattern_id, pattern in self.abstract_patterns.items():
            similarity = self._calculate_pattern_similarity(
                abstract_repr,
                pattern.abstract_representation
            )

            if similarity > 0.8:  # High similarity threshold
                return pattern_id

        return None

    def _calculate_pattern_similarity(self, repr1: str, repr2: str) -> float:
        """Calculate similarity between two abstract representations"""

        words1 = set(repr1.lower().split())
        words2 = set(repr2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def attempt_transfer(
        self,
        source_context: str,
        target_context: str,
        current_situation: str
    ) -> Dict:
        """
        Attempt to transfer learning from source context to target context.
        """

        # Find patterns from source context
        source_patterns = [
            pattern for pattern in self.abstract_patterns.values()
            if source_context in pattern.contexts_seen
        ]

        if not source_patterns:
            return {
                'transfer_possible': False,
                'reason': 'No patterns found in source context'
            }

        # Calculate context similarity
        context_similarity = self._calculate_context_similarity(
            source_context, target_context
        )

        # Select most transferable pattern
        best_pattern = max(
            source_patterns,
            key=lambda p: p.transferability_score * p.confidence
        )

        # Attempt transfer
        transfer_success = context_similarity > 0.3  # Threshold for transfer

        # Determine adaptation needed
        if context_similarity > 0.7:
            adaptation = "minimal"
        elif context_similarity > 0.5:
            adaptation = "moderate"
        else:
            adaptation = "significant"

        # Record attempt
        transfer_id = f"transfer_{self.transfer_id_counter}"
        self.transfer_id_counter += 1

        attempt = TransferAttempt(
            transfer_id=transfer_id,
            source_context=source_context,
            target_context=target_context,
            pattern_transferred=best_pattern.pattern_id,
            success=transfer_success,
            similarity_score=context_similarity,
            adaptation_required=adaptation,
            outcome="success" if transfer_success else "failed"
        )

        self.transfer_attempts.append(attempt)

        # Update pattern transferability
        if transfer_success:
            best_pattern.successful_transfers += 1
            best_pattern.transferability_score = min(
                1.0,
                best_pattern.transferability_score + 0.05
            )
            self.successful_transfer_paths.append((source_context, target_context))
        else:
            best_pattern.failed_transfers += 1
            best_pattern.transferability_score = max(
                0.1,
                best_pattern.transferability_score - 0.03
            )

        # Generate adapted recommendation
        adapted_recommendation = self._adapt_pattern_to_context(
            best_pattern,
            target_context,
            current_situation
        )

        return {
            'transfer_possible': True,
            'pattern': best_pattern.abstract_representation,
            'confidence': best_pattern.confidence * best_pattern.transferability_score * context_similarity,
            'adaptation_needed': adaptation,
            'recommendation': adapted_recommendation,
            'similar_past_situations': self._find_similar_past_situations(current_situation, 3)
        }

    def _calculate_context_similarity(self, context1: str, context2: str) -> float:
        """Calculate similarity between two contexts"""

        # Check cache
        key = tuple(sorted([context1, context2]))
        if key in self.context_similarities:
            return self.context_similarities[key]

        # Calculate similarity based on:
        # 1. Shared topics
        topics1 = set(context1.lower().split())
        topics2 = set(context2.lower().split())

        word_similarity = len(topics1 & topics2) / max(len(topics1), len(topics2)) if topics1 or topics2 else 0

        # 2. Shared patterns
        patterns1 = {
            p.pattern_id for p in self.abstract_patterns.values()
            if context1 in p.contexts_seen
        }
        patterns2 = {
            p.pattern_id for p in self.abstract_patterns.values()
            if context2 in p.contexts_seen
        }

        pattern_similarity = len(patterns1 & patterns2) / max(len(patterns1), len(patterns2)) if patterns1 or patterns2 else 0

        # Combined similarity
        similarity = 0.6 * word_similarity + 0.4 * pattern_similarity

        # Cache it
        self.context_similarities[key] = similarity

        return similarity

    def _adapt_pattern_to_context(
        self,
        pattern: AbstractPattern,
        target_context: str,
        current_situation: str
    ) -> str:
        """Adapt an abstract pattern to a specific context"""

        base_repr = pattern.abstract_representation

        # Simple adaptation: inject context
        adapted = f"In the context of {target_context}, {base_repr.lower()}"

        # Add situation-specific details if available
        if current_situation:
            adapted += f". Specifically for: {current_situation}"

        return adapted

    def _find_similar_past_situations(
        self,
        current_situation: str,
        top_k: int = 3
    ) -> List[Dict]:
        """Find similar past situations across all sessions"""

        similar_situations = []

        # Search through all patterns
        for pattern in self.abstract_patterns.values():
            for instance in pattern.concrete_instances:
                description = instance.get('description', '')

                similarity = self._calculate_pattern_similarity(
                    current_situation,
                    description
                )

                if similarity > 0.3:
                    similar_situations.append({
                        'description': description,
                        'context': instance.get('context', []),
                        'session': instance.get('session', 'unknown'),
                        'similarity': similarity,
                        'pattern_type': pattern.pattern_type
                    })

        # Sort by similarity
        similar_situations.sort(key=lambda x: x['similarity'], reverse=True)

        return similar_situations[:top_k]

    def get_cross_session_insights(self) -> Dict:
        """Get insights aggregated across all sessions"""

        if not self.session_summaries:
            return {'message': 'No sessions to analyze yet'}

        # Most discussed topics across all sessions
        all_topics = defaultdict(int)
        for summary in self.session_summaries:
            for topic in summary.topics_discussed:
                all_topics[topic] += 1

        top_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:10]

        # Most transferable patterns
        transferable_patterns = sorted(
            self.abstract_patterns.values(),
            key=lambda p: p.transferability_score * p.confidence,
            reverse=True
        )[:5]

        # Transfer success rate
        total_transfers = len(self.transfer_attempts)
        successful_transfers = sum(1 for t in self.transfer_attempts if t.success)
        transfer_success_rate = successful_transfers / total_transfers if total_transfers > 0 else 0

        # Average learning quality
        avg_learning_quality = (
            sum(s.learning_quality for s in self.session_summaries) /
            len(self.session_summaries)
        )

        # Most successful transfer paths
        transfer_path_counts = defaultdict(int)
        for source, target in self.successful_transfer_paths:
            transfer_path_counts[(source, target)] += 1

        top_transfer_paths = sorted(
            transfer_path_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'total_sessions': len(self.session_summaries),
            'total_patterns_learned': len(self.abstract_patterns),
            'top_topics_across_sessions': top_topics,
            'most_transferable_patterns': [
                {
                    'pattern': p.abstract_representation,
                    'transferability': p.transferability_score,
                    'confidence': p.confidence,
                    'contexts': list(p.contexts_seen)
                }
                for p in transferable_patterns
            ],
            'transfer_learning_stats': {
                'total_attempts': total_transfers,
                'successful_transfers': successful_transfers,
                'success_rate': transfer_success_rate
            },
            'avg_learning_quality': avg_learning_quality,
            'top_transfer_paths': [
                {
                    'from': path[0],
                    'to': path[1],
                    'success_count': count
                }
                for path, count in top_transfer_paths
            ]
        }

    def get_topic_knowledge_aggregation(self, topic: str) -> Dict:
        """Get aggregated knowledge about a topic from all sessions"""

        if topic not in self.global_topic_knowledge:
            return {'message': f'No knowledge about topic: {topic}'}

        knowledge = self.global_topic_knowledge[topic]

        # Find patterns related to this topic
        related_patterns = [
            pattern for pattern in self.abstract_patterns.values()
            if topic.lower() in pattern.abstract_representation.lower()
        ]

        # Find related insights from sessions
        related_insights = []
        for summary in self.session_summaries:
            if topic in summary.topics_discussed:
                related_insights.extend(summary.key_insights)

        return {
            'topic': topic,
            'total_discussions': knowledge['total_discussions'],
            'contexts_discussed': list(knowledge['contexts']),
            'related_topics': sorted(
                knowledge['related_topics'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'related_patterns': [
                {
                    'pattern': p.abstract_representation,
                    'confidence': p.confidence,
                    'contexts': list(p.contexts_seen)
                }
                for p in related_patterns[:5]
            ],
            'key_insights': related_insights[:5]
        }

    def save_state(self, filepath: str):
        """Save transfer learning state"""
        state = {
            'abstract_patterns': {
                k: v.to_dict() for k, v in self.abstract_patterns.items()
            },
            'transfer_attempts': [t.to_dict() for t in self.transfer_attempts[-100:]],
            'session_summaries': [s.to_dict() for s in self.session_summaries[-50:]],
            'global_topic_knowledge': {
                k: {
                    'total_discussions': v['total_discussions'],
                    'contexts': list(v['contexts']),
                    'related_topics': dict(v['related_topics']),
                    'typical_patterns': v['typical_patterns']
                }
                for k, v in self.global_topic_knowledge.items()
            },
            'context_similarities': {
                f"{k[0]}||{k[1]}": v for k, v in self.context_similarities.items()
            },
            'successful_transfer_paths': self.successful_transfer_paths[-50:],
            'counters': {
                'pattern': self.pattern_id_counter,
                'transfer': self.transfer_id_counter,
                'session': self.session_id_counter
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load transfer learning state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Load patterns
            self.abstract_patterns = {
                k: AbstractPattern.from_dict(v)
                for k, v in state.get('abstract_patterns', {}).items()
            }

            # Load transfer attempts
            self.transfer_attempts = [
                TransferAttempt(**t)
                for t in state.get('transfer_attempts', [])
            ]

            # Load session summaries
            self.session_summaries = [
                SessionSummary(**s)
                for s in state.get('session_summaries', [])
            ]

            # Load global topic knowledge
            for topic, knowledge in state.get('global_topic_knowledge', {}).items():
                self.global_topic_knowledge[topic] = {
                    'total_discussions': knowledge['total_discussions'],
                    'contexts': set(knowledge['contexts']),
                    'related_topics': defaultdict(int, knowledge['related_topics']),
                    'typical_patterns': knowledge['typical_patterns']
                }

            # Load context similarities
            for key_str, similarity in state.get('context_similarities', {}).items():
                contexts = key_str.split('||')
                self.context_similarities[tuple(contexts)] = similarity

            # Load transfer paths
            self.successful_transfer_paths = [
                tuple(path) for path in state.get('successful_transfer_paths', [])
            ]

            # Load counters
            counters = state.get('counters', {})
            self.pattern_id_counter = counters.get('pattern', 0)
            self.transfer_id_counter = counters.get('transfer', 0)
            self.session_id_counter = counters.get('session', 0)

        except FileNotFoundError:
            pass
