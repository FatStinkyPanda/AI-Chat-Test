"""
Predictive Conversation Trajectory System

This system predicts where conversations are likely to go based on learned patterns,
enabling proactive and anticipatory responses.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import json
import numpy as np


@dataclass
class ConversationState:
    """Represents the current state of a conversation"""
    current_topics: List[str]
    recent_intents: List[str]  # Last few user intents
    emotional_trajectory: List[str]  # Emotional progression
    turn_count: int
    topic_shifts: int
    engagement_level: float  # 0-1
    conversation_depth: float  # How deep/superficial (0-1)

    def to_dict(self) -> Dict:
        return {
            'current_topics': self.current_topics,
            'recent_intents': self.recent_intents,
            'emotional_trajectory': self.emotional_trajectory,
            'turn_count': self.turn_count,
            'topic_shifts': self.topic_shifts,
            'engagement_level': self.engagement_level,
            'conversation_depth': self.conversation_depth
        }


@dataclass
class TrajectoryPrediction:
    """Represents a predicted conversation trajectory"""
    predicted_topics: List[Tuple[str, float]]  # (topic, probability)
    predicted_intents: List[Tuple[str, float]]  # (intent, probability)
    predicted_emotions: List[Tuple[str, float]]  # (emotion, probability)
    likely_duration: int  # Predicted number of turns remaining
    conversation_ending_probability: float
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict:
        return {
            'predicted_topics': self.predicted_topics,
            'predicted_intents': self.predicted_intents,
            'predicted_emotions': self.predicted_emotions,
            'likely_duration': self.likely_duration,
            'conversation_ending_probability': self.conversation_ending_probability,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }


@dataclass
class ConversationPattern:
    """Learned pattern of how conversations evolve"""
    pattern_id: str
    state_sequence: List[Dict]  # Sequence of conversation states
    transition_probabilities: Dict[str, Dict[str, float]]  # State transitions
    typical_duration: int
    success_rate: float  # How often this pattern leads to satisfactory conversations
    occurrence_count: int
    last_seen: float

    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'state_sequence': self.state_sequence,
            'transition_probabilities': self.transition_probabilities,
            'typical_duration': self.typical_duration,
            'success_rate': self.success_rate,
            'occurrence_count': self.occurrence_count,
            'last_seen': self.last_seen
        }


class ConversationPredictor:
    """
    Predicts conversation trajectories based on learned patterns and current state.
    """

    def __init__(self):
        self.conversation_patterns: Dict[str, ConversationPattern] = {}
        self.topic_transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.intent_transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.emotion_transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.topic_duration_stats: Dict[str, List[int]] = defaultdict(list)
        self.conversation_histories: List[Dict] = []

        # Ending signals
        self.ending_signals = [
            'goodbye', 'bye', 'see you', 'talk later', 'got to go', 'gotta go',
            'thanks for', 'thank you', 'appreciated', 'that helps', 'perfect'
        ]

    def update_state(
        self,
        previous_state: Optional[ConversationState],
        new_topics: List[str],
        new_intent: str,
        new_emotion: str,
        user_text: str
    ) -> ConversationState:
        """Update conversation state with new turn"""

        if previous_state is None:
            # Initialize new conversation
            return ConversationState(
                current_topics=new_topics,
                recent_intents=[new_intent],
                emotional_trajectory=[new_emotion],
                turn_count=1,
                topic_shifts=0,
                engagement_level=0.7,  # Assume moderate initial engagement
                conversation_depth=0.3  # Start shallow
            )

        # Update existing conversation
        topic_shifted = not any(topic in previous_state.current_topics for topic in new_topics)

        # Calculate engagement (based on message length, specificity)
        engagement = self._calculate_engagement(user_text, previous_state)

        # Calculate depth (increases over time, with topic discussion)
        depth = min(1.0, previous_state.conversation_depth + 0.05)

        return ConversationState(
            current_topics=new_topics,
            recent_intents=(previous_state.recent_intents + [new_intent])[-5:],  # Keep last 5
            emotional_trajectory=(previous_state.emotional_trajectory + [new_emotion])[-10:],
            turn_count=previous_state.turn_count + 1,
            topic_shifts=previous_state.topic_shifts + (1 if topic_shifted else 0),
            engagement_level=engagement,
            conversation_depth=depth
        )

    def _calculate_engagement(self, user_text: str, state: ConversationState) -> float:
        """Calculate user engagement level"""
        engagement = state.engagement_level

        # Long messages = high engagement
        word_count = len(user_text.split())
        if word_count > 30:
            engagement += 0.1
        elif word_count < 5:
            engagement -= 0.1

        # Questions = high engagement
        if '?' in user_text:
            engagement += 0.05

        # Exclamations = high engagement
        if '!' in user_text:
            engagement += 0.05

        # Very short responses = low engagement
        if word_count <= 2 and user_text.lower() in ['ok', 'okay', 'sure', 'fine', 'yeah', 'yes', 'no']:
            engagement -= 0.15

        return max(0.0, min(1.0, engagement))

    def learn_transitions(
        self,
        previous_topic: str,
        new_topic: str,
        previous_intent: str,
        new_intent: str,
        previous_emotion: str,
        new_emotion: str
    ):
        """Learn how topics, intents, and emotions transition"""

        # Learn topic transitions
        if previous_topic and new_topic:
            self.topic_transitions[previous_topic][new_topic] += 1

        # Learn intent transitions
        if previous_intent and new_intent:
            self.intent_transitions[previous_intent][new_intent] += 1

        # Learn emotion transitions
        if previous_emotion and new_emotion:
            self.emotion_transitions[previous_emotion][new_emotion] += 1

    def predict_trajectory(
        self,
        current_state: ConversationState,
        horizon: int = 5
    ) -> TrajectoryPrediction:
        """
        Predict where the conversation is likely to go in the next few turns.
        """

        # Predict next topics
        predicted_topics = self._predict_next_topics(
            current_state.current_topics,
            horizon=horizon
        )

        # Predict next intents
        predicted_intents = self._predict_next_intents(
            current_state.recent_intents,
            horizon=horizon
        )

        # Predict emotional trajectory
        predicted_emotions = self._predict_next_emotions(
            current_state.emotional_trajectory,
            horizon=horizon
        )

        # Predict conversation ending
        ending_prob = self._predict_ending_probability(current_state)

        # Predict duration
        likely_duration = self._predict_remaining_duration(current_state)

        # Generate reasoning
        reasoning = self._generate_trajectory_reasoning(
            current_state,
            predicted_topics,
            predicted_intents,
            ending_prob
        )

        # Overall confidence based on amount of training data
        confidence = self._calculate_prediction_confidence(current_state)

        return TrajectoryPrediction(
            predicted_topics=predicted_topics[:5],
            predicted_intents=predicted_intents[:5],
            predicted_emotions=predicted_emotions[:3],
            likely_duration=likely_duration,
            conversation_ending_probability=ending_prob,
            confidence=confidence,
            reasoning=reasoning
        )

    def _predict_next_topics(self, current_topics: List[str], horizon: int) -> List[Tuple[str, float]]:
        """Predict likely next topics"""
        predictions = defaultdict(float)

        for topic in current_topics:
            if topic in self.topic_transitions:
                transitions = self.topic_transitions[topic]
                total = sum(transitions.values())

                for next_topic, count in transitions.items():
                    probability = count / total
                    predictions[next_topic] += probability

        # Normalize and sort
        if predictions:
            max_prob = max(predictions.values())
            if max_prob > 0:
                predictions = {k: v / max_prob for k, v in predictions.items()}

        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:horizon]

    def _predict_next_intents(self, recent_intents: List[str], horizon: int) -> List[Tuple[str, float]]:
        """Predict likely next user intents"""
        predictions = defaultdict(float)

        # Use most recent intents for prediction
        for intent in recent_intents[-3:]:
            if intent in self.intent_transitions:
                transitions = self.intent_transitions[intent]
                total = sum(transitions.values())

                for next_intent, count in transitions.items():
                    probability = count / total
                    predictions[next_intent] += probability

        # Normalize
        if predictions:
            total_prob = sum(predictions.values())
            if total_prob > 0:
                predictions = {k: v / total_prob for k, v in predictions.items()}

        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:horizon]

    def _predict_next_emotions(self, emotional_trajectory: List[str], horizon: int) -> List[Tuple[str, float]]:
        """Predict emotional trajectory"""
        predictions = defaultdict(float)

        # Use recent emotions
        if emotional_trajectory:
            current_emotion = emotional_trajectory[-1]

            if current_emotion in self.emotion_transitions:
                transitions = self.emotion_transitions[current_emotion]
                total = sum(transitions.values())

                for next_emotion, count in transitions.items():
                    probability = count / total
                    predictions[next_emotion] = probability

        # Default to maintaining current emotion if no data
        if not predictions and emotional_trajectory:
            predictions[emotional_trajectory[-1]] = 0.7

        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_predictions[:horizon]

    def _predict_ending_probability(self, state: ConversationState) -> float:
        """Predict probability that conversation will end soon"""
        ending_prob = 0.0

        # Base rate: longer conversations more likely to end
        if state.turn_count > 20:
            ending_prob += 0.3
        elif state.turn_count > 10:
            ending_prob += 0.15

        # Low engagement = likely ending
        if state.engagement_level < 0.3:
            ending_prob += 0.25
        elif state.engagement_level < 0.5:
            ending_prob += 0.10

        # Multiple topic shifts = restlessness, may end
        if state.topic_shifts > state.turn_count * 0.5:
            ending_prob += 0.15

        # Recent "thank you" intents
        if any(intent in ['gratitude', 'acknowledgment'] for intent in state.recent_intents[-2:]):
            ending_prob += 0.20

        return min(1.0, ending_prob)

    def _predict_remaining_duration(self, state: ConversationState) -> int:
        """Predict how many more turns the conversation will last"""

        # Base prediction on current length and engagement
        if state.engagement_level > 0.7:
            # High engagement = longer conversation
            base = max(5, 15 - state.turn_count)
        elif state.engagement_level > 0.4:
            # Medium engagement
            base = max(3, 10 - state.turn_count)
        else:
            # Low engagement = ending soon
            base = max(1, 5 - state.turn_count)

        # Adjust for topic depth
        if state.conversation_depth > 0.7:
            base = int(base * 1.3)  # Deep conversations last longer

        return max(1, base)

    def _calculate_prediction_confidence(self, state: ConversationState) -> float:
        """Calculate confidence in predictions based on available data"""
        confidence = 0.5  # Base confidence

        # More data = higher confidence
        total_transitions = sum(
            sum(trans.values()) for trans in self.topic_transitions.values()
        )

        if total_transitions > 100:
            confidence += 0.3
        elif total_transitions > 50:
            confidence += 0.2
        elif total_transitions > 20:
            confidence += 0.1

        # Longer current conversation = more context = higher confidence
        if state.turn_count > 5:
            confidence += 0.1

        return min(1.0, confidence)

    def _generate_trajectory_reasoning(
        self,
        state: ConversationState,
        predicted_topics: List[Tuple[str, float]],
        predicted_intents: List[Tuple[str, float]],
        ending_prob: float
    ) -> str:
        """Generate explanation for trajectory prediction"""
        parts = []

        # Talk about engagement
        if state.engagement_level > 0.7:
            parts.append("High engagement suggests continued, deep discussion")
        elif state.engagement_level < 0.4:
            parts.append("Lower engagement may indicate conversation winding down")

        # Talk about predicted topics
        if predicted_topics:
            top_topic = predicted_topics[0]
            parts.append(f"Likely to shift toward '{top_topic[0]}' (p={top_topic[1]:.2f})")

        # Talk about ending
        if ending_prob > 0.5:
            parts.append(f"Conversation likely approaching end (p={ending_prob:.2f})")
        elif ending_prob < 0.2:
            parts.append("Conversation likely to continue for several more turns")

        return ". ".join(parts) if parts else "Insufficient data for detailed reasoning"

    def detect_conversation_patterns(self, min_occurrences: int = 3) -> List[ConversationPattern]:
        """
        Detect recurring patterns in conversation trajectories.
        """
        patterns = []

        # Simple pattern detection: look for common topic sequences
        topic_sequences = defaultdict(int)

        # Extract sequences from transition data
        for topic1, transitions in self.topic_transitions.items():
            for topic2, count in transitions.items():
                if count >= min_occurrences:
                    sequence_key = f"{topic1} -> {topic2}"
                    topic_sequences[sequence_key] = count

        # Create pattern objects
        for sequence, count in topic_sequences.items():
            topics = sequence.split(' -> ')

            pattern = ConversationPattern(
                pattern_id=f"pattern_{len(patterns)}",
                state_sequence=[{'topic': t} for t in topics],
                transition_probabilities={},
                typical_duration=len(topics),
                success_rate=0.8,  # Could be calculated from actual data
                occurrence_count=count,
                last_seen=datetime.now().timestamp()
            )
            patterns.append(pattern)

            # Store pattern
            self.conversation_patterns[pattern.pattern_id] = pattern

        return patterns

    def get_anticipatory_insights(self, state: ConversationState) -> Dict:
        """
        Get insights for anticipatory/proactive responses.
        """
        prediction = self.predict_trajectory(state)

        insights = {
            'should_prepare_for_ending': prediction.conversation_ending_probability > 0.5,
            'likely_next_topics': [t[0] for t in prediction.predicted_topics[:3]],
            'likely_user_needs': [i[0] for i in prediction.predicted_intents[:3]],
            'recommended_strategy': self._recommend_strategy(state, prediction),
            'proactive_suggestions': self._generate_proactive_suggestions(state, prediction)
        }

        return insights

    def _recommend_strategy(self, state: ConversationState, prediction: TrajectoryPrediction) -> str:
        """Recommend response strategy based on prediction"""

        if prediction.conversation_ending_probability > 0.6:
            return "wrap_up_conversation"
        elif state.engagement_level < 0.4:
            return "increase_engagement"
        elif state.conversation_depth < 0.4 and state.turn_count > 3:
            return "deepen_conversation"
        elif state.topic_shifts > state.turn_count * 0.6:
            return "stabilize_topic"
        else:
            return "maintain_flow"

    def _generate_proactive_suggestions(
        self,
        state: ConversationState,
        prediction: TrajectoryPrediction
    ) -> List[str]:
        """Generate proactive suggestions for what to say"""
        suggestions = []

        # Based on predicted topics
        if prediction.predicted_topics:
            top_topic = prediction.predicted_topics[0][0]
            if prediction.predicted_topics[0][1] > 0.6:
                suggestions.append(f"Proactively introduce '{top_topic}' topic")

        # Based on engagement
        if state.engagement_level < 0.5:
            suggestions.append("Ask engaging question to increase involvement")

        # Based on depth
        if state.conversation_depth < 0.4 and state.turn_count > 4:
            suggestions.append("Share deeper insight to increase conversation depth")

        # Based on ending probability
        if prediction.conversation_ending_probability > 0.5:
            suggestions.append("Prepare closing statement or offer to continue later")

        return suggestions

    def save_state(self, filepath: str):
        """Save predictor state"""
        state = {
            'topic_transitions': {
                k: dict(v) for k, v in self.topic_transitions.items()
            },
            'intent_transitions': {
                k: dict(v) for k, v in self.intent_transitions.items()
            },
            'emotion_transitions': {
                k: dict(v) for k, v in self.emotion_transitions.items()
            },
            'patterns': {
                k: v.to_dict() for k, v in self.conversation_patterns.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load predictor state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.topic_transitions = defaultdict(
                lambda: defaultdict(int),
                {k: defaultdict(int, v) for k, v in state.get('topic_transitions', {}).items()}
            )

            self.intent_transitions = defaultdict(
                lambda: defaultdict(int),
                {k: defaultdict(int, v) for k, v in state.get('intent_transitions', {}).items()}
            )

            self.emotion_transitions = defaultdict(
                lambda: defaultdict(int),
                {k: defaultdict(int, v) for k, v in state.get('emotion_transitions', {}).items()}
            )

            # Load patterns
            for pattern_id, pattern_data in state.get('patterns', {}).items():
                pattern = ConversationPattern(**pattern_data)
                self.conversation_patterns[pattern_id] = pattern

        except FileNotFoundError:
            pass

    def get_statistics(self) -> Dict:
        """Get statistics about learned conversation patterns"""
        return {
            'total_topic_transitions': sum(sum(t.values()) for t in self.topic_transitions.values()),
            'total_intent_transitions': sum(sum(t.values()) for t in self.intent_transitions.values()),
            'total_emotion_transitions': sum(sum(t.values()) for t in self.emotion_transitions.values()),
            'detected_patterns': len(self.conversation_patterns),
            'most_common_topic_transitions': sorted(
                [(f"{k1} -> {k2}", v2)
                 for k1, v1 in self.topic_transitions.items()
                 for k2, v2 in v1.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
