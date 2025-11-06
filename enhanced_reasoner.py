"""
Enhanced Reasoning and Learning System
Significantly more powerful at:
- Learning from conversations
- Finding associations and patterns
- Generating inferences from natural language
- Making connections across memories
"""

from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import re
import numpy as np


@dataclass
class ConversationalPattern:
    """A learned pattern from conversations"""
    trigger_topics: List[str]
    related_topics: List[str]
    confidence: float
    examples: List[str]
    frequency: int


@dataclass
class Association:
    """An association between concepts"""
    concept1: str
    concept2: str
    strength: float
    co_occurrence_count: int
    relationship_type: str  # topic, emotional, sequential, etc.


class EnhancedReasoner:
    """
    Enhanced reasoning that works with natural conversational language
    Learns patterns, finds associations, makes smart inferences
    """

    def __init__(self):
        # Learned patterns and associations
        self.topic_associations: Dict[str, List[Association]] = defaultdict(list)
        self.conversational_patterns: List[ConversationalPattern] = []
        self.topic_frequencies: Counter = Counter()
        self.topic_co_occurrences: Dict[Tuple[str, str], int] = defaultdict(int)

        # Conversation understanding
        self.common_topics = self._init_common_topics()
        self.emotional_indicators = self._init_emotional_indicators()
        self.interest_verbs = ['enjoy', 'like', 'love', 'prefer', 'appreciate',
                              'interested in', 'passionate about', 'care about']

    def _init_common_topics(self) -> Dict[str, List[str]]:
        """Initialize common topic categories and related words"""
        return {
            'food': ['food', 'eat', 'taste', 'meal', 'cook', 'delicious', 'flavor',
                    'restaurant', 'dish', 'cuisine', 'hungry', 'tasty'],
            'conversation': ['talk', 'chat', 'discuss', 'conversation', 'speak',
                           'communicate', 'share', 'tell', 'listen'],
            'emotions': ['feel', 'emotion', 'happy', 'sad', 'excited', 'nervous',
                        'love', 'enjoy', 'like', 'dislike'],
            'activities': ['do', 'activity', 'hobby', 'pastime', 'enjoy', 'play',
                          'practice', 'pursue', 'engage'],
            'learning': ['learn', 'study', 'understand', 'know', 'discover',
                        'explore', 'education', 'knowledge'],
            'relationships': ['friend', 'family', 'people', 'social', 'together',
                            'relationship', 'connection', 'bond'],
        }

    def _init_emotional_indicators(self) -> Dict[str, List[str]]:
        """Initialize emotional indicator words"""
        return {
            'positive': ['enjoy', 'love', 'happy', 'great', 'wonderful', 'amazing',
                        'excited', 'glad', 'pleased', 'delighted'],
            'negative': ['hate', 'dislike', 'sad', 'terrible', 'awful', 'bad',
                        'disappointed', 'frustrated', 'angry'],
            'interest': ['interested', 'curious', 'fascinated', 'intrigued',
                        'wonder', 'keen'],
        }

    def learn_from_conversation(self, user_input: str, ai_response: str,
                               memories: List[Dict]):
        """
        Learn patterns and associations from the conversation
        This makes the AI smarter over time
        """
        # Extract topics from input
        topics = self._extract_topics_detailed(user_input)

        # Update topic frequencies
        for topic in topics:
            self.topic_frequencies[topic] += 1

        # Learn topic co-occurrences
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                pair = tuple(sorted([topic1, topic2]))
                self.topic_co_occurrences[pair] += 1

        # Learn associations with memory
        self._learn_associations_from_memory(topics, memories)

        # Learn conversational patterns
        self._learn_conversational_pattern(user_input, topics)

    def find_intelligent_associations(self, current_input: str,
                                     memories: List[Dict]) -> List[Dict]:
        """
        Find intelligent associations between current input and memories
        Much smarter than simple keyword matching
        """
        associations_found = []

        # Extract topics from current input
        current_topics = self._extract_topics_detailed(current_input)

        # Find directly related memories
        for memory in memories:
            memory_content = memory.get('content', '')
            memory_topics = self._extract_topics_detailed(memory_content)

            # Calculate topic overlap
            overlap = set(current_topics) & set(memory_topics)

            if overlap:
                associations_found.append({
                    'memory': memory,
                    'shared_topics': list(overlap),
                    'association_strength': len(overlap) / max(len(current_topics), 1),
                    'type': 'direct_topic'
                })

        # Find semantically related memories (even without exact topic match)
        for memory in memories:
            memory_content = memory.get('content', '')
            memory_topics = self._extract_topics_detailed(memory_content)

            # Check for learned associations
            for current_topic in current_topics:
                for memory_topic in memory_topics:
                    pair = tuple(sorted([current_topic, memory_topic]))
                    if pair in self.topic_co_occurrences:
                        associations_found.append({
                            'memory': memory,
                            'related_topics': [current_topic, memory_topic],
                            'association_strength': min(1.0, self.topic_co_occurrences[pair] * 0.2),
                            'type': 'learned_association'
                        })

        # Find thematic connections (same category)
        for memory in memories:
            memory_content = memory.get('content', '')
            connection_strength = self._find_thematic_connection(
                current_input, memory_content
            )

            if connection_strength > 0.3:
                associations_found.append({
                    'memory': memory,
                    'connection_type': 'thematic',
                    'association_strength': connection_strength,
                    'type': 'thematic'
                })

        # Remove duplicates and sort by strength
        unique_associations = {}
        for assoc in associations_found:
            mem_id = assoc['memory'].get('id')
            if mem_id not in unique_associations or \
               assoc['association_strength'] > unique_associations[mem_id]['association_strength']:
                unique_associations[mem_id] = assoc

        sorted_associations = sorted(
            unique_associations.values(),
            key=lambda x: x['association_strength'],
            reverse=True
        )

        return sorted_associations[:5]

    def generate_conversational_inferences(self, user_input: str,
                                          memories: List[Dict],
                                          associations: List[Dict]) -> List[str]:
        """
        Generate intelligent inferences from natural conversation
        Understands what the user is actually communicating
        """
        inferences = []

        # Detect what the user is expressing
        user_intent = self._detect_user_intent(user_input)
        topics = self._extract_topics_detailed(user_input)

        # Inference 1: What the user cares about
        if user_intent.get('expressing_interest'):
            interests = user_intent['interest_objects']
            for interest in interests:
                inferences.append(f"The user has expressed interest in {interest}")

        # Inference 2: Patterns in conversation
        if len(topics) > 1:
            # Multiple topics = exploring a theme
            common_theme = self._find_common_theme(topics)
            if common_theme:
                inferences.append(f"This conversation is exploring the theme of {common_theme}")

        # Inference 3: Connections to past conversations
        if associations:
            for assoc in associations[:2]:
                if assoc['type'] == 'learned_association':
                    topics_mentioned = assoc.get('related_topics', [])
                    if len(topics_mentioned) == 2:
                        inferences.append(
                            f"When the user mentions {topics_mentioned[0]}, "
                            f"they often talk about {topics_mentioned[1]}"
                        )

        # Inference 4: Emotional state
        emotional_tone = self._detect_emotional_tone(user_input)
        if emotional_tone and emotional_tone != 'neutral':
            inferences.append(f"The user seems to be feeling {emotional_tone}")

        # Inference 5: What user might want to discuss
        if self.conversational_patterns:
            predicted_topics = self._predict_next_topics(topics)
            if predicted_topics:
                inferences.append(
                    f"Based on patterns, the user might want to discuss: {', '.join(predicted_topics[:2])}"
                )

        # Inference 6: Topic progression
        if len(memories) > 2:
            topic_evolution = self._analyze_topic_evolution(memories, topics)
            if topic_evolution:
                inferences.append(topic_evolution)

        return inferences

    def decide_response_strategy(self, user_input: str, inferences: List[str],
                                associations: List[Dict], confidence: float) -> Dict:
        """
        Decide HOW to respond based on understanding
        More intelligent response planning
        """
        strategy = {
            'approach': 'engage',
            'should_ask_question': False,
            'should_share_understanding': False,
            'should_make_connection': False,
            'suggested_topics': [],
            'tone': 'friendly'
        }

        user_intent = self._detect_user_intent(user_input)

        # If user is sharing something they enjoy/like
        if user_intent.get('expressing_interest'):
            strategy['approach'] = 'show_interest'
            strategy['should_ask_question'] = True
            strategy['suggested_topics'] = user_intent.get('interest_objects', [])

        # If we have strong associations
        if associations and any(a['association_strength'] > 0.6 for a in associations):
            strategy['should_make_connection'] = True
            strategy['approach'] = 'connect_memories'

        # If we have inferences
        if len(inferences) > 2:
            strategy['should_share_understanding'] = True

        # If user is asking a question
        if '?' in user_input:
            strategy['approach'] = 'answer_thoughtfully'
            if confidence < 0.5:
                strategy['should_ask_question'] = True  # Ask for clarification

        # Detect emotional content
        emotional_tone = self._detect_emotional_tone(user_input)
        if emotional_tone in ['positive', 'excited']:
            strategy['tone'] = 'enthusiastic'
        elif emotional_tone in ['negative', 'sad']:
            strategy['tone'] = 'empathetic'

        return strategy

    def _extract_topics_detailed(self, text: str) -> List[str]:
        """Extract topics with better understanding of natural language"""
        topics = []
        text_lower = text.lower()

        # Extract nouns and important words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)

        # Remove common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'was',
            'it', 'from', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
            'also', 'well', 'just', 'about', 'very', 'what', 'when', 'where'
        }

        for word in words:
            if word not in stop_words:
                topics.append(word)

        # Add compound topics (phrases)
        for category, keywords in self.common_topics.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics.append(category)
                    break

        return list(dict.fromkeys(topics))  # Remove duplicates, preserve order

    def _detect_user_intent(self, user_input: str) -> Dict:
        """Detect what the user is trying to communicate"""
        intent = {
            'expressing_interest': False,
            'asking_question': False,
            'sharing_information': False,
            'expressing_emotion': False,
            'interest_objects': []
        }

        text_lower = user_input.lower()

        # Detect interest expression
        for verb in self.interest_verbs:
            if verb in text_lower:
                intent['expressing_interest'] = True
                # Extract what they're interested in
                pattern = f"{verb}\\s+([\\w\\s]+?)(?:\\.|$|,|!)"
                match = re.search(pattern, text_lower)
                if match:
                    interest_object = match.group(1).strip()
                    intent['interest_objects'].append(interest_object)

        # Detect questions
        if '?' in user_input or any(word in text_lower for word in ['what', 'why', 'how', 'when', 'where', 'who']):
            intent['asking_question'] = True

        # Detect information sharing
        if any(word in text_lower for word in ['i think', 'i believe', 'actually', 'fact']):
            intent['sharing_information'] = True

        # Detect emotion expression
        for emotion_type, words in self.emotional_indicators.items():
            if any(word in text_lower for word in words):
                intent['expressing_emotion'] = True
                intent['emotion_type'] = emotion_type
                break

        return intent

    def _detect_emotional_tone(self, text: str) -> str:
        """Detect emotional tone of text"""
        text_lower = text.lower()

        for emotion_type, words in self.emotional_indicators.items():
            for word in words:
                if word in text_lower:
                    return emotion_type

        return 'neutral'

    def _find_thematic_connection(self, text1: str, text2: str) -> float:
        """Find thematic connection strength"""
        topics1 = set(self._extract_topics_detailed(text1))
        topics2 = set(self._extract_topics_detailed(text2))

        # Direct overlap
        overlap = topics1 & topics2
        if overlap:
            return len(overlap) / max(len(topics1), len(topics2), 1)

        # Check if topics belong to same category
        for category, keywords in self.common_topics.items():
            has_topic1 = any(t in keywords or category in topics1 for t in topics1)
            has_topic2 = any(t in keywords or category in topics2 for t in topics2)
            if has_topic1 and has_topic2:
                return 0.4  # Moderate connection via category

        return 0.0

    def _find_common_theme(self, topics: List[str]) -> Optional[str]:
        """Find common theme among topics"""
        for category, keywords in self.common_topics.items():
            matches = sum(1 for topic in topics if topic in keywords or topic == category)
            if matches >= 2:
                return category
        return None

    def _predict_next_topics(self, current_topics: List[str]) -> List[str]:
        """Predict what topics user might discuss next"""
        predicted = []

        for topic in current_topics:
            # Find topics that co-occur with this one
            related = []
            for pair, count in self.topic_co_occurrences.items():
                if topic in pair and count > 1:
                    other_topic = pair[0] if pair[1] == topic else pair[1]
                    related.append((other_topic, count))

            # Sort by frequency
            related.sort(key=lambda x: x[1], reverse=True)
            predicted.extend([t[0] for t in related[:2]])

        return list(dict.fromkeys(predicted))[:3]  # Top 3 unique predictions

    def _analyze_topic_evolution(self, memories: List[Dict],
                                 current_topics: List[str]) -> Optional[str]:
        """Analyze how topics have evolved in conversation"""
        if not memories:
            return None

        # Get topics from recent memories
        recent_topics = []
        for memory in memories[-3:]:
            content = memory.get('content', '')
            topics = self._extract_topics_detailed(content)
            recent_topics.extend(topics)

        # Find new topics
        new_topics = set(current_topics) - set(recent_topics)

        if new_topics:
            return f"The conversation is expanding to new topics: {', '.join(list(new_topics)[:2])}"

        # Find persisting topics
        persistent = set(current_topics) & set(recent_topics)
        if persistent:
            return f"Continuing to discuss: {', '.join(list(persistent)[:2])}"

        return None

    def _learn_associations_from_memory(self, topics: List[str],
                                       memories: List[Dict]):
        """Learn associations between topics from memory"""
        for memory in memories[-5:]:  # Last 5 memories
            memory_content = memory.get('content', '')
            memory_topics = self._extract_topics_detailed(memory_content)

            # Create associations between current and memory topics
            for current_topic in topics:
                for memory_topic in memory_topics:
                    if current_topic != memory_topic:
                        # Check if association exists
                        existing = None
                        for assoc in self.topic_associations[current_topic]:
                            if assoc.concept2 == memory_topic:
                                existing = assoc
                                break

                        if existing:
                            existing.co_occurrence_count += 1
                            existing.strength = min(1.0, existing.co_occurrence_count * 0.15)
                        else:
                            new_assoc = Association(
                                concept1=current_topic,
                                concept2=memory_topic,
                                strength=0.3,
                                co_occurrence_count=1,
                                relationship_type='conversational'
                            )
                            self.topic_associations[current_topic].append(new_assoc)

    def _learn_conversational_pattern(self, user_input: str, topics: List[str]):
        """Learn conversational patterns over time"""
        if not topics:
            return

        # Find if we have a similar pattern
        for pattern in self.conversational_patterns:
            overlap = set(topics) & set(pattern.trigger_topics)
            if len(overlap) >= 1:
                # Update existing pattern
                pattern.frequency += 1
                pattern.confidence = min(1.0, pattern.frequency * 0.1)
                pattern.examples.append(user_input[:50])
                # Add new related topics
                for topic in topics:
                    if topic not in pattern.related_topics:
                        pattern.related_topics.append(topic)
                return

        # Create new pattern
        new_pattern = ConversationalPattern(
            trigger_topics=topics[:3],
            related_topics=topics,
            confidence=0.3,
            examples=[user_input[:50]],
            frequency=1
        )
        self.conversational_patterns.append(new_pattern)
