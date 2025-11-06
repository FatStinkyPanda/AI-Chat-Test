"""
Advanced Response Generation System
Natural, intelligent conversation with pattern matching and intent recognition
"""

import re
import random
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum


class IntentType(Enum):
    """Types of user intents"""
    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION_WHAT = "question_what"
    QUESTION_WHO = "question_who"
    QUESTION_WHERE = "question_where"
    QUESTION_WHEN = "question_when"
    QUESTION_WHY = "question_why"
    QUESTION_HOW = "question_how"
    QUESTION_YES_NO = "question_yes_no"
    STATEMENT_FACT = "statement_fact"
    STATEMENT_OPINION = "statement_opinion"
    STATEMENT_FEELING = "statement_feeling"
    REQUEST_ACTION = "request_action"
    ACKNOWLEDGMENT = "acknowledgment"
    CONFUSION = "confusion"
    GRATITUDE = "gratitude"
    UNKNOWN = "unknown"


class ResponseGenerator:
    """
    Advanced response generation with intent recognition and pattern matching
    """

    def __init__(self):
        self._initialize_patterns()
        self._initialize_templates()

    def _initialize_patterns(self):
        """Initialize pattern matching rules for intent detection"""
        self.intent_patterns = {
            IntentType.GREETING: [
                r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
                r'^(hi|hello|hey|yo|sup)[\s!.?]*$',
            ],
            IntentType.FAREWELL: [
                r'\b(bye|goodbye|farewell|see you|talk later|gotta go|gtg)\b',
                r'\b(good night|take care)\b',
            ],
            IntentType.QUESTION_WHAT: [
                r'\bwhat (is|are|was|were|do|does|did|can|could|would|should)',
                r'^what\b',
            ],
            IntentType.QUESTION_WHO: [
                r'\bwho (is|are|was|were|do|does|did|can|could|would|should)',
                r'^who\b',
            ],
            IntentType.QUESTION_WHERE: [
                r'\bwhere (is|are|was|were|do|does|did|can|could|would|should)',
                r'^where\b',
            ],
            IntentType.QUESTION_WHEN: [
                r'\bwhen (is|are|was|were|do|does|did|can|could|would|should)',
                r'^when\b',
            ],
            IntentType.QUESTION_WHY: [
                r'\bwhy (is|are|was|were|do|does|did|can|could|would|should)',
                r'^why\b',
            ],
            IntentType.QUESTION_HOW: [
                r'\bhow (is|are|was|were|do|does|did|can|could|would|should)',
                r'^how\b',
            ],
            IntentType.QUESTION_YES_NO: [
                r'^(do|does|did|is|are|was|were|can|could|would|should|will|shall|have|has|had)\b',
                r'^(am i|are you|is it|can you|could you|would you|will you)\b',
            ],
            IntentType.GRATITUDE: [
                r'\b(thank you|thanks|thank|appreciate|grateful)\b',
                r'^(ty|thx|tyvm)\b',
            ],
            IntentType.REQUEST_ACTION: [
                r'\b(please|can you|could you|would you|will you|help me|show me|tell me|explain)\b',
                r'^(make|create|build|generate|write|do|perform)\b',
            ],
            IntentType.STATEMENT_FEELING: [
                r'\bi (feel|am feeling|felt|am|was)\s+(happy|sad|angry|excited|worried|anxious|nervous|confused)',
                r'\bi (love|hate|like|dislike|enjoy|prefer)',
            ],
            IntentType.ACKNOWLEDGMENT: [
                r'^(ok|okay|alright|sure|fine|yes|yeah|yep|yup|no|nope|nah)[\s!.?]*$',
                r'^(got it|i see|understood|makes sense)[\s!.?]*$',
            ],
            IntentType.CONFUSION: [
                r'\b(confused|don\'t understand|what do you mean|not sure|unclear)\b',
                r'^\?\?+$',
                r'\bhuh\b',
            ],
        }

    def _initialize_templates(self):
        """Initialize diverse response templates for each intent"""
        self.response_templates = {
            IntentType.GREETING: [
                "Hello! How can I help you today?",
                "Hi there! What's on your mind?",
                "Hey! Great to hear from you. What would you like to talk about?",
                "Hello! I'm here to chat. What can I do for you?",
                "Hi! How are you doing?",
            ],
            IntentType.FAREWELL: [
                "Goodbye! Feel free to come back anytime.",
                "See you later! It was nice talking with you.",
                "Take care! Looking forward to our next conversation.",
                "Bye! Have a great day!",
                "Farewell! I'll be here when you need me.",
            ],
            IntentType.QUESTION_WHAT: [
                "That's an interesting question. Let me think about that.",
                "Good question! From what I understand, {context}",
                "Based on what we've discussed, I'd say {context}",
                "Let me consider that carefully. {context}",
                "That's something worth exploring. {context}",
            ],
            IntentType.QUESTION_WHO: [
                "That's a question about people or identity. {context}",
                "Interesting question! {context}",
                "Let me think about who that might be. {context}",
            ],
            IntentType.QUESTION_WHERE: [
                "That's a location-related question. {context}",
                "Regarding location, {context}",
                "Let me think about where that might be. {context}",
            ],
            IntentType.QUESTION_WHEN: [
                "That's a timing question. {context}",
                "Regarding the timeframe, {context}",
                "Let me consider when that might be. {context}",
            ],
            IntentType.QUESTION_WHY: [
                "That's a deep question about reasons. {context}",
                "Why questions are always interesting! {context}",
                "Let me think about the reasoning. {context}",
                "That gets to the heart of the matter. {context}",
            ],
            IntentType.QUESTION_HOW: [
                "That's a procedural question. {context}",
                "Let me explain how that works. {context}",
                "Good question about the process. {context}",
                "Here's how I understand it: {context}",
            ],
            IntentType.QUESTION_YES_NO: [
                "That's a yes/no question. {context}",
                "Let me think about that. {context}",
                "From my understanding, {context}",
            ],
            IntentType.GRATITUDE: [
                "You're welcome! Happy to help.",
                "My pleasure! Let me know if you need anything else.",
                "Glad I could help! Feel free to ask more questions.",
                "No problem at all! That's what I'm here for.",
                "You're very welcome!",
            ],
            IntentType.REQUEST_ACTION: [
                "I'd be happy to help with that. {context}",
                "Let me assist you with that. {context}",
                "Sure, I can help. {context}",
                "I'll do my best to help you with that. {context}",
            ],
            IntentType.STATEMENT_FACT: [
                "That's interesting information.",
                "I understand. Thanks for sharing that.",
                "Noted. That's good to know.",
                "I see. That makes sense.",
                "Interesting! Tell me more.",
            ],
            IntentType.STATEMENT_OPINION: [
                "I appreciate you sharing your perspective.",
                "That's an interesting viewpoint.",
                "I see where you're coming from.",
                "Thanks for sharing your thoughts on that.",
                "That's a valid point.",
            ],
            IntentType.STATEMENT_FEELING: [
                "I hear you. Feelings are important.",
                "Thank you for sharing how you feel.",
                "I understand. Your feelings are valid.",
                "It's good that you're expressing that.",
                "I appreciate you being open about your feelings.",
            ],
            IntentType.ACKNOWLEDGMENT: [
                "Great! What else would you like to discuss?",
                "Understood. Anything else on your mind?",
                "Got it! Is there anything else you'd like to talk about?",
                "Okay! Feel free to continue.",
            ],
            IntentType.CONFUSION: [
                "Let me try to clarify that for you.",
                "I'll explain that more clearly.",
                "No worries! Let me rephrase.",
                "I understand the confusion. Let me help clarify.",
            ],
            IntentType.UNKNOWN: [
                "I'm listening. Could you tell me more?",
                "That's interesting. Can you elaborate?",
                "I'd like to understand better. Could you explain more?",
                "Tell me more about that.",
                "I'm curious to learn more. Please continue.",
            ],
        }

        # Context-aware additions
        self.context_modifiers = {
            'has_memory': [
                "Based on what you've told me before,",
                "Considering our previous conversation,",
                "From what I remember,",
                "Building on what we discussed,",
            ],
            'first_interaction': [
                "I'm just getting to know you, but",
                "As we start our conversation,",
                "From this first impression,",
            ],
            'positive_sentiment': [
                "I'm glad to hear that!",
                "That sounds positive!",
                "That's great!",
            ],
            'negative_sentiment': [
                "I understand that must be difficult.",
                "That sounds challenging.",
                "I hear the concern in that.",
            ],
        }

    def detect_intent(self, text: str) -> Tuple[IntentType, float]:
        """
        Detect the user's intent from their message
        Returns: (intent_type, confidence)
        """
        text_lower = text.lower().strip()

        # Check each intent pattern
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0

        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Calculate confidence based on pattern specificity
                    confidence = 0.8 if len(patterns) == 1 else 0.7
                    if confidence > best_confidence:
                        best_intent = intent_type
                        best_confidence = confidence

        # If no specific pattern matched, try to infer from structure
        if best_confidence == 0.0:
            if '?' in text:
                best_intent = IntentType.UNKNOWN
                best_confidence = 0.3
            elif len(text.split()) > 10:
                best_intent = IntentType.STATEMENT_FACT
                best_confidence = 0.4
            else:
                best_intent = IntentType.UNKNOWN
                best_confidence = 0.2

        return best_intent, best_confidence

    def extract_key_info(self, text: str) -> Dict[str, Any]:
        """
        Extract key information from the text
        """
        info = {
            'has_question': '?' in text,
            'is_short': len(text.split()) < 5,
            'is_long': len(text.split()) > 20,
            'has_personal_info': bool(re.search(r'\b(my|i am|i\'m|i have|i like|i love)\b', text.lower())),
            'has_negation': bool(re.search(r'\b(not|no|never|neither|don\'t|doesn\'t|didn\'t|won\'t|can\'t)\b', text.lower())),
            'keywords': self._extract_keywords(text),
        }
        return info

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
            'it', 'from', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]
        return keywords[:5]  # Top 5 keywords

    def generate_response(self,
                         user_input: str,
                         intent: Optional[IntentType] = None,
                         sentiment: str = "neutral",
                         has_context: bool = False,
                         memory_count: int = 0,
                         relevant_memories: Optional[List[Dict]] = None) -> str:
        """
        Generate a natural, intelligent response

        Args:
            user_input: The user's message
            intent: Detected intent (will auto-detect if None)
            sentiment: Sentiment of the input (positive/negative/neutral)
            has_context: Whether we have conversation context
            memory_count: Number of memories stored
        """
        # Detect intent if not provided
        if intent is None:
            intent, confidence = self.detect_intent(user_input)

        # Extract key information
        info = self.extract_key_info(user_input)

        # Select base template
        templates = self.response_templates.get(intent, self.response_templates[IntentType.UNKNOWN])
        base_response = random.choice(templates)

        # Build context string
        context_parts = []

        # Add specific memory-based responses
        if relevant_memories and len(relevant_memories) > 0:
            memory_content = self._extract_memory_insights(relevant_memories, user_input)
            if memory_content:
                context_parts.append(memory_content)

        # Add memory context
        if memory_count > 5:
            context_parts.append(random.choice([
                "I've been learning from our conversations.",
                "We've talked quite a bit, haven't we?",
                "I'm building up my understanding of our discussions.",
            ]))
        elif memory_count > 0:
            context_parts.append(random.choice([
                "I'm starting to get to know you.",
                "Our conversation is growing.",
            ]))

        # Add sentiment modifier
        if sentiment == "positive" and random.random() > 0.5:
            context_parts.append(random.choice(self.context_modifiers['positive_sentiment']))
        elif sentiment == "negative" and random.random() > 0.5:
            context_parts.append(random.choice(self.context_modifiers['negative_sentiment']))

        # Add intent-specific context
        if intent in [IntentType.QUESTION_WHAT, IntentType.QUESTION_HOW, IntentType.QUESTION_WHY]:
            if info['keywords']:
                context_parts.append(f"Regarding {info['keywords'][0]},")

        # Handle personal information statements
        if info['has_personal_info']:
            context_parts.append(random.choice([
                "Thanks for sharing that with me!",
                "I'll remember that.",
                "That's good to know about you.",
            ]))

        # Build final context
        context = " ".join(context_parts) if context_parts else "let me think about that."

        # Fill template
        if '{context}' in base_response:
            response = base_response.format(context=context)
        else:
            # For templates without context placeholder, append if we have context
            if context_parts and random.random() > 0.3:
                response = f"{base_response} {context}"
            else:
                response = base_response

        return response

    def generate_followup_question(self, user_input: str, keywords: List[str]) -> Optional[str]:
        """
        Generate an intelligent follow-up question
        """
        if not keywords:
            return None

        followup_templates = [
            f"What made you think about {keywords[0]}?",
            f"Can you tell me more about {keywords[0]}?",
            f"How do you feel about {keywords[0]}?",
            f"Is {keywords[0]} something you're interested in?",
            f"What's your experience with {keywords[0]}?",
        ]

        return random.choice(followup_templates)

    def should_ask_followup(self, intent: IntentType, response_count: int) -> bool:
        """
        Determine if we should ask a follow-up question
        """
        # Ask follow-ups for statements, but not too frequently
        if intent in [IntentType.STATEMENT_FACT, IntentType.STATEMENT_OPINION]:
            return response_count % 3 == 0  # Every 3rd response

        # Less frequently for other types
        return response_count % 5 == 0 and random.random() > 0.5

    def _extract_memory_insights(self, memories: List[Dict], current_input: str) -> Optional[str]:
        """
        Extract insights from relevant memories to use in response
        """
        if not memories:
            return None

        current_keywords = set(self._extract_keywords(current_input))

        # Look for memories that connect to current topic
        relevant_content = []
        for memory in memories[:3]:  # Check top 3 most relevant
            content = memory.get('content', '')

            # Skip if it's a response (we want user's previous messages)
            if memory.get('metadata', {}).get('role') == 'assistant':
                continue

            memory_keywords = set(self._extract_keywords(content))
            overlap = current_keywords & memory_keywords

            if overlap:
                relevant_content.append((content, len(overlap)))

        if relevant_content:
            # Sort by keyword overlap
            relevant_content.sort(key=lambda x: x[1], reverse=True)
            best_memory = relevant_content[0][0]

            # Generate context from memory
            responses = [
                f"You mentioned before about {best_memory[:50]}...",
                f"This reminds me of when you said '{best_memory[:50]}...'",
                f"Earlier you talked about {best_memory[:50]}...",
                f"I remember you said something about {best_memory[:50]}...",
            ]
            return random.choice(responses)

        # If no keyword overlap but we have memories, use general reference
        if len(memories) > 0:
            user_memories = [m for m in memories if m.get('metadata', {}).get('role') == 'user']
            if user_memories:
                responses = [
                    "Based on what you've told me,",
                    "From our previous conversations,",
                    "Connecting this with what we discussed before,",
                ]
                return random.choice(responses)

        return None
