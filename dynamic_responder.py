"""
Dynamic Response System
Constructs responses from understanding rather than templates
Gives the AI freedom to think for itself
"""

import random
from typing import List, Dict, Optional, Any
from response_generator import IntentType, ResponseGenerator


class DynamicResponder:
    """
    Generates responses by understanding context and constructing thoughts
    rather than using fixed templates
    """

    def __init__(self):
        self.intent_detector = ResponseGenerator()  # Use for intent detection only

    def construct_response(self,
                          user_input: str,
                          relevant_memories: List[Dict],
                          input_emotions: Dict,
                          reasoning_paths: List,
                          turn_count: int) -> str:
        """
        Dynamically construct a response by analyzing and understanding the input
        """
        # Understand what the user is communicating
        intent, confidence = self.intent_detector.detect_intent(user_input)
        key_info = self.intent_detector.extract_key_info(user_input)

        # Build understanding from multiple sources
        understanding = self._build_understanding(
            user_input=user_input,
            intent=intent,
            key_info=key_info,
            memories=relevant_memories,
            emotions=input_emotions,
            reasoning=reasoning_paths
        )

        # Construct response from understanding
        response = self._construct_from_understanding(
            understanding=understanding,
            intent=intent,
            turn_count=turn_count
        )

        return response

    def _build_understanding(self,
                           user_input: str,
                           intent: IntentType,
                           key_info: Dict,
                           memories: List[Dict],
                           emotions: Dict,
                           reasoning: List) -> Dict[str, Any]:
        """
        Build a comprehensive understanding of the situation
        """
        understanding = {
            'user_said': user_input,
            'intent': intent,
            'is_question': intent.name.startswith('QUESTION_'),
            'is_greeting': intent == IntentType.GREETING,
            'is_farewell': intent == IntentType.FAREWELL,
            'question_type': intent.name.split('_')[1] if intent.name.startswith('QUESTION_') else None,
            'keywords': key_info['keywords'],
            'has_personal_info': key_info['has_personal_info'],
            'is_emotional': any(v > 0.3 for v in emotions.values()) if emotions else False,
            'dominant_emotion': max(emotions.items(), key=lambda x: x[1])[0].value if emotions else 'neutral',
            'has_memories': len(memories) > 0,
            'memory_connections': [],
            'reasoning_available': len(reasoning) > 0,
        }

        # Analyze memory connections
        if memories:
            for memory in memories[:3]:
                if memory.get('metadata', {}).get('role') == 'user':
                    understanding['memory_connections'].append({
                        'content': memory.get('content', ''),
                        'relevance': memory.get('relevance', 0),
                    })

        return understanding

    def _construct_from_understanding(self,
                                    understanding: Dict,
                                    intent: IntentType,
                                    turn_count: int) -> str:
        """
        Construct response by thinking through the understanding
        """
        # Start building response components
        components = []

        # Handle greetings naturally
        if understanding['is_greeting']:
            return self._respond_to_greeting(understanding, turn_count)

        # Handle farewells naturally
        if understanding['is_farewell']:
            return self._respond_to_farewell(understanding)

        # Handle questions by thinking through them
        if understanding['is_question']:
            return self._respond_to_question(understanding)

        # Handle emotional content with empathy
        if understanding['is_emotional']:
            return self._respond_to_emotion(understanding)

        # Handle personal information
        if understanding['has_personal_info']:
            return self._respond_to_personal_info(understanding)

        # Handle statements with memory integration
        if understanding['has_memories']:
            return self._respond_with_memory(understanding)

        # Default: engage with interest
        return self._respond_with_engagement(understanding)

    def _respond_to_greeting(self, understanding: Dict, turn_count: int) -> str:
        """Respond to greetings naturally"""
        greeting_word = random.choice(['Hello', 'Hi', 'Hey', 'Greetings'])

        if turn_count == 1:
            return f"{greeting_word}! I'm here to chat and learn. What's on your mind?"
        elif turn_count < 5:
            return f"{greeting_word}! How are things going? What would you like to talk about?"
        else:
            if understanding['has_memories']:
                return f"{greeting_word} again! Good to continue our conversation. What would you like to discuss?"
            return f"{greeting_word}! What brings you here today?"

    def _respond_to_farewell(self, understanding: Dict) -> str:
        """Respond to farewells naturally"""
        farewell_word = random.choice(['Goodbye', 'See you', 'Take care', 'Bye', 'Until next time'])

        if understanding['has_memories']:
            memory_note = random.choice([
                "I'll remember our conversation.",
                "Looking forward to talking again.",
                "I've enjoyed learning from you.",
            ])
            return f"{farewell_word}! {memory_note}"

        return f"{farewell_word}! Feel free to come back anytime."

    def _respond_to_question(self, understanding: Dict) -> str:
        """Respond to questions by thinking through them"""
        question_type = understanding['question_type']
        keywords = understanding['keywords']

        # Acknowledge the question
        acknowledgment = random.choice([
            "That's an interesting question.",
            "Let me think about that.",
            "Good question.",
            "I appreciate you asking that.",
        ])

        # Try to use memory if relevant
        if understanding['memory_connections']:
            for connection in understanding['memory_connections']:
                if any(kw in connection['content'].lower() for kw in keywords):
                    return f"{acknowledgment} {self._reference_memory(connection['content'])} Is there something specific you'd like to know?"

        # Respond based on question type
        if question_type == 'WHAT':
            if keywords:
                return f"{acknowledgment} You're asking about {keywords[0]}. {self._express_limitation()} What specifically would help you understand it better?"
            return f"{acknowledgment} {self._express_limitation()} Could you provide more context?"

        elif question_type == 'WHY':
            return f"{acknowledgment} The 'why' behind things is always fascinating. {self._express_limitation()} What's your own thinking on this?"

        elif question_type == 'HOW':
            if keywords:
                return f"{acknowledgment} You want to understand how {keywords[0]} works. {self._express_limitation()} Would it help if I ask what you already know about it?"
            return f"{acknowledgment} {self._express_limitation()} Can you tell me more about what you're trying to understand?"

        elif question_type == 'WHO':
            return f"{acknowledgment} You're asking about someone. {self._express_limitation()} What would you like to know about them?"

        elif question_type == 'WHERE' or question_type == 'WHEN':
            return f"{acknowledgment} That's about location or timing. {self._express_limitation()} What context can you share?"

        else:
            return f"{acknowledgment} {self._express_thoughtfulness()} What would be most helpful for you to know?"

    def _respond_to_emotion(self, understanding: Dict) -> str:
        """Respond with emotional intelligence"""
        emotion = understanding['dominant_emotion']

        # Emotional validation
        if emotion in ['joy', 'surprise']:
            validation = random.choice([
                "That sounds wonderful!",
                "I'm glad to hear that positivity!",
                "That's great!",
            ])
        elif emotion in ['sadness', 'fear', 'anger']:
            validation = random.choice([
                "I hear that this is difficult.",
                "That sounds challenging.",
                "I understand that can be tough.",
            ])
        else:
            validation = "I hear you."

        # Add empathy
        empathy = random.choice([
            "Your feelings are valid.",
            "It's important to express how you feel.",
            "Thank you for sharing that with me.",
        ])

        # Offer support
        support = random.choice([
            "Would you like to talk more about it?",
            "Is there something specific on your mind about this?",
            "What's been going through your mind?",
        ])

        return f"{validation} {empathy} {support}"

    def _respond_to_personal_info(self, understanding: Dict) -> str:
        """Respond when user shares personal information"""
        keywords = understanding['keywords']

        # Acknowledge sharing
        acknowledgment = random.choice([
            "Thank you for sharing that with me.",
            "I appreciate you telling me that.",
            "That's good to know about you.",
        ])

        # Express interest
        if keywords:
            interest = f"So {keywords[0]} is part of your life."
        else:
            interest = "I'm learning more about you."

        # Encourage continuation
        encouragement = random.choice([
            "Tell me more if you'd like.",
            "What else would you like to share?",
            "I'm curious to hear more.",
        ])

        return f"{acknowledgment} {interest} {encouragement}"

    def _respond_with_memory(self, understanding: Dict) -> str:
        """Respond by connecting to memories"""
        keywords = understanding['keywords']

        # Find relevant memory connection
        best_connection = None
        for connection in understanding['memory_connections']:
            if any(kw in connection['content'].lower() for kw in keywords):
                best_connection = connection
                break

        if best_connection:
            # Make explicit connection
            memory_ref = self._reference_memory(best_connection['content'])
            current_thought = self._process_current_input(understanding['user_said'])
            return f"{memory_ref} {current_thought} How do these connect in your mind?"

        # General memory reference
        return f"This relates to things you've mentioned before. {self._process_current_input(understanding['user_said'])} What's your current thinking?"

    def _respond_with_engagement(self, understanding: Dict) -> str:
        """Engage with genuine interest"""
        keywords = understanding['keywords']

        # Show interest
        interest = random.choice([
            "That's interesting.",
            "I'm thinking about what you said.",
            "You've given me something to consider.",
        ])

        # Ask for elaboration
        if keywords:
            elaboration = f"When you mention {keywords[0]}, what comes to mind?"
        else:
            elaboration = random.choice([
                "Can you tell me more about your thoughts on this?",
                "What else would you like to explore about this?",
                "What made you bring this up?",
            ])

        return f"{interest} {elaboration}"

    def _reference_memory(self, memory_content: str) -> str:
        """Reference a memory naturally"""
        templates = [
            f"You mentioned earlier: \"{memory_content[:60]}...\"",
            f"I remember you said \"{memory_content[:60]}...\"",
            f"Earlier you told me \"{memory_content[:60]}...\"",
            f"This connects to when you said \"{memory_content[:60]}...\"",
        ]
        return random.choice(templates)

    def _process_current_input(self, input_text: str) -> str:
        """Process the current input and show understanding"""
        if len(input_text) > 100:
            return "You're sharing quite a bit of detail, which helps me understand."
        elif len(input_text.split()) < 5:
            return "That's concise."
        else:
            return "I'm following along."

    def _express_limitation(self) -> str:
        """Honestly express limitations"""
        return random.choice([
            "While I'm still learning,",
            "I'm working with what I know, and",
            "Let me be honest -",
            "I want to help, though",
        ])

    def _express_thoughtfulness(self) -> str:
        """Express genuine thought process"""
        return random.choice([
            "I'm thinking through this.",
            "Let me process what you're asking.",
            "I'm considering different angles on this.",
            "That requires some thought.",
        ])
