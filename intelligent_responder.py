"""
Intelligent Response Generator
Generates thoughtful, varied, intelligent responses
Uses understanding from deliberation, inferences, and associations
"""

import random
from typing import List, Dict, Optional


class IntelligentResponder:
    """
    Generates intelligent responses based on deep understanding
    No more generic responses - truly thinks about what to say
    """

    def __init__(self):
        pass

    def generate_response(self, user_input: str, understanding: Dict,
                         inferences: List[str], associations: List[Dict],
                         strategy: Dict, memories: List[Dict]) -> str:
        """
        Generate intelligent response based on complete understanding
        The AI decides what to say based on what it learned
        """

        approach = strategy.get('approach', 'engage')

        # Route to appropriate response builder
        if approach == 'show_interest':
            return self._respond_to_interest(user_input, understanding, strategy, inferences)

        elif approach == 'connect_memories':
            return self._respond_with_connections(user_input, associations, inferences)

        elif approach == 'answer_thoughtfully':
            return self._respond_to_question(user_input, inferences, strategy)

        elif approach == 'engage':
            return self._engage_naturally(user_input, inferences, associations, strategy)

        else:
            # Fallback to intelligent engagement
            return self._engage_naturally(user_input, inferences, associations, strategy)

    def _respond_to_interest(self, user_input: str, understanding: Dict,
                            strategy: Dict, inferences: List[str]) -> str:
        """
        Respond when user expresses interest in something
        Show genuine curiosity and engagement
        """
        interest_objects = strategy.get('suggested_topics', [])

        if not interest_objects:
            return self._engage_naturally(user_input, inferences, [], strategy)

        main_interest = interest_objects[0]

        # Build thoughtful response
        parts = []

        # Acknowledge the interest
        acknowledgments = [
            f"I find it interesting that you enjoy {main_interest}.",
            f"It's great that you appreciate {main_interest}.",
            f"That's wonderful that you're into {main_interest}.",
        ]
        parts.append(random.choice(acknowledgments))

        # If we have inferences, show understanding
        if inferences:
            understanding_shown = False
            for inference in inferences:
                if 'interest' in inference or 'expressed' in inference:
                    parts.append(f"I'm noticing your interests and learning about what matters to you.")
                    understanding_shown = True
                    break

        # Ask a thoughtful question
        questions = [
            f"What is it about {main_interest} that appeals to you most?",
            f"How did you first discover your interest in {main_interest}?",
            f"What aspects of {main_interest} do you find most engaging?",
            f"Is there something specific about {main_interest} you'd like to explore?",
        ]
        parts.append(random.choice(questions))

        return " ".join(parts)

    def _respond_with_connections(self, user_input: str, associations: List[Dict],
                                 inferences: List[str]) -> str:
        """
        Respond by making intelligent connections to past conversations
        """
        parts = []

        # Find the strongest association
        top_assoc = associations[0] if associations else None

        if top_assoc:
            memory_content = top_assoc['memory'].get('content', '')
            assoc_type = top_assoc.get('type', 'direct_topic')

            # Different connection phrasings based on type
            if assoc_type == 'learned_association':
                topics = top_assoc.get('related_topics', [])
                if len(topics) >= 2:
                    parts.append(f"This connects to something interesting - when you mentioned {topics[0]} before, "
                               f"you also brought up {topics[1]}.")

            elif assoc_type == 'thematic':
                parts.append(f"This reminds me of our earlier conversation.")

            else:  # direct_topic
                shared = top_assoc.get('shared_topics', [])
                if shared:
                    parts.append(f"This relates to when you mentioned {memory_content[:40]}...")

        # Add inference-based understanding
        if inferences:
            # Find pattern or theme inferences
            for inference in inferences:
                if 'theme' in inference.lower() or 'pattern' in inference.lower():
                    parts.append(inference)
                    break

        # Thoughtful follow-up
        follow_ups = [
            "I'm connecting these ideas. What's your thinking on this?",
            "I'm seeing patterns in what you're sharing. Want to explore this further?",
            "These connections are interesting. What else comes to mind?",
        ]
        parts.append(random.choice(follow_ups))

        return " ".join(parts)

    def _respond_to_question(self, user_input: str, inferences: List[str],
                            strategy: Dict) -> str:
        """
        Respond thoughtfully to questions
        Honest about limitations, thoughtful about what we know
        """
        parts = []

        # Acknowledge the question
        acknowledgments = [
            "That's a thought-provoking question.",
            "Let me think about that.",
            "That's something worth considering.",
        ]
        parts.append(random.choice(acknowledgments))

        # If we have understanding from inferences
        if inferences:
            # Share relevant insights
            relevant_inferences = [inf for inf in inferences if len(inf) > 10]
            if relevant_inferences:
                parts.append(f"From what I understand, {relevant_inferences[0].lower()}")

        # Be honest about limitations
        if strategy.get('should_ask_question'):
            clarifications = [
                "To give you a more complete answer, could you tell me more about what specifically interests you?",
                "I want to make sure I understand correctly - what aspect would you like me to focus on?",
                "Could you help me understand what you're most curious about regarding this?",
            ]
            parts.append(random.choice(clarifications))
        else:
            # Encourage further exploration
            parts.append("What's your own perspective on this?")

        return " ".join(parts)

    def _engage_naturally(self, user_input: str, inferences: List[str],
                         associations: List[Dict], strategy: Dict) -> str:
        """
        Natural engagement based on understanding
        Varied, thoughtful, contextual responses
        """
        parts = []

        # Use different opening based on inferences
        if inferences:
            # If we understand something specific
            if any('interest' in inf or 'feeling' in inf for inf in inferences):
                openings = [
                    "I'm following what you're sharing.",
                    "I'm taking in what you're saying.",
                    "I'm thinking about what you've told me.",
                ]
            elif any('theme' in inf or 'topic' in inf for inf in inferences):
                openings = [
                    "I'm noticing themes in our conversation.",
                    "There's interesting depth to what you're saying.",
                    "I'm seeing connections in what you're sharing.",
                ]
            else:
                openings = [
                    "That's interesting.",
                    "I'm considering what you said.",
                    "Let me think about that.",
                ]

            parts.append(random.choice(openings))

            # Share an inference if relevant
            relevant_inf = None
            for inf in inferences:
                # Skip meta-inferences about predictions
                if 'might want to discuss' not in inf and 'might discuss' not in inf:
                    relevant_inf = inf
                    break

            if relevant_inf:
                # Make it conversational
                if relevant_inf.startswith("The user"):
                    # Rephrase to be more natural
                    parts.append("I'm learning about what matters to you.")
                elif relevant_inf.startswith("This conversation"):
                    parts.append(relevant_inf.replace("This conversation is", "We're"))
                else:
                    parts.append(relevant_inf)

        else:
            # No inferences - still engage thoughtfully
            openings = [
                "I'm here and listening.",
                "I appreciate you sharing that.",
                "I'm taking that in.",
            ]
            parts.append(random.choice(openings))

        # If we have associations, reference them
        if associations and random.random() > 0.5:  # 50% chance to mention connection
            parts.append("This connects to things you've mentioned before.")

        # Thoughtful follow-up based on strategy
        tone = strategy.get('tone', 'friendly')

        if tone == 'enthusiastic':
            follow_ups = [
                "Tell me more!",
                "What else about this?",
                "I'd love to hear more.",
            ]
        elif tone == 'empathetic':
            follow_ups = [
                "I'm here to listen.",
                "What would help to talk about?",
                "How can we explore this together?",
            ]
        else:  # friendly
            follow_ups = [
                "What else is on your mind?",
                "Where shall we go with this?",
                "What would you like to explore?",
                "What else comes up for you?",
            ]

        parts.append(random.choice(follow_ups))

        return " ".join(parts)
