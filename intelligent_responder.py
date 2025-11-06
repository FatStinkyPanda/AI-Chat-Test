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
        NOW WITH: Much more variety and naturalness
        """
        parts = []

        # Find the strongest association
        top_assoc = associations[0] if associations else None

        # Vary the opening based on what we learned
        if inferences:
            # Use inferences as the main content
            primary_inference = inferences[0]

            # Make it conversational
            if "interest" in primary_inference.lower():
                parts.append(f"I'm noticing what matters to you. {primary_inference}")
            elif "theme" in primary_inference.lower():
                # Extract the theme and make it natural
                parts.append(primary_inference.replace("This conversation is exploring the theme of",
                                                       "We're diving into"))
            elif "expressed" in primary_inference.lower():
                parts.append("I'm picking up on your interests and starting to understand what draws your attention.")
            else:
                # Use the inference directly
                parts.append(primary_inference)

            # Add a second inference if it's different
            if len(inferences) > 1:
                second_inf = inferences[1]
                if "pattern" in second_inf.lower() or "when" in second_inf.lower():
                    parts.append(second_inf)

        # Optionally add connection to memory (but not always - variety!)
        if top_assoc and random.random() > 0.5:  # Only 50% of the time
            memory_content = top_assoc['memory'].get('content', '')
            assoc_type = top_assoc.get('type', 'direct_topic')

            connection_phrases = [
                f"This builds on what you said about {memory_content[:30]}...",
                f"Earlier you brought up {memory_content[:30]}... - I'm seeing how that connects.",
                f"That relates to your point about {memory_content[:30]}...",
            ]

            if assoc_type == 'learned_association':
                topics = top_assoc.get('related_topics', [])
                if len(topics) >= 2:
                    connection_phrases.append(
                        f"Interesting - you often link {topics[0]} with {topics[1]} in your thinking."
                    )

            # Only add if we haven't already said too much
            if len(parts) < 2:
                parts.append(random.choice(connection_phrases))

        # Varied follow-ups
        follow_ups = [
            "What direction would you like to take this?",
            "Where does this lead in your mind?",
            "What else is connected to this for you?",
            "How are you thinking about this?",
            "What's your perspective on where this connects?",
            "I'm curious what else this brings up for you.",
            "What other thoughts does this spark?",
        ]

        # Only add follow-up if we haven't said enough
        if len(parts) < 2:
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
        NOW WITH: More variety, better inference usage, varied structure
        """
        parts = []

        # Use inferences as primary content
        if inferences and len(inferences) > 0:
            # Get the most relevant inference
            primary_inf = inferences[0]

            # Skip meta-inferences about predictions
            for inf in inferences:
                if 'might want to discuss' not in inf and 'might discuss' not in inf and 'predicted' not in inf:
                    primary_inf = inf
                    break

            # Build response around the inference
            if 'interest' in primary_inf.lower() or 'expressed' in primary_inf.lower():
                openings = [
                    "I'm starting to understand what catches your attention.",
                    "I'm picking up on what draws your interest.",
                    "I'm seeing what resonates with you.",
                ]
                parts.append(random.choice(openings))

            elif 'theme' in primary_inf.lower():
                # Make theme natural
                theme_response = primary_inf.replace("This conversation is exploring the theme of", "We're exploring")
                theme_response = theme_response.replace("This conversation", "We're talking about")
                parts.append(theme_response + ".")

            elif 'feeling' in primary_inf.lower() or 'emotion' in primary_inf.lower():
                parts.append(f"{primary_inf}. I'm tuned into that.")

            else:
                # Use inference directly but make it conversational
                if primary_inf.startswith("The user"):
                    parts.append("I'm learning what matters to you as we talk.")
                elif primary_inf.startswith("When the user mentions"):
                    # Extract the pattern
                    parts.append(f"I notice patterns in how you think about things.")
                else:
                    parts.append(primary_inf)

            # Maybe add a second inference for depth
            if len(inferences) > 1 and len(parts) == 1 and random.random() > 0.6:
                second_inf = inferences[1]
                if 'might want' not in second_inf and len(second_inf) < 80:
                    parts.append(second_inf)

        else:
            # No inferences - use varied engagement
            varied_openings = [
                "I'm with you on this.",
                "I'm following along.",
                "You've given me something to think about.",
                "I'm processing what you're saying.",
                "That's worth considering.",
                "I hear you.",
                "I'm listening closely.",
            ]
            parts.append(random.choice(varied_openings))

        # Varied follow-ups based on tone
        tone = strategy.get('tone', 'friendly')

        if tone == 'enthusiastic':
            follow_ups = [
                "What else?",
                "Tell me more!",
                "I want to hear more about this.",
                "Keep going!",
                "What else is there to this?",
            ]
        elif tone == 'empathetic':
            follow_ups = [
                "I'm here.",
                "What else would help to explore?",
                "How does this sit with you?",
                "What would be useful to discuss?",
            ]
        else:  # friendly/curious
            follow_ups = [
                "What else comes to mind?",
                "Where would you like to go with this?",
                "What direction interests you?",
                "How are you thinking about this?",
                "What's next in your thinking?",
                "What else connects to this?",
                "Where does your mind go from here?",
            ]

        # Only add if response isn't already complete
        if len(" ".join(parts).split()) < 15:
            parts.append(random.choice(follow_ups))

        return " ".join(parts)
