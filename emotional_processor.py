"""
Emotional Processing System
Models emotional understanding and affective associations
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from brain_core import EmotionalValence


class EmotionalProcessor:
    """
    Processes emotional content and maintains affective associations
    """

    def __init__(self):
        # Emotion lexicons (simplified - can be expanded)
        self.emotion_lexicon = {
            EmotionalValence.JOY: [
                'happy', 'joy', 'delighted', 'pleased', 'glad', 'cheerful', 'excited',
                'wonderful', 'amazing', 'great', 'love', 'excellent', 'fantastic',
                'brilliant', 'awesome', 'beautiful', 'good', 'nice', 'fun', 'laugh'
            ],
            EmotionalValence.SADNESS: [
                'sad', 'unhappy', 'depressed', 'miserable', 'sorrowful', 'upset',
                'disappointed', 'hurt', 'lonely', 'sorry', 'unfortunate', 'bad',
                'terrible', 'awful', 'cry', 'tears', 'grief', 'loss'
            ],
            EmotionalValence.ANGER: [
                'angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated',
                'hate', 'rage', 'outraged', 'hostile', 'aggressive', 'violent'
            ],
            EmotionalValence.FEAR: [
                'afraid', 'scared', 'fearful', 'anxious', 'worried', 'nervous',
                'terrified', 'panic', 'frightened', 'alarmed', 'concerned', 'stress'
            ],
            EmotionalValence.SURPRISE: [
                'surprised', 'amazed', 'astonished', 'shocked', 'stunned',
                'unexpected', 'sudden', 'wow', 'oh'
            ],
            EmotionalValence.TRUST: [
                'trust', 'believe', 'faith', 'reliable', 'honest', 'sincere',
                'loyal', 'confident', 'sure', 'certain'
            ],
            EmotionalValence.ANTICIPATION: [
                'hope', 'expect', 'anticipate', 'look forward', 'await', 'eager',
                'ready', 'prepared', 'planning', 'will', 'going to'
            ]
        }

        # Build reverse index
        self.word_to_emotions: Dict[str, List[EmotionalValence]] = defaultdict(list)
        for emotion, words in self.emotion_lexicon.items():
            for word in words:
                self.word_to_emotions[word].append(emotion)

        # Emotion intensifiers
        self.intensifiers = {
            'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'incredibly': 2.0,
            'absolutely': 1.8, 'totally': 1.5, 'completely': 1.7,
            'somewhat': 0.5, 'slightly': 0.4, 'a bit': 0.5, 'kind of': 0.6
        }

        # Negation words
        self.negations = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', "n't"}

    def analyze_emotion(self, text: str) -> Dict[EmotionalValence, float]:
        """
        Analyze emotional content of text
        Returns emotional valence scores (0.0 to 1.0)
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        emotion_scores = defaultdict(float)
        total_emotion_words = 0

        for i, word in enumerate(words):
            if word in self.word_to_emotions:
                # Check for intensifiers
                intensity = 1.0
                if i > 0 and words[i-1] in self.intensifiers:
                    intensity = self.intensifiers[words[i-1]]

                # Check for negation
                negated = False
                if i > 0 and words[i-1] in self.negations:
                    negated = True
                elif i > 1 and words[i-2] in self.negations:
                    negated = True

                # Add emotion scores
                for emotion in self.word_to_emotions[word]:
                    if negated:
                        # Negation flips or neutralizes emotion
                        if emotion == EmotionalValence.JOY:
                            emotion_scores[EmotionalValence.SADNESS] += 0.5 * intensity
                        elif emotion == EmotionalValence.SADNESS:
                            emotion_scores[EmotionalValence.JOY] += 0.3 * intensity
                    else:
                        emotion_scores[emotion] += intensity

                total_emotion_words += 1

        # Normalize scores
        if total_emotion_words > 0:
            max_possible = total_emotion_words * 2.0
            emotion_scores = {
                emotion: min(1.0, score / max_possible)
                for emotion, score in emotion_scores.items()
            }

        # Add neutral if no emotions detected
        if not emotion_scores:
            emotion_scores[EmotionalValence.NEUTRAL] = 1.0

        return dict(emotion_scores)

    def get_dominant_emotion(self, emotion_scores: Dict[EmotionalValence, float]) -> Tuple[EmotionalValence, float]:
        """Get the strongest emotion"""
        if not emotion_scores:
            return EmotionalValence.NEUTRAL, 1.0

        dominant = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant

    def emotional_distance(self, emotions1: Dict[EmotionalValence, float],
                          emotions2: Dict[EmotionalValence, float]) -> float:
        """
        Calculate emotional distance between two emotional profiles
        """
        all_emotions = set(emotions1.keys()) | set(emotions2.keys())

        vector1 = np.array([emotions1.get(e, 0.0) for e in EmotionalValence])
        vector2 = np.array([emotions2.get(e, 0.0) for e in EmotionalValence])

        # Euclidean distance
        distance = np.linalg.norm(vector1 - vector2)
        return float(distance)

    def blend_emotions(self, emotions_list: List[Dict[EmotionalValence, float]],
                      weights: List[float] = None) -> Dict[EmotionalValence, float]:
        """Blend multiple emotional profiles"""
        if not emotions_list:
            return {EmotionalValence.NEUTRAL: 1.0}

        if weights is None:
            weights = [1.0] * len(emotions_list)

        blended = defaultdict(float)
        total_weight = sum(weights)

        for emotions, weight in zip(emotions_list, weights):
            for emotion, score in emotions.items():
                blended[emotion] += score * weight / total_weight

        return dict(blended)

    def detect_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Simple sentiment detection: positive, negative, or neutral
        Returns: (sentiment, confidence)
        """
        emotions = self.analyze_emotion(text)

        positive_emotions = {EmotionalValence.JOY, EmotionalValence.TRUST, EmotionalValence.SURPRISE}
        negative_emotions = {EmotionalValence.SADNESS, EmotionalValence.ANGER, EmotionalValence.FEAR}

        positive_score = sum(emotions.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotions.get(e, 0) for e in negative_emotions)

        if positive_score > negative_score and positive_score > 0.1:
            return "positive", positive_score
        elif negative_score > positive_score and negative_score > 0.1:
            return "negative", negative_score
        else:
            return "neutral", emotions.get(EmotionalValence.NEUTRAL, 0.5)

    def generate_emotional_response(self, input_emotions: Dict[EmotionalValence, float],
                                   empathy_level: float = 0.8) -> str:
        """
        Generate appropriate emotional response markers
        This can guide response generation
        """
        dominant_emotion, strength = self.get_dominant_emotion(input_emotions)

        response_markers = {
            EmotionalValence.JOY: ["That's wonderful!", "I'm glad to hear that!", "How exciting!"],
            EmotionalValence.SADNESS: ["I'm sorry to hear that.", "That must be difficult.", "I understand."],
            EmotionalValence.ANGER: ["I can see why that's frustrating.", "That sounds difficult.", "I hear you."],
            EmotionalValence.FEAR: ["That's understandable.", "It's okay to feel that way.", "I'm here to help."],
            EmotionalValence.SURPRISE: ["That's unexpected!", "Interesting!", "Really?"],
            EmotionalValence.TRUST: ["I appreciate your trust.", "Thank you.", "I'm glad we understand each other."],
            EmotionalValence.ANTICIPATION: ["That sounds promising!", "I'm curious to hear more.", "Let's explore that."],
            EmotionalValence.NEUTRAL: ["I see.", "Understood.", "Okay."]
        }

        markers = response_markers.get(dominant_emotion, [""])

        if strength > 0.5 * empathy_level:
            # Return empathetic marker
            import random
            return random.choice(markers)

        return ""

    def emotional_congruence(self, text: str, expected_emotion: EmotionalValence) -> float:
        """
        Check if text matches expected emotional tone
        Returns congruence score (0.0 to 1.0)
        """
        emotions = self.analyze_emotion(text)
        return emotions.get(expected_emotion, 0.0)
