"""
Advanced Causal Chain Discovery System

This system discovers, builds, and reasons about causal relationships in conversations.
It creates explicit causal graphs that show how concepts, events, and ideas are causally connected.
"""

import json
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import re
from datetime import datetime


@dataclass
class CausalRelation:
    """Represents a single causal relationship between two concepts"""
    cause: str
    effect: str
    confidence: float
    evidence: List[str] = field(default_factory=list)  # Supporting memory IDs
    strength: float = 0.5  # How strong is this causal link
    temporal_gap: float = 0.0  # Time between cause and effect
    co_occurrence_count: int = 1
    discovered_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'cause': self.cause,
            'effect': self.effect,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'strength': self.strength,
            'temporal_gap': self.temporal_gap,
            'co_occurrence_count': self.co_occurrence_count,
            'discovered_at': self.discovered_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CausalRelation':
        return cls(**data)


@dataclass
class CausalChain:
    """Represents a chain of causal relationships"""
    chain: List[str]  # [A, B, C] means A → B → C
    relations: List[CausalRelation]
    total_confidence: float
    chain_strength: float

    def to_dict(self) -> Dict:
        return {
            'chain': self.chain,
            'relations': [r.to_dict() for r in self.relations],
            'total_confidence': self.total_confidence,
            'chain_strength': self.chain_strength
        }


class CausalDiscoveryEngine:
    """
    Advanced engine for discovering and reasoning about causal relationships.
    Uses multiple detection strategies to find causes and effects in conversations.
    """

    # Linguistic patterns indicating causality
    CAUSAL_PATTERNS = [
        # Strong causal indicators
        (r'because\s+(.*?)\s*,?\s*(?:therefore|thus|so|then|hence)?\s*(.*?)(?:\.|$)', 0.9),
        (r'since\s+(.*?)\s*,?\s*(.*?)(?:\.|$)', 0.85),
        (r'(.*?)\s*(?:caused|causes|led to|results in|produces)\s*(.*?)(?:\.|$)', 0.95),
        (r'(.*?)\s*,?\s*(?:therefore|thus|consequently|hence)\s*(.*?)(?:\.|$)', 0.9),
        (r'if\s+(.*?)\s*,?\s*then\s*(.*?)(?:\.|$)', 0.85),
        (r'when\s+(.*?)\s*,?\s*(.*?)(?:\.|$)', 0.75),

        # Medium causal indicators
        (r'(.*?)\s*(?:makes|made)\s*(.*?)\s*(?:happen|occur|possible)(?:\.|$)', 0.8),
        (r'(.*?)\s*(?:triggers|triggered)\s*(.*?)(?:\.|$)', 0.85),
        (r'(.*?)\s*(?:enables|enabled|allows|allowed)\s*(.*?)(?:\.|$)', 0.8),
        (r'due to\s+(.*?)\s*,?\s*(.*?)(?:\.|$)', 0.85),
        (r'as a result of\s+(.*?)\s*,?\s*(.*?)(?:\.|$)', 0.9),

        # Weaker causal indicators
        (r'after\s+(.*?)\s*,?\s*(.*?)(?:\.|$)', 0.6),
        (r'(.*?)\s*(?:affects|affected|influences|influenced)\s*(.*?)(?:\.|$)', 0.7),
        (r'(.*?)\s*(?:contributes to|contributes towards)\s*(.*?)(?:\.|$)', 0.75),
    ]

    def __init__(self):
        self.causal_relations: Dict[Tuple[str, str], CausalRelation] = {}
        self.concept_causes: Dict[str, Set[str]] = defaultdict(set)  # What causes this concept
        self.concept_effects: Dict[str, Set[str]] = defaultdict(set)  # What this concept causes
        self.causal_chains: List[CausalChain] = []
        self.temporal_sequences: List[List[Tuple[str, float]]] = []  # Track temporal event sequences

    def discover_causal_relations(self, text: str, memory_id: str, timestamp: float) -> List[CausalRelation]:
        """
        Discover causal relationships in text using multiple detection strategies.
        """
        discovered = []

        # Strategy 1: Pattern-based detection
        for pattern, confidence in self.CAUSAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    cause = self._clean_concept(groups[0])
                    effect = self._clean_concept(groups[1])

                    if cause and effect and cause != effect:
                        relation = self._add_or_update_relation(
                            cause, effect, confidence, memory_id, timestamp
                        )
                        discovered.append(relation)

        # Strategy 2: Temporal sequence analysis
        # Extract concepts and track their order for implicit causality
        concepts = self._extract_concepts(text)
        if len(concepts) >= 2:
            self._add_temporal_sequence(concepts, timestamp)

        return discovered

    def _clean_concept(self, text: str) -> str:
        """Clean and normalize concept text"""
        text = text.strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.lower() if text else ""

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text (nouns, noun phrases)"""
        # Simple extraction - could be enhanced with NLP
        concepts = []
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            # Extract noun-like phrases (capitalized words or common patterns)
            words = sentence.split()
            for i, word in enumerate(words):
                if len(word) > 3:  # Meaningful words
                    concept = self._clean_concept(word)
                    if concept and len(concept) > 2:
                        concepts.append(concept)
        return concepts

    def _add_temporal_sequence(self, concepts: List[str], timestamp: float):
        """Track temporal sequences for implicit causal discovery"""
        sequence = [(c, timestamp) for c in concepts]
        self.temporal_sequences.append(sequence)

        # Keep only recent sequences (last 100)
        if len(self.temporal_sequences) > 100:
            self.temporal_sequences = self.temporal_sequences[-100:]

    def _add_or_update_relation(
        self,
        cause: str,
        effect: str,
        confidence: float,
        memory_id: str,
        timestamp: float
    ) -> CausalRelation:
        """Add a new causal relation or update existing one"""
        key = (cause, effect)

        if key in self.causal_relations:
            # Update existing relation
            relation = self.causal_relations[key]
            relation.co_occurrence_count += 1
            relation.confidence = min(0.99, relation.confidence + 0.05)  # Increase confidence
            relation.strength = min(1.0, relation.strength + 0.1)
            if memory_id not in relation.evidence:
                relation.evidence.append(memory_id)
        else:
            # Create new relation
            relation = CausalRelation(
                cause=cause,
                effect=effect,
                confidence=confidence,
                evidence=[memory_id],
                strength=confidence * 0.7,  # Initial strength based on pattern confidence
                discovered_at=timestamp
            )
            self.causal_relations[key] = relation

            # Update indices
            self.concept_causes[effect].add(cause)
            self.concept_effects[cause].add(effect)

        return relation

    def find_causal_chains(self, concept: str, max_depth: int = 4) -> List[CausalChain]:
        """
        Find causal chains starting from a concept.
        Explores both forward (what does this cause?) and backward (what caused this?).
        """
        chains = []

        # Forward chains (what does this concept cause?)
        forward_chains = self._explore_forward_chains(concept, max_depth)
        chains.extend(forward_chains)

        # Backward chains (what caused this concept?)
        backward_chains = self._explore_backward_chains(concept, max_depth)
        chains.extend(backward_chains)

        # Sort by confidence
        chains.sort(key=lambda x: x.total_confidence, reverse=True)

        return chains

    def _explore_forward_chains(self, start: str, max_depth: int) -> List[CausalChain]:
        """Explore forward causal chains (what does this cause?)"""
        chains = []

        # BFS to find chains
        queue = deque([(start, [start], [], 1.0, 1.0)])  # (current, chain, relations, confidence, strength)
        visited = set()

        while queue:
            current, chain, relations, confidence, strength = queue.popleft()

            # Get effects of current concept
            if current in self.concept_effects:
                for effect in self.concept_effects[current]:
                    key = (current, effect)
                    relation = self.causal_relations[key]

                    # Avoid cycles
                    if effect not in chain and len(chain) < max_depth:
                        new_chain = chain + [effect]
                        new_relations = relations + [relation]
                        new_confidence = confidence * relation.confidence
                        new_strength = (strength + relation.strength) / 2

                        # Add as a discovered chain
                        if len(new_chain) >= 2:
                            chains.append(CausalChain(
                                chain=new_chain,
                                relations=new_relations,
                                total_confidence=new_confidence,
                                chain_strength=new_strength
                            ))

                        # Continue exploring
                        state = (effect, tuple(new_chain))
                        if state not in visited:
                            visited.add(state)
                            queue.append((effect, new_chain, new_relations, new_confidence, new_strength))

        return chains

    def _explore_backward_chains(self, start: str, max_depth: int) -> List[CausalChain]:
        """Explore backward causal chains (what caused this?)"""
        chains = []

        # BFS to find chains (going backward)
        queue = deque([(start, [start], [], 1.0, 1.0)])
        visited = set()

        while queue:
            current, chain, relations, confidence, strength = queue.popleft()

            # Get causes of current concept
            if current in self.concept_causes:
                for cause in self.concept_causes[current]:
                    key = (cause, current)
                    relation = self.causal_relations[key]

                    # Avoid cycles
                    if cause not in chain and len(chain) < max_depth:
                        # Build chain in reverse order (cause comes first)
                        new_chain = [cause] + chain
                        new_relations = [relation] + relations
                        new_confidence = confidence * relation.confidence
                        new_strength = (strength + relation.strength) / 2

                        # Add as a discovered chain
                        if len(new_chain) >= 2:
                            chains.append(CausalChain(
                                chain=new_chain,
                                relations=new_relations,
                                total_confidence=new_confidence,
                                chain_strength=new_strength
                            ))

                        # Continue exploring
                        state = (cause, tuple(new_chain))
                        if state not in visited:
                            visited.add(state)
                            queue.append((cause, new_chain, new_relations, new_confidence, new_strength))

        return chains

    def predict_effect(self, cause: str, depth: int = 3) -> List[Tuple[str, float]]:
        """
        Predict what effects might result from a cause.
        Returns list of (effect, confidence) tuples.
        """
        predictions = []
        visited = set()

        # Multi-hop prediction
        queue = deque([(cause, 1.0, 0)])  # (concept, confidence, current_depth)

        while queue:
            current, confidence, current_depth = queue.popleft()

            if current in visited or current_depth >= depth:
                continue

            visited.add(current)

            # Get direct effects
            if current in self.concept_effects:
                for effect in self.concept_effects[current]:
                    key = (current, effect)
                    relation = self.causal_relations[key]

                    effect_confidence = confidence * relation.confidence * relation.strength

                    if effect != cause:  # Avoid circular predictions
                        predictions.append((effect, effect_confidence))

                        # Continue predicting downstream effects
                        if current_depth + 1 < depth:
                            queue.append((effect, effect_confidence, current_depth + 1))

        # Aggregate predictions (same effect might be reached via multiple paths)
        aggregated = defaultdict(float)
        for effect, conf in predictions:
            aggregated[effect] = max(aggregated[effect], conf)  # Take highest confidence path

        # Sort by confidence
        result = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        return result

    def predict_cause(self, effect: str, depth: int = 3) -> List[Tuple[str, float]]:
        """
        Predict what causes might have led to an effect.
        Returns list of (cause, confidence) tuples.
        """
        predictions = []
        visited = set()

        # Multi-hop prediction backward
        queue = deque([(effect, 1.0, 0)])

        while queue:
            current, confidence, current_depth = queue.popleft()

            if current in visited or current_depth >= depth:
                continue

            visited.add(current)

            # Get direct causes
            if current in self.concept_causes:
                for cause in self.concept_causes[current]:
                    key = (cause, current)
                    relation = self.causal_relations[key]

                    cause_confidence = confidence * relation.confidence * relation.strength

                    if cause != effect:
                        predictions.append((cause, cause_confidence))

                        # Continue predicting upstream causes
                        if current_depth + 1 < depth:
                            queue.append((cause, cause_confidence, current_depth + 1))

        # Aggregate predictions
        aggregated = defaultdict(float)
        for cause, conf in predictions:
            aggregated[cause] = max(aggregated[cause], conf)

        result = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
        return result

    def get_causal_explanation(self, effect: str, max_explanations: int = 5) -> List[CausalChain]:
        """
        Get causal explanations for why something happened.
        Returns the most confident causal chains leading to the effect.
        """
        chains = self._explore_backward_chains(effect, max_depth=4)

        # Sort by confidence and return top explanations
        chains.sort(key=lambda x: x.total_confidence * x.chain_strength, reverse=True)
        return chains[:max_explanations]

    def get_causal_insights(self, concept: str) -> Dict:
        """Get comprehensive causal insights about a concept"""
        return {
            'concept': concept,
            'direct_causes': list(self.concept_causes.get(concept, set())),
            'direct_effects': list(self.concept_effects.get(concept, set())),
            'predicted_effects': self.predict_effect(concept, depth=3)[:5],
            'predicted_causes': self.predict_cause(concept, depth=3)[:5],
            'causal_chains': [c.to_dict() for c in self.find_causal_chains(concept, max_depth=3)[:5]],
            'total_causal_relations': len(self.concept_causes.get(concept, set())) + len(self.concept_effects.get(concept, set()))
        }

    def save_state(self, filepath: str):
        """Save causal discovery state to file"""
        state = {
            'causal_relations': {
                f"{k[0]}||{k[1]}": v.to_dict()
                for k, v in self.causal_relations.items()
            },
            'concept_causes': {k: list(v) for k, v in self.concept_causes.items()},
            'concept_effects': {k: list(v) for k, v in self.concept_effects.items()},
            'temporal_sequences': self.temporal_sequences[-50:]  # Save recent sequences
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load causal discovery state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Restore relations
            self.causal_relations = {}
            for key_str, rel_data in state.get('causal_relations', {}).items():
                cause, effect = key_str.split('||')
                self.causal_relations[(cause, effect)] = CausalRelation.from_dict(rel_data)

            # Restore indices
            self.concept_causes = {
                k: set(v) for k, v in state.get('concept_causes', {}).items()
            }
            self.concept_effects = {
                k: set(v) for k, v in state.get('concept_effects', {}).items()
            }

            # Restore temporal sequences
            self.temporal_sequences = state.get('temporal_sequences', [])

        except FileNotFoundError:
            pass  # Start fresh

    def get_statistics(self) -> Dict:
        """Get statistics about discovered causal relationships"""
        return {
            'total_causal_relations': len(self.causal_relations),
            'total_concepts_with_causes': len(self.concept_causes),
            'total_concepts_with_effects': len(self.concept_effects),
            'strongest_relations': sorted(
                [(f"{k[0]} → {k[1]}", v.strength, v.confidence)
                 for k, v in self.causal_relations.items()],
                key=lambda x: x[1] * x[2],
                reverse=True
            )[:10],
            'most_causal_concepts': sorted(
                [(k, len(v)) for k, v in self.concept_effects.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'most_affected_concepts': sorted(
                [(k, len(v)) for k, v in self.concept_causes.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
