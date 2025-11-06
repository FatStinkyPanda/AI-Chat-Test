"""
Advanced Reasoning Engine
Performs inference, finds connections, identifies gaps, evaluates truthfulness
Enables the AI to truly think and reason
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
from datetime import datetime
import re


class Inference:
    """Represents a single inference made by the system"""
    def __init__(self, premise_ids: List[str], conclusion: str,
                 confidence: float, reasoning_type: str):
        self.premise_ids = premise_ids
        self.conclusion = conclusion
        self.confidence = confidence
        self.reasoning_type = reasoning_type
        self.timestamp = datetime.now()


class InformationGap:
    """Represents a gap in knowledge"""
    def __init__(self, topic: str, context: str, question: str, importance: float):
        self.topic = topic
        self.context = context
        self.question = question
        self.importance = importance


class ReasoningEngine:
    """
    Advanced reasoning engine that performs human-like thinking
    """

    def __init__(self):
        self.inference_patterns = self._initialize_inference_patterns()
        self.contradiction_patterns = self._initialize_contradiction_patterns()

    def _initialize_inference_patterns(self) -> Dict:
        """Initialize patterns for different types of reasoning"""
        return {
            'causal': {
                'if_then': r'\b(if|when)\b.*\b(then|therefore|thus|so)\b',
                'because': r'\b(because|since|due to|as a result of)\b',
                'causes': r'\b(causes|leads to|results in|produces)\b',
            },
            'definitional': {
                'is_a': r'\b(is a|are|is an|is the)\b',
                'means': r'\b(means|refers to|defined as)\b',
            },
            'comparative': {
                'similar': r'\b(similar|like|resembles|comparable to)\b',
                'different': r'\b(different|unlike|contrasts|opposite)\b',
                'better_worse': r'\b(better|worse|superior|inferior)\b',
            },
            'temporal': {
                'before_after': r'\b(before|after|during|while)\b',
                'sequence': r'\b(first|then|next|finally|subsequently)\b',
            },
            'modal': {
                'possibility': r'\b(might|may|could|possibly)\b',
                'necessity': r'\b(must|should|need to|have to)\b',
                'probability': r'\b(probably|likely|unlikely|certainly)\b',
            }
        }

    def _initialize_contradiction_patterns(self) -> List:
        """Patterns that indicate potential contradictions"""
        return [
            (r'\bnot\b', r'\bis\b'),
            (r'\bnever\b', r'\balways\b'),
            (r'\bimpossible\b', r'\bpossible\b'),
            (r'\bcan\'t\b', r'\bcan\b'),
        ]

    def perform_inference(self, memories: List[Dict],
                         current_context: str) -> List[Inference]:
        """
        Perform inferences from memories and current context
        This is where the AI truly thinks
        """
        inferences = []

        # Extract facts from memories
        facts = self._extract_facts(memories)

        # Perform different types of reasoning
        inferences.extend(self._causal_reasoning(facts, current_context))
        inferences.extend(self._analogical_reasoning(facts))
        inferences.extend(self._deductive_reasoning(facts))
        inferences.extend(self._abductive_reasoning(facts, current_context))

        # Sort by confidence
        inferences.sort(key=lambda x: x.confidence, reverse=True)

        return inferences[:10]  # Top 10 inferences

    def _extract_facts(self, memories: List[Dict]) -> List[Dict]:
        """Extract factual statements from memories"""
        facts = []

        for memory in memories:
            content = memory.get('content', '')

            # Skip very short or question statements
            if len(content.split()) < 3 or '?' in content:
                continue

            # Extract sentences
            sentences = re.split(r'[.!;]', content)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Meaningful sentence
                    facts.append({
                        'statement': sentence,
                        'memory_id': memory.get('id'),
                        'content': content,
                        'metadata': memory.get('metadata', {})
                    })

        return facts

    def _causal_reasoning(self, facts: List[Dict], context: str) -> List[Inference]:
        """Reason about causes and effects"""
        inferences = []

        # Look for causal patterns
        for fact in facts:
            statement = fact['statement'].lower()

            # If X causes Y, and we see X, infer Y might happen
            if re.search(self.inference_patterns['causal']['causes'], statement):
                parts = re.split(r'\bcauses\b|\bleads to\b|\bresults in\b', statement)
                if len(parts) >= 2:
                    cause = parts[0].strip()
                    effect = parts[1].strip()

                    # Check if cause is mentioned in context
                    if any(word in context.lower() for word in cause.split()[:3]):
                        inference = Inference(
                            premise_ids=[fact['memory_id']],
                            conclusion=f"Given {cause}, {effect} may occur",
                            confidence=0.7,
                            reasoning_type='causal'
                        )
                        inferences.append(inference)

        return inferences

    def _analogical_reasoning(self, facts: List[Dict]) -> List[Inference]:
        """Find analogies and transfer knowledge"""
        inferences = []

        # Find similar facts that might transfer
        for i, fact1 in enumerate(facts[:20]):  # Limit for performance
            for fact2 in facts[i+1:20]:
                similarity = self._semantic_similarity_simple(
                    fact1['statement'],
                    fact2['statement']
                )

                if 0.3 < similarity < 0.8:  # Similar but not identical
                    # Create analogical inference
                    inference = Inference(
                        premise_ids=[fact1['memory_id'], fact2['memory_id']],
                        conclusion=f"Concepts in '{fact1['statement'][:50]}' relate to '{fact2['statement'][:50]}'",
                        confidence=similarity,
                        reasoning_type='analogical'
                    )
                    inferences.append(inference)

        return inferences[:5]  # Top 5

    def _deductive_reasoning(self, facts: List[Dict]) -> List[Inference]:
        """Perform logical deduction"""
        inferences = []

        # Look for if-then patterns
        conditionals = []
        for fact in facts:
            statement = fact['statement'].lower()
            if re.search(self.inference_patterns['causal']['if_then'], statement):
                conditionals.append(fact)

        # Try to apply conditionals
        for conditional in conditionals:
            statement = conditional['statement']
            # Extract condition and consequence
            parts = re.split(r'\bthen\b|\btherefore\b', statement, maxsplit=1)
            if len(parts) == 2:
                condition = parts[0].replace('if', '').replace('when', '').strip()
                consequence = parts[1].strip()

                # Check if condition is satisfied by other facts
                for fact in facts:
                    if condition[:30].lower() in fact['statement'].lower():
                        inference = Inference(
                            premise_ids=[conditional['memory_id'], fact['memory_id']],
                            conclusion=f"Therefore: {consequence}",
                            confidence=0.8,
                            reasoning_type='deductive'
                        )
                        inferences.append(inference)

        return inferences

    def _abductive_reasoning(self, facts: List[Dict], context: str) -> List[Inference]:
        """Infer best explanations (abduction)"""
        inferences = []

        # Given observations in context, what might explain them?
        context_lower = context.lower()

        # Look for potential explanations in facts
        for fact in facts:
            statement = fact['statement']

            # If fact contains "because" or "explains", it's a potential explanation
            if any(word in statement.lower() for word in ['because', 'explains', 'reason', 'due to']):
                # Check relevance to context
                words = set(context_lower.split())
                fact_words = set(statement.lower().split())
                overlap = len(words & fact_words)

                if overlap > 2:
                    inference = Inference(
                        premise_ids=[fact['memory_id']],
                        conclusion=f"This might explain the situation: {statement[:80]}",
                        confidence=min(0.6, overlap * 0.1),
                        reasoning_type='abductive'
                    )
                    inferences.append(inference)

        return inferences

    def _semantic_similarity_simple(self, text1: str, text2: str) -> float:
        """Simple semantic similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def find_connections(self, node_id: str, all_nodes: Dict,
                        all_edges: List, max_depth: int = 3) -> List[Dict]:
        """
        Find connections between concepts using graph traversal
        Returns interesting connection paths
        """
        connections = []

        # BFS to find paths
        queue = [(node_id, [node_id], 0)]
        visited = {node_id}

        while queue:
            current, path, depth = queue.pop(0)

            if depth >= max_depth:
                continue

            # Get neighbors
            for edge in all_edges:
                neighbor = None
                if edge.source_id == current and edge.target_id not in visited:
                    neighbor = edge.target_id
                elif edge.target_id == current and edge.source_id not in visited:
                    neighbor = edge.source_id

                if neighbor:
                    new_path = path + [neighbor]
                    visited.add(neighbor)
                    queue.append((neighbor, new_path, depth + 1))

                    # If path is interesting (crosses different edge types)
                    if depth >= 1:
                        connections.append({
                            'path': new_path,
                            'depth': depth + 1,
                            'edge_type': edge.edge_type,
                            'strength': edge.strength
                        })

        # Sort by interestingness (depth and strength)
        connections.sort(key=lambda x: x['depth'] * x['strength'], reverse=True)

        return connections[:10]

    def identify_information_gaps(self, memories: List[Dict],
                                 current_input: str) -> List[InformationGap]:
        """
        Identify gaps in knowledge
        What does the AI not know that it should?
        """
        gaps = []

        # Extract topics from current input
        topics = self._extract_topics(current_input)

        for topic in topics:
            # Check how much we know about this topic
            relevant_memories = [
                m for m in memories
                if topic.lower() in m.get('content', '').lower()
            ]

            # If we have few memories about this topic, it's a gap
            if len(relevant_memories) < 3:
                gap = InformationGap(
                    topic=topic,
                    context=current_input,
                    question=f"What else should I know about {topic}?",
                    importance=0.8
                )
                gaps.append(gap)

            # Check for incomplete information
            for memory in relevant_memories:
                content = memory.get('content', '')

                # Questions often indicate gaps
                if '?' in content:
                    question = content.split('?')[0] + '?'
                    gap = InformationGap(
                        topic=topic,
                        context=content,
                        question=question,
                        importance=0.6
                    )
                    gaps.append(gap)

        # Sort by importance
        gaps.sort(key=lambda x: x.importance, reverse=True)

        return gaps[:5]

    def evaluate_truthfulness(self, statement: str, memories: List[Dict]) -> Tuple[float, str]:
        """
        Evaluate the truthfulness/consistency of a statement
        Returns: (confidence, reasoning)
        """
        # Extract facts from memories
        facts = self._extract_facts(memories)

        # Check for contradictions
        contradictions = []
        for fact in facts:
            if self._are_contradictory(statement, fact['statement']):
                contradictions.append(fact)

        # Check for supporting evidence
        support = []
        for fact in facts:
            if self._supports(statement, fact['statement']):
                support.append(fact)

        # Calculate confidence
        if contradictions and not support:
            confidence = 0.2
            reasoning = f"Contradicts {len(contradictions)} known facts"
        elif support and not contradictions:
            confidence = 0.9
            reasoning = f"Supported by {len(support)} known facts"
        elif contradictions and support:
            confidence = 0.5
            reasoning = f"Mixed evidence: {len(support)} support, {len(contradictions)} contradict"
        else:
            confidence = 0.5
            reasoning = "Insufficient evidence to evaluate"

        return confidence, reasoning

    def _are_contradictory(self, statement1: str, statement2: str) -> bool:
        """Check if two statements contradict"""
        s1_lower = statement1.lower()
        s2_lower = statement2.lower()

        # Check for explicit negation
        for neg_pattern, pos_pattern in self.contradiction_patterns:
            if re.search(neg_pattern, s1_lower) and re.search(pos_pattern, s2_lower):
                # Check if they're about the same topic
                if self._semantic_similarity_simple(s1_lower, s2_lower) > 0.4:
                    return True

        return False

    def _supports(self, statement: str, evidence: str) -> bool:
        """Check if evidence supports statement"""
        similarity = self._semantic_similarity_simple(statement, evidence)
        return similarity > 0.5

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        # Remove common words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
            'it', 'from', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'i', 'you', 'he', 'she', 'we', 'they'
        }

        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        topics = [w for w in words if w not in stop_words]

        return topics[:5]  # Top 5 topics

    def generate_thought(self, context: str, memories: List[Dict],
                        inferences: List[Inference]) -> Optional[str]:
        """
        Generate an internal thought based on reasoning
        This thought will be stored as a memory node
        """
        if not inferences:
            return None

        # Take the most confident inference
        best_inference = inferences[0]

        # Generate thought text
        thought_templates = [
            f"I'm thinking: {best_inference.conclusion}",
            f"Based on what I know, {best_inference.conclusion}",
            f"My reasoning: {best_inference.conclusion}",
            f"I infer that {best_inference.conclusion}",
            f"Connecting the dots: {best_inference.conclusion}",
        ]

        import random
        thought = random.choice(thought_templates)

        # Add confidence qualifier
        if best_inference.confidence < 0.5:
            thought = "Uncertain, but " + thought.lower()
        elif best_inference.confidence > 0.8:
            thought = "Confidently: " + thought

        return thought
