"""
Multi-Step Inference Engine

This system builds chains of inferences to reach deeper, more sophisticated conclusions.
It can perform forward chaining (data-driven) and backward chaining (goal-driven) inference.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
import re


@dataclass
class InferenceStep:
    """Represents a single step in an inference chain"""
    step_number: int
    premise_ids: List[str]  # What this step is based on
    conclusion: str
    reasoning_type: str  # deductive, inductive, abductive, analogical, causal
    confidence: float
    method: str  # How this step was derived
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'step_number': self.step_number,
            'premise_ids': self.premise_ids,
            'conclusion': self.conclusion,
            'reasoning_type': self.reasoning_type,
            'confidence': self.confidence,
            'method': self.method,
            'timestamp': self.timestamp
        }


@dataclass
class InferenceChain:
    """Represents a complete chain of reasoning"""
    chain_id: str
    goal: Optional[str]  # Goal for backward chaining
    steps: List[InferenceStep]
    final_conclusion: str
    total_confidence: float
    chain_type: str  # forward, backward, hybrid
    depth: int
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'chain_id': self.chain_id,
            'goal': self.goal,
            'steps': [s.to_dict() for s in self.steps],
            'final_conclusion': self.final_conclusion,
            'total_confidence': self.total_confidence,
            'chain_type': self.chain_type,
            'depth': self.depth,
            'created_at': self.created_at
        }


@dataclass
class InferenceRule:
    """Represents a learned inference rule"""
    rule_id: str
    pattern: str  # Pattern to match
    conclusion_template: str  # How to generate conclusion
    confidence: float
    usage_count: int = 0
    success_rate: float = 1.0

    def to_dict(self) -> Dict:
        return {
            'rule_id': self.rule_id,
            'pattern': self.pattern,
            'conclusion_template': self.conclusion_template,
            'confidence': self.confidence,
            'usage_count': self.usage_count,
            'success_rate': self.success_rate
        }


class MultiStepInferenceEngine:
    """
    Advanced inference engine that chains multiple reasoning steps together
    to reach deeper conclusions than single-step inference.
    """

    def __init__(self):
        self.inference_chains: List[InferenceChain] = []
        self.learned_rules: Dict[str, InferenceRule] = {}
        self.fact_base: Dict[str, Dict] = {}  # Known facts with metadata
        self.inference_history: List[Dict] = []

        # Initialize with some basic inference rules
        self._initialize_rules()

    def _initialize_rules(self):
        """Initialize with basic inference rules"""
        basic_rules = [
            InferenceRule(
                rule_id="transitivity",
                pattern="if A→B and B→C then A→C",
                conclusion_template="transitivity",
                confidence=0.95
            ),
            InferenceRule(
                rule_id="modus_ponens",
                pattern="if A and A→B then B",
                conclusion_template="modus_ponens",
                confidence=0.98
            ),
            InferenceRule(
                rule_id="generalization",
                pattern="if X has property P multiple times, then likely always has P",
                conclusion_template="generalization",
                confidence=0.75
            ),
            InferenceRule(
                rule_id="specialization",
                pattern="if category C has property P, then instance I of C has P",
                conclusion_template="specialization",
                confidence=0.85
            ),
            InferenceRule(
                rule_id="analogy",
                pattern="if A:B::C:? and we know A→X, then C→Y where Y is analogous to X",
                conclusion_template="analogy",
                confidence=0.70
            ),
            InferenceRule(
                rule_id="causal_prediction",
                pattern="if A causes B in past, and A occurs, then B likely",
                conclusion_template="causal_prediction",
                confidence=0.80
            ),
            InferenceRule(
                rule_id="correlation_inference",
                pattern="if A and B frequently co-occur, and A present, then B likely present",
                conclusion_template="correlation",
                confidence=0.65
            ),
            InferenceRule(
                rule_id="negation_inference",
                pattern="if not A, and A→B, then possibly not B",
                conclusion_template="negation",
                confidence=0.60
            )
        ]

        for rule in basic_rules:
            self.learned_rules[rule.rule_id] = rule

    def add_fact(self, fact_id: str, content: str, confidence: float = 1.0, metadata: Dict = None):
        """Add a fact to the knowledge base"""
        self.fact_base[fact_id] = {
            'content': content,
            'confidence': confidence,
            'metadata': metadata or {},
            'timestamp': datetime.now().timestamp()
        }

    def forward_chain(
        self,
        premises: List[Dict],
        max_depth: int = 5,
        min_confidence: float = 0.5
    ) -> List[InferenceChain]:
        """
        Forward chaining: Start with premises and derive all possible conclusions.
        Data-driven reasoning.
        """
        chains = []
        chain_id = f"forward_{datetime.now().timestamp()}"

        # Track what we've inferred to avoid redundant work
        inferred_conclusions = set()

        # Start with initial premises
        current_facts = {f"premise_{i}": p for i, p in enumerate(premises)}
        all_facts = current_facts.copy()

        for depth in range(max_depth):
            new_inferences = []

            # Try to apply inference rules to current facts
            for rule_id, rule in self.learned_rules.items():
                inferences = self._apply_rule(rule, list(all_facts.values()), all_facts)

                for inference in inferences:
                    if inference['conclusion'] not in inferred_conclusions:
                        if inference['confidence'] >= min_confidence:
                            new_inferences.append(inference)
                            inferred_conclusions.add(inference['conclusion'])

            # If no new inferences, stop
            if not new_inferences:
                break

            # Create inference chain from these inferences
            if new_inferences:
                steps = []
                for i, inf in enumerate(new_inferences):
                    step = InferenceStep(
                        step_number=depth * 10 + i,
                        premise_ids=inf['premise_ids'],
                        conclusion=inf['conclusion'],
                        reasoning_type=inf['reasoning_type'],
                        confidence=inf['confidence'],
                        method=inf['method']
                    )
                    steps.append(step)

                    # Add to fact base for next iteration
                    fact_id = f"inferred_{depth}_{i}"
                    all_facts[fact_id] = {
                        'content': inf['conclusion'],
                        'confidence': inf['confidence']
                    }

                # Create chain
                if steps:
                    final_step = steps[-1]
                    chain = InferenceChain(
                        chain_id=f"{chain_id}_depth{depth}",
                        goal=None,
                        steps=steps,
                        final_conclusion=final_step.conclusion,
                        total_confidence=self._calculate_chain_confidence(steps),
                        chain_type="forward",
                        depth=depth + 1
                    )
                    chains.append(chain)

        return chains

    def backward_chain(
        self,
        goal: str,
        premises: List[Dict],
        max_depth: int = 5,
        min_confidence: float = 0.5
    ) -> Optional[InferenceChain]:
        """
        Backward chaining: Start with a goal and work backward to find supporting premises.
        Goal-driven reasoning.
        """
        chain_id = f"backward_{datetime.now().timestamp()}"

        # Track visited goals to avoid cycles
        visited = set()
        steps = []

        # Recursive backward search
        result = self._backward_search(
            goal=goal,
            premises=premises,
            visited=visited,
            depth=0,
            max_depth=max_depth,
            min_confidence=min_confidence,
            steps=steps
        )

        if result['success']:
            return InferenceChain(
                chain_id=chain_id,
                goal=goal,
                steps=result['steps'],
                final_conclusion=goal,
                total_confidence=result['confidence'],
                chain_type="backward",
                depth=len(result['steps'])
            )

        return None

    def _backward_search(
        self,
        goal: str,
        premises: List[Dict],
        visited: Set[str],
        depth: int,
        max_depth: int,
        min_confidence: float,
        steps: List[InferenceStep]
    ) -> Dict:
        """Recursive backward search"""

        # Base case: goal already in premises
        for i, premise in enumerate(premises):
            if goal.lower() in premise.get('content', '').lower():
                return {
                    'success': True,
                    'confidence': premise.get('confidence', 1.0),
                    'steps': []
                }

        # Base case: max depth reached
        if depth >= max_depth:
            return {'success': False, 'confidence': 0.0, 'steps': []}

        # Avoid cycles
        if goal in visited:
            return {'success': False, 'confidence': 0.0, 'steps': []}

        visited.add(goal)

        # Try to find rules that can conclude the goal
        for rule_id, rule in self.learned_rules.items():
            # Check if this rule could derive the goal
            subgoals = self._identify_subgoals(goal, rule, premises)

            if subgoals:
                # Try to prove all subgoals
                all_subgoals_proven = True
                subgoal_confidences = []
                subgoal_steps = []

                for subgoal in subgoals:
                    result = self._backward_search(
                        subgoal, premises, visited.copy(),
                        depth + 1, max_depth, min_confidence, []
                    )

                    if result['success']:
                        subgoal_confidences.append(result['confidence'])
                        subgoal_steps.extend(result['steps'])
                    else:
                        all_subgoals_proven = False
                        break

                if all_subgoals_proven:
                    # All subgoals proven, create inference step
                    avg_confidence = sum(subgoal_confidences) / len(subgoal_confidences) if subgoal_confidences else 0
                    final_confidence = avg_confidence * rule.confidence

                    if final_confidence >= min_confidence:
                        step = InferenceStep(
                            step_number=depth,
                            premise_ids=[f"subgoal_{i}" for i in range(len(subgoals))],
                            conclusion=goal,
                            reasoning_type="backward_chaining",
                            confidence=final_confidence,
                            method=f"rule_{rule_id}"
                        )

                        all_steps = subgoal_steps + [step]

                        return {
                            'success': True,
                            'confidence': final_confidence,
                            'steps': all_steps
                        }

        return {'success': False, 'confidence': 0.0, 'steps': []}

    def _identify_subgoals(self, goal: str, rule: InferenceRule, premises: List[Dict]) -> List[str]:
        """Identify what subgoals would allow us to conclude the goal using this rule"""
        subgoals = []

        # Different strategies based on rule type
        if rule.rule_id == "modus_ponens":
            # If goal is B, need to find A and A→B
            # Look for implications in premises
            for premise in premises:
                content = premise.get('content', '')
                # Check for implication patterns
                if '→' in content or 'implies' in content or 'then' in content:
                    # Extract antecedent as subgoal
                    parts = re.split(r'→|implies|then', content)
                    if len(parts) >= 2:
                        antecedent = parts[0].strip()
                        consequent = parts[1].strip()
                        if goal.lower() in consequent.lower():
                            subgoals.append(antecedent)

        elif rule.rule_id == "transitivity":
            # If goal is A→C, need to find B such that A→B and B→C
            # This is complex, simplified here
            pass

        elif rule.rule_id == "causal_prediction":
            # If goal is "B will happen", need to find "A causes B" and "A is happening"
            cause_patterns = [
                r'(.*?)\s+causes?\s+' + re.escape(goal),
                r'(.*?)\s+leads? to\s+' + re.escape(goal)
            ]
            for premise in premises:
                content = premise.get('content', '')
                for pattern in cause_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        cause = match.group(1).strip()
                        subgoals.append(cause)

        return subgoals

    def _apply_rule(self, rule: InferenceRule, facts: List[Dict], fact_base: Dict) -> List[Dict]:
        """Apply an inference rule to current facts"""
        inferences = []

        if rule.rule_id == "transitivity":
            inferences.extend(self._apply_transitivity(facts, fact_base))
        elif rule.rule_id == "modus_ponens":
            inferences.extend(self._apply_modus_ponens(facts, fact_base))
        elif rule.rule_id == "generalization":
            inferences.extend(self._apply_generalization(facts, fact_base))
        elif rule.rule_id == "specialization":
            inferences.extend(self._apply_specialization(facts, fact_base))
        elif rule.rule_id == "causal_prediction":
            inferences.extend(self._apply_causal_prediction(facts, fact_base))
        elif rule.rule_id == "correlation_inference":
            inferences.extend(self._apply_correlation(facts, fact_base))

        # Update rule usage
        if inferences:
            rule.usage_count += len(inferences)

        return inferences

    def _apply_transitivity(self, facts: List[Dict], fact_base: Dict) -> List[Dict]:
        """Apply transitive reasoning: if A→B and B→C, then A→C"""
        inferences = []

        # Find implications in facts
        implications = []
        for fact_id, fact in fact_base.items():
            content = fact.get('content', '')
            # Look for implication patterns
            matches = re.findall(r'(\w+)\s*(?:→|implies|leads to)\s*(\w+)', content)
            for match in matches:
                implications.append({
                    'from': match[0],
                    'to': match[1],
                    'fact_id': fact_id,
                    'confidence': fact.get('confidence', 1.0)
                })

        # Find transitive chains
        for imp1 in implications:
            for imp2 in implications:
                if imp1['to'] == imp2['from'] and imp1['from'] != imp2['to']:
                    # Found A→B and B→C
                    confidence = imp1['confidence'] * imp2['confidence'] * 0.95
                    inferences.append({
                        'premise_ids': [imp1['fact_id'], imp2['fact_id']],
                        'conclusion': f"{imp1['from']} implies {imp2['to']} (by transitivity)",
                        'reasoning_type': 'deductive',
                        'confidence': confidence,
                        'method': 'transitivity'
                    })

        return inferences

    def _apply_modus_ponens(self, facts: List[Dict], fact_base: Dict) -> List[Dict]:
        """Apply modus ponens: if A and A→B, then B"""
        inferences = []

        # Find implications and facts
        implications = []
        present_facts = []

        for fact_id, fact in fact_base.items():
            content = fact.get('content', '')

            # Check if it's an implication
            if '→' in content or 'implies' in content or 'if' in content:
                matches = re.findall(r'(\w+)\s*(?:→|implies|leads to)\s*(\w+)', content)
                for match in matches:
                    implications.append({
                        'antecedent': match[0],
                        'consequent': match[1],
                        'fact_id': fact_id,
                        'confidence': fact.get('confidence', 1.0)
                    })
            else:
                # It's a fact
                present_facts.append({
                    'content': content,
                    'fact_id': fact_id,
                    'confidence': fact.get('confidence', 1.0)
                })

        # Apply modus ponens
        for imp in implications:
            for pfact in present_facts:
                # Check if fact matches antecedent
                if imp['antecedent'].lower() in pfact['content'].lower():
                    confidence = imp['confidence'] * pfact['confidence'] * 0.98
                    inferences.append({
                        'premise_ids': [imp['fact_id'], pfact['fact_id']],
                        'conclusion': imp['consequent'],
                        'reasoning_type': 'deductive',
                        'confidence': confidence,
                        'method': 'modus_ponens'
                    })

        return inferences

    def _apply_generalization(self, facts: List[Dict], fact_base: Dict) -> List[Dict]:
        """Generalize from multiple similar instances"""
        inferences = []

        # Group facts by similarity
        patterns = defaultdict(list)

        for fact_id, fact in fact_base.items():
            content = fact.get('content', '')
            # Extract patterns (simplified - could use NLP)
            words = content.lower().split()
            for word in words:
                if len(word) > 3:
                    patterns[word].append({
                        'fact_id': fact_id,
                        'content': content,
                        'confidence': fact.get('confidence', 1.0)
                    })

        # If we see a pattern multiple times, generalize
        for pattern, occurrences in patterns.items():
            if len(occurrences) >= 3:  # Need multiple instances
                avg_confidence = sum(o['confidence'] for o in occurrences) / len(occurrences)
                confidence = min(0.95, avg_confidence * 0.75 * (len(occurrences) / 5))

                inferences.append({
                    'premise_ids': [o['fact_id'] for o in occurrences],
                    'conclusion': f"Pattern involving '{pattern}' is common (generalization)",
                    'reasoning_type': 'inductive',
                    'confidence': confidence,
                    'method': 'generalization'
                })

        return inferences

    def _apply_specialization(self, facts: List[Dict], fact_base: Dict) -> List[Dict]:
        """Apply general rules to specific instances"""
        inferences = []

        # Find general statements and specific instances
        general_statements = []
        specific_instances = []

        for fact_id, fact in fact_base.items():
            content = fact.get('content', '').lower()

            if any(word in content for word in ['all', 'every', 'always', 'generally']):
                general_statements.append({
                    'fact_id': fact_id,
                    'content': fact['content'],
                    'confidence': fact.get('confidence', 1.0)
                })
            elif any(word in content for word in ['this', 'that', 'specific', 'particular']):
                specific_instances.append({
                    'fact_id': fact_id,
                    'content': fact['content'],
                    'confidence': fact.get('confidence', 1.0)
                })

        # Apply general to specific
        for general in general_statements:
            for specific in specific_instances:
                # Find common words (simplified matching)
                gen_words = set(general['content'].lower().split())
                spec_words = set(specific['content'].lower().split())
                common = gen_words & spec_words

                if len(common) >= 2:
                    confidence = general['confidence'] * specific['confidence'] * 0.85
                    inferences.append({
                        'premise_ids': [general['fact_id'], specific['fact_id']],
                        'conclusion': f"Specific instance follows general pattern (specialization)",
                        'reasoning_type': 'deductive',
                        'confidence': confidence,
                        'method': 'specialization'
                    })

        return inferences

    def _apply_causal_prediction(self, facts: List[Dict], fact_base: Dict) -> List[Dict]:
        """Predict effects based on known causal relations"""
        inferences = []

        # Find causal relations and current states
        causal_relations = []
        current_states = []

        for fact_id, fact in fact_base.items():
            content = fact.get('content', '')

            # Check for causal language
            if any(word in content.lower() for word in ['causes', 'leads to', 'results in', 'produces']):
                causal_relations.append({
                    'fact_id': fact_id,
                    'content': content,
                    'confidence': fact.get('confidence', 1.0)
                })
            else:
                current_states.append({
                    'fact_id': fact_id,
                    'content': content,
                    'confidence': fact.get('confidence', 1.0)
                })

        # Make predictions
        for causal in causal_relations:
            # Extract cause and effect (simplified)
            matches = re.findall(r'(.*?)\s+(?:causes?|leads? to|results? in)\s+(.*?)(?:\.|$)',
                                 causal['content'], re.IGNORECASE)

            for match in matches:
                cause = match[0].strip()
                effect = match[1].strip()

                # Check if cause is present in current states
                for state in current_states:
                    if cause.lower() in state['content'].lower():
                        confidence = causal['confidence'] * state['confidence'] * 0.80
                        inferences.append({
                            'premise_ids': [causal['fact_id'], state['fact_id']],
                            'conclusion': f"Predict: {effect} (because {cause} is present)",
                            'reasoning_type': 'causal',
                            'confidence': confidence,
                            'method': 'causal_prediction'
                        })

        return inferences

    def _apply_correlation(self, facts: List[Dict], fact_base: Dict) -> List[Dict]:
        """Infer based on correlations"""
        inferences = []

        # Track co-occurrences (simplified)
        concept_pairs = defaultdict(list)

        for fact_id, fact in fact_base.items():
            content = fact.get('content', '')
            words = [w.lower() for w in content.split() if len(w) > 3]

            # Track pairs
            for i, w1 in enumerate(words):
                for w2 in words[i+1:]:
                    pair = tuple(sorted([w1, w2]))
                    concept_pairs[pair].append({
                        'fact_id': fact_id,
                        'confidence': fact.get('confidence', 1.0)
                    })

        # If concepts co-occur frequently, infer correlation
        for pair, occurrences in concept_pairs.items():
            if len(occurrences) >= 2:
                avg_confidence = sum(o['confidence'] for o in occurrences) / len(occurrences)
                confidence = min(0.85, avg_confidence * 0.65 * (len(occurrences) / 3))

                inferences.append({
                    'premise_ids': [o['fact_id'] for o in occurrences],
                    'conclusion': f"Concepts '{pair[0]}' and '{pair[1]}' are correlated",
                    'reasoning_type': 'inductive',
                    'confidence': confidence,
                    'method': 'correlation'
                })

        return inferences

    def _calculate_chain_confidence(self, steps: List[InferenceStep]) -> float:
        """Calculate overall confidence for a chain of inferences"""
        if not steps:
            return 0.0

        # Multiply confidences with diminishing returns
        confidence = 1.0
        for i, step in enumerate(steps):
            # Each step reduces confidence, but with diminishing impact
            decay = 0.95 ** i
            confidence *= (step.confidence * decay)

        return max(0.0, min(1.0, confidence))

    def get_best_inference_chains(self, top_k: int = 5) -> List[InferenceChain]:
        """Get the top K most confident inference chains"""
        sorted_chains = sorted(
            self.inference_chains,
            key=lambda c: c.total_confidence * c.depth,
            reverse=True
        )
        return sorted_chains[:top_k]

    def explain_conclusion(self, conclusion: str) -> List[InferenceChain]:
        """Find all inference chains that lead to a specific conclusion"""
        matching_chains = []

        for chain in self.inference_chains:
            if conclusion.lower() in chain.final_conclusion.lower():
                matching_chains.append(chain)

        matching_chains.sort(key=lambda c: c.total_confidence, reverse=True)
        return matching_chains

    def get_statistics(self) -> Dict:
        """Get statistics about inference chains"""
        return {
            'total_chains': len(self.inference_chains),
            'average_depth': sum(c.depth for c in self.inference_chains) / len(self.inference_chains) if self.inference_chains else 0,
            'average_confidence': sum(c.total_confidence for c in self.inference_chains) / len(self.inference_chains) if self.inference_chains else 0,
            'forward_chains': len([c for c in self.inference_chains if c.chain_type == 'forward']),
            'backward_chains': len([c for c in self.inference_chains if c.chain_type == 'backward']),
            'total_rules': len(self.learned_rules),
            'most_used_rules': sorted(
                [(r.rule_id, r.usage_count) for r in self.learned_rules.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
