"""
Deliberation Engine
Allows the AI to think iteratively and decide when it's ready to respond
The AI thinks for itself and determines when it has sufficient understanding
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import random


@dataclass
class ThinkingIteration:
    """Represents one iteration of thinking"""
    iteration_number: int
    thoughts: List[str]
    inferences: List[Any]
    information_gaps: List[Any]
    confidence: float
    timestamp: datetime
    new_insights: int


@dataclass
class DeliberationResult:
    """Result of the deliberation process"""
    total_iterations: int
    thinking_iterations: List[ThinkingIteration]
    final_confidence: float
    readiness_score: float
    stopping_reason: str
    key_insights: List[str]
    response_direction: str  # Suggested direction for response
    associations: List[Dict] = None  # NEW: Intelligent associations found
    enhanced_inferences: List[str] = None  # NEW: Enhanced inferences
    response_strategy: Dict = None  # NEW: Strategy for response


class DeliberationEngine:
    """
    Manages the AI's internal thinking process
    Decides autonomously when enough thinking has been done
    Now uses enhanced reasoning for smarter thinking
    """

    def __init__(self, reasoning_engine, enhanced_reasoner=None):
        self.reasoning_engine = reasoning_engine
        self.enhanced_reasoner = enhanced_reasoner  # New enhanced reasoning

        # Deliberation parameters
        self.min_iterations = 1
        self.max_iterations = 5
        self.confidence_threshold = 0.75
        self.readiness_threshold = 0.70
        self.diminishing_returns_threshold = 0.15  # Stop if new insights < this

    def deliberate(self, user_input: str,
                   relevant_memories: List[Dict],
                   initial_inferences: List,
                   initial_gaps: List,
                   emotional_context: Dict) -> DeliberationResult:
        """
        Think iteratively about the input and decide when ready to respond

        The AI will:
        1. Think about the input from multiple angles
        2. Generate inferences and identify gaps
        3. Evaluate its own understanding
        4. Decide when it has thought enough
        5. Return its deliberation results

        NOW ENHANCED: Uses intelligent association finding and inference generation
        """

        iterations = []
        previous_insights = set()
        all_associations = []
        all_inferences = []

        print(f"\n[Deliberation] Starting enhanced thinking process...")

        # Use enhanced reasoner if available
        if self.enhanced_reasoner:
            # Find intelligent associations
            all_associations = self.enhanced_reasoner.find_intelligent_associations(
                user_input, relevant_memories
            )
            print(f"[Enhanced] Found {len(all_associations)} intelligent associations")

            # Generate conversational inferences
            all_inferences = self.enhanced_reasoner.generate_conversational_inferences(
                user_input, relevant_memories, all_associations
            )
            print(f"[Enhanced] Generated {len(all_inferences)} conversational inferences")

        for i in range(self.max_iterations):
            iteration_num = i + 1

            # Perform thinking for this iteration
            iteration_result = self._think_iteration(
                iteration_num=iteration_num,
                user_input=user_input,
                memories=relevant_memories,
                previous_iterations=iterations,
                emotional_context=emotional_context,
                associations=all_associations,
                enhanced_inferences=all_inferences
            )

            iterations.append(iteration_result)

            # Track new insights
            current_insights = self._extract_insight_keys(iteration_result)
            new_insights_count = len(current_insights - previous_insights)
            iteration_result.new_insights = new_insights_count

            # Log thinking
            print(f"[Deliberation] Iteration {iteration_num}:")
            print(f"  - Generated {len(iteration_result.thoughts)} thoughts")
            print(f"  - Found {len(iteration_result.inferences)} inferences")
            print(f"  - Identified {len(iteration_result.information_gaps)} gaps")
            print(f"  - New insights: {new_insights_count}")
            print(f"  - Confidence: {iteration_result.confidence:.2f}")

            # Evaluate readiness after minimum iterations
            if iteration_num >= self.min_iterations:
                is_ready, readiness_score, reason = self._evaluate_readiness(
                    iterations=iterations,
                    current_iteration=iteration_result,
                    new_insights_count=new_insights_count
                )

                if is_ready:
                    print(f"[Deliberation] Ready to respond after {iteration_num} iterations")
                    print(f"[Deliberation] Reason: {reason}")

                    # Compile deliberation results
                    result = self._compile_results(
                        iterations=iterations,
                        readiness_score=readiness_score,
                        stopping_reason=reason,
                        user_input=user_input,
                        relevant_memories=relevant_memories,
                        associations=all_associations,
                        enhanced_inferences=all_inferences
                    )

                    return result

            # Update insights for next iteration
            previous_insights.update(current_insights)

        # Reached max iterations
        print(f"[Deliberation] Completed maximum {self.max_iterations} iterations")

        final_iteration = iterations[-1]
        result = self._compile_results(
            iterations=iterations,
            readiness_score=final_iteration.confidence,
            stopping_reason=f"Reached maximum {self.max_iterations} thinking iterations",
            user_input=user_input,
            relevant_memories=relevant_memories,
            associations=all_associations,
            enhanced_inferences=all_inferences
        )

        return result

    def _think_iteration(self, iteration_num: int,
                        user_input: str,
                        memories: List[Dict],
                        previous_iterations: List[ThinkingIteration],
                        emotional_context: Dict,
                        associations: List[Dict] = None,
                        enhanced_inferences: List[str] = None) -> ThinkingIteration:
        """
        Perform one iteration of thinking
        NOW ENHANCED: Uses intelligent associations and inferences
        """

        # Build context from previous thinking
        thinking_context = self._build_thinking_context(
            user_input=user_input,
            previous_iterations=previous_iterations
        )

        # Use enhanced inferences if available, otherwise fallback to reasoning engine
        if enhanced_inferences and len(enhanced_inferences) > 0:
            # Convert string inferences to Inference objects for compatibility
            inferences = enhanced_inferences  # Use as-is, they're more useful as strings
        else:
            # Generate inferences using reasoning engine
            inferences = self.reasoning_engine.perform_inference(
                memories=memories,
                current_context=thinking_context
            )

        # Identify information gaps
        gaps = self.reasoning_engine.identify_information_gaps(
            memories=memories,
            current_input=thinking_context
        )

        # Generate thoughts about the situation
        thoughts = self._generate_deliberation_thoughts(
            user_input=user_input,
            iteration_num=iteration_num,
            inferences=inferences,
            gaps=gaps,
            emotional_context=emotional_context,
            previous_iterations=previous_iterations,
            associations=associations
        )

        # Calculate confidence for this iteration
        confidence = self._calculate_iteration_confidence(
            inferences=inferences,
            gaps=gaps,
            thoughts=thoughts,
            previous_iterations=previous_iterations
        )

        return ThinkingIteration(
            iteration_number=iteration_num,
            thoughts=thoughts,
            inferences=inferences,
            information_gaps=gaps,
            confidence=confidence,
            timestamp=datetime.now(),
            new_insights=0  # Will be set by caller
        )

    def _build_thinking_context(self, user_input: str,
                                previous_iterations: List[ThinkingIteration]) -> str:
        """
        Build context for current thinking iteration
        Includes original input and insights from previous thinking
        """

        if not previous_iterations:
            return user_input

        # Add recent thoughts to context
        context_parts = [user_input]

        for iteration in previous_iterations[-2:]:  # Last 2 iterations
            if iteration.thoughts:
                context_parts.append(iteration.thoughts[0])  # Most confident thought

        return " | ".join(context_parts)

    def _generate_deliberation_thoughts(self, user_input: str,
                                       iteration_num: int,
                                       inferences: List,
                                       gaps: List,
                                       emotional_context: Dict,
                                       previous_iterations: List[ThinkingIteration],
                                       associations: List[Dict] = None) -> List[str]:
        """
        Generate thoughts during deliberation
        These are different from response - they're internal thinking
        NOW ENHANCED: Uses associations and string-based inferences
        """

        thoughts = []

        # Think about inferences (now they can be strings from enhanced reasoner)
        if inferences:
            if isinstance(inferences[0], str):
                # Enhanced string inferences
                top_inference = inferences[0]
                thoughts.append(f"Understanding: {top_inference}")
            else:
                # Traditional Inference objects
                top_inference = inferences[0]
                thought = f"Considering: {top_inference.conclusion}"
                thoughts.append(thought)

                if top_inference.confidence > 0.7:
                    thoughts.append(f"This seems quite certain ({top_inference.reasoning_type} reasoning)")
                else:
                    thoughts.append(f"This is somewhat uncertain (confidence: {top_inference.confidence:.2f})")

        # Think about associations
        if associations:
            strong_assocs = [a for a in associations if a.get('association_strength', 0) > 0.5]
            if strong_assocs:
                thoughts.append(f"Found {len(strong_assocs)} strong connections to past conversations")

        # Think about gaps
        if gaps:
            gap_topics = [g.topic for g in gaps[:2]]
            thoughts.append(f"I need more information about: {', '.join(gap_topics)}")

        # Think about emotional context
        if emotional_context:
            emotions = [k for k, v in emotional_context.items() if v > 0.3]
            if emotions:
                thoughts.append(f"The emotional tone seems {emotions[0]}")

        # Meta-thinking (thinking about own thinking)
        if iteration_num > 1:
            if previous_iterations:
                prev_confidence = previous_iterations[-1].confidence
                curr_confidence = self._calculate_iteration_confidence(
                    inferences, gaps, thoughts, previous_iterations
                )

                if curr_confidence > prev_confidence + 0.1:
                    thoughts.append("My understanding is improving with more thought")
                elif curr_confidence < prev_confidence - 0.05:
                    thoughts.append("Additional thinking isn't adding much clarity")

        # Think about response approach
        if iteration_num >= 2 and inferences:
            thoughts.append(self._plan_response_approach(inferences, gaps, emotional_context))

        return thoughts

    def _plan_response_approach(self, inferences: List, gaps: List,
                               emotional_context: Dict) -> str:
        """
        Plan what kind of response would be appropriate
        NOW HANDLES: String inferences from enhanced reasoner
        """

        approaches = []

        if gaps and len(gaps) > 2:
            approaches.append("Should ask clarifying questions")

        # Handle both string and Inference object inferences
        if inferences:
            if isinstance(inferences[0], str):
                # Enhanced string inferences - having multiple means we understand well
                if len(inferences) >= 3:
                    approaches.append("Can share confident insights")
            elif any(hasattr(inf, 'confidence') and inf.confidence > 0.75 for inf in inferences):
                approaches.append("Can share confident insights")

        if emotional_context:
            high_emotion = any(v > 0.5 for v in emotional_context.values())
            if high_emotion:
                approaches.append("Should respond with emotional awareness")

        if inferences and gaps:
            approaches.append("Balance sharing what I know with acknowledging gaps")

        if approaches:
            return f"Response approach: {'; '.join(approaches)}"
        else:
            return "Response approach: Engage with curiosity and openness"

    def _calculate_iteration_confidence(self, inferences: List, gaps: List,
                                       thoughts: List,
                                       previous_iterations: List[ThinkingIteration]) -> float:
        """
        Calculate confidence level for this iteration
        Higher confidence = better understanding
        NOW HANDLES: String inferences from enhanced reasoner
        """

        base_confidence = 0.5

        # More high-confidence inferences = higher confidence
        if inferences:
            if isinstance(inferences[0], str):
                # String inferences from enhanced reasoner
                # Having inferences means we understood something
                base_confidence += 0.3 * min(1.0, len(inferences) * 0.2)
            else:
                # Traditional Inference objects with confidence scores
                avg_inference_conf = sum(inf.confidence for inf in inferences) / len(inferences)
                base_confidence += avg_inference_conf * 0.3

        # Fewer gaps = higher confidence
        gap_penalty = min(0.2, len(gaps) * 0.05)
        base_confidence -= gap_penalty

        # More thoughts = better understanding
        thought_bonus = min(0.15, len(thoughts) * 0.03)
        base_confidence += thought_bonus

        # Improvement over previous iterations
        if previous_iterations and len(previous_iterations) > 0:
            prev_conf = previous_iterations[-1].confidence
            # If we're making progress, bonus
            if inferences and len(inferences) > len(previous_iterations[-1].inferences):
                base_confidence += 0.1

        return max(0.0, min(1.0, base_confidence))

    def _extract_insight_keys(self, iteration: ThinkingIteration) -> set:
        """
        Extract key insights from an iteration for tracking novelty
        NOW HANDLES: String inferences from enhanced reasoner
        """

        insights = set()

        # Add inference conclusions (handle both string and Inference objects)
        for inf in iteration.inferences:
            if isinstance(inf, str):
                insights.add(inf[:50])  # First 50 chars as key
            elif hasattr(inf, 'conclusion'):
                insights.add(inf.conclusion[:50])  # First 50 chars as key

        # Add gap topics
        for gap in iteration.information_gaps:
            insights.add(f"gap:{gap.topic}")

        # Add unique thought elements
        for thought in iteration.thoughts:
            # Extract key words
            words = thought.lower().split()
            key_words = [w for w in words if len(w) > 5][:3]  # 3 longest words
            if key_words:
                insights.add("-".join(key_words))

        return insights

    def _evaluate_readiness(self, iterations: List[ThinkingIteration],
                           current_iteration: ThinkingIteration,
                           new_insights_count: int) -> Tuple[bool, float, str]:
        """
        Evaluate if the AI is ready to respond
        Returns: (is_ready, readiness_score, reason)

        The AI decides autonomously when it's done thinking
        """

        # Calculate readiness score
        readiness_factors = []

        # Factor 1: Confidence level
        confidence_score = current_iteration.confidence
        readiness_factors.append(("confidence", confidence_score, 0.4))

        # Factor 2: Diminishing returns (not learning much new)
        if len(iterations) > 1:
            diminishing = new_insights_count < 2
            diminishing_score = 1.0 if diminishing else 0.3
        else:
            diminishing_score = 0.5
        readiness_factors.append(("diminishing_returns", diminishing_score, 0.3))

        # Factor 3: Has some inferences to work with
        has_inferences = len(current_iteration.inferences) > 0
        inference_score = 1.0 if has_inferences else 0.3
        readiness_factors.append(("has_inferences", inference_score, 0.2))

        # Factor 4: Not too many unresolved gaps
        gap_score = max(0.0, 1.0 - (len(current_iteration.information_gaps) * 0.15))
        readiness_factors.append(("gap_resolution", gap_score, 0.1))

        # Calculate weighted readiness score
        readiness_score = sum(score * weight for _, score, weight in readiness_factors)

        # Determine if ready based on criteria
        ready = False
        reason = ""

        # Criterion 1: High confidence reached
        if confidence_score >= self.confidence_threshold:
            ready = True
            reason = f"High confidence reached ({confidence_score:.2f})"

        # Criterion 2: Diminishing returns
        elif diminishing_score > 0.8 and len(iterations) >= 2:
            ready = True
            reason = f"Diminishing returns - not generating new insights (only {new_insights_count} new)"

        # Criterion 3: Overall readiness threshold
        elif readiness_score >= self.readiness_threshold:
            ready = True
            reason = f"Overall readiness threshold met ({readiness_score:.2f})"

        # Criterion 4: Has thought enough with reasonable understanding
        elif len(iterations) >= 3 and confidence_score > 0.6:
            ready = True
            reason = f"Sufficient thinking with reasonable understanding ({len(iterations)} iterations, {confidence_score:.2f} confidence)"

        return ready, readiness_score, reason

    def _compile_results(self, iterations: List[ThinkingIteration],
                        readiness_score: float,
                        stopping_reason: str,
                        user_input: str,
                        relevant_memories: List[Dict] = None,
                        associations: List[Dict] = None,
                        enhanced_inferences: List[str] = None) -> DeliberationResult:
        """
        Compile the results of deliberation
        NOW ENHANCED: Uses enhanced reasoner for response strategy
        """

        final_iteration = iterations[-1]

        # Extract key insights from all iterations
        key_insights = []

        # Prefer enhanced inferences
        if enhanced_inferences:
            key_insights.extend(enhanced_inferences[:5])

        # Add insights from iterations
        for iteration in iterations:
            # Handle both string inferences and Inference objects
            for inf in iteration.inferences[:2]:
                if isinstance(inf, str):
                    key_insights.append(inf)
                elif hasattr(inf, 'confidence') and inf.confidence > 0.6:
                    key_insights.append(inf.conclusion)

            # Important thoughts
            for thought in iteration.thoughts[:2]:
                if "approach" in thought.lower() or "understanding" in thought.lower():
                    key_insights.append(thought)

        # Deduplicate
        key_insights = list(dict.fromkeys(key_insights))[:5]

        # Determine response direction and strategy using enhanced reasoner
        if self.enhanced_reasoner:
            response_strategy = self.enhanced_reasoner.decide_response_strategy(
                user_input=user_input,
                inferences=enhanced_inferences or [],
                associations=associations or [],
                confidence=final_iteration.confidence
            )
            response_direction = response_strategy.get('approach', 'engage')
        else:
            # Fallback to original method
            response_direction = self._determine_response_direction(
                iterations=iterations,
                user_input=user_input
            )
            response_strategy = {'approach': response_direction}

        return DeliberationResult(
            total_iterations=len(iterations),
            thinking_iterations=iterations,
            final_confidence=final_iteration.confidence,
            readiness_score=readiness_score,
            stopping_reason=stopping_reason,
            key_insights=key_insights,
            response_direction=response_direction,
            associations=associations,
            enhanced_inferences=enhanced_inferences,
            response_strategy=response_strategy
        )

    def _determine_response_direction(self, iterations: List[ThinkingIteration],
                                     user_input: str) -> str:
        """
        Determine what direction the response should take
        Based on deliberation results
        """

        final_iteration = iterations[-1]

        # Check for strong inferences
        if final_iteration.inferences:
            confident_inferences = [inf for inf in final_iteration.inferences
                                   if inf.confidence > 0.7]
            if confident_inferences:
                return "share_insights"

        # Check for many gaps
        if len(final_iteration.information_gaps) > 3:
            return "ask_questions"

        # Check for emotional content
        if any("emotional" in t.lower() for t in final_iteration.thoughts):
            return "empathetic_response"

        # Check if understanding improved over iterations
        if len(iterations) > 1:
            confidence_delta = final_iteration.confidence - iterations[0].confidence
            if confidence_delta > 0.2:
                return "share_understanding"

        # Default: engage naturally
        return "natural_engagement"
