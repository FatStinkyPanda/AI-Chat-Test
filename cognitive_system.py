"""
Unified Cognitive System
Integrates all components into a coherent learning conversational AI
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import uuid
import os
from collections import deque

from brain_core import BrainCore, MemoryNode, CognitiveEdge, EdgeType, EmotionalValence
from semantic_processor import SemanticProcessor, ContextManager
from emotional_processor import EmotionalProcessor
from vector_memory import VectorMemory
from dynamic_responder import DynamicResponder
from reasoning_engine import ReasoningEngine
from deliberation_engine import DeliberationEngine
from enhanced_reasoner import EnhancedReasoner
from intelligent_responder import IntelligentResponder

# NEW ADVANCED SYSTEMS
from causal_discovery import CausalDiscoveryEngine
from multi_step_inference import MultiStepInferenceEngine
from conversation_predictor import ConversationPredictor, ConversationState
from attention_mechanism import AttentionMechanism
from knowledge_graph import KnowledgeGraph
from meta_learning import MetaLearningSystem
from transfer_learning import TransferLearningSystem


class CognitiveSystem:
    """
    The main cognitive system that orchestrates all components
    """

    def __init__(self, memory_dir: str = "./brain_memory",
                 state_file: str = "./brain_state.json"):
        """Initialize the cognitive system"""
        print("Initializing Cognitive System...")

        # Core components
        self.brain = BrainCore()
        self.semantic_processor = SemanticProcessor()
        self.emotional_processor = EmotionalProcessor()
        self.vector_memory = VectorMemory(persist_directory=memory_dir)
        self.context_manager = ContextManager(self.semantic_processor)
        self.responder = DynamicResponder()
        self.reasoning_engine = ReasoningEngine()

        # Enhanced learning and reasoning (NEW!)
        self.enhanced_reasoner = EnhancedReasoner()
        self.intelligent_responder = IntelligentResponder()

        # Deliberation engine with enhanced reasoning
        self.deliberation_engine = DeliberationEngine(
            self.reasoning_engine,
            self.enhanced_reasoner
        )

        # ADVANCED COGNITIVE SYSTEMS (NEWEST!)
        print("Initializing advanced cognitive systems...")
        self.causal_engine = CausalDiscoveryEngine()
        self.inference_engine = MultiStepInferenceEngine()
        self.conversation_predictor = ConversationPredictor()
        self.attention_mechanism = AttentionMechanism(embedding_dim=384)
        self.knowledge_graph = KnowledgeGraph()
        self.meta_learning = MetaLearningSystem()
        self.transfer_learning = TransferLearningSystem()
        print("Advanced systems initialized.")

        # Conversation state tracking
        self.current_conversation_state: Optional[ConversationState] = None
        self.session_start_time = datetime.now().timestamp()

        # System state
        self.state_file = state_file
        self.conversation_id = str(uuid.uuid4())
        self.turn_count = 0

        # Learning parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 0.7
        self.context_window = 10

        # Response generation state
        self.response_patterns: Dict[str, List[str]] = {}
        self.learned_associations: Dict[str, List[str]] = {}

        # Load previous state if exists
        if os.path.exists(state_file):
            try:
                self.brain.load_state(state_file)
                print(f"Loaded previous brain state: {len(self.brain.nodes)} nodes")
            except Exception as e:
                print(f"Could not load previous state: {e}")

        # Load advanced system states
        self._load_advanced_states()

        print("Cognitive System initialized.")
        print(f"Memory statistics: {self.vector_memory.get_memory_stats()}")

    def process_input(self, user_input: str, user_id: str = "user") -> str:
        """
        Process user input through the cognitive pipeline
        """
        self.turn_count += 1
        timestamp = datetime.now()

        # 1. PERCEPTION: Encode input
        input_embedding = self.semantic_processor.encode(user_input)
        input_emotions = self.emotional_processor.analyze_emotion(user_input)

        # 2. MEMORY CREATION: Create memory node
        node_id = f"input_{self.turn_count}_{uuid.uuid4().hex[:8]}"
        memory_node = MemoryNode(
            id=node_id,
            content=user_input,
            embedding=input_embedding,
            timestamp=timestamp,
            emotional_valence=input_emotions,
            metadata={
                'role': 'user',
                'user_id': user_id,
                'turn': self.turn_count,
                'conversation_id': self.conversation_id
            }
        )
        self.brain.add_node(memory_node)

        # 3. STORE IN VECTOR MEMORY
        emotion_dict = {k.value: v for k, v in input_emotions.items()}
        self.vector_memory.store_episode(
            content=user_input,
            embedding=input_embedding,
            metadata={
                'role': 'user',
                'turn': self.turn_count,
                'emotions': emotion_dict
            },
            memory_id=node_id
        )

        # 4. CONTEXT UPDATE
        self.context_manager.add_turn('user', user_input, {'timestamp': timestamp})

        # 5. RETRIEVAL: Find relevant memories WITH ATTENTION
        # Use attention mechanism to intelligently select relevant information
        relevant_memories = self._retrieve_relevant_context_with_attention(
            input_embedding, user_input
        )

        # 6. KNOWLEDGE EXTRACTION: Extract entities, relationships, and facts
        knowledge_extracted = self.knowledge_graph.extract_knowledge(
            user_input, node_id
        )

        # 7. CAUSAL DISCOVERY: Discover causal relationships
        causal_relations = self.causal_engine.discover_causal_relations(
            user_input, node_id, timestamp.timestamp()
        )

        # 8. ASSOCIATION: Create cognitive edges (enhanced with causal edges)
        self._create_associations_enhanced(
            node_id, relevant_memories, input_emotions, causal_relations
        )

        # 9. ACTIVATION: Spread activation in network
        relevant_node_ids = [m['id'] for m in relevant_memories[:5]]
        self.brain.activate_network([node_id] + relevant_node_ids, activation_spread=2)

        # 10. REASONING: Multi-hop reasoning for deeper understanding
        reasoning_paths = self._perform_reasoning(node_id)

        # 11. INFERENCE: Generate initial inferences from memories and context
        inferences = self._perform_inference(relevant_memories, user_input)

        # 12. MULTI-STEP INFERENCE: Build inference chains for deeper conclusions
        inference_chains = self._perform_multi_step_inference(relevant_memories, user_input)

        # 13. GAP DETECTION: Identify initial information gaps
        information_gaps = self._identify_gaps(relevant_memories, user_input)

        # 14. CONVERSATION PREDICTION: Predict trajectory and get anticipatory insights
        topics = self.semantic_processor.extract_keywords(user_input, top_n=5)
        dominant_emotion_tuple = self.emotional_processor.get_dominant_emotion(input_emotions)
        dominant_emotion = dominant_emotion_tuple[0].value if dominant_emotion_tuple else "neutral"
        intent = self._detect_intent(user_input)

        # Update conversation state
        self.current_conversation_state = self.conversation_predictor.update_state(
            previous_state=self.current_conversation_state,
            new_topics=topics,
            new_intent=intent,
            new_emotion=dominant_emotion,
            user_text=user_input
        )

        # Get predictive insights
        trajectory_prediction = self.conversation_predictor.predict_trajectory(
            self.current_conversation_state,
            horizon=5
        )

        anticipatory_insights = self.conversation_predictor.get_anticipatory_insights(
            self.current_conversation_state
        )

        # 15. META-LEARNING: Select optimal learning strategy
        learning_context = f"{intent}_{dominant_emotion}" if intent and dominant_emotion else "general"
        learning_strategy = self.meta_learning.select_learning_strategy(
            information=user_input,
            context=learning_context,
            learning_goal=None
        )

        print(f"\n[Meta-Learning] Selected strategy: {learning_strategy.name}")
        print(f"[Conversation Prediction] Ending probability: {trajectory_prediction.conversation_ending_probability:.2f}")
        print(f"[Strategy] Recommended: {anticipatory_insights.get('recommended_strategy', 'maintain_flow')}")

        # 16. DELIBERATION: Think iteratively and decide when ready to respond
        # This is where the AI truly thinks for itself
        print(f"\n{'='*60}")
        print("AI is thinking about what to say...")
        print(f"{'='*60}")

        deliberation_result = self.deliberation_engine.deliberate(
            user_input=user_input,
            relevant_memories=relevant_memories,
            initial_inferences=inferences,
            initial_gaps=information_gaps,
            emotional_context=input_emotions
        )

        print(f"\n{'='*60}")
        print(f"AI has completed thinking and is ready to respond")
        print(f"Reason: {deliberation_result.stopping_reason}")
        print(f"Confidence: {deliberation_result.final_confidence:.2f}")
        print(f"{'='*60}\n")

        # 17. THOUGHT RECORDING: Save deliberation thoughts as memory
        thought_nodes = self._record_deliberation_thoughts(
            deliberation_result=deliberation_result,
            context_node_id=node_id
        )

        # 18. LEARNING: Update patterns and associations
        self._learn_patterns(user_input, relevant_memories)

        # 19. CONVERSATION PATTERN LEARNING
        if self.turn_count > 1 and self.current_conversation_state:
            # Get previous topic from current state
            prev_topics = self.current_conversation_state.current_topics if hasattr(self.current_conversation_state, 'current_topics') else []
            prev_intent = self.current_conversation_state.recent_intents[-2] if len(self.current_conversation_state.recent_intents) > 1 else ""
            prev_emotion = self.current_conversation_state.emotional_trajectory[-2] if len(self.current_conversation_state.emotional_trajectory) > 1 else ""

            if prev_topics or prev_intent:
                self.conversation_predictor.learn_transitions(
                    previous_topic=prev_topics[-1] if prev_topics else "",
                    new_topic=topics[0] if topics else "",
                    previous_intent=prev_intent,
                    new_intent=intent,
                    previous_emotion=prev_emotion,
                    new_emotion=dominant_emotion
                )

        # 20. GENERATION: Generate intelligent response based on deliberation
        response = self._generate_intelligent_response(
            user_input,
            input_embedding,
            input_emotions,
            relevant_memories,
            reasoning_paths,
            deliberation_result,  # Pass deliberation results
            anticipatory_insights  # Pass anticipatory insights
        )

        # 21. STORE RESPONSE
        self._store_response(response, input_embedding, input_emotions)

        # 22. META-LEARNING: Record learning experience
        retention_score = deliberation_result.final_confidence
        application_score = 0.8  # Simplified - could be calculated based on response quality
        speed_score = 1.0 - (deliberation_result.total_iterations / 10.0)  # Fewer iterations = faster

        self.meta_learning.record_learning_experience(
            context=learning_context,
            strategy=learning_strategy,
            information=user_input,
            success=deliberation_result.final_confidence > 0.7,
            retention_score=retention_score,
            application_score=application_score,
            speed_score=speed_score
        )

        # 23. LEARNING: Learn from this conversation turn (ENHANCED!)
        self.enhanced_reasoner.learn_from_conversation(
            user_input=user_input,
            ai_response=response,
            memories=relevant_memories
        )

        # 24. MEMORY CONSOLIDATION
        self.brain.decay_activations(rate=0.05)
        self.brain.update_working_memory(node_id)

        # 25. PERIODIC SAVE (enhanced with new systems)
        if self.turn_count % 10 == 0:
            self._save_state()

        # 26. SESSION AGGREGATION (every 20 turns)
        if self.turn_count % 20 == 0:
            self._aggregate_session_patterns()

        return response

    def _retrieve_relevant_context(self, query_embedding: np.ndarray,
                                   query_text: str) -> List[Dict]:
        """Retrieve relevant context from memory"""
        # Hybrid retrieval from vector memory
        vector_results = self.vector_memory.hybrid_search(
            query_embedding,
            query_text,
            n_results=8
        )

        # Get graph-based associations
        working_memory = self.brain.get_working_memory_nodes()

        # Combine results
        all_results = []

        # Add vector search results
        for result in vector_results:
            all_results.append({
                'id': result['id'],
                'content': result['content'],
                'relevance': result.get('hybrid_score', 0.5),
                'source': 'vector',
                'metadata': result.get('metadata', {})
            })

        # Add working memory
        for node in working_memory[-5:]:
            all_results.append({
                'id': node.id,
                'content': node.content,
                'relevance': node.activation_level,
                'source': 'working_memory',
                'metadata': node.metadata
            })

        # Sort by relevance
        all_results.sort(key=lambda x: x['relevance'], reverse=True)

        return all_results[:10]

    def _create_associations(self, node_id: str, relevant_memories: List[Dict],
                           emotions: Dict[EmotionalValence, float]):
        """Create cognitive edges between memories"""
        for memory in relevant_memories[:5]:
            target_id = memory['id']

            if target_id == node_id:
                continue

            # Semantic edge (always create based on relevance)
            if memory['relevance'] > 0.5:
                semantic_edge = CognitiveEdge(
                    source_id=node_id,
                    target_id=target_id,
                    edge_type=EdgeType.SEMANTIC,
                    strength=memory['relevance']
                )
                self.brain.add_edge(semantic_edge)

            # Emotional edge if emotions match
            target_node = self.brain.get_node(target_id)
            if target_node and target_node.emotional_valence:
                emotional_similarity = self._emotional_similarity(
                    emotions,
                    target_node.emotional_valence
                )
                if emotional_similarity > 0.4:
                    emotional_edge = CognitiveEdge(
                        source_id=node_id,
                        target_id=target_id,
                        edge_type=EdgeType.EMOTIONAL,
                        strength=emotional_similarity
                    )
                    self.brain.add_edge(emotional_edge)

            # Temporal edge for recent memories
            if memory['source'] == 'working_memory':
                temporal_edge = CognitiveEdge(
                    source_id=node_id,
                    target_id=target_id,
                    edge_type=EdgeType.TEMPORAL,
                    strength=0.8
                )
                self.brain.add_edge(temporal_edge)

            # Contextual edge (co-occurrence in conversation)
            if memory.get('metadata', {}).get('conversation_id') == self.conversation_id:
                contextual_edge = CognitiveEdge(
                    source_id=node_id,
                    target_id=target_id,
                    edge_type=EdgeType.CONTEXTUAL,
                    strength=0.7
                )
                self.brain.add_edge(contextual_edge)

    def _emotional_similarity(self, emotions1: Dict[EmotionalValence, float],
                             emotions2: Dict[EmotionalValence, float]) -> float:
        """Calculate emotional similarity"""
        distance = self.emotional_processor.emotional_distance(emotions1, emotions2)
        similarity = max(0.0, 1.0 - distance / 2.0)
        return similarity

    def _perform_reasoning(self, start_node_id: str) -> List[List[MemoryNode]]:
        """Perform multi-hop reasoning across different edge types"""
        # Try different reasoning patterns
        reasoning_patterns = [
            [EdgeType.SEMANTIC, EdgeType.CAUSAL],
            [EdgeType.CONTEXTUAL, EdgeType.TEMPORAL],
            [EdgeType.EMOTIONAL, EdgeType.SEMANTIC],
        ]

        all_paths = []
        for pattern in reasoning_patterns:
            paths = self.brain.multi_hop_traverse(
                start_node_id,
                pattern,
                max_hops=2,
                min_strength=0.3
            )
            all_paths.extend(paths)

        return all_paths[:5]  # Return top 5 reasoning paths

    def _learn_patterns(self, user_input: str, relevant_memories: List[Dict]):
        """Learn patterns from interactions"""
        # Extract keywords
        keywords = self.semantic_processor.extract_keywords(user_input, top_n=5)

        # Update pattern associations
        for keyword in keywords:
            if keyword not in self.learned_associations:
                self.learned_associations[keyword] = []

            # Add contexts where this keyword appeared
            for memory in relevant_memories[:3]:
                if memory['content'] not in self.learned_associations[keyword]:
                    self.learned_associations[keyword].append(memory['content'])

            # Keep only recent associations
            self.learned_associations[keyword] = self.learned_associations[keyword][-20:]

    def _perform_inference(self, relevant_memories: List[Dict],
                          current_context: str) -> List:
        """
        Perform advanced inference using the reasoning engine
        This is where the AI truly thinks and makes connections
        """
        # Convert memories to format expected by reasoning engine
        memory_list = [
            {
                'content': m['content'],
                'id': m['id'],
                'metadata': m.get('metadata', {})
            }
            for m in relevant_memories
        ]

        # Perform inference
        inferences = self.reasoning_engine.perform_inference(
            memories=memory_list,
            current_context=current_context
        )

        return inferences

    def _identify_gaps(self, relevant_memories: List[Dict],
                      current_input: str) -> List:
        """
        Identify gaps in knowledge using the reasoning engine
        """
        memory_list = [
            {
                'content': m['content'],
                'id': m['id'],
                'metadata': m.get('metadata', {})
            }
            for m in relevant_memories
        ]

        # Identify information gaps
        gaps = self.reasoning_engine.identify_information_gaps(
            memories=memory_list,
            current_input=current_input
        )

        return gaps

    def _record_deliberation_thoughts(self, deliberation_result,
                                      context_node_id: str) -> List[str]:
        """
        Record thoughts from deliberation process as memory nodes
        This allows the AI to learn from its own thinking process
        """
        thought_node_ids = []

        # Record key insights from deliberation
        for i, insight in enumerate(deliberation_result.key_insights):
            thought_id = f"thought_{self.turn_count}_{i}_{uuid.uuid4().hex[:8]}"
            thought_embedding = self.semantic_processor.encode(insight)

            thought_node = MemoryNode(
                id=thought_id,
                content=insight,
                embedding=thought_embedding,
                timestamp=datetime.now(),
                emotional_valence={},
                metadata={
                    'type': 'deliberation_thought',
                    'role': 'internal',
                    'confidence': deliberation_result.final_confidence,
                    'iterations': deliberation_result.total_iterations,
                    'turn': self.turn_count,
                    'conversation_id': self.conversation_id
                }
            )
            self.brain.add_node(thought_node)

            # Store in vector memory (semantic memory for thoughts)
            self.vector_memory.store_semantic(
                content=insight,
                embedding=thought_embedding,
                metadata={
                    'type': 'deliberation_thought',
                    'role': 'internal',
                    'confidence': deliberation_result.final_confidence,
                    'iterations': deliberation_result.total_iterations,
                    'turn': self.turn_count
                },
                memory_id=thought_id
            )

            # Connect thought to current context
            context_edge = CognitiveEdge(
                source_id=context_node_id,
                target_id=thought_id,
                edge_type=EdgeType.CONTEXTUAL,
                strength=deliberation_result.final_confidence
            )
            self.brain.add_edge(context_edge)

            thought_node_ids.append(thought_id)

        # Record the deliberation process summary
        if deliberation_result.total_iterations > 1:
            summary = (f"Deliberated for {deliberation_result.total_iterations} iterations. "
                      f"{deliberation_result.stopping_reason}")

            summary_id = f"deliberation_{self.turn_count}_{uuid.uuid4().hex[:8]}"
            summary_embedding = self.semantic_processor.encode(summary)

            summary_node = MemoryNode(
                id=summary_id,
                content=summary,
                embedding=summary_embedding,
                timestamp=datetime.now(),
                emotional_valence={},
                metadata={
                    'type': 'deliberation_summary',
                    'role': 'internal',
                    'readiness_score': deliberation_result.readiness_score,
                    'turn': self.turn_count,
                    'conversation_id': self.conversation_id
                }
            )
            self.brain.add_node(summary_node)

            # Store in semantic memory
            self.vector_memory.store_semantic(
                content=summary,
                embedding=summary_embedding,
                metadata={
                    'type': 'deliberation_summary',
                    'role': 'internal',
                    'readiness_score': deliberation_result.readiness_score,
                    'turn': self.turn_count
                },
                memory_id=summary_id
            )

            # Connect to context
            summary_edge = CognitiveEdge(
                source_id=context_node_id,
                target_id=summary_id,
                edge_type=EdgeType.PROCEDURAL,
                strength=0.7
            )
            self.brain.add_edge(summary_edge)

            thought_node_ids.append(summary_id)

        return thought_node_ids


    def _store_response(self, response: str, input_embedding: np.ndarray,
                       input_emotions: Dict[EmotionalValence, float]):
        """Store the assistant's response in memory"""
        response_embedding = self.semantic_processor.encode(response)
        response_emotions = self.emotional_processor.analyze_emotion(response)

        # Create response node
        response_node_id = f"response_{self.turn_count}_{uuid.uuid4().hex[:8]}"
        response_node = MemoryNode(
            id=response_node_id,
            content=response,
            embedding=response_embedding,
            timestamp=datetime.now(),
            emotional_valence=response_emotions,
            metadata={
                'role': 'assistant',
                'turn': self.turn_count,
                'conversation_id': self.conversation_id
            }
        )
        self.brain.add_node(response_node)

        # Store in vector memory
        self.vector_memory.store_episode(
            content=response,
            embedding=response_embedding,
            metadata={
                'role': 'assistant',
                'turn': self.turn_count,
                'emotions': {k.value: v for k, v in response_emotions.items()}
            },
            memory_id=response_node_id
        )

        # Update context
        self.context_manager.add_turn('assistant', response)

        # Update working memory
        self.brain.update_working_memory(response_node_id)

    def _save_state(self):
        """Save brain state to disk"""
        try:
            self.brain.save_state(self.state_file)
            print(f"Brain state saved ({len(self.brain.nodes)} nodes)")
        except Exception as e:
            print(f"Error saving state: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        brain_stats = self.brain.get_statistics()
        memory_stats = self.vector_memory.get_memory_stats()

        return {
            'conversation_turns': self.turn_count,
            'conversation_id': self.conversation_id,
            'brain_stats': brain_stats,
            'memory_stats': memory_stats,
            'learned_patterns': len(self.learned_associations)
        }

    def reset_conversation(self):
        """Start a new conversation while keeping long-term memory"""
        self.conversation_id = str(uuid.uuid4())
        self.turn_count = 0
        self.brain.working_memory = []
        self.context_manager = ContextManager(self.semantic_processor)
        print("Started new conversation")

    def shutdown(self):
        """Gracefully shutdown and save state"""
        print("Shutting down cognitive system...")
        self._save_state()
        print("State saved. Goodbye!")

    # ========== NEW ADVANCED METHODS ==========

    def _retrieve_relevant_context_with_attention(
        self,
        query_embedding: np.ndarray,
        query_text: str
    ) -> List[Dict]:
        """Retrieve relevant context using attention mechanism"""
        # First get candidates using standard retrieval
        candidates = self._retrieve_relevant_context(query_embedding, query_text)

        if not candidates:
            return []

        # Prepare for attention mechanism
        items = []
        embeddings = []

        for candidate in candidates:
            # Ensure timestamp is a float
            timestamp_raw = candidate.get('metadata', {}).get('timestamp', datetime.now().timestamp())
            if isinstance(timestamp_raw, str):
                try:
                    # Try parsing ISO format or float string
                    timestamp = datetime.fromisoformat(timestamp_raw).timestamp()
                except:
                    timestamp = float(timestamp_raw) if timestamp_raw.replace('.', '').isdigit() else datetime.now().timestamp()
            elif isinstance(timestamp_raw, datetime):
                timestamp = timestamp_raw.timestamp()
            else:
                timestamp = float(timestamp_raw) if timestamp_raw else datetime.now().timestamp()

            items.append({
                'id': candidate['id'],
                'content': candidate['content'],
                'timestamp': timestamp,
                'importance': candidate.get('relevance', 0.5),
                'activation_level': candidate.get('relevance', 0.5)
            })

            # Get embedding (reuse or encode)
            node = self.brain.get_node(candidate['id'])
            if node and node.embedding is not None:
                embeddings.append(node.embedding)
            else:
                embeddings.append(self.semantic_processor.encode(candidate['content']))

        # Apply attention mechanism
        attention_scores = self.attention_mechanism.compute_attention(
            query_embedding=query_embedding,
            items=items,
            item_embeddings=embeddings,
            top_k=10,
            attention_mode="balanced"
        )

        # Convert back to standard format with attention weights
        results = []
        for att_score in attention_scores:
            # Find original candidate
            orig = next(c for c in candidates if c['id'] == att_score.item_id)
            orig['relevance'] = att_score.score
            orig['attention_weight'] = att_score.final_weight
            results.append(orig)

        return results

    def _create_associations_enhanced(
        self,
        node_id: str,
        relevant_memories: List[Dict],
        emotions: Dict[EmotionalValence, float],
        causal_relations: List
    ):
        """Create enhanced associations including causal edges"""
        # Standard associations
        self._create_associations(node_id, relevant_memories, emotions)

        # Add causal edges
        for causal_rel in causal_relations:
            # Find if we have nodes for cause and effect
            cause_content = causal_rel.cause.lower()
            effect_content = causal_rel.effect.lower()

            # Search for matching nodes
            for memory in relevant_memories[:10]:
                if cause_content in memory['content'].lower():
                    # Create causal edge
                    causal_edge = CognitiveEdge(
                        source_id=memory['id'],
                        target_id=node_id,
                        edge_type=EdgeType.CAUSAL,
                        strength=causal_rel.confidence
                    )
                    self.brain.add_edge(causal_edge)

    def _perform_multi_step_inference(
        self,
        relevant_memories: List[Dict],
        current_input: str
    ) -> List:
        """Perform multi-step inference to build inference chains"""

        # Convert memories to facts for inference engine
        for memory in relevant_memories[:10]:
            self.inference_engine.add_fact(
                fact_id=memory['id'],
                content=memory['content'],
                confidence=memory.get('relevance', 0.7)
            )

        # Try forward chaining
        premises = [
            {
                'id': m['id'],
                'content': m['content'],
                'confidence': m.get('relevance', 0.7)
            }
            for m in relevant_memories[:5]
        ]

        forward_chains = self.inference_engine.forward_chain(
            premises=premises,
            max_depth=3,
            min_confidence=0.5
        )

        return forward_chains

    def _detect_intent(self, user_input: str) -> str:
        """Detect user intent from input"""
        text_lower = user_input.lower()

        # Question intents
        if '?' in user_input:
            return 'question'

        # Emotional expressions
        emotion_words = ['feel', 'felt', 'feeling', 'emotion', 'happy', 'sad', 'angry', 'excited']
        if any(word in text_lower for word in emotion_words):
            return 'emotional_expression'

        # Information sharing
        sharing_words = ['i', 'my', 'me', 'today', 'yesterday', 'recently']
        if any(word in text_lower for word in sharing_words):
            return 'information_sharing'

        # Interest expression
        interest_words = ['interesting', 'curious', 'wonder', 'fascinated', 'tell me']
        if any(word in text_lower for word in interest_words):
            return 'interest_expression'

        # Gratitude
        if any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            return 'gratitude'

        # Greeting
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'

        # Farewell
        if any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'later']):
            return 'farewell'

        return 'general'

    def _aggregate_session_patterns(self):
        """Aggregate patterns from current session for transfer learning"""
        if self.turn_count < 5:
            return  # Too early

        # Gather session data - extract topics from conversation state
        topics = []
        if self.current_conversation_state:
            topics = self.current_conversation_state.current_topics[:20]

        # Also gather from topic transitions if available
        if hasattr(self.conversation_predictor, 'topic_transitions'):
            topics.extend(list(self.conversation_predictor.topic_transitions.keys())[:20])

        # Remove duplicates while preserving order
        seen = set()
        topics = [t for t in topics if not (t in seen or seen.add(t))][:20]

        patterns = []  # Could gather from enhanced reasoner

        # Extract entities and relationships from knowledge graph
        entities = [entity.name for entity in self.knowledge_graph.entities.values()]
        relationships = []
        for rel in list(self.knowledge_graph.relationships.values())[:20]:
            if rel.subject_id in self.knowledge_graph.entities and rel.object_id in self.knowledge_graph.entities:
                relationships.append(
                    f"{self.knowledge_graph.entities[rel.subject_id].name} {rel.predicate} {self.knowledge_graph.entities[rel.object_id].name}"
                )

        # Get insights from deliberation history
        insights = []
        if hasattr(self.deliberation_engine, 'deliberation_history'):
            for delib in self.deliberation_engine.deliberation_history[-5:]:
                insights.extend(delib.key_insights[:3])

        # Calculate session duration
        duration = datetime.now().timestamp() - self.session_start_time

        # Aggregate into transfer learning system
        session_summary = self.transfer_learning.aggregate_session_patterns(
            topics=topics,
            patterns=patterns,
            insights=insights,
            entities=entities[:20],
            relationships=relationships[:20],
            duration=duration,
            turn_count=self.turn_count
        )

        print(f"\n[Transfer Learning] Session aggregated: {session_summary.session_id}")
        print(f"[Transfer Learning] Learning quality: {session_summary.learning_quality:.2f}")

    def _load_advanced_states(self):
        """Load states for all advanced systems"""
        try:
            # Causal engine
            causal_state_file = "./causal_state.json"
            if os.path.exists(causal_state_file):
                self.causal_engine.load_state(causal_state_file)
                print(f"Loaded causal engine: {len(self.causal_engine.causal_relations)} relations")

            # Knowledge graph
            kg_state_file = "./knowledge_graph_state.json"
            if os.path.exists(kg_state_file):
                self.knowledge_graph.load_graph(kg_state_file)
                print(f"Loaded knowledge graph: {len(self.knowledge_graph.entities)} entities")

            # Conversation predictor
            conv_state_file = "./conversation_predictor_state.json"
            if os.path.exists(conv_state_file):
                self.conversation_predictor.load_state(conv_state_file)
                print(f"Loaded conversation predictor")

            # Attention mechanism
            attention_state_file = "./attention_state.json"
            if os.path.exists(attention_state_file):
                self.attention_mechanism.load_state(attention_state_file)
                print(f"Loaded attention mechanism")

            # Meta-learning
            meta_state_file = "./meta_learning_state.json"
            if os.path.exists(meta_state_file):
                self.meta_learning.load_state(meta_state_file)
                print(f"Loaded meta-learning system")

            # Transfer learning
            transfer_state_file = "./transfer_learning_state.json"
            if os.path.exists(transfer_state_file):
                self.transfer_learning.load_state(transfer_state_file)
                print(f"Loaded transfer learning system")

        except Exception as e:
            print(f"Error loading advanced states: {e}")

    def _save_state(self):
        """Enhanced save state - saves all systems"""
        try:
            # Original brain state
            self.brain.save_state(self.state_file)

            # Save all advanced systems
            self.causal_engine.save_state("./causal_state.json")
            self.knowledge_graph.save_graph("./knowledge_graph_state.json")
            self.conversation_predictor.save_state("./conversation_predictor_state.json")
            self.attention_mechanism.save_state("./attention_state.json")
            self.meta_learning.save_state("./meta_learning_state.json")
            self.transfer_learning.save_state("./transfer_learning_state.json")

            print(f"All systems saved ({len(self.brain.nodes)} nodes)")
        except Exception as e:
            print(f"Error saving state: {e}")

    def _generate_intelligent_response(
        self,
        user_input: str,
        input_embedding: np.ndarray,
        input_emotions: Dict[EmotionalValence, float],
        relevant_memories: List[Dict],
        reasoning_paths: List[List[MemoryNode]],
        deliberation_result=None,
        anticipatory_insights=None
    ) -> str:
        """
        Enhanced intelligent response generation with anticipatory insights
        """

        if deliberation_result and deliberation_result.response_strategy:
            # Use intelligent responder with enhanced understanding
            understanding = {
                'user_said': user_input,
                'keywords': self.semantic_processor.extract_keywords(user_input, top_n=5),
                'emotions': input_emotions,
                'memories': relevant_memories,
                'anticipatory_insights': anticipatory_insights or {}
            }

            response = self.intelligent_responder.generate_response(
                user_input=user_input,
                understanding=understanding,
                inferences=deliberation_result.enhanced_inferences or [],
                associations=deliberation_result.associations or [],
                strategy=deliberation_result.response_strategy,
                memories=relevant_memories
            )
        else:
            # Fallback to dynamic responder
            response = self.responder.construct_response(
                user_input=user_input,
                relevant_memories=relevant_memories,
                input_emotions=input_emotions,
                reasoning_paths=reasoning_paths,
                turn_count=self.turn_count,
                deliberation_result=deliberation_result
            )

        return response
