"""
Unified Cognitive System
Integrates all components into a coherent learning conversational AI
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import uuid
import os

from brain_core import BrainCore, MemoryNode, CognitiveEdge, EdgeType, EmotionalValence
from semantic_processor import SemanticProcessor, ContextManager
from emotional_processor import EmotionalProcessor
from vector_memory import VectorMemory
from dynamic_responder import DynamicResponder
from reasoning_engine import ReasoningEngine


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

        # 5. RETRIEVAL: Find relevant memories
        relevant_memories = self._retrieve_relevant_context(input_embedding, user_input)

        # 6. ASSOCIATION: Create cognitive edges
        self._create_associations(node_id, relevant_memories, input_emotions)

        # 7. ACTIVATION: Spread activation in network
        relevant_node_ids = [m['id'] for m in relevant_memories[:5]]
        self.brain.activate_network([node_id] + relevant_node_ids, activation_spread=2)

        # 8. REASONING: Multi-hop reasoning for deeper understanding
        reasoning_paths = self._perform_reasoning(node_id)

        # 8.5 INFERENCE: Generate inferences from memories and context
        inferences = self._perform_inference(relevant_memories, user_input)

        # 8.6 GAP DETECTION: Identify information gaps
        information_gaps = self._identify_gaps(relevant_memories, user_input)

        # 8.7 THOUGHT GENERATION & RECORDING: Create and save thoughts as memory
        thought_nodes = self._generate_and_record_thoughts(
            user_input,
            relevant_memories,
            inferences,
            information_gaps,
            node_id
        )

        # 9. LEARNING: Update patterns and associations
        self._learn_patterns(user_input, relevant_memories)

        # 10. GENERATION: Generate response
        response = self._generate_response(
            user_input,
            input_embedding,
            input_emotions,
            relevant_memories,
            reasoning_paths
        )

        # 11. STORE RESPONSE
        self._store_response(response, input_embedding, input_emotions)

        # 12. MEMORY CONSOLIDATION
        self.brain.decay_activations(rate=0.05)
        self.brain.update_working_memory(node_id)

        # 13. PERIODIC SAVE
        if self.turn_count % 10 == 0:
            self._save_state()

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

    def _generate_and_record_thoughts(self, user_input: str,
                                      relevant_memories: List[Dict],
                                      inferences: List,
                                      information_gaps: List,
                                      context_node_id: str) -> List[str]:
        """
        Generate thoughts from inferences and save them as memory nodes
        This allows the AI to learn from its own thinking
        """
        thought_node_ids = []

        # Generate thought from inferences
        if inferences:
            memory_list = [
                {
                    'content': m['content'],
                    'id': m['id'],
                    'metadata': m.get('metadata', {})
                }
                for m in relevant_memories
            ]

            thought_text = self.reasoning_engine.generate_thought(
                context=user_input,
                memories=memory_list,
                inferences=inferences
            )

            if thought_text:
                # Create thought node
                thought_id = f"thought_{self.turn_count}_{uuid.uuid4().hex[:8]}"
                thought_embedding = self.semantic_processor.encode(thought_text)

                thought_node = MemoryNode(
                    id=thought_id,
                    content=thought_text,
                    embedding=thought_embedding,
                    timestamp=datetime.now(),
                    emotional_valence={},
                    metadata={
                        'type': 'thought',
                        'role': 'internal',
                        'reasoning_type': inferences[0].reasoning_type if inferences else 'general',
                        'confidence': inferences[0].confidence if inferences else 0.5,
                        'turn': self.turn_count,
                        'conversation_id': self.conversation_id
                    }
                )
                self.brain.add_node(thought_node)

                # Store in vector memory (semantic memory for thoughts)
                self.vector_memory.store_semantic(
                    content=thought_text,
                    embedding=thought_embedding,
                    metadata={
                        'type': 'thought',
                        'role': 'internal',
                        'reasoning_type': inferences[0].reasoning_type if inferences else 'general',
                        'confidence': inferences[0].confidence if inferences else 0.5,
                        'turn': self.turn_count
                    },
                    memory_id=thought_id
                )

                # Create edges from premises to thought (causal edges)
                if inferences and hasattr(inferences[0], 'premise_ids'):
                    for premise_id in inferences[0].premise_ids:
                        # Check if premise node exists
                        if self.brain.get_node(premise_id):
                            edge = CognitiveEdge(
                                source_id=premise_id,
                                target_id=thought_id,
                                edge_type=EdgeType.CAUSAL,
                                strength=inferences[0].confidence
                            )
                            self.brain.add_edge(edge)

                # Connect thought to current context
                context_edge = CognitiveEdge(
                    source_id=context_node_id,
                    target_id=thought_id,
                    edge_type=EdgeType.CONTEXTUAL,
                    strength=0.8
                )
                self.brain.add_edge(context_edge)

                thought_node_ids.append(thought_id)

        # Record information gaps as thoughts
        if information_gaps:
            for gap in information_gaps[:2]:  # Top 2 gaps
                gap_thought = f"Information gap: {gap.question}"
                gap_id = f"gap_{self.turn_count}_{uuid.uuid4().hex[:8]}"
                gap_embedding = self.semantic_processor.encode(gap_thought)

                gap_node = MemoryNode(
                    id=gap_id,
                    content=gap_thought,
                    embedding=gap_embedding,
                    timestamp=datetime.now(),
                    emotional_valence={},
                    metadata={
                        'type': 'information_gap',
                        'role': 'internal',
                        'topic': gap.topic,
                        'importance': gap.importance,
                        'turn': self.turn_count,
                        'conversation_id': self.conversation_id
                    }
                )
                self.brain.add_node(gap_node)

                # Store in vector memory
                self.vector_memory.store_semantic(
                    content=gap_thought,
                    embedding=gap_embedding,
                    metadata={
                        'type': 'information_gap',
                        'role': 'internal',
                        'topic': gap.topic,
                        'importance': gap.importance,
                        'turn': self.turn_count
                    },
                    memory_id=gap_id
                )

                # Connect to context
                gap_edge = CognitiveEdge(
                    source_id=context_node_id,
                    target_id=gap_id,
                    edge_type=EdgeType.CONTEXTUAL,
                    strength=gap.importance
                )
                self.brain.add_edge(gap_edge)

                thought_node_ids.append(gap_id)

        return thought_node_ids

    def _generate_response(self, user_input: str,
                          input_embedding: np.ndarray,
                          input_emotions: Dict[EmotionalValence, float],
                          relevant_memories: List[Dict],
                          reasoning_paths: List[List[MemoryNode]]) -> str:
        """
        Generate a response based on all cognitive processes
        Uses dynamic responder for natural, intelligent conversation
        """
        # Use the dynamic responder to construct a natural response
        response = self.responder.construct_response(
            user_input=user_input,
            relevant_memories=relevant_memories,
            input_emotions=input_emotions,
            reasoning_paths=reasoning_paths,
            turn_count=self.turn_count
        )

        return response

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
