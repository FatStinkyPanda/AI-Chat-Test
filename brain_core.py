"""
Brain-Inspired Conversational AI Core
A novel multi-graph architecture with multiple edge types for human-like cognition
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from datetime import datetime
import json
from enum import Enum
import pickle
from collections import defaultdict


class EdgeType(Enum):
    """Different types of cognitive connections"""
    SEMANTIC = "semantic"  # Meaning similarity
    EMOTIONAL = "emotional"  # Emotional associations
    TEMPORAL = "temporal"  # Time-based connections
    CAUSAL = "causal"  # Cause-effect relationships
    CONTEXTUAL = "contextual"  # Co-occurrence patterns
    ANALOGICAL = "analogical"  # Metaphorical/analogical links
    PROCEDURAL = "procedural"  # Action-response patterns
    EPISODIC = "episodic"  # Memory episode links


class EmotionalValence(Enum):
    """Emotional dimensions"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


@dataclass
class MemoryNode:
    """A node in the cognitive graph"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    activation_level: float = 1.0
    access_count: int = 0
    emotional_valence: Dict[EmotionalValence, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def activate(self, strength: float = 0.1):
        """Increase activation level (simulating memory recall)"""
        self.activation_level = min(1.0, self.activation_level + strength)
        self.access_count += 1

    def decay(self, rate: float = 0.01):
        """Natural decay of activation over time"""
        self.activation_level = max(0.0, self.activation_level - rate)


@dataclass
class CognitiveEdge:
    """An edge connecting memory nodes"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)

    def reinforce(self, amount: float = 0.1):
        """Strengthen the connection (Hebbian learning)"""
        self.strength = min(1.0, self.strength + amount)
        self.last_reinforced = datetime.now()

    def weaken(self, amount: float = 0.01):
        """Weaken the connection over time"""
        self.strength = max(0.0, self.strength - amount)


class BrainCore:
    """
    The core cognitive architecture implementing a multi-graph memory system
    """

    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[CognitiveEdge] = []
        self.edge_index: Dict[EdgeType, List[CognitiveEdge]] = defaultdict(list)
        self.node_edges: Dict[str, List[CognitiveEdge]] = defaultdict(list)
        self.working_memory: List[str] = []  # Current conversation context
        self.working_memory_limit: int = 10
        self.global_context: Dict[str, Any] = {}

    def add_node(self, node: MemoryNode) -> str:
        """Add a new memory node to the graph"""
        self.nodes[node.id] = node
        return node.id

    def add_edge(self, edge: CognitiveEdge):
        """Add a new cognitive edge"""
        self.edges.append(edge)
        self.edge_index[edge.edge_type].append(edge)
        self.node_edges[edge.source_id].append(edge)
        self.node_edges[edge.target_id].append(edge)

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a node by ID"""
        return self.nodes.get(node_id)

    def find_edges(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[CognitiveEdge]:
        """Find all edges connected to a node, optionally filtered by type"""
        edges = self.node_edges.get(node_id, [])
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None,
                     min_strength: float = 0.0) -> List[Tuple[MemoryNode, CognitiveEdge]]:
        """Get neighboring nodes connected via edges"""
        neighbors = []
        edges = self.find_edges(node_id, edge_type)

        for edge in edges:
            if edge.strength < min_strength:
                continue

            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbor = self.get_node(neighbor_id)
            if neighbor:
                neighbors.append((neighbor, edge))

        return neighbors

    def multi_hop_traverse(self, start_node_id: str, edge_types: List[EdgeType],
                          max_hops: int = 3, min_strength: float = 0.3) -> List[List[MemoryNode]]:
        """
        Perform multi-hop traversal across different edge types
        This enables complex reasoning patterns
        """
        paths = []
        visited = set()

        def dfs(current_id: str, path: List[MemoryNode], remaining_types: List[EdgeType], hops: int):
            if hops >= max_hops or not remaining_types:
                if len(path) > 1:
                    paths.append(path.copy())
                return

            visited.add(current_id)
            current_edge_type = remaining_types[0]
            neighbors = self.get_neighbors(current_id, current_edge_type, min_strength)

            for neighbor, edge in neighbors:
                if neighbor.id not in visited:
                    path.append(neighbor)
                    dfs(neighbor.id, path, remaining_types[1:], hops + 1)
                    path.pop()

            visited.remove(current_id)

        start_node = self.get_node(start_node_id)
        if start_node:
            dfs(start_node_id, [start_node], edge_types, 0)

        return paths

    def activate_network(self, node_ids: List[str], activation_spread: int = 2):
        """
        Spreading activation across the network
        Simulates how thinking about one concept activates related concepts
        """
        activation_wave = {node_id: 1.0 for node_id in node_ids}

        for depth in range(activation_spread):
            next_wave = {}
            decay_factor = 0.5 ** (depth + 1)

            for node_id, activation in activation_wave.items():
                node = self.get_node(node_id)
                if node:
                    node.activate(activation * 0.5)

                # Spread to neighbors
                neighbors = self.get_neighbors(node_id)
                for neighbor, edge in neighbors:
                    spread_activation = activation * edge.strength * decay_factor
                    next_wave[neighbor.id] = max(
                        next_wave.get(neighbor.id, 0),
                        spread_activation
                    )

            activation_wave = next_wave

    def decay_activations(self, rate: float = 0.01):
        """Natural decay of all node activations"""
        for node in self.nodes.values():
            node.decay(rate)

    def update_working_memory(self, node_id: str):
        """Update working memory with recent context"""
        self.working_memory.append(node_id)
        if len(self.working_memory) > self.working_memory_limit:
            self.working_memory.pop(0)

    def get_working_memory_nodes(self) -> List[MemoryNode]:
        """Retrieve nodes in working memory"""
        return [self.get_node(nid) for nid in self.working_memory if self.get_node(nid)]

    def reinforce_path(self, node_ids: List[str], edge_type: EdgeType, strength: float = 0.1):
        """Reinforce connections along a path (Hebbian learning)"""
        for i in range(len(node_ids) - 1):
            # Find existing edge or create new one
            existing_edge = None
            for edge in self.find_edges(node_ids[i], edge_type):
                if edge.target_id == node_ids[i+1] or edge.source_id == node_ids[i+1]:
                    existing_edge = edge
                    break

            if existing_edge:
                existing_edge.reinforce(strength)
            else:
                new_edge = CognitiveEdge(
                    source_id=node_ids[i],
                    target_id=node_ids[i+1],
                    edge_type=edge_type,
                    strength=strength
                )
                self.add_edge(new_edge)

    def save_state(self, filepath: str):
        """Persist the brain state to disk"""
        state = {
            'nodes': {nid: {
                'id': n.id,
                'content': n.content,
                'embedding': n.embedding.tolist() if n.embedding is not None else None,
                'timestamp': n.timestamp.isoformat(),
                'activation_level': n.activation_level,
                'access_count': n.access_count,
                'emotional_valence': {k.value: v for k, v in n.emotional_valence.items()},
                'metadata': n.metadata
            } for nid, n in self.nodes.items()},
            'edges': [{
                'source_id': e.source_id,
                'target_id': e.target_id,
                'edge_type': e.edge_type.value,
                'strength': e.strength,
                'metadata': e.metadata,
                'created_at': e.created_at.isoformat(),
                'last_reinforced': e.last_reinforced.isoformat()
            } for e in self.edges],
            'working_memory': self.working_memory,
            'global_context': self.global_context
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load brain state from disk"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Restore nodes
        self.nodes = {}
        for nid, ndata in state['nodes'].items():
            node = MemoryNode(
                id=ndata['id'],
                content=ndata['content'],
                embedding=np.array(ndata['embedding']) if ndata['embedding'] else None,
                timestamp=datetime.fromisoformat(ndata['timestamp']),
                activation_level=ndata['activation_level'],
                access_count=ndata['access_count'],
                emotional_valence={EmotionalValence(k): v for k, v in ndata['emotional_valence'].items()},
                metadata=ndata['metadata']
            )
            self.nodes[nid] = node

        # Restore edges
        self.edges = []
        self.edge_index = defaultdict(list)
        self.node_edges = defaultdict(list)

        for edata in state['edges']:
            edge = CognitiveEdge(
                source_id=edata['source_id'],
                target_id=edata['target_id'],
                edge_type=EdgeType(edata['edge_type']),
                strength=edata['strength'],
                metadata=edata['metadata'],
                created_at=datetime.fromisoformat(edata['created_at']),
                last_reinforced=datetime.fromisoformat(edata['last_reinforced'])
            )
            self.add_edge(edge)

        self.working_memory = state['working_memory']
        self.global_context = state['global_context']

    def get_statistics(self) -> Dict[str, Any]:
        """Get brain statistics"""
        edge_counts = defaultdict(int)
        for edge in self.edges:
            edge_counts[edge.edge_type.value] += 1

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'edge_type_counts': dict(edge_counts),
            'working_memory_size': len(self.working_memory),
            'average_activation': np.mean([n.activation_level for n in self.nodes.values()]) if self.nodes else 0,
            'most_accessed_nodes': sorted(
                [(n.id, n.access_count, n.content[:50]) for n in self.nodes.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
