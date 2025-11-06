"""
Knowledge Graph Construction System

This system builds and maintains a structured knowledge graph from conversations,
extracting entities, relationships, and facts to create an organized knowledge base.
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import json
import re


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    entity_id: str
    name: str
    entity_type: str  # person, place, thing, concept, event, etc.
    aliases: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    first_mentioned: float = field(default_factory=lambda: datetime.now().timestamp())
    last_mentioned: float = field(default_factory=lambda: datetime.now().timestamp())
    mention_count: int = 0

    def to_dict(self) -> Dict:
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'entity_type': self.entity_type,
            'aliases': list(self.aliases),
            'attributes': self.attributes,
            'confidence': self.confidence,
            'first_mentioned': self.first_mentioned,
            'last_mentioned': self.last_mentioned,
            'mention_count': self.mention_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        data['aliases'] = set(data.get('aliases', []))
        return cls(**data)


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    relationship_id: str
    subject_id: str  # Entity ID
    predicate: str  # Type of relationship
    object_id: str  # Entity ID
    confidence: float = 1.0
    bidirectional: bool = False
    evidence: List[str] = field(default_factory=list)  # Supporting memory IDs
    strength: float = 0.5
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            'relationship_id': self.relationship_id,
            'subject_id': self.subject_id,
            'predicate': self.predicate,
            'object_id': self.object_id,
            'confidence': self.confidence,
            'bidirectional': self.bidirectional,
            'evidence': self.evidence,
            'strength': self.strength,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Relationship':
        return cls(**data)


@dataclass
class Fact:
    """Represents a factual statement"""
    fact_id: str
    statement: str
    subject_ids: List[str]  # Related entities
    confidence: float
    source_memory_ids: List[str]
    fact_type: str  # attribute, state, event, definition, etc.
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    verified: bool = False

    def to_dict(self) -> Dict:
        return {
            'fact_id': self.fact_id,
            'statement': self.statement,
            'subject_ids': self.subject_ids,
            'confidence': self.confidence,
            'source_memory_ids': self.source_memory_ids,
            'fact_type': self.fact_type,
            'timestamp': self.timestamp,
            'verified': self.verified
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Fact':
        return cls(**data)


class KnowledgeGraph:
    """
    Constructs and maintains a knowledge graph from conversations.
    Extracts entities, relationships, and facts.
    """

    # Common relationship types
    RELATIONSHIP_TYPES = [
        'is_a', 'has_a', 'part_of', 'located_in', 'works_at', 'knows',
        'likes', 'dislikes', 'causes', 'prevents', 'similar_to',
        'opposite_of', 'created_by', 'used_for', 'made_of', 'owns',
        'member_of', 'leads_to', 'requires', 'enables'
    ]

    # Entity type patterns
    ENTITY_TYPE_PATTERNS = {
        'person': ['he', 'she', 'they', 'person', 'people', 'who', 'name', 'someone'],
        'place': ['where', 'location', 'city', 'country', 'area', 'region', 'in', 'at'],
        'concept': ['idea', 'concept', 'theory', 'principle', 'notion', 'belief'],
        'event': ['happened', 'occurred', 'event', 'incident', 'when'],
        'thing': ['object', 'item', 'thing', 'what', 'something'],
        'organization': ['company', 'organization', 'group', 'team', 'firm']
    }

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.facts: Dict[str, Fact] = {}

        # Indices for fast lookup
        self.entity_by_name: Dict[str, str] = {}  # name -> entity_id
        self.relationships_by_subject: Dict[str, Set[str]] = defaultdict(set)
        self.relationships_by_object: Dict[str, Set[str]] = defaultdict(set)
        self.facts_by_entity: Dict[str, Set[str]] = defaultdict(set)

        # Statistics
        self.extraction_count = 0
        self.entity_id_counter = 0
        self.relationship_id_counter = 0
        self.fact_id_counter = 0

    def extract_knowledge(self, text: str, memory_id: str) -> Dict:
        """
        Extract entities, relationships, and facts from text.
        """
        self.extraction_count += 1

        extracted = {
            'entities': [],
            'relationships': [],
            'facts': []
        }

        # Extract entities
        entities = self._extract_entities(text, memory_id)
        extracted['entities'] = entities

        # Extract relationships between entities
        if len(entities) >= 2:
            relationships = self._extract_relationships(text, entities, memory_id)
            extracted['relationships'] = relationships

        # Extract facts
        facts = self._extract_facts(text, entities, memory_id)
        extracted['facts'] = facts

        return extracted

    def _extract_entities(self, text: str, memory_id: str) -> List[Entity]:
        """Extract entities from text"""
        entities = []

        # Simple entity extraction (could be enhanced with NLP)

        # 1. Extract capitalized words/phrases (likely proper nouns)
        capitalized_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        matches = re.findall(capitalized_pattern, text)

        for match in matches:
            # Skip common words that happen to be capitalized
            if match.lower() in ['i', 'the', 'a', 'an', 'this', 'that']:
                continue

            # Check if entity exists
            entity_id = self._find_or_create_entity(
                name=match,
                entity_type=self._infer_entity_type(match, text),
                memory_id=memory_id
            )

            entity = self.entities[entity_id]
            entities.append(entity)

        # 2. Extract quoted phrases (often refer to specific things)
        quoted_pattern = r'"([^"]+)"'
        quotes = re.findall(quoted_pattern, text)

        for quote in quotes:
            if len(quote.split()) <= 5:  # Short quotes might be entity names
                entity_id = self._find_or_create_entity(
                    name=quote,
                    entity_type='thing',
                    memory_id=memory_id
                )
                entity = self.entities[entity_id]
                entities.append(entity)

        # 3. Extract pronouns with references
        # "my X", "your X", "their X" patterns
        possessive_pattern = r'\b(?:my|your|his|her|their|our)\s+([a-z]+(?:\s+[a-z]+)*)'
        possessives = re.findall(possessive_pattern, text.lower())

        for poss in possessives:
            if len(poss.split()) <= 3:
                entity_id = self._find_or_create_entity(
                    name=poss,
                    entity_type=self._infer_entity_type(poss, text),
                    memory_id=memory_id
                )
                entity = self.entities[entity_id]
                entities.append(entity)

        return entities

    def _infer_entity_type(self, entity_name: str, context: str) -> str:
        """Infer entity type from name and context"""
        entity_lower = entity_name.lower()
        context_lower = context.lower()

        # Check patterns
        for entity_type, patterns in self.ENTITY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if pattern in entity_lower or pattern in context_lower:
                    return entity_type

        # Default
        if entity_name[0].isupper():
            return 'proper_noun'

        return 'concept'

    def _find_or_create_entity(
        self,
        name: str,
        entity_type: str,
        memory_id: str
    ) -> str:
        """Find existing entity or create new one"""

        # Normalize name
        normalized_name = name.lower().strip()

        # Check if exists
        if normalized_name in self.entity_by_name:
            entity_id = self.entity_by_name[normalized_name]
            entity = self.entities[entity_id]

            # Update
            entity.mention_count += 1
            entity.last_mentioned = datetime.now().timestamp()

            return entity_id

        # Create new entity
        entity_id = f"entity_{self.entity_id_counter}"
        self.entity_id_counter += 1

        entity = Entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            aliases={normalized_name},
            mention_count=1
        )

        self.entities[entity_id] = entity
        self.entity_by_name[normalized_name] = entity_id

        return entity_id

    def _extract_relationships(
        self,
        text: str,
        entities: List[Entity],
        memory_id: str
    ) -> List[Relationship]:
        """Extract relationships between entities"""
        relationships = []

        if len(entities) < 2:
            return relationships

        # Pattern-based relationship extraction
        text_lower = text.lower()

        # Try each relationship type pattern
        for rel_type in self.RELATIONSHIP_TYPES:
            # Create patterns for this relationship type
            patterns = self._get_relationship_patterns(rel_type)

            for pattern in patterns:
                # Try to match with entity pairs
                for i, entity1 in enumerate(entities):
                    for entity2 in entities[i+1:]:
                        # Check if pattern connects these entities
                        if self._matches_relationship_pattern(
                            text_lower, entity1.name.lower(),
                            entity2.name.lower(), pattern
                        ):
                            # Create or update relationship
                            rel_id = self._find_or_create_relationship(
                                subject_id=entity1.entity_id,
                                predicate=rel_type,
                                object_id=entity2.entity_id,
                                memory_id=memory_id
                            )

                            relationship = self.relationships[rel_id]
                            relationships.append(relationship)

        return relationships

    def _get_relationship_patterns(self, rel_type: str) -> List[str]:
        """Get linguistic patterns for relationship type"""
        patterns = {
            'is_a': ['is a', 'is an', 'are', 'am'],
            'has_a': ['has', 'have', 'owns', 'possess'],
            'part_of': ['part of', 'belongs to', 'within'],
            'located_in': ['in', 'at', 'located in', 'situated in'],
            'works_at': ['works at', 'employed by', 'working for'],
            'knows': ['knows', 'familiar with', 'acquainted with'],
            'likes': ['likes', 'loves', 'enjoys', 'appreciates'],
            'dislikes': ['dislikes', 'hates', 'avoids'],
            'causes': ['causes', 'leads to', 'results in'],
            'similar_to': ['similar to', 'like', 'resembles'],
            'used_for': ['used for', 'serves to', 'helps with']
        }

        return patterns.get(rel_type, [rel_type])

    def _matches_relationship_pattern(
        self,
        text: str,
        entity1: str,
        entity2: str,
        pattern: str
    ) -> bool:
        """Check if pattern connects two entities in text"""
        # Simple check: entity1 [pattern] entity2
        regex = rf'\b{re.escape(entity1)}\b.*?\b{re.escape(pattern)}\b.*?\b{re.escape(entity2)}\b'
        if re.search(regex, text):
            return True

        # Reverse: entity2 [pattern] entity1
        regex = rf'\b{re.escape(entity2)}\b.*?\b{re.escape(pattern)}\b.*?\b{re.escape(entity1)}\b'
        if re.search(regex, text):
            return True

        return False

    def _find_or_create_relationship(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        memory_id: str
    ) -> str:
        """Find or create relationship"""

        # Check if exists
        for rel_id, rel in self.relationships.items():
            if (rel.subject_id == subject_id and
                rel.predicate == predicate and
                rel.object_id == object_id):

                # Update existing
                rel.strength = min(1.0, rel.strength + 0.1)
                rel.confidence = min(1.0, rel.confidence + 0.05)
                if memory_id not in rel.evidence:
                    rel.evidence.append(memory_id)
                rel.updated_at = datetime.now().timestamp()

                return rel_id

        # Create new
        rel_id = f"rel_{self.relationship_id_counter}"
        self.relationship_id_counter += 1

        relationship = Relationship(
            relationship_id=rel_id,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            evidence=[memory_id]
        )

        self.relationships[rel_id] = relationship

        # Update indices
        self.relationships_by_subject[subject_id].add(rel_id)
        self.relationships_by_object[object_id].add(rel_id)

        return rel_id

    def _extract_facts(
        self,
        text: str,
        entities: List[Entity],
        memory_id: str
    ) -> List[Fact]:
        """Extract factual statements"""
        facts = []

        # Split into sentences
        sentences = re.split(r'[.!?]', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Too short to be meaningful
                continue

            # Check if sentence contains factual indicators
            factual_indicators = [
                'is', 'are', 'was', 'were', 'has', 'have', 'can', 'will',
                'always', 'never', 'typically', 'usually', 'generally'
            ]

            if any(indicator in sentence.lower() for indicator in factual_indicators):
                # Find which entities this fact refers to
                related_entities = []
                for entity in entities:
                    if entity.name.lower() in sentence.lower():
                        related_entities.append(entity.entity_id)

                # Create fact
                fact_id = f"fact_{self.fact_id_counter}"
                self.fact_id_counter += 1

                fact = Fact(
                    fact_id=fact_id,
                    statement=sentence,
                    subject_ids=related_entities,
                    confidence=0.7,  # Base confidence
                    source_memory_ids=[memory_id],
                    fact_type=self._infer_fact_type(sentence)
                )

                self.facts[fact_id] = fact
                facts.append(fact)

                # Update indices
                for entity_id in related_entities:
                    self.facts_by_entity[entity_id].add(fact_id)

        return facts

    def _infer_fact_type(self, statement: str) -> str:
        """Infer type of fact"""
        statement_lower = statement.lower()

        if any(word in statement_lower for word in ['is', 'are', 'am']):
            return 'attribute'
        elif any(word in statement_lower for word in ['happened', 'occurred', 'took place']):
            return 'event'
        elif any(word in statement_lower for word in ['can', 'able to', 'capable']):
            return 'capability'
        elif any(word in statement_lower for word in ['always', 'never', 'typically']):
            return 'pattern'
        else:
            return 'general'

    def query_entity(self, entity_name: str) -> Optional[Dict]:
        """Query knowledge about an entity"""
        normalized = entity_name.lower().strip()

        if normalized not in self.entity_by_name:
            return None

        entity_id = self.entity_by_name[normalized]
        entity = self.entities[entity_id]

        # Get relationships
        outgoing_rels = []
        for rel_id in self.relationships_by_subject.get(entity_id, set()):
            rel = self.relationships[rel_id]
            outgoing_rels.append({
                'predicate': rel.predicate,
                'object': self.entities[rel.object_id].name,
                'confidence': rel.confidence
            })

        incoming_rels = []
        for rel_id in self.relationships_by_object.get(entity_id, set()):
            rel = self.relationships[rel_id]
            incoming_rels.append({
                'subject': self.entities[rel.subject_id].name,
                'predicate': rel.predicate,
                'confidence': rel.confidence
            })

        # Get facts
        related_facts = []
        for fact_id in self.facts_by_entity.get(entity_id, set()):
            fact = self.facts[fact_id]
            related_facts.append({
                'statement': fact.statement,
                'confidence': fact.confidence,
                'type': fact.fact_type
            })

        return {
            'entity': entity.to_dict(),
            'outgoing_relationships': outgoing_rels,
            'incoming_relationships': incoming_rels,
            'facts': related_facts
        }

    def find_path(
        self,
        start_entity_name: str,
        end_entity_name: str,
        max_depth: int = 4
    ) -> Optional[List[Dict]]:
        """Find path between two entities in the knowledge graph"""

        start_id = self.entity_by_name.get(start_entity_name.lower().strip())
        end_id = self.entity_by_name.get(end_entity_name.lower().strip())

        if not start_id or not end_id:
            return None

        # BFS
        queue = deque([(start_id, [])])
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()

            if current_id == end_id:
                return path

            if len(path) >= max_depth:
                continue

            # Explore outgoing relationships
            for rel_id in self.relationships_by_subject.get(current_id, set()):
                rel = self.relationships[rel_id]
                next_id = rel.object_id

                if next_id not in visited:
                    visited.add(next_id)
                    new_path = path + [{
                        'from': self.entities[current_id].name,
                        'relationship': rel.predicate,
                        'to': self.entities[next_id].name
                    }]
                    queue.append((next_id, new_path))

        return None

    def get_entity_neighborhood(
        self,
        entity_name: str,
        depth: int = 1
    ) -> Dict:
        """Get entities within N relationships of given entity"""

        entity_id = self.entity_by_name.get(entity_name.lower().strip())
        if not entity_id:
            return {}

        neighborhood = {
            'center': self.entities[entity_id].to_dict(),
            'neighbors': defaultdict(list)
        }

        visited = {entity_id}
        current_level = {entity_id}

        for level in range(depth):
            next_level = set()

            for current_id in current_level:
                # Outgoing
                for rel_id in self.relationships_by_subject.get(current_id, set()):
                    rel = self.relationships[rel_id]
                    neighbor_id = rel.object_id

                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_level.add(neighbor_id)

                        neighborhood['neighbors'][level + 1].append({
                            'entity': self.entities[neighbor_id].to_dict(),
                            'relationship': rel.predicate,
                            'direction': 'outgoing'
                        })

                # Incoming
                for rel_id in self.relationships_by_object.get(current_id, set()):
                    rel = self.relationships[rel_id]
                    neighbor_id = rel.subject_id

                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_level.add(neighbor_id)

                        neighborhood['neighbors'][level + 1].append({
                            'entity': self.entities[neighbor_id].to_dict(),
                            'relationship': rel.predicate,
                            'direction': 'incoming'
                        })

            current_level = next_level

        return dict(neighborhood)

    def save_graph(self, filepath: str):
        """Save knowledge graph to file"""
        state = {
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'relationships': {k: v.to_dict() for k, v in self.relationships.items()},
            'facts': {k: v.to_dict() for k, v in self.facts.items()},
            'entity_by_name': self.entity_by_name,
            'relationships_by_subject': {k: list(v) for k, v in self.relationships_by_subject.items()},
            'relationships_by_object': {k: list(v) for k, v in self.relationships_by_object.items()},
            'facts_by_entity': {k: list(v) for k, v in self.facts_by_entity.items()},
            'counters': {
                'entity': self.entity_id_counter,
                'relationship': self.relationship_id_counter,
                'fact': self.fact_id_counter
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_graph(self, filepath: str):
        """Load knowledge graph from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Load entities
            self.entities = {
                k: Entity.from_dict(v) for k, v in state.get('entities', {}).items()
            }

            # Load relationships
            self.relationships = {
                k: Relationship.from_dict(v) for k, v in state.get('relationships', {}).items()
            }

            # Load facts
            self.facts = {
                k: Fact.from_dict(v) for k, v in state.get('facts', {}).items()
            }

            # Load indices
            self.entity_by_name = state.get('entity_by_name', {})
            self.relationships_by_subject = defaultdict(
                set,
                {k: set(v) for k, v in state.get('relationships_by_subject', {}).items()}
            )
            self.relationships_by_object = defaultdict(
                set,
                {k: set(v) for k, v in state.get('relationships_by_object', {}).items()}
            )
            self.facts_by_entity = defaultdict(
                set,
                {k: set(v) for k, v in state.get('facts_by_entity', {}).items()}
            )

            # Load counters
            counters = state.get('counters', {})
            self.entity_id_counter = counters.get('entity', 0)
            self.relationship_id_counter = counters.get('relationship', 0)
            self.fact_id_counter = counters.get('fact', 0)

        except FileNotFoundError:
            pass

    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'total_facts': len(self.facts),
            'extractions_performed': self.extraction_count,
            'entity_types': self._count_by_type(self.entities, 'entity_type'),
            'relationship_types': self._count_by_type(self.relationships, 'predicate'),
            'fact_types': self._count_by_type(self.facts, 'fact_type'),
            'most_connected_entities': self._get_most_connected_entities(5),
            'most_mentioned_entities': self._get_most_mentioned_entities(5)
        }

    def _count_by_type(self, items: Dict, type_field: str) -> Dict:
        """Count items by type"""
        counts = defaultdict(int)
        for item in items.values():
            type_value = getattr(item, type_field, 'unknown')
            counts[type_value] += 1
        return dict(counts)

    def _get_most_connected_entities(self, top_k: int) -> List[Tuple[str, int]]:
        """Get entities with most relationships"""
        connection_counts = defaultdict(int)

        for entity_id in self.entities:
            count = (
                len(self.relationships_by_subject.get(entity_id, set())) +
                len(self.relationships_by_object.get(entity_id, set()))
            )
            if count > 0:
                connection_counts[self.entities[entity_id].name] = count

        return sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _get_most_mentioned_entities(self, top_k: int) -> List[Tuple[str, int]]:
        """Get most frequently mentioned entities"""
        mentions = [
            (entity.name, entity.mention_count)
            for entity in self.entities.values()
        ]

        return sorted(mentions, key=lambda x: x[1], reverse=True)[:top_k]
