"""
Vector Memory System using ChromaDB
Persistent semantic memory with fast retrieval
Falls back to in-memory storage if ChromaDB is unavailable
"""

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("WARNING: ChromaDB not available. Running in limited mode (no persistence).")
    print("To enable persistence: python install_chromadb.py")

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
import os
import json
from collections import defaultdict


class VectorMemory:
    """
    Manages semantic memory using ChromaDB for persistent vector storage
    Falls back to in-memory storage if ChromaDB unavailable
    """

    def __init__(self, persist_directory: str = "./brain_memory"):
        """Initialize vector memory (ChromaDB or fallback)"""
        self.persist_directory = persist_directory
        self.use_chromadb = CHROMADB_AVAILABLE

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        if self.use_chromadb:
            # Initialize ChromaDB with persistent storage
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))

            # Create collections for different memory types
            self.episodic_memory = self._get_or_create_collection("episodic_memory")
            self.semantic_memory = self._get_or_create_collection("semantic_memory")
            self.emotional_memory = self._get_or_create_collection("emotional_memory")
        else:
            # Fallback: in-memory storage
            self.episodic_memory = InMemoryCollection("episodic_memory")
            self.semantic_memory = InMemoryCollection("semantic_memory")
            self.emotional_memory = InMemoryCollection("emotional_memory")

    def _get_or_create_collection(self, name: str):
        """Get existing collection or create new one"""
        try:
            return self.client.get_collection(name=name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

    def store_episode(self, content: str, embedding: np.ndarray,
                     metadata: Optional[Dict[str, Any]] = None,
                     memory_id: Optional[str] = None) -> str:
        """
        Store an episodic memory (conversation turn, event)
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        metadata['timestamp'] = metadata.get('timestamp', datetime.now().isoformat())
        metadata['memory_type'] = 'episodic'

        self.episodic_memory.add(
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )

        return memory_id

    def store_semantic(self, content: str, embedding: np.ndarray,
                      metadata: Optional[Dict[str, Any]] = None,
                      memory_id: Optional[str] = None) -> str:
        """
        Store semantic memory (facts, knowledge)
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        metadata['timestamp'] = metadata.get('timestamp', datetime.now().isoformat())
        metadata['memory_type'] = 'semantic'

        self.semantic_memory.add(
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )

        return memory_id

    def store_emotional(self, content: str, embedding: np.ndarray,
                       emotional_data: Dict[str, float],
                       metadata: Optional[Dict[str, Any]] = None,
                       memory_id: Optional[str] = None) -> str:
        """
        Store emotional memory
        """
        if memory_id is None:
            memory_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        metadata['timestamp'] = metadata.get('timestamp', datetime.now().isoformat())
        metadata['memory_type'] = 'emotional'
        metadata.update({f'emotion_{k}': v for k, v in emotional_data.items()})

        self.emotional_memory.add(
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )

        return memory_id

    def retrieve_similar(self, query_embedding: np.ndarray,
                        collection_name: str = "episodic_memory",
                        n_results: int = 5,
                        where: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve similar memories from specified collection
        """
        collection = getattr(self, collection_name)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )

        # Format results
        memories = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                memory = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                memories.append(memory)

        return memories

    def retrieve_recent(self, collection_name: str = "episodic_memory",
                       n_results: int = 10) -> List[Dict]:
        """Retrieve most recent memories"""
        collection = getattr(self, collection_name)

        try:
            results = collection.get(
                limit=n_results,
                include=['embeddings', 'documents', 'metadatas']
            )

            memories = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    memory = {
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    memories.append(memory)

            # Sort by timestamp
            memories.sort(
                key=lambda x: x['metadata'].get('timestamp', ''),
                reverse=True
            )

            return memories[:n_results]

        except Exception as e:
            print(f"Error retrieving recent memories: {e}")
            return []

    def update_memory(self, memory_id: str, collection_name: str,
                     metadata_updates: Dict[str, Any]):
        """Update metadata of existing memory"""
        collection = getattr(self, collection_name)

        try:
            collection.update(
                ids=[memory_id],
                metadatas=[metadata_updates]
            )
        except Exception as e:
            print(f"Error updating memory: {e}")

    def delete_memory(self, memory_id: str, collection_name: str):
        """Delete a specific memory"""
        collection = getattr(self, collection_name)
        collection.delete(ids=[memory_id])

    def search_by_metadata(self, collection_name: str,
                          where: Dict,
                          n_results: int = 10) -> List[Dict]:
        """Search memories by metadata filters"""
        collection = getattr(self, collection_name)

        try:
            results = collection.get(
                where=where,
                limit=n_results,
                include=['embeddings', 'documents', 'metadatas']
            )

            memories = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    memory = {
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    memories.append(memory)

            return memories

        except Exception as e:
            print(f"Error searching by metadata: {e}")
            return []

    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about stored memories"""
        return {
            'episodic_count': self.episodic_memory.count(),
            'semantic_count': self.semantic_memory.count(),
            'emotional_count': self.emotional_memory.count(),
            'total_memories': (
                self.episodic_memory.count() +
                self.semantic_memory.count() +
                self.emotional_memory.count()
            )
        }

    def clear_collection(self, collection_name: str):
        """Clear all memories from a collection"""
        try:
            if self.use_chromadb:
                self.client.delete_collection(name=collection_name)
                setattr(self, collection_name, self._get_or_create_collection(collection_name))
            else:
                collection = getattr(self, collection_name)
                collection.data = {}
        except Exception as e:
            print(f"Error clearing collection: {e}")

    def hybrid_search(self, query_embedding: np.ndarray,
                     query_text: str,
                     n_results: int = 5,
                     vector_weight: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining vector similarity and keyword matching
        """
        # Get vector search results
        vector_results = self.retrieve_similar(
            query_embedding,
            "episodic_memory",
            n_results * 2
        )

        # Simple keyword scoring
        query_keywords = set(query_text.lower().split())

        scored_results = []
        for result in vector_results:
            content_keywords = set(result['content'].lower().split())
            keyword_overlap = len(query_keywords & content_keywords) / max(len(query_keywords), 1)

            # Combine scores
            vector_score = 1.0 - result['distance'] if result['distance'] else 0.0
            hybrid_score = (vector_weight * vector_score +
                          (1 - vector_weight) * keyword_overlap)

            result['hybrid_score'] = hybrid_score
            scored_results.append(result)

        # Sort by hybrid score
        scored_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return scored_results[:n_results]


class InMemoryCollection:
    """
    Fallback in-memory collection when ChromaDB is not available
    Mimics ChromaDB API but stores in RAM only
    """

    def __init__(self, name: str):
        self.name = name
        self.data: Dict[str, Dict] = {}

    def add(self, embeddings: List, documents: List[str],
            metadatas: List[Dict], ids: List[str]):
        """Add documents to collection"""
        for i, doc_id in enumerate(ids):
            self.data[doc_id] = {
                'embedding': np.array(embeddings[i]),
                'document': documents[i],
                'metadata': metadatas[i]
            }

    def query(self, query_embeddings: List, n_results: int = 5,
             where: Optional[Dict] = None) -> Dict:
        """Query similar documents"""
        if not self.data:
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

        query_emb = np.array(query_embeddings[0])

        # Calculate similarities
        similarities = []
        for doc_id, data in self.data.items():
            # Filter by metadata if provided
            if where:
                match = all(data['metadata'].get(k) == v for k, v in where.items())
                if not match:
                    continue

            # Cosine similarity
            emb = data['embedding']
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            distance = 1.0 - sim
            similarities.append((doc_id, distance))

        # Sort by distance
        similarities.sort(key=lambda x: x[1])
        top_results = similarities[:n_results]

        # Format results
        ids = [[doc_id for doc_id, _ in top_results]]
        documents = [[self.data[doc_id]['document'] for doc_id, _ in top_results]]
        metadatas = [[self.data[doc_id]['metadata'] for doc_id, _ in top_results]]
        distances = [[dist for _, dist in top_results]]

        return {
            'ids': ids,
            'documents': documents,
            'metadatas': metadatas,
            'distances': distances
        }

    def get(self, limit: int = 10, include: List[str] = None,
            where: Optional[Dict] = None) -> Dict:
        """Get documents from collection"""
        items = list(self.data.items())

        # Filter by metadata if provided
        if where:
            items = [(doc_id, data) for doc_id, data in items
                    if all(data['metadata'].get(k) == v for k, v in where.items())]

        items = items[:limit]

        return {
            'ids': [doc_id for doc_id, _ in items],
            'documents': [data['document'] for _, data in items],
            'metadatas': [data['metadata'] for _, data in items]
        }

    def update(self, ids: List[str], metadatas: List[Dict]):
        """Update document metadata"""
        for doc_id, metadata in zip(ids, metadatas):
            if doc_id in self.data:
                self.data[doc_id]['metadata'].update(metadata)

    def delete(self, ids: List[str]):
        """Delete documents"""
        for doc_id in ids:
            if doc_id in self.data:
                del self.data[doc_id]

    def count(self) -> int:
        """Get count of documents"""
        return len(self.data)
