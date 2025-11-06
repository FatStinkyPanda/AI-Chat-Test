"""
Vector Memory System using ChromaDB
Persistent semantic memory with fast retrieval
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
import os


class VectorMemory:
    """
    Manages semantic memory using ChromaDB for persistent vector storage
    """

    def __init__(self, persist_directory: str = "./brain_memory"):
        """Initialize ChromaDB for persistent storage"""
        self.persist_directory = persist_directory

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Create collections for different memory types
        self.episodic_memory = self._get_or_create_collection("episodic_memory")
        self.semantic_memory = self._get_or_create_collection("semantic_memory")
        self.emotional_memory = self._get_or_create_collection("emotional_memory")

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
            self.client.delete_collection(name=collection_name)
            setattr(self, collection_name, self._get_or_create_collection(collection_name))
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
