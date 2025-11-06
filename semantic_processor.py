"""
Semantic Processing Layer
Handles encoding, similarity, and semantic understanding
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter


class SemanticProcessor:
    """
    Handles semantic understanding and encoding using transformer models
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model
        all-MiniLM-L6-v2: Fast, good quality, 384 dimensions
        """
        print(f"Loading semantic model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode(self, text: str) -> np.ndarray:
        """Encode text into semantic embedding"""
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts efficiently"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        return float(cosine_similarity(embedding1, embedding2)[0][0])

    def find_similar(self, query_embedding: np.ndarray,
                    candidate_embeddings: List[np.ndarray],
                    top_k: int = 5,
                    threshold: float = 0.3) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings from candidates
        Returns: List of (index, similarity_score) tuples
        """
        if not candidate_embeddings:
            return []

        query = query_embedding.reshape(1, -1)
        candidates = np.vstack([e.reshape(1, -1) for e in candidate_embeddings])

        similarities = cosine_similarity(query, candidates)[0]

        # Get top k indices above threshold
        indices_above_threshold = np.where(similarities >= threshold)[0]
        sorted_indices = indices_above_threshold[np.argsort(-similarities[indices_above_threshold])]

        results = [(int(idx), float(similarities[idx])) for idx in sorted_indices[:top_k]]
        return results

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction using word frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Remove common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
            'it', 'from', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had'
        }

        filtered_words = [w for w in words if w not in stop_words]
        word_freq = Counter(filtered_words)

        return [word for word, _ in word_freq.most_common(top_n)]

    def semantic_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split text into semantically coherent chunks
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def calculate_semantic_distance(self, text1: str, text2: str) -> float:
        """Calculate semantic distance between two texts"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return 1.0 - self.similarity(emb1, emb2)

    def find_analogies(self, a: str, b: str, c: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Find analogies: a is to b as c is to ?
        Example: "king" is to "queen" as "man" is to "woman"
        """
        emb_a = self.encode(a)
        emb_b = self.encode(b)
        emb_c = self.encode(c)

        # Calculate the analogy vector
        analogy_vector = emb_b - emb_a + emb_c

        # Encode candidates
        candidate_embeddings = self.encode_batch(candidates)

        # Find most similar to analogy vector
        similarities = cosine_similarity(analogy_vector.reshape(1, -1), candidate_embeddings)[0]

        results = [(candidates[i], float(similarities[i])) for i in range(len(candidates))]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:5]

    def aggregate_embeddings(self, embeddings: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Aggregate multiple embeddings into one
        Useful for creating context representations
        """
        if not embeddings:
            return np.zeros(self.embedding_dim)

        if weights is None:
            weights = [1.0] * len(embeddings)

        weighted_embeddings = [emb * w for emb, w in zip(embeddings, weights)]
        aggregated = np.sum(weighted_embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm

        return aggregated


class ContextManager:
    """
    Manages conversation context and builds rich contextual representations
    """

    def __init__(self, semantic_processor: SemanticProcessor):
        self.semantic_processor = semantic_processor
        self.conversation_history: List[Dict[str, any]] = []
        self.topic_stack: List[str] = []
        self.current_context_embedding: Optional[np.ndarray] = None

    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a conversation turn"""
        turn = {
            'role': role,
            'content': content,
            'embedding': self.semantic_processor.encode(content),
            'timestamp': metadata.get('timestamp') if metadata else None,
            'metadata': metadata or {}
        }
        self.conversation_history.append(turn)
        self._update_context()

    def _update_context(self):
        """Update the current context embedding"""
        if not self.conversation_history:
            self.current_context_embedding = None
            return

        # Weight recent messages more heavily
        recent_turns = self.conversation_history[-10:]
        embeddings = [turn['embedding'] for turn in recent_turns]

        # Exponential decay weights (more recent = higher weight)
        weights = [0.5 ** (len(recent_turns) - i - 1) for i in range(len(recent_turns))]

        self.current_context_embedding = self.semantic_processor.aggregate_embeddings(
            embeddings, weights
        )

    def get_context_embedding(self) -> Optional[np.ndarray]:
        """Get the current context representation"""
        return self.current_context_embedding

    def get_relevant_history(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant parts of conversation history"""
        if not self.conversation_history:
            return []

        query_embedding = self.semantic_processor.encode(query)
        history_embeddings = [turn['embedding'] for turn in self.conversation_history]

        similar_indices = self.semantic_processor.find_similar(
            query_embedding,
            history_embeddings,
            top_k=top_k,
            threshold=0.2
        )

        relevant_turns = [self.conversation_history[idx] for idx, _ in similar_indices]
        return relevant_turns

    def detect_topic_shift(self, new_message: str, threshold: float = 0.4) -> bool:
        """Detect if there's a topic shift in the conversation"""
        if self.current_context_embedding is None:
            return True

        new_embedding = self.semantic_processor.encode(new_message)
        similarity = self.semantic_processor.similarity(
            new_embedding,
            self.current_context_embedding
        )

        return similarity < threshold

    def summarize_context(self, max_items: int = 5) -> str:
        """Generate a text summary of current context"""
        if not self.conversation_history:
            return "No conversation history."

        recent = self.conversation_history[-max_items:]
        summary_parts = []

        for turn in recent:
            role = turn['role']
            content = turn['content'][:100]
            summary_parts.append(f"{role}: {content}")

        return "\n".join(summary_parts)
