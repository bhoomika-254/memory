"""
Embedding Service for Memory System.

This module handles text-to-vector conversion using sentence-transformers.
It provides a consistent interface for embedding text chunks that will be
stored in the Qdrant vector database for semantic search.

Key Features:
- Uses sentence-transformers for high-quality embeddings
- Handles batch processing for efficiency
- Caches the model to avoid repeated loading
- Provides both single text and batch embedding methods
"""

import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """
    Service for converting text to embeddings using sentence-transformers.
    
    This service provides a centralized way to generate embeddings for text
    that will be stored in our vector database. It uses the configured
    sentence-transformer model and handles batching for efficiency.
    """
    
    def __init__(self):
        """Initialize the embedding service with the configured model."""
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.vector_size = 384  # sentence-transformers/all-MiniLM-L6-v2 dimension
        self._model: Optional[SentenceTransformer] = None
        
        print(f"🔧 Embedding service configured with model: {self.model_name}")
    
    def _load_model(self) -> SentenceTransformer:
        """
        Lazy load the sentence transformer model.
        
        This approach loads the model only when first needed, which
        speeds up application startup and saves memory if embeddings
        aren't immediately required.
        
        Returns:
            Loaded SentenceTransformer model
        """
        if self._model is None:
            print(f"📥 Loading embedding model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name)
                print(f"✅ Embedding model loaded successfully")
                
                # Verify the model produces vectors of expected size
                test_embedding = self._model.encode("test")
                actual_size = len(test_embedding)
                
                if actual_size != self.vector_size:
                    print(f"⚠️  Warning: Model produces {actual_size}D vectors, expected {self.vector_size}D")
                    # Update to match actual model
                    self.vector_size = actual_size
                    print(f"🔧 Updated vector size to {actual_size}")
                
            except Exception as e:
                print(f"❌ Failed to load embedding model: {str(e)}")
                raise
        
        return self._model
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array representing the text embedding
            
        Raises:
            ValueError: If text is empty or None
            RuntimeError: If model fails to generate embedding
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        try:
            model = self._load_model()
            embedding = model.encode(text.strip())
            
            # Ensure we return a numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        This is more efficient than calling embed_text multiple times
        as it can leverage batch processing in the underlying model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of numpy arrays, one embedding per input text
            
        Raises:
            ValueError: If texts list is empty
            RuntimeError: If batch embedding fails
        """
        if not texts:
            raise ValueError("Cannot embed empty text list")
        
        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
        
        if not valid_texts:
            raise ValueError("No valid texts to embed after filtering")
        
        try:
            model = self._load_model()
            
            print(f"🔄 Generating embeddings for {len(valid_texts)} texts...")
            embeddings = model.encode(valid_texts, show_progress_bar=len(valid_texts) > 10)
            
            # Convert to list of numpy arrays
            if isinstance(embeddings, np.ndarray):
                embeddings = [embeddings[i] for i in range(len(embeddings))]
            
            # Create result list with None placeholders for invalid texts
            result = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                original_index = valid_indices[i]
                result[original_index] = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
            
            print(f"✅ Generated {len(valid_texts)} embeddings successfully")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding specifically for search queries.
        
        This is essentially the same as embed_text but provides a
        semantic distinction for query embeddings vs document embeddings.
        Some models have separate query/document encoders.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding as numpy array
        """
        return self.embed_text(query)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        This utility method helps with debugging and manual similarity
        calculations outside of the vector database.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        model = self._load_model()
        
        return {
            "model_name": self.model_name,
            "vector_size": self.vector_size,
            "max_sequence_length": getattr(model, 'max_seq_length', 'Unknown'),
            "model_loaded": self._model is not None
        }

# Create a global embedding service instance
embedding_service = EmbeddingService()

# Example usage and testing
if __name__ == "__main__":
    # Test the embedding service
    test_texts = [
        "What is packing optimization?",
        "How can machine learning improve shipping efficiency?",
        "Tell me about bin packing algorithms"
    ]
    
    print("Testing embedding service...")
    
    # Test single embedding
    single_embedding = embedding_service.embed_text(test_texts[0])
    print(f"Single embedding shape: {single_embedding.shape}")
    
    # Test batch embedding
    batch_embeddings = embedding_service.embed_batch(test_texts)
    print(f"Batch embeddings: {len(batch_embeddings)} vectors")
    
    # Test similarity
    if len(batch_embeddings) >= 2:
        sim = embedding_service.similarity(batch_embeddings[0], batch_embeddings[1])
        print(f"Similarity between first two texts: {sim:.3f}")
    
    # Show model info
    info = embedding_service.get_model_info()
    print(f"Model info: {info}")
