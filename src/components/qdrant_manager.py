"""
Qdrant Vector Database Manager for Memory System.

This module handles all interactions with the Qdrant vector database,
including storing conversation embeddings and performing semantic search.
It provides a high-level interface for the memory system's vector operations.

Key Features:
- Automatic collection creation and management
- Efficient batch storage of embeddings
- Semantic search with metadata filtering
- Connection management and error handling
"""

import uuid
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, SearchRequest
)

from src.utils.embedding_service import embedding_service

class QdrantManager:
    """
    Manages all Qdrant vector database operations for the memory system.
    
    This class provides a high-level interface for storing conversation
    embeddings and performing semantic search. It handles connection
    management, collection setup, and all vector operations.
    """
    
    def __init__(self):
        """Initialize the Qdrant manager with connection settings."""
        self.collection_name = "memory_chunks"
        self.vector_size = 384  # sentence-transformers/all-MiniLM-L6-v2 dimension
        
        # Check for cloud configuration
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Connection mode preference: cloud > local server > in-memory
        if self.qdrant_url and self.qdrant_api_key:
            self.connection_mode = "cloud"
            print(f"🌐 Qdrant manager configured for cloud mode: {self.qdrant_url}")
        elif self._check_local_server():
            self.connection_mode = "local"
            self.host = "localhost"
            self.port = 6333
            print(f"🔧 Qdrant manager configured for local server: {self.host}:{self.port}")
        else:
            self.connection_mode = "memory"
            print(f"🧠 Qdrant manager configured for in-memory mode")
        
        self._client: Optional[QdrantClient] = None
        self._collection_exists = False
    
    def _check_local_server(self) -> bool:
        """Check if local Qdrant server is available."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 6333))
            sock.close()
            return result == 0
        except:
            return False
    
    def _get_client(self) -> QdrantClient:
        """
        Get or create Qdrant client connection.
        
        Uses lazy loading to establish connection only when needed.
        Tries cloud first, then local server, then in-memory.
        
        Returns:
            Connected QdrantClient instance
        """
        if self._client is None:
            if self.connection_mode == "cloud":
                try:
                    print(f"🌐 Connecting to Qdrant Cloud: {self.qdrant_url}")
                    self._client = QdrantClient(
                        url=self.qdrant_url,
                        api_key=self.qdrant_api_key,
                        timeout=60,
                        prefer_grpc=True  # Use gRPC for cloud connections
                    )
                    # Test connection
                    collections = self._client.get_collections()
                    print(f"✅ Connected to Qdrant Cloud successfully ({len(collections.collections)} collections)")
                except Exception as e:
                    print(f"❌ Failed to connect to Qdrant Cloud: {str(e)}")
                    print("💡 Trying alternative connection method...")
                    try:
                        # Try without prefer_grpc
                        self._client = QdrantClient(
                            url=self.qdrant_url,
                            api_key=self.qdrant_api_key,
                            timeout=60
                        )
                        collections = self._client.get_collections()
                        print(f"✅ Connected to Qdrant Cloud (HTTP) successfully ({len(collections.collections)} collections)")
                    except Exception as e2:
                        print(f"❌ Alternative connection also failed: {str(e2)}")
                        print("🔄 Falling back to in-memory mode...")
                        self.connection_mode = "memory"
            
            if self.connection_mode == "local":
                try:
                    print(f"🔌 Connecting to local Qdrant server at {self.host}:{self.port}")
                    self._client = QdrantClient(host=self.host, port=self.port, timeout=60)
                    
                    # Test connection
                    collections = self._client.get_collections()
                    print(f"✅ Connected to local Qdrant server successfully ({len(collections.collections)} collections)")
                    
                except Exception as e:
                    print(f"❌ Failed to connect to local Qdrant server: {str(e)}")
                    print("� Falling back to in-memory mode...")
                    self.connection_mode = "memory"
            
            if self.connection_mode == "memory":
                try:
                    print("🧠 Creating in-memory Qdrant client")
                    self._client = QdrantClient(":memory:")
                    print("✅ In-memory Qdrant client created successfully (no persistence)")
                except Exception as e:
                    print(f"❌ Failed to create in-memory client: {str(e)}")
                    raise
        
        return self._client
    
    def _ensure_collection(self) -> None:
        """
        Ensure the conversation memory collection exists.
        
        Creates the collection if it doesn't exist, with proper
        vector configuration for our embedding model.
        """
        if self._collection_exists:
            return
        
        client = self._get_client()
        
        try:
            # Check if collection exists
            collections = client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                print(f"✅ Collection '{self.collection_name}' already exists")
                self._collection_exists = True
                return
            
            # Create collection
            print(f"🏗️  Creating collection '{self.collection_name}'...")
            
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE  # Cosine similarity for semantic search
                )
            )
            
            print(f"✅ Collection '{self.collection_name}' created successfully")
            self._collection_exists = True
            
        except Exception as e:
            print(f"❌ Failed to create collection: {str(e)}")
            raise
    
    def store_conversation_chunks(self, chunks: List[Dict[str, Any]], 
                                conversation_id: str) -> List[str]:
        """
        Store conversation chunks with their embeddings in Qdrant.
        
        This method takes chunked conversation data, generates embeddings,
        and stores everything in Qdrant with appropriate metadata.
        
        Args:
            chunks: List of chunk dictionaries from text_chunker
            conversation_id: Unique conversation identifier
            
        Returns:
            List of point IDs that were stored
        """
        if not chunks:
            return []
        
        self._ensure_collection()
        client = self._get_client()
        
        try:
            print(f"💾 Storing {len(chunks)} chunks for conversation {conversation_id}")
            
            # Extract texts for batch embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings for all chunks
            embeddings = embedding_service.embed_batch(texts)
            
            # Prepare points for Qdrant
            points = []
            point_ids = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding is None:
                    print(f"⚠️  Skipping chunk {i} - failed to generate embedding")
                    continue
                
                # Generate unique point ID
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                # Prepare metadata payload
                payload = {
                    "conversation_id": conversation_id,
                    "chunk_index": chunk.get("chunk_index", i),
                    "token_count": chunk.get("token_count", 0),
                    "text": chunk["text"],
                    "turn_number": chunk.get("turn_number"),
                    "user_message": chunk.get("user_message", ""),
                    "assistant_message": chunk.get("assistant_message", ""),
                    "message_type": chunk.get("message_type", "conversation_turn"),
                    "timestamp": datetime.now().isoformat(),
                    "sentence_start": chunk.get("sentence_start"),
                    "sentence_end": chunk.get("sentence_end")
                }
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),  # Convert numpy array to list
                    payload=payload
                )
                
                points.append(point)
            
            # Store all points in batch
            if points:
                client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                print(f"✅ Stored {len(points)} chunks successfully")
            else:
                print("⚠️  No valid chunks to store")
            
            return point_ids
            
        except Exception as e:
            print(f"❌ Failed to store chunks: {str(e)}")
            raise
    
    def semantic_search(self, query: str, top_k: int = None, 
                       conversation_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search for relevant conversation chunks.
        
        Uses the query embedding to find the most semantically similar
        stored conversation chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return (default from config)
            conversation_id: Optional filter by conversation ID
            
        Returns:
            List of search results with chunks and similarity scores
        """
        if top_k is None:
            top_k = 20  # Default semantic search limit
        
        self._ensure_collection()
        client = self._get_client()
        
        try:
            print(f"🔍 Semantic search for: '{query[:50]}...' (top {top_k})")
            
            # Generate query embedding
            query_embedding = embedding_service.embed_query(query)
            
            # Prepare search filter if conversation_id specified
            search_filter = None
            if conversation_id:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="conversation_id",
                            match=MatchValue(value=conversation_id)
                        )
                    ]
                )
            
            # Perform search
            search_results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload["text"],
                    "conversation_id": result.payload["conversation_id"],
                    "turn_number": result.payload.get("turn_number"),
                    "timestamp": result.payload["timestamp"],
                    "chunk_index": result.payload.get("chunk_index"),
                    "token_count": result.payload.get("token_count"),
                    "user_message": result.payload.get("user_message", ""),
                    "assistant_message": result.payload.get("assistant_message", ""),
                    "metadata": result.payload
                })
            
            print(f"✅ Found {len(results)} semantic matches")
            return results
            
        except Exception as e:
            print(f"❌ Semantic search failed: {str(e)}")
            return []
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific conversation.
        
        Args:
            conversation_id: Conversation to retrieve
            
        Returns:
            List of all chunks for the conversation, sorted by turn and chunk index
        """
        self._ensure_collection()
        client = self._get_client()
        
        try:
            # Search with conversation filter
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id)
                    )
                ]
            )
            
            # Get all chunks for this conversation
            results = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=1000,  # Adjust based on expected conversation length
                with_payload=True,
                with_vectors=False
            )
            
            chunks = []
            for point in results[0]:  # results is a tuple (points, next_page_offset)
                chunks.append({
                    "id": point.id,
                    "text": point.payload["text"],
                    "turn_number": point.payload.get("turn_number"),
                    "chunk_index": point.payload.get("chunk_index"),
                    "timestamp": point.payload["timestamp"],
                    "metadata": point.payload
                })
            
            # Sort by turn number and chunk index
            chunks.sort(key=lambda x: (x.get("turn_number", 0), x.get("chunk_index", 0)))
            
            return chunks
            
        except Exception as e:
            print(f"❌ Failed to retrieve conversation history: {str(e)}")
            return []
    
    def clear_conversation_memory(self, conversation_id: str) -> bool:
        """
        Delete all memory chunks for a specific conversation.
        
        Args:
            conversation_id: Conversation to clear
            
        Returns:
            True if successful, False otherwise
        """
        self._ensure_collection()
        client = self._get_client()
        
        try:
            print(f"🗑️  Clearing memory for conversation {conversation_id}")
            
            # Create filter for this conversation
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id)
                    )
                ]
            )
            
            # Delete all points matching the filter
            client.delete(
                collection_name=self.collection_name,
                points_selector=delete_filter
            )
            
            print(f"✅ Cleared memory for conversation {conversation_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to clear conversation memory: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection.
        
        Returns:
            Dictionary with collection statistics
        """
        self._ensure_collection()
        client = self._get_client()
        
        try:
            collection_info = client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "status": collection_info.status
            }
            
        except Exception as e:
            print(f"❌ Failed to get collection stats: {str(e)}")
            return {}

# Create global Qdrant manager instance
qdrant_manager = QdrantManager()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Qdrant manager...")
    
    # Test connection and collection creation
    try:
        stats = qdrant_manager.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Test storing a sample chunk
        sample_chunks = [{
            "text": "This is a test conversation about machine learning and AI.",
            "chunk_index": 0,
            "token_count": 12,
            "turn_number": 1,
            "user_message": "Tell me about AI",
            "assistant_message": "AI is about machine learning...",
            "message_type": "conversation_turn"
        }]
        
        test_conversation_id = "test_conv_123"
        point_ids = qdrant_manager.store_conversation_chunks(sample_chunks, test_conversation_id)
        print(f"Stored test chunks with IDs: {point_ids}")
        
        # Test semantic search
        results = qdrant_manager.semantic_search("machine learning", top_k=3)
        print(f"Search results: {len(results)} found")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Make sure Qdrant is running!")
