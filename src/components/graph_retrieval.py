"""
Graph-Based Retrieval System for Memory.

This module replaces the previous Qdrant + Whoosh system with a unified
Neo4j graph database approach, implementing the LightRAG methodology.

Key Features:
- Semantic similarity search using vector embeddings
- Fulltext keyword search using Neo4j's built-in indexing
- Graph traversal for contextual relationship discovery
- Reciprocal Rank Fusion (RRF) for result combination
- Entity and topic-aware retrieval (future enhancement)
"""

from typing import Dict, List, Any, Optional
from src.components.neo4j_manager import Neo4jManager


class GraphRetrieval:
    """
    Graph-based retrieval system using Neo4j.
    
    This class provides the same interface as the previous retrieval_fusion
    component but uses Neo4j graph database for all operations.
    """
    
    def __init__(self):
        """Initialize the graph retrieval system."""
        self.neo4j_manager = Neo4jManager()
        print("🕸️  Graph Retrieval System initialized")
    
    def hybrid_search(self, query: str, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic, lexical, and graph methods.
        
        Args:
            query: Search query text
            conversation_id: Conversation context for filtering
            limit: Maximum number of results to return
            
        Returns:
            List of ranked search results with metadata
        """
        try:
            print(f"🔍 Performing graph-based hybrid search...")
            
            # Use Neo4j's hybrid search
            results = self.neo4j_manager.hybrid_search(
                query_text=query,
                conversation_id=conversation_id,
                limit=limit
            )
            
            # Transform results to match expected format
            formatted_results = []
            for result in results:
                formatted_result = {
                    "text": result["text"],
                    "conversation_id": result["conversation_id"],
                    "turn_number": result["turn_number"],
                    "user_message": result["user_message"],
                    "assistant_message": result["assistant_message"],
                    "timestamp": result["timestamp"],
                    "rrf_score": result.get("rrf_score", 0.0),
                    "source": result.get("source", "graph_hybrid"),
                    "message_id": result["id"]
                }
                formatted_results.append(formatted_result)
            
            print(f"✅ Graph hybrid search completed: {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in graph hybrid search: {str(e)}")
            return []
    
    def semantic_search(self, query: str, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search only.
        
        Args:
            query: Search query text
            conversation_id: Conversation context for filtering  
            limit: Maximum number of results to return
            
        Returns:
            List of semantically similar results
        """
        try:
            results = self.neo4j_manager.semantic_search(
                query_text=query,
                conversation_id=conversation_id,
                limit=limit
            )
            
            # Transform to expected format
            formatted_results = []
            for result in results:
                formatted_result = {
                    "text": result["text"],
                    "conversation_id": result["conversation_id"],
                    "turn_number": result["turn_number"],
                    "similarity_score": result.get("similarity_score", 0.0),
                    "source": "semantic",
                    "message_id": result["id"]
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in semantic search: {str(e)}")
            return []
    
    def lexical_search(self, query: str, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform lexical/keyword search only.
        
        Args:
            query: Search query text
            conversation_id: Conversation context for filtering
            limit: Maximum number of results to return
            
        Returns:
            List of keyword-matched results
        """
        try:
            results = self.neo4j_manager.fulltext_search(
                query_text=query,
                conversation_id=conversation_id,
                limit=limit
            )
            
            # Transform to expected format
            formatted_results = []
            for result in results:
                formatted_result = {
                    "text": result["text"],
                    "conversation_id": result["conversation_id"],
                    "turn_number": result["turn_number"],
                    "relevance_score": result.get("relevance_score", 0.0),
                    "source": "lexical",
                    "message_id": result["id"]
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in lexical search: {str(e)}")
            return []
    
    def graph_traversal_search(self, seed_message_ids: List[str], hops: int = 1) -> List[Dict[str, Any]]:
        """
        Perform graph traversal search from seed messages.
        
        Args:
            seed_message_ids: Starting message IDs for traversal
            hops: Number of relationship hops to traverse
            
        Returns:
            List of related messages found through graph traversal
        """
        try:
            results = self.neo4j_manager.graph_traversal_search(
                message_ids=seed_message_ids,
                hops=hops
            )
            
            # Transform to expected format
            formatted_results = []
            for result in results:
                formatted_result = {
                    "text": result["text"],
                    "conversation_id": result["conversation_id"],
                    "turn_number": result["turn_number"],
                    "source": "graph_traversal",
                    "message_id": result["id"]
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in graph traversal search: {str(e)}")
            return []
    
    def close(self):
        """Close connections and cleanup resources."""
        if self.neo4j_manager:
            self.neo4j_manager.close()
            print("🔌 Graph Retrieval System closed")


# Test the graph retrieval system
if __name__ == "__main__":
    try:
        print("🧪 Testing Graph Retrieval System...")
        
        retrieval = GraphRetrieval()
        
        # Test hybrid search
        results = retrieval.hybrid_search("Neo4j graph database", "test_conversation", limit=5)
        
        if results:
            print(f"✅ Graph retrieval test successful! Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"   {i}. Score: {result.get('rrf_score', 0):.4f} - {result['text'][:80]}...")
        else:
            print("❌ No results found")
        
        retrieval.close()
        print("✅ Graph Retrieval test completed!")
        
    except Exception as e:
        print(f"❌ Graph Retrieval test failed: {str(e)}")
        import traceback
        traceback.print_exc()
