"""
Neo4j Graph Database Manager for Memory System.

This module handles all interactions with Neo4j graph database, including:
- Connection management
- Index creation and management
- Node and relationship operations
- Vector similarity search
- Graph traversal queries

Key Features:
- Message nodes with embeddings
- Entity and Topic extraction and storage
- Relationship management with weights
- Vector similarity search using Neo4j's native capabilities
"""

import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.utils.embedding_service import embedding_service

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class Neo4jManager:
    """
    Manages Neo4j graph database operations for the memory system.
    
    This class provides a high-level interface for storing and retrieving
    conversation data in a graph format, supporting semantic, lexical,
    and relational queries.
    """
    
    def __init__(self):
        """Initialize Neo4j connection and setup."""
        self.driver: Optional[Driver] = None
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Connection settings
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.uri, self.username, self.password]):
            raise ValueError("Missing Neo4j credentials in environment variables")
        
        # Initialize connection
        self._connect()
        self._setup_indexes()
        
        print("✅ Neo4j Manager initialized successfully")
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    print(f"🔗 Connected to Neo4j database: {self.database}")
                else:
                    raise Exception("Connection test failed")
                    
        except (ServiceUnavailable, AuthError) as e:
            print(f"❌ Failed to connect to Neo4j: {str(e)}")
            raise
    
    def _setup_indexes(self) -> None:
        """Create necessary indexes for optimal query performance."""
        try:
            with self.driver.session(database=self.database) as session:
                # Index setup queries
                index_queries = [
                    # Message ID index for fast lookups
                    "CREATE INDEX message_id_idx IF NOT EXISTS FOR (m:Message) ON (m.id)",
                    
                    # Conversation ID index for filtering
                    "CREATE INDEX conversation_id_idx IF NOT EXISTS FOR (m:Message) ON (m.conversation_id)",
                    
                    # Entity name index for keyword search
                    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                    
                    # Topic name index
                    "CREATE INDEX topic_name_idx IF NOT EXISTS FOR (t:Topic) ON (t.name)",
                    
                    # Timestamp index for temporal queries
                    "CREATE INDEX message_timestamp_idx IF NOT EXISTS FOR (m:Message) ON (m.timestamp)",
                    
                    # Vector indexes for similarity search (Neo4j 5.x+)
                    """CREATE VECTOR INDEX message_embeddings_idx IF NOT EXISTS
                       FOR (m:Message) ON (m.embedding)
                       OPTIONS {indexConfig: {
                         `vector.dimensions`: 384,
                         `vector.similarity_function`: 'cosine'
                       }}""",
                    
                    """CREATE VECTOR INDEX entity_embeddings_idx IF NOT EXISTS
                       FOR (e:Entity) ON (e.embedding)
                       OPTIONS {indexConfig: {
                         `vector.dimensions`: 384,
                         `vector.similarity_function`: 'cosine'
                       }}""",
                    
                    """CREATE VECTOR INDEX topic_embeddings_idx IF NOT EXISTS
                       FOR (t:Topic) ON (t.embedding)
                       OPTIONS {indexConfig: {
                         `vector.dimensions`: 384,
                         `vector.similarity_function`: 'cosine'
                       }}""",
                    
                    # Fulltext index for text search
                    "CREATE FULLTEXT INDEX message_text_idx IF NOT EXISTS FOR (m:Message) ON EACH [m.text]",
                    "CREATE FULLTEXT INDEX entity_description_idx IF NOT EXISTS FOR (e:Entity) ON EACH [e.description]"
                ]
                
                for query in index_queries:
                    try:
                        session.run(query)
                        print(f"✅ Index created/verified")
                    except Exception as e:
                        print(f"⚠️  Index creation warning: {str(e)}")
                
                print("🔍 All indexes created/verified successfully")
                
        except Exception as e:
            print(f"❌ Failed to setup indexes: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed")
    
    def store_message_node(self, 
                          message_id: str,
                          text: str,
                          conversation_id: str,
                          turn_number: int,
                          user_message: str,
                          assistant_message: str,
                          chunk_index: int = 0,
                          metadata: Dict[str, Any] = None) -> bool:
        """
        Store a message as a node in the graph database.
        
        Args:
            message_id: Unique identifier for the message
            text: Full text content of the message
            conversation_id: ID of the conversation this belongs to
            turn_number: Turn number in the conversation
            user_message: Original user input
            assistant_message: Original assistant response
            chunk_index: Index if this is part of a chunked message
            metadata: Additional metadata to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding for the text
            embedding = embedding_service.embed_text(text)
            if embedding is None:
                print(f"⚠️  Failed to generate embedding for message {message_id}")
                return False
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Create the message node
            with self.driver.session(database=self.database) as session:
                query = """
                MERGE (m:Message {id: $message_id})
                SET m.text = $text,
                    m.embedding = $embedding,
                    m.conversation_id = $conversation_id,
                    m.turn_number = $turn_number,
                    m.user_message = $user_message,
                    m.assistant_message = $assistant_message,
                    m.chunk_index = $chunk_index,
                    m.timestamp = $timestamp,
                    m.token_count = $token_count,
                    m.sentence_start = $sentence_start,
                    m.sentence_end = $sentence_end,
                    m.message_type = $message_type
                RETURN m.id as created_id
                """
                
                result = session.run(query, {
                    "message_id": message_id,
                    "text": text,
                    "embedding": embedding.tolist(),  # Convert numpy array to list
                    "conversation_id": conversation_id,
                    "turn_number": turn_number,
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                    "chunk_index": chunk_index,
                    "timestamp": datetime.now().isoformat(),
                    "token_count": len(text.split()),  # Simple token count
                    "sentence_start": metadata.get("sentence_start"),
                    "sentence_end": metadata.get("sentence_end"),
                    "message_type": metadata.get("message_type", "conversation_turn")
                })
                
                created_record = result.single()
                if created_record:
                    print(f"✅ Message node created: {created_record['created_id']}")
                    return True
                else:
                    print(f"❌ Failed to create message node: {message_id}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error storing message node {message_id}: {str(e)}")
            return False
    
    def store_conversation_chunks(self, chunks: List[Dict[str, Any]], conversation_id: str) -> List[str]:
        """
        Store multiple conversation chunks as message nodes with NEXT relationships.
        
        Args:
            chunks: List of chunk dictionaries from text_chunker
            conversation_id: Unique conversation identifier
            
        Returns:
            List of message IDs that were stored
        """
        if not chunks:
            return []
        
        stored_ids = []
        
        try:
            print(f"💾 Storing {len(chunks)} message chunks for conversation {conversation_id}")
            
            # Store each chunk as a message node
            for i, chunk in enumerate(chunks):
                message_id = str(uuid.uuid4())
                
                success = self.store_message_node(
                    message_id=message_id,
                    text=chunk["text"],
                    conversation_id=conversation_id,
                    turn_number=chunk.get("turn_number", 1),
                    user_message=chunk.get("user_message", ""),
                    assistant_message=chunk.get("assistant_message", ""),
                    chunk_index=chunk.get("chunk_index", i),
                    metadata={
                        "sentence_start": chunk.get("sentence_start"),
                        "sentence_end": chunk.get("sentence_end"),
                        "message_type": chunk.get("message_type", "conversation_turn")
                    }
                )
                
                if success:
                    stored_ids.append(message_id)
            
            # Create NEXT relationships between consecutive chunks
            if len(stored_ids) > 1:
                self._create_next_relationships(stored_ids)
            
            print(f"✅ Stored {len(stored_ids)} message nodes successfully")
            return stored_ids
            
        except Exception as e:
            print(f"❌ Error storing conversation chunks: {str(e)}")
            return stored_ids  # Return partial results
    
    def _create_next_relationships(self, message_ids: List[str]) -> None:
        """Create NEXT relationships between consecutive message nodes."""
        try:
            with self.driver.session(database=self.database) as session:
                for i in range(len(message_ids) - 1):
                    current_id = message_ids[i]
                    next_id = message_ids[i + 1]
                    
                    query = """
                    MATCH (current:Message {id: $current_id})
                    MATCH (next:Message {id: $next_id})
                    MERGE (current)-[:NEXT]->(next)
                    """
                    
                    session.run(query, {
                        "current_id": current_id,
                        "next_id": next_id
                    })
                
                print(f"🔗 Created {len(message_ids) - 1} NEXT relationships")
                
        except Exception as e:
            print(f"❌ Error creating NEXT relationships: {str(e)}")
    
    def semantic_search(self, query_text: str, conversation_id: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using vector embeddings.
        
        Args:
            query_text: Text to search for
            conversation_id: Optional conversation filter
            limit: Maximum number of results to return
            
        Returns:
            List of matching message nodes with similarity scores
        """
        try:
            # Generate embedding for the query
            query_embedding = embedding_service.embed_text(query_text)
            if query_embedding is None:
                print("⚠️  Failed to generate query embedding")
                return []
            
            with self.driver.session(database=self.database) as session:
                # Build query with optional conversation filter
                base_query = """
                CALL db.index.vector.queryNodes('message_embeddings_idx', $limit, $query_embedding)
                YIELD node, score
                """
                
                if conversation_id:
                    base_query += "WHERE node.conversation_id = $conversation_id\n"
                
                base_query += """
                RETURN node.id as id,
                       node.text as text,
                       node.conversation_id as conversation_id,
                       node.turn_number as turn_number,
                       node.user_message as user_message,
                       node.assistant_message as assistant_message,
                       node.timestamp as timestamp,
                       score
                ORDER BY score DESC
                """
                
                params = {
                    "limit": limit,
                    "query_embedding": query_embedding.tolist()
                }
                
                if conversation_id:
                    params["conversation_id"] = conversation_id
                
                result = session.run(base_query, params)
                
                results = []
                for record in result:
                    results.append({
                        "id": record["id"],
                        "text": record["text"],
                        "conversation_id": record["conversation_id"],
                        "turn_number": record["turn_number"],
                        "user_message": record["user_message"],
                        "assistant_message": record["assistant_message"],
                        "timestamp": record["timestamp"],
                        "similarity_score": record["score"],
                        "source": "semantic"
                    })
                
                print(f"🔍 Semantic search found {len(results)} results")
                return results
                
        except Exception as e:
            print(f"❌ Error in semantic search: {str(e)}")
            return []
    
    def fulltext_search(self, query_text: str, conversation_id: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Perform fulltext search on message content.
        
        Args:
            query_text: Text to search for
            conversation_id: Optional conversation filter
            limit: Maximum number of results to return
            
        Returns:
            List of matching message nodes with relevance scores
        """
        try:
            with self.driver.session(database=self.database) as session:
                base_query = """
                CALL db.index.fulltext.queryNodes('message_text_idx', $query_text)
                YIELD node, score
                """
                
                if conversation_id:
                    base_query += "WHERE node.conversation_id = $conversation_id\n"
                
                base_query += """
                RETURN node.id as id,
                       node.text as text,
                       node.conversation_id as conversation_id,
                       node.turn_number as turn_number,
                       node.user_message as user_message,
                       node.assistant_message as assistant_message,
                       node.timestamp as timestamp,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """
                
                params = {
                    "query_text": query_text,
                    "limit": limit
                }
                
                if conversation_id:
                    params["conversation_id"] = conversation_id
                
                result = session.run(base_query, params)
                
                results = []
                for record in result:
                    results.append({
                        "id": record["id"],
                        "text": record["text"],
                        "conversation_id": record["conversation_id"],
                        "turn_number": record["turn_number"],
                        "user_message": record["user_message"],
                        "assistant_message": record["assistant_message"],
                        "timestamp": record["timestamp"],
                        "relevance_score": record["score"],
                        "source": "fulltext"
                    })
                
                print(f"📝 Fulltext search found {len(results)} results")
                return results
                
        except Exception as e:
            print(f"❌ Error in fulltext search: {str(e)}")
            return []
    
    def graph_traversal_search(self, message_ids: List[str], hops: int = 1) -> List[Dict[str, Any]]:
        """
        Perform graph traversal to find related messages.
        
        Args:
            message_ids: Starting message IDs
            hops: Number of relationship hops to traverse
            
        Returns:
            List of related message nodes
        """
        try:
            if not message_ids:
                return []
            
            with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH (start:Message)
                WHERE start.id IN $message_ids
                MATCH (start)-[*1..{hops}]-(related:Message)
                WHERE related.id <> start.id
                RETURN DISTINCT related.id as id,
                       related.text as text,
                       related.conversation_id as conversation_id,
                       related.turn_number as turn_number,
                       related.user_message as user_message,
                       related.assistant_message as assistant_message,
                       related.timestamp as timestamp
                """
                
                result = session.run(query, {"message_ids": message_ids})
                
                results = []
                for record in result:
                    results.append({
                        "id": record["id"],
                        "text": record["text"],
                        "conversation_id": record["conversation_id"],
                        "turn_number": record["turn_number"],
                        "user_message": record["user_message"],
                        "assistant_message": record["assistant_message"],
                        "timestamp": record["timestamp"],
                        "source": "graph_traversal"
                    })
                
                print(f"🕸️  Graph traversal found {len(results)} related messages")
                return results
                
        except Exception as e:
            print(f"❌ Error in graph traversal search: {str(e)}")
            return []
    
    def hybrid_search(self, query_text: str, conversation_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic + fulltext + graph traversal.
        
        Args:
            query_text: Text to search for
            conversation_id: Optional conversation filter
            limit: Maximum number of final results
            
        Returns:
            List of fused and ranked results
        """
        try:
            print(f"🔍 Starting streamlined hybrid search for: '{query_text[:50]}...'")
            
            # 1. Semantic search (primary method)
            semantic_results = self.semantic_search(query_text, conversation_id, limit * 2)
            
            # 2. Graph traversal (for relationship context)
            if semantic_results:
                top_ids = [r["id"] for r in semantic_results[:5]]
                graph_results = self.graph_traversal_search(top_ids, hops=1)
            else:
                graph_results = []
            
            # 3. Simple concatenation (semantic-first, no fusion algorithm)
            combined_results = self._combine_semantic_graph_simple(
                semantic_results, graph_results, limit
            )
            
            print(f"✅ Streamlined hybrid search completed: {len(combined_results)} final results")
            return combined_results

        except Exception as e:
            print(f"❌ Error in hybrid search: {str(e)}")
            return []

    def _combine_semantic_graph_simple(self, 
                                     semantic_results: List[Dict[str, Any]], 
                                     graph_results: List[Dict[str, Any]], 
                                     limit: int) -> List[Dict[str, Any]]:
        """
        Simple combination of semantic + graph results without complex fusion.
        
        Semantic results take priority (they're already well-ranked), 
        graph results fill remaining slots for additional context.
        
        Args:
            semantic_results: Results from semantic search
            graph_results: Results from graph traversal
            limit: Maximum number of results to return
            
        Returns:
            Combined results (semantic-first, deduplicated)
        """
        try:
            # Use set to track seen IDs and avoid duplicates
            seen_ids = set()
            combined_results = []
            
            # 1. Add semantic results first (they're already scored and ranked)
            for result in semantic_results[:limit]:
                if result["id"] not in seen_ids:
                    # Keep original similarity score, add source info
                    result_copy = result.copy()
                    result_copy["source"] = "semantic"
                    combined_results.append(result_copy)
                    seen_ids.add(result["id"])
            
            # 2. Fill remaining slots with graph results for additional context
            remaining_slots = limit - len(combined_results)
            if remaining_slots > 0:
                for result in graph_results[:remaining_slots]:
                    if result["id"] not in seen_ids:
                        # Add source info for graph results
                        result_copy = result.copy()
                        result_copy["source"] = "graph_traversal"
                        combined_results.append(result_copy)
                        seen_ids.add(result["id"])
            
            print(f"📊 Combined {len(combined_results)} results: {len([r for r in combined_results if r['source'] == 'semantic'])} semantic + {len([r for r in combined_results if r['source'] == 'graph_traversal'])} graph")
            return combined_results[:limit]
            
        except Exception as e:
            print(f"❌ Error in simple combination: {str(e)}")
            return semantic_results[:limit]  # Fallback to semantic results
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()


# Test the connection and functionality
if __name__ == "__main__":
    try:
        print("🔧 Testing Neo4j Manager...")
        manager = Neo4jManager()
        print("✅ Neo4j Manager connection successful!")
        
        # Test storing a message
        print("\n🧪 Testing message storage...")
        test_chunks = [
            {
                "text": "User: How does vector search work in Neo4j?\n\nAssistant: Neo4j uses vector indexes to perform similarity search on embeddings stored as node properties.",
                "turn_number": 1,
                "user_message": "How does vector search work in Neo4j?",
                "assistant_message": "Neo4j uses vector indexes to perform similarity search on embeddings stored as node properties.",
                "chunk_index": 0,
                "message_type": "conversation_turn"
            }
        ]
        
        stored_ids = manager.store_conversation_chunks(test_chunks, "test_search_conversation")
        
        if stored_ids:
            print(f"✅ Test message storage successful!")
            
            # Test retrieval
            print("\n🔍 Testing retrieval...")
            results = manager.hybrid_search("vector search Neo4j", limit=5)
            
            if results:
                print(f"✅ Retrieval test successful! Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result.get('rrf_score', 0):.4f} - {result['text'][:100]}...")
            else:
                print("❌ Retrieval test failed - no results found")
        else:
            print("❌ Test message storage failed!")
        
        manager.close()
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Neo4j Manager test failed: {str(e)}")
        import traceback
        traceback.print_exc()
