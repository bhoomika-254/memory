"""
Whoosh Lexical Search Manager for Memory System.

This module handles keyword-based (BM25) search using Whoosh, providing
the lexical component of our hybrid retrieval system. It complements
the semantic search by finding exact keyword matches and term frequency patterns.

Key Features:
- BM25 scoring for relevance ranking
- Full-text indexing of conversation chunks
- Metadata filtering and faceted search
- Automatic index management and updates
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from whoosh import index
from whoosh.fields import Schema, TEXT, ID, DATETIME, KEYWORD, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import And, Term
from whoosh.scoring import BM25F

class WhooshManager:
    """
    Manages all Whoosh lexical search operations for the memory system.
    
    This class provides keyword-based search capabilities that complement
    the semantic search. It uses BM25 scoring to find chunks with high
    keyword relevance to user queries.
    """
    
    def __init__(self):
        """Initialize the Whoosh manager with index settings."""
        self.index_dir = "whoosh_index"
        self.top_k = 20  # How many results to return from lexical search
        
        self._index = None
        self._schema = None
        
        # Ensure index directory exists
        os.makedirs(self.index_dir, exist_ok=True)
        
        print(f"🔧 Whoosh manager configured with index dir: {self.index_dir}")
    
    def _get_schema(self) -> Schema:
        """
        Define the Whoosh schema for conversation chunks.
        
        The schema defines the structure of our search index,
        including which fields are searchable and how they're analyzed.
        
        Returns:
            Whoosh Schema object
        """
        if self._schema is None:
            # Create schema with stemming analyzer for better text matching
            analyzer = StemmingAnalyzer()
            
            self._schema = Schema(
                # Unique identifier for each chunk
                id=ID(stored=True, unique=True),
                
                # Main searchable text content
                text=TEXT(stored=True, analyzer=analyzer, phrase=True),
                
                # Conversation metadata
                conversation_id=ID(stored=True),
                turn_number=NUMERIC(stored=True),
                chunk_index=NUMERIC(stored=True),
                
                # Separate fields for user and assistant messages for targeted search
                user_message=TEXT(stored=True, analyzer=analyzer),
                assistant_message=TEXT(stored=True, analyzer=analyzer),
                
                # Message type for filtering
                message_type=KEYWORD(stored=True),
                
                # Timestamp for temporal filtering
                timestamp=DATETIME(stored=True),
                
                # Token count for relevance weighting
                token_count=NUMERIC(stored=True)
            )
        
        return self._schema
    
    def _get_index(self):
        """
        Get or create the Whoosh index.
        
        Uses lazy loading to create/open the index only when needed.
        
        Returns:
            Whoosh Index object
        """
        if self._index is None:
            schema = self._get_schema()
            
            try:
                if index.exists_in(self.index_dir):
                    print(f"📂 Opening existing Whoosh index at {self.index_dir}")
                    self._index = index.open_dir(self.index_dir)
                else:
                    print(f"🏗️  Creating new Whoosh index at {self.index_dir}")
                    self._index = index.create_in(self.index_dir, schema)
                
                print(f"✅ Whoosh index ready")
                
            except Exception as e:
                print(f"❌ Failed to open/create Whoosh index: {str(e)}")
                raise
        
        return self._index
    
    def index_conversation_chunks(self, chunks: List[Dict[str, Any]], 
                                conversation_id: str) -> int:
        """
        Index conversation chunks for lexical search.
        
        This method takes the same chunk format as the Qdrant manager
        and indexes them for keyword-based search.
        
        Args:
            chunks: List of chunk dictionaries from text_chunker
            conversation_id: Unique conversation identifier
            
        Returns:
            Number of chunks successfully indexed
        """
        if not chunks:
            return 0
        
        idx = self._get_index()
        
        try:
            print(f"📝 Indexing {len(chunks)} chunks for lexical search")
            
            writer = idx.writer()
            indexed_count = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    # Create document for indexing
                    doc = {
                        "id": f"{conversation_id}_{chunk.get('chunk_index', i)}_{int(datetime.now().timestamp())}",
                        "text": chunk["text"],
                        "conversation_id": conversation_id,
                        "turn_number": chunk.get("turn_number", 0),
                        "chunk_index": chunk.get("chunk_index", i),
                        "user_message": chunk.get("user_message", ""),
                        "assistant_message": chunk.get("assistant_message", ""),
                        "message_type": chunk.get("message_type", "conversation_turn"),
                        "timestamp": datetime.now(),
                        "token_count": chunk.get("token_count", 0)
                    }
                    
                    writer.add_document(**doc)
                    indexed_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Failed to index chunk {i}: {str(e)}")
                    continue
            
            writer.commit()
            print(f"✅ Indexed {indexed_count} chunks successfully")
            
            return indexed_count
            
        except Exception as e:
            print(f"❌ Failed to index chunks: {str(e)}")
            return 0
    
    def lexical_search(self, query: str, top_k: int = None, 
                      conversation_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform BM25-based lexical search for relevant chunks.
        
        Uses keyword matching and BM25 scoring to find chunks with
        high lexical relevance to the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return (default from config)
            conversation_id: Optional filter by conversation ID
            
        Returns:
            List of search results with chunks and BM25 scores
        """
        if top_k is None:
            top_k = self.top_k
        
        idx = self._get_index()
        
        try:
            print(f"🔍 Lexical search for: '{query[:50]}...' (top {top_k})")
            
            with idx.searcher(weighting=BM25F()) as searcher:
                # Create multi-field query parser
                # Search across text, user_message, and assistant_message fields
                parser = MultifieldParser(
                    ["text", "user_message", "assistant_message"], 
                    schema=idx.schema
                )
                
                # Parse the query
                parsed_query = parser.parse(query)
                
                # Add conversation filter if specified
                if conversation_id:
                    conv_filter = Term("conversation_id", conversation_id)
                    parsed_query = And([parsed_query, conv_filter])
                
                # Perform search
                search_results = searcher.search(parsed_query, limit=top_k)
                
                # Format results
                results = []
                for result in search_results:
                    results.append({
                        "id": result["id"],
                        "score": result.score,
                        "text": result["text"],
                        "conversation_id": result["conversation_id"],
                        "turn_number": result["turn_number"],
                        "chunk_index": result["chunk_index"],
                        "timestamp": result["timestamp"].isoformat() if result["timestamp"] else None,
                        "token_count": result["token_count"],
                        "user_message": result["user_message"],
                        "assistant_message": result["assistant_message"],
                        "message_type": result["message_type"],
                        "search_type": "lexical",
                        "highlights": self._get_highlights(result, query)
                    })
                
                print(f"✅ Found {len(results)} lexical matches")
                return results
                
        except Exception as e:
            print(f"❌ Lexical search failed: {str(e)}")
            return []
    
    def _get_highlights(self, result, query: str) -> List[str]:
        """
        Extract highlighted snippets showing keyword matches.
        
        Args:
            result: Whoosh search result
            query: Original search query
            
        Returns:
            List of highlighted text snippets
        """
        try:
            # Get highlighted text for the main text field
            highlights = result.highlights("text", top=3)
            if highlights:
                return [highlights]
            else:
                # If no highlights, return first 100 characters
                text = result.get("text", "")
                return [text[:100] + "..." if len(text) > 100 else text]
        except:
            return []
    
    def get_conversation_chunks(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all indexed chunks for a specific conversation.
        
        Args:
            conversation_id: Conversation to retrieve
            
        Returns:
            List of all chunks for the conversation
        """
        idx = self._get_index()
        
        try:
            with idx.searcher() as searcher:
                # Search for all chunks in this conversation
                query = Term("conversation_id", conversation_id)
                results = searcher.search(query, limit=None)  # Get all results
                
                chunks = []
                for result in results:
                    chunks.append({
                        "id": result["id"],
                        "text": result["text"],
                        "turn_number": result["turn_number"],
                        "chunk_index": result["chunk_index"],
                        "timestamp": result["timestamp"].isoformat() if result["timestamp"] else None,
                        "token_count": result["token_count"],
                        "user_message": result["user_message"],
                        "assistant_message": result["assistant_message"]
                    })
                
                # Sort by turn number and chunk index
                chunks.sort(key=lambda x: (x["turn_number"], x["chunk_index"]))
                
                return chunks
                
        except Exception as e:
            print(f"❌ Failed to retrieve conversation chunks: {str(e)}")
            return []
    
    def clear_conversation_index(self, conversation_id: str) -> bool:
        """
        Remove all indexed chunks for a specific conversation.
        
        Args:
            conversation_id: Conversation to clear from index
            
        Returns:
            True if successful, False otherwise
        """
        idx = self._get_index()
        
        try:
            print(f"🗑️  Clearing index for conversation {conversation_id}")
            
            writer = idx.writer()
            # Delete all documents with this conversation_id
            writer.delete_by_term("conversation_id", conversation_id)
            writer.commit()
            
            print(f"✅ Cleared index for conversation {conversation_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to clear conversation index: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Whoosh index.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            idx = self._get_index()
            
            with idx.searcher() as searcher:
                doc_count = searcher.doc_count()
                
                # Count unique conversations
                conv_facets = searcher.field_terms("conversation_id")
                conversation_count = len(list(conv_facets))
                
                return {
                    "index_dir": self.index_dir,
                    "document_count": doc_count,
                    "conversation_count": conversation_count,
                    "schema_fields": list(idx.schema.names()),
                    "index_exists": True
                }
                
        except Exception as e:
            print(f"❌ Failed to get index stats: {str(e)}")
            return {"index_exists": False, "error": str(e)}

# Create global Whoosh manager instance
whoosh_manager = WhooshManager()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Whoosh manager...")
    
    try:
        # Test index stats
        stats = whoosh_manager.get_index_stats()
        print(f"Index stats: {stats}")
        
        # Test indexing a sample chunk
        sample_chunks = [{
            "text": "This is a test conversation about machine learning algorithms and optimization techniques.",
            "chunk_index": 0,
            "token_count": 15,
            "turn_number": 1,
            "user_message": "Tell me about machine learning",
            "assistant_message": "Machine learning involves algorithms and optimization...",
            "message_type": "conversation_turn"
        }]
        
        test_conversation_id = "test_conv_lexical_123"
        indexed_count = whoosh_manager.index_conversation_chunks(sample_chunks, test_conversation_id)
        print(f"Indexed {indexed_count} test chunks")
        
        # Test lexical search
        results = whoosh_manager.lexical_search("machine learning optimization", top_k=3)
        print(f"Search results: {len(results)} found")
        
        for result in results:
            print(f"- Score: {result['score']:.3f}, Text: {result['text'][:50]}...")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
