"""
Token Optimization Manager for Gemini Service.

This module implements aggressive token optimization strategies:
1. Query transformation caching
2. Context compression caching  
3. Conversation summary storage
4. Smart input truncation
5. Token counting and limits
"""

import hashlib
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

class TokenOptimizer:
    """
    Manages token optimization strategies to reduce Gemini API usage.
    """
    
    def __init__(self, neo4j_driver=None):
        self.driver = neo4j_driver
        self.max_context_tokens = 4000  # Aggressive limit for context
        self.max_query_tokens = 500     # Limit for query transformation
        self.max_history_tokens = 1500  # Limit for conversation history
        
        # Cache settings
        self.cache_expiry_hours = 24
        
        print("🎯 Token optimizer initialized with aggressive limits")
        print(f"   - Max context tokens: {self.max_context_tokens}")
        print(f"   - Max query tokens: {self.max_query_tokens}")
        print(f"   - Max history tokens: {self.max_history_tokens}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for English)."""
        return len(text) // 4
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit."""
        if not text:
            return text
            
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= max_tokens:
            return text
        
        # Truncate to roughly max_tokens * 4 characters
        target_chars = max_tokens * 4
        if len(text) > target_chars:
            # Try to cut at sentence boundaries
            truncated = text[:target_chars]
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            
            cut_point = max(last_period, last_newline)
            if cut_point > target_chars * 0.8:  # If we can cut at a good point
                return text[:cut_point + 1]
            else:
                return text[:target_chars] + "..."
        
        return text
    
    def _get_cache_key(self, content: str, operation: str) -> str:
        """Generate cache key for content."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{operation}_{content_hash}"
    
    def get_cached_result(self, content: str, operation: str) -> Optional[str]:
        """Get cached result if available and not expired."""
        if not self.driver:
            return None
            
        try:
            cache_key = self._get_cache_key(content, operation)
            
            query = """
            MATCH (c:Cache {cache_key: $cache_key})
            WHERE c.created_at > $expiry_time
            RETURN c.result as result
            """
            
            expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
            result = self.driver.execute_query(
                query, 
                cache_key=cache_key,
                expiry_time=expiry_time.isoformat()
            )
            
            if result.records:
                print(f"✅ Cache hit for {operation}")
                return result.records[0]["result"]
                
        except Exception as e:
            print(f"⚠️ Cache lookup failed: {e}")
        
        return None
    
    def cache_result(self, content: str, operation: str, result: str):
        """Cache operation result."""
        if not self.driver or not result:
            return
            
        try:
            cache_key = self._get_cache_key(content, operation)
            
            query = """
            MERGE (c:Cache {cache_key: $cache_key})
            SET c.content = $content,
                c.operation = $operation,
                c.result = $result,
                c.created_at = $created_at
            """
            
            self.driver.execute_query(
                query,
                cache_key=cache_key,
                content=content[:500],  # Store truncated content for reference
                operation=operation,
                result=result,
                created_at=datetime.now().isoformat()
            )
            
            print(f"💾 Cached result for {operation}")
            
        except Exception as e:
            print(f"⚠️ Cache storage failed: {e}")
    
    def optimize_query_transformation(self, user_query: str, conversation_context: List[Dict[str, Any]] = None) -> Tuple[str, bool]:
        """
        Optimize query transformation with caching and smart skipping.
        
        Returns:
            Tuple of (optimized_prompt, should_skip_transformation)
        """
        # Skip transformation for simple, clear queries
        simple_patterns = [
            user_query.lower().startswith(('what is', 'how to', 'explain', 'define', 'tell me about')),
            len(user_query.split()) <= 3,
            '?' not in user_query and len(user_query) < 30
        ]
        
        if any(simple_patterns):
            print(f"⚡ Skipping transformation for simple query: '{user_query}'")
            return user_query, True
        
        # Check cache first
        context_str = ""
        if conversation_context:
            recent_turns = conversation_context[-2:]  # Reduced from 3 to 2
            context_parts = []
            for turn in recent_turns:
                if turn.get("user"):
                    context_parts.append(f"U: {turn['user'][:100]}")  # Truncate
                if turn.get("assistant"):
                    context_parts.append(f"A: {turn['assistant'][:100]}")  # Truncate
            context_str = "\n".join(context_parts)
        
        cache_content = f"{user_query}|{context_str}"
        cached = self.get_cached_result(cache_content, "query_transform")
        if cached:
            return cached, True
        
        # Optimize prompt for fewer tokens
        optimized_prompt = f"""Transform to clear search query:

Context: {context_str[:200] if context_str else "None"}
Query: "{user_query}"
Output only the transformed query:"""
        
        return self._truncate_to_tokens(optimized_prompt, self.max_query_tokens), False
    
    def optimize_context_compression(self, retrieved_chunks: List[Dict[str, Any]], query: str) -> Tuple[str, bool]:
        """
        Optimize context compression with intelligent chunking and caching.
        
        Returns:
            Tuple of (optimized_prompt, use_fallback)
        """
        if not retrieved_chunks:
            return "No relevant context found.", True
        
        # Generate cache key from chunk content
        chunk_texts = [chunk.get('text', '')[:100] for chunk in retrieved_chunks[:5]]  # Limit chunks
        cache_content = f"{query}|{'|'.join(chunk_texts)}"
        
        cached = self.get_cached_result(cache_content, "context_compress")
        if cached:
            return cached, True
        
        # Aggressive chunk optimization
        optimized_chunks = []
        total_tokens = 0
        
        for i, chunk in enumerate(retrieved_chunks[:3]):  # Limit to top 3 chunks
            chunk_text = chunk.get('text', '')
            
            # Truncate individual chunks
            truncated_chunk = self._truncate_to_tokens(chunk_text, 800)  # 800 tokens per chunk max
            
            chunk_info = f"[{i+1}] {truncated_chunk}"
            
            chunk_tokens = self._estimate_tokens(chunk_info)
            if total_tokens + chunk_tokens > self.max_context_tokens:
                break
                
            optimized_chunks.append(chunk_info)
            total_tokens += chunk_tokens
        
        if not optimized_chunks:
            return "Context too large, using fallback.", True
        
        combined_chunks = "\n\n".join(optimized_chunks)
        
        # Ultra-compact compression prompt
        optimized_prompt = f"""Compress this context for: "{query[:100]}"

Context:
{combined_chunks}

Compressed summary:"""
        
        return self._truncate_to_tokens(optimized_prompt, self.max_context_tokens), False
    
    def optimize_response_generation(self, user_query: str, compressed_context: str, 
                                   conversation_history: List[Dict[str, Any]]) -> str:
        """
        Optimize response generation with minimal context.
        """
        # Drastically limit conversation history
        history_str = ""
        if conversation_history:
            recent_turns = conversation_history[-2:]  # Only last 2 turns
            history_parts = []
            for turn in recent_turns:
                if turn.get("user"):
                    history_parts.append(f"U: {turn['user'][:150]}")  # Heavily truncated
                if turn.get("assistant"):
                    history_parts.append(f"A: {turn['assistant'][:150]}")  # Heavily truncated
            history_str = "\n".join(history_parts)
        
        # Ultra-compact generation prompt
        optimized_prompt = f"""Context: {self._truncate_to_tokens(compressed_context, 1000)}

History: {self._truncate_to_tokens(history_str, 500) if history_str else "None"}

Q: "{user_query}"
A:"""
        
        return self._truncate_to_tokens(optimized_prompt, 2000)  # Total limit for generation
    
    def store_conversation_summary(self, conversation_id: str, summary: str):
        """Store conversation summary to avoid regeneration."""
        if not self.driver:
            return
            
        try:
            query = """
            MATCH (c:Conversation {conversation_id: $conversation_id})
            SET c.summary = $summary,
                c.summary_generated_at = $timestamp
            """
            
            self.driver.execute_query(
                query,
                conversation_id=conversation_id,
                summary=summary,
                timestamp=datetime.now().isoformat()
            )
            
            print(f"💾 Stored summary for conversation {conversation_id}")
            
        except Exception as e:
            print(f"⚠️ Failed to store summary: {e}")
    
    def get_stored_summary(self, conversation_id: str) -> Optional[str]:
        """Get stored conversation summary."""
        if not self.driver:
            return None
            
        try:
            query = """
            MATCH (c:Conversation {conversation_id: $conversation_id})
            WHERE c.summary IS NOT NULL
            RETURN c.summary as summary
            """
            
            result = self.driver.execute_query(query, conversation_id=conversation_id)
            
            if result.records:
                print(f"✅ Retrieved stored summary for {conversation_id}")
                return result.records[0]["summary"]
                
        except Exception as e:
            print(f"⚠️ Failed to retrieve summary: {e}")
        
        return None
    
    def cleanup_old_cache(self):
        """Clean up expired cache entries."""
        if not self.driver:
            return
            
        try:
            expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
            
            query = """
            MATCH (c:Cache)
            WHERE c.created_at < $expiry_time
            DETACH DELETE c
            """
            
            result = self.driver.execute_query(query, expiry_time=expiry_time.isoformat())
            print(f"🧹 Cleaned up old cache entries")
            
        except Exception as e:
            print(f"⚠️ Cache cleanup failed: {e}")

# Global instance will be created when needed
token_optimizer = None

def get_token_optimizer(neo4j_driver=None):
    """Get or create global token optimizer instance."""
    global token_optimizer
    if token_optimizer is None:
        token_optimizer = TokenOptimizer(neo4j_driver)
    return token_optimizer
