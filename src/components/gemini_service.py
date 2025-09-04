"""
Google Gemini LLM Service for Memory System.

This module handles all interactions with Google's Gemini models, providing
three key functions:
1. Query transformation (rewriting vague queries into clear, searchable ones)
2. Context compression (summarizing retrieved chunks)
3. Final response generation (creating conversational responses)

Key Features:
- Multiple Gemini model support (Pro, Flash)
- Temperature control for different tasks
- Retry logic and error handling
- Token counting and optimization
"""

import time
from typing import List, Dict, Any, Optional
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiService:
    """
    Service for interacting with Google Gemini models.
    
    This service provides specialized methods for different LLM tasks
    in the memory system, each optimized for its specific purpose.
    """
    
    def __init__(self):
        """Initialize the Gemini service with API configuration."""
        # Get API key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Model configurations
        self.query_model_name = "gemini-1.5-flash"
        self.compression_model_name = "gemini-1.5-flash"  
        self.generation_model_name = "gemini-1.5-flash"
        
        # Temperature settings for different tasks
        self.query_temp = 0.3       # Low temp for query transformation
        self.compression_temp = 0.1  # Very low temp for compression
        self.generation_temp = 0.7   # Higher temp for creative responses
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1
        
        self._models = {}  # Cache for model instances
        
        self._configure_api()
        
        print(f"🔧 Gemini service configured with models:")
        print(f"   - Query: {self.query_model_name} (temp: {self.query_temp})")
        print(f"   - Compression: {self.compression_model_name} (temp: {self.compression_temp})")
        print(f"   - Generation: {self.generation_model_name} (temp: {self.generation_temp})")
    
    def _configure_api(self):
        """Configure the Google Generative AI API."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        try:
            genai.configure(api_key=self.api_key)
            print("✅ Gemini API configured successfully")
        except Exception as e:
            print(f"❌ Failed to configure Gemini API: {str(e)}")
            raise
    
    def _get_model(self, model_name: str, temperature: float):
        """
        Get or create a Gemini model instance with caching.
        
        Args:
            model_name: Name of the Gemini model
            temperature: Temperature setting for generation
            
        Returns:
            Configured GenerativeModel instance
        """
        cache_key = f"{model_name}_{temperature}"
        
        if cache_key not in self._models:
            try:
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=8192,  # Generous limit for all tasks
                    top_p=0.95,
                    top_k=64
                )
                
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                self._models[cache_key] = model
                print(f"📥 Loaded model: {model_name} with temperature {temperature}")
                
            except Exception as e:
                print(f"❌ Failed to load model {model_name}: {str(e)}")
                raise
        
        return self._models[cache_key]
    
    def _generate_with_retry(self, model, prompt: str, task_name: str) -> Optional[str]:
        """
        Generate text with retry logic for reliability.
        
        Args:
            model: Gemini model instance
            prompt: Input prompt
            task_name: Name of the task for logging
            
        Returns:
            Generated text or None if all retries failed
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"🤖 {task_name} - Attempt {attempt}/{self.max_retries}")
                
                response = model.generate_content(prompt)
                
                if response.text:
                    print(f"✅ {task_name} completed successfully")
                    return response.text.strip()
                else:
                    print(f"⚠️  {task_name} - Empty response received")
                    
            except Exception as e:
                print(f"❌ {task_name} - Attempt {attempt} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    print(f"⏳ Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"💥 {task_name} - All attempts failed")
        
        return None
    
    def transform_query(self, user_query: str, conversation_context: List[Dict[str, Any]] = None) -> str:
        """
        Transform a user query into a clear, self-contained search query.
        
        This function takes potentially vague or contextual queries and rewrites
        them into clear, specific queries suitable for retrieval.
        
        Args:
            user_query: Original user input
            conversation_context: Recent conversation turns for context
            
        Returns:
            Transformed, clear query string
        """
        # Prepare conversation context
        context_str = ""
        if conversation_context:
            recent_turns = conversation_context[-3:]  # Last 3 turns for context
            context_parts = []
            for turn in recent_turns:
                if turn.get("user"):
                    context_parts.append(f"User: {turn['user']}")
                if turn.get("assistant"):
                    context_parts.append(f"Assistant: {turn['assistant']}")
            context_str = "\n".join(context_parts)
        
        # Create transformation prompt
        prompt = f"""You are a query transformation specialist. Your job is to rewrite user queries into clear, self-contained search queries that can effectively retrieve relevant information from a conversation memory database.

TASK: Transform the user's query into a clear, specific search query.

RULES:
1. Make the query self-contained (no ambiguous references like "it", "that", "the previous discussion")
2. Expand abbreviations and unclear terms
3. Add context from recent conversation if needed
4. Keep the core intent intact
5. Make it suitable for both semantic and keyword search
6. Output ONLY the transformed query, no explanations

RECENT CONVERSATION CONTEXT:
{context_str if context_str else "No recent conversation context available."}

USER QUERY: "{user_query}"

TRANSFORMED QUERY:"""

        model = self._get_model(self.query_model_name, self.query_temp)
        transformed = self._generate_with_retry(model, prompt, "Query Transformation")
        
        # Fallback to original query if transformation fails
        if not transformed:
            print("⚠️  Query transformation failed, using original query")
            return user_query
        
        # Clean up the response (remove quotes, extra whitespace)
        transformed = transformed.strip().strip('"').strip("'")
        
        print(f"🔄 Query transformed: '{user_query}' → '{transformed}'")
        return transformed
    
    def compress_context(self, retrieved_chunks: List[Dict[str, Any]], 
                        query: str) -> str:
        """
        Compress retrieved context chunks into a concise summary.
        
        This function takes multiple retrieved chunks and creates a coherent,
        compressed summary that preserves the most relevant information.
        
        Args:
            retrieved_chunks: List of chunks from hybrid retrieval
            query: The search query for relevance filtering
            
        Returns:
            Compressed context summary
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        # Prepare chunks for compression
        chunks_text = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_info = f"[Chunk {i}]"
            
            # Add source information
            if chunk.get("appears_in"):
                sources = ", ".join(chunk["appears_in"])
                chunk_info += f" (Sources: {sources})"
            
            # Add relevance scores
            if chunk.get("rrf_score"):
                chunk_info += f" (Relevance: {chunk['rrf_score']:.3f})"
            
            chunk_info += f"\n{chunk['text']}\n"
            chunks_text.append(chunk_info)
        
        combined_chunks = "\n".join(chunks_text)
        
        # Create compression prompt
        prompt = f"""You are an expert at extractive summarization. Your job is to compress multiple conversation chunks into a concise, coherent summary that preserves the most important information relevant to the user's query.

TASK: Create a compressed summary of the retrieved conversation chunks.

RULES:
1. Preserve key facts, concepts, and details relevant to the query
2. Maintain logical flow and coherence
3. Remove redundant information across chunks
4. Keep important context like examples, explanations, or specific details
5. Aim for 2-3 paragraphs maximum
6. Focus on information that directly relates to the query
7. Preserve any specific numbers, names, or technical terms

USER QUERY: "{query}"

RETRIEVED CHUNKS:
{combined_chunks}

COMPRESSED SUMMARY:"""

        model = self._get_model(self.compression_model_name, self.compression_temp)
        compressed = self._generate_with_retry(model, prompt, "Context Compression")
        
        if not compressed:
            # Fallback: just concatenate first few chunks
            print("⚠️  Context compression failed, using fallback")
            fallback_chunks = retrieved_chunks[:3]
            return "\n\n".join([chunk["text"] for chunk in fallback_chunks])
        
        print(f"📝 Compressed {len(retrieved_chunks)} chunks into summary")
        return compressed
    
    def generate_response(self, user_query: str, compressed_context: str, 
                         conversation_history: List[Dict[str, Any]]) -> str:
        """
        Generate the final conversational response.
        
        This is the main generation function that creates the assistant's
        response using the compressed context and conversation history.
        
        Args:
            user_query: Original user query
            compressed_context: Compressed retrieved context
            conversation_history: Recent conversation turns
            
        Returns:
            Generated response string
        """
        # Prepare conversation history
        history_str = ""
        if conversation_history:
            recent_turns = conversation_history[-5:]  # Last 5 turns for context
            history_parts = []
            for turn in recent_turns:
                if turn.get("user"):
                    history_parts.append(f"User: {turn['user']}")
                if turn.get("assistant"):
                    history_parts.append(f"Assistant: {turn['assistant']}")
            history_str = "\n".join(history_parts)
        
        # Create generation prompt
        prompt = f"""You are an intelligent AI assistant with access to conversation memory. You can remember and reference previous discussions to provide helpful, contextual responses.

INSTRUCTIONS:
1. Answer the user's query using both the recent conversation and retrieved memory context
2. Be conversational, natural and friendly in tone. Act as a friend of the user.
3. If the context doesn't contain relevant information, say so honestly
5. Provide helpful, accurate, and complete responses
6. Ask clarifying questions if the query is ambiguous
7. Listen and do what the user says. Don't contradict them.

RECENT CONVERSATION:
{history_str if history_str else "This is the start of our conversation."}

RETRIEVED MEMORY CONTEXT:
{compressed_context}

USER QUERY: "{user_query}"

RESPONSE:"""

        model = self._get_model(self.generation_model_name, self.generation_temp)
        response = self._generate_with_retry(model, prompt, "Response Generation")
        
        if not response:
            # Fallback response
            return "I apologize, but I'm having trouble generating a response right now. Could you please try rephrasing your question?"
        
        print(f"💬 Generated response ({len(response)} characters)")
        return response
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Gemini service.
        
        Returns:
            Dictionary with service status information
        """
        return {
            "api_configured": bool(self.api_key),
            "models_loaded": len(self._models),
            "available_models": {
                "query_transformation": self.query_model_name,
                "context_compression": self.compression_model_name,
                "response_generation": self.generation_model_name
            },
            "temperature_settings": {
                "query": self.query_temp,
                "compression": self.compression_temp,
                "generation": self.generation_temp
            }
        }

# Create global Gemini service instance
gemini_service = GeminiService()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Gemini service...")
    
    try:
        # Test service status
        status = gemini_service.get_service_status()
        print(f"Service status: {status}")
        
        # Test query transformation
        test_query = "What was that optimization thing we talked about?"
        test_context = [
            {"user": "Tell me about packing optimization", "assistant": "Packing optimization involves algorithms like bin packing..."},
            {"user": "How does it work in practice?", "assistant": "In practice, you use heuristics like first-fit decreasing..."}
        ]
        
        transformed = gemini_service.transform_query(test_query, test_context)
        print(f"Transformed query: {transformed}")
        
        # Test context compression
        test_chunks = [
            {"text": "Bin packing is an NP-hard problem that involves packing items into bins.", "rrf_score": 0.95, "appears_in": ["semantic"]},
            {"text": "First-fit decreasing is a popular heuristic for bin packing.", "rrf_score": 0.87, "appears_in": ["lexical"]},
            {"text": "Genetic algorithms can also be used for optimization problems.", "rrf_score": 0.76, "appears_in": ["semantic", "lexical"]}
        ]
        
        compressed = gemini_service.compress_context(test_chunks, transformed)
        print(f"Compressed context: {compressed[:100]}...")
        
        # Test response generation
        response = gemini_service.generate_response(test_query, compressed, test_context)
        print(f"Generated response: {response[:100]}...")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Make sure your GEMINI_API_KEY is valid!")
