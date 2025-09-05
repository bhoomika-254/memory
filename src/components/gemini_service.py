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
- Aggressive caching and token limits
"""

import time
from typing import List, Dict, Any, Optional
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from .token_optimizer import get_token_optimizer

# Load environment variables
load_dotenv()

class GeminiService:
    """
    Service for interacting with Google Gemini models.
    
    This service provides specialized methods for different LLM tasks
    in the memory system, each optimized for its specific purpose.
    """
    
    def __init__(self, neo4j_driver=None):
        """Initialize the Gemini service with API configuration."""
        # Get API key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize token optimizer
        self.optimizer = get_token_optimizer(neo4j_driver)
        
        # Model configurations - using Flash for everything to save costs
        self.query_model_name = "gemini-1.5-flash"
        self.compression_model_name = "gemini-1.5-flash"  
        self.generation_model_name = "gemini-1.5-flash"
        
        # Temperature settings for different tasks
        self.query_temp = 0.2       # Lower temp for query transformation
        self.compression_temp = 0.0  # Minimal temp for compression
        self.generation_temp = 0.5   # Reduced temp for responses
        
        # Retry settings
        self.max_retries = 2  # Reduced from 3 to save quota
        self.retry_delay = 1
        
        self._models = {}  # Cache for model instances
        
        self._configure_api()
        
        print(f"🔧 Optimized Gemini service configured:")
        print(f"   - All models: {self.query_model_name} (Flash for cost efficiency)")
        print(f"   - Reduced temperatures and retries")
        print(f"   - Token optimization enabled")
    
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
                    max_output_tokens=1024,  # Aggressively reduced from 2048
                    top_p=0.9,              # Reduced for more focused responses
                    top_k=40                # Reduced for faster generation
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
        Uses aggressive optimization and caching to minimize API calls.
        """
        # Use optimizer to check cache and optimize prompt
        optimized_prompt, should_skip = self.optimizer.optimize_query_transformation(
            user_query, conversation_context
        )
        
        # If we should skip transformation, return the result
        if should_skip:
            return optimized_prompt
        
        model = self._get_model(self.query_model_name, self.query_temp)
        transformed = self._generate_with_retry(model, optimized_prompt, "Query Transformation")
        
        # Cache the result
        if transformed:
            cache_content = f"{user_query}|{str(conversation_context) if conversation_context else ''}"
            self.optimizer.cache_result(cache_content, "query_transform", transformed)
        
        # Fallback to original query if transformation fails
        if not transformed:
            print("⚠️  Query transformation failed, using original query")
            return user_query
        
        # Clean up the response
        transformed = transformed.strip().strip('"').strip("'")
        print(f"🔄 Query transformed: '{user_query}' → '{transformed}'")
        return transformed
    
    def compress_context(self, retrieved_chunks: List[Dict[str, Any]], 
                        query: str) -> str:
        """
        Compress retrieved context chunks with aggressive optimization.
        """
        # Use optimizer for caching and optimization
        optimized_prompt, use_fallback = self.optimizer.optimize_context_compression(
            retrieved_chunks, query
        )
        
        # If we should use fallback, return the result directly
        if use_fallback:
            return optimized_prompt
        
        model = self._get_model(self.compression_model_name, self.compression_temp)
        compressed = self._generate_with_retry(model, optimized_prompt, "Context Compression")
        
        # Cache the result
        if compressed:
            chunk_texts = [chunk.get('text', '')[:100] for chunk in retrieved_chunks[:5]]
            cache_content = f"{query}|{'|'.join(chunk_texts)}"
            self.optimizer.cache_result(cache_content, "context_compress", compressed)
        
        if not compressed:
            # Fallback: use simple concatenation of first 2 chunks
            print("⚠️  Context compression failed, using simple fallback")
            fallback_chunks = retrieved_chunks[:2]
            return "\n\n".join([chunk["text"][:500] for chunk in fallback_chunks])
        
        print(f"📝 Compressed {len(retrieved_chunks)} chunks into summary")
        return compressed
    
    def generate_response(self, user_query: str, compressed_context: str, 
                         conversation_history: List[Dict[str, Any]]) -> str:
        """
        Generate the final conversational response with minimal token usage.
        """
        # Use optimizer to create ultra-compact prompt
        optimized_prompt = self.optimizer.optimize_response_generation(
            user_query, compressed_context, conversation_history
        )
        
        model = self._get_model(self.generation_model_name, self.generation_temp)
        response = self._generate_with_retry(model, optimized_prompt, "Response Generation")
        
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

# Global Gemini service instance will be created when Neo4j driver is available
gemini_service = None

def initialize_gemini_service(neo4j_driver=None):
    """Initialize global gemini service with Neo4j driver for optimization."""
    global gemini_service
    if gemini_service is None:
        gemini_service = GeminiService(neo4j_driver)
    return gemini_service

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
