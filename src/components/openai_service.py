"""
Azure OpenAI GPT-4o Service for Memory System.

This module handles all interactions with Azure OpenAI GPT-4o models, providing
three key functions:
1. Query transformation (rewriting vague queries into clear, searchable ones)
2. Context compression (summarizing retrieved chunks)
3. Final response generation (creating conversational responses)

Key Features:
- Azure OpenAI GPT-4o integration
- Temperature control for different tasks
- Retry logic and error handling
- Token counting and optimization
"""

import time
from typing import List, Dict, Any, Optional
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OpenAIService:
    """
    Service for interacting with Azure OpenAI GPT-4o models.
    
    This service provides specialized methods for different LLM tasks
    in the memory system, each optimized for its specific purpose.
    """
    
    def __init__(self):
        """Initialize the Azure OpenAI service with API configuration."""
        # Get Azure OpenAI credentials from environment
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI credentials not found in environment variables")
        
        # Model configurations (all using GPT-4o deployment)
        self.query_model_name = self.deployment_name
        self.compression_model_name = self.deployment_name
        self.generation_model_name = self.deployment_name
        
        # Temperature settings for different tasks
        self.query_temp = 0.3       # Low temp for query transformation
        self.compression_temp = 0.1  # Very low temp for compression
        self.generation_temp = 0.7   # Higher temp for creative responses
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1
        
        self._client = None
        
        self._configure_client()
        
        print(f"🔧 Azure OpenAI service configured with GPT-4o:")
        print(f"   - Query: {self.query_model_name} (temp: {self.query_temp})")
        print(f"   - Compression: {self.compression_model_name} (temp: {self.compression_temp})")
        print(f"   - Generation: {self.generation_model_name} (temp: {self.generation_temp})")
    
    def _configure_client(self):
        """Configure the Azure OpenAI client."""
        try:
            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
            print("✅ Azure OpenAI client configured successfully")
        except Exception as e:
            print(f"❌ Failed to configure Azure OpenAI client: {str(e)}")
            raise
    
    def _make_request_with_retry(self, messages: List[Dict[str, str]], temperature: float, task_name: str) -> Dict[str, Any]:
        """
        Make a request to Azure OpenAI with retry logic.
        
        Args:
            messages: List of message objects for the chat
            temperature: Temperature setting for generation
            task_name: Name of the task for logging
            
        Returns:
            Dictionary with 'text' and 'token_usage' fields, or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Make the API call
                response = self._client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=4000,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                # Extract the response text and usage
                if response.choices and len(response.choices) > 0:
                    result = response.choices[0].message.content
                    if result and result.strip():
                        token_usage = {
                            'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                            'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                            'total_tokens': response.usage.total_tokens if response.usage else 0
                        }
                        return {
                            'text': result.strip(),
                            'token_usage': token_usage
                        }
                
                print(f"⚠️ Empty response from Azure OpenAI for {task_name} (attempt {attempt + 1})")
                
            except Exception as e:
                print(f"❌ Error calling Azure OpenAI for {task_name} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"💥 All {self.max_retries} attempts failed for {task_name}")
                    return None
        
        return None
    
    def transform_query(self, user_query: str, conversation_context: List[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
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

        messages = [
            {"role": "system", "content": "You are a query transformation specialist. Transform user queries into clear, self-contained search queries."},
            {"role": "user", "content": prompt}
        ]
        
        result = self._make_request_with_retry(messages, self.query_temp, "Query Transformation")
        
        # Fallback to original query if transformation fails
        if not result or not result.get('text'):
            print("⚠️  Query transformation failed, using original query")
            return user_query, {'total_tokens': 0}
        
        # Clean up the response (remove quotes, extra whitespace)
        transformed = result['text'].strip().strip('"').strip("'")
        
        print(f"🔄 Query transformed: '{user_query}' → '{transformed}'")
        print(f"🔢 Tokens used: {result['token_usage']['total_tokens']}")
        return transformed, result['token_usage']
        
        # Clean up the response (remove quotes, extra whitespace)
        transformed = transformed.strip().strip('"').strip("'")
        
        print(f"🔄 Query transformed: '{user_query}' → '{transformed}'")
        return transformed
    
    def compress_context(self, retrieved_chunks: List[Dict[str, Any]], 
                        query: str) -> tuple[str, Dict[str, Any]]:
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

        messages = [
            {"role": "system", "content": "You are an expert at extractive summarization. Create concise, coherent summaries from multiple text chunks."},
            {"role": "user", "content": prompt}
        ]
        
        result = self._make_request_with_retry(messages, self.compression_temp, "Context Compression")
        
        if not result or not result.get('text'):
            # Fallback: just concatenate first few chunks
            print("⚠️  Context compression failed, using fallback")
            fallback_chunks = retrieved_chunks[:3]
            return "\n\n".join([chunk["text"] for chunk in fallback_chunks]), {'total_tokens': 0}
        
        print(f"📝 Compressed {len(retrieved_chunks)} chunks into summary")
        print(f"🔢 Tokens used: {result['token_usage']['total_tokens']}")
        return result['text'], result['token_usage']
    
    def generate_response(self, user_query: str, compressed_context: str, 
                         conversation_history: List[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
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
        prompt = f"""You are an intelligent and friendly AI assistant with access to conversation memory. You can remember and reference previous discussions to provide helpful, contextual responses.

INSTRUCTIONS:
1. Answer the user's query using both the recent conversation and retrieved memory context, but don't repeatedly keep telling the user about the previous conversation.
2. Be conversational, natural, friendly and emotional in tone. Act as a friend of the user.
3. If the user asks you a question and the context doesn't contain relevant information, say so honestly.
4. Provide helpful, accurate, and complete responses.
5. Ask clarifying questions if the query is ambiguous.
6. Listen and do what the user says. Don't contradict them.
7. Answer to the point if the query is straightforward.

RECENT CONVERSATION:
{history_str if history_str else "This is the start of our conversation."}

RETRIEVED MEMORY CONTEXT:
{compressed_context}

USER QUERY: "{user_query}"

RESPONSE:"""

        messages = [
            {"role": "system", "content": "You are an intelligent and friendly AI assistant with access to conversation memory. Be conversational, natural, and helpful."},
            {"role": "user", "content": prompt}
        ]
        
        result = self._make_request_with_retry(messages, self.generation_temp, "Response Generation")
        
        if not result or not result.get('text'):
            # Fallback response
            return "I apologize, but I'm having trouble generating a response right now. Could you please try rephrasing your question?", {'total_tokens': 0}
        
        print(f"💬 Generated response ({len(result['text'])} characters)")
        print(f"🔢 Tokens used: {result['token_usage']['total_tokens']}")
        return result['text'], result['token_usage']
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Azure OpenAI service.
        
        Returns:
            Dictionary with service status information
        """
        return {
            "api_configured": bool(self.api_key and self.endpoint),
            "client_ready": self._client is not None,
            "available_models": {
                "query_transformation": self.query_model_name,
                "context_compression": self.compression_model_name,
                "response_generation": self.generation_model_name
            },
            "temperature_settings": {
                "query": self.query_temp,
                "compression": self.compression_temp,
                "generation": self.generation_temp
            },
            "deployment_name": self.deployment_name,
            "api_version": self.api_version
        }

# Create global Azure OpenAI service instance
openai_service = OpenAIService()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Azure OpenAI service...")
    
    try:
        # Test service status
        status = openai_service.get_service_status()
        print(f"Service status: {status}")
        
        # Test query transformation
        test_query = "What was that optimization thing we talked about?"
        test_context = [
            {"user": "Tell me about packing optimization", "assistant": "Packing optimization involves algorithms like bin packing..."},
            {"user": "How does it work in practice?", "assistant": "In practice, you use heuristics like first-fit decreasing..."}
        ]
        
        transformed = openai_service.transform_query(test_query, test_context)
        print(f"Transformed query: {transformed}")
        
        # Test context compression
        test_chunks = [
            {"text": "Bin packing is an NP-hard problem that involves packing items into bins.", "rrf_score": 0.95, "appears_in": ["semantic"]},
            {"text": "First-fit decreasing is a popular heuristic for bin packing.", "rrf_score": 0.87, "appears_in": ["lexical"]},
            {"text": "Genetic algorithms can also be used for optimization problems.", "rrf_score": 0.76, "appears_in": ["semantic", "lexical"]}
        ]
        
        compressed = openai_service.compress_context(test_chunks, transformed)
        print(f"Compressed context: {compressed[:100]}...")
        
        # Test response generation
        response = openai_service.generate_response(test_query, compressed, test_context)
        print(f"Generated response: {response[:100]}...")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Make sure your Azure OpenAI credentials are valid!")
