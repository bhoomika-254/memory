"""
Text Chunking Utility for Memory System.

This module handles breaking down conversation text into manageable chunks
that can be embedded and stored in the vector database. The chunking strategy
uses fixed token sizes with percentage-based overlap to ensure context preservation.

Key Features:
- Fixed token size chunks (300-500 tokens)
- Configurable overlap (10-20%)
- Preserves sentence boundaries when possible
- Handles edge cases like very short or very long inputs
"""

import tiktoken
from typing import List, Dict, Any

class TextChunker:
    """
    Handles intelligent text chunking for memory storage.
    
    This class takes conversation text and breaks it into overlapping chunks
    that are optimal for embedding and retrieval. The chunking preserves
    context while keeping chunks within token limits.
    """
    
    def __init__(self):
        """Initialize the text chunker with tokenizer."""
        # Using GPT-3.5-turbo tokenizer as a good general-purpose tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Direct configuration values
        self.chunk_size = 400  # tokens per chunk (between 300-500)
        self.chunk_overlap = 60  # 15% overlap (60 tokens out of 400)
        self.max_chunk_length = 500  # maximum allowed chunk size
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        return len(self.tokenizer.encode(text))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving context.
        
        This is a simple sentence splitter that looks for common
        sentence endings. For production, you might want to use
        a more sophisticated NLP library like spaCy.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on common punctuation
        import re
        
        # Split on sentence endings, but keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks from input text.
        
        This is the main chunking function that:
        1. Splits text into sentences
        2. Groups sentences into chunks of target size
        3. Adds overlap between chunks
        4. Preserves metadata for each chunk
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Default metadata
        if metadata is None:
            metadata = {}
        
        # If text is smaller than chunk size, return as single chunk
        if self.count_tokens(text) <= self.chunk_size:
            return [{
                "text": text.strip(),
                "chunk_index": 0,
                "token_count": self.count_tokens(text),
                **metadata
            }]
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save the current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_index": chunk_index,
                    "token_count": current_tokens,
                    "sentence_start": max(0, i - len(current_chunk.split('. '))),
                    "sentence_end": i - 1,
                    **metadata
                })
                
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._create_overlap(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text
                current_tokens = self.count_tokens(overlap_text)
            
            # Add current sentence to chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens
            
            i += 1
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "token_count": current_tokens,
                "sentence_start": max(0, len(sentences) - len(current_chunk.split('. '))),
                "sentence_end": len(sentences) - 1,
                **metadata
            })
        
        return chunks
    
    def _create_overlap(self, text: str, overlap_tokens: int) -> str:
        """
        Create overlap text from the end of current chunk.
        
        Takes the last `overlap_tokens` worth of text from the current
        chunk to provide context for the next chunk.
        
        Args:
            text: Current chunk text
            overlap_tokens: Number of tokens to overlap
            
        Returns:
            Overlap text for next chunk
        """
        if overlap_tokens <= 0:
            return ""
        
        tokens = self.tokenizer.encode(text)
        
        # If overlap is larger than text, return the whole text
        if overlap_tokens >= len(tokens):
            return text
        
        # Take the last overlap_tokens and decode back to text
        overlap_tokens_list = tokens[-overlap_tokens:]
        overlap_text = self.tokenizer.decode(overlap_tokens_list)
        
        return overlap_text
    
    def chunk_conversation_turn(self, user_message: str, assistant_message: str, 
                               turn_number: int, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Chunk a complete conversation turn (user + assistant).
        
        This method handles a full conversation exchange and creates
        appropriately chunked and labeled pieces for storage.
        
        Args:
            user_message: User's input message
            assistant_message: Assistant's response
            turn_number: The turn number in the conversation
            conversation_id: Unique conversation identifier
            
        Returns:
            List of chunks with conversation metadata
        """
        # Combine user and assistant messages with clear markers
        full_turn = f"User: {user_message}\n\nAssistant: {assistant_message}"
        
        # Create metadata for this conversation turn
        turn_metadata = {
            "conversation_id": conversation_id,
            "turn_number": turn_number,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "message_type": "conversation_turn"
        }
        
        # Create chunks
        chunks = self.create_chunks(full_turn, turn_metadata)
        
        return chunks

# Create a global chunker instance
chunker = TextChunker()

# Example usage and testing
if __name__ == "__main__":
    # Test the chunker with sample conversation
    test_text = """
    This is a sample conversation about packing optimization. The user asked about 
    efficient packing strategies for shipping containers. We discussed various algorithms 
    including bin packing, first-fit decreasing, and genetic algorithms. The conversation 
    covered both theoretical aspects and practical implementation details. We also talked 
    about how machine learning can be applied to optimize packing efficiency in real-world 
    scenarios with irregular shaped objects.
    """
    
    sample_chunks = chunker.create_chunks(test_text, {"test": True})
    
    print(f"Created {len(sample_chunks)} chunks:")
    for i, chunk in enumerate(sample_chunks):
        print(f"\nChunk {i+1} ({chunk['token_count']} tokens):")
        print(f"Text: {chunk['text'][:100]}...")
