"""
LangGraph Orchestration Agent for Memory System.

This module implements the central orchestration logic using LangGraph to manage
conversation flow, state tracking, and routing decisions. It acts as the conductor
that coordinates all components of the memory system.

Key Features:
- State management for conversation context
- Routing logic for different query types
- Tool usage tracking and optimization
- Memory management and persistence
- Error handling and fallback strategies
"""

from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END

from src.utils.text_chunker import chunker

# Import classes instead of instances to avoid dependency issues
try:
    from src.components.gemini_service import GeminiService
    from src.components.graph_retrieval import GraphRetrieval
    from src.components.neo4j_manager import Neo4jManager
    from src.components.conversation_history_manager import ConversationHistoryManager
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False

class ConversationState(TypedDict):
    """
    State structure for the conversation agent.
    
    This defines all the information the agent tracks as the
    conversation evolves, including memory, preferences, and routing signals.
    """
    # Core conversation data
    conversation_id: str
    session_id: str
    messages: List[Dict[str, str]]  # Short-term conversation buffer
    
    # Current query processing
    current_query: str
    transformed_query: str
    retrieved_context: List[Dict[str, Any]]
    compressed_context: str
    final_response: str
    
    # Routing and decision flags
    needs_retrieval: bool
    needs_compression: bool
    retrieval_completed: bool
    compression_completed: bool
    
    # User preferences and session data
    user_preferences: Dict[str, Any]
    conversation_summary: str
    
    # Tool usage tracking
    tools_used: List[str]
    tool_results: Dict[str, Any]
    
    # Error handling
    errors: List[str]
    retry_count: int
    
    # Metadata
    timestamp: str
    turn_number: int

class MemoryOrchestrator:
    """
    LangGraph-based orchestrator for the memory-enhanced conversation system.
    
    This class manages the entire conversation flow from user input to
    final response, coordinating all components and maintaining state.
    """
    
    def __init__(self):
        """Initialize the memory orchestrator with LangGraph workflow."""
        # Initialize components if available
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Required components not available")
        
        try:
            self.gemini_service = GeminiService()
            self.neo4j_manager = Neo4jManager()
            self.graph_retrieval = GraphRetrieval()
            self.history_manager = ConversationHistoryManager()
            print("✅ All graph-based components initialized successfully")
        except Exception as e:
            raise ImportError(f"Failed to initialize components: {e}")
        
        self.graph = self._build_graph()
        self.active_conversations: Dict[str, ConversationState] = {}
        
        print("🎭 Memory orchestrator initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow for conversation processing.
        
        The graph defines the flow of conversation processing:
        1. Query analysis and transformation
        2. Retrieval routing decision
        3. Hybrid retrieval (if needed)
        4. Context compression (if needed)
        5. Response generation
        6. Memory storage and state update
        
        Returns:
            Configured StateGraph
        """
        # Create the state graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes for each processing step
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("transform_query", self._transform_query)
        workflow.add_node("route_retrieval", self._route_retrieval)
        workflow.add_node("hybrid_retrieval", self._hybrid_retrieval)
        workflow.add_node("compress_context", self._compress_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("store_memory", self._store_memory)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the workflow edges
        workflow.set_entry_point("analyze_query")
        
        # Query analysis leads to transformation
        workflow.add_edge("analyze_query", "transform_query")
        
        # After transformation, decide if retrieval is needed
        workflow.add_edge("transform_query", "route_retrieval")
        
        # Conditional routing based on retrieval needs
        workflow.add_conditional_edges(
            "route_retrieval",
            self._should_retrieve,
            {
                "retrieve": "hybrid_retrieval",
                "skip_retrieval": "generate_response"
            }
        )
        
        # After retrieval, check if compression is needed
        workflow.add_conditional_edges(
            "hybrid_retrieval",
            self._should_compress,
            {
                "compress": "compress_context",
                "skip_compression": "generate_response"
            }
        )
        
        # After compression, generate response
        workflow.add_edge("compress_context", "generate_response")
        
        # After generation, store memory
        workflow.add_edge("generate_response", "store_memory")
        
        # End after storing memory
        workflow.add_edge("store_memory", END)
        
        # Error handling routes to error node
        workflow.add_edge("handle_error", END)
        
        print("🔗 LangGraph workflow built successfully")
        return workflow.compile()
    
    def _initialize_state(self, conversation_id: str, user_query: str) -> ConversationState:
        """
        Initialize conversation state for a new or existing conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            user_query: Current user input
            
        Returns:
            Initialized ConversationState
        """
        # Get existing state or create new one
        if conversation_id in self.active_conversations:
            state = self.active_conversations[conversation_id].copy()
            state["turn_number"] += 1
        else:
            state = ConversationState(
                conversation_id=conversation_id,
                session_id=str(uuid.uuid4()),
                messages=[],
                current_query="",
                transformed_query="",
                retrieved_context=[],
                compressed_context="",
                final_response="",
                needs_retrieval=False,
                needs_compression=False,
                retrieval_completed=False,
                compression_completed=False,
                user_preferences={},
                conversation_summary="",
                tools_used=[],
                tool_results={},
                errors=[],
                retry_count=0,
                timestamp="",
                turn_number=1
            )
        
        # Update current query and timestamp
        state["current_query"] = user_query
        state["timestamp"] = datetime.now().isoformat()
        state["errors"] = []  # Reset errors for new turn
        
        return state
    
    def _analyze_query(self, state: ConversationState) -> ConversationState:
        """
        Analyze the user query to understand intent and requirements.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with analysis results
        """
        try:
            print(f"🔍 Analyzing query: '{state['current_query']}'")
            
            query = state["current_query"].strip()
            
            # Simple heuristics for routing decisions
            # In a production system, you might use a more sophisticated classifier
            
            # Check if query seems to reference previous conversation
            memory_indicators = [
                "we discussed", "we talked about", "earlier", "before",
                "that", "it", "the previous", "last time", "remember"
            ]
            
            has_memory_reference = any(indicator in query.lower() for indicator in memory_indicators)
            
            # Check if query is complex enough to warrant retrieval
            is_complex_query = len(query.split()) > 3
            
            # Set routing flags
            state["needs_retrieval"] = has_memory_reference or is_complex_query
            state["needs_compression"] = True  # Always compress if we retrieve
            
            state["tools_used"].append("query_analysis")
            
            print(f"📊 Query analysis: retrieval_needed={state['needs_retrieval']}")
            
        except Exception as e:
            state["errors"].append(f"Query analysis failed: {str(e)}")
            print(f"❌ Query analysis error: {str(e)}")
        
        return state
    
    def _transform_query(self, state: ConversationState) -> ConversationState:
        """
        Transform the query into a clear, searchable form.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with transformed query
        """
        try:
            print("🔄 Transforming query...")
            
            # Get recent conversation context
            recent_messages = state["messages"][-6:]  # Last 6 messages (3 turns)
            conversation_context = []
            
            for i in range(0, len(recent_messages), 2):
                if i + 1 < len(recent_messages):
                    conversation_context.append({
                        "user": recent_messages[i].get("content", ""),
                        "assistant": recent_messages[i + 1].get("content", "")
                    })
            
            # Transform the query
            transformed = self.gemini_service.transform_query(
                state["current_query"],
                conversation_context
            )
            
            state["transformed_query"] = transformed
            state["tools_used"].append("query_transformation")
            
            print(f"✅ Query transformed successfully")
            
        except Exception as e:
            state["errors"].append(f"Query transformation failed: {str(e)}")
            state["transformed_query"] = state["current_query"]  # Fallback
            print(f"❌ Query transformation error: {str(e)}")
        
        return state
    
    def _route_retrieval(self, state: ConversationState) -> ConversationState:
        """
        Make routing decision about whether retrieval is needed.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with routing decision
        """
        print(f"🛤️  Routing decision: needs_retrieval = {state['needs_retrieval']}")
        state["tools_used"].append("routing_decision")
        return state
    
    def _should_retrieve(self, state: ConversationState) -> str:
        """
        Conditional edge function to determine retrieval path.
        
        Args:
            state: Current conversation state
            
        Returns:
            "retrieve" or "skip_retrieval"
        """
        return "retrieve" if state["needs_retrieval"] else "skip_retrieval"
    
    def _hybrid_retrieval(self, state: ConversationState) -> ConversationState:
        """
        Perform hybrid retrieval using semantic and lexical search.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with retrieved context
        """
        try:
            print("🔍 Performing graph-based hybrid retrieval...")
            
            # Use the graph retrieval component
            results = self.graph_retrieval.hybrid_search(
                query=state["transformed_query"],
                conversation_id=state["conversation_id"]
            )
            
            state["retrieved_context"] = results
            state["retrieval_completed"] = True
            state["tools_used"].append("graph_hybrid_retrieval")
            
            print(f"✅ Retrieved {len(results)} relevant chunks from graph")
            
        except Exception as e:
            state["errors"].append(f"Hybrid retrieval failed: {str(e)}")
            state["retrieved_context"] = []
            print(f"❌ Hybrid retrieval error: {str(e)}")
        
        return state
    
    def _should_compress(self, state: ConversationState) -> str:
        """
        Conditional edge function to determine compression path.
        
        Args:
            state: Current conversation state
            
        Returns:
            "compress" or "skip_compression"
        """
        # Compress if we have retrieved context and it's substantial
        should_compress = (
            state["needs_compression"] and 
            len(state["retrieved_context"]) > 0
        )
        
        return "compress" if should_compress else "skip_compression"
    
    def _compress_context(self, state: ConversationState) -> ConversationState:
        """
        Compress retrieved context into a concise summary.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with compressed context
        """
        try:
            print("📝 Compressing context...")
            
            compressed = self.gemini_service.compress_context(
                state["retrieved_context"],
                state["transformed_query"]
            )
            
            state["compressed_context"] = compressed
            state["compression_completed"] = True
            state["tools_used"].append("context_compression")
            
            print("✅ Context compressed successfully")
            
        except Exception as e:
            state["errors"].append(f"Context compression failed: {str(e)}")
            # Fallback: concatenate first few chunks
            fallback_chunks = state["retrieved_context"][:3]
            state["compressed_context"] = "\n\n".join([
                chunk["text"] for chunk in fallback_chunks
            ])
            print(f"❌ Context compression error: {str(e)}")
        
        return state
    
    def _generate_response(self, state: ConversationState) -> ConversationState:
        """
        Generate the final response using all available context.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with generated response
        """
        try:
            print("💬 Generating response...")
            
            # Prepare conversation history
            conversation_history = []
            recent_messages = state["messages"][-10:]  # Last 10 messages
            
            for i in range(0, len(recent_messages), 2):
                if i + 1 < len(recent_messages):
                    conversation_history.append({
                        "user": recent_messages[i].get("content", ""),
                        "assistant": recent_messages[i + 1].get("content", "")
                    })
            
            # Generate response
            response = self.gemini_service.generate_response(
                user_query=state["current_query"],
                compressed_context=state["compressed_context"],
                conversation_history=conversation_history
            )
            
            state["final_response"] = response
            state["tools_used"].append("response_generation")
            
            print("✅ Response generated successfully")
            
        except Exception as e:
            state["errors"].append(f"Response generation failed: {str(e)}")
            state["final_response"] = "I apologize, but I'm having trouble generating a response right now. Could you please try again?"
            print(f"❌ Response generation error: {str(e)}")
        
        return state
    
    def _store_memory(self, state: ConversationState) -> ConversationState:
        """
        Store the conversation turn in memory for future retrieval.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state after memory storage
        """
        try:
            print("💾 Storing conversation in graph database...")
            
            # Create chunks for the current conversation turn
            chunks = chunker.chunk_conversation_turn(
                user_message=state["current_query"],
                assistant_message=state["final_response"],
                turn_number=state["turn_number"],
                conversation_id=state["conversation_id"]
            )
            
            # Store in graph database (Neo4j)
            stored_ids = self.neo4j_manager.store_conversation_chunks(
                chunks, state["conversation_id"]
            )
            
            # Update session state
            state["messages"].extend([
                {"role": "user", "content": state["current_query"]},
                {"role": "assistant", "content": state["final_response"]}
            ])
            
            # Keep only recent messages in active memory
            max_messages = 10  # Keep last 10 messages (5 conversation turns)
            if len(state["messages"]) > max_messages:
                state["messages"] = state["messages"][-max_messages:]
            
            state["tools_used"].append("graph_memory_storage")
            state["tool_results"]["stored_chunks"] = len(chunks)
            state["tool_results"]["graph_message_nodes"] = len(stored_ids)
            
            print(f"✅ Stored {len(chunks)} chunks in graph database")
            
        except Exception as e:
            state["errors"].append(f"Memory storage failed: {str(e)}")
            print(f"❌ Memory storage error: {str(e)}")
        
        return state
    
    def _handle_error(self, state: ConversationState) -> ConversationState:
        """
        Handle errors that occur during processing.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with error handling
        """
        print("🚨 Handling errors...")
        
        if state["errors"]:
            error_summary = "; ".join(state["errors"])
            state["final_response"] = f"I encountered some issues while processing your request: {error_summary}. Please try again."
        
        return state
    
    def process_conversation_turn(self, conversation_id: str, user_input: str) -> Dict[str, Any]:
        """
        Process a complete conversation turn from user input to response.
        
        This is the main public method that handles a complete conversation cycle.
        
        Args:
            conversation_id: Unique conversation identifier
            user_input: User's input message
            
        Returns:
            Dictionary with response and processing metadata
        """
        try:
            print(f"\n🎭 Processing conversation turn for: {conversation_id}")
            print(f"📝 User input: '{user_input}'")
            
            # Initialize state
            initial_state = self._initialize_state(conversation_id, user_input)
            
            # Run the LangGraph workflow
            final_state = self.graph.invoke(initial_state)
            
            # Store the updated state
            self.active_conversations[conversation_id] = final_state
            
            # Prepare response
            response = {
                "response": final_state["final_response"],
                "conversation_id": conversation_id,
                "turn_number": final_state["turn_number"],
                "tools_used": final_state["tools_used"],
                "retrieved_chunks": len(final_state["retrieved_context"]),
                "errors": final_state["errors"],
                "processing_metadata": {
                    "needs_retrieval": final_state["needs_retrieval"],
                    "retrieval_completed": final_state["retrieval_completed"],
                    "compression_completed": final_state["compression_completed"],
                    "transformed_query": final_state["transformed_query"]
                }
            }
            
            print(f"✅ Conversation turn completed successfully")
            return response
            
        except Exception as e:
            error_msg = f"Conversation processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "conversation_id": conversation_id,
                "turn_number": 0,
                "tools_used": [],
                "retrieved_chunks": 0,
                "errors": [error_msg],
                "processing_metadata": {}
            }
    
    def get_conversation_state(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get the current state of a conversation.
        
        Args:
            conversation_id: Conversation to retrieve
            
        Returns:
            ConversationState or None if not found
        """
        return self.active_conversations.get(conversation_id)
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear conversation from active memory (but not from database).
        
        Args:
            conversation_id: Conversation to clear
            
        Returns:
            True if successful
        """
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            print(f"🗑️  Cleared active conversation: {conversation_id}")
            return True
        return False
    
    def clear_conversation_memory(self, conversation_id: str) -> bool:
        """
        Clear conversation from both active memory and database.
        
        Args:
            conversation_id: Conversation to clear completely
            
        Returns:
            True if successful
        """
        # Clear from active memory
        self.clear_conversation(conversation_id)
        
        # TODO: Implement Neo4j conversation clearing
        # For now, just clear active memory
        print(f"🧹 Cleared active memory for conversation: {conversation_id}")
        print("⚠️  Note: Neo4j conversation clearing not yet implemented")
        
        return True
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation including metadata.
        
        Args:
            conversation_id: Conversation to summarize
            
        Returns:
            Dictionary with conversation summary
        """
        try:
            if self.history_manager:
                conv_details = self.history_manager.get_conversation_details(conversation_id)
                if conv_details:
                    return {
                        "conversation_id": conversation_id,
                        "summary": conv_details.get("summary", "No summary available"),
                        "topics": conv_details.get("topics", []),
                        "total_turns": conv_details.get("total_turns", 0),
                        "total_messages": conv_details.get("total_messages", 0),
                        "start_time": conv_details.get("start_time"),
                        "end_time": conv_details.get("end_time")
                    }
            
            # Fallback to basic state info
            state = self.get_conversation_state(conversation_id)
            if state:
                return {
                    "conversation_id": conversation_id,
                    "summary": f"Conversation with {len(state.get('messages', []))} messages",
                    "topics": [],
                    "total_turns": state.get("turn_number", 0),
                    "total_messages": len(state.get("messages", [])),
                    "start_time": state.get("timestamp"),
                    "end_time": state.get("timestamp")
                }
            
            return {}
            
        except Exception as e:
            print(f"❌ Error getting conversation summary: {str(e)}")
            return {}

# Create global orchestrator instance if components are available
try:
    if COMPONENTS_AVAILABLE:
        memory_orchestrator = MemoryOrchestrator()
        print("✅ Memory orchestrator created successfully")
    else:
        memory_orchestrator = None
        print("❌ Memory orchestrator not created - dependencies missing")
except Exception as e:
    memory_orchestrator = None
    print(f"❌ Failed to create memory orchestrator: {e}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing Memory Orchestrator...")
    
    try:
        # Test conversation processing
        test_conversation_id = "test_orchestrator_123"
        
        # First turn
        response1 = memory_orchestrator.process_conversation_turn(
            test_conversation_id,
            "Tell me about machine learning algorithms"
        )
        print(f"Response 1: {response1}")
        
        # Second turn with memory reference
        response2 = memory_orchestrator.process_conversation_turn(
            test_conversation_id,
            "What did we discuss about that earlier?"
        )
        print(f"Response 2: {response2}")
        
        # Check conversation state
        state = memory_orchestrator.get_conversation_state(test_conversation_id)
        print(f"Conversation state: {len(state['messages'])} messages")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print("Make sure all dependencies are properly configured!")
