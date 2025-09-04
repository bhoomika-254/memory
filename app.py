"""
Streamlit User Interface for Memory-Enhanced Conversational AI.

This module provides the web-based user interface using Streamlit, allowing users
to interact with the memory-enhanced AI system. It includes conversation display,
memory management controls, and system status information.

Key Features:
- Chat interface with conversation history
- Memory management buttons (clear conversation, clear memory)
- System status and debugging information
- Error handling and user feedback
- Session state management for UI persistence
"""

import sys
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import streamlit as st
import uuid
from typing import Dict, List, Any, Optional

# Configure Streamlit page FIRST, before any other Streamlit calls
st.set_page_config(
    page_title="Memory-Enhanced AI Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import components, handle missing dependencies gracefully
try:
    from src.components.memory_orchestrator import memory_orchestrator
    MEMORY_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    st.error(f"Memory orchestrator not available: {e}")
    MEMORY_ORCHESTRATOR_AVAILABLE = False
    memory_orchestrator = None

try:
    from src.components.gemini_service import gemini_service
    from src.components.qdrant_manager import qdrant_manager  
    from src.components.whoosh_manager import whoosh_manager
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Components not available: {e}")
    COMPONENTS_AVAILABLE = False
    gemini_service = qdrant_manager = whoosh_manager = None

class MemoryUI:
    """
    Main UI class for the memory-enhanced conversational AI system.
    
    This class manages all UI components, state management, and user interactions
    for the Streamlit application.
    """
    
    def __init__(self):
        """Initialize the UI with session state management."""
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        # Core conversation state
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # UI state
        if "show_debug" not in st.session_state:
            st.session_state.show_debug = False
        
        if "show_retrieved_context" not in st.session_state:
            st.session_state.show_retrieved_context = False
        
        # System status
        if "system_status" not in st.session_state:
            st.session_state.system_status = {}
    
    def render_header(self):
        """Render the application header with title and description."""
        st.title("🧠 Memory-Enhanced AI Chat")
        
        st.markdown("""
        Welcome to your AI assistant with **persistent memory**! I can remember our previous conversations 
        and reference them in future discussions. Feel free to ask me about anything we've talked about before.
        """)
        
        # Add system status indicator
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Conversation ID:** `{st.session_state.conversation_id[:8]}...`")
        
        with col2:
            if st.button("🆕 New Chat"):
                self._start_new_conversation()
        
        with col3:
            if st.button("ℹ️ System Status"):
                st.session_state.show_debug = not st.session_state.show_debug
    
    def render_sidebar(self):
        """Render the sidebar with controls and system information."""
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Settings
            st.subheader("⚙️ Settings")
            
            st.session_state.show_retrieved_context = st.checkbox(
                "Show retrieved context",
                value=st.session_state.show_retrieved_context,
                help="Display the context retrieved from memory (for debugging)"
            )
            
            st.divider()
            
            # System information
            st.subheader("📊 System Info")
            self._render_system_info()
    
    def _render_system_info(self):
        """Render system status information in the sidebar."""
        if not COMPONENTS_AVAILABLE:
            st.error("❌ System components not available")
            return
            
        try:
            # Get system status
            gemini_status = gemini_service.get_service_status()
            qdrant_stats = qdrant_manager.get_collection_stats()
            whoosh_stats = whoosh_manager.get_index_stats()
            
            # Gemini status
            st.markdown("**🤖 Gemini Service**")
            if gemini_status.get("api_configured"):
                st.success("✅ Connected")
            else:
                st.error("❌ Not configured")
            
            # Vector database status
            st.markdown("**🗄️ Vector Database**")
            if qdrant_stats.get("points_count", 0) > 0:
                st.success(f"✅ {qdrant_stats['points_count']} memories stored")
            else:
                st.info("📝 No memories yet")
            
            # Search index status
            st.markdown("**🔍 Search Index**")
            if whoosh_stats.get("document_count", 0) > 0:
                st.success(f"✅ {whoosh_stats['document_count']} documents indexed")
            else:
                st.info("📝 No documents indexed")
            
            # Conversation stats
            if MEMORY_ORCHESTRATOR_AVAILABLE and memory_orchestrator:
                conversation_state = memory_orchestrator.get_conversation_state(st.session_state.conversation_id)
                if conversation_state:
                    st.markdown("**💬 Current Session**")
                    st.info(f"📈 Turn {conversation_state.get('turn_number', 0)}")
                    st.info(f"💾 {len(conversation_state.get('messages', []))} messages in buffer")
            
        except Exception as e:
            st.error(f"Error getting system status: {str(e)}")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        # Display conversation history
        self._display_conversation_history()
        
        # Chat input
        user_input = st.chat_input("Ask me anything... I'll remember our conversation!")
        
        if user_input:
            self._process_user_input(user_input)
    
    def _display_conversation_history(self):
        """Display the conversation history with messages."""
        # Display messages from session state
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata for assistant messages if debug is enabled
                if (message["role"] == "assistant" and 
                    st.session_state.show_debug and 
                    "metadata" in message):
                    
                    with st.expander("🔍 Debug Information"):
                        self._display_message_metadata(message["metadata"])
    
    def _display_message_metadata(self, metadata: Dict[str, Any]):
        """Display debug metadata for a message."""
        st.json(metadata)
    
    def _process_user_input(self, user_input: str):
        """Process user input and generate response."""
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process the conversation turn
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking and searching my memory..."):
                try:
                    # Check if memory orchestrator is available
                    if not MEMORY_ORCHESTRATOR_AVAILABLE:
                        st.error("❌ Memory orchestrator not available")
                        return
                    
                    # Get response from orchestrator
                    response_data = memory_orchestrator.process_conversation_turn(
                        st.session_state.conversation_id,
                        user_input
                    )
                    
                    # Display the response
                    response_text = response_data["response"]
                    st.markdown(response_text)
                    
                    # Show retrieved context if enabled
                    if (st.session_state.show_retrieved_context and 
                        response_data.get("retrieved_chunks", 0) > 0):
                        
                        with st.expander(f"📚 Retrieved Context ({response_data['retrieved_chunks']} chunks)"):
                            # Get the actual retrieved context from the orchestrator state
                            conversation_state = memory_orchestrator.get_conversation_state(st.session_state.conversation_id)
                            if conversation_state and conversation_state.get("retrieved_context"):
                                for i, chunk in enumerate(conversation_state["retrieved_context"], 1):
                                    st.markdown(f"**Chunk {i}** (Score: {chunk.get('rrf_score', 0):.3f})")
                                    st.markdown(f"```\n{chunk['text']}\n```")
                    
                    # Show processing information if debug is enabled
                    if st.session_state.show_debug:
                        with st.expander("🔧 Processing Details"):
                            st.json(response_data)
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": response_text,
                        "metadata": response_data
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Show any errors
                    if response_data.get("errors"):
                        st.warning(f"⚠️ Some issues occurred: {'; '.join(response_data['errors'])}")
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    
                    # Add error message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "metadata": {"error": str(e)}
                    })
    
    def _start_new_conversation(self):
        """Start a new conversation session while preserving memory context."""
        # Keep the same conversation ID to maintain memory context
        # Only clear the UI session state - do NOT clear orchestrator memory
        st.session_state.messages = []
        
        st.success("🆕 Started new chat! I can still remember our previous conversations.")
        st.rerun()
    
    def render_debug_panel(self):
        """Render debug information panel if enabled."""
        if st.session_state.show_debug:
            st.subheader("🔧 Debug Information")
            
            with st.expander("System Status", expanded=True):
                if not COMPONENTS_AVAILABLE or not MEMORY_ORCHESTRATOR_AVAILABLE:
                    st.error("❌ System components not fully available")
                    return
                    
                try:
                    # Get comprehensive system status
                    status = {
                        "conversation_id": st.session_state.conversation_id,
                        "session_messages": len(st.session_state.messages),
                        "gemini_service": gemini_service.get_service_status(),
                        "qdrant_stats": qdrant_manager.get_collection_stats(),
                        "whoosh_stats": whoosh_manager.get_index_stats()
                    }
                    
                    # Get orchestrator state
                    conversation_state = memory_orchestrator.get_conversation_state(st.session_state.conversation_id)
                    if conversation_state:
                        status["orchestrator_state"] = {
                            "turn_number": conversation_state.get("turn_number"),
                            "tools_used": conversation_state.get("tools_used", []),
                            "errors": conversation_state.get("errors", [])
                        }
                    
                    st.json(status)
                    
                except Exception as e:
                    st.error(f"Error generating debug info: {str(e)}")
    
    def run(self):
        """Main method to run the Streamlit application."""
        try:
            # Render all UI components
            self.render_header()
            self.render_sidebar()
            self.render_chat_interface()
            self.render_debug_panel()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.exception(e)

def main():
    """Main function to run the Streamlit app."""
    # Initialize and run the UI
    ui = MemoryUI()
    ui.run()

if __name__ == "__main__":
    main()
