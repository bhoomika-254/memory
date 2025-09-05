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
    from src.components.openai_service import openai_service
    from src.components.neo4j_manager import Neo4jManager
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Components not available: {e}")
    COMPONENTS_AVAILABLE = False
    openai_service = None

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
        
        # UI state (removed debug and context features)
        
        # System status
        if "system_status" not in st.session_state:
            st.session_state.system_status = {}
    
    def render_header(self):
        """Render the application header with title and description."""
        st.title("🧠 Memory-Enhanced AI Chat")
        
        st.markdown("""
                    AI assistant with intelligent human like memory capabilities.
                    Note: Upon refreshing the convo id changes, and the AI will switch to a different memory context.
                    Test new chat, to see how assistant retains the memory context. 
        """)
        
        # Add system status indicator
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Conversation ID:** `{st.session_state.conversation_id[:8]}...`")
        
        with col2:
            if st.button("🆕 New Chat"):
                self._start_new_conversation()
    
    def render_sidebar(self):
        """Render the sidebar with controls and system information."""
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Navigation tabs
            tab1, tab2 = st.tabs(["⚙️ Settings", "📚 History"])
            
            with tab1:
                self._render_settings_tab()
                
            with tab2:
                self._render_history_tab()
    
    def _render_system_info(self):
        """Render system status information in the sidebar."""
        if not COMPONENTS_AVAILABLE:
            st.error("❌ System components not available")
            return
            
        try:
            # Get system status
            openai_status = openai_service.get_service_status()
            
            # Azure OpenAI status
            st.markdown("**🤖 Azure OpenAI Service**")
            if openai_status.get("api_configured"):
                st.success("✅ Connected")
            else:
                st.error("❌ Not configured")
            
            # Graph database status
            st.markdown("**�️ Neo4j Graph Database**")
            try:
                # Quick test of Neo4j connection
                neo4j_manager = Neo4jManager()
                st.success("✅ Connected to Neo4j")
                neo4j_manager.close()
            except Exception as neo4j_error:
                st.error(f"❌ Neo4j connection failed: {str(neo4j_error)}")
            
            # Conversation stats
            if MEMORY_ORCHESTRATOR_AVAILABLE and memory_orchestrator:
                conversation_state = memory_orchestrator.get_conversation_state(st.session_state.conversation_id)
                if conversation_state:
                    st.markdown("**💬 Current Session**")
                    st.info(f"📈 Turn {conversation_state.get('turn_number', 0)}")
                    st.info(f"💾 {len(conversation_state.get('messages', []))} messages in buffer")
            
        except Exception as e:
            st.error(f"Error getting system status: {str(e)}")
    
    def _render_settings_tab(self):
        """Render the settings tab in sidebar."""
        st.divider()
        
        # System information
        st.subheader("📊 System Info")
        self._render_system_info()
    
    def _render_history_tab(self):
        """Render the conversation history tab."""
        try:
            # Create temporary history manager for history
            from src.components.conversation_history_manager import ConversationHistoryManager
            history_manager = ConversationHistoryManager()
            
            st.subheader("📚 Conversation History")
            
            # Get all conversations
            with st.spinner("Loading conversations..."):
                conversations = history_manager.get_all_conversations(limit=50)
            
            if conversations:
                st.write(f"Found {len(conversations)} conversations")
                
                # Display conversations
                for conv in conversations:
                    with st.container():
                        # Create columns for conversation info, go to chat button, and delete button
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            # Conversation preview
                            conv_id_short = conv['conversation_id'][:8]
                            st.write(f"**ID:** {conv_id_short}...")
                            st.write(f"**Messages:** {conv['message_count']}")
                            st.write(f"**Last:** {conv['last_message'][:19] if len(conv['last_message']) > 19 else conv['last_message']}")
                            
                            # Show preview if available
                            if conv.get('preview'):
                                st.write(f"**Preview:** {conv['preview'][:100]}...")
                        
                        with col2:
                            # Go to chat button
                            if st.button(f"💬 Open", key=f"open_{conv['conversation_id']}"):
                                self._load_conversation(conv['conversation_id'])
                                st.rerun()
                        
                        with col3:
                            # Delete button
                            if st.button(f"🗑️ Delete", key=f"del_{conv['conversation_id']}"):
                                # Confirm deletion
                                if st.session_state.get(f"confirm_delete_{conv['conversation_id']}", False):
                                    # Actually delete the conversation
                                    try:
                                        if history_manager.delete_conversation(conv['conversation_id']):
                                            st.success(f"Deleted conversation {conv_id_short}")
                                            st.rerun()
                                        else:
                                            st.error("Failed to delete conversation")
                                    except Exception as e:
                                        st.error(f"Error deleting: {str(e)}")
                                else:
                                    # Set confirmation flag
                                    st.session_state[f"confirm_delete_{conv['conversation_id']}"] = True
                                    st.warning(f"Click delete again to confirm deletion of {conv_id_short}")
                        
                        st.divider()
            
            else:
                st.info("No conversation history found")
            
            # Clear all conversations button
            if conversations:
                st.divider()
                if st.button("🧹 Clear All History", type="secondary"):
                    if st.session_state.get("confirm_clear_all", False):
                        # Clear all conversations
                        try:
                            deleted_count = 0
                            for conv in conversations:
                                if history_manager.delete_conversation(conv['conversation_id']):
                                    deleted_count += 1
                            st.success(f"Cleared {deleted_count} conversations")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing history: {str(e)}")
                    else:
                        st.session_state["confirm_clear_all"] = True
                        st.warning("Click again to confirm clearing ALL conversation history")
            
            history_manager.close()
        
        except Exception as e:
            st.error(f"Error loading history: {str(e)}")
    
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
                
                # Show timing information for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    processing_time = message["metadata"].get("processing_time_seconds", 0)
                    retrieved_chunks = message["metadata"].get("retrieved_chunks", 0)
                    tools_used = message["metadata"].get("tools_used", [])
                    total_tokens = message["metadata"].get("total_tokens_used", 0)
                    
                    if processing_time > 0:
                        timing_info = f"⏱️ {processing_time}s"
                        if total_tokens > 0:
                            timing_info += f" • 🎫 {total_tokens:,} tokens"
                        if retrieved_chunks > 0:
                            timing_info += f" • 🔍 {retrieved_chunks} chunks"
                        if tools_used:
                            timing_info += f" • 🛠️ {len(tools_used)} tools"
                        st.caption(timing_info)
    
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
            with st.spinner("🧠 Thinking..."):
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
                    
                    # Show timing information
                    processing_time = response_data.get("processing_time_seconds", 0)
                    retrieved_chunks = response_data.get("retrieved_chunks", 0)
                    tools_used = response_data.get("tools_used", [])
                    total_tokens = response_data.get("total_tokens_used", 0)
                    
                    if processing_time > 0:
                        timing_info = f"⏱️ {processing_time}s"
                        if total_tokens > 0:
                            timing_info += f" • 🎫 {total_tokens:,} tokens"
                        if retrieved_chunks > 0:
                            timing_info += f" • 🔍 {retrieved_chunks} chunks"
                        if tools_used:
                            timing_info += f" • 🛠️ {len(tools_used)} tools"
                        st.caption(timing_info)
                    
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
    
    def _load_conversation(self, conversation_id: str):
        """Load a specific conversation and its messages into the current session."""
        try:
            # Create temporary history manager to get conversation details
            from src.components.conversation_history_manager import ConversationHistoryManager
            history_manager = ConversationHistoryManager()
            
            # Get conversation details
            conv_details = history_manager.get_conversation_details(conversation_id)
            
            if conv_details and conv_details.get('turns'):
                # Update session state
                st.session_state.conversation_id = conversation_id
                st.session_state.messages = []
                
                # Load messages in the correct format for Streamlit chat
                # Sort turns by turn number
                sorted_turns = sorted(conv_details['turns'].items(), key=lambda x: int(x[0]))
                
                for turn_number, turn_messages in sorted_turns:
                    # Get the first message from this turn (it should contain both user and assistant messages)
                    if turn_messages:
                        turn_msg = turn_messages[0]  # Take first chunk of the turn
                        
                        if turn_msg.get('user_message'):
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": turn_msg['user_message']
                            })
                        if turn_msg.get('assistant_message'):
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": turn_msg['assistant_message']
                            })
                
                st.success(f"📂 Loaded conversation {conversation_id[:8]}... ({conv_details['total_turns']} turns)")
            else:
                st.warning("No messages found for this conversation")
            
            history_manager.close()
            
        except Exception as e:
            st.error(f"Error loading conversation: {str(e)}")
    
    def run(self):
        """Main method to run the Streamlit application."""
        try:
            # Render all UI components
            self.render_header()
            self.render_sidebar()
            self.render_chat_interface()
            
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
