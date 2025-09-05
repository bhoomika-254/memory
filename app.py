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
    from src.components.neo4j_manager import Neo4jManager
    from src.components.conversation_history_manager import ConversationHistoryManager
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Components not available: {e}")
    COMPONENTS_AVAILABLE = False
    gemini_service = None

class MemoryUI:
    """
    Main UI class for the memory-enhanced conversational AI system.
    
    This class manages all UI components, state management, and user interactions
    for the Streamlit application.
    """
    
    def __init__(self):
        """Initialize the UI with session state management."""
        self._initialize_session_state()
        
        # Initialize history manager if available
        try:
            if COMPONENTS_AVAILABLE:
                self.history_manager = ConversationHistoryManager()
            else:
                self.history_manager = None
        except Exception as e:
            st.error(f"Failed to initialize history manager: {e}")
            self.history_manager = None
        
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
        
        # History UI state
        if "show_history" not in st.session_state:
            st.session_state.show_history = False
            
        if "selected_conversation" not in st.session_state:
            st.session_state.selected_conversation = None
            
        if "history_search_query" not in st.session_state:
            st.session_state.history_search_query = ""
            
        if "show_analytics" not in st.session_state:
            st.session_state.show_analytics = False
    
    def render_header(self):
        """Render the application header with title and description."""
        st.title("🧠 Memory-Enhanced AI Chat")
        
        st.markdown("""
                    AI assistant with intelligent human like memory capabilities.
                    Note: Upon refreshing the convo id changes, and the AI will switch to a different memory context.
                    Test new chat, to see how assistant retains the memory context. 
        """)
        
        # Add system status indicator
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Conversation ID:** `{st.session_state.conversation_id[:8]}...`")
        
        with col2:
            if st.button("🆕 New Chat"):
                self._start_new_conversation()
        
        with col3:
            if st.button("📚 History"):
                st.session_state.show_history = not st.session_state.show_history
    
    def render_sidebar(self):
        """Render the sidebar with controls and system information."""
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Navigation tabs
            tab1, tab2, tab3 = st.tabs(["⚙️ Settings", "📚 History", "📊 Analytics"])
            
            with tab1:
                self._render_settings_tab()
            
            with tab2:
                self._render_history_tab()
                
            with tab3:
                self._render_analytics_tab()
    
    def _render_system_info(self):
        """Render system status information in the sidebar."""
        if not COMPONENTS_AVAILABLE:
            st.error("❌ System components not available")
            return
            
        try:
            # Get system status
            gemini_status = gemini_service.get_service_status()
            
            # Gemini status
            st.markdown("**🤖 Gemini Service**")
            if gemini_status.get("api_configured"):
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
        st.session_state.show_retrieved_context = st.checkbox(
            "Show retrieved context",
            value=st.session_state.show_retrieved_context,
            help="Display the context retrieved from memory (for debugging)"
        )
        
        st.session_state.show_debug = st.checkbox(
            "Show debug info",
            value=st.session_state.show_debug,
            help="Show debug information and system status"
        )
        
        st.divider()
        
        # System information
        st.subheader("📊 System Info")
        self._render_system_info()
    
    def _render_history_tab(self):
        """Render the conversation history tab."""
        if not self.history_manager:
            st.error("History manager not available")
            return
        
        try:
            # Search conversations
            search_query = st.text_input(
                "🔍 Search conversations",
                value=st.session_state.history_search_query,
                placeholder="Search through your conversation history..."
            )
            
            if search_query != st.session_state.history_search_query:
                st.session_state.history_search_query = search_query
                st.rerun()
            
            # Date filters
            col1, col2 = st.columns(2)
            with col1:
                date_from = st.date_input("From date", value=None)
            with col2:
                date_to = st.date_input("To date", value=None)
            
            # Get conversations
            if search_query:
                conversations = self.history_manager.search_conversations(
                    query=search_query,
                    date_from=date_from.isoformat() if date_from else None,
                    date_to=date_to.isoformat() if date_to else None,
                    limit=20
                )
                st.write(f"🔍 Found {len(conversations)} matching conversations")
            else:
                conversations = self.history_manager.get_all_conversations(limit=20)
                st.write(f"📚 Recent conversations ({len(conversations)})")
            
            # Display conversations
            for conv in conversations:
                with st.expander(f"💬 {conv.get('preview', 'No preview')[:50]}..."):
                    if 'conversation_id' in conv:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**ID:** `{conv['conversation_id'][:8]}...`")
                            if 'topics' in conv:
                                topics = ", ".join(conv['topics'][:3])
                                st.write(f"**Topics:** {topics}")
                        
                        with col2:
                            if 'total_turns' in conv:
                                st.metric("Turns", conv['total_turns'])
                            if 'message_count' in conv:
                                st.metric("Messages", conv['message_count'])
                        
                        with col3:
                            if st.button("👁️ View", key=f"view_{conv['conversation_id']}"):
                                st.session_state.selected_conversation = conv['conversation_id']
                                st.rerun()
                            
                            if st.button("🗑️ Delete", key=f"del_{conv['conversation_id']}"):
                                if self.history_manager.delete_conversation(conv['conversation_id']):
                                    st.success("Conversation deleted!")
                                    st.rerun()
                    else:
                        # This is a search result
                        st.write(f"**Text:** {conv.get('text', '')[:200]}...")
                        st.write(f"**Timestamp:** {conv.get('timestamp', 'Unknown')}")
                        st.write(f"**Relevance:** {conv.get('relevance_score', 0):.3f}")
            
            # Export functionality
            st.divider()
            st.subheader("📤 Export Data")
            
            col1, col2 = st.columns(2)
            with col1:
                export_format = st.selectbox("Format", ["json", "csv"])
            with col2:
                include_embeddings = st.checkbox("Include embeddings")
            
            if st.button("📤 Export All Conversations"):
                with st.spinner("Exporting conversations..."):
                    filepath = self.history_manager.export_conversation_data(
                        format_type=export_format,
                        include_embeddings=include_embeddings
                    )
                    if filepath:
                        st.success(f"✅ Exported to: `{filepath}`")
                    else:
                        st.error("❌ Export failed")
        
        except Exception as e:
            st.error(f"Error in history tab: {str(e)}")
    
    def _render_analytics_tab(self):
        """Render the analytics tab."""
        if not self.history_manager:
            st.error("Analytics not available")
            return
        
        try:
            # Analytics period selector
            period_days = st.selectbox(
                "📊 Analytics Period",
                [7, 14, 30, 90],
                index=2,  # Default to 30 days
                format_func=lambda x: f"Last {x} days"
            )
            
            # Get analytics
            with st.spinner("Loading analytics..."):
                analytics = self.history_manager.get_conversation_analytics(days=period_days)
            
            if analytics:
                # Key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Conversations",
                        analytics.get('total_conversations', 0)
                    )
                
                with col2:
                    st.metric(
                        "Total Messages",
                        analytics.get('total_messages', 0)
                    )
                
                with col3:
                    st.metric(
                        "Avg Turns/Conv",
                        analytics.get('avg_turns_per_conversation', 0)
                    )
                
                # Daily activity chart
                daily_activity = analytics.get('daily_activity', [])
                if daily_activity:
                    st.subheader("📈 Daily Activity")
                    
                    # Create a simple chart using Streamlit's built-in charting
                    import pandas as pd
                    df = pd.DataFrame(daily_activity)
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        st.line_chart(df['message_count'])
                
                # Trending topics
                trending_topics = analytics.get('trending_topics', [])
                if trending_topics:
                    st.subheader("🔥 Trending Topics")
                    
                    for topic in trending_topics[:5]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{topic['topic'].title()}**")
                        with col2:
                            st.write(f"{topic['conversation_count']} convs ({topic['percentage']}%)")
            
            else:
                st.info("No analytics data available")
        
        except Exception as e:
            st.error(f"Error loading analytics: {str(e)}")
    
    def _render_conversation_view(self):
        """Render detailed view of a selected conversation."""
        if not self.history_manager:
            st.error("History manager not available")
            return
        
        try:
            # Header with back button
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button("← Back"):
                    st.session_state.selected_conversation = None
                    st.rerun()
            
            with col2:
                st.subheader(f"💬 Conversation Details")
            
            # Get conversation details
            with st.spinner("Loading conversation..."):
                conv_details = self.history_manager.get_conversation_details(
                    st.session_state.selected_conversation
                )
            
            if not conv_details:
                st.error("Could not load conversation details")
                return
            
            # Conversation metadata
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Turns", conv_details.get('total_turns', 0))
            
            with col2:
                st.metric("Total Messages", conv_details.get('total_messages', 0))
            
            with col3:
                if conv_details.get('start_time'):
                    start_date = conv_details['start_time'][:10]  # Extract date
                    st.metric("Start Date", start_date)
            
            with col4:
                topics = conv_details.get('topics', [])
                if topics:
                    st.metric("Topics", len(topics))
            
            # Conversation summary
            if conv_details.get('summary'):
                st.subheader("📝 Summary")
                st.write(conv_details['summary'])
            
            # Topics
            if topics:
                st.subheader("🏷️ Topics")
                topic_cols = st.columns(min(len(topics), 3))
                for i, topic in enumerate(topics[:3]):
                    with topic_cols[i]:
                        st.info(f"#{topic}")
            
            st.divider()
            
            # Conversation turns
            st.subheader("💬 Conversation")
            
            turns_data = conv_details.get('turns', {})
            if turns_data:
                for turn_num in sorted(turns_data.keys()):
                    turn_messages = turns_data[turn_num]
                    
                    # Get the full turn data
                    user_msg = ""
                    assistant_msg = ""
                    
                    for msg in turn_messages:
                        if msg.get('user_message'):
                            user_msg = msg['user_message']
                        if msg.get('assistant_message'):
                            assistant_msg = msg['assistant_message']
                    
                    if user_msg or assistant_msg:
                        with st.expander(f"Turn {turn_num}", expanded=True):
                            if user_msg:
                                with st.chat_message("user"):
                                    st.write(user_msg)
                            
                            if assistant_msg:
                                with st.chat_message("assistant"):
                                    st.write(assistant_msg)
            
            # Export this conversation
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 Export This Conversation"):
                    filepath = self.history_manager.export_conversation_data(
                        conversation_ids=[st.session_state.selected_conversation],
                        format_type="json"
                    )
                    if filepath:
                        st.success(f"✅ Exported to: `{filepath}`")
            
            with col2:
                if st.button("🗑️ Delete This Conversation"):
                    if st.button("⚠️ Confirm Delete", type="secondary"):
                        if self.history_manager.delete_conversation(st.session_state.selected_conversation):
                            st.success("Conversation deleted!")
                            st.session_state.selected_conversation = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error rendering conversation view: {str(e)}")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        # Check if viewing a specific conversation
        if st.session_state.selected_conversation:
            self._render_conversation_view()
        else:
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
                        "neo4j_available": True  # Neo4j status checked via connection test
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
        
        # History panel
        if st.session_state.show_history:
            self._render_history_panel()
    
    def _render_history_panel(self):
        """Render the history panel in main area."""
        st.subheader("📚 Conversation History")
        
        if not self.history_manager:
            st.error("History manager not available")
            return
        
        try:
            # Quick search
            search_col, filter_col = st.columns([3, 1])
            
            with search_col:
                quick_search = st.text_input(
                    "🔍 Quick search",
                    placeholder="Search conversations..."
                )
            
            with filter_col:
                time_filter = st.selectbox(
                    "Time",
                    ["All time", "Last 7 days", "Last 30 days", "Last 90 days"]
                )
            
            # Get conversations based on search
            if quick_search:
                conversations = self.history_manager.search_conversations(
                    query=quick_search,
                    limit=10
                )
            else:
                conversations = self.history_manager.get_all_conversations(limit=10)
            
            # Display in a more compact format
            for i, conv in enumerate(conversations):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    preview = conv.get('preview', 'No preview')
                    st.write(f"**{preview[:60]}...**")
                    
                    if 'topics' in conv and conv['topics']:
                        topics_str = " • ".join([f"#{topic}" for topic in conv['topics'][:2]])
                        st.caption(topics_str)
                
                with col2:
                    if 'last_message' in conv:
                        date_str = conv['last_message'][:10]  # Extract date
                        st.caption(f"📅 {date_str}")
                    
                    if 'total_turns' in conv:
                        st.caption(f"💬 {conv['total_turns']} turns")
                
                with col3:
                    if st.button("View", key=f"hist_view_{i}"):
                        st.session_state.selected_conversation = conv.get('conversation_id')
                        st.session_state.show_history = False
                        st.rerun()
            
            if not conversations:
                st.info("No conversations found")
        
        except Exception as e:
            st.error(f"Error in history panel: {str(e)}")
    
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
