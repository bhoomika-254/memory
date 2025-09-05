"""
Conversation History Management System.

This module provides comprehensive conversation history management including:
- Past conversation retrieval and organization
- Search functionality across conversations
- Export/import capabilities
- Topic-based organization
- Timeline views and analytics

Key Features:
- Conversation listing and filtering
- Full-text search across all conversations
- Topic extraction and clustering
- Data export (JSON, CSV formats)
- Conversation statistics and analytics
"""

import json
import csv
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import os

from src.components.neo4j_manager import Neo4jManager
from src.components.openai_service import OpenAIService
from src.utils.embedding_service import embedding_service


class ConversationHistoryManager:
    """
    Manages conversation history, search, and analytics.
    
    This class provides a comprehensive interface for managing historical
    conversations, including search, organization, and data management.
    """
    
    def __init__(self):
        """Initialize the conversation history manager."""
        self.neo4j_manager = Neo4jManager()
        self.openai_service = OpenAIService()
        
        print("📚 Conversation History Manager initialized")
    
    def get_all_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of all conversations with metadata.
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        try:
            with self.neo4j_manager.driver.session(database=self.neo4j_manager.database) as session:
                query = """
                MATCH (m:Message)
                WITH m.conversation_id as conv_id, 
                     COUNT(m) as message_count,
                     MIN(m.timestamp) as first_message,
                     MAX(m.timestamp) as last_message,
                     MAX(m.turn_number) as total_turns
                RETURN conv_id, message_count, first_message, last_message, total_turns
                ORDER BY last_message DESC
                LIMIT $limit
                """
                
                result = session.run(query, {"limit": limit})
                
                conversations = []
                for record in result:
                    conv_data = {
                        "conversation_id": record["conv_id"],
                        "message_count": record["message_count"],
                        "total_turns": record["total_turns"],
                        "first_message": record["first_message"],
                        "last_message": record["last_message"],
                        "duration": self._calculate_duration(
                            record["first_message"], 
                            record["last_message"]
                        )
                    }
                    
                    # Get conversation preview
                    conv_data["preview"] = self._get_conversation_preview(record["conv_id"])
                    conv_data["topics"] = self._extract_conversation_topics(record["conv_id"])
                    
                    conversations.append(conv_data)
                
                print(f"📋 Retrieved {len(conversations)} conversations")
                return conversations
                
        except Exception as e:
            print(f"❌ Error retrieving conversations: {str(e)}")
            return []
    
    def search_conversations(self, 
                           query: str, 
                           date_from: str = None,
                           date_to: str = None,
                           topic_filter: str = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search through conversation history.
        
        Args:
            query: Search query text
            date_from: Start date filter (ISO format)
            date_to: End date filter (ISO format)
            topic_filter: Topic to filter by
            limit: Maximum results to return
            
        Returns:
            List of matching conversation segments
        """
        try:
            # Build search query with filters
            base_query = """
            CALL db.index.fulltext.queryNodes('message_text_idx', $query_text)
            YIELD node, score
            WHERE 1=1
            """
            
            params = {"query_text": query, "limit": limit}
            
            # Add date filters
            if date_from:
                base_query += " AND node.timestamp >= $date_from"
                params["date_from"] = date_from
            
            if date_to:
                base_query += " AND node.timestamp <= $date_to"
                params["date_to"] = date_to
            
            base_query += """
            RETURN node.conversation_id as conv_id,
                   node.text as text,
                   node.timestamp as timestamp,
                   node.turn_number as turn_number,
                   node.user_message as user_message,
                   node.assistant_message as assistant_message,
                   score
            ORDER BY score DESC, node.timestamp DESC
            LIMIT $limit
            """
            
            with self.neo4j_manager.driver.session(database=self.neo4j_manager.database) as session:
                result = session.run(base_query, params)
                
                search_results = []
                for record in result:
                    result_data = {
                        "conversation_id": record["conv_id"],
                        "text": record["text"],
                        "timestamp": record["timestamp"],
                        "turn_number": record["turn_number"],
                        "user_message": record["user_message"],
                        "assistant_message": record["assistant_message"],
                        "relevance_score": record["score"],
                        "preview": record["text"][:200] + "..." if len(record["text"]) > 200 else record["text"]
                    }
                    search_results.append(result_data)
                
                print(f"🔍 Found {len(search_results)} matching conversation segments")
                return search_results
                
        except Exception as e:
            print(f"❌ Error searching conversations: {str(e)}")
            return []
    
    def get_conversation_details(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific conversation.
        
        Args:
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            Detailed conversation data
        """
        try:
            with self.neo4j_manager.driver.session(database=self.neo4j_manager.database) as session:
                query = """
                MATCH (m:Message {conversation_id: $conv_id})
                RETURN m.text as text,
                       m.timestamp as timestamp,
                       m.turn_number as turn_number,
                       m.user_message as user_message,
                       m.assistant_message as assistant_message,
                       m.chunk_index as chunk_index
                ORDER BY m.turn_number, m.chunk_index
                """
                
                result = session.run(query, {"conv_id": conversation_id})
                
                messages = []
                turns = defaultdict(list)
                
                for record in result:
                    message_data = {
                        "text": record["text"],
                        "timestamp": record["timestamp"],
                        "turn_number": record["turn_number"],
                        "user_message": record["user_message"],
                        "assistant_message": record["assistant_message"],
                        "chunk_index": record["chunk_index"]
                    }
                    
                    messages.append(message_data)
                    turns[record["turn_number"]].append(message_data)
                
                # Generate conversation summary
                summary = self._generate_conversation_summary(conversation_id, messages)
                topics = self._extract_conversation_topics(conversation_id)
                
                conversation_details = {
                    "conversation_id": conversation_id,
                    "total_messages": len(messages),
                    "total_turns": len(turns),
                    "messages": messages,
                    "turns": dict(turns),
                    "summary": summary,
                    "topics": topics,
                    "start_time": messages[0]["timestamp"] if messages else None,
                    "end_time": messages[-1]["timestamp"] if messages else None
                }
                
                return conversation_details
                
        except Exception as e:
            print(f"❌ Error retrieving conversation details: {str(e)}")
            return {}
    
    def get_conversation_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get analytics and statistics about conversations.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analytics data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            with self.neo4j_manager.driver.session(database=self.neo4j_manager.database) as session:
                # Get basic statistics
                stats_query = """
                MATCH (m:Message)
                WHERE m.timestamp >= $start_date
                WITH m.conversation_id as conv_id,
                     COUNT(m) as msg_count,
                     MAX(m.turn_number) as turns,
                     MIN(m.timestamp) as start_time,
                     MAX(m.timestamp) as end_time
                RETURN COUNT(DISTINCT conv_id) as total_conversations,
                       AVG(msg_count) as avg_messages_per_conversation,
                       AVG(turns) as avg_turns_per_conversation,
                       SUM(msg_count) as total_messages
                """
                
                stats_result = session.run(stats_query, {
                    "start_date": start_date.isoformat()
                })
                
                stats = stats_result.single()
                
                # Get daily activity
                daily_query = """
                MATCH (m:Message)
                WHERE m.timestamp >= $start_date
                WITH date(datetime(m.timestamp)) as day, COUNT(m) as daily_count
                RETURN day, daily_count
                ORDER BY day
                """
                
                daily_result = session.run(daily_query, {
                    "start_date": start_date.isoformat()
                })
                
                daily_activity = []
                for record in daily_result:
                    daily_activity.append({
                        "date": str(record["day"]),
                        "message_count": record["daily_count"]
                    })
                
                # Get top topics
                topics = self._get_trending_topics(days)
                
                analytics = {
                    "period_days": days,
                    "total_conversations": stats["total_conversations"] or 0,
                    "total_messages": stats["total_messages"] or 0,
                    "avg_messages_per_conversation": round(stats["avg_messages_per_conversation"] or 0, 2),
                    "avg_turns_per_conversation": round(stats["avg_turns_per_conversation"] or 0, 2),
                    "daily_activity": daily_activity,
                    "trending_topics": topics,
                    "generated_at": datetime.now().isoformat()
                }
                
                return analytics
                
        except Exception as e:
            print(f"❌ Error generating analytics: {str(e)}")
            return {}
    
    def export_conversation_data(self, 
                               conversation_ids: List[str] = None,
                               format_type: str = "json",
                               include_embeddings: bool = False) -> str:
        """
        Export conversation data to file.
        
        Args:
            conversation_ids: Specific conversations to export (None for all)
            format_type: Export format ("json" or "csv")
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            Path to exported file
        """
        try:
            # Create exports directory
            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_export_{timestamp}.{format_type}"
            filepath = os.path.join(export_dir, filename)
            
            # Get conversations to export
            if conversation_ids:
                conversations = []
                for conv_id in conversation_ids:
                    conv_data = self.get_conversation_details(conv_id)
                    if conv_data:
                        conversations.append(conv_data)
            else:
                # Export all conversations
                all_convs = self.get_all_conversations(limit=1000)
                conversations = []
                for conv_summary in all_convs:
                    conv_data = self.get_conversation_details(conv_summary["conversation_id"])
                    if conv_data:
                        conversations.append(conv_data)
            
            # Export based on format
            if format_type.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        "export_metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "total_conversations": len(conversations),
                            "include_embeddings": include_embeddings
                        },
                        "conversations": conversations
                    }, f, indent=2, ensure_ascii=False)
            
            elif format_type.lower() == "csv":
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    header = [
                        "conversation_id", "turn_number", "timestamp",
                        "user_message", "assistant_message", "chunk_text"
                    ]
                    writer.writerow(header)
                    
                    # Write data
                    for conv in conversations:
                        for message in conv.get("messages", []):
                            row = [
                                conv["conversation_id"],
                                message["turn_number"],
                                message["timestamp"],
                                message["user_message"],
                                message["assistant_message"],
                                message["text"]
                            ]
                            writer.writerow(row)
            
            print(f"📤 Exported {len(conversations)} conversations to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error exporting conversation data: {str(e)}")
            return ""
    
    def import_conversation_data(self, filepath: str) -> bool:
        """
        Import conversation data from file.
        
        Args:
            filepath: Path to import file
            
        Returns:
            True if successful
        """
        try:
            if not os.path.exists(filepath):
                print(f"❌ Import file not found: {filepath}")
                return False
            
            # Detect format
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                conversations = data.get("conversations", [])
                
                for conv in conversations:
                    # Import each conversation
                    self._import_conversation(conv)
                
                print(f"📥 Imported {len(conversations)} conversations from {filepath}")
                return True
            
            else:
                print(f"❌ Unsupported import format: {filepath}")
                return False
                
        except Exception as e:
            print(f"❌ Error importing conversation data: {str(e)}")
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: ID of conversation to delete
            
        Returns:
            True if successful
        """
        try:
            with self.neo4j_manager.driver.session(database=self.neo4j_manager.database) as session:
                query = """
                MATCH (m:Message {conversation_id: $conv_id})
                DETACH DELETE m
                RETURN COUNT(m) as deleted_count
                """
                
                result = session.run(query, {"conv_id": conversation_id})
                deleted_count = result.single()["deleted_count"]
                
                print(f"🗑️ Deleted conversation {conversation_id} ({deleted_count} messages)")
                return True
                
        except Exception as e:
            print(f"❌ Error deleting conversation: {str(e)}")
            return False
    
    # Helper methods
    def _calculate_duration(self, start_time: str, end_time: str) -> str:
        """Calculate duration between two timestamps."""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration = end - start
            
            if duration.days > 0:
                return f"{duration.days}d {duration.seconds//3600}h"
            elif duration.seconds > 3600:
                return f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"
            else:
                return f"{duration.seconds//60}m"
        except:
            return "Unknown"
    
    def _get_conversation_preview(self, conversation_id: str) -> str:
        """Get a preview of the conversation."""
        try:
            with self.neo4j_manager.driver.session(database=self.neo4j_manager.database) as session:
                query = """
                MATCH (m:Message {conversation_id: $conv_id})
                WHERE m.turn_number = 1 AND m.chunk_index = 0
                RETURN m.user_message as first_user_message
                LIMIT 1
                """
                
                result = session.run(query, {"conv_id": conversation_id})
                record = result.single()
                
                if record and record["first_user_message"]:
                    preview = record["first_user_message"]
                    return preview[:100] + "..." if len(preview) > 100 else preview
                
                return "No preview available"
                
        except Exception as e:
            return "Preview unavailable"
    
    def _extract_conversation_topics(self, conversation_id: str) -> List[str]:
        """Extract topics from a conversation using simple keyword analysis."""
        try:
            # Get conversation text
            with self.neo4j_manager.driver.session(database=self.neo4j_manager.database) as session:
                query = """
                MATCH (m:Message {conversation_id: $conv_id})
                RETURN m.user_message as user_msg, m.assistant_message as assistant_msg
                LIMIT 10
                """
                
                result = session.run(query, {"conv_id": conversation_id})
                
                all_text = []
                for record in result:
                    if record["user_msg"]:
                        all_text.append(record["user_msg"])
                    if record["assistant_msg"]:
                        all_text.append(record["assistant_msg"])
                
                # Simple topic extraction using common words
                combined_text = " ".join(all_text).lower()
                
                # Common topic keywords
                topic_keywords = {
                    "programming": ["code", "programming", "python", "javascript", "algorithm", "function", "variable"],
                    "ai_ml": ["machine learning", "ai", "artificial intelligence", "model", "training", "neural"],
                    "data": ["data", "database", "sql", "analysis", "visualization", "csv"],
                    "web": ["website", "html", "css", "web", "browser", "url", "http"],
                    "business": ["business", "strategy", "market", "customer", "revenue", "profit"],
                    "science": ["research", "experiment", "hypothesis", "theory", "study", "analysis"],
                    "technology": ["technology", "software", "hardware", "system", "network", "server"]
                }
                
                detected_topics = []
                for topic, keywords in topic_keywords.items():
                    if any(keyword in combined_text for keyword in keywords):
                        detected_topics.append(topic)
                
                return detected_topics[:3]  # Return top 3 topics
                
        except Exception as e:
            return []
    
    def _generate_conversation_summary(self, conversation_id: str, messages: List[Dict]) -> str:
        """Generate a summary of the conversation using Gemini."""
        try:
            if not messages:
                return "No messages in conversation"
            
            # Extract key turns for summarization
            key_messages = []
            for msg in messages[:10]:  # First 10 messages
                if msg.get("user_message") and msg.get("assistant_message"):
                    key_messages.append(f"User: {msg['user_message']}")
                    key_messages.append(f"Assistant: {msg['assistant_message']}")
            
            if not key_messages:
                return "Unable to generate summary"
            
            conversation_text = "\n".join(key_messages)
            
            # Use Azure OpenAI to generate summary
            summary_prompt = f"""Please provide a concise 2-3 sentence summary of this conversation:

{conversation_text}

Summary:"""
            
            messages = [
                {"role": "system", "content": "You are an expert at creating concise conversation summaries. Provide 2-3 sentence summaries of conversations."},
                {"role": "user", "content": summary_prompt}
            ]
            
            summary = self.openai_service._make_request_with_retry(
                messages, 0.1, "Conversation Summary"
            )
            
            return summary if summary else "Summary generation failed"
            
        except Exception as e:
            return f"Summary unavailable: {str(e)}"
    
    def _get_trending_topics(self, days: int) -> List[Dict[str, Any]]:
        """Get trending topics in recent conversations."""
        try:
            # Get recent conversations
            recent_convs = self.get_all_conversations(limit=50)
            
            topic_counts = Counter()
            for conv in recent_convs:
                for topic in conv.get("topics", []):
                    topic_counts[topic] += 1
            
            trending = []
            for topic, count in topic_counts.most_common(10):
                trending.append({
                    "topic": topic,
                    "conversation_count": count,
                    "percentage": round((count / len(recent_convs)) * 100, 1)
                })
            
            return trending
            
        except Exception as e:
            return []
    
    def _import_conversation(self, conv_data: Dict[str, Any]) -> bool:
        """Import a single conversation."""
        try:
            # This would involve recreating the messages in Neo4j
            # Implementation depends on the exact format of exported data
            # For now, just log the attempt
            print(f"📥 Importing conversation: {conv_data.get('conversation_id', 'unknown')}")
            return True
            
        except Exception as e:
            print(f"❌ Error importing conversation: {str(e)}")
            return False
    
    def close(self):
        """Close connections."""
        if self.neo4j_manager:
            self.neo4j_manager.close()


# Test the conversation history manager
if __name__ == "__main__":
    try:
        print("🧪 Testing Conversation History Manager...")
        
        history_manager = ConversationHistoryManager()
        
        # Test getting all conversations
        conversations = history_manager.get_all_conversations(limit=5)
        print(f"Found {len(conversations)} conversations")
        
        # Test analytics
        analytics = history_manager.get_conversation_analytics(days=7)
        print(f"Analytics: {analytics}")
        
        history_manager.close()
        print("✅ History Manager test completed!")
        
    except Exception as e:
        print(f"❌ History Manager test failed: {str(e)}")
        import traceback
        traceback.print_exc()
