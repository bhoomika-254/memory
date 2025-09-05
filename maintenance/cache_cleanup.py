"""
Cache Maintenance Script for Token Optimization.

Run this script periodically to clean up expired cache entries
and optimize the database for better performance.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.neo4j_manager import Neo4jManager
from src.components.token_optimizer import get_token_optimizer
from datetime import datetime

def main():
    """Run cache cleanup and maintenance tasks."""
    print("🧹 Starting cache maintenance...")
    
    try:
        # Initialize Neo4j connection
        neo4j_manager = Neo4jManager()
        optimizer = get_token_optimizer(neo4j_manager.driver)
        
        # Clean up old cache entries
        optimizer.cleanup_old_cache()
        
        # Add database indexes for better performance
        create_cache_indexes(neo4j_manager.driver)
        
        print("✅ Cache maintenance completed successfully")
        
    except Exception as e:
        print(f"❌ Cache maintenance failed: {e}")

def create_cache_indexes(driver):
    """Create indexes for better cache performance."""
    try:
        indexes = [
            "CREATE INDEX cache_key_index IF NOT EXISTS FOR (c:Cache) ON (c.cache_key)",
            "CREATE INDEX cache_created_index IF NOT EXISTS FOR (c:Cache) ON (c.created_at)",
            "CREATE INDEX conversation_summary_index IF NOT EXISTS FOR (c:Conversation) ON (c.summary)"
        ]
        
        for index_query in indexes:
            driver.execute_query(index_query)
            
        print("📊 Database indexes created/verified")
        
    except Exception as e:
        print(f"⚠️ Index creation failed: {e}")

if __name__ == "__main__":
    main()
