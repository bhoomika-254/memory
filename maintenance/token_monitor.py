"""
Token Usage Monitor for Gemini API.

This script helps monitor and analyze token usage patterns
to further optimize API consumption.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.components.neo4j_manager import Neo4jManager
from datetime import datetime, timedelta
import json

class TokenUsageMonitor:
    """Monitor and analyze token usage patterns."""
    
    def __init__(self):
        self.neo4j_manager = Neo4jManager()
        self.driver = self.neo4j_manager.driver
    
    def log_api_call(self, operation: str, input_tokens: int, output_tokens: int, success: bool):
        """Log an API call for monitoring."""
        try:
            query = """
            CREATE (l:APILog {
                operation: $operation,
                input_tokens: $input_tokens,
                output_tokens: $output_tokens,
                total_tokens: $total_tokens,
                success: $success,
                timestamp: $timestamp
            })
            """
            
            self.driver.execute_query(
                query,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                success=success,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"⚠️ Failed to log API call: {e}")
    
    def get_usage_stats(self, days: int = 7) -> dict:
        """Get token usage statistics for the last N days."""
        try:
            since_date = datetime.now() - timedelta(days=days)
            
            query = """
            MATCH (l:APILog)
            WHERE l.timestamp > $since_date
            RETURN 
                l.operation as operation,
                count(l) as call_count,
                sum(l.input_tokens) as total_input_tokens,
                sum(l.output_tokens) as total_output_tokens,
                sum(l.total_tokens) as total_tokens,
                avg(l.total_tokens) as avg_tokens_per_call,
                sum(CASE WHEN l.success THEN 1 ELSE 0 END) as successful_calls
            ORDER BY total_tokens DESC
            """
            
            result = self.driver.execute_query(query, since_date=since_date.isoformat())
            
            stats = {}
            total_tokens = 0
            total_calls = 0
            
            for record in result.records:
                operation = record["operation"]
                stats[operation] = {
                    "call_count": record["call_count"],
                    "total_input_tokens": record["total_input_tokens"],
                    "total_output_tokens": record["total_output_tokens"],
                    "total_tokens": record["total_tokens"],
                    "avg_tokens_per_call": round(record["avg_tokens_per_call"], 2),
                    "successful_calls": record["successful_calls"],
                    "success_rate": round(record["successful_calls"] / record["call_count"] * 100, 2)
                }
                total_tokens += record["total_tokens"]
                total_calls += record["call_count"]
            
            stats["summary"] = {
                "total_tokens": total_tokens,
                "total_calls": total_calls,
                "avg_tokens_per_call": round(total_tokens / total_calls, 2) if total_calls > 0 else 0,
                "days_analyzed": days
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get usage stats: {e}")
            return {}
    
    def print_usage_report(self, days: int = 7):
        """Print a detailed usage report."""
        stats = self.get_usage_stats(days)
        
        if not stats:
            print("❌ No usage data available")
            return
        
        print(f"\n📊 TOKEN USAGE REPORT (Last {days} days)")
        print("=" * 50)
        
        summary = stats.pop("summary")
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Tokens Used: {summary['total_tokens']:,}")
        print(f"Average Tokens per Call: {summary['avg_tokens_per_call']}")
        print(f"Estimated Cost (USD): ${summary['total_tokens'] * 0.000002:.4f}")  # Rough Gemini Flash pricing
        
        print("\nPER OPERATION BREAKDOWN:")
        print("-" * 30)
        
        for operation, data in stats.items():
            print(f"\n{operation.upper()}:")
            print(f"  Calls: {data['call_count']}")
            print(f"  Total Tokens: {data['total_tokens']:,}")
            print(f"  Avg Tokens/Call: {data['avg_tokens_per_call']}")
            print(f"  Success Rate: {data['success_rate']}%")
            print(f"  Input Tokens: {data['total_input_tokens']:,}")
            print(f"  Output Tokens: {data['total_output_tokens']:,}")
    
    def cleanup_old_logs(self, days: int = 30):
        """Clean up old API logs."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
            MATCH (l:APILog)
            WHERE l.timestamp < $cutoff_date
            DETACH DELETE l
            """
            
            result = self.driver.execute_query(query, cutoff_date=cutoff_date.isoformat())
            print(f"🧹 Cleaned up API logs older than {days} days")
            
        except Exception as e:
            print(f"⚠️ Failed to cleanup logs: {e}")

def main():
    """Run token usage monitoring."""
    monitor = TokenUsageMonitor()
    
    # Print usage report
    monitor.print_usage_report(7)
    
    # Cleanup old logs
    monitor.cleanup_old_logs(30)

if __name__ == "__main__":
    main()
