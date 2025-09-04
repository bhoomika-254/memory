"""
Retrieval Fusion Component for Memory System.

This module implements Reciprocal Rank Fusion (RRF) to combine results
from semantic search (Qdrant) and lexical search (Whoosh), creating a
unified ranking that leverages the strengths of both approaches.

Key Features:
- Reciprocal Rank Fusion algorithm
- Score normalization and ranking
- Duplicate removal and result merging
- Configurable fusion parameters
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
import numpy as np

from src.components.qdrant_manager import qdrant_manager
from src.components.whoosh_manager import whoosh_manager

class RetrievalFusion:
    """
    Combines semantic and lexical search results using Reciprocal Rank Fusion.
    
    RRF is a method that combines multiple ranked lists by using the reciprocal
    of the rank positions, giving higher weight to items that appear highly
    ranked in multiple lists.
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize the retrieval fusion component.
        
        Args:
            k: RRF parameter that controls the fusion weighting.
               Lower values give more importance to top-ranked items.
               Typical values are between 10-100, with 60 being a good default.
        """
        self.k = k
        self.semantic_top_k = 20  # How many results to get from semantic search
        self.lexical_top_k = 20   # How many results to get from lexical search  
        self.final_top_k = 5      # How many final results to return after fusion
        
        print(f"🔧 Retrieval fusion configured: semantic({self.semantic_top_k}) + lexical({self.lexical_top_k}) → final({self.final_top_k})")
    
    def hybrid_search(self, query: str, conversation_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and lexical retrieval.
        
        This is the main method that orchestrates the entire hybrid retrieval
        process: semantic search, lexical search, and fusion.
        
        Args:
            query: Search query text
            conversation_id: Optional conversation filter
            
        Returns:
            List of fused and ranked search results
        """
        print(f"🔍 Starting hybrid search for: '{query[:50]}...'")
        
        # Perform parallel retrieval
        semantic_results = self._semantic_search(query, conversation_id)
        lexical_results = self._lexical_search(query, conversation_id)
        
        # Combine results using RRF
        fused_results = self._reciprocal_rank_fusion(semantic_results, lexical_results)
        
        # Limit to final top_k
        final_results = fused_results[:self.final_top_k]
        
        print(f"✅ Hybrid search complete: {len(semantic_results)} semantic + {len(lexical_results)} lexical → {len(final_results)} final")
        
        return final_results
    
    def _semantic_search(self, query: str, conversation_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using Qdrant.
        
        Args:
            query: Search query
            conversation_id: Optional conversation filter
            
        Returns:
            List of semantic search results
        """
        try:
            results = qdrant_manager.semantic_search(
                query=query,
                top_k=self.semantic_top_k,
                conversation_id=conversation_id
            )
            
            # Add search type marker
            for result in results:
                result["search_type"] = "semantic"
                result["original_rank"] = results.index(result) + 1
            
            return results
            
        except Exception as e:
            print(f"❌ Semantic search failed: {str(e)}")
            return []
    
    def _lexical_search(self, query: str, conversation_id: str = None) -> List[Dict[str, Any]]:
        """
        Perform lexical search using Whoosh.
        
        Args:
            query: Search query
            conversation_id: Optional conversation filter
            
        Returns:
            List of lexical search results
        """
        try:
            results = whoosh_manager.lexical_search(
                query=query,
                top_k=self.lexical_top_k,
                conversation_id=conversation_id
            )
            
            # Add search type marker
            for result in results:
                result["search_type"] = "lexical"
                result["original_rank"] = results.index(result) + 1
            
            return results
            
        except Exception as e:
            print(f"❌ Lexical search failed: {str(e)}")
            return []
    
    def _reciprocal_rank_fusion(self, semantic_results: List[Dict[str, Any]], 
                               lexical_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion to combine search results.
        
        RRF Formula: score(d) = Σ(1 / (k + rank(d)))
        Where k is the RRF parameter and rank(d) is the rank of document d in each list.
        
        Args:
            semantic_results: Results from semantic search
            lexical_results: Results from lexical search
            
        Returns:
            List of fused results sorted by RRF score
        """
        print(f"🔀 Applying RRF fusion with k={self.k}")
        
        # Dictionary to accumulate RRF scores for each unique document
        # Key: unique identifier, Value: document info with accumulated score
        fusion_scores = defaultdict(lambda: {
            "rrf_score": 0.0,
            "semantic_rank": None,
            "lexical_rank": None,
            "semantic_score": None,
            "lexical_score": None,
            "appears_in": []
        })
        
        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            doc_id = self._get_document_id(result)
            
            # Calculate RRF contribution: 1 / (k + rank)
            rrf_contribution = 1.0 / (self.k + rank)
            
            fusion_scores[doc_id]["rrf_score"] += rrf_contribution
            fusion_scores[doc_id]["semantic_rank"] = rank
            fusion_scores[doc_id]["semantic_score"] = result.get("score", 0.0)
            fusion_scores[doc_id]["appears_in"].append("semantic")
            fusion_scores[doc_id]["document"] = result
        
        # Process lexical results
        for rank, result in enumerate(lexical_results, 1):
            doc_id = self._get_document_id(result)
            
            # Calculate RRF contribution
            rrf_contribution = 1.0 / (self.k + rank)
            
            fusion_scores[doc_id]["rrf_score"] += rrf_contribution
            fusion_scores[doc_id]["lexical_rank"] = rank
            fusion_scores[doc_id]["lexical_score"] = result.get("score", 0.0)
            fusion_scores[doc_id]["appears_in"].append("lexical")
            
            # If document wasn't in semantic results, store it
            if "document" not in fusion_scores[doc_id]:
                fusion_scores[doc_id]["document"] = result
        
        # Create final ranked list
        fused_results = []
        
        for doc_id, score_info in fusion_scores.items():
            document = score_info["document"].copy()
            
            # Add fusion metadata
            document.update({
                "rrf_score": score_info["rrf_score"],
                "semantic_rank": score_info["semantic_rank"],
                "lexical_rank": score_info["lexical_rank"],
                "semantic_score": score_info["semantic_score"],
                "lexical_score": score_info["lexical_score"],
                "appears_in": score_info["appears_in"],
                "fusion_method": "rrf"
            })
            
            fused_results.append(document)
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # Add final rank information
        for rank, result in enumerate(fused_results, 1):
            result["final_rank"] = rank
        
        print(f"📊 RRF fusion stats:")
        print(f"   - Unique documents after fusion: {len(fused_results)}")
        
        # Count documents by source
        semantic_only = sum(1 for r in fused_results if r["appears_in"] == ["semantic"])
        lexical_only = sum(1 for r in fused_results if r["appears_in"] == ["lexical"])
        both = sum(1 for r in fused_results if len(r["appears_in"]) == 2)
        
        print(f"   - Semantic only: {semantic_only}, Lexical only: {lexical_only}, Both: {both}")
        
        return fused_results
    
    def _get_document_id(self, result: Dict[str, Any]) -> str:
        """
        Generate a unique identifier for a document to handle duplicates.
        
        We use the text content as the primary identifier since the same
        conversation chunk might be indexed with different IDs in Qdrant vs Whoosh.
        
        Args:
            result: Search result document
            
        Returns:
            Unique identifier string
        """
        # Use a combination of conversation_id, turn_number, and chunk_index
        # This should uniquely identify each chunk
        conv_id = result.get("conversation_id", "unknown")
        turn_num = result.get("turn_number", 0)
        chunk_idx = result.get("chunk_index", 0)
        
        return f"{conv_id}_{turn_num}_{chunk_idx}"
    
    def explain_fusion_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Provide detailed explanation of fusion results for debugging.
        
        Args:
            results: Fused search results
            
        Returns:
            Dictionary with fusion analysis
        """
        if not results:
            return {"message": "No results to analyze"}
        
        analysis = {
            "total_results": len(results),
            "score_distribution": {
                "min_rrf_score": min(r["rrf_score"] for r in results),
                "max_rrf_score": max(r["rrf_score"] for r in results),
                "avg_rrf_score": np.mean([r["rrf_score"] for r in results])
            },
            "source_distribution": {
                "semantic_only": len([r for r in results if r["appears_in"] == ["semantic"]]),
                "lexical_only": len([r for r in results if r["appears_in"] == ["lexical"]]),
                "both_sources": len([r for r in results if len(r["appears_in"]) == 2])
            },
            "top_results_detail": []
        }
        
        # Add details for top 3 results
        for i, result in enumerate(results[:3]):
            detail = {
                "rank": i + 1,
                "rrf_score": result["rrf_score"],
                "text_preview": result["text"][:100] + "...",
                "semantic_rank": result.get("semantic_rank"),
                "lexical_rank": result.get("lexical_rank"),
                "semantic_score": result.get("semantic_score"),
                "lexical_score": result.get("lexical_score"),
                "appears_in": result["appears_in"]
            }
            analysis["top_results_detail"].append(detail)
        
        return analysis

# Create global retrieval fusion instance
retrieval_fusion = RetrievalFusion()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Retrieval Fusion...")
    
    # Mock results for testing
    mock_semantic = [
        {"id": "1", "text": "Machine learning algorithms", "score": 0.95, "conversation_id": "conv1", "turn_number": 1, "chunk_index": 0},
        {"id": "2", "text": "Deep learning networks", "score": 0.87, "conversation_id": "conv1", "turn_number": 2, "chunk_index": 0},
        {"id": "3", "text": "Neural network optimization", "score": 0.82, "conversation_id": "conv1", "turn_number": 3, "chunk_index": 0}
    ]
    
    mock_lexical = [
        {"id": "1", "text": "Machine learning algorithms", "score": 2.5, "conversation_id": "conv1", "turn_number": 1, "chunk_index": 0},
        {"id": "4", "text": "Optimization techniques", "score": 2.1, "conversation_id": "conv1", "turn_number": 4, "chunk_index": 0},
        {"id": "5", "text": "Algorithm performance", "score": 1.8, "conversation_id": "conv1", "turn_number": 5, "chunk_index": 0}
    ]
    
    # Test RRF fusion
    fusion = RetrievalFusion(k=60)
    fused_results = fusion._reciprocal_rank_fusion(mock_semantic, mock_lexical)
    
    print(f"Fused {len(fused_results)} results:")
    for result in fused_results:
        print(f"  - RRF: {result['rrf_score']:.3f}, Text: {result['text']}, Sources: {result['appears_in']}")
    
    # Test explanation
    explanation = fusion.explain_fusion_results(fused_results)
    print(f"Fusion explanation: {explanation}")
