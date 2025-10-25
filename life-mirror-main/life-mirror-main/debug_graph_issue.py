#!/usr/bin/env python3
"""
Debug script to test GraphExecutor and identify the issue
"""

import os
import sys
sys.path.append('.')

from src.agents.graph_workflow import GraphExecutor
import uuid

def debug_graph_executor():
    """Debug the GraphExecutor to see what's happening"""
    print("🔍 Debugging GraphExecutor...")
    
    # Set mock mode
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    try:
        print("   - Creating GraphExecutor...")
        graph_executor = GraphExecutor()
        
        # Test the graph directly
        initial_state = {
            "media_id": "debug_test",
            "url": "test_person.jpg",
            "context": {"user_id": "test_user"},
            "embedding": None,
            "face": None,
            "fashion": None,
            "posture": None,
            "bio": None,
            "aggregated": None,
            "final_result": None,
            "social_analysis_result": None,
            "enhancement_results": None,
            "langsmith_run_id": str(uuid.uuid4()),
            "langsmith_run_ids": None
        }
        
        print("   - Compiling and invoking graph...")
        compiled_graph = graph_executor.graph.compile()
        
        # Add debug logging to see which nodes execute
        print("   - Graph nodes:", list(graph_executor.graph.nodes.keys()))
        print("   - Graph edges:", list(graph_executor.graph.edges))
        
        final_state = compiled_graph.invoke(initial_state)
        
        print("   - Graph execution completed")
        print("   - Final state keys:", list(final_state.keys()))
        
        print(f"\n📊 FINAL STATE ANALYSIS:")
        print(f"   - Keys: {list(final_state.keys())}")
        
        # Check each agent result
        for agent in ['embedding', 'face', 'fashion', 'posture', 'bio']:
            result = final_state.get(agent)
            print(f"   - {agent}: {type(result)} = {result}")
        
        print(f"   - aggregated: {final_state.get('aggregated')}")
        print(f"   - final_result: {final_state.get('final_result')}")
        
        return final_state
        
    except Exception as e:
        print(f"❌ GraphExecutor failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Starting GraphExecutor Debug")
    print("=" * 50)
    
    result = debug_graph_executor()
    
    print("\n" + "=" * 50)
    if result and result.get('final_result'):
        print("✅ GraphExecutor is working correctly")
    else:
        print("❌ GraphExecutor has issues")
        print("   - final_result is None or missing")