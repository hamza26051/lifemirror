#!/usr/bin/env python3
"""
Debug script to trace the exact state flow in GraphExecutor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph_workflow import GraphExecutor
import uuid

# Set mock mode
os.environ["LIFEMIRROR_MODE"] = "mock"

def debug_graph_execution():
    print("🔍 Debugging GraphExecutor state flow...")
    
    # Create GraphExecutor
    executor = GraphExecutor()
    
    # Test data
    media_id = "test-media-123"
    url = "http://localhost:8000/test.jpg"
    context = {
        "user_consent": {
            "face_analysis": True,
            "fashion_analysis": True,
            "posture_analysis": True
        }
    }
    
    # Create initial state
    initial_state = {
        "media_id": media_id,
        "url": url,
        "context": context,
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
    
    print(f"📝 Initial state keys: {list(initial_state.keys())}")
    print(f"📝 Initial final_result: {initial_state['final_result']}")
    
    try:
        # Compile and execute graph
        compiled_graph = executor.graph.compile()
        print("✅ Graph compiled successfully")
        
        # Execute graph
        print("🚀 Executing graph...")
        final_state = compiled_graph.invoke(initial_state)
        
        print(f"📋 Final state keys: {list(final_state.keys())}")
        print(f"📋 Final state final_result: {final_state.get('final_result')}")
        
        # Check each step
        print("\n🔍 State analysis:")
        print(f"  - embedding: {type(final_state.get('embedding'))} - {bool(final_state.get('embedding'))}")
        print(f"  - face: {type(final_state.get('face'))} - {bool(final_state.get('face'))}")
        print(f"  - aggregated: {type(final_state.get('aggregated'))} - {bool(final_state.get('aggregated'))}")
        print(f"  - final_result: {type(final_state.get('final_result'))} - {bool(final_state.get('final_result'))}")
        
        if final_state.get('final_result'):
            print(f"\n✅ Final result structure: {final_state['final_result'].keys() if isinstance(final_state['final_result'], dict) else 'Not a dict'}")
        else:
            print("\n❌ final_result is None or empty!")
            
        # Test the execute method
        print("\n🧪 Testing execute method...")
        result = executor.execute(media_id, url, context)
        print(f"Execute result: {result}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_graph_execution()