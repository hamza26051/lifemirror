#!/usr/bin/env python3
"""
Debug graph execution to see which nodes are actually called
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph_workflow import GraphExecutor
import uuid

# Set mock mode
os.environ["LIFEMIRROR_MODE"] = "mock"

def test_graph_execution_with_debug():
    print("🔍 Testing graph execution with debug output...")
    
    executor = GraphExecutor()
    
    # Create test state
    initial_state = {
        "media_id": "test-123",
        "url": "http://localhost:8000/test.jpg",
        "context": {"user_consent": {"face_analysis": True}},
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
    
    print("\n🕸️ Compiling and executing graph...")
    try:
        compiled_graph = executor.graph.compile()
        print("   ✅ Graph compiled successfully")
        
        # Execute with debug output
        print("\n🚀 Starting graph execution...")
        final_state = compiled_graph.invoke(initial_state)
        print("   ✅ Graph execution completed")
        
        # Analyze final state
        print("\n📊 Final state analysis:")
        for key in ['embedding', 'face', 'fashion', 'posture', 'bio', 'aggregated', 'final_result']:
            value = final_state.get(key)
            if value is None:
                print(f"   ❌ {key}: None")
            elif isinstance(value, dict):
                success = value.get('success', False)
                print(f"   {'✅' if success else '❌'} {key}: {type(value)} (success: {success})")
            else:
                print(f"   ✅ {key}: {type(value)}")
        
        # Check if formatter was called
        if final_state.get('final_result') is None:
            print("\n❌ ISSUE: final_result is None")
            print("   This means the formatter node was not called or failed")
            
            # Check if aggregated data exists
            aggregated = final_state.get('aggregated')
            if aggregated:
                print(f"   📊 Aggregated data exists: {aggregated.get('success', False)}")
                print("   🔧 Trying to call formatter manually...")
                
                try:
                    manual_result = executor.run_formatter(final_state)
                    print(f"   ✅ Manual formatter call successful: {bool(manual_result.get('final_result'))}")
                    if manual_result.get('final_result'):
                        print("   🎯 SOLUTION: Formatter works manually but not in graph")
                        print("   🔧 Issue is with LangGraph execution flow")
                except Exception as e:
                    print(f"   ❌ Manual formatter call failed: {e}")
            else:
                print("   ❌ No aggregated data - aggregator may have failed")
        else:
            print("\n✅ SUCCESS: final_result exists!")
            final_result = final_state['final_result']
            print(f"   📊 Type: {type(final_result)}")
            if isinstance(final_result, dict):
                print(f"   📊 Keys: {list(final_result.keys())}")
        
        return final_state.get('final_result') is not None
        
    except Exception as e:
        print(f"   ❌ Graph execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_execution_with_debug()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Graph execution debug completed")