#!/usr/bin/env python3
"""
Test individual graph nodes to identify where the execution stops
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph_workflow import GraphExecutor
import uuid

# Set mock mode
os.environ["LIFEMIRROR_MODE"] = "mock"

def test_individual_nodes():
    print("🧪 Testing individual graph nodes...")
    
    executor = GraphExecutor()
    
    # Create test state
    test_state = {
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
    
    print("\n1️⃣ Testing embedder node...")
    try:
        result1 = executor.run_embedder(test_state)
        print(f"   ✅ Embedder: {bool(result1.get('embedding'))}")
        test_state.update(result1)
    except Exception as e:
        print(f"   ❌ Embedder failed: {e}")
        return False
    
    print("\n2️⃣ Testing face node...")
    try:
        result2 = executor.run_face(test_state)
        print(f"   ✅ Face: {bool(result2.get('face'))}")
        test_state.update(result2)
    except Exception as e:
        print(f"   ❌ Face failed: {e}")
        return False
    
    print("\n3️⃣ Testing aggregator node...")
    try:
        result3 = executor.run_aggregator(test_state)
        print(f"   ✅ Aggregator: {bool(result3.get('aggregated'))}")
        test_state.update(result3)
    except Exception as e:
        print(f"   ❌ Aggregator failed: {e}")
        return False
    
    print("\n4️⃣ Testing formatter node...")
    try:
        result4 = executor.run_formatter(test_state)
        print(f"   ✅ Formatter: {bool(result4.get('final_result'))}")
        test_state.update(result4)
        
        if result4.get('final_result'):
            final_result = result4['final_result']
            print(f"   📊 Final result type: {type(final_result)}")
            if isinstance(final_result, dict):
                print(f"   📊 Final result keys: {list(final_result.keys())}")
                if 'data' in final_result:
                    print(f"   📊 Data keys: {list(final_result['data'].keys())}")
        
    except Exception as e:
        print(f"   ❌ Formatter failed: {e}")
        return False
    
    print("\n✅ All individual nodes work correctly!")
    
    # Now test the compiled graph
    print("\n🕸️ Testing compiled graph...")
    try:
        compiled_graph = executor.graph.compile()
        
        # Reset state for graph test
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
        
        final_state = compiled_graph.invoke(initial_state)
        print(f"   📊 Graph execution completed")
        print(f"   📊 Final result: {bool(final_state.get('final_result'))}")
        
        if not final_state.get('final_result'):
            print("   ❌ Graph execution failed to produce final_result")
            print(f"   📊 Available keys: {list(final_state.keys())}")
            for key in ['embedding', 'face', 'aggregated', 'final_result']:
                value = final_state.get(key)
                print(f"   📊 {key}: {type(value)} = {bool(value)}")
            return False
        else:
            print("   ✅ Graph execution successful!")
            return True
            
    except Exception as e:
        print(f"   ❌ Graph compilation/execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_individual_nodes()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Individual node testing completed")