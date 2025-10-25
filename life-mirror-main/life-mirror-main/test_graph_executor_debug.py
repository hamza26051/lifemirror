#!/usr/bin/env python3
"""
Debug test for GraphExecutor to identify hanging issues
"""

import os
import time
import threading
from src.agents.graph_workflow import GraphExecutor

def test_graph_executor_with_timeout():
    """Test GraphExecutor with timeout to detect hanging"""
    print("🔍 Testing GraphExecutor with timeout detection...")
    
    # Set mock mode
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    def run_graph_executor():
        try:
            print("   - Creating GraphExecutor...")
            graph_executor = GraphExecutor()
            
            print("   - Starting execution...")
            result = graph_executor.execute(
                media_id="test_debug",
                url="test_person.jpg",
                context={"user_id": "test_user"}
            )
            
            print(f"✅ GraphExecutor completed successfully!")
            print(f"   - Type: {type(result)}")
            print(f"   - Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            if isinstance(result, dict):
                print(f"   - Success: {result.get('success', 'Missing')}")
                print(f"   - Error: {result.get('error', 'None')}")
                if 'data' in result:
                    data = result['data']
                    print(f"   - Data type: {type(data)}")
                    if isinstance(data, dict):
                        print(f"   - Data keys: {list(data.keys())}")
            
            return result
        except Exception as e:
            print(f"❌ GraphExecutor failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Run with timeout
    result = [None]
    
    def target():
        result[0] = run_graph_executor()
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    # Wait for completion with timeout
    print("   - Waiting for completion (30s timeout)...")
    thread.join(timeout=30)  # 30 second timeout
    
    if thread.is_alive():
        print("⚠️  GRAPH EXECUTOR HANGING DETECTED!")
        print("   - Process did not complete within 30 seconds")
        print("   - This is likely causing Streamlit analysis to hang")
        print("   - Issue is in LangGraph compilation or execution")
        return False
    else:
        print("✅ GraphExecutor completed within timeout")
        return result[0] is not None

def test_direct_orchestrator_vs_graph():
    """Compare direct orchestrator vs graph executor"""
    print("\n🔄 Comparing Direct Orchestrator vs GraphExecutor...")
    
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    # Test direct orchestrator
    print("\n1️⃣ Testing Direct Orchestrator:")
    try:
        from src.agents.orchestrator import Orchestrator
        start_time = time.time()
        orchestrator = Orchestrator()
        result1 = orchestrator.analyze_media(
            media_id="test_compare",
            url="test_person.jpg",
            context={"user_id": "test_user"}
        )
        end_time = time.time()
        print(f"   ✅ Direct Orchestrator: {end_time - start_time:.2f}s")
        print(f"   - Result type: {type(result1)}")
        direct_success = True
    except Exception as e:
        print(f"   ❌ Direct Orchestrator failed: {e}")
        direct_success = False
    
    # Test graph executor
    print("\n2️⃣ Testing GraphExecutor:")
    try:
        start_time = time.time()
        graph_executor = GraphExecutor()
        result2 = graph_executor.execute(
            media_id="test_compare",
            url="test_person.jpg",
            context={"user_id": "test_user"}
        )
        end_time = time.time()
        print(f"   ✅ GraphExecutor: {end_time - start_time:.2f}s")
        print(f"   - Result type: {type(result2)}")
        graph_success = True
    except Exception as e:
        print(f"   ❌ GraphExecutor failed: {e}")
        graph_success = False
    
    print("\n📊 COMPARISON RESULTS:")
    print(f"   - Direct Orchestrator: {'✅ WORKING' if direct_success else '❌ FAILED'}")
    print(f"   - GraphExecutor: {'✅ WORKING' if graph_success else '❌ FAILED'}")
    
    if direct_success and not graph_success:
        print("\n🎯 SOLUTION FOUND:")
        print("   - Use Direct Orchestrator instead of GraphExecutor")
        print("   - GraphExecutor has issues with LangGraph")
        print("   - This will fix the Streamlit hanging problem")
    
    return direct_success, graph_success

if __name__ == "__main__":
    print("🚀 Starting GraphExecutor Debug Test")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test 1: GraphExecutor with timeout
    success = test_graph_executor_with_timeout()
    
    # Test 2: Compare both approaches
    direct_success, graph_success = test_direct_orchestrator_vs_graph()
    
    end_time = time.time()
    
    print("\n" + "=" * 60)
    print(f"🏁 All tests completed in {end_time - start_time:.2f} seconds")
    
    if not success or not graph_success:
        print("\n❌ GRAPH EXECUTOR HAS HANGING ISSUES")
        print("   - This explains why Streamlit analysis gets stuck")
        print("   - Recommendation: Switch to Direct Orchestrator")
    else:
        print("\n✅ GRAPH EXECUTOR IS WORKING")
        print("   - Hanging issue may be elsewhere")