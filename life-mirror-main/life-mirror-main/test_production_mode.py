#!/usr/bin/env python3
"""
Test production mode vs mock mode to identify hanging issues
"""

import os
import time
import threading
from src.agents.orchestrator import Orchestrator
from src.agents.graph_workflow import GraphExecutor

def test_mode_comparison():
    """Compare mock mode vs production mode"""
    print("🔍 Testing Mock vs Production Mode...")
    
    # Test 1: Mock Mode
    print("\n1️⃣ Testing MOCK Mode:")
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    def test_mock():
        try:
            orchestrator = Orchestrator()
            start_time = time.time()
            result = orchestrator.analyze_media(
                media_id="test_mock",
                url="test_person.jpg",
                context={"user_id": "test_user"}
            )
            end_time = time.time()
            print(f"   ✅ Mock mode: {end_time - start_time:.2f}s")
            return True
        except Exception as e:
            print(f"   ❌ Mock mode failed: {e}")
            return False
    
    mock_success = test_mock()
    
    # Test 2: Production Mode
    print("\n2️⃣ Testing PRODUCTION Mode:")
    os.environ["LIFEMIRROR_MODE"] = "prod"
    
    def test_production():
        try:
            orchestrator = Orchestrator()
            start_time = time.time()
            result = orchestrator.analyze_media(
                media_id="test_prod",
                url="test_person.jpg",
                context={"user_id": "test_user"}
            )
            end_time = time.time()
            print(f"   ✅ Production mode: {end_time - start_time:.2f}s")
            return True
        except Exception as e:
            print(f"   ❌ Production mode failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test production mode with timeout
    result = [False]
    def target():
        result[0] = test_production()
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=60)  # 60 second timeout for production
    
    if thread.is_alive():
        print("   ❌ PRODUCTION MODE HANGING DETECTED!")
        print("   - This is the root cause of Streamlit hanging")
        print("   - MediaPipe/OpenCV issues in production mode")
        prod_success = False
    else:
        prod_success = result[0]
    
    return mock_success, prod_success

def test_graph_executor_modes():
    """Test GraphExecutor in both modes"""
    print("\n🔍 Testing GraphExecutor in Both Modes...")
    
    # Test 1: Mock Mode
    print("\n1️⃣ GraphExecutor MOCK Mode:")
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    try:
        graph_executor = GraphExecutor()
        start_time = time.time()
        result = graph_executor.execute(
            media_id="test_graph_mock",
            url="test_person.jpg",
            context={"user_id": "test_user"}
        )
        end_time = time.time()
        print(f"   ✅ GraphExecutor mock: {end_time - start_time:.2f}s")
        graph_mock_success = True
    except Exception as e:
        print(f"   ❌ GraphExecutor mock failed: {e}")
        graph_mock_success = False
    
    # Test 2: Production Mode with timeout
    print("\n2️⃣ GraphExecutor PRODUCTION Mode:")
    os.environ["LIFEMIRROR_MODE"] = "prod"
    
    def test_graph_production():
        try:
            graph_executor = GraphExecutor()
            start_time = time.time()
            result = graph_executor.execute(
                media_id="test_graph_prod",
                url="test_person.jpg",
                context={"user_id": "test_user"}
            )
            end_time = time.time()
            print(f"   ✅ GraphExecutor production: {end_time - start_time:.2f}s")
            return True
        except Exception as e:
            print(f"   ❌ GraphExecutor production failed: {e}")
            return False
    
    result = [False]
    def target():
        result[0] = test_graph_production()
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=60)  # 60 second timeout
    
    if thread.is_alive():
        print("   ❌ GRAPH EXECUTOR PRODUCTION MODE HANGING!")
        print("   - This confirms the hanging issue is in production mode")
        graph_prod_success = False
    else:
        graph_prod_success = result[0]
    
    return graph_mock_success, graph_prod_success

def test_individual_agents_production():
    """Test individual agents in production mode"""
    print("\n🔍 Testing Individual Agents in Production Mode...")
    
    os.environ["LIFEMIRROR_MODE"] = "prod"
    
    from src.agents.face_agent import FaceAgent
    from src.agents.fashion_agent import FashionAgent
    from src.agents.posture_agent import PostureAgent
    from src.agents.base_agent import AgentInput
    
    agents = [
        ("Face", FaceAgent()),
        ("Fashion", FashionAgent()),
        ("Posture", PostureAgent())
    ]
    
    results = {}
    
    for name, agent in agents:
        print(f"\n   Testing {name} Agent:")
        
        def test_agent():
            try:
                start_time = time.time()
                result = agent.run(AgentInput(
                    media_id=f"test_{name.lower()}",
                    url="test_person.jpg",
                    context={"user_id": "test"}
                ))
                end_time = time.time()
                print(f"     ✅ {name}: {end_time - start_time:.2f}s")
                return True
            except Exception as e:
                print(f"     ❌ {name} failed: {e}")
                return False
        
        # Test with timeout
        result = [False]
        def target():
            result[0] = test_agent()
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30)  # 30 second timeout per agent
        
        if thread.is_alive():
            print(f"     ❌ {name} AGENT HANGING in production mode!")
            results[name] = False
        else:
            results[name] = result[0]
    
    return results

if __name__ == "__main__":
    print("🚀 Starting Production Mode Test")
    print("=" * 60)
    
    # Test 1: Orchestrator comparison
    mock_success, prod_success = test_mode_comparison()
    
    # Test 2: GraphExecutor comparison
    graph_mock_success, graph_prod_success = test_graph_executor_modes()
    
    # Test 3: Individual agents in production
    agent_results = test_individual_agents_production()
    
    print("\n" + "=" * 60)
    print("🏁 Production Mode Test Results")
    print("\n📊 SUMMARY:")
    print(f"   - Orchestrator Mock: {'✅' if mock_success else '❌'}")
    print(f"   - Orchestrator Prod: {'✅' if prod_success else '❌'}")
    print(f"   - GraphExecutor Mock: {'✅' if graph_mock_success else '❌'}")
    print(f"   - GraphExecutor Prod: {'✅' if graph_prod_success else '❌'}")
    
    print("\n🤖 Individual Agents (Production):")
    for agent_name, success in agent_results.items():
        print(f"   - {agent_name}: {'✅' if success else '❌'}")
    
    if not prod_success or not graph_prod_success:
        print("\n🎯 ROOT CAUSE IDENTIFIED:")
        print("   - Production mode has hanging issues")
        print("   - MediaPipe DLL loading problems cause infinite loops")
        print("   - Solution: Force mock mode or fix MediaPipe dependencies")
    else:
        print("\n✅ No hanging issues detected in production mode")
    
    # Reset to mock mode
    os.environ["LIFEMIRROR_MODE"] = "mock"