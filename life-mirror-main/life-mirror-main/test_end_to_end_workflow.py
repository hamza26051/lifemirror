#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Complete Image Analysis Workflow
Tests the entire pipeline from image input to formatted analysis output
"""

import sys
import os
import time
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph_workflow import GraphExecutor
from src.agents.orchestrator import Orchestrator
from src.agents.face_agent import FaceAgent
from src.agents.fashion_agent import FashionAgent
from src.agents.posture_agent import PostureAgent
from src.agents.bio_agent import BioAgent
from src.agents.embedder_agent import EmbedderAgent
from src.schemas.agents import AgentInput

def test_individual_agents():
    """Test each agent individually to ensure they work"""
    print("🔍 Testing Individual Agents...")
    print("-" * 40)
    
    # Set mock mode
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    test_input = AgentInput(
        media_id="e2e_test",
        url="test_person.jpg",
        context={"user_id": "test_user"}
    )
    
    agents = {
        "Embedder": EmbedderAgent(),
        "Face": FaceAgent(),
        "Fashion": FashionAgent(),
        "Posture": PostureAgent(),
        "Bio": BioAgent()
    }
    
    results = {}
    
    for name, agent in agents.items():
        try:
            start_time = time.time()
            result = agent.run(test_input)
            end_time = time.time()
            
            success = result.success if hasattr(result, 'success') else bool(result)
            results[name] = {
                "success": success,
                "time": round(end_time - start_time, 2),
                "has_data": bool(result.data if hasattr(result, 'data') else result)
            }
            
            status = "✅" if success else "❌"
            print(f"   {status} {name} Agent: {results[name]['time']}s")
            
        except Exception as e:
            results[name] = {"success": False, "error": str(e), "time": 0}
            print(f"   ❌ {name} Agent: Failed - {e}")
    
    return results

def test_orchestrator_workflow():
    """Test the direct orchestrator workflow"""
    print("\n🔄 Testing Direct Orchestrator Workflow...")
    print("-" * 40)
    
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    try:
        start_time = time.time()
        orchestrator = Orchestrator()
        result = orchestrator.analyze_media(
            media_id="e2e_orchestrator_test",
            url="test_person.jpg",
            context={"user_id": "test_user"}
        )
        end_time = time.time()
        
        success = result and result.get('success', False)
        
        print(f"   ✅ Orchestrator: {round(end_time - start_time, 2)}s")
        print(f"   📊 Success: {success}")
        
        if result:
            print(f"   📈 Overall Score: {result.get('overall_score', 'N/A')}")
            print(f"   🎯 Confidence: {result.get('confidence', 'N/A')}")
            
            # Check data structure
            data = result.get('data', {})
            if data:
                components = {
                    'face_analysis': '👤',
                    'fashion_analysis': '👗', 
                    'posture_analysis': '🧍',
                    'bio_analysis': '🧬',
                    'embedding_analysis': '🔢'
                }
                
                for component, emoji in components.items():
                    has_component = bool(data.get(component))
                    status = "✅" if has_component else "❌"
                    print(f"   {status} {emoji} {component.replace('_', ' ').title()}")
        
        return {"success": success, "result": result, "time": round(end_time - start_time, 2)}
        
    except Exception as e:
        print(f"   ❌ Orchestrator failed: {e}")
        return {"success": False, "error": str(e), "time": 0}

def test_graph_executor_workflow():
    """Test the LangGraph-based workflow"""
    print("\n🕸️ Testing GraphExecutor Workflow...")
    print("-" * 40)
    
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    try:
        start_time = time.time()
        executor = GraphExecutor()
        result = executor.execute(
            media_id="e2e_graph_test",
            url="test_person.jpg",
            context={"user_id": "test_user"}
        )
        end_time = time.time()
        
        success = result and result.get('success', False)
        
        print(f"   ✅ GraphExecutor: {round(end_time - start_time, 2)}s")
        print(f"   📊 Success: {success}")
        
        if result:
            print(f"   📈 Overall Score: {result.get('overall_score', 'N/A')}")
            print(f"   🎯 Confidence: {result.get('confidence', 'N/A')}")
            
            # Check if we have formatted analysis
            if 'analysis' in result:
                analysis = result['analysis']
                print(f"   📝 Analysis Summary: Present")
                print(f"   💪 Key Strengths: {len(analysis.get('key_strengths', []))} items")
                print(f"   🔧 Improvement Areas: {len(analysis.get('improvement_areas', []))} items")
            
            # Check data structure
            data = result.get('data', {})
            if data:
                components = {
                    'face_analysis': '👤',
                    'fashion_analysis': '👗', 
                    'posture_analysis': '🧍',
                    'bio_analysis': '🧬',
                    'embedding_analysis': '🔢'
                }
                
                for component, emoji in components.items():
                    has_component = bool(data.get(component))
                    status = "✅" if has_component else "❌"
                    print(f"   {status} {emoji} {component.replace('_', ' ').title()}")
        
        return {"success": success, "result": result, "time": round(end_time - start_time, 2)}
        
    except Exception as e:
        print(f"   ❌ GraphExecutor failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "time": 0}

def test_production_mode():
    """Test both workflows in production mode"""
    print("\n🏭 Testing Production Mode...")
    print("-" * 40)
    
    os.environ["LIFEMIRROR_MODE"] = "prod"
    
    results = {}
    
    # Test Orchestrator in production
    print("   Testing Orchestrator (Production):")
    try:
        start_time = time.time()
        orchestrator = Orchestrator()
        result = orchestrator.analyze_media(
            media_id="e2e_prod_orchestrator",
            url="test_person.jpg",
            context={"user_id": "test_user"}
        )
        end_time = time.time()
        
        success = result and result.get('success', False)
        results['orchestrator_prod'] = {"success": success, "time": round(end_time - start_time, 2)}
        print(f"     ✅ Orchestrator Production: {results['orchestrator_prod']['time']}s")
        
    except Exception as e:
        results['orchestrator_prod'] = {"success": False, "error": str(e)}
        print(f"     ❌ Orchestrator Production failed: {e}")
    
    # Test GraphExecutor in production
    print("   Testing GraphExecutor (Production):")
    try:
        start_time = time.time()
        executor = GraphExecutor()
        result = executor.execute(
            media_id="e2e_prod_graph",
            url="test_person.jpg",
            context={"user_id": "test_user"}
        )
        end_time = time.time()
        
        success = result and result.get('success', False)
        results['graph_prod'] = {"success": success, "time": round(end_time - start_time, 2)}
        print(f"     ✅ GraphExecutor Production: {results['graph_prod']['time']}s")
        
    except Exception as e:
        results['graph_prod'] = {"success": False, "error": str(e)}
        print(f"     ❌ GraphExecutor Production failed: {e}")
    
    return results

def run_comprehensive_test():
    """Run all end-to-end tests"""
    print("🚀 Starting Comprehensive End-to-End Workflow Test")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test 1: Individual Agents
    agent_results = test_individual_agents()
    
    # Test 2: Orchestrator Workflow
    orchestrator_results = test_orchestrator_workflow()
    
    # Test 3: GraphExecutor Workflow
    graph_results = test_graph_executor_workflow()
    
    # Test 4: Production Mode
    production_results = test_production_mode()
    
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    
    # Summary Report
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    print("\n🔍 Individual Agent Results:")
    for agent, result in agent_results.items():
        status = "✅" if result.get('success') else "❌"
        print(f"   {status} {agent}: {result.get('time', 0)}s")
    
    print("\n🔄 Workflow Results:")
    orchestrator_status = "✅" if orchestrator_results.get('success') else "❌"
    graph_status = "✅" if graph_results.get('success') else "❌"
    print(f"   {orchestrator_status} Orchestrator: {orchestrator_results.get('time', 0)}s")
    print(f"   {graph_status} GraphExecutor: {graph_results.get('time', 0)}s")
    
    print("\n🏭 Production Mode Results:")
    for test_name, result in production_results.items():
        status = "✅" if result.get('success') else "❌"
        print(f"   {status} {test_name.replace('_', ' ').title()}: {result.get('time', 0)}s")
    
    # Overall Assessment
    print("\n🎯 OVERALL ASSESSMENT:")
    
    agent_success_count = sum(1 for r in agent_results.values() if r.get('success'))
    total_agents = len(agent_results)
    
    workflow_success = orchestrator_results.get('success') and graph_results.get('success')
    production_success = all(r.get('success') for r in production_results.values())
    
    print(f"   📈 Agent Success Rate: {agent_success_count}/{total_agents} ({(agent_success_count/total_agents)*100:.1f}%)")
    print(f"   🔄 Workflow Success: {'✅ PASS' if workflow_success else '❌ FAIL'}")
    print(f"   🏭 Production Ready: {'✅ YES' if production_success else '❌ NO'}")
    print(f"   ⏱️ Total Test Time: {total_time}s")
    
    # Final Verdict
    overall_success = (agent_success_count >= total_agents * 0.8 and 
                      workflow_success and 
                      production_success)
    
    if overall_success:
        print("\n🎉 ALL SYSTEMS GO! The image analysis workflow is ready for production.")
        print("   ✅ Individual agents are working")
        print("   ✅ Both orchestrator and graph workflows are functional")
        print("   ✅ Production mode is operational")
        print("   ✅ End-to-end pipeline is complete")
    else:
        print("\n⚠️ ISSUES DETECTED! Some components need attention.")
        if agent_success_count < total_agents * 0.8:
            print("   ❌ Some individual agents are failing")
        if not workflow_success:
            print("   ❌ Workflow execution has issues")
        if not production_success:
            print("   ❌ Production mode needs fixes")
    
    return overall_success

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)