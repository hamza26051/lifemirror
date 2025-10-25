import os
import sys
import requests
import json
import time
from pathlib import Path

# Test the complete API workflow
def test_api_health():
    print("🏥 Testing API Health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend API: Healthy")
            return True
        else:
            print(f"❌ Backend API: Unhealthy (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Backend API: Connection failed - {e}")
        return False

def test_streamlit_frontend():
    print("\n🎨 Testing Streamlit Frontend...")
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Streamlit Frontend: Accessible")
            return True
        else:
            print(f"❌ Streamlit Frontend: Issues (status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Streamlit Frontend: Not running or connection refused")
        return False
    except Exception as e:
        print(f"❌ Streamlit Frontend: Connection failed - {e}")
        return False

def test_mock_mode_agents():
    print("\n🤖 Testing Agents in Mock Mode...")
    
    # Set to mock mode to avoid MediaPipe issues
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    sys.path.append('src')
    from src.agents.face_agent import FaceAgent
    from src.agents.fashion_agent import FashionAgent
    from src.agents.posture_agent import PostureAgent
    from src.schemas.agents import AgentInput
    
    test_input = AgentInput(
        media_id="test_mock_workflow",
        url="test_person.jpg",
        user_id="test_user",
        metadata={}
    )
    
    results = {}
    
    # Test Face Agent in Mock Mode
    try:
        face_agent = FaceAgent()
        face_result = face_agent.run(test_input)
        results['face'] = face_result.success
        print(f"✅ Face Agent (Mock): {'SUCCESS' if face_result.success else 'FAILED'}")
    except Exception as e:
        print(f"❌ Face Agent (Mock): EXCEPTION - {e}")
        results['face'] = False
    
    # Test Fashion Agent in Mock Mode
    try:
        fashion_agent = FashionAgent()
        fashion_result = fashion_agent.run(test_input)
        results['fashion'] = fashion_result.success
        print(f"✅ Fashion Agent (Mock): {'SUCCESS' if fashion_result.success else 'FAILED'}")
    except Exception as e:
        print(f"❌ Fashion Agent (Mock): EXCEPTION - {e}")
        results['fashion'] = False
    
    # Test Posture Agent in Mock Mode
    try:
        posture_agent = PostureAgent()
        posture_result = posture_agent.run(test_input)
        results['posture'] = posture_result.success
        print(f"✅ Posture Agent (Mock): {'SUCCESS' if posture_result.success else 'FAILED'}")
    except Exception as e:
        print(f"❌ Posture Agent (Mock): EXCEPTION - {e}")
        results['posture'] = False
    
    success_count = sum(results.values())
    print(f"\n📊 Mock Mode Success Rate: {success_count}/3 ({success_count/3*100:.1f}%)")
    
    return success_count == 3

def test_orchestrator():
    print("\n🎼 Testing Orchestrator...")
    
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    try:
        sys.path.append('src')
        from src.agents.orchestrator import Orchestrator
        from src.schemas.agents import AgentInput
        
        orchestrator = Orchestrator()
        test_input = AgentInput(
            media_id="test_orchestrator",
            url="test_person.jpg",
            user_id="test_user",
            metadata={}
        )
        
        result = orchestrator.analyze_media(
            media_id="test_orchestrator",
            url="test_person.jpg",
            context={"user_id": "test_user"}
        )
        
        if result and isinstance(result, dict):
            print("✅ Orchestrator: SUCCESS")
            print(f"   - Analysis completed with overall score: {result.get('overall_score', 'N/A')}")
            print(f"   - Confidence: {result.get('confidence', 'N/A')}")
            print(f"   - Agents processed: {len(result.get('detailed_analysis', {}))}")
            return True
        else:
            print(f"❌ Orchestrator: FAILED - {result if isinstance(result, str) else 'Unknown error'}")
            return False
            
    except Exception as e:
        print(f"❌ Orchestrator: EXCEPTION - {e}")
        return False

def main():
    print("🚀 LIFE MIRROR - API WORKFLOW TEST")
    print("=" * 50)
    
    # Test components
    api_healthy = test_api_health()
    frontend_accessible = test_streamlit_frontend()
    agents_working = test_mock_mode_agents()
    orchestrator_working = test_orchestrator()
    
    print("\n🏁 FINAL RESULTS")
    print("=" * 30)
    print(f"Backend API: {'✅ WORKING' if api_healthy else '❌ ISSUES'}")
    print(f"Frontend: {'✅ WORKING' if frontend_accessible else '❌ ISSUES'}")
    print(f"Agents (Mock): {'✅ WORKING' if agents_working else '❌ ISSUES'}")
    print(f"Orchestrator: {'✅ WORKING' if orchestrator_working else '❌ ISSUES'}")
    
    all_working = api_healthy and frontend_accessible and agents_working and orchestrator_working
    
    if all_working:
        print("\n🎊 COMPLETE SYSTEM IS FUNCTIONAL!")
        print("\n📋 NEXT STEPS:")
        print("1. Fix MediaPipe DLL issues for production mode")
        print("2. Test image upload through Streamlit frontend")
        print("3. Verify end-to-end analysis workflow")
        print("\n💡 TIP: You can now test the app by:")
        print("   - Opening http://localhost:8501 in your browser")
        print("   - Uploading an image through the Streamlit interface")
        print("   - Viewing the analysis results")
    else:
        print("\n⚠️  System has issues that need attention.")
        print("\n🔧 TROUBLESHOOTING:")
        if not api_healthy:
            print("   - Check if backend is running: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        if not frontend_accessible:
            print("   - Check if Streamlit is running: streamlit run streamlit_frontend.py --server.port 8501")
        if not agents_working:
            print("   - Check agent implementations and dependencies")
        if not orchestrator_working:
            print("   - Check orchestrator configuration and agent integration")

if __name__ == "__main__":
    main()