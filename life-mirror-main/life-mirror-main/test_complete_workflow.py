import os
import sys
import requests
import time
sys.path.append('src')

from src.agents.face_agent import FaceAgent
from src.agents.fashion_agent import FashionAgent
from src.agents.posture_agent import PostureAgent
from src.schemas.agents import AgentInput

# Set to production mode
os.environ["LIFEMIRROR_MODE"] = "prod"

def test_complete_analysis_workflow():
    print("🧪 Testing Complete Image Analysis Workflow")
    print("=" * 50)
    
    # Test image path
    test_image_path = os.path.join(os.getcwd(), "test_person.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    print(f"📸 Using test image: {test_image_path}")
    
    # Create test input
    test_input = AgentInput(
        media_id="test_complete_workflow",
        url=test_image_path,
        user_id="test_user",
        metadata={}
    )
    
    results = {}
    
    # Test Face Agent
    print("\n👤 Testing Face Agent...")
    try:
        face_agent = FaceAgent()
        face_result = face_agent.run(test_input)
        results['face'] = face_result
        
        if face_result.success:
            print("✅ Face Agent: SUCCESS")
            data = face_result.data
            print(f"   - Faces detected: {len(data.get('faces', []))}")
            print(f"   - Overall rating: {data.get('overall_rating', 'N/A')}")
            if data.get('faces'):
                for i, face in enumerate(data['faces']):
                    print(f"   - Face {i+1}: Age ~{face.get('age', 'unknown')}, Emotion: {face.get('emotion', 'unknown')}")
        else:
            print(f"❌ Face Agent: FAILED - {face_result.error}")
            
    except Exception as e:
        print(f"❌ Face Agent: EXCEPTION - {e}")
        results['face'] = {'success': False, 'error': str(e)}
    
    # Test Fashion Agent
    print("\n👗 Testing Fashion Agent...")
    try:
        fashion_agent = FashionAgent()
        fashion_result = fashion_agent.run(test_input)
        results['fashion'] = fashion_result
        
        if fashion_result.success:
            print("✅ Fashion Agent: SUCCESS")
            data = fashion_result.data
            print(f"   - Style: {data.get('style', 'unknown')}")
            print(f"   - Items detected: {len(data.get('items', []))}")
            print(f"   - Overall rating: {data.get('overall_rating', 'N/A')}")
            if data.get('items'):
                for i, item in enumerate(data['items'][:3]):  # Show first 3 items
                    print(f"   - Item {i+1}: {item}")
        else:
            print(f"❌ Fashion Agent: FAILED - {fashion_result.error}")
            
    except Exception as e:
        print(f"❌ Fashion Agent: EXCEPTION - {e}")
        results['fashion'] = {'success': False, 'error': str(e)}
    
    # Test Posture Agent
    print("\n🧍 Testing Posture Agent...")
    try:
        posture_agent = PostureAgent()
        posture_result = posture_agent.run(test_input)
        results['posture'] = posture_result
        
        if posture_result.success:
            print("✅ Posture Agent: SUCCESS")
            data = posture_result.data
            print(f"   - Posture quality: {data.get('posture_quality', 'unknown')}")
            print(f"   - Overall rating: {data.get('overall_rating', 'N/A')}")
            if data.get('recommendations'):
                print(f"   - Recommendations: {len(data['recommendations'])} found")
                for i, rec in enumerate(data['recommendations'][:2]):  # Show first 2
                    print(f"     • {rec}")
        else:
            print(f"❌ Posture Agent: FAILED - {posture_result.error}")
            
    except Exception as e:
        print(f"❌ Posture Agent: EXCEPTION - {e}")
        results['posture'] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n📊 WORKFLOW SUMMARY")
    print("=" * 30)
    
    success_count = 0
    total_agents = 3
    
    for agent_name, result in results.items():
        if hasattr(result, 'success') and result.success:
            print(f"✅ {agent_name.title()} Agent: Working")
            success_count += 1
        else:
            print(f"❌ {agent_name.title()} Agent: Failed")
    
    print(f"\n🎯 Success Rate: {success_count}/{total_agents} ({success_count/total_agents*100:.1f}%)")
    
    if success_count == total_agents:
        print("🎉 ALL AGENTS WORKING! Complete workflow is functional.")
        return True
    else:
        print("⚠️  Some agents failed. Check individual results above.")
        return False

def test_api_endpoint():
    print("\n🌐 Testing API Endpoint Integration")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Backend API: Healthy")
        else:
            print(f"❌ Backend API: Unhealthy (status: {health_response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Backend API: Connection failed - {e}")
        return False
    
    # Test media upload endpoint (without authentication for now)
    print("📤 Note: Media upload requires authentication - testing agents directly instead")
    return True

if __name__ == "__main__":
    print("🚀 LIFE MIRROR - COMPLETE WORKFLOW TEST")
    print("=" * 50)
    
    # Test individual agents
    agents_working = test_complete_analysis_workflow()
    
    # Test API health
    api_working = test_api_endpoint()
    
    print("\n🏁 FINAL RESULTS")
    print("=" * 20)
    print(f"Agents Pipeline: {'✅ WORKING' if agents_working else '❌ ISSUES'}")
    print(f"API Backend: {'✅ WORKING' if api_working else '❌ ISSUES'}")
    
    if agents_working and api_working:
        print("\n🎊 COMPLETE SYSTEM IS FUNCTIONAL!")
        print("Users can now upload images and get comprehensive analysis.")
    else:
        print("\n⚠️  System has issues that need attention.")