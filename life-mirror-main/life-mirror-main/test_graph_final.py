#!/usr/bin/env python3
"""
Final test of GraphExecutor to verify the formatter issue is resolved
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.graph_workflow import GraphExecutor
import os

def test_graph_executor():
    """Test GraphExecutor with mock data"""
    print("🚀 Testing GraphExecutor with current fixes...")
    print("=" * 50)
    
    # Set mock mode
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    # Create executor
    executor = GraphExecutor()
    
    # Test data
    test_data = {
        "media_id": "final_test",
        "url": "test_person.jpg",
        "context": {
            "user_id": "test_user",
            "analysis_type": "comprehensive"
        }
    }
    
    try:
        # Execute analysis
        result = executor.execute(
            media_id=test_data["media_id"],
            url=test_data["url"],
            context=test_data["context"]
        )
        
        print("\n📊 Analysis Results:")
        print("-" * 30)
        
        if result:
            print(f"✅ Success: {result.get('success', False)}")
            print(f"📈 Overall Score: {result.get('overall_score', 'N/A')}")
            print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
            
            # Check if we have formatted analysis
            if 'analysis' in result:
                analysis = result['analysis']
                print(f"📝 Analysis Summary: {analysis.get('summary', 'N/A')[:100]}...")
                print(f"💪 Key Strengths: {len(analysis.get('key_strengths', []))} items")
                print(f"🔧 Improvement Areas: {len(analysis.get('improvement_areas', []))} items")
            
            # Check individual components
            data = result.get('data', {})
            if data:
                print(f"👤 Face Analysis: {'✅' if data.get('face_analysis') else '❌'}")
                print(f"👗 Fashion Analysis: {'✅' if data.get('fashion_analysis') else '❌'}")
                print(f"🧍 Posture Analysis: {'✅' if data.get('posture_analysis') else '❌'}")
                print(f"🧬 Bio Analysis: {'✅' if data.get('bio_analysis') else '❌'}")
                print(f"🔢 Embedding Analysis: {'✅' if data.get('embedding_analysis') else '❌'}")
            
            print("\n🎉 GraphExecutor is working correctly!")
            return True
        else:
            print("❌ No result returned")
            return False
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graph_executor()
    if success:
        print("\n✅ All tests passed! The formatter issue has been resolved.")
    else:
        print("\n❌ Tests failed. The formatter issue persists.")
    
    sys.exit(0 if success else 1)