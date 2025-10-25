#!/usr/bin/env python3
"""
Test the FormatterAgent directly to see if it's working
"""

import os
import sys
sys.path.append('.')

from src.agents.formatter_agent import FormatterAgent
from src.agents.base_agent import AgentInput

def test_formatter_direct():
    """Test FormatterAgent directly"""
    print("🔍 Testing FormatterAgent directly...")
    
    # Set mock mode
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    try:
        formatter_agent = FormatterAgent()
        
        # Create test aggregated result
        aggregated_result = {
            "media_id": "test_media",
            "overall_score": 7.5,
            "attractiveness_score": 8.0,
            "style_score": 7.0,
            "presence_score": 7.5,
            "confidence": 0.8,
            "face_analysis": {"num_faces": 1},
            "fashion_analysis": None,
            "posture_analysis": None,
            "bio_analysis": None,
            "embedding_analysis": {"vector": [0.1, 0.2, 0.3]},
            "langsmith_run_ids": {"face": "test_id"},
            "processing_metadata": {"mode": "mock"},
            "key_strengths": ["Good lighting"],
            "improvement_areas": ["Better posture"],
            "explanation": "Test explanation"
        }
        
        formatter_context = {
            "aggregated_result": aggregated_result,
            "user_consent": {},
            "request_metadata": {},
            "langsmith_run_id": "test_run_id"
        }
        
        formatter_input = AgentInput(
            media_id="test_media",
            url="test_url",
            context=formatter_context
        )
        
        print("   - Running FormatterAgent...")
        result = formatter_agent.run(formatter_input)
        
        print(f"   - Success: {result.success}")
        print(f"   - Error: {result.error}")
        print(f"   - Data keys: {list(result.data.keys()) if result.data else 'None'}")
        
        if result.success and result.data:
            print("✅ FormatterAgent is working correctly")
            return True
        else:
            print("❌ FormatterAgent failed")
            return False
            
    except Exception as e:
        print(f"❌ FormatterAgent error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing FormatterAgent")
    print("=" * 40)
    
    success = test_formatter_direct()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ FormatterAgent works - issue is in graph execution")
    else:
        print("❌ FormatterAgent has issues")