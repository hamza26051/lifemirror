#!/usr/bin/env python3
"""
Debug test for orchestrator to identify hanging issues
"""

import os
import time
import threading
from src.agents.orchestrator import Orchestrator

def test_orchestrator_with_timeout():
    """Test orchestrator with timeout to detect hanging"""
    print("🔍 Testing Orchestrator with timeout detection...")
    
    # Set mock mode
    os.environ["LIFEMIRROR_MODE"] = "mock"
    
    def run_orchestrator():
        try:
            orchestrator = Orchestrator()
            result = orchestrator.analyze_media(
                media_id="test_debug",
                url="test_person.jpg",
                context={"user_id": "test_user"}
            )
            print(f"✅ Orchestrator completed successfully!")
            print(f"   - Type: {type(result)}")
            print(f"   - Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            if isinstance(result, dict):
                print(f"   - Overall Score: {result.get('overall_score', 'Missing')}")
                print(f"   - Confidence: {result.get('confidence', 'Missing')}")
                print(f"   - Summary: {result.get('summary', 'Missing')[:100]}...")
                print(f"   - Has detailed_analysis: {'detailed_analysis' in result}")
            
            return result
        except Exception as e:
            print(f"❌ Orchestrator failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Run with timeout
    result = [None]
    
    def target():
        result[0] = run_orchestrator()
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    # Wait for completion with timeout
    thread.join(timeout=30)  # 30 second timeout
    
    if thread.is_alive():
        print("⚠️  ORCHESTRATOR HANGING DETECTED!")
        print("   - Process did not complete within 30 seconds")
        print("   - This explains why Streamlit analysis gets stuck")
        return False
    else:
        print("✅ Orchestrator completed within timeout")
        return result[0] is not None

if __name__ == "__main__":
    print("🚀 Starting Orchestrator Debug Test")
    print("=" * 50)
    
    start_time = time.time()
    success = test_orchestrator_with_timeout()
    end_time = time.time()
    
    print("\n" + "=" * 50)
    print(f"🏁 Test completed in {end_time - start_time:.2f} seconds")
    
    if success:
        print("✅ ORCHESTRATOR IS WORKING CORRECTLY")
        print("   - No hanging issues detected")
        print("   - Streamlit issues may be elsewhere")
    else:
        print("❌ ORCHESTRATOR HAS ISSUES")
        print("   - This is likely causing Streamlit to hang")
        print("   - Check agent dependencies and infinite loops")